#!/usr/bin/env python
"""Calibrate the ball adapter's common rolling-direction sign from a forward run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from common.caster_frame import load_caster_frames
from common.config import load_config, update_yaml


def infer_direction_sign(
    frames_path: str | Path,
    current_sign: float,
    *,
    min_samples: int = 5,
    min_roll_rate_rad_s: float = 1e-3,
    min_forward_cos: float = 0.5,
    min_consistency: float = 0.8,
) -> tuple[float, dict[str, Any]]:
    """Return the sign correction that makes a known-forward clip point car +x."""

    sign = float(current_sign)
    if sign not in {-1.0, 1.0}:
        raise ValueError("current ball.roll_direction_sign must be +1 or -1")
    frames, metadata = load_caster_frames(frames_path)
    if metadata.get("R_bc_calibrated") is False or metadata.get(
        "R_ball_to_car_calibrated"
    ) is False:
        raise ValueError("known-forward frames require calibrated ball axis transforms")
    heading_x: list[float] = []
    z = np.array([0.0, 0.0, 1.0])
    for frame in frames:
        if not frame.roll_valid or not frame.heading_valid:
            continue
        if frame.omega_roll < float(min_roll_rate_rad_s):
            continue
        raw_heading = frame.raw.get("rolling_heading_car")
        if raw_heading is None:
            heading = np.cross(z, frame.roll_axis_car)[:2]
        else:
            heading = np.asarray(raw_heading, dtype=float).reshape(-1)
            if heading.shape != (2,) or not np.all(np.isfinite(heading)):
                continue
        norm = float(np.linalg.norm(heading))
        if norm <= 1e-9:
            continue
        x_value = float(heading[0] / norm)
        if abs(x_value) >= float(min_forward_cos):
            heading_x.append(x_value)
    if len(heading_x) < int(min_samples):
        raise ValueError(
            f"known-forward stream has only {len(heading_x)} usable samples; "
            f"need at least {int(min_samples)}"
        )
    values = np.asarray(heading_x, dtype=float)
    median = float(np.median(values))
    correction = float(np.sign(median))
    consistency = float(np.mean(np.sign(values) == np.sign(median)))
    if consistency < float(min_consistency):
        raise ValueError(
            f"known-forward heading consistency is only {consistency:.1%}; trim turns/stops"
        )
    return sign * correction, {
        "sample_count": len(values),
        "median_current_heading_x": median,
        "direction_consistency": consistency,
        "input_metadata": metadata,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--caster-frames", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--min-roll-rate-rad-s", type=float, default=1e-3)
    parser.add_argument("--min-forward-cos", type=float, default=0.5)
    parser.add_argument("--min-consistency", type=float, default=0.8)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config, config_path = load_config(args.config)
        report_path = args.report.expanduser().resolve() if args.report else None
        if report_path == config_path:
            raise ValueError("--report must not overwrite the calibration config")
        ball = config.get("ball", {}) or {}
        calibrated_sign, detail = infer_direction_sign(
            args.caster_frames,
            float(ball.get("roll_direction_sign", -1.0)),
            min_samples=args.min_samples,
            min_roll_rate_rad_s=args.min_roll_rate_rad_s,
            min_forward_cos=args.min_forward_cos,
            min_consistency=args.min_consistency,
        )
        update_yaml(
            config_path,
            {
                "ball": {
                    "roll_direction_sign": calibrated_sign,
                    "direction_sign_calibrated": True,
                }
            },
        )
        report = {
            "method": "known-forward CasterFrame stream",
            "input": str(args.caster_frames.expanduser().resolve()),
            "roll_direction_sign": calibrated_sign,
            "direction_sign_calibrated": True,
            "config_written": str(config_path),
            **detail,
        }
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = report_path.with_suffix(report_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(report_path)
        print(f"Ball roll direction sign: {calibrated_sign:+g}")
        print(f"Updated ball.direction_sign_calibrated in: {config_path}")
        if report_path is not None:
            print(f"Report: {report_path}")
        return 0
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "infer_direction_sign", "main"]
