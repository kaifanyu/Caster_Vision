#!/usr/bin/env python
"""Calibrate the loaded ball-caster rolling radius from known travel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from common.config import load_config, update_yaml


def effective_radius(distance_m: float, angle_rad: float) -> float:
    """Return ``distance / |angle|`` after validating a physical trial."""

    distance = float(distance_m)
    angle = abs(float(angle_rad))
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("distance_m must be positive and finite")
    if not np.isfinite(angle) or angle <= 1e-9:
        raise ValueError("the measured rolling angle must be nonzero and finite")
    return distance / angle


def _measured_angle(args: argparse.Namespace) -> tuple[float, str, dict[str, float]]:
    has_endpoints = args.phi_start is not None or args.phi_end is not None
    if has_endpoints and (args.phi_start is None or args.phi_end is None):
        raise ValueError("--phi-start and --phi-end must be supplied together")
    mode_count = sum(
        (
            args.revolutions is not None,
            args.angle_rad is not None,
            has_endpoints,
        )
    )
    if mode_count != 1:
        raise ValueError(
            "choose exactly one angle source: --revolutions, --angle-rad, "
            "or --phi-start with --phi-end"
        )

    if args.revolutions is not None:
        revolutions = abs(float(args.revolutions))
        if not np.isfinite(revolutions) or revolutions <= 0.0:
            raise ValueError("--revolutions must be nonzero and finite")
        return (
            2.0 * np.pi * revolutions,
            "known distance and revolution count",
            {"revolutions": revolutions},
        )
    if args.angle_rad is not None:
        angle = abs(float(args.angle_rad))
        return angle, "known distance and total rolling angle", {"angle_rad": angle}

    start, end = float(args.phi_start), float(args.phi_end)
    return (
        abs(end - start),
        "known distance and rolling-angle endpoints",
        {"phi_start_rad": start, "phi_end_rad": end},
    )


def _write_report(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--distance-m", type=float, required=True)
    parser.add_argument("--revolutions", type=float)
    parser.add_argument(
        "--angle-rad",
        type=float,
        help="Signed or unsigned total rolling angle over the known distance",
    )
    parser.add_argument("--phi-start", type=float, help="Starting unwrapped roll angle (rad)")
    parser.add_argument("--phi-end", type=float, help="Ending unwrapped roll angle (rad)")
    parser.add_argument("--report", type=Path, help="Optional JSON calibration report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _, config_path = load_config(args.config)
        if (
            args.report is not None
            and args.report.expanduser().resolve() == config_path
        ):
            raise ValueError("--report must not overwrite the calibration config")
        angle, method, detail = _measured_angle(args)
        radius = effective_radius(args.distance_m, angle)
        update_yaml(config_path, {"ball": {"r_eff_m": radius}})
        report = {
            "method": method,
            "known_distance_m": float(args.distance_m),
            "delta_angle_rad": angle,
            "r_eff_m": radius,
            **detail,
            "config_key": "ball.r_eff_m",
            "config_written": str(config_path),
        }
        report_path = _write_report(args.report, report) if args.report else None
        print(f"Effective ball radius: {radius:.9g} m")
        print(f"Updated ball.r_eff_m in: {config_path}")
        if report_path is not None:
            print(f"Report: {report_path}")
        return 0
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "effective_radius", "main"]
