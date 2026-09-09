#!/usr/bin/env python
"""Run the Approach-A ball-caster pipeline and common comparison metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from ball.adapter import adapt_ball_pipeline
from ball.diagnostics import summarize_quality, write_run_outputs
from ball.pipeline import PipelineResult, run_pipeline
from common.caster_frame import CasterFrame, save_caster_frames
from common.config import (
    camera_matrix,
    configured_circle,
    distortion_coefficients,
    load_config,
    resolve_from_config,
)
from common.io_frames import FrameSource, open_frame_source
from common.kinematics import load_car_track
from common.rotation import is_rotation_matrix
from metrics.diagnostics import save_metric_diagnostics
from metrics.metrics import MetricConfig, compute_metrics


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _metric_config(values: Mapping[str, Any]) -> MetricConfig:
    translated = {
        "min_contact_speed_mps": values.get("speed_epsilon_mps", 0.005),
        "min_longitudinal_speed_mps": values.get(
            "slip_speed_epsilon_mps", 0.02
        ),
        "min_confidence": values.get("min_confidence", 0.0),
        "lag_max_s": values.get("max_lag_s", 1.0),
        "settling_threshold_deg": values.get("settling_threshold_deg", 5.0),
        "settling_dwell_s": values.get("settling_dwell_s", 0.25),
        "shimmy_min_frequency_hz": values.get("shimmy_min_hz", 0.5),
        "shimmy_max_frequency_hz": values.get("shimmy_max_hz"),
    }
    segment = values.get("shimmy_segment_s")
    if isinstance(segment, (list, tuple)) and len(segment) == 2:
        translated["shimmy_start_s"], translated["shimmy_end_s"] = map(
            float, segment
        )
    return MetricConfig.from_value(translated)


def _check_assumptions(config: Mapping[str, Any]) -> None:
    assumptions = config.get("assumptions", {}) or {}
    if not bool(assumptions.get("camera_fixed_to_chassis", False)):
        raise SystemExit(
            "assumptions.camera_fixed_to_chassis must be true; "
            "a room-fixed camera violates the ball rotation-only model"
        )
    if assumptions.get("ball_center_stationary_in_image") is False:
        raise SystemExit(
            "assumptions.ball_center_stationary_in_image cannot be false; "
            "the projected ball center must remain fixed"
        )
    ball = config.get("ball", {}) or {}
    circle = ball.get("circle", config.get("circle", {})) or {}
    if bool(circle.get("refit_each_frame", False)):
        raise SystemExit(
            "ball.circle.refit_each_frame=true is unsupported by the fixed-center model"
        )


def _as_rotation(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if not is_rotation_matrix(matrix, atol=1e-6):
        raise SystemExit(f"{name} must be a finite, proper 3x3 rotation matrix")
    return matrix


def _resolve_ball_calibration(
    ball: Mapping[str, Any], *, tracking_only: bool
) -> tuple[np.ndarray, np.ndarray, float, dict[str, bool]]:
    """Resolve strict measurement calibration or explicit tracking placeholders."""

    r_bc_value = ball.get("R_bc")
    ball_to_car_value = ball.get("R_ball_to_car")
    radius_value = ball.get("r_eff_m")
    direction_calibrated = bool(ball.get("direction_sign_calibrated", False))
    if r_bc_value is None and not tracking_only:
        raise SystemExit(
            "ball.R_bc is null. Calibrate the ball axes or pass --tracking-only."
        )
    if radius_value is None and not tracking_only:
        raise SystemExit(
            "ball.r_eff_m is null. Calibrate effective radius or pass --tracking-only."
        )
    if ball_to_car_value is None and not tracking_only:
        raise SystemExit(
            "ball.R_ball_to_car is null. Re-run ball axis calibration after camera "
            "extrinsics, or pass --tracking-only."
        )
    if not direction_calibrated and not tracking_only:
        raise SystemExit(
            "ball.direction_sign_calibrated is false. Run the known-forward "
            "direction calibration or pass --tracking-only."
        )
    R_bc = (
        np.eye(3)
        if r_bc_value is None
        else _as_rotation(r_bc_value, "ball.R_bc")
    )
    R_ball_to_car = _as_rotation(
        np.eye(3) if ball_to_car_value is None else ball_to_car_value,
        "ball.R_ball_to_car",
    )
    r_eff = 1.0 if radius_value is None else float(radius_value)
    if not np.isfinite(r_eff) or r_eff <= 0.0:
        raise SystemExit("ball.r_eff_m must be finite and positive")
    return R_bc, R_ball_to_car, r_eff, {
        "R_bc_calibrated": r_bc_value is not None,
        "R_ball_to_car_calibrated": ball_to_car_value is not None,
        "direction_sign_calibrated": direction_calibrated,
        "r_eff_calibrated": radius_value is not None,
    }


def _timestamped_records(
    source: FrameSource, fps: float, time_offset_s: float
) -> Iterable[tuple[np.ndarray, float, str]]:
    """Use the selected FPS consistently and put caster time on the car clock."""

    for record in source:
        yield record.image, record.index / fps + time_offset_s, record.source


def _step_valid(result: PipelineResult, name: str, index: int) -> bool:
    increments = getattr(result, f"{name}_increments")
    if increments[index] is None:
        return False
    values = np.asarray(getattr(result, f"{name}_step_valid"), dtype=bool)
    if values.shape == (len(result.timestamps_s),):
        return bool(values[index + 1])
    if values.shape == (len(result.timestamps_s) - 1,):
        return bool(values[index])
    raise ValueError(f"{name}_step_valid has an unexpected shape")


def _increment_payload(result: PipelineResult) -> dict[str, Any]:
    times = np.asarray(result.timestamps_s, dtype=float)
    interval_count = len(times) - 1
    return {
        "time_start_s": times[:-1],
        "time_end_s": times[1:],
        "time_mid_s": 0.5 * (times[:-1] + times[1:]),
        "top_camera": list(result.top_increments),
        "bottom_camera": list(result.bottom_increments),
        "top_valid": [
            _step_valid(result, "top", index) for index in range(interval_count)
        ],
        "bottom_valid": [
            _step_valid(result, "bottom", index)
            for index in range(interval_count)
        ],
    }


def _write_increment_csv(result: PipelineResult, output: Path) -> Path:
    """Write lossless camera-frame increment matrices and explicit validity."""

    destination = output / "increments.csv"
    matrix_names = [f"r{row}{column}" for row in range(3) for column in range(3)]
    fields = [
        "interval",
        "time_start_s",
        "time_end_s",
        "time_mid_s",
        "top_valid",
        "bottom_valid",
        *[f"top_{name}" for name in matrix_names],
        *[f"bottom_{name}" for name in matrix_names],
    ]
    times = np.asarray(result.timestamps_s, dtype=float)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (top, bottom) in enumerate(
            zip(result.top_increments, result.bottom_increments)
        ):
            row: dict[str, Any] = {
                "interval": index,
                "time_start_s": times[index],
                "time_end_s": times[index + 1],
                "time_mid_s": 0.5 * (times[index] + times[index + 1]),
                "top_valid": _step_valid(result, "top", index),
                "bottom_valid": _step_valid(result, "bottom", index),
            }
            for prefix, matrix in (("top", top), ("bottom", bottom)):
                flat = [] if matrix is None else np.asarray(matrix, dtype=float).reshape(9)
                for matrix_index, name in enumerate(matrix_names):
                    row[f"{prefix}_{name}"] = (
                        "" if matrix is None else flat[matrix_index]
                    )
            writer.writerow(row)
    return destination


def _motion_payload(result: PipelineResult) -> dict[str, Any] | None:
    motion = result.motion
    if motion is None:
        return None
    return {
        "alpha_rad": motion.alpha,
        "beta_top_rad": motion.beta_top,
        "beta_bottom_rad": motion.beta_bottom,
        "alpha_top_rad": motion.alpha_top,
        "alpha_bottom_rad": motion.alpha_bottom,
        "gamma_top_rad": motion.gamma_top,
        "gamma_bottom_rad": motion.gamma_bottom,
        "valid_top": motion.valid_top,
        "valid_bottom": motion.valid_bottom,
    }


def _resolve_car_track(
    argument: Path | None,
    input_config: Mapping[str, Any],
    config_path: Path,
) -> Path | None:
    if argument is not None:
        return argument.expanduser().resolve()
    value = input_config.get("car_track")
    if not value:
        return None
    candidate = resolve_from_config(config_path, value)
    return candidate if candidate.is_file() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--clip", type=Path)
    parser.add_argument("--car-track", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--tracking-only",
        action="store_true",
        help=(
            "Allow missing R_bc/R_ball_to_car/r_eff/direction calibration, mark "
            "output uncalibrated, and skip metrics"
        ),
    )
    parser.add_argument(
        "--no-overlay", action="store_true", help="Do not write the tracking video"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, config_path = load_config(args.config)
    _check_assumptions(config)
    input_config = config.get("input", {}) or {}
    clip_value = args.clip if args.clip is not None else input_config.get("ball_clip")
    if clip_value is None:
        raise SystemExit("input.ball_clip is missing and --clip was not supplied")
    clip = (
        args.clip.expanduser().resolve()
        if args.clip is not None
        else resolve_from_config(config_path, clip_value)
    )
    source = open_frame_source(
        clip,
        input_type=str(input_config.get("type", "auto")),
        max_frames=input_config.get("max_frames"),
        image_fps=input_config.get("fps_override"),
    )
    if source.frame_count is not None and source.frame_count < 2:
        raise SystemExit("the ball clip must contain at least two frames")
    fps = float(input_config.get("fps_override") or source.fps or 0.0)
    if not np.isfinite(fps) or fps <= 0.0:
        raise SystemExit("no reliable FPS is available; set input.fps_override")

    camera_config = config.get("camera", {}) or {}
    K, approximated = camera_matrix(
        camera_config, (source.size[1], source.size[0], 3)
    )
    dist = distortion_coefficients(camera_config)
    ball_config = config.get("ball", {}) or {}
    R_bc, R_ball_to_car, r_eff, calibration = _resolve_ball_calibration(
        ball_config, tracking_only=args.tracking_only
    )
    if approximated and not args.tracking_only:
        raise SystemExit(
            "camera.K is uncalibrated. Run intrinsics calibration or pass --tracking-only."
        )
    if approximated:
        warnings.warn(
            "tracking-only mode is using an approximate FOV camera matrix; output is untrusted",
            RuntimeWarning,
        )
    if args.tracking_only and not all(calibration.values()):
        warnings.warn(
            "tracking-only mode has one or more uncalibrated axis, direction, or "
            "radius values; do not interpret common velocities or metric scale",
            RuntimeWarning,
        )

    circle_config = ball_config.get("circle", config.get("circle", {})) or {}
    circle = configured_circle(circle_config)
    output_config = config.get("output", {}) or {}
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else resolve_from_config(config_path, output_config.get("dir", "out"))
        / "ball"
    )
    output.mkdir(parents=True, exist_ok=True)
    save_overlay = bool(output_config.get("save_overlay_video", True)) and not args.no_overlay
    overlay = output / "tracking_overlay.mp4" if save_overlay else None
    time_offset = float((config.get("car_track", {}) or {}).get("time_offset_s", 0.0))
    result = run_pipeline(
        _timestamped_records(source, fps, time_offset),
        K=K,
        dist=dist,
        circle=circle,
        R_bc=R_bc,
        segment_config=ball_config.get("segment", {}),
        track_config=config.get("track", {}),
        estimate_config=config.get("estimate", {}),
        fps=fps,
        overlay_path=overlay,
        overlay_codec=str(output_config.get("overlay_codec", "mp4v")),
    )
    if result.frame_count < 2 or result.motion is None:
        raise RuntimeError("ball pipeline returned no decomposable frame interval")
    caster_frames = adapt_ball_pipeline(
        result,
        R_bc=R_bc,
        R_ball_to_car=R_ball_to_car,
        r_eff=r_eff,
        roll_direction_sign=float(ball_config.get("roll_direction_sign", -1.0)),
    )
    summary = summarize_quality(
        result.qualities, result.motion, result.top_absolute, result.bottom_absolute
    )
    trusted = bool(
        not args.tracking_only
        and calibration["R_bc_calibrated"]
        and calibration["R_ball_to_car_calibrated"]
        and calibration["direction_sign_calibrated"]
        and calibration["r_eff_calibrated"]
    )
    metadata: dict[str, Any] = {
        "device": "ball",
        "input": str(clip),
        "config": str(config_path),
        "frame_count": result.frame_count,
        "fps": fps,
        "time_offset_s": time_offset,
        "K": K,
        "dist": dist,
        "K_approximated": approximated,
        "circle": result.circle,
        "R_bc": R_bc,
        "R_ball_to_car": R_ball_to_car,
        "r_eff_m": r_eff,
        **calibration,
        "tracking_only": bool(args.tracking_only),
        "clocks_synchronized": bool(
            (config.get("assumptions", {}) or {}).get(
                "car_and_caster_clocks_synchronized", False
            )
        ),
        "trusted_measurement": trusted,
        "summary": summary,
    }

    # Retain the Approach-A CSV and diagnostic plots.  Its results.json is
    # replaced below by a superset that includes the raw rotations.
    ball_diagnostic_paths = write_run_outputs(
        output,
        result.timestamps_s,
        result.motion,
        result.qualities,
        result.top_absolute,
        result.bottom_absolute,
        metadata=metadata,
    )
    increment_csv = _write_increment_csv(result, output)

    metric_payload = None
    metric_paths: dict[str, Path] = {}
    car_path = _resolve_car_track(args.car_track, input_config, config_path)
    if not args.tracking_only and car_path is not None and len(caster_frames) >= 2:
        if not bool(
            (config.get("assumptions", {}) or {}).get(
                "car_and_caster_clocks_synchronized", False
            )
        ):
            raise SystemExit(
                "car/caster clocks are not marked synchronized. Run sync-event or "
                "sync-signals before computing shared metrics. Tracking artifacts were written."
            )
        track_config = config.get("car_track", {}) or {}
        car_track = load_car_track(
            car_path,
            smooth_window=int(track_config.get("savgol_window", 9)),
            polyorder=int(track_config.get("savgol_polyorder", 2)),
            theta_unit=str(track_config.get("theta_unit", "rad")),
        )
        metric_result = compute_metrics(
            caster_frames,
            car_track,
            r_arm_car=np.asarray(
                track_config.get("r_arm_car_m", [0.0, 0.0]), dtype=float
            ),
            config=_metric_config(config.get("metrics", {}) or {}),
        )
        metric_paths = save_metric_diagnostics(
            metric_result, output / "metrics", title="Ball caster"
        )
        metric_payload = metric_result.to_dict()
        metadata["metrics_summary"] = metric_payload["summary"]
    elif not args.tracking_only:
        warnings.warn(
            "car track was unavailable (or the clip had fewer than three frames); "
            "tracking outputs were written but shared metrics were skipped",
            RuntimeWarning,
        )

    caster_path = save_caster_frames(
        output / "caster_frames.json", caster_frames, metadata=metadata
    )
    payload = {
        "schema": "ball-pipeline-results-v1",
        "metadata": metadata,
        "increments": _increment_payload(result),
        "top_step_valid": result.top_step_valid,
        "bottom_step_valid": result.bottom_step_valid,
        "top_absolute_camera": result.top_absolute,
        "bottom_absolute_camera": result.bottom_absolute,
        "motion": _motion_payload(result),
        "quality": result.qualities,
        "metrics": metric_payload,
        "artifacts": {
            "caster_frames": caster_path,
            "results_csv": output / "results.csv",
            "increments_csv": increment_csv,
            "overlay": result.overlay_path,
            "ball_diagnostics": ball_diagnostic_paths,
            "metric_diagnostics": metric_paths,
        },
    }
    result_path = output / "results.json"
    result_path.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8"
    )

    top_coverage = float(np.mean(payload["increments"]["top_valid"]))
    bottom_coverage = float(np.mean(payload["increments"]["bottom_valid"]))
    print(f"Processed {result.frame_count} frames at {fps:.3f} FPS")
    print(f"Top/bottom rotation coverage: {top_coverage:.1%} / {bottom_coverage:.1%}")
    print(f"Measurement calibration: {'trusted' if trusted else 'TRACKING ONLY / UNTRUSTED'}")
    print(f"Results: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
