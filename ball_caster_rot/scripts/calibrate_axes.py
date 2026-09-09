#!/usr/bin/env python
"""Estimate the fixed ball-to-camera frame from pure roll and swivel clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.spatial.transform import Rotation

from ballrot.config import (
    camera_matrix,
    configured_circle,
    distortion_coefficients,
    load_config,
    resolve_from_config,
    update_yaml,
)
from ballrot.integrate import calibrate_ball_frame
from ballrot.io_frames import FrameSource, open_frame_source
from ballrot.pipeline import PipelineResult, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Capture requirements:
  * Both clips must start at the mechanical home pose.
  * --roll must contain only roll; --swivel must contain only swivel.
  * Move monotonically in the positive right-hand-rule direction by default.
  * Axis direction cannot be recovered from imagery alone. If a clip was shot
    in the negative direction, pass the corresponding --*-sign -1 option.
""",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument(
        "--roll",
        required=True,
        help="Pure-roll video, image file, image directory, or image glob",
    )
    parser.add_argument(
        "--swivel",
        required=True,
        help="Pure-swivel video, image file, image directory, or image glob",
    )
    parser.add_argument(
        "--roll-type",
        choices=("auto", "video", "images"),
        default="auto",
        help="Input type for --roll (default: auto)",
    )
    parser.add_argument(
        "--swivel-type",
        choices=("auto", "video", "images"),
        default="auto",
        help="Input type for --swivel (default: auto)",
    )
    parser.add_argument(
        "--roll-sign",
        type=int,
        choices=(-1, 1),
        default=1,
        help="+1 if roll moved in positive +x direction, -1 if it moved negative",
    )
    parser.add_argument(
        "--swivel-sign",
        type=int,
        choices=(-1, 1),
        default=1,
        help="+1 if swivel moved in positive +z direction, -1 if it moved negative",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS for image sequences (default: input.fps_override, then 60)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Optional frame limit per clip"
    )
    parser.add_argument(
        "--min-axis-step-deg",
        type=float,
        default=0.20,
        help=(
            "Minimum incremental rotation used for axis fitting and direction "
            "validation (default: 0.20 deg)"
        ),
    )
    parser.add_argument(
        "--max-axis-spread-deg",
        type=float,
        default=15.0,
        help="Maximum allowed p95 increment-axis spread (default: 15 deg)",
    )
    parser.add_argument(
        "--min-axis-separation-deg",
        type=float,
        default=70.0,
        help="Minimum acute separation between raw roll/swivel axes (default: 70 deg)",
    )
    parser.add_argument(
        "--min-pca-explained",
        type=float,
        default=0.90,
        help="Minimum shared-axis PCA explained ratio (default: 0.90)",
    )
    parser.add_argument(
        "--min-valid-steps",
        type=int,
        default=8,
        help="Minimum non-trivial top-hemisphere increments per clip (default: 8)",
    )
    parser.add_argument(
        "--min-direction-consistency",
        type=float,
        default=0.80,
        help="Minimum fraction of increments moving in the labeled direction",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path (default: <output.dir>/axis_calibration_report.json)",
    )
    return parser


def _resolve_input(value: str) -> str:
    """Resolve ordinary CLI paths while retaining glob metacharacters."""

    expanded = str(Path(value).expanduser())
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return str(candidate)


def _image_fps(args: argparse.Namespace, config: dict[str, Any]) -> float:
    configured = config.get("input", {}).get("fps_override")
    value = args.fps if args.fps is not None else configured
    fps = 60.0 if value is None else float(value)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("--fps/input.fps_override must be positive and finite")
    return fps


def _open_source(
    path: str,
    input_type: str,
    max_frames: int | None,
    image_fps: float,
) -> FrameSource:
    source = open_frame_source(
        _resolve_input(path),
        input_type=input_type,
        max_frames=max_frames,
        image_fps=image_fps,
    )
    if source.frame_count is not None and source.frame_count < 3:
        raise ValueError(f"calibration input needs at least 3 frames: {path}")
    return source


def _run_clip(
    source: FrameSource,
    *,
    K: np.ndarray,
    dist: np.ndarray,
    circle: tuple[float, float, float],
    config: dict[str, Any],
    fallback_fps: float,
) -> PipelineResult:
    return run_pipeline(
        source.frames(),
        K=K,
        dist=dist,
        circle=circle,
        R_bc=None,
        segment_config=config.get("segment", {}),
        track_config=config.get("track", {}),
        estimate_config=config.get("estimate", {}),
        fps=source.fps or fallback_fps,
    )


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _pipeline_summary(result: PipelineResult) -> dict[str, Any]:
    top_success = np.array([estimate.top.success for estimate in result.estimates])
    bottom_success = np.array([estimate.bottom.success for estimate in result.estimates])
    top_counts = np.array([match.top.count for match in result.matches], dtype=float)
    bottom_counts = np.array([match.bottom.count for match in result.matches], dtype=float)
    top_residual = np.array(
        [estimate.top.mean_inlier_residual_deg for estimate in result.estimates], dtype=float
    )
    bottom_residual = np.array(
        [estimate.bottom.mean_inlier_residual_deg for estimate in result.estimates],
        dtype=float,
    )
    return {
        "frame_count": result.frame_count,
        "step_count": len(result.estimates),
        "top_valid_steps": int(top_success.sum()),
        "bottom_valid_steps": int(bottom_success.sum()),
        "top_median_matches": _finite_or_none(float(np.median(top_counts))),
        "bottom_median_matches": _finite_or_none(float(np.median(bottom_counts))),
        "top_median_inlier_residual_deg": _finite_or_none(
            float(np.nanmedian(top_residual))
        ),
        "bottom_median_inlier_residual_deg": _finite_or_none(
            float(np.nanmedian(bottom_residual))
        ),
    }


def _nontrivial_rotation_vectors(
    increments: Iterable[np.ndarray | None],
    min_step_deg: float,
) -> tuple[np.ndarray, int]:
    if not np.isfinite(min_step_deg) or min_step_deg <= 0:
        raise ValueError("min_step_deg must be positive and finite")

    vectors: list[np.ndarray] = []
    raw_step_count = 0
    min_step_rad = np.deg2rad(min_step_deg)
    for increment in increments:
        if increment is None:
            continue
        vector = Rotation.from_matrix(np.asarray(increment, dtype=float)).as_rotvec()
        raw_step_count += 1
        if np.linalg.norm(vector) >= min_step_rad:
            vectors.append(vector)
    return np.asarray(vectors, dtype=float).reshape(-1, 3), raw_step_count


def _direction_diagnostics(
    increments: Iterable[np.ndarray | None],
    axis: np.ndarray,
    expected_sign: int,
    min_step_deg: float,
) -> dict[str, Any]:
    vectors, raw_step_count = _nontrivial_rotation_vectors(
        increments, min_step_deg
    )
    retained_step_count = len(vectors)
    counts = {
        "raw_step_count": raw_step_count,
        "rejected_low_motion_count": raw_step_count - retained_step_count,
        "retained_step_count": retained_step_count,
        "nontrivial_step_count": retained_step_count,
        "min_axis_step_deg": float(min_step_deg),
    }
    if len(vectors) == 0:
        return {
            **counts,
            "consistent_direction_fraction": None,
            "median_signed_step_deg": None,
        }
    signed = vectors @ np.asarray(axis, dtype=float)
    consistent = signed * expected_sign > 0.0
    return {
        **counts,
        "consistent_direction_fraction": float(np.mean(consistent)),
        "median_signed_step_deg": float(np.rad2deg(np.median(signed))),
    }


def _validate(
    diagnostics: dict[str, Any],
    roll_direction: dict[str, Any],
    swivel_direction: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    failures: list[str] = []
    for name in ("roll", "swivel"):
        values = diagnostics[name]
        if int(values["sample_count"]) < args.min_valid_steps:
            failures.append(
                f"{name}: only {values['sample_count']} steps met the "
                f"{args.min_axis_step_deg:.3f} deg motion floor "
                f"(need {args.min_valid_steps}; "
                f"rejected {values['rejected_low_motion_count']} low-motion steps)"
            )
        if float(values["axis_spread_p95_deg"]) > args.max_axis_spread_deg:
            failures.append(
                f"{name}: p95 axis spread {values['axis_spread_p95_deg']:.2f} deg "
                f"exceeds {args.max_axis_spread_deg:.2f} deg"
            )
        if float(values["pca_explained_ratio"]) < args.min_pca_explained:
            failures.append(
                f"{name}: PCA explained ratio {values['pca_explained_ratio']:.3f} "
                f"is below {args.min_pca_explained:.3f}"
            )
    separation = float(diagnostics["raw_axis_separation_deg"])
    if separation < args.min_axis_separation_deg:
        failures.append(
            f"raw axes are separated by only {separation:.2f} deg "
            f"(need at least {args.min_axis_separation_deg:.2f} deg)"
        )
    for name, values in (
        ("roll", roll_direction),
        ("swivel", swivel_direction),
    ):
        consistency = values["consistent_direction_fraction"]
        if consistency is None or consistency < args.min_direction_consistency:
            rendered = "unavailable" if consistency is None else f"{consistency:.3f}"
            failures.append(
                f"{name}: direction consistency {rendered} is below "
                f"{args.min_direction_consistency:.3f}; motion may not be monotonic"
            )
    return failures


def _report_path(
    explicit: Path | None, config: dict[str, Any], config_path: Path
) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    output_value = config.get("output", {}).get("dir", "out")
    output_dir = resolve_from_config(config_path, output_value)
    return output_dir / "axis_calibration_report.json"


def _validate_cli_thresholds(args: argparse.Namespace) -> None:
    if args.max_frames is not None and args.max_frames < 3:
        raise ValueError("--max-frames must be at least 3")
    if not np.isfinite(args.min_axis_step_deg) or args.min_axis_step_deg <= 0:
        raise ValueError("--min-axis-step-deg must be positive and finite")
    if args.max_axis_spread_deg <= 0:
        raise ValueError("--max-axis-spread-deg must be positive")
    if not 0 < args.min_axis_separation_deg <= 90:
        raise ValueError("--min-axis-separation-deg must be in (0, 90]")
    if not 0 < args.min_pca_explained <= 1:
        raise ValueError("--min-pca-explained must be in (0, 1]")
    if args.min_valid_steps < 3:
        raise ValueError("--min-valid-steps must be at least 3")
    if not 0 < args.min_direction_consistency <= 1:
        raise ValueError("--min-direction-consistency must be in (0, 1]")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_cli_thresholds(args)
        config, config_path = load_config(args.config)
        camera_config = config.get("camera", {})
        if camera_config.get("K") is None:
            raise ValueError(
                "camera.K must be calibrated before axis calibration; "
                "run scripts/calibrate_camera.py first"
            )
        circle = configured_circle(config.get("circle", {}))
        if circle is None:
            raise ValueError(
                "circle.u0/v0/r_px must be configured before axis calibration; "
                "run scripts/calibrate_circle.py first"
            )
        image_fps = _image_fps(args, config)
        roll_source = _open_source(
            args.roll, args.roll_type, args.max_frames, image_fps
        )
        swivel_source = _open_source(
            args.swivel, args.swivel_type, args.max_frames, image_fps
        )
        if swivel_source.size != roll_source.size:
            raise ValueError(
                "roll and swivel clips must use the same resolution; got "
                f"{roll_source.size} and {swivel_source.size}"
            )
        # K is mandatory, so camera_matrix only validates it here and can never
        # select its approximate-FOV branch.
        K, approximate = camera_matrix(
            camera_config, (roll_source.size[1], roll_source.size[0], 3)
        )
        assert not approximate
        dist = distortion_coefficients(camera_config)

        print("Assumption: camera and ball center are fixed in both clips.")
        print("Assumption: clips start at home and contain isolated, monotonic motion.")
        print("Axis source: top-hemisphere increments in both labeled clips.")
        print(
            "Axis motion floor: retaining increments of at least "
            f"{args.min_axis_step_deg:.3f} deg."
        )
        print(
            "Direction ambiguity: imagery determines an axis line, not its sign; "
            f"using roll_sign={args.roll_sign:+d}, swivel_sign={args.swivel_sign:+d}."
        )

        roll_result = _run_clip(
            roll_source,
            K=K,
            dist=dist,
            circle=circle,
            config=config,
            fallback_fps=image_fps,
        )
        swivel_result = _run_clip(
            swivel_source,
            K=K,
            dist=dist,
            circle=circle,
            config=config,
            fallback_fps=image_fps,
        )
        R_bc, diagnostics = calibrate_ball_frame(
            roll_result.top_increments,
            swivel_result.top_increments,
            roll_sign=args.roll_sign,
            swivel_sign=args.swivel_sign,
            min_step_deg=args.min_axis_step_deg,
        )
        roll_direction = _direction_diagnostics(
            roll_result.top_increments,
            R_bc[:, 0],
            args.roll_sign,
            args.min_axis_step_deg,
        )
        swivel_direction = _direction_diagnostics(
            swivel_result.top_increments,
            R_bc[:, 2],
            args.swivel_sign,
            args.min_axis_step_deg,
        )
        failures = _validate(
            diagnostics, roll_direction, swivel_direction, args
        )
        passed = not failures

        report_path = _report_path(args.report, config, config_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any] = {
            "passed": passed,
            "config": str(config_path),
            "config_updated": passed,
            "inputs": {
                "roll": str(roll_source.path),
                "swivel": str(swivel_source.path),
                "roll_type": roll_source.kind,
                "swivel_type": swivel_source.kind,
            },
            "assumptions": {
                "camera_fixed_to_chassis": True,
                "ball_center_fixed": True,
                "clips_start_at_home": True,
                "motions_are_isolated_and_monotonic": True,
                "axis_source_hemisphere": "top",
                "axis_direction_is_user_labeled": True,
            },
            "signs": {"roll": args.roll_sign, "swivel": args.swivel_sign},
            "K": K.tolist(),
            "dist": dist.tolist(),
            "circle": {"u0": circle[0], "v0": circle[1], "r_px": circle[2]},
            "R_bc": R_bc.tolist(),
            "axes_camera": {
                "x_roll": R_bc[:, 0].tolist(),
                "y_derived": R_bc[:, 1].tolist(),
                "z_swivel": R_bc[:, 2].tolist(),
            },
            "diagnostics": diagnostics,
            "direction_diagnostics": {
                "roll": roll_direction,
                "swivel": swivel_direction,
            },
            "pipeline": {
                "roll": _pipeline_summary(roll_result),
                "swivel": _pipeline_summary(swivel_result),
            },
            "thresholds": {
                "min_axis_step_deg": args.min_axis_step_deg,
                "max_axis_spread_p95_deg": args.max_axis_spread_deg,
                "min_raw_axis_separation_deg": args.min_axis_separation_deg,
                "min_pca_explained_ratio": args.min_pca_explained,
                "min_valid_steps": args.min_valid_steps,
                "min_direction_consistency": args.min_direction_consistency,
            },
            "failures": failures,
        }
        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        if passed:
            update_yaml(config_path, {"frame_calib": {"R_bc": R_bc.tolist()}})
            print("PASS: isolated-motion axis calibration passed all diagnostics.")
            print(
                "Axis spread p95 (roll/swivel): "
                f"{diagnostics['roll']['axis_spread_p95_deg']:.3f} / "
                f"{diagnostics['swivel']['axis_spread_p95_deg']:.3f} deg"
            )
            print(
                "Retained axis steps (roll/swivel): "
                f"{diagnostics['roll']['retained_step_count']} / "
                f"{diagnostics['swivel']['retained_step_count']} "
                f"at >= {args.min_axis_step_deg:.3f} deg"
            )
            print(
                "Raw roll/swivel separation: "
                f"{diagnostics['raw_axis_separation_deg']:.3f} deg"
            )
            print(f"Updated frame_calib.R_bc in: {config_path}")
        else:
            print("FAIL: axis calibration diagnostics did not pass; config was not changed.")
            for failure in failures:
                print(f"  - {failure}")
        print(f"Report: {report_path}")
        print(
            "If recovered roll or swivel has the wrong sign, re-run with "
            "--roll-sign -1 and/or --swivel-sign -1."
        )
        return 0 if passed else 2
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
