#!/usr/bin/env python
"""Run the measured real-footage ball-caster rotation pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from ballrot.config import (
    camera_matrix,
    configured_circle,
    configured_frame,
    distortion_coefficients,
    load_config,
    measurement_frame,
    resolve_from_config,
)
from ballrot.diagnostics import _json_clean, summarize_quality, unconstrained_result, write_run_outputs
from ballrot.integrate import axis_disagreement_deg, inter_shell_swivel_axis
from ballrot.io_frames import FrameSource
from ballrot.pipeline import run_pipeline
from ballrot.offline_observations import save_observations


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    result.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 when self-consistency targets are missed",
    )
    result.add_argument("--no-overlay", action="store_true")
    result.add_argument("--output-dir", type=Path, default=None,
                        help="Write this run to a separate directory (relative to the current directory)")
    return result


def _check_assumptions(config: dict) -> None:
    assumptions = config.get("assumptions", {})
    required = ("camera_fixed_to_chassis", "ball_center_stationary_in_image")
    rejected = [name for name in required if not bool(assumptions.get(name, False))]
    if rejected:
        raise SystemExit(
            "STOP: the rotation-only model requires these config assumptions to be true: "
            + ", ".join(rejected)
            + ". This footage is outside the pipeline's stated model."
        )
    if bool(config.get("circle", {}).get("refit_each_frame", False)):
        raise SystemExit(
            "STOP: circle.refit_each_frame=true is incompatible with the required "
            "stationary-center assumption in this implementation. Fix the camera mount "
            "or use a different motion model."
        )
    segment_mode = str(config.get("segment", {}).get("mode", "color")).lower()
    if segment_mode == "color" and not bool(
        assumptions.get("two_speckle_colors", False)
    ):
        raise SystemExit(
            "STOP: segment.mode=color requires two distinguishable speckle colors. "
            "Use segment.mode=equator for one-color footage."
        )


def _print_startup(
    source: FrameSource,
    fps: float,
    approximate_K: bool,
    circle,
    config: dict,
) -> None:
    assumptions = config.get("assumptions", {})
    segment = config.get("segment", {})
    print("Resolved assumptions and inputs")
    print("-------------------------------")
    print(f"input:              {source.path}")
    print(f"input type:         {source.kind}")
    print(f"frames reported:    {source.frame_count if source.frame_count is not None else 'unknown'}")
    print(f"frame rate:         {fps:g} fps")
    print(f"camera fixed:       {assumptions.get('camera_fixed_to_chassis')}")
    print(f"center stationary:  {assumptions.get('ball_center_stationary_in_image')}")
    print(f"intrinsics source:  {'FOV approximation (UNTRUSTED)' if approximate_K else 'calibrated config K'}")
    print(f"circle source:      {'auto-fit first frame' if circle is None else 'configured fixed circle'}")
    print(f"segmentation:       {segment.get('mode', 'color')}")
    print(f"paint support:      {'enabled' if segment.get('paint_support', {}).get('enabled', False) else 'disabled'}")
    print("axis frame source:  calibrated R_bc with initial roll offset")
    print(f"initial roll:       {config.get('frame_calib', {}).get('initial_roll_deg', 0.0):g} deg")
    print(f"temporal tracking:  {'persistent tracks + nearby keyframes' if config.get('temporal', {}).get('enabled', False) else 'disabled (frame pairs)'}")
    print(f"offline refinement: {'spherical pixel fit + backward observations' if config.get('offline', {}).get('enabled', False) else 'disabled'}")
    print(f"mechanical fit:     {'shared roll + independent swivel' if config.get('mechanical', {}).get('enabled', False) else 'disabled'}")
    if config.get("offline", {}).get("enabled", False) and config.get("offline", {}).get("window_frames", 0):
        print(f"offline windows:    {config['offline']['window_frames']} frames with {config['offline'].get('window_overlap_frames', 12)} overlapping frames")
    if config.get("mechanical", {}).get("enabled", False) and config.get("mechanical", {}).get("window_frames", 0):
        print(f"mechanical windows: {config['mechanical']['window_frames']} frames with {config['mechanical'].get('overlap_frames', 0)} overlapping frames")
    print()


def _swivel_axis_check(
    result, R_bc, max_disagreement_deg: float = 5.0
) -> dict[str, object]:
    """Compare the clip's own differential shell axis with calibrated ``R_bc``.

    Every other check in this runner is blind to a caster that was re-seated
    between axis calibration and the measurement: tracking stays clean and the
    residuals stay small while every reported angle is expressed in the wrong
    frame.  This one is not, because the differential shell motion observes the
    swivel axis directly, in the measurement clip, without using ``R_bc``.
    """

    axis, diagnostics = inter_shell_swivel_axis(
        result.top_absolute,
        result.bottom_absolute,
        valid_top=result.top_step_valid,
        valid_bottom=result.bottom_step_valid,
    )
    report: dict[str, object] = {
        "target_max_deg": max_disagreement_deg,
        **diagnostics,
    }
    if axis is None:
        report["status"] = "not observable"
        report["note"] = (
            "The shells never spun far enough apart for this clip to observe "
            "its own swivel axis; R_bc remains unverified against this footage."
        )
        return report
    disagreement = axis_disagreement_deg(axis, np.asarray(R_bc, dtype=float)[:, 2])
    report["measured_axis_camera"] = [float(value) for value in axis]
    report["calibrated_axis_camera"] = [
        float(value) for value in np.asarray(R_bc, dtype=float)[:, 2]
    ]
    report["disagreement_deg"] = disagreement
    report["status"] = "ok" if disagreement <= max_disagreement_deg else "mismatch"
    return report


def _acceptance(summary: dict, min_inliers: int) -> tuple[bool, list[str]]:
    failures = []
    independent = summary.get("unconstrained_checks", summary)
    for name in ("top", "bottom"):
        values = summary[name]
        residual = values["mean_inlier_residual_deg_max"]
        ratio = values["inlier_ratio_min"]
        fb_error = values["forward_backward_error_px_max"]
        tracked_min = values["tracked_count_min"]
        success_rate = values["success_rate"]
        gamma = independent[name]["gamma_residual_deg_median"]
        if residual is None or residual > 0.5:
            failures.append(f"{name} maximum Kabsch residual is {residual} deg (target <= 0.5)")
        if ratio is None or ratio < 0.7:
            failures.append(f"{name} minimum inlier ratio is {ratio} (target >= 0.7)")
        if fb_error is None or fb_error > 1.0:
            failures.append(f"{name} maximum per-frame median forward/backward error is {fb_error} px (target <= 1)")
        if tracked_min is None or tracked_min < min_inliers:
            failures.append(
                f"{name} minimum tracked count is {tracked_min} (target >= {min_inliers})"
            )
        if success_rate is None or success_rate < 1.0:
            failures.append(
                f"{name} solve success rate is {success_rate} (strict target = 1.0)"
            )
        if gamma is None or gamma > 1.0:
            failures.append(
                f"{name} median gamma residual is {gamma} deg (target <= 1.0)"
            )
    disagreement = independent["alpha_top_bottom_disagreement_deg_median"]
    if disagreement is None or disagreement > 1.0:
        failures.append(
            f"top/bottom alpha disagreement is {disagreement} deg (target <= 1.0)"
        )
    swivel = summary.get("swivel_axis_check", {})
    final_tracking = summary.get("mechanical_tracking", summary.get("offline_tracking", summary.get("temporal_tracking", {})))
    for name, report in final_tracking.items():
        missing = report.get("unresolved_frames", [])
        if missing:
            failures.append(f"{name} has {len(missing)} unresolved poses; see results.json tracking diagnostics")
    if swivel.get("status") == "mismatch":
        failures.append(
            "this clip's own swivel axis is "
            f"{swivel['disagreement_deg']:.1f} deg from the calibrated R_bc "
            f"z-axis (target <= {swivel['target_max_deg']:.1f}); check the "
            "initial pose and tracking drift before interpreting alpha/beta"
        )
    return not failures, failures


def _print_mechanical_status(report: dict | None) -> None:
    """Explain missing moving overlays separately from feature detection."""
    if report is None:
        return
    print(f"MECHANICAL FIT: {report.get('status', 'unknown')}")
    if "windows" in report:
        print(f"  Window attempts: {len(report['windows'])}; see mechanical_report.json for overlap and gap diagnostics")
    if report.get("solver_message"):
        print(f"  Solver: {report['solver_message']} (evaluations: {report.get('solver_nfev', 'unknown')})")
    for name in ("top", "bottom"):
        summary = report.get("summary", {}).get(name, {})
        print(f"  {name}: {summary.get('valid_frames', 0)} valid poses; "
              f"{len(summary.get('unresolved_frames', []))} unresolved")
    if not any(item.get("valid_frames", 0) for item in report.get("summary", {}).values()):
        print("  No accepted mechanical poses: the moving axes/grid will be hidden. "
              "This does not imply the color masks were empty; inspect mechanical_report.json.")


def main() -> int:
    args = parser().parse_args()
    config, config_path = load_config(args.config)
    _check_assumptions(config)
    input_cfg = config.get("input", {})
    if not input_cfg.get("path"):
        raise SystemExit("input.path is empty in the configuration")
    input_path = resolve_from_config(config_path, input_cfg["path"])
    source = FrameSource(
        input_path,
        input_type=input_cfg.get("type", "auto"),
        max_frames=input_cfg.get("max_frames"),
        image_fps=input_cfg.get("fps_override"),
    )
    if source.frame_count is not None and source.frame_count < 2:
        raise SystemExit("At least two frames are required to measure rotation.")
    fps_override = input_cfg.get("fps_override")
    fps = float(fps_override if fps_override is not None else (source.fps or 0.0))
    if fps <= 0:
        raise SystemExit(
            "No reliable FPS is available. Set input.fps_override in config.yaml "
            "(required for image sequences and videos without FPS metadata)."
        )
    first = next(iter(source), None)
    if first is None:
        raise SystemExit("input contains no decodable frames")
    K, approximate = camera_matrix(config.get("camera", {}), first.image.shape)
    dist = distortion_coefficients(config.get("camera", {}))
    circle = configured_circle(config.get("circle", {}))
    R_bc_home = configured_frame(config.get("frame_calib", {}))
    R_bc = measurement_frame(config.get("frame_calib", {}))
    if R_bc is None:
        raise SystemExit(
            "frame_calib.R_bc is null. Record labeled pure-roll and pure-swivel clips, "
            "then run: python scripts/calibrate_axes.py --config config.yaml "
            "--roll data/axis_calibration/pure_roll.mp4 "
            "--swivel data/axis_calibration/pure_swivel.mp4"
        )
    _print_startup(source, fps, approximate, circle, config)

    output_cfg = config.get("output", {})
    output_dir = (args.output_dir.resolve() if args.output_dir is not None else
                  resolve_from_config(config_path, output_cfg.get("dir", "out/real")))
    save_overlay = bool(output_cfg.get("save_overlay_video", True)) and not args.no_overlay
    overlay_path = output_dir / "tracking_overlay.mp4" if save_overlay else None
    # Preserve native video timestamps unless the user explicitly retimes it.
    result = run_pipeline(
        source.frames() if fps_override is not None else source.records(),
        K=K,
        dist=dist,
        circle=circle,
        R_bc=R_bc,
        segment_config=config.get("segment", {}),
        track_config=config.get("track", {}),
        estimate_config=config.get("estimate", {}),
        temporal_config=config.get("temporal", {}),
        offline_config=config.get("offline", {}),
        mechanical_config=config.get("mechanical", {}),
        top_shell_sign=int(config.get("frame_calib", {}).get("top_shell_sign", 1)),
        fps=fps,
        overlay_path=overlay_path,
        overlay_codec=output_cfg.get("overlay_codec", "mp4v"),
        progress=lambda message: print(message, flush=True),
    )
    if result.frame_count < 2:
        raise SystemExit("At least two frames are required to measure rotation.")
    if result.motion is None:  # Kept explicit as an invariant guard.
        raise RuntimeError("decomposition unexpectedly missing despite configured R_bc")
    independent = None
    check_result = result
    if result.unconstrained_motion is not None:
        independent = unconstrained_result(
            result.unconstrained_motion, result.unconstrained_rotations["top"],
            result.unconstrained_rotations["bottom"], result.qualities)
        check_result = SimpleNamespace(
            top_absolute=result.unconstrained_rotations["top"],
            bottom_absolute=result.unconstrained_rotations["bottom"],
            top_step_valid=result.unconstrained_valid["top"],
            bottom_step_valid=result.unconstrained_valid["bottom"])
    swivel_check = _swivel_axis_check(check_result, R_bc)
    if independent is not None:
        swivel_check["pose_source"] = "unconstrained estimates before mechanical fitting"
    paths = write_run_outputs(
        output_dir,
        result.timestamps_s,
        result.motion,
        result.qualities,
        result.top_absolute,
        result.bottom_absolute,
        metadata={
            "config": str(config_path),
            "input": str(input_path),
            "frame_count": result.frame_count,
            "fps": fps,
            "K_was_approximated": approximate,
            "K": K,
            "dist": dist,
            "circle": result.circle,
            "R_bc": R_bc,
            "R_bc_home": R_bc_home,
            "initial_roll_deg": float(config.get("frame_calib", {}).get("initial_roll_deg", 0.0)),
            "top_shell_sign": int(config.get("frame_calib", {}).get("top_shell_sign", 1)),
            "timing_source": "fps_override" if fps_override is not None else getattr(source, "timing_source", "frame_records"),
            "overlay_timing": "constant-rate diagnostic playback; use time_s for measurement",
            "tracking_overlay_stage": "forward tracking before optional offline refinement",
            "segmentation": config.get("segment", {}),
            "feature_tracking": config.get("track", {}),
            "increment_estimation": config.get("estimate", {}),
            **({"mechanical_model": {**result.mechanical["config"],
                                      "model": result.mechanical["model"]}}
               if result.mechanical is not None else {}),
            "assumptions": config.get("assumptions", {}),
        },
        extra_summary={"swivel_axis_check": swivel_check,
                       **({"temporal_tracking": result.temporal["summary"]}
                          if result.temporal is not None else {}),
                       **({"offline_tracking": result.offline["summary"]}
                          if result.offline is not None else {})},
        temporal=result.temporal,
        offline=result.offline,
        mechanical=result.mechanical,
        unconstrained=independent,
    )
    if result.mechanical is not None:
        paths["mechanical_report"] = output_dir / "mechanical_report.json"
        paths["mechanical_report"].write_text(
            json.dumps(_json_clean(result.mechanical), indent=2, allow_nan=False), encoding="utf-8")
    if result.offline_observations is not None:
        paths["offline_observations"] = save_observations(
            output_dir / "offline_observations.npz", result.offline_observations,
            timestamps=result.timestamps_s, K=result.K, center=result.sphere_center,
            radius=result.sphere_radius, initial_rotations=result.initial_rotations,
            initial_valid=result.initial_valid,
            refined_rotations={"top": result.top_absolute, "bottom": result.bottom_absolute},
            refined_valid={"top": result.top_step_valid, "bottom": result.bottom_step_valid},
            landmark_sources=result.offline_landmark_sources,
            unconstrained_rotations=result.unconstrained_rotations,
            unconstrained_valid=result.unconstrained_valid)
    summary = summarize_quality(
        result.qualities, result.motion, result.top_absolute, result.bottom_absolute
    )
    summary["swivel_axis_check"] = swivel_check
    if result.temporal is not None:
        summary["temporal_tracking"] = result.temporal["summary"]
    if result.offline is not None:
        summary["offline_tracking"] = result.offline["summary"]
    if result.mechanical is not None:
        summary["mechanical_tracking"] = result.mechanical["summary"]
        summary["unconstrained_checks"] = independent["summary"]
        summary["constraint_metrics_note"] = "Zero gamma/shared roll are imposed; acceptance uses independent pre-constraint checks."
    passed, failures = _acceptance(
        summary, int(config.get("estimate", {}).get("min_inliers", 8))
    )
    print(f"Processed {result.frame_count} frames.")
    print(json.dumps(summary, indent=2))
    print(f"SELF-CONSISTENCY: {'PASS' if passed else 'WARN'}")
    _print_mechanical_status(result.mechanical)
    for failure in failures:
        print(f"  - {failure}")
    if approximate:
        print("  - Camera intrinsics were approximated; do not treat this run as calibrated.")
    print("Outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    if result.overlay_path:
        print(f"  overlay: {result.overlay_path}")
    return 2 if args.strict and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
