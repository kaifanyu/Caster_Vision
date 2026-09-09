#!/usr/bin/env python
"""Run the calibrated swivel-caster pipeline on a real clip."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from common.caster_frame import save_caster_frames
from common.config import (
    camera_matrix,
    distortion_coefficients,
    load_config,
    resolve_from_config,
)
from common.io_frames import open_frame_source
from common.kinematics import load_car_track
from metrics.diagnostics import save_metric_diagnostics
from metrics.metrics import MetricConfig, compute_metrics
from swivel.geometry import SwivelGeometry
from swivel.angular import angular_intervals, write_angular_csv
from swivel.pipeline import SwivelPipelineResult, run_pipeline


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _metric_config(values: Mapping[str, Any]) -> MetricConfig:
    translated = {
        "min_contact_speed_mps": values.get("speed_epsilon_mps", 0.005),
        "min_longitudinal_speed_mps": values.get("slip_speed_epsilon_mps", 0.02),
        "lag_max_s": values.get("max_lag_s", 1.0),
        "settling_threshold_deg": values.get("settling_threshold_deg", 5.0),
        "settling_dwell_s": values.get("settling_dwell_s", 0.25),
        "shimmy_min_frequency_hz": values.get("shimmy_min_hz", 0.5),
        "shimmy_max_frequency_hz": values.get("shimmy_max_hz"),
    }
    segment = values.get("shimmy_segment_s")
    if isinstance(segment, (list, tuple)) and len(segment) == 2:
        translated["shimmy_start_s"], translated["shimmy_end_s"] = map(float, segment)
    return MetricConfig.from_value(translated)


def _summary(result: SwivelPipelineResult, *, loop_closure: bool) -> dict[str, Any]:
    valid_tags = [item for item in result.tag_observations if item.valid]
    tag_errors = np.array([item.reprojection_error_px for item in valid_tags], dtype=float)
    successful = [item for item, q in zip(result.roll_estimates, result.interval_quality)
                  if item is not None and q.get("roll_valid", False)]
    inliers = np.array([item.inlier_ratio for item in successful], dtype=float)
    off_axis = np.array([np.rad2deg(item.off_axis_residual_rad) for item in successful], dtype=float)
    off_axis = off_axis[np.isfinite(off_axis)]
    roll_valid = np.asarray([bool(item.get("roll_valid")) for item in result.interval_quality])
    delta = np.array([
        item.delta_phi if item is not None and q.get("roll_valid", False) else np.nan
        for item, q in zip(result.roll_estimates, result.interval_quality)
    ])
    finite_delta = delta[np.isfinite(delta) & (np.abs(delta) > np.deg2rad(0.05))]
    monotonic_fraction = (
        float(max(np.mean(finite_delta >= 0), np.mean(finite_delta <= 0)))
        if len(finite_delta) else float("nan")
    )
    summary: dict[str, Any] = {
        "tag_detection_coverage": len(valid_tags) / result.frame_count,
        "tag_reprojection_error_px_median": float(np.median(tag_errors)) if len(tag_errors) else float("nan"),
        "tag_reprojection_error_px_max": float(np.max(tag_errors)) if len(tag_errors) else float("nan"),
        "roll_valid_coverage": float(np.mean(roll_valid)) if len(roll_valid) else 0.0,
        "roll_inlier_ratio_median": float(np.median(inliers)) if len(inliers) else float("nan"),
        "roll_off_axis_residual_deg_median": float(np.median(off_axis)) if len(off_axis) else float("nan"),
        "steady_roll_monotonic_fraction": monotonic_fraction,
        "accumulated_phi_complete_at_end": bool(result.phi_valid[-1]),
        "accumulated_phi_valid_coverage": float(np.mean(result.phi_valid)),
        "failure_counts": dict(Counter(
            ("OK" if q.get("roll_valid") else str(q.get("failure_reason", "unknown")))
            for q in result.interval_quality)),
        "tag_failure_counts": dict(Counter(getattr(o, "failure_reason", "unknown") for o in result.tag_observations if not o.valid)),
    }
    if loop_closure:
        valid_psi = result.psi[result.psi_valid]
        valid_phi = result.phi if result.phi_valid[0] and result.phi_valid[-1] else np.array([])
        psi_delta = valid_psi[-1] - valid_psi[0] if len(valid_psi) >= 2 else float("nan")
        phi_delta = valid_phi[-1] - valid_phi[0] if len(valid_phi) >= 2 else float("nan")
        wrap = lambda angle: (float(angle) + np.pi) % (2.0 * np.pi) - np.pi
        summary["loop_closure"] = {
            "psi_closure_error_deg": float(np.rad2deg(wrap(psi_delta))),
            "phi_closure_error_deg": float(np.rad2deg(wrap(phi_delta))),
            "psi_net_unwrapped_deg": float(np.rad2deg(psi_delta)),
            "phi_net_unwrapped_deg": float(np.rad2deg(phi_delta)),
            "phi_net_turns": float(phi_delta / (2.0 * np.pi)),
            "note": (
                "Closure errors are modulo 360 degrees. Net unwrapped motion is also "
                "reported so valid full wheel turns are not mistaken for closure error."
            ),
        }
    summary["self_consistency_targets"] = {
        "tag_reprojection_small": summary["tag_reprojection_error_px_median"] < 2.0,
        "roll_inlier_ratio_gt_0_7": summary["roll_inlier_ratio_median"] > 0.7,
        "off_axis_residual_small": summary["roll_off_axis_residual_deg_median"] < 2.0,
        "tag_coverage_ge_0_95": summary["tag_detection_coverage"] >= .95,
        "roll_coverage_ge_0_80": summary["roll_valid_coverage"] >= .80,
        "accumulated_phi_complete": bool(result.phi_valid[-1]),
    }
    return summary


def _write_track(result: SwivelPipelineResult, output: Path) -> Path:
    path = output / "swivel_track.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "phi_rad", "psi_rad", "phi_valid", "psi_valid"])
        for row in zip(result.timestamps_s, result.phi, result.psi, result.phi_valid, result.psi_valid):
            writer.writerow(row)
    return path


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
            "Allow approximate K and missing psi0/r_eff, mark output untrusted, "
            "and skip metrics; camera extrinsics and usable geometry remain required"
        ),
    )
    parser.add_argument("--loop-closure", action="store_true", help="Report end-vs-start angle differences for a marked loop clip")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, config_path = load_config(args.config)
    assumptions = config.get("assumptions", {})
    if not bool(assumptions.get("camera_fixed_to_chassis", False)):
        raise SystemExit("assumptions.camera_fixed_to_chassis must be true; a room-fixed camera is unsupported")
    print("Resolved assumptions:")
    print("  camera is rigidly fixed to chassis: yes")
    print(f"  clocks synchronized: {bool(assumptions.get('car_and_caster_clocks_synchronized', False))}")
    print("  camera frame: OpenCV x-right/y-down/z-forward; car frame: x-forward/y-left/z-up")

    input_cfg = config.get("input", {})
    clip = args.clip.resolve() if args.clip else resolve_from_config(config_path, input_cfg["swivel_clip"])
    source = open_frame_source(
        clip,
        input_type=str(input_cfg.get("type", "auto")),
        max_frames=input_cfg.get("max_frames"),
        image_fps=input_cfg.get("fps_override"),
    )
    fps = float(input_cfg.get("fps_override") or source.fps or 0.0)
    if not np.isfinite(fps) or fps <= 0.0:
        raise SystemExit("no reliable FPS is available; set input.fps_override")
    camera_cfg = config.get("camera", {})
    K, approximated = camera_matrix(camera_cfg, (source.size[1], source.size[0], 3))
    if approximated and not args.tracking_only:
        raise SystemExit(
            "camera.K is uncalibrated. Run intrinsics calibration or pass --tracking-only."
        )
    if approximated:
        warnings.warn(
            "tracking-only mode is using an approximate FOV camera matrix; output is untrusted",
            RuntimeWarning,
        )
    dist = distortion_coefficients(camera_cfg)
    geometry = SwivelGeometry.from_mapping(config)
    swivel_cfg = config.get("swivel", {})
    marker = swivel_cfg.get("marker", {})
    geometry_cfg = swivel_cfg.get("geometry", {})
    geometry_calibrated = bool(geometry_cfg.get("calibrated", False))
    direction_sign_calibrated = bool(
        swivel_cfg.get("direction_sign_calibrated", False)
    )
    if not args.tracking_only and not geometry_calibrated:
        raise SystemExit(
            "swivel.geometry is not marked calibrated. Run the geometry subcommand "
            "or pass --tracking-only."
        )
    if not args.tracking_only and not direction_sign_calibrated:
        raise SystemExit(
            "swivel direction sign is not calibrated. Run roll-sign or pass --tracking-only."
        )
    psi0 = marker.get("psi0_rad")
    if psi0 is None:
        if not args.tracking_only:
            raise SystemExit(
                "swivel.marker.psi0_rad is null. Run psi-zero calibration or pass --tracking-only."
            )
        warnings.warn(
            "tracking-only mode is using psi0=0; absolute heading is untrusted",
            RuntimeWarning,
        )
        psi0 = 0.0
    r_eff = swivel_cfg.get("r_eff_m")
    if r_eff is None and not args.tracking_only:
        raise SystemExit("swivel.r_eff_m is null. Run radius calibration or pass --tracking-only.")
    output_cfg = config.get("output", {})
    output = (
        args.output.resolve()
        if args.output
        else resolve_from_config(config_path, output_cfg.get("dir", "out")) / "swivel"
    )
    output.mkdir(parents=True, exist_ok=True)
    overlay = output / "tracking_overlay.mp4" if bool(output_cfg.get("save_overlay_video", True)) else None
    timestamped_source = (
        (record.image, record.index / fps, record.source) for record in source
    )
    result = run_pipeline(
        timestamped_source,
        K=K,
        dist=dist,
        geometry=geometry,
        marker_size_m=float(marker.get("size_m", 0.04)),
        marker_id=int(marker.get("id", 0)),
        marker_dictionary=str(marker.get("dictionary", "DICT_4X4_50")),
        psi0_rad=float(psi0),
        max_tag_reprojection_error_px=float(marker.get("max_reprojection_error_px", 2.0)),
        track_config=config.get("track", {}),
        estimate_config=config.get("estimate", {}),
        reference_dot_config=swivel_cfg.get("reference_dot", {}),
        sidewall_inner_fraction=float(swivel_cfg.get("geometry", {}).get("sidewall_inner_fraction", 0.20)),
        edge_on_min_cos=float(swivel_cfg.get("geometry", {}).get("edge_on_min_cos", 0.18)),
        max_off_axis_deg=float(config.get("estimate", {}).get("max_off_axis_deg", 2.0)),
        r_eff=None if r_eff is None else float(r_eff),
        roll_direction_sign=float(swivel_cfg.get("roll_direction_sign", 1.0)),
        heading_direction_sign=float(swivel_cfg.get("heading_direction_sign", 1.0)),
        fps=fps,
        time_offset_s=float(config.get("car_track", {}).get("time_offset_s", 0.0)),
        overlay_path=overlay,
        marker_config=marker, mask_config=swivel_cfg.get("mask", {}),
    )
    _write_track(result, output)
    summary = _summary(result, loop_closure=args.loop_closure)
    metadata = {
        "device": "swivel",
        "input": str(clip),
        "config": str(config_path),
        "frame_count": result.frame_count,
        "fps": fps,
        "time_offset_s": float(
            config.get("car_track", {}).get("time_offset_s", 0.0)
        ),
        "camera": {
            "K": K,
            "dist": dist,
            "T_car_from_cam": camera_cfg.get("T_car_from_cam"),
        },
        "swivel_calibration": {
            "marker": dict(marker),
            "geometry": dict(swivel_cfg.get("geometry", {})),
            "r_eff_m": r_eff,
            "roll_direction_sign": float(
                swivel_cfg.get("roll_direction_sign", 1.0)
            ),
            "heading_direction_sign": float(
                swivel_cfg.get("heading_direction_sign", 1.0)
            ),
        },
        "K_approximated": approximated,
        "psi0_calibrated": marker.get("psi0_rad") is not None,
        "geometry_calibrated": geometry_calibrated,
        "direction_sign_calibrated": direction_sign_calibrated,
        "r_eff_calibrated": r_eff is not None,
        "tracking_only": bool(args.tracking_only),
        "clocks_synchronized": bool(
            assumptions.get("car_and_caster_clocks_synchronized", False)
        ),
        "trusted_measurement": bool(
            not args.tracking_only
            and not approximated
            and marker.get("psi0_rad") is not None
            and geometry_calibrated
            and direction_sign_calibrated
            and r_eff is not None
            and all(summary["self_consistency_targets"].values())
        ),
        "summary": summary,
        "processing_config": {
            "track": config.get("track", {}), "estimate": config.get("estimate", {}),
            "mask": swivel_cfg.get("mask", {}),
            "reference_dot": swivel_cfg.get("reference_dot", {}),
        },
    }
    if result.caster_frames:
        save_caster_frames(output / "caster_frames.json", result.caster_frames, metadata=metadata)

    metric_payload = None
    car_path = args.car_track
    if car_path is None and input_cfg.get("car_track"):
        candidate = resolve_from_config(config_path, input_cfg["car_track"])
        car_path = candidate if candidate.is_file() else None
    if result.caster_frames and car_path is not None and not args.tracking_only:
        if not bool(assumptions.get("car_and_caster_clocks_synchronized", False)):
            raise SystemExit(
                "car/caster clocks are not marked synchronized. Run sync-event or "
                "sync-signals before computing shared metrics. Tracking artifacts were written."
            )
        track_cfg = config.get("car_track", {})
        car = load_car_track(
            car_path,
            smooth_window=int(track_cfg.get("savgol_window", 9)),
            polyorder=int(track_cfg.get("savgol_polyorder", 2)),
            theta_unit=str(track_cfg.get("theta_unit", "rad")),
        )
        metric = compute_metrics(
            result.caster_frames,
            car,
            r_arm_car=np.asarray(track_cfg.get("r_arm_car_m", [0.0, 0.0]), dtype=float),
            config=_metric_config(config.get("metrics", {})),
        )
        save_metric_diagnostics(metric, output / "metrics", title="Swivel caster")
        metric_payload = metric.to_dict()
    elif not args.tracking_only:
        warnings.warn("car track not found; tracking outputs were written but comparison metrics were skipped", RuntimeWarning)

    payload = {
        "metadata": metadata,
        "timestamps_s": result.timestamps_s,
        "phi_rad": result.phi,
        "psi_rad": result.psi,
        "phi_valid": result.phi_valid,
        "psi_valid": result.psi_valid,
        "interval_quality": result.interval_quality,
        "phi_segment_id": result.phi_segment_id,
        "reference_corrected": result.reference_corrected,
        "tag_marker_id": [o.marker_id for o in result.tag_observations],
        "tag_contributing_ids": [o.contributing_ids for o in result.tag_observations],
        "tag_failure_reason": [o.failure_reason for o in result.tag_observations],
        "reference_phase_rad": result.reference_phase,
        "reference_phase_error_rad": result.reference_phase_error,
        "reference_valid": [item.valid for item in result.reference_observations],
        "metrics": metric_payload,
    }
    payload["angular_intervals"] = angular_intervals(payload)
    metadata["angular_rate_convention"] = "Signed phi/psi interval averages; null means unavailable; no smoothing"
    metadata["timestamp_source"] = "frame_index / reported_fps"
    write_angular_csv(output / "angular_motion.csv", payload["angular_intervals"])
    result_path = output / "results.json"
    result_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    print(f"Processed {result.frame_count} frames at {fps:.3f} FPS")
    print(f"Tag coverage:  {summary['tag_detection_coverage']:.1%}")
    print(f"Roll coverage: {summary['roll_valid_coverage']:.1%}")
    print(f"Accumulated roll complete: {summary['accumulated_phi_complete_at_end']}")
    print(f"Results: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
