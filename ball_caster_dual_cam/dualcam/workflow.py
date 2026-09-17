"""User-facing calibration and motion workflows with provenance checks."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import (CAMERA_NAMES, load_config, load_rig, read_yaml, rotation,
                     sha256, write_json, write_yaml)
from .solver import fit_joint
from .tracking import (initialize_angles, initialize_axes, initialize_pivot,
                       save_tracks, track_session)


class WorkflowError(ValueError):
    """A failed workflow has saved diagnostics, but no accepted calibration."""


def _output_directory(output):
    output = Path(output).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def calibration_hashes(cfg):
    return {**{name: sha256(cfg["cameras"][name]["intrinsics"]) for name in CAMERA_NAMES},
            "stereo": sha256(cfg["stereo"]["path"])}


def _solver_options(cfg):
    options = dict(cfg.get("solver", {}))
    options.setdefault("min_track_frames", int(cfg.get("tracking", {}).get("min_track_length", 3)))
    options["red_sign"] = cfg["geometry"].get("red_shell_sign", 1)
    return options


def _record_provenance(report, cfg, intrinsics):
    timing_verified = bool(cfg["timing"].get("verified", False))
    profiles = {name: bool(info.get("profile_verified", info.get("capture_profile") is not None))
                for name, info in zip(CAMERA_NAMES, intrinsics)}
    report["provenance"] = {
        "intrinsics_capture_profile_verified": profiles,
        "timing_verified_by_operator": timing_verified,
        "timestamp_source": "host receive timestamps; not hardware exposure timestamps",
        "validation_scope": "Solver success checks image reprojection, track support, and applicable observability. It does not independently verify exposure synchronization or sensor latency.",
    }
    warnings = []
    for name, verified in profiles.items():
        if not verified:
            warnings.append(f"{name}: intrinsics have no recorded capture_profile; matching focus, zoom and camera settings have not been verified against intrinsic calibration.")
    if not timing_verified:
        warnings.append("Camera timing is unverified (timing.verified=false). Host receive timestamps do not establish exposure synchronization; solver consistency does not validate sensor latency. Measure a shared event and set the inter-camera offset before claiming motion accuracy.")
    report["warnings"] = warnings
    for warning in warnings:
        print(f"Warning: {warning}", flush=True)


def load_axes(cfg):
    """Reject axes estimated with different intrinsics, extrinsics or geometry."""
    path = Path(cfg["axes"]["path"])
    axes = read_yaml(path)
    F = rotation(axes.get("R_bc"), "axes.R_bc")
    pivot = np.asarray(axes.get("pivot_c920_m"), dtype=float)
    if pivot.shape != (3,) or not np.isfinite(pivot).all():
        raise ValueError("axes.pivot_c920_m must be a finite metric C920 position")
    if axes.get("calibration_hashes") != calibration_hashes(cfg):
        raise ValueError("Axis calibration is stale: intrinsic/stereo calibration hashes changed or are missing. Recalibrate axes.")
    for key in ("radius_m", "gap_m"):
        if key not in axes or not np.isclose(axes[key], cfg["geometry"][key], rtol=1e-12, atol=1e-12):
            raise ValueError(f"Axis calibration geometry.{key} differs from the rig. Recalibrate axes.")
    if axes.get("red_shell_sign") != cfg["geometry"].get("red_shell_sign", 1):
        raise ValueError("Axis calibration red_shell_sign differs from the rig. Recalibrate axes.")
    saved_offset = axes.get("timing", {}).get("brio_offset_s")
    if saved_offset is None or not np.isclose(saved_offset, cfg["timing"].get("brio_offset_s", 0), rtol=0, atol=1e-12):
        raise ValueError("Axis calibration camera time offset differs from the rig. Recalibrate axes using the corrected timing.")
    return {**axes, "R_bc": F, "pivot_c920_m": pivot, "sha256": sha256(path), "path": str(path)}


def _dataset(tracked, cameras, F, mode, initial_roll_rad=0.0):
    return {"times": tracked["times"], "observations": tracked["observations"],
            "mode": mode, "initial_roll_rad": initial_roll_rad,
            "initial_angles": initialize_angles(tracked, cameras, F, mode,
                                                  initial_roll_rad=initial_roll_rad)}


def _session_info(tracked):
    return {"session": tracked["session"], "pairs": tracked["pairs"],
            "session_report": tracked["session_report"], "mode": tracked["session_mode"]}


def calibrate_axes(config_path, roll_session, swivel_session, output, *,
                   max_frames=180, roll_sign=1, swivel_sign=1):
    """Jointly refine axes/pivot from separate pure-motion, known-home clips.

    Both recordings must begin with the caster held at the SAME known home pose.
    Their initial shell phases define zero independently and are nuisance gauges.
    A failed fit still produces report.json, but never overwrites accepted axes.
    """
    output = _output_directory(output)
    report_path = output / "report.json"
    report = {"kind": "axis_calibration", "success": False, "status": "failed",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "roll_session": str(Path(roll_session).expanduser().resolve()),
              "swivel_session": str(Path(swivel_session).expanduser().resolve())}
    try:
        if roll_sign not in (-1, 1) or swivel_sign not in (-1, 1):
            raise ValueError("roll_sign and swivel_sign must be +1 or -1")
        cfg = load_config(config_path)
        cameras, intrinsics, stereo = load_rig(cfg)
        _record_provenance(report, cfg, intrinsics)
        report["config"] = cfg
        report["calibration_hashes"] = calibration_hashes(cfg)
        tracked = []
        for label, session in (("roll", roll_session), ("swivel", swivel_session)):
            clip = track_session(session, cfg, max_frames=max_frames)
            save_tracks(output / f"{label}_tracks.npz", clip)
            report[label] = _session_info(clip)
            if clip["session_mode"] != label:
                raise ValueError(f"Expected a --mode {label} recording, got {clip['session_mode']!r} for {session}")
            tracked.append(clip)
        F, initialization = initialize_axes(*tracked, cameras, cfg,
                                            roll_sign=roll_sign, swivel_sign=swivel_sign)
        pivot = initialize_pivot(tracked[0], cameras, cfg)
        report["initialization"] = {"axes": initialization, "R_bc": F, "pivot_c920_m": pivot}
        datasets = [_dataset(clip, cameras, F, label) for clip, label in zip(tracked, ("roll", "swivel"))]
        geo = cfg["geometry"]
        print(f"Fitting shared axes and pivot: {len(datasets[0]['times'])} roll pairs and {len(datasets[1]['times'])} swivel pairs. This joint optimization may take time.", flush=True)
        result = fit_joint(datasets, cameras, F, pivot, geo["radius_m"], geo["gap_m"],
                           calibrate_axes=True, refine_pivot=True, options=_solver_options(cfg))
        report["fit"] = result
        report["success"] = bool(result["success"])
        report["status"] = "passed" if result["success"] else "rejected"
        report["accepted_axes_path"] = cfg["axes"]["path"] if result["success"] else None
        # Diagnostics are materialized before the only accepted-calibration write.
        write_json(report_path, report)
        if result["success"]:
            axes = {"calibration_kind": "dual_camera_axes", "created_utc": report["created_utc"],
                    "R_bc": result["F"], "pivot_c920_m": result["pivot"],
                    "radius_m": geo["radius_m"], "gap_m": geo["gap_m"],
                    "red_shell_sign": geo.get("red_shell_sign", 1),
                    "calibration_hashes": report["calibration_hashes"], "timing": cfg["timing"],
                    "home_reference": {"description": "Both clips start at the same known caster home; angles at each clip's first paired frame are [0,0,0].",
                                       "roll_session": report["roll_session"], "swivel_session": report["swivel_session"],
                                       "frame0_angles_rad": [0., 0., 0.], "roll_sign": roll_sign, "swivel_sign": swivel_sign},
                    "diagnostics_report": str(report_path)}
            write_yaml(cfg["axes"]["path"], axes)
        return report
    except Exception as exc:
        report.update(success=False, status="failed", error=str(exc))
        write_json(report_path, report)
        raise WorkflowError(f"{exc}\nDiagnostics: {report_path}. Accepted axes were not replaced.") from exc


def _write_motion_csv(path, fit):
    dataset = fit["datasets"][0]
    times = np.asarray(dataset["times"], float)
    angles = np.asarray(dataset["angles"], float)
    valid = np.asarray(dataset["valid"], bool)
    if angles.shape != (len(times), 3) or valid.shape != angles.shape:
        raise ValueError("Solver returned inconsistent angle/valid/time shapes")
    valid &= np.isfinite(angles)
    if not fit["success"]:
        valid[:] = False
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "time_s", "alpha_rad", "beta_red_rad", "beta_green_rad",
                         "alpha_deg", "beta_red_deg", "beta_green_deg", "valid_alpha", "valid_red", "valid_green"])
        for frame, (time, values, supported) in enumerate(zip(times, angles, valid)):
            radians = [float(value) if okay else "" for value, okay in zip(values, supported)]
            degrees = [float(np.degrees(value)) if okay else "" for value, okay in zip(values, supported)]
            writer.writerow([frame, float(time), *radians, *degrees, *[int(x) for x in supported]])


def run_motion(config_path, session, output, *, initial_roll_deg, max_frames=180):
    """Fit one motion clip; alpha starts at a supplied absolute roll angle.

    Both shell spins are relative to this recording's first paired frame. No
    absolute painted-feature correspondence across sessions is assumed.
    """
    output = _output_directory(output)
    report_path = output / "results.json"
    report = {"kind": "motion", "success": False, "status": "failed",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "session": str(Path(session).expanduser().resolve()),
              "initial_roll_deg": initial_roll_deg,
              "spin_reference": "Independent red/green shell spins are zero at the first paired frame."}
    try:
        if not np.isfinite(initial_roll_deg):
            raise ValueError("initial_roll_deg must be a known finite angle relative to the calibrated home")
        cfg = load_config(config_path)
        cameras, intrinsics, stereo = load_rig(cfg)
        _record_provenance(report, cfg, intrinsics)
        axes = load_axes(cfg)
        report.update(config=cfg, geometry=cfg["geometry"], calibration_hashes=calibration_hashes(cfg),
                      axes_sha256=axes["sha256"])
        tracked = track_session(session, cfg, max_frames=max_frames)
        save_tracks(output / "tracks.npz", tracked)
        report.update(pairs=tracked["pairs"], session_report=tracked["session_report"])
        if tracked["session_mode"] != "motion":
            raise ValueError(f"Expected a --mode motion recording, got {tracked['session_mode']!r}")
        F, pivot = axes["R_bc"], axes["pivot_c920_m"]
        dataset = _dataset(tracked, cameras, F, "motion", initial_roll_rad=float(np.deg2rad(initial_roll_deg)))
        geo = cfg["geometry"]
        print(f"Fitting motion: {len(dataset['times'])} paired frames, fixed calibrated axes/pivot. This joint optimization may take time.", flush=True)
        result = fit_joint([dataset], cameras, F, pivot, geo["radius_m"], geo["gap_m"],
                           calibrate_axes=False, refine_pivot=False, options=_solver_options(cfg))
        report.update(fit=result, success=bool(result["success"]),
                      status="passed" if result["success"] else "rejected")
        write_json(report_path, report)
        _write_motion_csv(output / "results.csv", result)
        return report
    except Exception as exc:
        report.update(success=False, status="failed", error=str(exc))
        write_json(report_path, report)
        raise WorkflowError(f"{exc}\nDiagnostics: {report_path}") from exc
