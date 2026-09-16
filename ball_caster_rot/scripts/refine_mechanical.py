#!/usr/bin/env python
"""Jointly refit saved image observations with the caster's mechanical constraints.

This uses the saved feature observations, without decoding or tracking the video
again. Camera/circle geometry must match the source run. The calibrated frame,
initial roll, shell mapping and configured shell gap may be changed for this fit.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.spatial.transform import Rotation

from ballrot.config import (
    camera_matrix, configured_circle, configured_frame, distortion_coefficients,
    load_config, measurement_frame, resolve_from_config,
)
from ballrot.diagnostics import (
    FrameQuality, HemisphereQuality, _json_clean, unconstrained_result, write_run_outputs,
)
from ballrot.integrate import decompose_hemispheres, mechanical_motion
from ballrot.mechanical import MechanicalConfig, refine_mechanical_trajectory
from ballrot.offline import OfflineConfig, SurfaceObservation
from ballrot.offline_observations import save_observations
from ballrot.rotation import is_rotation_matrix
from ballrot.sphere import sphere_pose_from_circle
from scripts.run import _check_assumptions, _print_mechanical_status


SHELLS = ("top", "bottom")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--results", type=Path, required=True, help="Source results.json")
    result.add_argument(
        "--observations", type=Path,
        help="Numeric observation archive (default: offline_observations.npz beside results)",
    )
    result.add_argument("--output", type=Path, required=True, help="Separate output directory")
    return result


def _same_array(name: str, actual, expected, *, atol=1e-8) -> None:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if (actual.shape != expected.shape or not np.all(np.isfinite(actual))
            or not np.all(np.isfinite(expected))
            or not np.allclose(actual, expected, atol=atol, rtol=1e-10)):
        raise ValueError(f"{name} differs from source measurements; use the matching archive/config")


def _source_quality(payload: Mapping[str, Any], count: int) -> list[FrameQuality]:
    items = payload.get("quality")
    if not isinstance(items, list) or len(items) != count - 1:
        raise ValueError("source must have one quality record for every frame pair")
    result = []
    for index, item in enumerate(items, start=1):
        if item.get("frame_index") != index:
            raise ValueError("source quality frame indices must be consecutive, starting at 1")
        shells = {}
        for name in SHELLS:
            quality = dict(item[name])
            if not isinstance(quality.get("success"), bool):
                raise ValueError(f"source {name} quality.success must be a boolean")
            for field in (
                "inlier_ratio", "mean_residual_deg", "median_residual_deg", "median_fb_error_px",
            ):
                if quality.get(field) is None:
                    quality[field] = float("nan")
            shells[name] = HemisphereQuality(**quality)
        result.append(FrameQuality(index, **shells))
    return result


def _rotations(arrays, key: str, count: int) -> np.ndarray:
    matrices = np.asarray(arrays[key], dtype=float)
    if (matrices.shape != (count, 3, 3) or not np.all(np.isfinite(matrices))
            or not np.allclose(matrices.transpose(0, 2, 1) @ matrices, np.eye(3), atol=1e-6)
            or not np.allclose(np.linalg.det(matrices), 1.0, atol=1e-6)):
        raise ValueError(f"archive {key} must contain {count} proper rotation matrices")
    if not np.allclose(matrices[0], np.eye(3), atol=1e-6):
        raise ValueError(f"archive {key} frame zero must be identity")
    return matrices.copy()


def _validity(arrays, key: str, count: int) -> np.ndarray:
    flags = np.asarray(arrays[key])
    if flags.shape != (count,) or flags.dtype.kind != "b":
        raise ValueError(f"archive {key} must contain {count} boolean flags")
    return flags.copy()


def load_archive(path: Path, payload: Mapping[str, Any]) -> SimpleNamespace:
    """Read only numeric arrays and verify the archive belongs to these results."""
    metadata, frames = payload["metadata"], payload["frames"]
    times = np.asarray(frames["time_s"], dtype=float)
    count = len(times)
    if (times.shape != (count,) or count < 2 or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("source must contain at least two strictly increasing finite timestamps")
    if metadata.get("frame_count") != count:
        raise ValueError("source metadata.frame_count differs from timestamps")
    if any(not isinstance(values, list) or len(values) != count for values in frames.values()):
        raise ValueError("source frame arrays have inconsistent lengths")
    frame = np.asarray(metadata["R_bc"], dtype=float)
    if not is_rotation_matrix(frame, atol=1e-6):
        raise ValueError("source metadata.R_bc must be a proper rotation matrix")
    qualities = _source_quality(payload, count)
    with np.load(path, allow_pickle=False) as archive:
        version = np.asarray(archive.get("schema_version", 0))
        if version.shape != () or version.dtype.kind not in "iu" or int(version) < 2:
            raise ValueError("archive schema 2 or later with source-template identity provenance is required")
        required = {
            "coordinate_system", "landmark_id", "landmark_family", "landmark_source_frame",
            "landmark_source_track_id", "timestamps_s", "K", "sphere_center", "sphere_radius",
        }
        for name in SHELLS:
            required.update(f"{name}_{key}" for key in (
                "frame_index", "track_id", "uv", "weight", "initial_rotations", "initial_valid",
                "refined_rotations", "refined_valid",
            ))
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError("archive is missing identity/measurement fields: " + ", ".join(missing))
        if str(archive["coordinate_system"].item()) != "undistorted_pixels; camera_relative_rotations":
            raise ValueError("archive coordinate system must use undistorted pixels and camera-relative rotations")
        _same_array("archive timestamps", archive["timestamps_s"], times, atol=1e-10)
        K = np.asarray(archive["K"], dtype=float)
        _same_array("archive K", K, metadata["K"])
        center = np.asarray(archive["sphere_center"], dtype=float)
        radius = float(archive["sphere_radius"])
        expected_center, expected_radius = sphere_pose_from_circle(*metadata["circle"], K)
        _same_array("archive sphere_center", center, expected_center)
        _same_array("archive sphere_radius", radius, expected_radius)
        if "sphere_center" in metadata:
            _same_array("metadata sphere_center", center, metadata["sphere_center"])
        if "sphere_radius" in metadata:
            _same_array("metadata sphere_radius", radius, metadata["sphere_radius"])
        identifiers = np.asarray(archive["landmark_id"])
        families = np.asarray(archive["landmark_family"])
        source_frames = np.asarray(archive["landmark_source_frame"])
        source_ids = np.asarray(archive["landmark_source_track_id"])
        if (identifiers.ndim != 1 or identifiers.dtype.kind not in "iu"
                or len(np.unique(identifiers)) != len(identifiers)
                or families.shape != identifiers.shape or families.dtype.kind not in "US"
                or source_frames.shape != identifiers.shape or source_frames.dtype.kind not in "iu"
                or source_ids.shape != identifiers.shape or source_ids.dtype.kind not in "iu"
                or np.any(source_frames < 0) or np.any(source_frames >= count)
                or not np.all(np.isin(families, ["adjacent", "anchor", "reverse"]))):
            raise ValueError("archive landmark source-template identity provenance is invalid")
        landmark_sources = {
            int(identifier): (str(family), int(source_index), int(source_id))
            for identifier, family, source_index, source_id
            in zip(identifiers, families, source_frames, source_ids)
        }
        observations, initial_rotations, initial_valid = {}, {}, {}
        unconstrained_rotations, unconstrained_valid = {}, {}
        for name in SHELLS:
            indices = np.asarray(archive[f"{name}_frame_index"])
            ids = np.asarray(archive[f"{name}_track_id"])
            uv = np.asarray(archive[f"{name}_uv"], dtype=float)
            weights = np.asarray(archive[f"{name}_weight"], dtype=float)
            if (indices.ndim != 1 or indices.dtype.kind not in "iu"
                    or ids.shape != indices.shape or ids.dtype.kind not in "iu"
                    or uv.shape != (len(indices), 2) or not np.all(np.isfinite(uv))
                    or weights.shape != indices.shape or not np.all(np.isfinite(weights))
                    or np.any(weights <= 0) or np.any(indices < 0) or np.any(indices >= count)
                    or not np.all(np.isin(ids, identifiers))):
                raise ValueError(f"archive {name} observations have invalid indices, pixels, weights or identities")
            observations[name] = [
                SurfaceObservation(int(index), int(identifier), point.copy(), float(weight))
                for index, identifier, point, weight in zip(indices, ids, uv, weights)
            ]
            initial_rotations[name] = _rotations(archive, f"{name}_initial_rotations", count)
            initial_valid[name] = _validity(archive, f"{name}_initial_valid", count)
            refined = _rotations(archive, f"{name}_refined_rotations", count)
            refined_valid = _validity(archive, f"{name}_refined_valid", count)
            flags = frames.get(f"valid_{name}")
            if (flags is None or any(not isinstance(flag, bool) for flag in flags)
                    or not np.array_equal(refined_valid, flags)):
                raise ValueError(f"archive {name} refined validity differs from source results")
            angles = np.column_stack([
                np.asarray(frames[f"{component}_{name}_rad"], dtype=float)
                for component in ("alpha", "gamma", "beta")
            ])
            if np.any(refined_valid):
                if not np.all(np.isfinite(angles[refined_valid])):
                    raise ValueError(f"source {name} valid poses have missing angles")
                matrices = frame @ Rotation.from_euler("XYZ", angles[refined_valid]).as_matrix() @ frame.T
                _same_array(f"archive {name} refined rotations", refined[refined_valid], matrices, atol=1e-6)
            raw_keys = [f"{name}_unconstrained_rotations", f"{name}_unconstrained_valid"]
            if any(key in archive for key in raw_keys) and not all(key in archive for key in raw_keys):
                raise ValueError(f"archive {name} unconstrained pose fields must occur together")
            if all(key in archive for key in raw_keys):
                unconstrained_rotations[name] = _rotations(archive, raw_keys[0], count)
                unconstrained_valid[name] = _validity(archive, raw_keys[1], count)
            elif payload.get("mechanical") is not None:
                raise ValueError("mechanical source archive lacks preserved unconstrained poses; use the original run")
            else:
                unconstrained_rotations[name], unconstrained_valid[name] = refined, refined_valid
    return SimpleNamespace(
        timestamps=times, frame_count=count, K=K, center=center, radius=radius,
        observations=observations, initial_rotations=initial_rotations, initial_valid=initial_valid,
        unconstrained_rotations=unconstrained_rotations, unconstrained_valid=unconstrained_valid,
        landmark_sources=landmark_sources, qualities=qualities,
    )


def _same_geometry(config, config_path: Path, metadata) -> None:
    input_path = config.get("input", {}).get("path")
    if not input_path or resolve_from_config(config_path, input_path) != Path(metadata["input"]).resolve():
        raise ValueError("input.path differs from source measurements; use the source video's configuration")
    camera = config.get("camera", {})
    if camera.get("K") is None:
        raise ValueError("set camera.K to the source run's saved K before refitting saved observations")
    K, _ = camera_matrix(camera, (1, 1, 3))
    circle = configured_circle(config.get("circle", {}))
    if circle is None:
        raise ValueError("set circle to the source run's saved circle before refitting observations")
    for name, value in (("K", K), ("dist", distortion_coefficients(camera)), ("circle", circle)):
        _same_array(f"configured {name}", value, metadata[name])


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config, config_path = load_config(args.config)
    _check_assumptions(config)
    source_results = args.results.expanduser().resolve()
    observations_path = (args.observations.expanduser().resolve() if args.observations else
                         source_results.with_name("offline_observations.npz"))
    output_dir = args.output.expanduser().resolve()
    if output_dir in (source_results.parent, observations_path.parent):
        raise ValueError("output must differ from the source results/archive directories; originals are preserved")
    payload = json.loads(source_results.read_text(encoding="utf-8"))
    _same_geometry(config, config_path, payload["metadata"])
    archive = load_archive(observations_path, payload)
    home = configured_frame(config.get("frame_calib", {}))
    frame = measurement_frame(config.get("frame_calib", {}))
    if home is None or frame is None:
        raise ValueError("frame_calib.R_bc is required")
    mechanical_config = MechanicalConfig.from_mapping({**config.get("mechanical", {}), "enabled": True})
    offline_config = OfflineConfig.from_mapping(
        config.get("offline", payload.get("offline", {}).get("config", {})))
    top_shell_sign = int(config.get("frame_calib", {}).get("top_shell_sign", 1))
    print(f"Refitting {archive.frame_count} saved frames with shared roll and independent shell spins.")
    print(f"Fixed rim gap / ball diameter: {mechanical_config.gap_fraction:g}")
    fitted = refine_mechanical_trajectory(
        archive.observations, archive.unconstrained_rotations, archive.unconstrained_valid,
        archive.K, archive.center, frame, radius=archive.radius, config=mechanical_config,
        offline_config=offline_config, top_shell_sign=top_shell_sign,
    )
    motion = mechanical_motion(fitted.alpha, fitted.beta, fitted.valid)
    raw_motion = decompose_hemispheres(
        archive.unconstrained_rotations["top"], archive.unconstrained_rotations["bottom"], frame,
        valid_top=archive.unconstrained_valid["top"], valid_bottom=archive.unconstrained_valid["bottom"],
    )
    raw = unconstrained_result(
        raw_motion, archive.unconstrained_rotations["top"], archive.unconstrained_rotations["bottom"],
        archive.qualities,
    )
    metadata = {
        **payload["metadata"], "config": str(config_path), "R_bc_home": home, "R_bc": frame,
        "initial_roll_deg": float(config.get("frame_calib", {}).get("initial_roll_deg", 0)),
        "top_shell_sign": top_shell_sign,
        "mechanical_model": {**asdict(mechanical_config), "model": "shared_roll_independent_spin"},
        "source_results": str(source_results), "source_observations": str(observations_path),
        "mechanical_refit_note": (
            "Joint fit of preserved undistorted image observations. Camera and sphere geometry "
            "are unchanged; the configured mechanism frame and gap were applied anew. "
            "Unconstrained poses and original tracking diagnostics are preserved. Invalid "
            "constrained poses are not substituted with independent shell motion. Shared "
            "roll and zero sideways tilt are imposed constraints, not accuracy measurements."
        ),
    }
    extra_summary = {
        key: payload.get("summary", {})[key] for key in ("temporal_tracking", "offline_tracking")
        if key in payload.get("summary", {})
    }
    extra_summary["mechanical_tracking"] = fitted.diagnostics["summary"]
    paths = write_run_outputs(
        output_dir, archive.timestamps, motion, archive.qualities,
        fitted.rotations["top"], fitted.rotations["bottom"], metadata=metadata,
        extra_summary=extra_summary, temporal=payload.get("temporal"), offline=payload.get("offline"),
        mechanical=fitted.diagnostics, unconstrained=raw,
    )
    report_path = output_dir / "mechanical_report.json"
    report_path.write_text(json.dumps(_json_clean(fitted.diagnostics), indent=2) + "\n", encoding="utf-8")
    paths["mechanical_report"] = report_path
    paths["observations"] = save_observations(
        output_dir / "offline_observations.npz", archive.observations,
        timestamps=archive.timestamps, K=archive.K, center=archive.center, radius=archive.radius,
        initial_rotations=archive.initial_rotations, initial_valid=archive.initial_valid,
        refined_rotations=fitted.rotations, refined_valid=fitted.valid,
        unconstrained_rotations=archive.unconstrained_rotations, unconstrained_valid=archive.unconstrained_valid,
        landmark_sources=archive.landmark_sources,
    )
    _print_mechanical_status(fitted.diagnostics)
    if any(not np.all(fitted.valid[name]) for name in SHELLS):
        print("UNRESOLVED: unsupported constrained poses remain missing; inspect mechanical_report.json.")
    print("Original results and observations preserved. Outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("Render the constrained results with:")
    print(f'  .\\.venv\\Scripts\\python.exe .\\scripts\\simulate_measured.py --config "{config_path}" '
          f'--results "{output_dir / "results.json"}" --output "{output_dir / "simulation"}" --tests axes replay')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
