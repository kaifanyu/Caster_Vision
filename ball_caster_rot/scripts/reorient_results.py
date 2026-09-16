#!/usr/bin/env python
"""Re-express saved tracking measurements in a corrected initial ball frame.

The original full XYZ decompositions retain the measured camera rotations.
Reconstructing them avoids repeating feature tracking. This cannot recover
failed frame-pair motion or retroactively change camera or circle calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.spatial.transform import Rotation

from ballrot.config import (
    camera_matrix,
    configured_circle,
    configured_frame,
    distortion_coefficients,
    load_config,
    measurement_frame,
    resolve_from_config,
)
from ballrot.diagnostics import (
    FrameQuality,
    HemisphereQuality,
    summarize_quality,
    write_run_outputs,
)
from ballrot.integrate import decompose_hemispheres
from ballrot.io_frames import FrameSource
from ballrot.rotation import is_rotation_matrix
from scripts.run import _acceptance, _check_assumptions, _swivel_axis_check


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--results", type=Path, required=True, help="Original results.json")
    result.add_argument(
        "--output", type=Path,
        help="Output directory (default: config output.dir); must differ from source",
    )
    result.add_argument("--strict", action="store_true", help="Exit 2 if consistency checks fail")
    return result


def reconstruct_camera_rotations(payload: Mapping[str, Any]) -> SimpleNamespace:
    """Recover measured rotations and saved pose-validity masks from JSON.

    Explicit pose validity takes precedence over raw frame-pair solve success:
    a keyframe may recover a pose after an unsuccessful adjacent solve, and a
    successful adjacent solve can still fail temporal quality checks. Missing
    poses hold the preceding orientation for display and remain invalid.
    """

    metadata = payload["metadata"]
    old_frame = np.asarray(metadata["R_bc"], dtype=float)
    if not is_rotation_matrix(old_frame, atol=1e-6):
        raise ValueError("source metadata.R_bc must be a proper rotation matrix")
    frames = payload["frames"]
    times = np.asarray(frames["time_s"], dtype=float)
    count = len(times)
    if times.shape != (count,) or count < 2 or not np.all(np.isfinite(times)):
        raise ValueError("source must contain at least two finite frame timestamps")
    if np.any(np.diff(times) <= 0):
        raise ValueError("source timestamps must be strictly increasing")
    if metadata.get("frame_count") != count:
        raise ValueError("source metadata.frame_count differs from frame arrays")
    for key, values in frames.items():
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"source frames.{key} has an inconsistent frame count")

    raw_quality = payload["quality"]
    if not isinstance(raw_quality, list) or len(raw_quality) != count - 1:
        raise ValueError("source must have one quality record for every frame pair")
    qualities: list[FrameQuality] = []
    for frame_index, item in enumerate(raw_quality, start=1):
        if item.get("frame_index") != frame_index:
            raise ValueError("source quality frame indices must be consecutive, starting at 1")
        hemispheres = {}
        for name in ("top", "bottom"):
            data = dict(item[name])
            if not isinstance(data.get("success"), bool):
                raise ValueError(f"source {name} success must be a boolean at frame {frame_index}")
            # JSON stores non-finite quality statistics as null.
            for key in (
                "inlier_ratio", "mean_residual_deg", "median_residual_deg", "median_fb_error_px"
            ):
                if data.get(key) is None:
                    data[key] = float("nan")
            hemispheres[name] = HemisphereQuality(**data)
        qualities.append(FrameQuality(frame_index, **hemispheres))

    temporal = payload.get("temporal")
    if temporal is not None and not isinstance(temporal, Mapping):
        raise ValueError("source temporal diagnostics must be a mapping")
    offline = payload.get("offline")
    if offline is not None and not isinstance(offline, Mapping):
        raise ValueError("source offline diagnostics must be a mapping")
    recovered: dict[str, Any] = {
        "qualities": qualities, "frame_count": count, "temporal": temporal,
        "offline": offline,
    }
    for name in ("top", "bottom"):
        angles = np.column_stack([
            np.asarray(frames[f"{component}_{name}_rad"], dtype=float)
            for component in ("alpha", "gamma", "beta")
        ])
        if angles.shape != (count, 3):
            raise ValueError(f"source {name} Euler arrays must be one-dimensional")
        validity_key = f"valid_{name}"
        if validity_key in frames:
            flags = frames[validity_key]
            if any(not isinstance(flag, bool) for flag in flags):
                raise ValueError(f"source frames.{validity_key} must contain only booleans")
            valid = np.asarray(flags, dtype=bool)
            if not valid[0]:
                raise ValueError(f"source {name} frame zero must have a valid identity pose")
        else:
            valid = np.array([True] + [getattr(q, name).success for q in qualities], dtype=bool)
        if not np.all(np.isfinite(angles[valid])):
            raise ValueError(f"source {name} has missing or non-finite angles for a successful frame")
        ball_rotations = Rotation.from_euler("XYZ", angles[valid]).as_matrix()
        if not np.allclose(ball_rotations[0], np.eye(3), atol=1e-6, rtol=0.0):
            raise ValueError(f"source {name} frame zero must represent identity rotation")
        absolute = np.empty((count, 3, 3), dtype=float)
        absolute[valid] = old_frame @ ball_rotations @ old_frame.T
        absolute[0] = np.eye(3)
        for index in range(1, count):
            if not valid[index]:
                absolute[index] = absolute[index - 1]
        recovered[f"{name}_absolute"] = absolute
        recovered[f"{name}_step_valid"] = valid
    return SimpleNamespace(**recovered)


def _same_geometry(
    config: Mapping[str, Any], metadata: Mapping[str, Any],
    source: FrameSource,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], bool]:
    old_input = metadata.get("input")
    if not old_input or Path(old_input).expanduser().resolve() != source.path.resolve():
        raise ValueError("input.path differs from source measurements; retrack the new input")
    width, height = source.size
    K, approximate = camera_matrix(config.get("camera", {}), (height, width, 3))
    dist = distortion_coefficients(config.get("camera", {}))
    circle = configured_circle(config.get("circle", {}))
    if circle is None:
        raise ValueError("set the source run's measured circle explicitly before reorienting")
    for name, new in (("K", K), ("dist", dist), ("circle", circle)):
        old = np.asarray(metadata.get(name), dtype=float)
        new = np.asarray(new, dtype=float)
        if old.shape != new.shape or not np.allclose(old, new, atol=1e-8, rtol=1e-10):
            raise ValueError(f"{name} differs from source measurements; geometry changes require retracking")
    return K, dist, circle, approximate


def _source_times(source: FrameSource, count: int, fps_override: float | None) -> np.ndarray:
    if fps_override is not None:
        fps = float(fps_override)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("input.fps_override must be finite and positive")
    timestamps = []
    records = source.records() if fps_override is None else source.frames()
    for index, record in enumerate(records):
        if fps_override is None and record.index != index:
            raise ValueError("input frame indices must start at zero and be consecutive")
        timestamps.append(record.timestamp_s if fps_override is None else index / fps)
    if len(timestamps) != count:
        raise ValueError(
            f"input decoded {len(timestamps)} frames, source results contain {count}; "
            "use the same input.max_frames and video as the original run"
        )
    if fps_override is not None:
        return np.arange(count, dtype=float) / fps
    times = np.asarray(timestamps, dtype=float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("input must supply finite, strictly increasing presentation timestamps")
    return times - times[0]


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config, config_path = load_config(args.config)
    _check_assumptions(config)
    source_results = args.results.expanduser().resolve()
    output_dir = (
        args.output.expanduser().resolve() if args.output is not None
        else resolve_from_config(config_path, config.get("output", {}).get("dir", "out/real"))
    )
    if output_dir == source_results.parent or (output_dir / "results.json").resolve() == source_results:
        raise ValueError("output must differ from the source results directory; originals are preserved")
    payload = json.loads(source_results.read_text(encoding="utf-8"))
    if (payload.get("metadata", {}).get("mechanical_model", {}).get("enabled", False)
            or payload.get("mechanical", {}).get("config", {}).get("enabled", False)):
        raise ValueError(
            "mechanically constrained results must be refitted to change the initial frame; "
            "use scripts/refine_mechanical.py with their saved offline_observations.npz")
    recovered = reconstruct_camera_rotations(payload)
    input_cfg = config.get("input", {})
    if not input_cfg.get("path"):
        raise ValueError("input.path is required")
    source = FrameSource(
        resolve_from_config(config_path, input_cfg["path"]),
        input_type=input_cfg.get("type", "auto"),
        max_frames=input_cfg.get("max_frames"),
        image_fps=input_cfg.get("fps_override"),
    )
    K, dist, circle, approximate = _same_geometry(config, payload["metadata"], source)
    home = configured_frame(config.get("frame_calib", {}))
    effective = measurement_frame(config.get("frame_calib", {}))
    if home is None or effective is None:
        raise ValueError("frame_calib.R_bc is required")
    times = _source_times(source, recovered.frame_count, input_cfg.get("fps_override"))
    motion = decompose_hemispheres(
        recovered.top_absolute, recovered.bottom_absolute, effective,
        valid_top=recovered.top_step_valid, valid_bottom=recovered.bottom_step_valid,
    )
    swivel_check = _swivel_axis_check(recovered, effective)
    summary = summarize_quality(
        recovered.qualities, motion, recovered.top_absolute, recovered.bottom_absolute
    )
    summary["swivel_axis_check"] = swivel_check
    extra_summary = {"swivel_axis_check": swivel_check}
    if recovered.temporal is not None:
        source_summary = recovered.temporal.get("summary", {})
        if recovered.offline is not None:
            # The temporal summary describes the original forward pass; final
            # offline validity belongs in its own summary below.
            temporal_summary = dict(source_summary)
        else:
            temporal_summary = {
                name: {
                    **source_summary.get(name, {}),
                    "unresolved_frames": np.flatnonzero(
                        ~getattr(recovered, f"{name}_step_valid")
                    ).tolist(),
                }
                for name in ("top", "bottom")
            }
        summary["temporal_tracking"] = temporal_summary
        extra_summary["temporal_tracking"] = temporal_summary
    if recovered.offline is not None:
        source_summary = recovered.offline.get("summary", {})
        offline_summary = {
            name: {
                **source_summary.get(name, {}),
                "unresolved_frames": np.flatnonzero(
                    ~getattr(recovered, f"{name}_step_valid")
                ).tolist(),
            }
            for name in ("top", "bottom")
        }
        summary["offline_tracking"] = offline_summary
        extra_summary["offline_tracking"] = offline_summary
    passed, failures = _acceptance(summary, int(config.get("estimate", {}).get("min_inliers", 8)))
    timing_source = (
        "input.fps_override" if input_cfg.get("fps_override") is not None
        else getattr(source, "timing_source", "video presentation timestamps")
    )
    metadata = {
        **payload["metadata"],
        "config": str(config_path), "input": str(source.path),
        "K": K, "dist": dist, "circle": circle, "K_was_approximated": approximate,
        "R_bc_home": home, "R_bc": effective,
        "initial_roll_deg": float(config.get("frame_calib", {}).get("initial_roll_deg", 0.0)),
        "top_shell_sign": int(config.get("frame_calib", {}).get("top_shell_sign", 1)),
        "source_results": str(source_results), "timing_source": timing_source,
        "reorientation_note": (
            "Camera rotations reconstructed from saved full intrinsic XYZ components. "
            "Offline-refined and recovered poses, final pose validity, and original "
            "temporal/offline diagnostics are preserved. No missing motion was interpolated. "
            "Rates were recomputed on native timestamps with local polynomial fits using "
            "the source offline settings; invalid gaps are never bridged. Shared roll "
            "rates require valid observations from both shells. No feature "
            "tracking or offline pose refinement was repeated."
            if recovered.offline is not None else
            "Camera rotations reconstructed from saved full intrinsic XYZ components. "
            "Unresolved poses remain invalid; keyframe-recovered absolute poses and "
            "temporal diagnostics are preserved. No missing motion was interpolated. "
            "Velocity gradients retain the original behavior of bridging valid samples "
            "across gaps. No feature tracking was repeated."
            if recovered.temporal is not None else
            "Camera rotations reconstructed from saved full intrinsic XYZ components. "
            "Failed frame pairs remain missing and hold their prior accumulated rotation. "
            "Reorientation cannot recover their missing motion; subsequent accumulated "
            "orientations can remain inaccurate. Velocity gradients retain the original "
            "behavior of bridging valid samples across gaps. No feature tracking was repeated."
        ),
    }
    paths = write_run_outputs(
        output_dir, times, motion, recovered.qualities,
        recovered.top_absolute, recovered.bottom_absolute, metadata=metadata,
        extra_summary=extra_summary, temporal=recovered.temporal, offline=recovered.offline,
    )
    print(f"Reoriented {recovered.frame_count} saved frames; original results preserved.")
    print(f"Initial roll: {metadata['initial_roll_deg']:g} deg; timing: {timing_source}")
    print(f"SELF-CONSISTENCY: {'PASS' if passed else 'WARN'}")
    for failure in failures:
        print(f"  - {failure}")
    if recovered.offline is not None:
        print("Offline poses and diagnostics were preserved; gap-safe rates were recomputed.")
    elif recovered.temporal is not None:
        print("Unresolved poses and recovered keyframe poses were preserved; tracking was not repeated.")
    else:
        print("Missing tracking steps were preserved; this operation cannot repair them.")
    print("Outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 2 if args.strict and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
