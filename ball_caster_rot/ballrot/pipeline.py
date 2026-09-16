"""Top-to-bottom speckle rotation pipeline used by synthetic and real runs."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

import cv2
import numpy as np

from .calibration import detect_circle_auto
from .camera import undistort_image
from .diagnostics import FrameQuality, HemisphereQuality
from .estimate import FrameRotationEstimate, solve_frame_increments
from .integrate import DecomposedMotion, accumulate_increments, decompose_hemispheres, mechanical_motion
from .mechanical import MechanicalConfig, refine_mechanical_trajectory
from .offline import OfflineConfig, refine_trajectory
from .offline_observations import OfflineObservationCollector
from .segment import SegmentationMasks, segment_frame
from .sphere import sphere_pose_from_circle
from .track import FrameMatches, KLTConfig, KLTTracker, PersistentKLTTracker
from .temporal import TemporalConfig, TemporalRotationTracker, safe_tracking_masks, temporal_report


@dataclass(frozen=True)
class InputFrame:
    index: int
    image: np.ndarray
    timestamp_s: float
    source: str = "memory"


@dataclass
class PipelineResult:
    """Measured poses and the adjacent-pair evidence used to obtain them.

    In temporal mode, ``*_step_valid`` identifies trustworthy absolute poses.
    A pose recovered after a gap is valid, but its one-frame increment is None.
    ``matches``, ``estimates`` and ``qualities`` retain the raw adjacent-pair
    evidence; ``temporal`` records correction/recovery decisions separately.
    """

    timestamps_s: np.ndarray
    top_increments: list[np.ndarray | None]
    bottom_increments: list[np.ndarray | None]
    top_absolute: np.ndarray
    bottom_absolute: np.ndarray
    top_step_valid: np.ndarray
    bottom_step_valid: np.ndarray
    motion: DecomposedMotion | None
    qualities: list[FrameQuality]
    matches: list[FrameMatches]
    estimates: list[FrameRotationEstimate]
    circle: tuple[float, float, float]
    sphere_center: np.ndarray
    sphere_radius: float
    K: np.ndarray
    dist: np.ndarray
    frame_count: int
    overlay_path: Path | None = None
    temporal: dict[str, Any] | None = None
    offline: dict[str, Any] | None = None
    offline_observations: dict | None = None
    initial_rotations: dict | None = None
    initial_valid: dict | None = None
    offline_landmark_sources: dict | None = None
    mechanical: dict | None = None
    unconstrained_rotations: dict | None = None
    unconstrained_valid: dict | None = None
    unconstrained_motion: DecomposedMotion | None = None


def _records(frames: Iterable[Any], fps: float = 30.0) -> Iterator[InputFrame]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    for fallback_index, value in enumerate(frames):
        if isinstance(value, np.ndarray):
            yield InputFrame(fallback_index, value, fallback_index / fps)
            continue
        if hasattr(value, "image"):
            index = int(getattr(value, "index", fallback_index))
            raw_timestamp = getattr(value, "timestamp_s", None)
            timestamp = index / fps if raw_timestamp is None else float(raw_timestamp)
            source = str(getattr(value, "source", "input"))
            yield InputFrame(index, np.asarray(value.image), timestamp, source)
            continue
        if isinstance(value, (tuple, list)) and len(value) in (2, 3):
            image = np.asarray(value[0])
            timestamp = float(value[1])
            source = str(value[2]) if len(value) == 3 else "input"
            yield InputFrame(fallback_index, image, timestamp, source)
            continue
        raise TypeError("frames must contain arrays, FrameRecord-like objects, or tuples")


def _segment(
    frame: np.ndarray,
    circle: tuple[float, float, float],
    config: Mapping[str, Any],
) -> SegmentationMasks:
    mode = str(config.get("mode", "color")).lower()
    yoke_config = config.get("yoke_hsv")
    if isinstance(yoke_config, Mapping) and not bool(yoke_config.get("enabled", True)):
        yoke_config = None
    equator_config = config.get("equator", {}) or {}
    if "angle_deg" in equator_config or "offset_px" in equator_config:
        equator = {
            "y": circle[1] + float(equator_config.get("offset_px", 0.0)),
            "slope": np.tan(np.deg2rad(float(equator_config.get("angle_deg", 0.0)))),
            "band_px": float(
                equator_config.get(
                    "deadband_px", equator_config.get("band_px", 0.0)
                )
            ),
        }
    else:
        equator = equator_config
    morphology = int(config.get("morphology_px", config.get("morph_kernel", 1)))
    if morphology > 1 and morphology % 2 == 0:
        morphology += 1
    return segment_frame(
        frame,
        circle,
        mode=mode,
        top_hsv=config.get("top_hsv"),
        bottom_hsv=config.get("bottom_hsv"),
        yoke_hsv=yoke_config,
        equator=equator,
        equator_band_px=float(equator_config.get("deadband_px", 0.0)),
        ball_margin_px=float(config.get("ball_margin_px", 1.0)),
        morph_kernel=max(1, morphology),
        morph_iterations=int(config.get("morph_iterations", 1)),
        grow_px=float(config.get("grow_px", 0.0)),
        separation_px=float(config.get("separation_px", 0.0)),
        yoke_dilate_px=float(config.get("yoke_dilate_px", 0.0)),
        color_wins_over_yoke=bool(config.get("color_wins_over_yoke", False)),
        paint_support=config.get("paint_support"),
    )


def _tracker_config(values: Mapping[str, Any]) -> KLTConfig:
    translated = dict(values)
    if "pyramid_levels" in translated and "max_level" not in translated:
        translated["max_level"] = translated["pyramid_levels"]
    return KLTConfig.from_mapping(translated)


def _quality(
    frame_index: int,
    matches: FrameMatches,
    estimates: FrameRotationEstimate,
) -> FrameQuality:
    def one(name: str) -> HemisphereQuality:
        tracked = matches[name]
        estimated = estimates[name]
        inlier_residual = estimated.residual_rad[estimated.inlier_mask]
        all_fb = np.asarray(tracked.fb_error_all, dtype=float)
        all_fb = all_fb[np.isfinite(all_fb)]
        return HemisphereQuality(
            matched_count=tracked.count,
            usable_count=estimated.valid_count,
            inlier_count=estimated.inlier_count,
            inlier_ratio=estimated.inlier_ratio,
            mean_residual_deg=estimated.mean_inlier_residual_deg,
            median_residual_deg=(
                float(np.rad2deg(np.median(inlier_residual)))
                if len(inlier_residual)
                else float("nan")
            ),
            median_fb_error_px=(
                float(np.median(all_fb))
                if len(all_fb)
                else float("nan")
            ),
            success=estimated.success,
        )

    return FrameQuality(frame_index=frame_index, top=one("top"), bottom=one("bottom"))


def run_pipeline(
    frames: Iterable[Any],
    *,
    K: np.ndarray,
    dist: np.ndarray | None = None,
    circle: tuple[float, float, float] | None = None,
    R_bc: np.ndarray | None = None,
    segment_config: Mapping[str, Any] | None = None,
    track_config: Mapping[str, Any] | None = None,
    estimate_config: Mapping[str, Any] | None = None,
    temporal_config: Mapping[str, Any] | None = None,
    offline_config: Mapping[str, Any] | None = None,
    mechanical_config: Mapping[str, Any] | None = None,
    top_shell_sign: int = 1,
    fps: float = 30.0,
    overlay_path: str | Path | None = None,
    overlay_codec: str = "mp4v",
    progress: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Process a video/image iterable with one shared implementation.

    Input images are undistorted before circle fitting, segmentation, and KLT.
    Consequently configured circle coordinates must refer to the undistorted
    image, as produced by ``scripts/calibrate_circle.py``.

    ``temporal_config.enabled`` opts into persistent tracks, quality gates,
    and local image anchors. It defaults off for compatibility with isolated
    axis calibration and callers that require raw frame-pair increments.
    """

    matrix = np.asarray(K, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("K must be 3x3")
    coefficients = np.zeros(5) if dist is None else np.asarray(dist, dtype=float).reshape(-1)
    iterator = iter(_records(frames, fps=fps))
    try:
        first_record = next(iterator)
    except StopIteration as exc:
        raise ValueError("input contains no frames") from exc
    first = undistort_image(first_record.image, matrix, coefficients)
    if circle is None:
        u0, v0, radius_px, _ = detect_circle_auto(first)
        circle = (u0, v0, radius_px)
    else:
        circle = tuple(map(float, circle))
        if len(circle) != 3 or circle[2] <= 0:
            raise ValueError("circle must be (u0, v0, positive r_px)")
    sphere_center, sphere_radius = sphere_pose_from_circle(*circle, matrix)

    segment_values = dict(segment_config or {})
    track_values = dict(track_config or {})
    estimate_values = dict(estimate_config or {})
    temporal_options = TemporalConfig.from_mapping(temporal_config)
    offline_options = OfflineConfig.from_mapping(offline_config)
    mechanical_options = MechanicalConfig.from_mapping(mechanical_config)
    if mechanical_options.enabled and (not offline_options.enabled or R_bc is None):
        raise ValueError("mechanical processing requires offline.enabled=true and calibrated R_bc")
    if top_shell_sign not in (-1, 1):
        raise ValueError("top_shell_sign must be +1 or -1")
    if offline_options.enabled and not temporal_options.enabled:
        raise ValueError("offline processing requires temporal.enabled=true for persistent feature identities")
    klt_config = _tracker_config(track_values)
    tracker = (PersistentKLTTracker(klt_config) if temporal_options.enabled
               else KLTTracker(klt_config))
    previous_masks = _segment(first, circle, segment_values)
    if temporal_options.enabled:
        previous_masks = safe_tracking_masks(previous_masks, temporal_options.boundary_margin_px)
    previous = first
    timestamps = [first_record.timestamp_s]
    top_increments: list[np.ndarray | None] = []
    bottom_increments: list[np.ndarray | None] = []
    match_history: list[FrameMatches] = []
    estimate_history: list[FrameRotationEstimate] = []
    qualities: list[FrameQuality] = []
    random = np.random.default_rng(int(estimate_values.get("random_seed", 7)))
    solver_options = {
        "limb_cull_deg": float(track_values.get("limb_cull_deg", 65.0)),
        "ransac_iters": int(estimate_values.get("ransac_iters", 200)),
        "ransac_inlier_deg": float(estimate_values.get("ransac_inlier_deg", 1.0)),
        "min_inliers": int(estimate_values.get("min_inliers", 8)),
    }
    temporal_trackers = {}
    absolute_history = {name: [np.eye(3)] for name in ("top", "bottom")}
    validity_history = {name: [True] for name in ("top", "bottom")}
    collector = (OfflineObservationCollector(
        offline_options, matrix, sphere_center, sphere_radius, circle[2],
        klt_config, solver_options, temporal_options,
        seed=int(estimate_values.get("random_seed", 7)) + 701)
        if offline_options.enabled else None)
    if temporal_options.enabled:
        tracker.initialize(first, previous_masks)
        gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY) if first.ndim == 3 else first
        for offset, name in enumerate(("top", "bottom")):
            temporal_trackers[name] = TemporalRotationTracker(
                matrix, sphere_center, sphere_radius, circle[2], klt_config,
                solver_options, temporal_options,
                seed=int(estimate_values.get("random_seed", 7)) + offset + 1)
            temporal_trackers[name].initialize(
                gray, previous_masks[name], *tracker.points(name), timestamp_s=first_record.timestamp_s)
        if collector is not None:
            collector.push(0, gray, previous_masks,
                           {name: tracker.points(name) for name in temporal_trackers},
                           {name: np.eye(3) for name in temporal_trackers},
                           {name: {**temporal_trackers[name].history[0],
                                   "sharpness": temporal_trackers[name].recent_sharpness[-1]}
                            for name in temporal_trackers})

    writer = None
    resolved_overlay: Path | None = None
    if overlay_path is not None:
        resolved_overlay = Path(overlay_path).expanduser().resolve()
        resolved_overlay.parent.mkdir(parents=True, exist_ok=True)
        codec = str(overlay_codec)
        if len(codec) != 4:
            raise ValueError("overlay codec must contain four characters")
        writer = cv2.VideoWriter(
            str(resolved_overlay),
            cv2.VideoWriter_fourcc(*codec),
            float(fps),
            (first.shape[1], first.shape[0]),
        )
        if not writer.isOpened():
            raise OSError(f"could not open overlay video writer: {resolved_overlay}")
        writer.write(_draw_base_overlay(first, circle, first_record.index))

    try:
        for logical_index, record in enumerate(iterator, start=1):
            current = undistort_image(record.image, matrix, coefficients)
            if current.shape != previous.shape:
                raise ValueError(
                    f"frame {record.index} has shape {current.shape}, expected {previous.shape}"
                )
            current_masks = _segment(current, circle, segment_values)
            if temporal_options.enabled:
                current_masks = safe_tracking_masks(current_masks, temporal_options.boundary_margin_px)
            matches = tracker.track_pair(previous, current, previous_masks, current_masks)
            estimates = solve_frame_increments(
                matches,
                matrix,
                sphere_center,
                sphere_radius,
                rng=random,
                **solver_options,
            )
            statuses = {}
            records = {}
            if temporal_options.enabled:
                gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY) if current.ndim == 3 else current
                for name, increments in (("top", top_increments), ("bottom", bottom_increments)):
                    pose, valid, diagnostic = temporal_trackers[name].update(
                        logical_index, gray, current_masks[name], matches[name], estimates[name],
                        *tracker.points(name), timestamp_s=record.timestamp_s)
                    # A recovered absolute pose spans a gap; it is not a measured
                    # one-frame increment and must not enter axis calibration.
                    increments.append(pose @ absolute_history[name][-1].T
                                      if valid and validity_history[name][-1] else None)
                    absolute_history[name].append(pose)
                    validity_history[name].append(valid)
                    statuses[name] = diagnostic["status"]
                    records[name] = diagnostic
                    if collector is not None:
                        collector.add_adjacent(name, logical_index, matches[name], estimates[name], diagnostic)
                        collector.add_anchors(name, logical_index, temporal_trackers[name])
                    if diagnostic["adjacent"]["accepted"]:
                        tracker.prune(name, matches[name].track_ids[estimates[name].inlier_mask])
                if collector is not None:
                    collector.push(logical_index, gray, current_masks,
                                   {name: tracker.points(name) for name in temporal_trackers},
                                   {name: temporal_trackers[name].prediction for name in temporal_trackers},
                                   records)
            else:
                top_increments.append(estimates.top.R)
                bottom_increments.append(estimates.bottom.R)
            match_history.append(matches)
            estimate_history.append(estimates)
            quality = _quality(logical_index, matches, estimates)
            qualities.append(quality)
            timestamps.append(record.timestamp_s)
            if writer is not None:
                writer.write(
                    _draw_pair_overlay(current, circle, logical_index, matches, estimates, statuses)
                )
            previous = current
            previous_masks = current_masks
            if progress is not None and logical_index % 60 == 0:
                progress(f"Tracked {logical_index + 1} frames.")
    finally:
        if writer is not None:
            writer.release()

    time_array = np.asarray(timestamps, dtype=float)
    if not np.all(np.isfinite(time_array)) or (len(time_array) > 1 and np.any(np.diff(time_array) <= 0)):
        if offline_options.enabled:
            raise ValueError("offline processing requires finite, strictly increasing timestamps")
        warnings.warn(
            "input timestamps were not strictly increasing; replacing them with fps spacing",
            RuntimeWarning,
        )
        time_array = np.arange(len(time_array), dtype=float) / fps
    if temporal_options.enabled:
        top_absolute, bottom_absolute = (np.stack(absolute_history[name]) for name in ("top", "bottom"))
        top_valid, bottom_valid = (np.asarray(validity_history[name], dtype=bool) for name in ("top", "bottom"))
    else:
        top_absolute, top_valid = accumulate_increments(top_increments)
        bottom_absolute, bottom_valid = accumulate_increments(bottom_increments)
    offline_report = observations = initial_rotations = initial_valid = None
    if collector is not None:
        observations = collector.lists()
        initial_rotations = {"top": top_absolute.copy(), "bottom": bottom_absolute.copy()}
        initial_valid = {"top": top_valid.copy(), "bottom": bottom_valid.copy()}
        fitted = {}
        for name in ("top", "bottom"):
            if progress is not None:
                progress(f"Refining {name} trajectory from {len(observations[name])} pixel observations...")
            fitted[name] = refine_trajectory(observations[name], initial_rotations[name],
                                             initial_valid[name], matrix, sphere_center,
                                             sphere_radius, offline_options)
        top_absolute, bottom_absolute = (fitted[name].rotations for name in ("top", "bottom"))
        top_valid, bottom_valid = (fitted[name].valid for name in ("top", "bottom"))
        top_increments, bottom_increments = (
            [value.rotations[i] @ value.rotations[i - 1].T
             if value.valid[i] and value.valid[i - 1] else None
             for i in range(1, len(time_array))]
            for value in (fitted["top"], fitted["bottom"]))
        offline_report = {
            "config": asdict(offline_options),
            "summary": {name: {
                **{key: value.diagnostics[key] for key in (
                    "refined_frames", "recovered_frames", "optimized_components", "rejected_components")},
                "unresolved_frames": np.flatnonzero(~value.valid).tolist(),
            } for name, value in fitted.items()},
            "shells": {name: value.diagnostics for name, value in fitted.items()},
            "reverse_observations": collector.reverse_diagnostics,
            "observation_identity": "Direct landmarks are scoped to their exact source image; forward LK identities are separate.",
            "note": "Final pose validity is stored in frames.valid_top/valid_bottom; temporal and tracking-overlay diagnostics describe the forward pass.",
        }
    mechanical_report = unconstrained_rotations = unconstrained_valid = unconstrained_motion = None
    mechanical_fit = None
    if mechanical_options.enabled:
        unconstrained_rotations = {"top": top_absolute.copy(), "bottom": bottom_absolute.copy()}
        unconstrained_valid = {"top": top_valid.copy(), "bottom": bottom_valid.copy()}
        unconstrained_motion = decompose_hemispheres(
            top_absolute, bottom_absolute, R_bc, valid_top=top_valid, valid_bottom=bottom_valid)
        if progress is not None:
            progress("Fitting shared roll and independent shell swivel to pixel observations...")
        mechanical_fit = refine_mechanical_trajectory(
            observations, unconstrained_rotations, unconstrained_valid,
            matrix, sphere_center, np.asarray(R_bc, dtype=float), radius=sphere_radius,
            config=mechanical_options, offline_config=offline_options,
            top_shell_sign=top_shell_sign)
        top_absolute, bottom_absolute = (mechanical_fit.rotations[name] for name in ("top", "bottom"))
        top_valid, bottom_valid = (mechanical_fit.valid[name] for name in ("top", "bottom"))
        top_increments, bottom_increments = (
            [mechanical_fit.rotations[name][i] @ mechanical_fit.rotations[name][i - 1].T
             if mechanical_fit.valid[name][i] and mechanical_fit.valid[name][i - 1] else None
             for i in range(1, len(time_array))] for name in ("top", "bottom"))
        mechanical_report = mechanical_fit.diagnostics
        offline_report["note"] = (
            "This report describes the independent offline fit before mechanical fitting. "
            "Final pose validity is in frames.valid_top/valid_bottom and mechanical diagnostics; "
            "the tracking overlay describes the forward pass.")
    motion = None
    if R_bc is not None:
        motion = decompose_hemispheres(
            top_absolute,
            bottom_absolute,
            np.asarray(R_bc, dtype=float),
            valid_top=top_valid,
            valid_bottom=bottom_valid,
        )
    if mechanical_fit is not None:
        motion = mechanical_motion(mechanical_fit.alpha, mechanical_fit.beta, mechanical_fit.valid)
    return PipelineResult(
        timestamps_s=time_array,
        top_increments=top_increments,
        bottom_increments=bottom_increments,
        top_absolute=top_absolute,
        bottom_absolute=bottom_absolute,
        top_step_valid=top_valid,
        bottom_step_valid=bottom_valid,
        motion=motion,
        qualities=qualities,
        matches=match_history,
        estimates=estimate_history,
        circle=circle,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        K=matrix,
        dist=coefficients,
        frame_count=len(time_array),
        overlay_path=resolved_overlay,
        temporal=temporal_report(temporal_options, temporal_trackers) if temporal_options.enabled else None,
        offline=offline_report,
        offline_observations=observations,
        initial_rotations=initial_rotations,
        initial_valid=initial_valid,
        offline_landmark_sources=collector.landmark_sources if collector is not None else None,
        mechanical=mechanical_report,
        unconstrained_rotations=unconstrained_rotations,
        unconstrained_valid=unconstrained_valid,
        unconstrained_motion=unconstrained_motion,
    )


def _draw_base_overlay(
    frame: np.ndarray, circle: tuple[float, float, float], frame_index: int
) -> np.ndarray:
    output = frame.copy()
    cv2.circle(
        output,
        (int(round(circle[0])), int(round(circle[1]))),
        int(round(circle[2])),
        (180, 180, 180),
        1,
    )
    cv2.putText(
        output,
        f"frame {frame_index}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _draw_pair_overlay(
    frame: np.ndarray,
    circle: tuple[float, float, float],
    frame_index: int,
    matches: FrameMatches,
    estimates: FrameRotationEstimate,
    statuses: Mapping[str, str] | None = None,
) -> np.ndarray:
    output = _draw_base_overlay(frame, circle, frame_index)
    for name, color in (("top", (255, 220, 0)), ("bottom", (180, 0, 255))):
        tracked = matches[name]
        estimated = estimates[name]
        for index, (previous, current) in enumerate(zip(tracked.uv_prev, tracked.uv_curr)):
            good = bool(estimated.inlier_mask[index])
            draw_color = color if good else (0, 0, 255)
            p0 = tuple(np.rint(previous).astype(int))
            p1 = tuple(np.rint(current).astype(int))
            cv2.line(output, p0, p1, draw_color, 1, cv2.LINE_AA)
            cv2.circle(output, p1, 2, draw_color, -1, cv2.LINE_AA)
    text = (
        f"top {estimates.top.inlier_count}/{estimates.top.valid_count}  "
        f"bottom {estimates.bottom.inlier_count}/{estimates.bottom.valid_count}"
    )
    cv2.putText(
        output,
        text,
        (10, output.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if statuses:
        cv2.putText(output, "  ".join(f"{name}: {state}" for name, state in statuses.items()),
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
    return output
