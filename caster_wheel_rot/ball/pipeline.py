"""Top-to-bottom speckle rotation pipeline used by synthetic and real runs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import cv2
import numpy as np

from .calibration import detect_circle_auto
from common.camera import undistort_image
from .diagnostics import FrameQuality, HemisphereQuality
from .estimate import FrameRotationEstimate, solve_frame_increments
from .integrate import DecomposedMotion, accumulate_increments, decompose_hemispheres
from .segment import SegmentationMasks, segment_frame
from .sphere import sphere_pose_from_circle
from common.track import FrameMatches, KLTConfig, KLTTracker


@dataclass(frozen=True)
class InputFrame:
    index: int
    image: np.ndarray
    timestamp_s: float
    source: str = "memory"


@dataclass
class PipelineResult:
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
    fps: float = 30.0,
    overlay_path: str | Path | None = None,
    overlay_codec: str = "mp4v",
) -> PipelineResult:
    """Process a video/image iterable with one shared implementation.

    Input images are undistorted before circle fitting, segmentation, and KLT.
    Consequently configured circle coordinates must refer to the undistorted
    image, as produced by ``scripts/calibrate_circle.py``.
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
    tracker = KLTTracker(_tracker_config(track_values))
    previous_masks = _segment(first, circle, segment_values)
    previous = first
    timestamps = [first_record.timestamp_s]
    top_increments: list[np.ndarray | None] = []
    bottom_increments: list[np.ndarray | None] = []
    match_history: list[FrameMatches] = []
    estimate_history: list[FrameRotationEstimate] = []
    qualities: list[FrameQuality] = []
    random = np.random.default_rng(int(estimate_values.get("random_seed", 7)))

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
            matches = tracker.track_pair(previous, current, previous_masks, current_masks)
            estimates = solve_frame_increments(
                matches,
                matrix,
                sphere_center,
                sphere_radius,
                limb_cull_deg=float(track_values.get("limb_cull_deg", 65.0)),
                ransac_iters=int(estimate_values.get("ransac_iters", 200)),
                ransac_inlier_deg=float(
                    estimate_values.get("ransac_inlier_deg", 1.0)
                ),
                min_inliers=int(estimate_values.get("min_inliers", 8)),
                rng=random,
            )
            top_increments.append(estimates.top.R)
            bottom_increments.append(estimates.bottom.R)
            match_history.append(matches)
            estimate_history.append(estimates)
            quality = _quality(logical_index, matches, estimates)
            qualities.append(quality)
            timestamps.append(record.timestamp_s)
            if writer is not None:
                writer.write(
                    _draw_pair_overlay(current, circle, logical_index, matches, estimates)
                )
            previous = current
            previous_masks = current_masks
    finally:
        if writer is not None:
            writer.release()

    time_array = np.asarray(timestamps, dtype=float)
    if len(time_array) > 1 and np.any(np.diff(time_array) <= 0):
        warnings.warn(
            "input timestamps were not strictly increasing; replacing them with fps spacing",
            RuntimeWarning,
        )
        time_array = np.arange(len(time_array), dtype=float) / fps
    top_absolute, top_valid = accumulate_increments(top_increments)
    bottom_absolute, bottom_valid = accumulate_increments(bottom_increments)
    motion = None
    if R_bc is not None:
        motion = decompose_hemispheres(
            top_absolute,
            bottom_absolute,
            np.asarray(R_bc, dtype=float),
            valid_top=top_valid,
            valid_bottom=bottom_valid,
        )
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
    return output
