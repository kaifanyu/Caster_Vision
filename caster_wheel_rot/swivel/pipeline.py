"""End-to-end fork-tag and sidewall-speckle swivel pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import cv2
import numpy as np

from common.camera import undistort_image
from common.caster_frame import CasterFrame
from common.rotation import Rz
from .adapter import SwivelSeries, adapt_swivel_series
from .geometry import SwivelGeometry, project_sidewall
from .roll import RollEstimate, RollEstimator
from .reference import ReferenceObservation, detect_reference_phase, unwrap_reference_phase
from .tag import ArucoTagTracker, TagObservation
from .tag import MultiArucoTagTracker
from .masking import wheel_tracking_mask
from .integration import integrate_roll


@dataclass(frozen=True)
class InputFrame:
    index: int
    image: np.ndarray
    timestamp_s: float
    source: str = "memory"


@dataclass
class SwivelPipelineResult:
    timestamps_s: np.ndarray
    phi: np.ndarray
    psi: np.ndarray
    phi_valid: np.ndarray
    psi_valid: np.ndarray
    tag_observations: list[TagObservation]
    roll_estimates: list[RollEstimate | None]
    interval_quality: list[dict[str, Any]]
    face_signs: np.ndarray
    reference_observations: list[ReferenceObservation]
    reference_phase: np.ndarray
    reference_phase_error: np.ndarray
    series: SwivelSeries
    caster_frames: list[CasterFrame]
    geometry: SwivelGeometry
    K: np.ndarray
    dist: np.ndarray
    frame_count: int
    overlay_path: Path | None = None
    phi_segment_id: np.ndarray | None = None
    reference_corrected: np.ndarray | None = None


def _records(frames: Iterable[Any], fps: float) -> Iterator[InputFrame]:
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    for fallback_index, value in enumerate(frames):
        if isinstance(value, np.ndarray):
            yield InputFrame(fallback_index, value, fallback_index / fps)
        elif hasattr(value, "image"):
            index = int(getattr(value, "index", fallback_index))
            timestamp = getattr(value, "timestamp_s", None)
            yield InputFrame(
                index,
                np.asarray(value.image),
                index / fps if timestamp is None else float(timestamp),
                str(getattr(value, "source", "input")),
            )
        elif isinstance(value, (tuple, list)) and len(value) in {2, 3}:
            yield InputFrame(
                fallback_index,
                np.asarray(value[0]),
                float(value[1]),
                str(value[2]) if len(value) == 3 else "input",
            )
        else:
            raise TypeError("frames must contain arrays, FrameRecord-like values, or tuples")


def _empty_roll(reason: str, face_sign: int, view_prev: float, view_curr: float) -> RollEstimate:
    return RollEstimate(
        delta_phi=float("nan"),
        R_roll_fork=None,
        R_unconstrained_fork=None,
        inlier_mask=np.zeros(0, dtype=bool),
        geometry_valid_mask=np.zeros(0, dtype=bool),
        residual_rad=np.zeros(0),
        off_axis_residual_rad=float("nan"),
        face_sign=int(face_sign),
        view_confidence_prev=float(view_prev),
        view_confidence_curr=float(view_curr),
        method="failed",
        failure_reason=reason,
    )


def run_pipeline(
    frames: Iterable[Any],
    *,
    K: np.ndarray,
    geometry: SwivelGeometry,
    marker_size_m: float,
    marker_id: int = 0,
    marker_dictionary: str = "DICT_4X4_50",
    psi0_rad: float = 0.0,
    dist: np.ndarray | None = None,
    track_config: Mapping[str, Any] | None = None,
    estimate_config: Mapping[str, Any] | None = None,
    reference_dot_config: Mapping[str, Any] | None = None,
    sidewall_inner_fraction: float = 0.20,
    sidewall_margin_px: int = 2,
    edge_on_min_cos: float = 0.08,
    max_tag_reprojection_error_px: float = 3.0,
    max_off_axis_deg: float = 2.0,
    r_eff: float | None = None,
    roll_direction_sign: float = 1.0,
    heading_direction_sign: float = 1.0,
    fps: float = 30.0,
    time_offset_s: float = 0.0,
    overlay_path: str | Path | None = None,
    overlay_codec: str = "mp4v",
    marker_config: Mapping[str, Any] | None = None,
    mask_config: Mapping[str, Any] | None = None,
) -> SwivelPipelineResult:
    """Track one clip and optionally emit calibrated common records.

    Input frames are undistorted once at the front end.  Tag PnP and wheel
    projection therefore both receive zero distortion afterward.
    """

    matrix = np.asarray(K, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("K must be a finite 3x3 matrix")
    coefficients = np.zeros(5) if dist is None else np.asarray(dist, dtype=float).reshape(-1)
    # Retain only undistorted images, not a second full copy of the source clip.
    images, times = [], []
    maps = None
    for record in _records(frames, fps):
        if images and record.image.shape != images[0].shape:
            raise ValueError("all frames must share one image shape")
        if np.any(coefficients):
            if maps is None:
                size = (record.image.shape[1], record.image.shape[0])
                maps = cv2.initUndistortRectifyMap(matrix, coefficients, None, matrix, size, cv2.CV_32FC1)
            images.append(cv2.remap(record.image, *maps, cv2.INTER_LINEAR))
        else:
            images.append(record.image.copy())
        times.append(record.timestamp_s + float(time_offset_s))
    if len(images) < 2:
        raise ValueError("input must contain at least two frames")
    timestamps = np.asarray(times)
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("frame timestamps plus time_offset_s must be finite and increasing")
    shape = images[0].shape
    if any(image.shape != shape for image in images):
        raise ValueError("all frames must share one image shape")

    tag_tracker = ArucoTagTracker(
        matrix,
        marker_size_m,
        marker_id=marker_id,
        dictionary=marker_dictionary,
        dist=np.zeros_like(coefficients),
        R_car_from_camera=geometry.R_car_from_camera,
        R_tag_to_car_zero=Rz(float(psi0_rad)),
        expected_normal_camera=geometry.car_vectors_to_camera(
            np.array([0.0, 0.0, 1.0])
        ),
        max_reprojection_error_px=max_tag_reprojection_error_px,
    )
    marker_values = dict(marker_config or {})
    zeros = marker_values.get("zero_rotations")
    if zeros:
        tag_tracker = MultiArucoTagTracker(
            matrix, marker_size_m, zero_rotations=zeros, dictionary=marker_dictionary,
            R_car_from_camera=geometry.R_car_from_camera,
            max_reprojection_error_px=max_tag_reprojection_error_px,
            max_disagreement_deg=float(marker_values.get("max_disagreement_deg", 8.0)),
            max_step_deg=float(marker_values.get("max_step_deg", 35.0)),
            max_unwrap_gap_frames=int(marker_values.get("max_unwrap_gap_frames", 5)))
    observations = [tag_tracker.track(image) for image in images]
    psi = np.array([item.psi if item.valid else np.nan for item in observations])
    psi_valid = np.array([item.valid for item in observations], dtype=bool)

    reference_values = dict(reference_dot_config or {})
    reference_enabled = bool(reference_values.get("enabled", True))
    reference_observations: list[ReferenceObservation] = []
    for image, angle, valid, observation in zip(images, psi, psi_valid, observations):
        if reference_enabled and valid and geometry.sidewall_view_confidence(angle) >= edge_on_min_cos:
            reference_mask_config = dict(mask_config or {})
            reference_mask_config["white_face"] = {"enabled": False}
            reference_mask, _ = wheel_tracking_mask(image, geometry, matrix, angle,
                inner_fraction=sidewall_inner_fraction, margin_px=sidewall_margin_px,
                config=reference_mask_config, observation=observation)
            reference_observations.append(
                detect_reference_phase(
                    image,
                    float(angle),
                    geometry,
                    matrix,
                    config=reference_values,
                    allowed_mask=reference_mask,
                )
            )
        else:
            reference_observations.append(
                ReferenceObservation(
                    False,
                    failure_reason="disabled" if not reference_enabled else "fork tag unavailable",
                )
            )
    reference_phase = unwrap_reference_phase(
        reference_observations,
        max_gap_frames=int(reference_values.get("max_gap_frames", 5)),
    )

    estimate_values = dict(estimate_config or {})
    roll_estimator = RollEstimator(
        geometry,
        matrix,
        dist=np.zeros_like(coefficients),
        track_config=track_config,
        min_view_confidence=edge_on_min_cos,
        ransac_iters=int(estimate_values.get("ransac_iters", 250)),
        ransac_inlier_deg=float(estimate_values.get("ransac_inlier_deg", 1.5)),
        min_inliers=int(estimate_values.get("min_inliers", 8)),
        random_seed=int(estimate_values.get("random_seed", 7)),
        circular_fallback=bool(estimate_values.get("circular_fallback", True)),
        motion_prediction=bool((track_config or {}).get("motion_prediction", False)),
        max_radial_error_fraction=estimate_values.get("max_radial_error_fraction"),
        max_reprojection_error_px=estimate_values.get("max_reprojection_error_px"),
        min_angular_span_deg=float(estimate_values.get("min_angular_span_deg", 0.0)),
    )
    max_off_axis = np.deg2rad(float(max_off_axis_deg))
    roll_estimates: list[RollEstimate | None] = []
    qualities: list[dict[str, Any]] = []
    face_signs: list[int] = []
    phi = np.zeros(len(images), dtype=float)
    phi_valid = np.ones(len(images), dtype=bool)
    predicted_delta = 0.0
    for index in range(len(images) - 1):
        if psi_valid[index] and psi_valid[index + 1] and observations[index+1].continuous:
            psi_mid = 0.5 * (psi[index] + psi[index + 1])
            face_sign = geometry.visible_face_sign(float(psi_mid))
            mask_prev, projection_prev = wheel_tracking_mask(
                images[index], geometry, matrix, float(psi[index]), face_sign=face_sign,
                inner_fraction=sidewall_inner_fraction, margin_px=sidewall_margin_px,
                config=mask_config, observation=observations[index],
            )
            mask_curr, projection_curr = wheel_tracking_mask(
                images[index+1], geometry, matrix, float(psi[index+1]), face_sign=face_sign,
                inner_fraction=sidewall_inner_fraction, margin_px=sidewall_margin_px,
                config=mask_config, observation=observations[index+1],
            )
            estimate = roll_estimator.estimate(
                images[index], images[index + 1], mask_prev,
                mask_curr, float(psi[index]), float(psi[index + 1]),
                face_sign=face_sign,
                predicted_delta_phi=predicted_delta,
            )
        else:
            probe_psi = float(psi[index]) if psi_valid[index] else (
                float(psi[index + 1]) if psi_valid[index + 1] else 0.0
            )
            face_sign = geometry.visible_face_sign(probe_psi)
            estimate = _empty_roll(
                "fork tag is missing at one or both interval endpoints",
                face_sign,
                geometry.sidewall_view_confidence(probe_psi, face_sign),
                geometry.sidewall_view_confidence(probe_psi, face_sign),
            )
        face_signs.append(face_sign)
        success = bool(
            estimate.success
            and ((np.isfinite(estimate.off_axis_residual_rad)
                  and estimate.off_axis_residual_rad <= max_off_axis)
                 or (estimate.method == "circular" and estimate.image_plane_check is not None
                     and abs(estimate.image_plane_disagreement_rad) <= np.deg2rad(float(estimate_values.get("ransac_inlier_deg", 1.5)))
                     and estimate.mean_inlier_residual_rad <= np.deg2rad(float(estimate_values.get("ransac_inlier_deg", 1.5)))))
        )
        if success and estimate.inlier_ratio < float(estimate_values.get("min_inlier_ratio", 0.0)):
            success = False
            estimate = replace(estimate, failure_reason="roll inlier ratio below configured minimum")
        predicted_delta = float(estimate.delta_phi) if success else 0.0
        if success:
            phi[index + 1] = phi[index] + float(estimate.delta_phi)
        else:
            phi[index + 1] = phi[index]
            phi_valid[index + 1] = False
        heading_ok = bool(psi_valid[index] and psi_valid[index + 1] and observations[index+1].continuous)
        tag_error = max(
            observations[index].reprojection_error_px if observations[index].valid else np.inf,
            observations[index + 1].reprojection_error_px if observations[index + 1].valid else np.inf,
        )
        quality = dict(estimate.quality)
        quality.update(
            {
                "roll_valid": success,
                "delta_phi_rad": float(estimate.delta_phi) if success else None,
                "heading_valid": heading_ok,
                "tag_reprojection_error_px": float(tag_error),
                "failure_reason": estimate.failure_reason if not success else None,
                "confidence": float(
                    np.clip(
                        min(estimate.inlier_ratio, estimate.view_confidence_prev, estimate.view_confidence_curr)
                        if success else 0.0,
                        0.0,
                        1.0,
                    )
                ),
            }
        )
        if estimate.success and not success and not quality["failure_reason"]:
            quality["failure_reason"] = (
                f"off-axis residual {np.rad2deg(estimate.off_axis_residual_rad):.3f}deg "
                f"exceeds {max_off_axis_deg:.3f}deg"
            )
        qualities.append(quality)
        roll_estimates.append(estimate)

    phi, phi_valid, segments, reference_phase_error, corrected = integrate_roll(
        [e.delta_phi for e in roll_estimates], [q["roll_valid"] for q in qualities],
        reference_observations, max_gap_frames=int(reference_values.get("max_gap_frames", 5)),
        max_resync_error_deg=float(reference_values.get("max_resync_error_deg", 45.)),
        max_step_deg=float(reference_values.get("max_step_deg", 30.)))
    for index, quality in enumerate(qualities):
        endpoint_errors = reference_phase_error[index:index+2]
        finite_errors = endpoint_errors[np.isfinite(endpoint_errors)]
        quality["reference_phase_error_deg"] = float(np.rad2deg(np.max(np.abs(finite_errors)))) if len(finite_errors) else None
        quality["reference_dot_valid"] = bool(reference_observations[index].valid or reference_observations[index+1].valid)
        quality["accumulated_phi_valid"] = bool(phi_valid[index+1])
        quality["phi_segment_id"] = int(segments[index+1])
        quality["reference_corrected"] = bool(corrected[index+1])
        quality["tag_ids"] = list(observations[index+1].contributing_ids)

    series = SwivelSeries(timestamps, phi, psi, phi_valid, psi_valid, qualities)
    caster_frames = []
    if r_eff is not None:
        caster_frames = adapt_swivel_series(
            series,
            r_eff=float(r_eff),
            axle0_car=geometry.axle_zero_car,
            roll_direction_sign=roll_direction_sign,
            heading_direction_sign=heading_direction_sign,
            swivel_axis_car=geometry.swivel_axis_car,
            hub_offset0_car=geometry.hub_offset_zero_car,
        )

    resolved_overlay = _write_overlay(
        images, timestamps, observations, qualities, geometry, matrix, psi,
        psi_valid, overlay_path, fps, overlay_codec,
        roll_estimates=roll_estimates, mask_config=mask_config,
        inner_fraction=sidewall_inner_fraction, margin_px=sidewall_margin_px,
    ) if overlay_path is not None else None
    return SwivelPipelineResult(
        timestamps_s=timestamps,
        phi=phi,
        psi=psi,
        phi_valid=phi_valid,
        psi_valid=psi_valid,
        tag_observations=observations,
        roll_estimates=roll_estimates,
        interval_quality=qualities,
        face_signs=np.asarray(face_signs, dtype=int),
        reference_observations=reference_observations,
        reference_phase=reference_phase,
        reference_phase_error=reference_phase_error,
        series=series,
        caster_frames=caster_frames,
        geometry=geometry,
        K=matrix,
        dist=coefficients,
        frame_count=len(images),
        overlay_path=resolved_overlay,
        phi_segment_id=segments, reference_corrected=corrected,
    )


def run_clip(
    clip_path: str | Path,
    config: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> SwivelPipelineResult:
    """High-level raw-clip entry used by the comparison harness."""

    from common.config import camera_matrix, distortion_coefficients
    from common.io_frames import open_frame_source

    input_values = config.get("input", {})
    swivel_values = config.get("swivel", {})
    if not isinstance(input_values, Mapping) or not isinstance(swivel_values, Mapping):
        raise ValueError("config.input and config.swivel must be mappings")
    fps_override = input_values.get("fps_override")
    source = open_frame_source(
        clip_path,
        input_type=str(input_values.get("type", "auto")),
        max_frames=input_values.get("max_frames"),
        image_fps=None if fps_override is None else float(fps_override),
    )
    fps = float(fps_override or source.fps or 0.0)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("raw swivel comparison has no reliable FPS; set input.fps_override")
    camera_values = config.get("camera", {})
    if not isinstance(camera_values, Mapping):
        raise ValueError("config.camera must be a mapping")
    K, approximate = camera_matrix(camera_values, (source.size[1], source.size[0], 3))
    if approximate:
        raise ValueError(
            "raw swivel comparison requires calibrated camera.K, not the FOV fallback"
        )
    dist = distortion_coefficients(camera_values)
    geometry = SwivelGeometry.from_mapping(config)
    marker = swivel_values.get("marker", {})
    geometry_values = swivel_values.get("geometry", {})
    if not isinstance(marker, Mapping) or not isinstance(geometry_values, Mapping):
        raise ValueError("swivel.marker and swivel.geometry must be mappings")
    if not bool(geometry_values.get("calibrated", False)):
        raise ValueError("raw swivel comparison requires calibrated swivel.geometry")
    if not bool(swivel_values.get("direction_sign_calibrated", False)):
        raise ValueError("raw swivel comparison requires calibrated swivel direction sign")
    if marker.get("psi0_rad") is None:
        raise ValueError("raw swivel comparison requires calibrated swivel.marker.psi0_rad")
    radius = swivel_values.get("r_eff_m")
    overlay = None
    if output_dir is not None and bool(config.get("output", {}).get("save_overlay_video", False)):
        overlay = Path(output_dir) / "tracking_overlay.mp4"
    timestamped_source = (
        (record.image, record.index / fps, record.source) for record in source
    )
    return run_pipeline(
        timestamped_source,
        K=K,
        dist=dist,
        geometry=geometry,
        marker_size_m=float(marker.get("size_m", 0.04)),
        marker_id=int(marker.get("id", 0)),
        marker_dictionary=str(marker.get("dictionary", "DICT_4X4_50")),
        psi0_rad=float(marker["psi0_rad"]),
        max_tag_reprojection_error_px=float(marker.get("max_reprojection_error_px", 3.0)),
        track_config=config.get("track", {}),
        estimate_config=config.get("estimate", {}),
        reference_dot_config=swivel_values.get("reference_dot", {}),
        sidewall_inner_fraction=float(geometry_values.get("sidewall_inner_fraction", 0.20)),
        edge_on_min_cos=float(geometry_values.get("edge_on_min_cos", 0.08)),
        max_off_axis_deg=float(config.get("estimate", {}).get("max_off_axis_deg", 2.0)),
        r_eff=None if radius is None else float(radius),
        roll_direction_sign=float(swivel_values.get("roll_direction_sign", 1.0)),
        heading_direction_sign=float(swivel_values.get("heading_direction_sign", 1.0)),
        fps=fps,
        time_offset_s=float(config.get("car_track", {}).get("time_offset_s", 0.0)),
        overlay_path=overlay,
        marker_config=marker, mask_config=swivel_values.get("mask", {}),
    )


def _write_overlay(
    images: list[np.ndarray],
    timestamps: np.ndarray,
    observations: list[TagObservation],
    qualities: list[dict[str, Any]],
    geometry: SwivelGeometry,
    K: np.ndarray,
    psi: np.ndarray,
    psi_valid: np.ndarray,
    path: str | Path,
    fps: float,
    codec: str,
    *, roll_estimates=None, mask_config=None, inner_fraction=.2, margin_px=2,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if len(codec) != 4:
        raise ValueError("overlay_codec must contain four characters")
    height, width = images[0].shape[:2]
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*codec), float(fps), (width, height))
    if not writer.isOpened():
        raise OSError(f"could not open overlay writer: {destination}")
    try:
        for index, image in enumerate(images):
            output = image.copy()
            observation = observations[index]
            if observation.corners_uv is not None:
                corners = np.rint(observation.corners_uv).astype(np.int32)
                cv2.polylines(output, [corners], True, (0, 220, 0), 2, cv2.LINE_AA)
            if psi_valid[index]:
                sign = geometry.visible_face_sign(float(psi[index]))
                mask, projection = wheel_tracking_mask(image, geometry, K, float(psi[index]),
                    face_sign=sign, inner_fraction=inner_fraction, margin_px=margin_px,
                    config=mask_config, observation=observation)
                output[mask] = (output[mask].astype(float)*.8 + np.array([0,180,0])*.2).astype(np.uint8)
                boundary = np.rint(projection.boundary_uv[np.all(np.isfinite(projection.boundary_uv), axis=1)]).astype(np.int32)
                if len(boundary) >= 3:
                    cv2.polylines(output, [boundary], True, (255, 180, 0), 1, cv2.LINE_AA)
            q = qualities[min(index, len(qualities) - 1)] if qualities else {}
            if roll_estimates and index < len(roll_estimates):
                estimate = roll_estimates[index]
                for point, inlier in zip(estimate.uv_prev, estimate.inlier_mask):
                    cv2.circle(output, tuple(np.rint(point).astype(int)), 3,
                               (0,255,0) if inlier else (0,0,255), 1, cv2.LINE_AA)
            for marker_id, corners in observation.detected_corners.items():
                corners = np.rint(corners).astype(np.int32)
                cv2.polylines(output, [corners], True, (0,220,255), 2)
                cv2.putText(output, f"ID {marker_id}", tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX,.6,(0,220,255),2)
            text = f"t={timestamps[index]:.3f}s psi={'%.1f' % np.rad2deg(psi[index]) if psi_valid[index] else 'DROP'} roll={'OK' if q.get('roll_valid') else 'DROP'}"
            cv2.putText(output, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(output, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            detail = f"inliers={q.get('inlier_count',0)} accumulated={'complete' if q.get('accumulated_phi_valid') else 'INCOMPLETE'} {q.get('failure_reason') or ''}"
            cv2.putText(output, detail, (10,48), cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(output, detail, (10,48), cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
            writer.write(output)
    finally:
        writer.release()
    return destination


__all__ = ["InputFrame", "SwivelPipelineResult", "run_clip", "run_pipeline"]
