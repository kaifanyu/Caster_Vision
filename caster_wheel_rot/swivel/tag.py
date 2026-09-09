"""ArUco fork-tag detection, pose solving, and swivel-angle time series.

The module avoids ``estimatePoseSingleMarkers`` because that API is absent in
some current Python wheels and deprecated in others.  Marker corners are fed
directly to the core ``solvePnPGeneric``/``solvePnP`` APIs using the documented
``SOLVEPNP_IPPE_SQUARE`` ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from common.camera import validate_camera_matrix
from common.rotation import geodesic_angle, is_rotation_matrix


FloatArray = NDArray[np.float64]


def _aruco_module() -> Any:
    module = getattr(cv2, "aruco", None)
    if module is None:
        raise RuntimeError(
            "this OpenCV build has no cv2.aruco module; install one OpenCV "
            "wheel with ArUco support (normally opencv-contrib-python)"
        )
    return module


def aruco_dictionary(dictionary: str | int | Any = "DICT_4X4_50") -> Any:
    """Resolve a predefined dictionary name/id or return a dictionary object."""

    module = _aruco_module()
    if isinstance(dictionary, str):
        if not hasattr(module, dictionary):
            raise ValueError(f"unknown ArUco dictionary name: {dictionary}")
        dictionary = int(getattr(module, dictionary))
    if isinstance(dictionary, (int, np.integer)):
        return module.getPredefinedDictionary(int(dictionary))
    if hasattr(dictionary, "bytesList"):
        return dictionary
    raise TypeError("dictionary must be a DICT_* name, integer id, or OpenCV dictionary")


def generate_marker_image(
    dictionary: str | int | Any,
    marker_id: int,
    side_pixels: int,
    *,
    border_bits: int = 1,
) -> NDArray[np.uint8]:
    """Generate a canonical marker across modern and legacy ArUco bindings."""

    module = _aruco_module()
    resolved = aruco_dictionary(dictionary)
    side = int(side_pixels)
    if side < 16:
        raise ValueError("side_pixels must be at least 16")
    if int(marker_id) < 0:
        raise ValueError("marker_id must be non-negative")
    if int(border_bits) < 1:
        raise ValueError("border_bits must be at least one")
    if hasattr(module, "generateImageMarker"):
        image = module.generateImageMarker(
            resolved, int(marker_id), side, borderBits=int(border_bits)
        )
    elif hasattr(resolved, "generateImageMarker"):
        image = resolved.generateImageMarker(
            int(marker_id), side, borderBits=int(border_bits)
        )
    elif hasattr(module, "drawMarker"):
        image = module.drawMarker(
            resolved, int(marker_id), side, borderBits=int(border_bits)
        )
    else:  # pragma: no cover - only very unusual/custom OpenCV builds.
        raise RuntimeError("OpenCV's ArUco bindings cannot generate marker images")
    result = np.asarray(image, dtype=np.uint8)
    if result.shape != (side, side):
        raise RuntimeError(f"OpenCV returned marker image shape {result.shape}, expected {(side, side)}")
    return result


def detector_parameters(*, refine_corners: bool = True) -> Any:
    """Construct detector parameters on both object and legacy APIs."""

    module = _aruco_module()
    if hasattr(module, "DetectorParameters"):
        parameters = module.DetectorParameters()
    elif hasattr(module, "DetectorParameters_create"):
        parameters = module.DetectorParameters_create()
    else:  # pragma: no cover - unsupported ancient/custom binding.
        raise RuntimeError("OpenCV's ArUco detector-parameter API is unavailable")
    if refine_corners and hasattr(parameters, "cornerRefinementMethod"):
        method = getattr(module, "CORNER_REFINE_SUBPIX", 1)
        parameters.cornerRefinementMethod = int(method)
    return parameters


class _DetectorBackend:
    def __init__(self, dictionary: Any, parameters: Any) -> None:
        module = _aruco_module()
        self.dictionary = dictionary
        self.parameters = parameters
        self.detector = (
            module.ArucoDetector(dictionary, parameters)
            if hasattr(module, "ArucoDetector")
            else None
        )

    def detect(self, image: NDArray[np.generic]) -> tuple[list[Any], Any, list[Any]]:
        module = _aruco_module()
        if self.detector is not None:
            return self.detector.detectMarkers(image)
        if not hasattr(module, "detectMarkers"):  # pragma: no cover
            raise RuntimeError("OpenCV's ArUco detection API is unavailable")
        return module.detectMarkers(
            image,
            self.dictionary,
            parameters=self.parameters,
        )


def marker_object_points(marker_length: float) -> FloatArray:
    """Return IPPE-square points matching detected TL/TR/BR/BL corners."""

    length = float(marker_length)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("marker_length must be positive and finite")
    half = 0.5 * length
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def _corners4(value: ArrayLike) -> FloatArray:
    corners = np.asarray(value, dtype=np.float64).reshape(-1, 2)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        raise ValueError("corners_uv must contain four finite 2-D corners")
    return corners


def _distortion(value: ArrayLike | None) -> FloatArray | None:
    if value is None:
        return None
    coefficients = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(coefficients) not in (4, 5, 8, 12, 14) or not np.all(np.isfinite(coefficients)):
        raise ValueError("dist must contain 4, 5, 8, 12, or 14 finite values")
    return coefficients


@dataclass(frozen=True)
class TagPose:
    """One target marker pose and its pixel-domain fit diagnostic."""

    corners_uv: FloatArray
    rvec: FloatArray
    tvec: FloatArray
    R_tag_to_camera: FloatArray
    R_tag_to_car: FloatArray
    reprojection_error_px: float
    solution_count: int


def solve_tag_pose(
    corners_uv: ArrayLike,
    marker_length: float,
    K: ArrayLike,
    *,
    dist: ArrayLike | None = None,
    R_car_from_camera: ArrayLike | None = None,
    previous_R_tag_to_camera: ArrayLike | None = None,
    expected_normal_camera: ArrayLike | None = None,
    continuity_weight_px_per_rad: float = 0.1,
) -> TagPose:
    """Solve a square marker pose with IPPE and choose its physical branch.

    Candidate points must all lie in front of the camera.  When supplied,
    ``expected_normal_camera`` rejects the planar mirror whose tag +z points
    the wrong way.  Reprojection error remains the primary selection signal;
    temporal rotation continuity only breaks near-ties.
    """

    corners = _corners4(corners_uv)
    matrix = validate_camera_matrix(K)
    coefficients = _distortion(dist)
    object_points = marker_object_points(marker_length)
    camera_to_car = (
        np.eye(3, dtype=float)
        if R_car_from_camera is None
        else np.asarray(R_car_from_camera, dtype=float)
    )
    if not is_rotation_matrix(camera_to_car):
        raise ValueError("R_car_from_camera must be a proper 3x3 rotation")
    previous = None
    if previous_R_tag_to_camera is not None:
        previous = np.asarray(previous_R_tag_to_camera, dtype=float)
        if not is_rotation_matrix(previous):
            raise ValueError("previous_R_tag_to_camera must be a proper rotation")
    expected = None
    if expected_normal_camera is not None:
        expected = np.asarray(expected_normal_camera, dtype=float)
        if expected.shape != (3,) or not np.all(np.isfinite(expected)):
            raise ValueError("expected_normal_camera must be a finite (3,) vector")
        norm = float(np.linalg.norm(expected))
        if norm <= np.finfo(float).eps:
            raise ValueError("expected_normal_camera must be non-zero")
        expected = expected / norm
    continuity_weight = float(continuity_weight_px_per_rad)
    if not np.isfinite(continuity_weight) or continuity_weight < 0.0:
        raise ValueError("continuity_weight_px_per_rad must be non-negative")

    rvecs: list[FloatArray] = []
    tvecs: list[FloatArray] = []
    try:
        generic = cv2.solvePnPGeneric(
            object_points,
            corners,
            matrix,
            coefficients,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if bool(generic[0]):
            rvecs = [np.asarray(item, dtype=float).reshape(3) for item in generic[1]]
            tvecs = [np.asarray(item, dtype=float).reshape(3) for item in generic[2]]
    except (AttributeError, cv2.error):
        # Some older Python bindings expose IPPE only through solvePnP.
        rvecs = []
        tvecs = []
    if not rvecs:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            corners,
            matrix,
            coefficients,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            raise RuntimeError("solvePnP failed for the detected marker")
        rvecs = [np.asarray(rvec, dtype=float).reshape(3)]
        tvecs = [np.asarray(tvec, dtype=float).reshape(3)]

    candidates: list[tuple[tuple[float, float, float], FloatArray, FloatArray, FloatArray, float]] = []
    for rvec, tvec in zip(rvecs, tvecs):
        rotation, _ = cv2.Rodrigues(rvec)
        camera_points = object_points @ rotation.T + tvec
        if not np.all(camera_points[:, 2] > 0.0):
            continue
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            matrix,
            coefficients,
        )
        residual = projected.reshape(4, 2) - corners
        error = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        facing_penalty = 0.0
        if expected is not None and float(rotation[:, 2] @ expected) <= 0.0:
            facing_penalty = 1.0
        continuity = (
            float(geodesic_angle(rotation, previous)) if previous is not None else 0.0
        )
        # Facing is a hard first key.  Error plus a small temporal term then
        # resolves the two planar IPPE solutions without overpowering pixels.
        key = (facing_penalty, error + continuity_weight * continuity, error)
        candidates.append((key, rvec, tvec, rotation, error))
    if not candidates:
        raise RuntimeError("IPPE returned no positive-depth marker pose")
    _, rvec, tvec, rotation, error = min(candidates, key=lambda item: item[0])
    return TagPose(
        corners_uv=corners,
        rvec=rvec,
        tvec=tvec,
        R_tag_to_camera=rotation,
        R_tag_to_car=camera_to_car @ rotation,
        reprojection_error_px=error,
        solution_count=len(candidates),
    )


def relative_swivel_yaw(
    R_tag_to_car: ArrayLike,
    R_tag_to_car_zero: ArrayLike,
) -> float:
    """Extract signed car-vertical yaw relative to the calibrated tag zero."""

    current = np.asarray(R_tag_to_car, dtype=float)
    zero = np.asarray(R_tag_to_car_zero, dtype=float)
    if not is_rotation_matrix(current) or not is_rotation_matrix(zero):
        raise ValueError("tag orientations must be proper 3x3 rotations")
    relative = current @ zero.T
    return float(np.arctan2(relative[1, 0], relative[0, 0]))


def unwrap_angles_with_gaps(
    angles: ArrayLike,
    *,
    max_gap_samples: int | None = None,
) -> FloatArray:
    """Unwrap finite samples while preserving missing values as ``NaN``.

    Gaps up to ``max_gap_samples`` are bridged under the usual less-than-pi
    motion assumption.  Longer gaps start a new absolute branch because the
    full-turn count is then unknowable.  ``None`` bridges every gap.
    """

    values = np.asarray(angles, dtype=float).reshape(-1)
    if max_gap_samples is not None and int(max_gap_samples) < 0:
        raise ValueError("max_gap_samples must be non-negative or None")
    output = np.full_like(values, np.nan)
    last_index: int | None = None
    last_wrapped = 0.0
    last_unwrapped = 0.0
    for index, wrapped in enumerate(values):
        if not np.isfinite(wrapped):
            continue
        if last_index is None:
            unwrapped = float(wrapped)
        else:
            gap = index - last_index - 1
            if max_gap_samples is not None and gap > int(max_gap_samples):
                unwrapped = float(wrapped)
            else:
                delta = (float(wrapped) - last_wrapped + np.pi) % (2.0 * np.pi) - np.pi
                unwrapped = last_unwrapped + delta
        output[index] = unwrapped
        last_index = index
        last_wrapped = float(wrapped)
        last_unwrapped = unwrapped
    return output


def differentiate_angles_with_gaps(angles: ArrayLike, timestamps_s: ArrayLike) -> FloatArray:
    """Differentiate each contiguous finite run without crossing dropouts."""

    values = np.asarray(angles, dtype=float).reshape(-1)
    time = np.asarray(timestamps_s, dtype=float).reshape(-1)
    if values.shape != time.shape:
        raise ValueError("angles and timestamps_s must have the same shape")
    if not np.all(np.isfinite(time)) or (len(time) > 1 and np.any(np.diff(time) <= 0.0)):
        raise ValueError("timestamps_s must be finite and strictly increasing")
    derivative = np.full_like(values, np.nan)
    finite = np.isfinite(values)
    index = 0
    while index < len(values):
        if not finite[index]:
            index += 1
            continue
        end = index + 1
        while end < len(values) and finite[end]:
            end += 1
        length = end - index
        if length >= 2:
            edge_order = 2 if length >= 3 else 1
            derivative[index:end] = np.gradient(
                values[index:end], time[index:end], edge_order=edge_order
            )
        index = end
    return derivative


@dataclass(frozen=True)
class TagObservation:
    """Pipeline-friendly tag result; invalid frames carry diagnostics, not exceptions."""

    valid: bool
    marker_id: int
    psi: float = float("nan")
    psi_wrapped: float = float("nan")
    reprojection_error_px: float = float("nan")
    corners_uv: FloatArray | None = None
    rvec: FloatArray | None = None
    tvec: FloatArray | None = None
    R_tag_to_camera: FloatArray | None = None
    R_tag_to_car: FloatArray | None = None
    failure_reason: str | None = None
    detected_corners: dict[int, FloatArray] = field(default_factory=dict)
    contributing_ids: tuple[int, ...] = ()
    continuous: bool = True
    source: str = "detected"


class ArucoTagTracker:
    """Detect one fork tag and maintain a gap-aware unwrapped ``psi``."""

    def __init__(
        self,
        K: ArrayLike,
        marker_length: float,
        *,
        marker_id: int = 0,
        dictionary: str | int | Any = "DICT_4X4_50",
        dist: ArrayLike | None = None,
        R_car_from_camera: ArrayLike | None = None,
        R_tag_to_car_zero: ArrayLike | None = None,
        expected_normal_camera: ArrayLike | None = None,
        max_reprojection_error_px: float = 3.0,
        max_unwrap_gap_frames: int | None = 5,
        refine_corners: bool = True,
    ) -> None:
        self.K = validate_camera_matrix(K)
        self.dist = _distortion(dist)
        self.marker_length = float(marker_length)
        marker_object_points(self.marker_length)  # validation
        self.marker_id = int(marker_id)
        if self.marker_id < 0:
            raise ValueError("marker_id must be non-negative")
        self.R_car_from_camera = (
            np.eye(3, dtype=float)
            if R_car_from_camera is None
            else np.asarray(R_car_from_camera, dtype=float)
        )
        if not is_rotation_matrix(self.R_car_from_camera):
            raise ValueError("R_car_from_camera must be a proper rotation")
        self.R_tag_to_car_zero = (
            np.eye(3, dtype=float)
            if R_tag_to_car_zero is None
            else np.asarray(R_tag_to_car_zero, dtype=float)
        )
        if not is_rotation_matrix(self.R_tag_to_car_zero):
            raise ValueError("R_tag_to_car_zero must be a proper rotation")
        self.expected_normal_camera = expected_normal_camera
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        if not np.isfinite(self.max_reprojection_error_px) or self.max_reprojection_error_px <= 0.0:
            raise ValueError("max_reprojection_error_px must be positive and finite")
        if max_unwrap_gap_frames is not None and int(max_unwrap_gap_frames) < 0:
            raise ValueError("max_unwrap_gap_frames must be non-negative or None")
        self.max_unwrap_gap_frames = max_unwrap_gap_frames
        self._backend = _DetectorBackend(
            aruco_dictionary(dictionary),
            detector_parameters(refine_corners=refine_corners),
        )
        self.reset()

    def reset(self) -> None:
        self._previous_rotation: FloatArray | None = None
        self._last_wrapped: float | None = None
        self._last_unwrapped: float | None = None
        self._gap_frames = 0

    def _target_corners(self, frame: NDArray[np.generic]) -> FloatArray | None:
        image = np.asarray(frame)
        if image.dtype != np.uint8 or image.ndim not in (2, 3):
            raise ValueError("frame must be a uint8 grayscale or color image")
        corners, ids, _ = self._backend.detect(image)
        if ids is None or not len(ids):
            return None
        flat_ids = np.asarray(ids, dtype=int).reshape(-1)
        matches = np.flatnonzero(flat_ids == self.marker_id)
        if not len(matches):
            return None
        candidates = [_corners4(corners[index]) for index in matches]
        # Duplicate ids should not normally exist; prefer the largest image
        # quadrilateral because it has the best-conditioned pose.
        def area(item: FloatArray) -> float:
            return abs(float(cv2.contourArea(item.astype(np.float32))))

        return max(candidates, key=area)

    def _invalid(self, reason: str) -> TagObservation:
        self._gap_frames += 1
        return TagObservation(False, self.marker_id, failure_reason=reason)

    def track(self, frame: NDArray[np.generic]) -> TagObservation:
        """Return one absolute, unwrapped swivel observation for ``frame``."""

        corners = self._target_corners(frame)
        return self.track_corners(corners)

    def track_corners(self, corners: ArrayLike | None) -> TagObservation:
        """Pose/unwrap a previously decoded marker (shared multi-tag detection)."""
        if corners is None:
            return self._invalid("target marker not detected")
        corners = _corners4(corners)
        try:
            pose = solve_tag_pose(
                corners,
                self.marker_length,
                self.K,
                dist=self.dist,
                R_car_from_camera=self.R_car_from_camera,
                previous_R_tag_to_camera=self._previous_rotation,
                expected_normal_camera=self.expected_normal_camera,
            )
        except (RuntimeError, ValueError, cv2.error) as exc:
            return self._invalid(f"pose solve failed: {exc}")
        wrapped = relative_swivel_yaw(
            pose.R_tag_to_car,
            self.R_tag_to_car_zero,
        )
        if pose.reprojection_error_px > self.max_reprojection_error_px:
            return self._invalid(
                f"reprojection error {pose.reprojection_error_px:.3f}px exceeds "
                f"{self.max_reprojection_error_px:.3f}px"
            )
        bridge = (
            self._last_wrapped is not None
            and (
                self.max_unwrap_gap_frames is None
                or self._gap_frames <= int(self.max_unwrap_gap_frames)
            )
        )
        if bridge:
            assert self._last_unwrapped is not None
            delta = (wrapped - self._last_wrapped + np.pi) % (2.0 * np.pi) - np.pi
            unwrapped = self._last_unwrapped + delta
        else:
            unwrapped = wrapped
        continuous = self._last_wrapped is None or bool(bridge)
        self._previous_rotation = pose.R_tag_to_camera
        self._last_wrapped = wrapped
        self._last_unwrapped = unwrapped
        self._gap_frames = 0
        return TagObservation(
            valid=True,
            marker_id=self.marker_id,
            psi=float(unwrapped),
            psi_wrapped=float(wrapped),
            reprojection_error_px=pose.reprojection_error_px,
            corners_uv=pose.corners_uv,
            rvec=pose.rvec,
            tvec=pose.tvec,
            R_tag_to_camera=pose.R_tag_to_camera,
            R_tag_to_car=pose.R_tag_to_car,
            detected_corners={self.marker_id: corners},
            contributing_ids=(self.marker_id,),
            continuous=continuous,
        )


class MultiArucoTagTracker:
    """Fuse only explicitly calibrated marker IDs in one fork yaw convention.

    Each zero rotation maps that marker into the car at fork psi=0. Markers
    may have different tilts. Detection runs once; missing frames are never
    replaced with an unlabelled prediction. Conflicting poses are rejected.
    """

    def __init__(self, K, marker_length, *, zero_rotations, dictionary="DICT_4X4_50",
                 R_car_from_camera=None, max_reprojection_error_px=2.0,
                 max_disagreement_deg=8.0, max_step_deg=35.0,
                 max_unwrap_gap_frames=5):
        if not zero_rotations:
            raise ValueError("multi-tag tracking requires calibrated zero_rotations")
        self.trackers = {}
        rotation = np.eye(3) if R_car_from_camera is None else np.asarray(R_car_from_camera)
        for key, zero in zero_rotations.items():
            zero = np.asarray(zero, dtype=float)
            self.trackers[int(key)] = ArucoTagTracker(
                K, marker_length, marker_id=int(key), dictionary=dictionary,
                R_car_from_camera=rotation, R_tag_to_car_zero=zero,
                expected_normal_camera=rotation.T @ zero[:, 2],
                max_reprojection_error_px=max_reprojection_error_px,
                max_unwrap_gap_frames=max_unwrap_gap_frames)
        self.backend = _DetectorBackend(aruco_dictionary(dictionary), detector_parameters())
        self.max_disagreement = np.deg2rad(float(max_disagreement_deg))
        self.max_step = np.deg2rad(float(max_step_deg))
        if not 0 < self.max_disagreement < np.pi or not 0 < self.max_step <= np.pi:
            raise ValueError("multi-tag angle limits must lie in (0,180] degrees")
        self.max_gap = int(max_unwrap_gap_frames)
        if self.max_gap < 0:
            raise ValueError("max_unwrap_gap_frames must be nonnegative")
        self.last_wrapped = self.last_unwrapped = None
        self.gap = 0

    def track(self, frame):
        corners, ids, _ = self.backend.detect(frame)
        found = {}
        for points, key in zip(corners, [] if ids is None else np.asarray(ids).ravel()):
            key = int(key)
            if key not in self.trackers:
                continue
            points = _corners4(points)
            if key not in found or abs(cv2.contourArea(points.astype(np.float32))) > abs(cv2.contourArea(found[key].astype(np.float32))):
                found[key] = points
        observations = [tracker.track_corners(found.get(key)) for key, tracker in self.trackers.items()]
        candidates = [o for o in observations if o.valid]
        wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
        if self.last_wrapped is not None and self.gap <= self.max_gap:
            candidates = [o for o in candidates if abs(wrap(o.psi_wrapped - self.last_wrapped)) <= min(np.pi, self.max_step * (self.gap + 1))]
        reason = "no calibrated marker pose passed detection/pose/motion gates"
        if candidates:
            best = min(candidates, key=lambda o: o.reprojection_error_px)
            if any(abs(wrap(o.psi_wrapped - best.psi_wrapped)) > self.max_disagreement for o in candidates):
                candidates = []
                reason = "calibrated markers disagree on fork yaw"
        if not candidates:
            self.gap += 1
            return TagObservation(False, -1, failure_reason=reason, detected_corners=found)
        weights = np.array([1 / max(o.reprojection_error_px, .25)**2 for o in candidates])
        angles = np.array([o.psi_wrapped for o in candidates])
        wrapped = float(np.arctan2(weights @ np.sin(angles), weights @ np.cos(angles)))
        bridge = self.last_wrapped is not None and self.gap <= self.max_gap
        continuous = self.last_wrapped is None or bridge
        unwrapped = self.last_unwrapped + wrap(wrapped - self.last_wrapped) if bridge else wrapped
        self.last_wrapped, self.last_unwrapped, self.gap = wrapped, unwrapped, 0
        return replace(best, psi=float(unwrapped), psi_wrapped=wrapped,
                       detected_corners=found, contributing_ids=tuple(o.marker_id for o in candidates),
                       continuous=continuous, source="multi_tag" if len(candidates) > 1 else "detected")


# Short name for callers that prefer the detector terminology in the spec.
TagDetector = ArucoTagTracker


__all__ = [
    "ArucoTagTracker",
    "TagDetector",
    "TagObservation",
    "TagPose",
    "aruco_dictionary",
    "detector_parameters",
    "differentiate_angles_with_gaps",
    "generate_marker_image",
    "marker_object_points",
    "relative_swivel_yaw",
    "solve_tag_pose",
    "unwrap_angles_with_gaps",
]
