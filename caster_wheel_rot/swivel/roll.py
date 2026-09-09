"""Wheel-sidewall speckle tracking and de-swiveled roll recovery.

Each pixel pair is lifted with the plane belonging to its own frame.  The hub
translation and fork swivel are then removed before fitting roll about the
known zero-fork axle.  Fitting directly in the camera frame is incorrect when
``psi`` changes, particularly for a caster with non-zero trail.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, field
from typing import Any, Mapping

import numpy as np
import cv2
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from common.camera import undistort_points, validate_camera_matrix
from common.estimate import angular_residuals, ransac_kabsch
from common.track import KLTConfig, detect_features, track_features
from swivel.geometry import SwivelGeometry, unproject_to_plane, project_camera_points


FloatArray = NDArray[np.float64]


def _pixel_pairs(first: ArrayLike, second: ArrayLike) -> tuple[FloatArray, FloatArray]:
    previous = np.asarray(first, dtype=float)
    current = np.asarray(second, dtype=float)
    if previous.ndim != 2 or previous.shape[1:] != (2,):
        raise ValueError("uv_prev must have shape (N, 2)")
    if current.shape != previous.shape:
        raise ValueError("uv_curr must match uv_prev")
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
        raise ValueError("tracked pixels must be finite")
    return previous, current


def _unit_axis(value: ArrayLike) -> FloatArray:
    axis = np.asarray(value, dtype=float)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("axis must be a finite vector with shape (3,)")
    norm = float(np.linalg.norm(axis))
    if norm <= np.finfo(float).eps:
        raise ValueError("axis must be non-zero")
    return axis / norm


def _wrapped(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=float)
    return (array + np.pi) % (2.0 * np.pi) - np.pi


def _axis_rotation(axis: FloatArray, angle: float) -> FloatArray:
    return Rotation.from_rotvec(axis * float(angle)).as_matrix()


def _signed_feature_angles(a: FloatArray, b: FloatArray, axis: FloatArray) -> FloatArray:
    cross = np.cross(a, b)
    sine = cross @ axis
    cosine = np.einsum("ij,ij->i", a, b)
    return np.arctan2(sine, cosine)


def _circular_mean(values: FloatArray) -> float:
    if not len(values):
        return float("nan")
    return float(np.arctan2(np.mean(np.sin(values)), np.mean(np.cos(values))))


@dataclass(frozen=True)
class ImagePlaneRollCheck:
    """Independent roll estimate from ellipse-normalized image coordinates.

    This check deliberately does not use ray--plane intersection, Kabsch, or
    the primary fit's inlier mask.  Each sidewall plane induces a homography
    from fork-frame disk coordinates to the image.  Inverting that homography
    turns the projected ellipse back into a circle, after which each tracked
    feature supplies a signed polar-angle change.
    """

    delta_phi: float
    valid_mask: NDArray[np.bool_]
    inlier_mask: NDArray[np.bool_]
    residual_rad: FloatArray
    homography_condition_prev: float
    homography_condition_curr: float

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def inlier_ratio(self) -> float:
        return self.inlier_count / self.valid_count if self.valid_count else 0.0

    @property
    def mean_inlier_residual_rad(self) -> float:
        if not self.inlier_count:
            return float("nan")
        return float(np.mean(self.residual_rad[self.inlier_mask]))


def _disk_to_image_homography(
    K: FloatArray,
    geometry: SwivelGeometry,
    psi: float,
    face_sign: int,
) -> tuple[FloatArray, float] | None:
    """Map ``[disk_u, disk_v, 1]`` to an undistorted image pixel."""

    center_camera, _ = geometry.face_plane_camera(float(psi), face_sign)
    first_fork, second_fork = geometry.radial_basis_zero()
    rotation = geometry.R_camera_from_fork(float(psi))
    first_camera = rotation @ first_fork
    second_camera = rotation @ second_fork
    homography = K @ np.column_stack(
        [first_camera, second_camera, center_camera]
    )
    condition = float(np.linalg.cond(homography))
    if not np.isfinite(condition) or abs(float(np.linalg.det(homography))) <= 1e-12:
        return None
    return homography, condition


def _image_to_disk(
    uv: FloatArray,
    homography: FloatArray,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    homogeneous = np.column_stack([uv, np.ones(len(uv), dtype=float)])
    disk_homogeneous = homogeneous @ np.linalg.inv(homography).T
    scale = disk_homogeneous[:, 2]
    valid = np.isfinite(disk_homogeneous).all(axis=1) & (np.abs(scale) > 1e-12)
    disk = np.full((len(uv), 2), np.nan, dtype=float)
    disk[valid] = disk_homogeneous[valid, :2] / scale[valid, None]
    return disk, valid


def image_plane_circular_check(
    uv_prev: ArrayLike,
    uv_curr: ArrayLike,
    K: ArrayLike,
    geometry: SwivelGeometry,
    psi_prev: float,
    psi_curr: float,
    *,
    face_sign: int | None = None,
    dist: ArrayLike | None = None,
    inlier_rad: float = np.deg2rad(1.0),
    min_inliers: int = 8,
    max_homography_condition: float = 1e12,
) -> ImagePlaneRollCheck | None:
    """Estimate roll by a robust circular mean in normalized image space.

    This is a diagnostic, not a replacement for :func:`fit_axis_rotation`.
    Perspective and elliptical foreshortening are removed by the exact planar
    homography for each frame.  Consensus is deterministic, so adding the
    check cannot perturb the primary RANSAC random stream.
    """

    previous, current = _pixel_pairs(uv_prev, uv_curr)
    matrix = validate_camera_matrix(K)
    threshold = float(inlier_rad)
    required = int(min_inliers)
    maximum_condition = float(max_homography_condition)
    if not np.isfinite(threshold) or not 0.0 < threshold <= np.pi:
        raise ValueError("inlier_rad must lie in (0, pi]")
    if required < 3:
        raise ValueError("min_inliers must be at least three")
    if not np.isfinite(maximum_condition) or maximum_condition <= 1.0:
        raise ValueError("max_homography_condition must be finite and greater than one")
    if len(previous) < required:
        return None

    if face_sign is None:
        sign_previous = geometry.visible_face_sign(float(psi_prev))
        sign_current = geometry.visible_face_sign(float(psi_curr))
        if sign_previous != sign_current:
            return None
        sign = sign_previous
    else:
        sign = int(face_sign)
        if sign not in {-1, 1}:
            raise ValueError("face_sign must be +1, -1, or None")

    previous_mapping = _disk_to_image_homography(
        matrix, geometry, float(psi_prev), sign
    )
    current_mapping = _disk_to_image_homography(
        matrix, geometry, float(psi_curr), sign
    )
    if previous_mapping is None or current_mapping is None:
        return None
    previous_homography, previous_condition = previous_mapping
    current_homography, current_condition = current_mapping
    if max(previous_condition, current_condition) > maximum_condition:
        return None

    previous_pixels = undistort_points(previous, matrix, dist)
    current_pixels = undistort_points(current, matrix, dist)
    previous_disk, previous_valid = _image_to_disk(
        previous_pixels, previous_homography
    )
    current_disk, current_valid = _image_to_disk(current_pixels, current_homography)
    radial_floor = max(np.finfo(float).eps, geometry.wheel_radius * 1e-7)
    previous_radius = np.linalg.norm(previous_disk, axis=1)
    current_radius = np.linalg.norm(current_disk, axis=1)
    valid = (
        previous_valid
        & current_valid
        & np.isfinite(previous_disk).all(axis=1)
        & np.isfinite(current_disk).all(axis=1)
        & (previous_radius > radial_floor)
        & (current_radius > radial_floor)
    )
    indices = np.flatnonzero(valid)
    if len(indices) < required:
        return None

    previous_angle = np.arctan2(
        previous_disk[indices, 1], previous_disk[indices, 0]
    )
    current_angle = np.arctan2(
        current_disk[indices, 1], current_disk[indices, 0]
    )
    changes = _wrapped(current_angle - previous_angle)

    # Every observation is a candidate mode.  The vectorized consensus table
    # keeps this deterministic without adding a per-feature Python loop to
    # every video interval (500 features require only about 2 MB here).
    candidate_residuals = np.abs(_wrapped(changes[None, :] - changes[:, None]))
    candidate_inliers = candidate_residuals <= threshold
    candidate_counts = np.count_nonzero(candidate_inliers, axis=1)
    candidate_medians = np.array(
        [
            np.median(row[mask]) if np.any(mask) else np.inf
            for row, mask in zip(candidate_residuals, candidate_inliers)
        ],
        dtype=float,
    )
    best_count = int(np.max(candidate_counts))
    contenders = np.flatnonzero(candidate_counts == best_count)
    best_index = int(contenders[np.argmin(candidate_medians[contenders])])
    best = candidate_inliers[best_index]
    if best_count < required:
        return None

    delta = _circular_mean(changes[best])
    local_residual = np.abs(_wrapped(changes - delta))
    local_inliers = local_residual <= threshold
    if np.count_nonzero(local_inliers) >= required:
        delta = _circular_mean(changes[local_inliers])
        local_residual = np.abs(_wrapped(changes - delta))
        local_inliers = local_residual <= threshold
    if np.count_nonzero(local_inliers) < required:
        return None

    inliers = np.zeros(len(previous), dtype=bool)
    inliers[indices[local_inliers]] = True
    residual = np.full(len(previous), np.nan, dtype=float)
    residual[indices] = local_residual
    return ImagePlaneRollCheck(
        delta_phi=float(delta),
        valid_mask=valid,
        inlier_mask=inliers,
        residual_rad=residual,
        homography_condition_prev=previous_condition,
        homography_condition_curr=current_condition,
    )


@dataclass(frozen=True)
class RollCorrespondences:
    """Frame-specific plane lifts expressed in the common zero-fork frame."""

    points_prev_camera: FloatArray
    points_curr_camera: FloatArray
    vectors_prev_fork: FloatArray
    vectors_curr_fork: FloatArray
    geometry_valid_mask: NDArray[np.bool_]
    face_sign: int
    view_confidence_prev: float
    view_confidence_curr: float


def prepare_roll_correspondences(
    uv_prev: ArrayLike,
    uv_curr: ArrayLike,
    K: ArrayLike,
    geometry: SwivelGeometry,
    psi_prev: float,
    psi_curr: float,
    *,
    face_sign: int | None = None,
    dist: ArrayLike | None = None,
    parallel_epsilon: float = 1e-8,
) -> RollCorrespondences:
    """Lift a pixel pair and remove hub motion plus fork swivel.

    If no side is forced, both frames must select the same visible face.  A
    face transition requires KLT re-seeding and therefore has no valid
    cross-frame sidewall correspondence.
    """

    previous, current = _pixel_pairs(uv_prev, uv_curr)
    matrix = validate_camera_matrix(K)
    if face_sign is None:
        sign_previous = geometry.visible_face_sign(float(psi_prev))
        sign_current = geometry.visible_face_sign(float(psi_curr))
        sign = sign_previous
        same_face = sign_previous == sign_current
    else:
        sign = int(face_sign)
        if sign not in (-1, 1):
            raise ValueError("face_sign must be +1, -1, or None")
        same_face = True

    plane_previous, normal_previous = geometry.face_plane_camera(psi_prev, sign)
    plane_current, normal_current = geometry.face_plane_camera(psi_curr, sign)
    points_previous, valid_previous = unproject_to_plane(
        previous,
        matrix,
        plane_previous,
        normal_previous,
        dist=dist,
        parallel_epsilon=parallel_epsilon,
    )
    points_current, valid_current = unproject_to_plane(
        current,
        matrix,
        plane_current,
        normal_current,
        dist=dist,
        parallel_epsilon=parallel_epsilon,
    )

    hub_previous = geometry.hub_center_camera(psi_prev)
    hub_current = geometry.hub_center_camera(psi_curr)
    vectors_previous_camera = points_previous - hub_previous
    vectors_current_camera = points_current - hub_current
    vectors_previous = geometry.camera_vectors_to_fork(
        np.nan_to_num(vectors_previous_camera), psi_prev
    )
    vectors_current = geometry.camera_vectors_to_fork(
        np.nan_to_num(vectors_current_camera), psi_curr
    )

    # Side-face points carry a constant +/-width/2 component along the axle.
    # Remove it so varying speckle radii cannot bias the constrained fit.
    axis = geometry.axle_zero_car
    vectors_previous -= (vectors_previous @ axis)[:, None] * axis
    vectors_current -= (vectors_current @ axis)[:, None] * axis
    radii_previous = np.linalg.norm(vectors_previous, axis=1)
    radii_current = np.linalg.norm(vectors_current, axis=1)
    radial_floor = max(np.finfo(float).eps, geometry.wheel_radius * 1e-7)
    valid = (
        np.asarray(valid_previous, dtype=bool)
        & np.asarray(valid_current, dtype=bool)
        & np.isfinite(vectors_previous).all(axis=1)
        & np.isfinite(vectors_current).all(axis=1)
        & (radii_previous > radial_floor)
        & (radii_current > radial_floor)
    )
    if not same_face:
        valid[:] = False
    vectors_previous[~valid] = np.nan
    vectors_current[~valid] = np.nan
    return RollCorrespondences(
        points_prev_camera=points_previous,
        points_curr_camera=points_current,
        vectors_prev_fork=vectors_previous,
        vectors_curr_fork=vectors_current,
        geometry_valid_mask=valid,
        face_sign=sign,
        view_confidence_prev=geometry.sidewall_view_confidence(psi_prev, sign),
        view_confidence_curr=geometry.sidewall_view_confidence(psi_curr, sign),
    )


@dataclass(frozen=True)
class AxisRotationFit:
    """Known-axis rotation fit before expansion to full pixel diagnostics."""

    delta_angle: float
    R_axis: FloatArray
    R_unconstrained: FloatArray | None
    inlier_mask: NDArray[np.bool_]
    residual_rad: FloatArray
    off_axis_residual_rad: float
    method: str


def circular_axis_fit(
    vectors_prev: ArrayLike,
    vectors_curr: ArrayLike,
    axis: ArrayLike,
    *,
    inlier_rad: float = np.deg2rad(1.0),
    min_inliers: int = 8,
    iterations: int = 200,
    rng: np.random.Generator | int | None = None,
) -> AxisRotationFit | None:
    """Robust one-axis circular fit, usable as an independent fallback."""

    previous = np.asarray(vectors_prev, dtype=float)
    current = np.asarray(vectors_curr, dtype=float)
    if previous.ndim != 2 or previous.shape[1:] != (3,) or current.shape != previous.shape:
        raise ValueError("vectors_prev/curr must have matching shape (N, 3)")
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
        raise ValueError("axis-fit vectors must be finite")
    direction = _unit_axis(axis)
    threshold = float(inlier_rad)
    required = int(min_inliers)
    if not np.isfinite(threshold) or not 0.0 < threshold <= np.pi:
        raise ValueError("inlier_rad must lie in (0, pi]")
    if required < 3:
        raise ValueError("min_inliers must be at least three")
    if len(previous) < required:
        return None
    angles = _signed_feature_angles(previous, current, direction)
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    best: NDArray[np.bool_] | None = None
    best_count = -1
    best_median = np.inf
    for _ in range(max(1, int(iterations))):
        candidate = float(angles[int(generator.integers(len(angles)))])
        residual = np.abs(_wrapped(angles - candidate))
        inliers = residual <= threshold
        count = int(np.count_nonzero(inliers))
        median = float(np.median(residual[inliers])) if count else np.inf
        if count > best_count or (count == best_count and median < best_median):
            best, best_count, best_median = inliers, count, median
    if best is None or best_count < required:
        return None
    angle = _circular_mean(angles[best])
    residual = np.abs(_wrapped(angles - angle))
    refined = residual <= threshold
    if np.count_nonzero(refined) >= required:
        angle = _circular_mean(angles[refined])
        residual = np.abs(_wrapped(angles - angle))
        refined = residual <= threshold
    return AxisRotationFit(
        delta_angle=angle,
        R_axis=_axis_rotation(direction, angle),
        R_unconstrained=None,
        inlier_mask=refined,
        residual_rad=residual,
        off_axis_residual_rad=float("nan"),
        method="circular",
    )


def fit_axis_rotation(
    vectors_prev: ArrayLike,
    vectors_curr: ArrayLike,
    axis: ArrayLike,
    *,
    ransac_iters: int = 200,
    inlier_rad: float = np.deg2rad(1.0),
    min_inliers: int = 8,
    rng: np.random.Generator | int | None = None,
    circular_fallback: bool = True,
) -> AxisRotationFit | None:
    """RANSAC/Kabsch fit followed by an exact known-axis refinement."""

    previous = np.asarray(vectors_prev, dtype=float)
    current = np.asarray(vectors_curr, dtype=float)
    if previous.ndim != 2 or previous.shape[1:] != (3,) or current.shape != previous.shape:
        raise ValueError("vectors_prev/curr must have matching shape (N, 3)")
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
        raise ValueError("axis-fit vectors must be finite")
    direction = _unit_axis(axis)
    threshold = float(inlier_rad)
    required = int(min_inliers)
    result = ransac_kabsch(
        previous,
        current,
        iters=int(ransac_iters),
        inlier_rad=threshold,
        min_inliers=required,
        rng=rng,
    )
    if result is None:
        if not circular_fallback:
            return None
        return circular_axis_fit(
            previous,
            current,
            direction,
            inlier_rad=threshold,
            min_inliers=required,
            iterations=ransac_iters,
            rng=rng,
        )

    unconstrained, initial_inliers = result
    rotvec = Rotation.from_matrix(unconstrained).as_rotvec()
    projected_angle = float(rotvec @ direction)
    angles = _signed_feature_angles(previous, current, direction)
    # Circular refinement respects wrapping and keeps the Kabsch/RANSAC
    # consensus.  Fall back to the projected rotvec only for a degenerate mean.
    angle = _circular_mean(angles[initial_inliers])
    if not np.isfinite(angle):
        angle = projected_angle
    constrained = _axis_rotation(direction, angle)
    residual = angular_residuals(previous, current, constrained)
    inliers = residual <= threshold
    if np.count_nonzero(inliers) >= required:
        angle = _circular_mean(angles[inliers])
        constrained = _axis_rotation(direction, angle)
        residual = angular_residuals(previous, current, constrained)
        inliers = residual <= threshold
    if np.count_nonzero(inliers) < required:
        if circular_fallback:
            return circular_axis_fit(
                previous,
                current,
                direction,
                inlier_rad=threshold,
                min_inliers=required,
                iterations=ransac_iters,
                rng=rng,
            )
        return None
    off_axis = float(np.linalg.norm(rotvec - projected_angle * direction))
    return AxisRotationFit(
        delta_angle=angle,
        R_axis=constrained,
        R_unconstrained=unconstrained,
        inlier_mask=inliers,
        residual_rad=residual,
        off_axis_residual_rad=off_axis,
        method="ransac_kabsch",
    )


@dataclass(frozen=True)
class RollEstimate:
    """One frame-pair roll increment with mask-aligned diagnostics."""

    delta_phi: float
    R_roll_fork: FloatArray | None
    R_unconstrained_fork: FloatArray | None
    inlier_mask: NDArray[np.bool_]
    geometry_valid_mask: NDArray[np.bool_]
    residual_rad: FloatArray
    off_axis_residual_rad: float
    face_sign: int
    view_confidence_prev: float
    view_confidence_curr: float
    method: str
    failure_reason: str | None = None
    detected_count: int = 0
    matched_count: int = 0
    median_fb_error_px: float = float("nan")
    image_plane_check: ImagePlaneRollCheck | None = None
    image_plane_disagreement_rad: float = float("nan")
    uv_prev: FloatArray = field(default_factory=lambda: np.empty((0, 2)))
    uv_curr: FloatArray = field(default_factory=lambda: np.empty((0, 2)))
    radial_rejected_count: int = 0
    reprojection_error_px: float = float("nan")
    angular_span_deg: float = 0.0

    @property
    def success(self) -> bool:
        return self.R_roll_fork is not None and np.isfinite(self.delta_phi)

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.geometry_valid_mask))

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def inlier_ratio(self) -> float:
        return self.inlier_count / self.valid_count if self.valid_count else 0.0

    @property
    def mean_inlier_residual_rad(self) -> float:
        if not self.inlier_count:
            return float("nan")
        return float(np.mean(self.residual_rad[self.inlier_mask]))

    @property
    def quality(self) -> dict[str, float | int | str | bool]:
        image_check = self.image_plane_check
        return {
            "success": self.success,
            "method": self.method,
            "detected_count": int(self.detected_count),
            "matched_count": int(self.matched_count),
            "valid_count": self.valid_count,
            "inlier_count": self.inlier_count,
            "inlier_ratio": self.inlier_ratio,
            "mean_inlier_residual_deg": float(np.rad2deg(self.mean_inlier_residual_rad)),
            "off_axis_residual_deg": float(np.rad2deg(self.off_axis_residual_rad)),
            "view_confidence": min(self.view_confidence_prev, self.view_confidence_curr),
            "median_fb_error_px": float(self.median_fb_error_px),
            "radial_rejected_count": self.radial_rejected_count,
            "roll_reprojection_error_px": self.reprojection_error_px,
            "angular_span_deg": self.angular_span_deg,
            "image_plane_circular_delta_phi_deg": (
                float(np.rad2deg(image_check.delta_phi))
                if image_check is not None
                else float("nan")
            ),
            "image_plane_circular_disagreement_deg": float(
                np.rad2deg(self.image_plane_disagreement_rad)
            ),
            "image_plane_circular_inlier_count": (
                image_check.inlier_count if image_check is not None else 0
            ),
            "image_plane_circular_inlier_ratio": (
                image_check.inlier_ratio if image_check is not None else 0.0
            ),
            "image_plane_circular_mean_residual_deg": (
                float(np.rad2deg(image_check.mean_inlier_residual_rad))
                if image_check is not None
                else float("nan")
            ),
        }


def _failed_estimate(
    count: int,
    *,
    reason: str,
    face_sign: int,
    valid_mask: NDArray[np.bool_] | None = None,
    view_previous: float = 0.0,
    view_current: float = 0.0,
    image_plane_check: ImagePlaneRollCheck | None = None,
) -> RollEstimate:
    geometry_valid = (
        np.zeros(count, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    return RollEstimate(
        delta_phi=float("nan"),
        R_roll_fork=None,
        R_unconstrained_fork=None,
        inlier_mask=np.zeros(count, dtype=bool),
        geometry_valid_mask=geometry_valid,
        residual_rad=np.full(count, np.nan, dtype=float),
        off_axis_residual_rad=float("nan"),
        face_sign=int(face_sign),
        view_confidence_prev=float(view_previous),
        view_confidence_curr=float(view_current),
        method="failed",
        failure_reason=reason,
        matched_count=count,
        image_plane_check=image_plane_check,
    )


def estimate_roll_increment(
    uv_prev: ArrayLike,
    uv_curr: ArrayLike,
    K: ArrayLike,
    geometry: SwivelGeometry,
    psi_prev: float,
    psi_curr: float,
    *,
    face_sign: int | None = None,
    dist: ArrayLike | None = None,
    min_view_confidence: float = 0.08,
    ransac_iters: int = 200,
    ransac_inlier_deg: float = 1.0,
    min_inliers: int = 8,
    rng: np.random.Generator | int | None = None,
    circular_fallback: bool = True,
) -> RollEstimate:
    """Recover signed ``delta_phi`` from already tracked sidewall pixels."""

    previous, current = _pixel_pairs(uv_prev, uv_curr)
    correspondences = prepare_roll_correspondences(
        previous,
        current,
        K,
        geometry,
        psi_prev,
        psi_curr,
        face_sign=face_sign,
        dist=dist,
    )
    image_check = image_plane_circular_check(
        previous,
        current,
        K,
        geometry,
        psi_prev,
        psi_curr,
        face_sign=correspondences.face_sign,
        dist=dist,
        inlier_rad=np.deg2rad(float(ransac_inlier_deg)),
        min_inliers=min_inliers,
    )
    count = len(previous)
    minimum_view = float(min_view_confidence)
    if not np.isfinite(minimum_view) or not 0.0 <= minimum_view <= 1.0:
        raise ValueError("min_view_confidence must lie in [0, 1]")
    view = min(
        correspondences.view_confidence_prev,
        correspondences.view_confidence_curr,
    )
    if view < minimum_view:
        return _failed_estimate(
            count,
            reason=f"sidewall view confidence {view:.3f} is below {minimum_view:.3f}",
            face_sign=correspondences.face_sign,
            valid_mask=correspondences.geometry_valid_mask,
            view_previous=correspondences.view_confidence_prev,
            view_current=correspondences.view_confidence_curr,
            image_plane_check=image_check,
        )
    indices = np.flatnonzero(correspondences.geometry_valid_mask)
    if len(indices) < max(3, int(min_inliers)):
        return _failed_estimate(
            count,
            reason=f"only {len(indices)} geometry-valid matches; need {max(3, int(min_inliers))}",
            face_sign=correspondences.face_sign,
            valid_mask=correspondences.geometry_valid_mask,
            view_previous=correspondences.view_confidence_prev,
            view_current=correspondences.view_confidence_curr,
            image_plane_check=image_check,
        )
    fit = fit_axis_rotation(
        correspondences.vectors_prev_fork[indices],
        correspondences.vectors_curr_fork[indices],
        geometry.axle_zero_car,
        ransac_iters=ransac_iters,
        inlier_rad=np.deg2rad(float(ransac_inlier_deg)),
        min_inliers=min_inliers,
        rng=rng,
        circular_fallback=circular_fallback,
    )
    if fit is None:
        return _failed_estimate(
            count,
            reason=f"roll fit found fewer than {min_inliers} inliers",
            face_sign=correspondences.face_sign,
            valid_mask=correspondences.geometry_valid_mask,
            view_previous=correspondences.view_confidence_prev,
            view_current=correspondences.view_confidence_curr,
            image_plane_check=image_check,
        )
    inliers = np.zeros(count, dtype=bool)
    inliers[indices[fit.inlier_mask]] = True
    residual = np.full(count, np.nan, dtype=float)
    residual[indices] = fit.residual_rad
    image_disagreement = (
        float(_wrapped(fit.delta_angle - image_check.delta_phi))
        if image_check is not None
        else float("nan")
    )
    return RollEstimate(
        delta_phi=fit.delta_angle,
        R_roll_fork=fit.R_axis,
        R_unconstrained_fork=fit.R_unconstrained,
        inlier_mask=inliers,
        geometry_valid_mask=correspondences.geometry_valid_mask,
        residual_rad=residual,
        off_axis_residual_rad=fit.off_axis_residual_rad,
        face_sign=correspondences.face_sign,
        view_confidence_prev=correspondences.view_confidence_prev,
        view_confidence_curr=correspondences.view_confidence_curr,
        method=fit.method,
        matched_count=count,
        image_plane_check=image_check,
        image_plane_disagreement_rad=image_disagreement,
    )


class RollEstimator:
    """Frame-pair KLT front end around :func:`estimate_roll_increment`."""

    def __init__(
        self,
        geometry: SwivelGeometry,
        K: ArrayLike,
        *,
        dist: ArrayLike | None = None,
        track_config: KLTConfig | Mapping[str, Any] | None = None,
        min_view_confidence: float = 0.08,
        ransac_iters: int = 200,
        ransac_inlier_deg: float = 1.0,
        min_inliers: int = 8,
        random_seed: int = 7,
        circular_fallback: bool = True,
        motion_prediction: bool = False,
        max_radial_error_fraction: float | None = None,
        max_reprojection_error_px: float | None = None,
        min_angular_span_deg: float = 0.0,
    ) -> None:
        self.geometry = geometry
        self.K = validate_camera_matrix(K)
        self.dist = None if dist is None else np.asarray(dist, dtype=float).reshape(-1)
        self.track_config = (
            track_config
            if isinstance(track_config, KLTConfig)
            else KLTConfig.from_mapping(track_config)
        )
        self.min_view_confidence = float(min_view_confidence)
        self.ransac_iters = int(ransac_iters)
        self.ransac_inlier_deg = float(ransac_inlier_deg)
        self.min_inliers = int(min_inliers)
        self.rng = np.random.default_rng(int(random_seed))
        self.circular_fallback = bool(circular_fallback)
        self.motion_prediction = bool(motion_prediction)
        self.max_radial_error_fraction = max_radial_error_fraction
        self.max_reprojection_error_px = max_reprojection_error_px
        self.min_angular_span_deg = float(min_angular_span_deg)
        for value in [max_radial_error_fraction, max_reprojection_error_px]:
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError("radial/reprojection limits must be positive")
        if not 0 <= self.min_angular_span_deg < 360:
            raise ValueError("min_angular_span_deg must lie in [0,360)")

    def predict_pixels(self, pixels, psi_prev, psi_curr, face_sign, delta_phi=0.):
        center, normal = self.geometry.face_plane_camera(psi_prev, face_sign)
        points, valid = unproject_to_plane(pixels, self.K, center, normal, dist=self.dist)
        vectors = self.geometry.camera_vectors_to_fork(np.nan_to_num(points-center), psi_prev)
        vectors = vectors @ _axis_rotation(self.geometry.axle_zero_car, delta_phi).T
        current_center, _ = self.geometry.face_plane_camera(psi_curr, face_sign)
        current = vectors @ self.geometry.R_camera_from_fork(psi_curr).T + current_center
        projected, front = project_camera_points(current, self.K)
        if self.dist is not None and np.any(self.dist) and np.any(front):
            distorted, _ = cv2.projectPoints(current[front], np.zeros(3), np.zeros(3), self.K, self.dist)
            projected[front] = distorted.reshape(-1, 2)
        valid &= front
        return projected, valid

    def estimate(
        self,
        frame_prev: NDArray[np.generic],
        frame_curr: NDArray[np.generic],
        mask_prev: NDArray[np.generic],
        mask_curr: NDArray[np.generic],
        psi_prev: float,
        psi_curr: float,
        *,
        face_sign: int | None = None,
        predicted_delta_phi: float = 0.0,
    ) -> RollEstimate:
        """Track a frame pair and return roll plus inlier/view diagnostics."""

        features = detect_features(frame_prev, mask_prev, self.track_config)
        sign = self.geometry.visible_face_sign(psi_prev) if face_sign is None else face_sign
        initial = None
        if self.motion_prediction and len(features):
            initial, good = self.predict_pixels(features, psi_prev, psi_curr, sign, predicted_delta_phi)
            initial[~good] = features[~good]
        tracked = track_features(
            frame_prev,
            frame_curr,
            features,
            prev_mask=mask_prev,
            curr_mask=mask_curr,
            config=self.track_config,
            initial_uv_curr=initial,
        )
        previous, current = tracked.uv_prev, tracked.uv_curr
        keep = np.ones(len(previous), dtype=bool)
        if self.max_radial_error_fraction is not None and len(previous):
            pairs = prepare_roll_correspondences(previous, current, self.K, self.geometry,
                                                 psi_prev, psi_curr, face_sign=sign, dist=self.dist)
            ra = np.linalg.norm(pairs.vectors_prev_fork, axis=1)
            rb = np.linalg.norm(pairs.vectors_curr_fork, axis=1)
            keep &= pairs.geometry_valid_mask & (np.abs(rb-ra) <= self.max_radial_error_fraction*self.geometry.wheel_radius)
        radial_rejected = int((~keep).sum())
        previous, current = previous[keep], current[keep]
        estimate = estimate_roll_increment(
            previous,
            current,
            self.K,
            self.geometry,
            psi_prev,
            psi_curr,
            face_sign=face_sign,
            dist=self.dist,
            min_view_confidence=self.min_view_confidence,
            ransac_iters=self.ransac_iters,
            ransac_inlier_deg=self.ransac_inlier_deg,
            min_inliers=self.min_inliers,
            rng=self.rng,
            circular_fallback=self.circular_fallback,
        )
        reprojection = float("nan")
        span = 0.
        if estimate.success and estimate.inlier_count:
            predicted, good = self.predict_pixels(previous, psi_prev, psi_curr, sign, estimate.delta_phi)
            usable = good & estimate.inlier_mask
            if usable.any():
                reprojection = float(np.median(np.linalg.norm(predicted[usable]-current[usable], axis=1)))
            pairs = prepare_roll_correspondences(previous, current, self.K, self.geometry,
                                                 psi_prev, psi_curr, face_sign=sign, dist=self.dist)
            first, second = self.geometry.radial_basis_zero()
            vectors = pairs.vectors_prev_fork[estimate.inlier_mask]
            angles = np.sort(np.mod(np.arctan2(vectors @ second, vectors @ first), 2*np.pi))
            if len(angles) >= 2:
                span = float(np.rad2deg(2*np.pi - np.diff(np.r_[angles, angles[0]+2*np.pi]).max()))
            reason = None
            if self.max_reprojection_error_px is not None and (not np.isfinite(reprojection) or reprojection > self.max_reprojection_error_px):
                reason = "wheel motion reprojection error exceeds pixel gate"
            if span < self.min_angular_span_deg:
                reason = "wheel inliers occupy too small an angular span"
            if reason:
                estimate = replace(estimate, R_roll_fork=None, failure_reason=reason)
        median_fb = (
            float(np.median(tracked.fb_error)) if len(tracked.fb_error) else float("nan")
        )
        return replace(
            estimate,
            detected_count=int(tracked.detected_count),
            matched_count=int(tracked.count),
            median_fb_error_px=median_fb,
            uv_prev=previous, uv_curr=current,
            radial_rejected_count=radial_rejected,
            reprojection_error_px=reprojection, angular_span_deg=span,
        )


__all__ = [
    "AxisRotationFit",
    "ImagePlaneRollCheck",
    "RollCorrespondences",
    "RollEstimate",
    "RollEstimator",
    "circular_axis_fit",
    "estimate_roll_increment",
    "fit_axis_rotation",
    "image_plane_circular_check",
    "prepare_roll_correspondences",
]
