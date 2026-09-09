"""Robust incremental rotation estimation from tracked speckles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .camera import undistort_points
from .sphere import unproject_to_sphere, viewing_angle


def _paired_vectors(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    if first.ndim != 2 or first.shape[1:] != (3,):
        raise ValueError("a must have shape (N, 3)")
    if second.shape != first.shape:
        raise ValueError("b must have the same (N, 3) shape as a")
    if len(first) < 3:
        raise ValueError("at least three correspondences are required")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("correspondence vectors must be finite")
    norms_a = np.linalg.norm(first, axis=1)
    norms_b = np.linalg.norm(second, axis=1)
    if np.any(norms_a <= np.finfo(float).eps) or np.any(
        norms_b <= np.finfo(float).eps
    ):
        raise ValueError("correspondence vectors must be non-zero")
    return first / norms_a[:, None], second / norms_b[:, None]


def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the proper rotation satisfying ``b ~= a @ R.T``.

    Equivalently, for column-vector notation this minimizes
    ``sum(||R @ a_i - b_i||^2)``.  The determinant correction prevents a
    reflected correspondence set from producing an improper rotation.
    """

    first, second = _paired_vectors(a, b)
    covariance = first.T @ second
    U, _, Vt = np.linalg.svd(covariance)
    correction = np.sign(np.linalg.det(Vt.T @ U.T))
    if correction == 0:  # Defensive; SVD factors should be orthogonal.
        correction = 1.0
    return Vt.T @ np.diag([1.0, 1.0, correction]) @ U.T


def angular_residuals(a: np.ndarray, b: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Per-correspondence angular errors in radians."""

    first, second = _paired_vectors(a, b)
    rotation = np.asarray(R, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("R must be a finite 3x3 matrix")
    prediction = first @ rotation.T
    cosine = np.einsum("ij,ij->i", prediction, second)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _coerce_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _sample_is_degenerate(points: np.ndarray) -> bool:
    # Two non-collinear directions determine a 3-D rotation.  Reject samples
    # that are effectively all parallel, which otherwise make Kabsch choose an
    # arbitrary twist about the common direction.
    cross_norms = np.linalg.norm(
        np.cross(points[[0, 0, 1]], points[[1, 2, 2]]), axis=1
    )
    return bool(np.max(cross_norms) < 1e-6)


def ransac_kabsch(
    a: np.ndarray,
    b: np.ndarray,
    iters: int = 200,
    inlier_rad: float = np.deg2rad(1.0),
    min_inliers: int = 8,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Robustly fit ``b ~= R a`` and return ``(R, inlier_mask)``.

    The public return form intentionally matches the build specification.
    Ties in consensus size are broken by the lower median angular residual.
    """

    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    if first.ndim != 2 or first.shape[1:] != (3,) or second.shape != first.shape:
        raise ValueError("a and b must have matching shape (N, 3)")
    count = len(first)
    if count < 3:
        return None
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("RANSAC inputs must be finite")
    if int(iters) <= 0:
        raise ValueError("iters must be positive")
    if not np.isfinite(inlier_rad) or not (0 < inlier_rad <= np.pi):
        raise ValueError("inlier_rad must lie in (0, pi]")
    if int(min_inliers) < 3:
        raise ValueError("min_inliers must be at least three")
    if count < int(min_inliers):
        return None

    norms_a = np.linalg.norm(first, axis=1)
    norms_b = np.linalg.norm(second, axis=1)
    if np.any(norms_a <= np.finfo(float).eps) or np.any(
        norms_b <= np.finfo(float).eps
    ):
        raise ValueError("RANSAC inputs must be non-zero")
    first = first / norms_a[:, None]
    second = second / norms_b[:, None]

    generator = _coerce_rng(rng)
    best_inliers: np.ndarray | None = None
    best_count = -1
    best_median = np.inf
    for _ in range(int(iters)):
        sample = generator.choice(count, 3, replace=False)
        if _sample_is_degenerate(first[sample]) or _sample_is_degenerate(
            second[sample]
        ):
            continue
        candidate = kabsch(first[sample], second[sample])
        residual = angular_residuals(first, second, candidate)
        inliers = residual <= inlier_rad
        inlier_count = int(np.count_nonzero(inliers))
        median = float(np.median(residual[inliers])) if inlier_count else np.inf
        if inlier_count > best_count or (
            inlier_count == best_count and median < best_median
        ):
            best_inliers = inliers
            best_count = inlier_count
            best_median = median

    if best_inliers is None or best_count < int(min_inliers):
        return None

    rotation = kabsch(first[best_inliers], second[best_inliers])
    # One local consensus/refit pass normally absorbs points just outside the
    # initial minimal-sample model without turning this into a complex solver.
    refined = angular_residuals(first, second, rotation) <= inlier_rad
    if np.count_nonzero(refined) >= int(min_inliers):
        rotation = kabsch(first[refined], second[refined])
        final = angular_residuals(first, second, rotation) <= inlier_rad
        if np.count_nonzero(final) >= int(min_inliers):
            refined = final
    return rotation, refined


@dataclass(frozen=True)
class SphereCorrespondences:
    """Unprojected frame-pair correspondences and their geometry gate."""

    dirs_prev: np.ndarray
    dirs_curr: np.ndarray
    valid_mask: np.ndarray
    view_angle_prev_rad: np.ndarray
    view_angle_curr_rad: np.ndarray

    def __post_init__(self) -> None:
        previous = np.asarray(self.dirs_prev, dtype=np.float64)
        current = np.asarray(self.dirs_curr, dtype=np.float64)
        valid = np.asarray(self.valid_mask, dtype=bool)
        angle_previous = np.asarray(self.view_angle_prev_rad, dtype=np.float64)
        angle_current = np.asarray(self.view_angle_curr_rad, dtype=np.float64)
        if previous.ndim != 2 or previous.shape[1:] != (3,):
            raise ValueError("dirs_prev must have shape (N, 3)")
        if current.shape != previous.shape:
            raise ValueError("dirs_curr must match dirs_prev")
        count = len(previous)
        if valid.shape != (count,):
            raise ValueError("valid_mask must have shape (N,)")
        if angle_previous.shape != (count,) or angle_current.shape != (count,):
            raise ValueError("view angles must have shape (N,)")
        object.__setattr__(self, "dirs_prev", previous)
        object.__setattr__(self, "dirs_curr", current)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "view_angle_prev_rad", angle_previous)
        object.__setattr__(self, "view_angle_curr_rad", angle_current)


@dataclass(frozen=True)
class HemisphereEstimate:
    """Incremental rotation plus diagnostics for one hemisphere."""

    R: np.ndarray | None
    inlier_mask: np.ndarray
    geometry_valid_mask: np.ndarray
    residual_rad: np.ndarray
    view_angle_prev_rad: np.ndarray
    view_angle_curr_rad: np.ndarray
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        inliers = np.asarray(self.inlier_mask, dtype=bool)
        valid = np.asarray(self.geometry_valid_mask, dtype=bool)
        residual = np.asarray(self.residual_rad, dtype=np.float64)
        previous = np.asarray(self.view_angle_prev_rad, dtype=np.float64)
        current = np.asarray(self.view_angle_curr_rad, dtype=np.float64)
        shape = valid.shape
        if len(shape) != 1:
            raise ValueError("diagnostic masks must be one-dimensional")
        if inliers.shape != shape or residual.shape != shape:
            raise ValueError("inlier and residual arrays must match valid mask")
        if previous.shape != shape or current.shape != shape:
            raise ValueError("view angle arrays must match valid mask")
        if np.any(inliers & ~valid):
            raise ValueError("an inlier must pass the geometry gate")
        if self.R is not None:
            rotation = np.asarray(self.R, dtype=np.float64)
            if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
                raise ValueError("R must be None or a finite 3x3 matrix")
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
                raise ValueError("R must be orthonormal")
            if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
                raise ValueError("R must be a proper rotation with determinant +1")
            object.__setattr__(self, "R", rotation)
        object.__setattr__(self, "inlier_mask", inliers)
        object.__setattr__(self, "geometry_valid_mask", valid)
        object.__setattr__(self, "residual_rad", residual)
        object.__setattr__(self, "view_angle_prev_rad", previous)
        object.__setattr__(self, "view_angle_curr_rad", current)

    @property
    def success(self) -> bool:
        return self.R is not None

    @property
    def match_count(self) -> int:
        return len(self.geometry_valid_mask)

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
    def mean_inlier_residual_deg(self) -> float:
        return float(np.rad2deg(self.mean_inlier_residual_rad))


@dataclass(frozen=True)
class FrameRotationEstimate:
    top: HemisphereEstimate
    bottom: HemisphereEstimate

    def __getitem__(self, hemisphere: str) -> HemisphereEstimate:
        if hemisphere not in {"top", "bottom"}:
            raise KeyError(hemisphere)
        return getattr(self, hemisphere)


def _pixel_pairs(
    uv_prev: np.ndarray, uv_curr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    previous = np.asarray(uv_prev, dtype=np.float64)
    current = np.asarray(uv_curr, dtype=np.float64)
    if previous.ndim != 2 or previous.shape[1:] != (2,):
        raise ValueError("uv_prev must have shape (N, 2)")
    if current.shape != previous.shape:
        raise ValueError("uv_curr must match uv_prev's shape")
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
        raise ValueError("tracked pixels must be finite")
    return previous, current


def prepare_sphere_correspondences(
    uv_prev: np.ndarray,
    uv_curr: np.ndarray,
    K: np.ndarray,
    C: np.ndarray,
    radius: float = 1.0,
    *,
    limb_cull_deg: float = 65.0,
    dist: np.ndarray | None = None,
) -> SphereCorrespondences:
    """Undistort, unproject, and apply ray-validity and limb gates."""

    previous, current = _pixel_pairs(uv_prev, uv_curr)
    if not np.isfinite(limb_cull_deg) or not (0 < limb_cull_deg <= 90):
        raise ValueError("limb_cull_deg must lie in (0, 90]")
    if dist is not None:
        previous = undistort_points(previous, K, dist)
        current = undistort_points(current, K, dist)

    dirs_previous, valid_previous = unproject_to_sphere(previous, K, C, radius)
    dirs_current, valid_current = unproject_to_sphere(current, K, C, radius)
    angles_previous = viewing_angle(dirs_previous, C, radius)
    angles_current = viewing_angle(dirs_current, C, radius)
    limb_limit = np.deg2rad(float(limb_cull_deg))
    valid = (
        np.asarray(valid_previous, dtype=bool)
        & np.asarray(valid_current, dtype=bool)
        & np.all(np.isfinite(dirs_previous), axis=1)
        & np.all(np.isfinite(dirs_current), axis=1)
        & np.isfinite(angles_previous)
        & np.isfinite(angles_current)
        & (angles_previous <= limb_limit)
        & (angles_current <= limb_limit)
    )
    return SphereCorrespondences(
        dirs_prev=dirs_previous,
        dirs_curr=dirs_current,
        valid_mask=valid,
        view_angle_prev_rad=angles_previous,
        view_angle_curr_rad=angles_current,
    )


def _failed_estimate(
    correspondences: SphereCorrespondences,
    reason: str,
) -> HemisphereEstimate:
    count = len(correspondences.valid_mask)
    return HemisphereEstimate(
        R=None,
        inlier_mask=np.zeros(count, dtype=bool),
        geometry_valid_mask=correspondences.valid_mask,
        residual_rad=np.full(count, np.nan, dtype=float),
        view_angle_prev_rad=correspondences.view_angle_prev_rad,
        view_angle_curr_rad=correspondences.view_angle_curr_rad,
        failure_reason=reason,
    )


def solve_hemisphere_increment(
    uv_prev: np.ndarray,
    uv_curr: np.ndarray,
    K: np.ndarray,
    C: np.ndarray,
    radius: float = 1.0,
    *,
    limb_cull_deg: float = 65.0,
    ransac_iters: int = 200,
    ransac_inlier_deg: float = 1.0,
    min_inliers: int = 8,
    rng: np.random.Generator | int | None = None,
    dist: np.ndarray | None = None,
) -> HemisphereEstimate:
    """Estimate one camera-frame increment from tracked pixel pairs."""

    correspondences = prepare_sphere_correspondences(
        uv_prev,
        uv_curr,
        K,
        C,
        radius,
        limb_cull_deg=limb_cull_deg,
        dist=dist,
    )
    valid_indices = np.flatnonzero(correspondences.valid_mask)
    required = max(3, int(min_inliers))
    if len(valid_indices) < required:
        return _failed_estimate(
            correspondences,
            f"only {len(valid_indices)} geometry-valid matches; need {required}",
        )

    result = ransac_kabsch(
        correspondences.dirs_prev[valid_indices],
        correspondences.dirs_curr[valid_indices],
        iters=ransac_iters,
        inlier_rad=np.deg2rad(float(ransac_inlier_deg)),
        min_inliers=min_inliers,
        rng=rng,
    )
    if result is None:
        return _failed_estimate(
            correspondences,
            f"RANSAC found fewer than {min_inliers} inliers",
        )

    rotation, local_inliers = result
    inliers = np.zeros(len(correspondences.valid_mask), dtype=bool)
    inliers[valid_indices[local_inliers]] = True
    residuals = np.full(len(inliers), np.nan, dtype=float)
    residuals[valid_indices] = angular_residuals(
        correspondences.dirs_prev[valid_indices],
        correspondences.dirs_curr[valid_indices],
        rotation,
    )
    return HemisphereEstimate(
        R=rotation,
        inlier_mask=inliers,
        geometry_valid_mask=correspondences.valid_mask,
        residual_rad=residuals,
        view_angle_prev_rad=correspondences.view_angle_prev_rad,
        view_angle_curr_rad=correspondences.view_angle_curr_rad,
    )


def _matches_for(matches: Any, name: str) -> Any:
    if isinstance(matches, Mapping):
        if name not in matches:
            raise KeyError(f"matches do not contain {name!r}")
        return matches[name]
    if hasattr(matches, name):
        return getattr(matches, name)
    raise TypeError("matches must be a FrameMatches-like object or mapping")


def _uv_from_matches(matches: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(matches, Mapping):
        try:
            return np.asarray(matches["uv_prev"]), np.asarray(matches["uv_curr"])
        except KeyError as exc:
            raise KeyError("hemisphere matches need uv_prev and uv_curr") from exc
    if hasattr(matches, "uv_prev") and hasattr(matches, "uv_curr"):
        return np.asarray(matches.uv_prev), np.asarray(matches.uv_curr)
    if isinstance(matches, (tuple, list)) and len(matches) == 2:
        return np.asarray(matches[0]), np.asarray(matches[1])
    raise TypeError("hemisphere matches need uv_prev and uv_curr")


def solve_frame_increments(
    matches: Any,
    K: np.ndarray,
    C: np.ndarray,
    radius: float = 1.0,
    **kwargs: Any,
) -> FrameRotationEstimate:
    """Solve both hemisphere increments from a ``FrameMatches``-like value."""

    top_previous, top_current = _uv_from_matches(_matches_for(matches, "top"))
    bottom_previous, bottom_current = _uv_from_matches(
        _matches_for(matches, "bottom")
    )
    # A shared Generator is intentionally consumed in sequence, giving each
    # solve independent samples while preserving full-run reproducibility.
    top = solve_hemisphere_increment(
        top_previous, top_current, K, C, radius, **kwargs
    )
    bottom = solve_hemisphere_increment(
        bottom_previous, bottom_current, K, C, radius, **kwargs
    )
    return FrameRotationEstimate(top=top, bottom=bottom)


# Readable aliases for callers that prefer "estimate" terminology.
estimate_hemisphere_increment = solve_hemisphere_increment
estimate_frame_increments = solve_frame_increments
