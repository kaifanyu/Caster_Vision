"""Device-independent rigid-rotation fitting utilities.

The row-array convention used throughout is ``b ~= a @ R.T``; equivalently,
for column vectors, ``b_i ~= R @ a_i``.
"""

from __future__ import annotations

import numpy as np


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
    """Return the proper rotation satisfying ``b ~= a @ R.T``."""

    first, second = _paired_vectors(a, b)
    covariance = first.T @ second
    U, _, Vt = np.linalg.svd(covariance)
    correction = np.sign(np.linalg.det(Vt.T @ U.T))
    if correction == 0:
        correction = 1.0
    return Vt.T @ np.diag([1.0, 1.0, correction]) @ U.T


def angular_residuals(a: np.ndarray, b: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Return per-correspondence angular errors in radians."""

    first, second = _paired_vectors(a, b)
    rotation = np.asarray(R, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("R must be a finite 3x3 matrix")
    prediction = first @ rotation.T
    cosine = np.einsum("ij,ij->i", prediction, second)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _sample_is_degenerate(points: np.ndarray) -> bool:
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
    """Robustly fit ``b ~= R a`` and return ``(R, inlier_mask)``."""

    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    if first.ndim != 2 or first.shape[1:] != (3,) or second.shape != first.shape:
        raise ValueError("a and b must have matching shape (N, 3)")
    if len(first) < max(3, int(min_inliers)):
        return None
    if int(iters) <= 0:
        raise ValueError("iters must be positive")
    if not np.isfinite(inlier_rad) or not (0.0 < inlier_rad <= np.pi):
        raise ValueError("inlier_rad must lie in (0, pi]")
    if int(min_inliers) < 3:
        raise ValueError("min_inliers must be at least three")
    first, second = _paired_vectors(first, second)
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    best: np.ndarray | None = None
    best_count = -1
    best_median = np.inf
    for _ in range(int(iters)):
        sample = generator.choice(len(first), 3, replace=False)
        if _sample_is_degenerate(first[sample]) or _sample_is_degenerate(second[sample]):
            continue
        candidate = kabsch(first[sample], second[sample])
        residual = angular_residuals(first, second, candidate)
        inliers = residual <= inlier_rad
        count = int(np.count_nonzero(inliers))
        median = float(np.median(residual[inliers])) if count else np.inf
        if count > best_count or (count == best_count and median < best_median):
            best, best_count, best_median = inliers, count, median
    if best is None or best_count < int(min_inliers):
        return None
    rotation = kabsch(first[best], second[best])
    refined = angular_residuals(first, second, rotation) <= inlier_rad
    if np.count_nonzero(refined) >= int(min_inliers):
        rotation = kabsch(first[refined], second[refined])
        final = angular_residuals(first, second, rotation) <= inlier_rad
        if np.count_nonzero(final) >= int(min_inliers):
            refined = final
    return rotation, refined
