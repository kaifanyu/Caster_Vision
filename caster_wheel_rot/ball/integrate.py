"""Rotation accumulation, ball-frame conversion, and axis calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from common.rotation import decompose_alpha_beta


def _as_rotation_matrix(value: np.ndarray, name: str = "rotation") -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(matrix), 1.0, atol=1e-6
    ):
        raise ValueError(f"{name} must be a proper rotation matrix")
    return matrix


def accumulate_increments(
    increments: Sequence[np.ndarray | None],
    *,
    include_identity: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Left-accumulate camera-frame increments.

    Each increment follows ``surface_curr = R_increment @ surface_prev``.  A
    missing increment holds the last valid orientation and is marked false in
    the returned validity array.  With ``include_identity=True`` the output has
    one more element than ``increments`` and begins at identity.
    """

    absolute: list[np.ndarray] = [np.eye(3)] if include_identity else []
    valid: list[bool] = [True] if include_identity else []
    current = np.eye(3)
    for index, increment in enumerate(increments):
        if increment is None:
            absolute.append(current.copy())
            valid.append(False)
            continue
        inc = _as_rotation_matrix(increment, f"increments[{index}]")
        current = inc @ current
        # Project away tiny numerical drift without altering the convention.
        u, _, vt = np.linalg.svd(current)
        current = u @ vt
        if np.linalg.det(current) < 0:
            u[:, -1] *= -1
            current = u @ vt
        absolute.append(current.copy())
        valid.append(True)
    return np.asarray(absolute), np.asarray(valid, dtype=bool)


def camera_to_ball(R_cam: np.ndarray, R_bc: np.ndarray) -> np.ndarray:
    """Express camera-coordinate rotations in fixed ball axes.

    ``R_bc`` maps ball-frame vectors into camera coordinates.
    """

    frame = _as_rotation_matrix(R_bc, "R_bc")
    rotations = np.asarray(R_cam, dtype=float)
    if rotations.shape[-2:] != (3, 3):
        raise ValueError("R_cam must end in shape (3, 3)")
    return np.einsum("ji,...jk,kl->...il", frame, rotations, frame)


def _unwrap_with_gaps(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.full_like(values, np.nan, dtype=float)
    indices = np.flatnonzero(valid & np.isfinite(values))
    if len(indices):
        output[indices] = np.unwrap(values[indices])
    return output


@dataclass(frozen=True)
class DecomposedMotion:
    alpha: np.ndarray
    beta_top: np.ndarray
    beta_bottom: np.ndarray
    alpha_top: np.ndarray
    alpha_bottom: np.ndarray
    gamma_top: np.ndarray
    gamma_bottom: np.ndarray
    valid_top: np.ndarray
    valid_bottom: np.ndarray


def decompose_hemispheres(
    R_top_cam: np.ndarray,
    R_bottom_cam: np.ndarray,
    R_bc: np.ndarray,
    *,
    valid_top: np.ndarray | None = None,
    valid_bottom: np.ndarray | None = None,
    unwrap: bool = True,
) -> DecomposedMotion:
    """Convert absolute hemisphere rotations to ``alpha, beta_top, beta_bottom``."""

    top = np.asarray(R_top_cam, dtype=float)
    bottom = np.asarray(R_bottom_cam, dtype=float)
    if top.shape != bottom.shape or top.ndim != 3 or top.shape[1:] != (3, 3):
        raise ValueError("top and bottom rotations must both have shape (N, 3, 3)")
    n_frames = len(top)
    vt = np.ones(n_frames, dtype=bool) if valid_top is None else np.asarray(valid_top, bool)
    vb = (
        np.ones(n_frames, dtype=bool)
        if valid_bottom is None
        else np.asarray(valid_bottom, bool)
    )
    if vt.shape != (n_frames,) or vb.shape != (n_frames,):
        raise ValueError("validity masks must have shape (N,)")

    top_ball = camera_to_ball(top, R_bc)
    bottom_ball = camera_to_ball(bottom, R_bc)
    top_components = np.array([decompose_alpha_beta(r) for r in top_ball])
    bottom_components = np.array([decompose_alpha_beta(r) for r in bottom_ball])
    alpha_top, beta_top, gamma_top = top_components.T
    alpha_bottom, beta_bottom, gamma_bottom = bottom_components.T

    if unwrap:
        alpha_top = _unwrap_with_gaps(alpha_top, vt)
        alpha_bottom = _unwrap_with_gaps(alpha_bottom, vb)
        beta_top = _unwrap_with_gaps(beta_top, vt)
        beta_bottom = _unwrap_with_gaps(beta_bottom, vb)
    else:
        alpha_top = np.where(vt, alpha_top, np.nan)
        alpha_bottom = np.where(vb, alpha_bottom, np.nan)
        beta_top = np.where(vt, beta_top, np.nan)
        beta_bottom = np.where(vb, beta_bottom, np.nan)
    gamma_top = np.where(vt, gamma_top, np.nan)
    gamma_bottom = np.where(vb, gamma_bottom, np.nan)

    weights = np.stack([vt.astype(float), vb.astype(float)])
    alpha_values = np.stack(
        [np.nan_to_num(alpha_top, nan=0.0), np.nan_to_num(alpha_bottom, nan=0.0)]
    )
    denominator = weights.sum(axis=0)
    alpha = np.divide(
        (alpha_values * weights).sum(axis=0),
        denominator,
        out=np.full(n_frames, np.nan),
        where=denominator > 0,
    )
    return DecomposedMotion(
        alpha=alpha,
        beta_top=beta_top,
        beta_bottom=beta_bottom,
        alpha_top=alpha_top,
        alpha_bottom=alpha_bottom,
        gamma_top=gamma_top,
        gamma_bottom=gamma_bottom,
        valid_top=vt,
        valid_bottom=vb,
    )


def angular_velocity(
    angles: np.ndarray, timestamps_s: np.ndarray
) -> np.ndarray:
    """Finite-difference an angle sequence while preserving missing samples."""

    angles = np.asarray(angles, dtype=float)
    times = np.asarray(timestamps_s, dtype=float)
    if angles.shape != times.shape or angles.ndim != 1:
        raise ValueError("angles and timestamps_s must be one-dimensional and equal length")
    result = np.full_like(angles, np.nan)
    finite = np.isfinite(angles) & np.isfinite(times)
    if finite.sum() >= 2:
        idx = np.flatnonzero(finite)
        if np.any(np.diff(times[idx]) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        result[idx] = np.gradient(angles[idx], times[idx])
    return result


def _rotation_vectors(
    increments: Iterable[np.ndarray | None], *, min_step_deg: float = 0.05
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Return rotation vectors large enough to provide a useful axis estimate."""

    if not np.isfinite(min_step_deg) or min_step_deg <= 0:
        raise ValueError("min_step_deg must be positive and finite")

    vectors: list[np.ndarray] = []
    raw_step_count = 0
    min_step_rad = np.deg2rad(min_step_deg)
    for increment in increments:
        if increment is None:
            continue
        matrix = _as_rotation_matrix(increment)
        vector = Rotation.from_matrix(matrix).as_rotvec()
        raw_step_count += 1
        if np.linalg.norm(vector) >= min_step_rad:
            vectors.append(vector)
    if not vectors:
        raise ValueError(
            "no valid rotations met the axis-calibration motion floor of "
            f"{min_step_deg:g} deg"
        )
    retained_step_count = len(vectors)
    counts: dict[str, float | int] = {
        "raw_step_count": raw_step_count,
        "rejected_low_motion_count": raw_step_count - retained_step_count,
        "retained_step_count": retained_step_count,
        # Retain this established field for callers and report compatibility.
        "sample_count": retained_step_count,
        "min_axis_step_deg": float(min_step_deg),
    }
    return np.asarray(vectors), counts


def estimate_common_axis(
    increments: Iterable[np.ndarray | None],
    *,
    expected_sign: float = 1.0,
    min_step_deg: float = 0.05,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Estimate a shared directed rotation axis from a pure-motion clip.

    The line is found with sign-invariant PCA. ``expected_sign`` then resolves
    the unavoidable axis direction ambiguity; use ``+1`` when the calibration
    clip was recorded in the documented positive direction and ``-1`` if not.
    """

    vectors, counts = _rotation_vectors(increments, min_step_deg=min_step_deg)
    scatter = vectors.T @ vectors
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    axis = eigenvectors[:, np.argmax(eigenvalues)]
    mean_projection = float(np.mean(vectors @ axis))
    if mean_projection * expected_sign < 0:
        axis *= -1
    axis /= np.linalg.norm(axis)
    unsigned = np.abs(vectors @ axis)
    residual = vectors - (vectors @ axis)[:, None] * axis
    angular_spread = np.arctan2(np.linalg.norm(residual, axis=1), unsigned)
    explained = float(eigenvalues[-1] / max(eigenvalues.sum(), np.finfo(float).eps))
    diagnostics: dict[str, float | int] = {
        **counts,
        "axis_spread_median_deg": float(np.rad2deg(np.median(angular_spread))),
        "axis_spread_p95_deg": float(np.rad2deg(np.percentile(angular_spread, 95))),
        "pca_explained_ratio": explained,
    }
    return axis, diagnostics


def calibrate_ball_frame(
    roll_increments: Iterable[np.ndarray | None],
    swivel_increments: Iterable[np.ndarray | None],
    *,
    roll_sign: float = 1.0,
    swivel_sign: float = 1.0,
    min_step_deg: float = 0.05,
) -> tuple[np.ndarray, dict[str, object]]:
    """Estimate ``R_bc`` from labeled pure positive roll and swivel clips."""

    x_raw, roll_diag = estimate_common_axis(
        roll_increments,
        expected_sign=roll_sign,
        min_step_deg=min_step_deg,
    )
    z, swivel_diag = estimate_common_axis(
        swivel_increments,
        expected_sign=swivel_sign,
        min_step_deg=min_step_deg,
    )
    # Gram-Schmidt retains the better observed z direction and forces x normal.
    x = x_raw - z * np.dot(z, x_raw)
    separation = float(np.rad2deg(np.arccos(np.clip(abs(np.dot(x_raw, z)), -1, 1))))
    if np.linalg.norm(x) < 1e-3:
        raise ValueError("roll and swivel axes are nearly parallel; calibration is invalid")
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    z /= np.linalg.norm(z)
    R_bc = np.column_stack([x, y, z])
    diagnostics: dict[str, object] = {
        "roll": roll_diag,
        "swivel": swivel_diag,
        "raw_axis_separation_deg": separation,
        "orthogonality_error": float(np.linalg.norm(R_bc.T @ R_bc - np.eye(3))),
        "determinant": float(np.linalg.det(R_bc)),
    }
    return R_bc, diagnostics
