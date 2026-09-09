"""Rotation primitives and the ball-caster two-axis convention.

All matrices in this project are *active*, right-handed rotations acting on
column vectors.  Thus ``b = R @ a`` (or, for an ``(N, 3)`` row array,
``b = a @ R.T``).  The modeled hemisphere orientation is
``Rx(alpha) @ Rz(beta)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation


FloatArray = NDArray[np.float64]


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def Rx(angle: float) -> FloatArray:
    """Return an active right-handed rotation about the positive x-axis."""

    angle = _finite_scalar(angle, "angle")
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float
    )


def Rz(angle: float) -> FloatArray:
    """Return an active right-handed rotation about the positive z-axis."""

    angle = _finite_scalar(angle, "angle")
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float
    )


def is_rotation_matrix(matrix: ArrayLike, *, atol: float = 1e-7) -> bool:
    """Return whether ``matrix`` is a finite, proper 3-D rotation matrix."""

    candidate = np.asarray(matrix, dtype=float)
    if candidate.shape != (3, 3) or not np.all(np.isfinite(candidate)):
        return False
    return bool(
        np.allclose(candidate.T @ candidate, np.eye(3), atol=atol, rtol=0.0)
        and np.isclose(np.linalg.det(candidate), 1.0, atol=atol, rtol=0.0)
    )


def geodesic_angle(R1: ArrayLike, R2: ArrayLike) -> float | FloatArray:
    """Return the smallest angle between rotations, in radians.

    Inputs may be individual ``(3, 3)`` matrices or broadcast-compatible
    stacks ending in ``(3, 3)``.  Clipping the trace-derived cosine makes the
    calculation safe against round-off at zero and pi.
    """

    first = np.asarray(R1, dtype=float)
    second = np.asarray(R2, dtype=float)
    if first.shape[-2:] != (3, 3) or second.shape[-2:] != (3, 3):
        raise ValueError("R1 and R2 must end in shape (3, 3)")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("rotation matrices must contain only finite values")
    try:
        relative = np.matmul(first, np.swapaxes(second, -1, -2))
    except ValueError as exc:
        raise ValueError("R1 and R2 have incompatible batch shapes") from exc
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return float(angle) if np.ndim(angle) == 0 else angle


def decompose_alpha_beta(R_ball: ArrayLike) -> tuple[float, float, float]:
    """Decompose a ball-frame rotation into ``(alpha, beta, residual)``.

    SciPy's uppercase ``XYZ`` convention is intrinsic and reconstructs as
    ``Rx(alpha) @ Ry(gamma) @ Rz(beta)``.  The returned ``gamma`` is therefore
    the residual outside the caster's modeled ``Rx(alpha) @ Rz(beta)``
    manifold.  Principal Euler values are returned in radians.

    ``R_ball`` must already be expressed in ball coordinates; camera-frame
    rotations must first be conjugated by the fixed ball-to-camera frame.
    """

    matrix = np.asarray(R_ball, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"R_ball must have shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("R_ball must contain only finite values")
    if not is_rotation_matrix(matrix, atol=1e-6):
        raise ValueError("R_ball must be a proper rotation matrix")

    alpha, gamma, beta = Rotation.from_matrix(matrix).as_euler("XYZ")
    return float(alpha), float(beta), float(gamma)


__all__ = [
    "Rx",
    "Rz",
    "decompose_alpha_beta",
    "geodesic_angle",
    "is_rotation_matrix",
]
