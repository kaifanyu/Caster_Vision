"""Pinhole-camera helpers used by tracking and sphere unprojection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
_VALID_DISTORTION_LENGTHS = {4, 5, 8, 12, 14}


def validate_camera_matrix(K: ArrayLike) -> FloatArray:
    """Validate and return a float copy of an OpenCV camera matrix."""

    matrix = np.asarray(K, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"K must have shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("K must contain only finite values")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-12, rtol=0.0):
        raise ValueError("K must have the standard pinhole final row [0, 0, 1]")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError("K focal lengths fx and fy must be positive")
    if abs(np.linalg.det(matrix)) <= np.finfo(float).eps:
        raise ValueError("K must be invertible")
    return matrix.copy()


def _distortion_array(dist: ArrayLike | None) -> FloatArray:
    if dist is None:
        return np.zeros(5, dtype=float)
    coefficients = np.asarray(dist, dtype=float).reshape(-1)
    if coefficients.size not in _VALID_DISTORTION_LENGTHS:
        allowed = ", ".join(str(value) for value in sorted(_VALID_DISTORTION_LENGTHS))
        raise ValueError(f"dist must contain one of {allowed} coefficients")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("dist must contain only finite values")
    return coefficients.copy()


def _point_array(uv: ArrayLike) -> FloatArray:
    points = np.asarray(uv, dtype=float)
    if points.ndim == 0 or points.shape[-1:] != (2,):
        raise ValueError(f"uv must end in shape (2,), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("uv must contain only finite values")
    return points


def pixel_to_ray(uv: ArrayLike, K: ArrayLike) -> FloatArray:
    """Convert undistorted pixels to unit rays in camera coordinates.

    The camera is at the origin and looks along positive z.  The leading shape
    of ``uv`` is preserved: an ``(N, 2)`` array yields ``(N, 3)`` and one
    ``(2,)`` pixel yields one ``(3,)`` ray.  Distorted observations must first
    be passed through :func:`undistort_points`.
    """

    points = _point_array(uv)
    matrix = validate_camera_matrix(K)
    flat = points.reshape(-1, 2)
    if len(flat) == 0:
        return np.empty(points.shape[:-1] + (3,), dtype=float)
    homogeneous = np.column_stack([flat, np.ones(len(flat), dtype=float)])
    directions = homogeneous @ np.linalg.inv(matrix).T
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("K maps at least one pixel to a degenerate ray")
    directions /= norms
    return directions.reshape(points.shape[:-1] + (3,))


def undistort_points(
    uv: ArrayLike, K: ArrayLike, dist: ArrayLike | None = None
) -> FloatArray:
    """Undistort pixel coordinates while keeping them in pixel units.

    ``dist`` follows OpenCV's standard perspective distortion model.  Passing
    ``None`` or all-zero coefficients is an exact no-op (apart from returning
    a safe float copy).
    """

    points = _point_array(uv)
    matrix = validate_camera_matrix(K)
    coefficients = _distortion_array(dist)
    if points.size == 0 or not np.any(coefficients):
        return points.copy()
    flat = points.reshape(-1, 1, 2)
    corrected = cv2.undistortPoints(flat, matrix, coefficients, P=matrix)
    return corrected.reshape(points.shape)


def undistort_image(
    image: NDArray[np.generic], K: ArrayLike, dist: ArrayLike | None = None
) -> NDArray[np.generic]:
    """Return an image corrected with the same OpenCV camera model."""

    frame = np.asarray(image)
    if frame.ndim not in (2, 3) or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("image must be a non-empty grayscale or color array")
    matrix = validate_camera_matrix(K)
    coefficients = _distortion_array(dist)
    if not np.any(coefficients):
        return frame.copy()
    return cv2.undistort(frame, matrix, coefficients)


def camera_matrix_from_fov(
    image_shape: tuple[int, ...], fov_deg: float
) -> FloatArray:
    """Approximate ``K`` from image shape and horizontal field of view.

    This fallback assumes square pixels and a centered principal point.  It is
    intentionally explicit so callers can warn that calibrated intrinsics are
    preferable for measurement work.
    """

    if len(image_shape) < 2:
        raise ValueError("image_shape must provide at least (height, width)")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    fov = float(fov_deg)
    if not np.isfinite(fov) or not 0.0 < fov < 180.0:
        raise ValueError("fov_deg must be finite and strictly between 0 and 180")
    focal = 0.5 * width / np.tan(0.5 * np.deg2rad(fov))
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


@dataclass(frozen=True)
class CameraModel:
    """A validated camera matrix and OpenCV distortion coefficients."""

    K: FloatArray
    dist: FloatArray

    def __init__(self, K: ArrayLike, dist: ArrayLike | None = None) -> None:
        object.__setattr__(self, "K", validate_camera_matrix(K))
        object.__setattr__(self, "dist", _distortion_array(dist))

    def undistort_points(self, uv: ArrayLike) -> FloatArray:
        return undistort_points(uv, self.K, self.dist)

    def pixel_to_ray(self, uv: ArrayLike) -> FloatArray:
        return pixel_to_ray(uv, self.K)

    def undistort_image(
        self, image: NDArray[np.generic]
    ) -> NDArray[np.generic]:
        return undistort_image(image, self.K, self.dist)


__all__ = [
    "CameraModel",
    "camera_matrix_from_fov",
    "pixel_to_ray",
    "undistort_image",
    "undistort_points",
    "validate_camera_matrix",
]
