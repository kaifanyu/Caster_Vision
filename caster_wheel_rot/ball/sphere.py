"""Projected-circle fitting and scale-free sphere geometry."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from common.camera import pixel_to_ray, validate_camera_matrix


FloatArray = NDArray[np.float64]


def _positive_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _sphere_parameters(C: ArrayLike, radius: float) -> tuple[FloatArray, float]:
    center = np.asarray(C, dtype=float)
    if center.shape != (3,):
        raise ValueError(f"C must have shape (3,), got {center.shape}")
    if not np.all(np.isfinite(center)):
        raise ValueError("C must contain only finite values")
    return center, _positive_scalar(radius, "radius")


def fit_circle_points(points: ArrayLike) -> tuple[float, float, float]:
    """Least-squares fit ``(u0, v0, radius_px)`` to 2-D boundary points."""

    samples = np.asarray(points, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {samples.shape}")
    if len(samples) < 3:
        raise ValueError("at least three boundary points are required")
    if not np.all(np.isfinite(samples)):
        raise ValueError("points must contain only finite values")

    # u^2 + v^2 = 2*u0*u + 2*v0*v + (r^2-u0^2-v0^2).
    design = np.column_stack(
        [2.0 * samples[:, 0], 2.0 * samples[:, 1], np.ones(len(samples))]
    )
    rhs = np.einsum("ij,ij->i", samples, samples)
    solution, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
    if rank < 3:
        raise ValueError("boundary points are collinear; a circle is undefined")
    u0, v0, offset = solution
    radius_squared = offset + u0 * u0 + v0 * v0
    if not np.isfinite(radius_squared) or radius_squared <= 0.0:
        raise ValueError("circle fit produced a non-positive radius")
    return float(u0), float(v0), float(np.sqrt(radius_squared))


def fit_circle(
    mask: ArrayLike, *, method: str = "least_squares"
) -> tuple[float, float, float]:
    """Fit the largest connected silhouette in a binary mask.

    Parameters
    ----------
    mask:
        A two-dimensional array whose nonzero pixels denote the ball.
    method:
        ``"least_squares"`` fits all points of the largest external contour;
        ``"min_enclosing"`` uses OpenCV's enclosing circle.  The former is
        less sensitive to isolated contour extremities; the latter guarantees
        that the selected contour is enclosed.
    """

    binary = np.asarray(mask)
    if binary.ndim != 2 or binary.shape[0] == 0 or binary.shape[1] == 0:
        raise ValueError("mask must be a non-empty two-dimensional array")
    if method not in {"least_squares", "min_enclosing"}:
        raise ValueError("method must be 'least_squares' or 'min_enclosing'")
    image = np.where(binary != 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("mask contains no foreground silhouette")
    contour = max(contours, key=cv2.contourArea)
    points = contour.reshape(-1, 2).astype(float)
    if len(points) < 3:
        raise ValueError("largest foreground component has too few boundary points")
    if method == "min_enclosing":
        (u0, v0), radius = cv2.minEnclosingCircle(points.astype(np.float32))
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("circle fit produced a non-positive radius")
        return float(u0), float(v0), float(radius)
    return fit_circle_points(points)


# An explicit name is convenient at call sites that also fit axes or ellipses.
fit_circle_from_mask = fit_circle


def sphere_pose_from_circle(
    u0: float, v0: float, r_px: float, K: ArrayLike
) -> tuple[FloatArray, float]:
    """Infer a scale-free camera-frame sphere from its projected circle.

    The physical radius is set to one because only surface directions enter
    Kabsch.  This follows the spec's near-axis/paraxial silhouette relation;
    strongly off-axis or wide-angle observations require a cone refinement.
    """

    u0 = float(u0)
    v0 = float(v0)
    if not np.isfinite(u0) or not np.isfinite(v0):
        raise ValueError("circle center coordinates must be finite")
    radius_px = _positive_scalar(r_px, "r_px")
    matrix = validate_camera_matrix(K)
    focal = 0.5 * (matrix[0, 0] + matrix[1, 1])
    angular_radius = np.arctan(radius_px / focal)
    distance = 1.0 / np.sin(angular_radius)
    center_ray = pixel_to_ray(np.array([u0, v0], dtype=float), matrix)
    return center_ray * distance, 1.0


def unproject_to_sphere(
    uv: ArrayLike, K: ArrayLike, C: ArrayLike, radius: float = 1.0
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Intersect camera rays with a sphere and return outward directions.

    ``uv`` may have any leading shape ending in two.  The returned directions
    have the corresponding leading shape ending in three, and ``valid`` has
    the leading shape.  The nearer positive intersection is selected.  Rays
    that miss, merely graze, or intersect only behind the camera are invalid;
    their direction rows are ``NaN`` to prevent accidental use by Kabsch.
    """

    center, radius = _sphere_parameters(C, radius)
    rays = pixel_to_ray(uv, K)
    leading_shape = rays.shape[:-1]
    flat_rays = rays.reshape(-1, 3)
    if len(flat_rays) == 0:
        return (
            np.empty(leading_shape + (3,), dtype=float),
            np.empty(leading_shape, dtype=bool),
        )

    projection = flat_rays @ center
    constant = float(center @ center - radius * radius)
    discriminant = projection * projection - constant
    root = np.sqrt(np.clip(discriminant, 0.0, None))
    near = projection - root
    far = projection + root
    # If the camera is inside the sphere, the near root is behind it and the
    # far root is the first physically observable intersection.
    epsilon = np.finfo(float).eps * max(1.0, float(np.linalg.norm(center)))
    distance = np.where(near > epsilon, near, far)
    valid_flat = (discriminant > 0.0) & (distance > epsilon)

    directions = np.full((len(flat_rays), 3), np.nan, dtype=float)
    if np.any(valid_flat):
        points = flat_rays[valid_flat] * distance[valid_flat, None]
        offsets = points - center
        norms = np.linalg.norm(offsets, axis=1)
        numerically_valid = norms > np.finfo(float).eps
        valid_indices = np.flatnonzero(valid_flat)
        if np.any(numerically_valid):
            directions[valid_indices[numerically_valid]] = (
                offsets[numerically_valid] / norms[numerically_valid, None]
            )
        valid_flat[valid_indices[~numerically_valid]] = False

    return directions.reshape(leading_shape + (3,)), valid_flat.reshape(leading_shape)


def viewing_angle(
    dirs: ArrayLike, C: ArrayLike, radius: float = 1.0
) -> FloatArray:
    """Return each surface normal's angle from the direction to the camera.

    Zero is the center of the visible cap and pi/2 is the silhouette limb.
    This is the angle used by ``limb_cull_deg``.  Nonfinite or zero-length
    direction rows produce ``NaN`` rather than contaminating neighboring rows.
    """

    center, radius = _sphere_parameters(C, radius)
    normals = np.asarray(dirs, dtype=float)
    if normals.ndim == 0 or normals.shape[-1:] != (3,):
        raise ValueError(f"dirs must end in shape (3,), got {normals.shape}")
    leading_shape = normals.shape[:-1]
    flat = normals.reshape(-1, 3)
    angles = np.full(len(flat), np.nan, dtype=float)
    if len(flat) == 0:
        return angles.reshape(leading_shape)

    normal_norm = np.linalg.norm(flat, axis=1)
    finite = np.all(np.isfinite(flat), axis=1)
    usable = finite & (normal_norm > np.finfo(float).eps)
    if np.any(usable):
        unit_normal = flat[usable] / normal_norm[usable, None]
        surface_points = center + radius * unit_normal
        toward_camera = -surface_points
        view_norm = np.linalg.norm(toward_camera, axis=1)
        view_valid = view_norm > np.finfo(float).eps
        dots = np.full(len(unit_normal), np.nan, dtype=float)
        if np.any(view_valid):
            dots[view_valid] = np.einsum(
                "ij,ij->i",
                unit_normal[view_valid],
                toward_camera[view_valid] / view_norm[view_valid, None],
            )
        selected = np.flatnonzero(usable)
        angles[selected] = np.arccos(np.clip(dots, -1.0, 1.0))
    return angles.reshape(leading_shape)


__all__ = [
    "fit_circle",
    "fit_circle_from_mask",
    "fit_circle_points",
    "sphere_pose_from_circle",
    "unproject_to_sphere",
    "viewing_angle",
]
