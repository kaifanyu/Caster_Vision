"""Calibrated swivel-caster geometry and ray--plane lifting.

The transform convention is deliberately explicit: ``R_camera_from_car`` and
``t_camera_from_car`` satisfy

``point_camera = R_camera_from_car @ point_car + t_camera_from_car``.

The wheel hub is not fixed in the car frame when the caster has trail.  Its
zero-swivel offset is rotated about ``swivel_axis_car`` for every value of
``psi``.  This detail is essential when roll and swivel occur together.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from common.camera import pixel_to_ray, undistort_points, validate_camera_matrix
from common.rotation import Rz, is_rotation_matrix


FloatArray = NDArray[np.float64]


def _vector3(value: ArrayLike, name: str, *, unit: bool = False) -> FloatArray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    if unit:
        norm = float(np.linalg.norm(vector))
        if norm <= np.finfo(float).eps:
            raise ValueError(f"{name} must be non-zero")
        vector = vector / norm
    return vector.copy()


def _points3(value: ArrayLike, name: str) -> tuple[FloatArray, tuple[int, ...]]:
    points = np.asarray(value, dtype=float)
    if points.ndim == 0 or points.shape[-1:] != (3,):
        raise ValueError(f"{name} must end in shape (3,), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    return points.reshape(-1, 3), points.shape


def _face_sign(value: int) -> int:
    sign = int(value)
    if sign not in (-1, 1):
        raise ValueError("face_sign must be +1 or -1")
    return sign


@dataclass(frozen=True)
class SwivelGeometry:
    """Rigid camera calibration plus the caster's zero-swivel geometry.

    ``hub_offset_zero_car`` points from the swivel-axis reference point to the
    wheel hub at ``psi == 0``.  Its horizontal component therefore contains
    the trail.  ``axle_zero_car`` is the signed roll axis at the same pose.
    A positive face is displaced by ``+wheel_width/2 * axle``.
    """

    R_camera_from_car: FloatArray
    t_camera_from_car: FloatArray
    swivel_axis_car: FloatArray
    hub_offset_zero_car: FloatArray
    axle_zero_car: FloatArray
    wheel_radius: float
    wheel_width: float = 0.0

    @classmethod
    def from_config(
        cls,
        camera: Mapping[str, Any],
        geometry: Mapping[str, Any],
    ) -> "SwivelGeometry":
        """Build from the repository's ``camera`` and ``swivel.geometry`` maps.

        Configuration stores ``T_car_from_cam`` because that direction is
        convenient when emitting car-frame measurements.  Internally this
        class uses its exact inverse, ``T_camera_from_car``.
        """

        transform_value = camera.get("T_car_from_cam")
        if transform_value is None:
            raise ValueError("camera.T_car_from_cam must be calibrated")
        transform = np.asarray(transform_value, dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("camera.T_car_from_cam must be a finite 4x4 matrix")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError("camera.T_car_from_cam must end in [0, 0, 0, 1]")
        R_car_from_camera = transform[:3, :3]
        if not is_rotation_matrix(R_car_from_camera):
            raise ValueError("camera.T_car_from_cam rotation is not proper")
        t_car_from_camera = transform[:3, 3]
        R_camera_from_car = R_car_from_camera.T
        t_camera_from_car = -R_camera_from_car @ t_car_from_camera
        try:
            swivel_axis = geometry["swivel_axis_car"]
            hub_offset = geometry["hub_offset0_car"]
            axle = geometry["axle0_car"]
            radius = geometry["wheel_radius_m"]
        except KeyError as exc:
            raise ValueError(f"missing swivel.geometry value: {exc.args[0]}") from exc
        return cls(
            R_camera_from_car=R_camera_from_car,
            t_camera_from_car=t_camera_from_car,
            swivel_axis_car=swivel_axis,
            hub_offset_zero_car=hub_offset,
            axle_zero_car=axle,
            wheel_radius=radius,
            wheel_width=float(geometry.get("wheel_width_m", 0.0)),
        )

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "SwivelGeometry":
        """Build directly from the root configuration mapping."""

        try:
            camera = config["camera"]
            geometry = config["swivel"]["geometry"]
        except (KeyError, TypeError) as exc:
            raise ValueError("config must contain camera and swivel.geometry mappings") from exc
        if not isinstance(camera, Mapping) or not isinstance(geometry, Mapping):
            raise ValueError("camera and swivel.geometry must be mappings")
        return cls.from_config(camera, geometry)

    def __post_init__(self) -> None:
        rotation = np.asarray(self.R_camera_from_car, dtype=float)
        if not is_rotation_matrix(rotation):
            raise ValueError("R_camera_from_car must be a proper 3x3 rotation")
        translation = _vector3(self.t_camera_from_car, "t_camera_from_car")
        swivel_axis = _vector3(self.swivel_axis_car, "swivel_axis_car")
        hub_offset = _vector3(self.hub_offset_zero_car, "hub_offset_zero_car")
        axle = _vector3(self.axle_zero_car, "axle_zero_car", unit=True)
        radius = float(self.wheel_radius)
        width = float(self.wheel_width)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("wheel_radius must be positive and finite")
        if not np.isfinite(width) or width < 0.0:
            raise ValueError("wheel_width must be non-negative and finite")
        # A near-vertical axle would not define a ground-plane rolling heading.
        if np.linalg.norm(np.cross([0.0, 0.0, 1.0], axle)) < 1e-6:
            raise ValueError("axle_zero_car must not be parallel to car +z")
        object.__setattr__(self, "R_camera_from_car", rotation.copy())
        object.__setattr__(self, "t_camera_from_car", translation)
        object.__setattr__(self, "swivel_axis_car", swivel_axis)
        object.__setattr__(self, "hub_offset_zero_car", hub_offset)
        object.__setattr__(self, "axle_zero_car", axle)
        object.__setattr__(self, "wheel_radius", radius)
        object.__setattr__(self, "wheel_width", width)

    @property
    def R_car_from_camera(self) -> FloatArray:
        """Rotation taking camera-frame vectors into the car frame."""

        return self.R_camera_from_car.T

    @property
    def T_camera_from_car(self) -> FloatArray:
        """Homogeneous point transform used directly by the renderer."""

        transform = np.eye(4, dtype=float)
        transform[:3, :3] = self.R_camera_from_car
        transform[:3, 3] = self.t_camera_from_car
        return transform

    @property
    def T_car_from_camera(self) -> FloatArray:
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = self.R_car_from_camera
        transform[:3, 3] = -self.R_car_from_camera @ self.t_camera_from_car
        return transform

    @property
    def heading_zero_car(self) -> FloatArray:
        """Ground-plane rolling direction implied by the signed zero axle."""

        heading = np.cross([0.0, 0.0, 1.0], self.axle_zero_car)
        heading[2] = 0.0
        return heading / np.linalg.norm(heading)

    @property
    def trail(self) -> float:
        """Signed horizontal trail along the zero rolling heading.

        A conventional trailing wheel has a positive value because its hub is
        behind the swivel axis: ``hub_offset_xy ~= -trail * heading_zero``.
        """

        return float(-self.hub_offset_zero_car[:2] @ self.heading_zero_car[:2])

    def R_car_from_fork(self, psi: float) -> FloatArray:
        """Return the zero-fork-frame to car-frame rotation at ``psi``."""

        return Rz(float(psi))

    def R_camera_from_fork(self, psi: float) -> FloatArray:
        """Return the zero-fork-frame to camera-frame rotation at ``psi``."""

        return self.R_camera_from_car @ self.R_car_from_fork(psi)

    def hub_center_car(self, psi: float) -> FloatArray:
        """Dynamic hub center, including the caster's trailing offset."""

        return self.swivel_axis_car + Rz(float(psi)) @ self.hub_offset_zero_car

    # Concise aliases used by the pipeline and synthetic renderer.
    def hub_car(self, psi: float) -> FloatArray:
        return self.hub_center_car(psi)

    def hub_center_camera(self, psi: float) -> FloatArray:
        return self.car_points_to_camera(self.hub_center_car(psi))

    def axle_car(self, psi: float) -> FloatArray:
        axle = Rz(float(psi)) @ self.axle_zero_car
        return axle / np.linalg.norm(axle)

    def axle_camera(self, psi: float) -> FloatArray:
        axle = self.car_vectors_to_camera(self.axle_car(psi))
        return axle / np.linalg.norm(axle)

    def face_center_car(self, psi: float, face_sign: int = 1) -> FloatArray:
        sign = _face_sign(face_sign)
        return self.hub_center_car(psi) + sign * 0.5 * self.wheel_width * self.axle_car(psi)

    def face_plane_camera(
        self, psi: float, face_sign: int = 1
    ) -> tuple[FloatArray, FloatArray]:
        """Return a point and outward unit normal for one wheel-side plane."""

        sign = _face_sign(face_sign)
        center = self.car_points_to_camera(self.face_center_car(psi, sign))
        normal = sign * self.axle_camera(psi)
        return center, normal

    def car_points_to_camera(self, points_car: ArrayLike) -> FloatArray:
        points, original_shape = _points3(points_car, "points_car")
        transformed = points @ self.R_camera_from_car.T + self.t_camera_from_car
        return transformed.reshape(original_shape)

    def camera_points_to_car(self, points_camera: ArrayLike) -> FloatArray:
        points, original_shape = _points3(points_camera, "points_camera")
        transformed = (points - self.t_camera_from_car) @ self.R_camera_from_car
        return transformed.reshape(original_shape)

    def car_vectors_to_camera(self, vectors_car: ArrayLike) -> FloatArray:
        vectors, original_shape = _points3(vectors_car, "vectors_car")
        return (vectors @ self.R_camera_from_car.T).reshape(original_shape)

    def camera_vectors_to_car(self, vectors_camera: ArrayLike) -> FloatArray:
        vectors, original_shape = _points3(vectors_camera, "vectors_camera")
        return (vectors @ self.R_camera_from_car).reshape(original_shape)

    def camera_vectors_to_fork(self, vectors_camera: ArrayLike, psi: float) -> FloatArray:
        """Undo camera extrinsics and swivel for free vectors."""

        vectors, original_shape = _points3(vectors_camera, "vectors_camera")
        # Row-vector inverse of v_camera = R_camera_from_fork @ v_fork.
        transformed = vectors @ self.R_camera_from_fork(psi)
        return transformed.reshape(original_shape)

    def radial_basis_zero(self) -> tuple[FloatArray, FloatArray]:
        """Return an oriented orthonormal basis of the zero wheel plane."""

        axis = self.axle_zero_car
        first = np.array([0.0, 0.0, 1.0]) - axis[2] * axis
        if np.linalg.norm(first) < 1e-8:
            seed = np.array([1.0, 0.0, 0.0])
            first = seed - (seed @ axis) * axis
        first /= np.linalg.norm(first)
        second = np.cross(axis, first)
        second /= np.linalg.norm(second)
        return first, second

    def sidewall_boundary_car(
        self,
        psi: float,
        face_sign: int = 1,
        *,
        radius: float | None = None,
        samples: int = 96,
    ) -> FloatArray:
        """Sample an ordered circular boundary on a wheel side face."""

        if int(samples) < 12:
            raise ValueError("samples must be at least 12")
        resolved_radius = self.wheel_radius if radius is None else float(radius)
        if not np.isfinite(resolved_radius) or resolved_radius <= 0.0:
            raise ValueError("radius must be positive and finite")
        first, second = self.radial_basis_zero()
        theta = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=False)
        radial_zero = resolved_radius * (
            np.cos(theta)[:, None] * first + np.sin(theta)[:, None] * second
        )
        radial_car = radial_zero @ Rz(float(psi)).T
        return self.face_center_car(psi, face_sign) + radial_car

    def sidewall_view_confidence(self, psi: float, face_sign: int | None = None) -> float:
        """Cosine-like face-on score in ``[0, 1]``.

        For a specified face the score is zero when that outward face points
        away from the camera.  With ``face_sign=None`` the better of the two
        speckled sidewalls is used.
        """

        if face_sign is None:
            return max(
                self.sidewall_view_confidence(psi, 1),
                self.sidewall_view_confidence(psi, -1),
            )
        center, outward = self.face_plane_camera(psi, _face_sign(face_sign))
        distance = float(np.linalg.norm(center))
        if distance <= np.finfo(float).eps:
            return 0.0
        return float(np.clip(outward @ (-center / distance), 0.0, 1.0))

    def visible_face_sign(self, psi: float) -> int:
        """Return the sidewall whose outward normal is most camera-facing."""

        plus = self.sidewall_view_confidence(psi, 1)
        minus = self.sidewall_view_confidence(psi, -1)
        return 1 if plus >= minus else -1

    def sidewall_mask(
        self,
        psi: float,
        K: ArrayLike,
        image_shape: tuple[int, ...],
        **kwargs: object,
    ) -> NDArray[np.bool_]:
        """Rasterize the projected wheel-face mask for pipeline KLT."""

        return projected_sidewall_mask(self, psi, K, image_shape, **kwargs)


def unproject_to_plane(
    uv: ArrayLike,
    K: ArrayLike,
    Q: ArrayLike,
    m: ArrayLike,
    *,
    dist: ArrayLike | None = None,
    parallel_epsilon: float = 1e-8,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Intersect camera rays with a plane.

    ``Q`` is any point on the plane and ``m`` its normal, both in camera
    coordinates.  Invalid parallel rays and intersections behind the camera
    are returned as ``NaN`` rows with a false validity flag.
    """

    points = np.asarray(uv, dtype=float)
    if points.ndim == 0 or points.shape[-1:] != (2,):
        raise ValueError(f"uv must end in shape (2,), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("uv must contain only finite values")
    matrix = validate_camera_matrix(K)
    plane_point = _vector3(Q, "Q")
    normal = _vector3(m, "m", unit=True)
    epsilon = float(parallel_epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("parallel_epsilon must be positive and finite")
    pixels = undistort_points(points, matrix, dist) if dist is not None else points
    rays = pixel_to_ray(pixels, matrix)
    flat_rays = rays.reshape(-1, 3)
    denominator = flat_rays @ normal
    numerator = float(plane_point @ normal)
    valid = np.isfinite(denominator) & (np.abs(denominator) > epsilon)
    parameter = np.full(len(flat_rays), np.nan, dtype=float)
    parameter[valid] = numerator / denominator[valid]
    valid &= np.isfinite(parameter) & (parameter > 0.0)
    intersections = np.full_like(flat_rays, np.nan, dtype=float)
    intersections[valid] = flat_rays[valid] * parameter[valid, None]
    output_shape = points.shape[:-1]
    return intersections.reshape(output_shape + (3,)), valid.reshape(output_shape)


def project_camera_points(
    points_camera: ArrayLike, K: ArrayLike
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Project camera-frame points through an undistorted pinhole model."""

    points, original_shape = _points3(points_camera, "points_camera")
    matrix = validate_camera_matrix(K)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.0)
    pixels = np.full((len(points), 2), np.nan, dtype=float)
    homogeneous = points[valid] @ matrix.T
    pixels[valid] = homogeneous[:, :2] / homogeneous[:, 2:3]
    leading = original_shape[:-1]
    return pixels.reshape(leading + (2,)), valid.reshape(leading)


@dataclass(frozen=True)
class SidewallProjection:
    """Projected sidewall mask plus the geometry used to construct it."""

    mask: NDArray[np.bool_]
    boundary_uv: FloatArray
    face_sign: int
    view_confidence: float
    center_camera: FloatArray
    normal_camera: FloatArray


def project_sidewall(
    geometry: SwivelGeometry,
    psi: float,
    K: ArrayLike,
    image_shape: tuple[int, ...],
    *,
    face_sign: int | None = None,
    inner_radius_fraction: float = 0.0,
    margin_px: int = 0,
    samples: int = 96,
) -> SidewallProjection:
    """Project one wheel face and rasterize its visible annular mask."""

    if len(image_shape) < 2:
        raise ValueError("image_shape must contain at least height and width")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    sign = geometry.visible_face_sign(psi) if face_sign is None else _face_sign(face_sign)
    inner = float(inner_radius_fraction)
    if not np.isfinite(inner) or not 0.0 <= inner < 1.0:
        raise ValueError("inner_radius_fraction must lie in [0, 1)")
    if int(margin_px) < 0:
        raise ValueError("margin_px must be non-negative")

    boundary_car = geometry.sidewall_boundary_car(psi, sign, samples=samples)
    boundary_camera = geometry.car_points_to_camera(boundary_car)
    boundary_uv, valid = project_camera_points(boundary_camera, K)
    mask_u8 = np.zeros((height, width), dtype=np.uint8)
    if np.count_nonzero(valid) >= 3:
        polygon = np.rint(boundary_uv[valid]).astype(np.int32)
        cv2.fillConvexPoly(mask_u8, polygon, 255, lineType=cv2.LINE_8)

    if inner > 0.0:
        inner_car = geometry.sidewall_boundary_car(
            psi,
            sign,
            radius=inner * geometry.wheel_radius,
            samples=samples,
        )
        inner_camera = geometry.car_points_to_camera(inner_car)
        inner_uv, inner_valid = project_camera_points(inner_camera, K)
        if np.count_nonzero(inner_valid) >= 3:
            cv2.fillConvexPoly(
                mask_u8,
                np.rint(inner_uv[inner_valid]).astype(np.int32),
                0,
                lineType=cv2.LINE_8,
            )
    if margin_px:
        size = 2 * int(margin_px) + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        mask_u8 = cv2.erode(mask_u8, kernel, iterations=1)

    center, normal = geometry.face_plane_camera(psi, sign)
    return SidewallProjection(
        mask=mask_u8.astype(bool),
        boundary_uv=np.asarray(boundary_uv, dtype=float),
        face_sign=sign,
        view_confidence=geometry.sidewall_view_confidence(psi, sign),
        center_camera=center,
        normal_camera=normal,
    )


def projected_sidewall_mask(
    geometry: SwivelGeometry,
    psi: float,
    K: ArrayLike,
    image_shape: tuple[int, ...],
    **kwargs: object,
) -> NDArray[np.bool_]:
    """Convenience wrapper returning only :func:`project_sidewall`'s mask."""

    return project_sidewall(geometry, psi, K, image_shape, **kwargs).mask


__all__ = [
    "SidewallProjection",
    "SwivelGeometry",
    "project_camera_points",
    "project_sidewall",
    "projected_sidewall_mask",
    "unproject_to_plane",
]
