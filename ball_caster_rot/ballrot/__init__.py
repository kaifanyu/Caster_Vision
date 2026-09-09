"""Speckle-based three-DOF ball-caster rotation measurement."""

from .camera import CameraModel, pixel_to_ray, undistort_points
from .rotation import Rx, Rz, decompose_alpha_beta, geodesic_angle
from .sphere import sphere_pose_from_circle, unproject_to_sphere, viewing_angle

__version__ = "0.1.0"

__all__ = [
    "CameraModel",
    "Rx",
    "Rz",
    "decompose_alpha_beta",
    "geodesic_angle",
    "pixel_to_ray",
    "sphere_pose_from_circle",
    "undistort_points",
    "unproject_to_sphere",
    "viewing_angle",
]
