"""Shared projection geometry for fitting and rendering the two caster shells.

``C`` is the assembly pivot. Radius is the curvature radius of either shell.
For separated hemispheres, each sphere center is displaced by half the rim gap
along the moving spin axis; spinning a shell does not move its center.
"""
from __future__ import annotations

import numpy as np

GEOMETRIES = ("common_sphere_caps", "separated_hemispheres")


def cap_boundary(gap_fraction, geometry):
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown shell geometry: {geometry}")
    return float(gap_fraction) if geometry == "common_sphere_caps" else 0.0


def shell_centers(orientations, C, radius, sign, gap_fraction, geometry):
    cap_boundary(gap_fraction, geometry)
    orientations = np.asarray(orientations, dtype=float)
    offset = float(radius) * float(gap_fraction) if geometry == "separated_hemispheres" else 0.0
    return np.asarray(C, dtype=float) + (np.asarray(sign) * offset)[..., None] * orientations[..., :, 2]


def surface_camera(points, orientations, C, radius, sign, gap_fraction, geometry):
    normals = np.einsum("...ij,...j->...i", orientations, points)
    return shell_centers(orientations, C, radius, sign, gap_fraction, geometry) + radius * normals


def unproject_shell(uv, K, C, radius, orientations, sign, gap_fraction, geometry):
    """Intersect each pixel ray with its pose-dependent shell sphere.

    Returns outward camera normals, valid intersection flags, and view angles.
    The cap restriction is imposed in landmark coordinates by the estimator.
    """
    pixels = np.asarray(uv, dtype=float)
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(K).T
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    centers = np.broadcast_to(shell_centers(orientations, C, radius, sign, gap_fraction, geometry), rays.shape)
    along = np.einsum("ij,ij->i", rays, centers)
    discriminant = along**2 - (np.sum(centers**2, axis=1) - radius**2)
    distances = along - np.sqrt(np.maximum(discriminant, 0.0))
    valid = (discriminant > 0) & (distances > 0)
    normals = (rays * distances[:, None] - centers) / radius
    normals[~valid] = np.nan
    angles = np.arccos(np.clip(-np.einsum("ij,ij->i", normals, rays), -1, 1))
    return normals, valid, angles
