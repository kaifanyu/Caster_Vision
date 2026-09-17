"""Metric separated-hemisphere geometry in camera-1 coordinates.

Pixels supplied to this module must be undistorted with their original K (no
rectification/new camera matrix). Camera dictionaries describe X_i = R_i X_1+t_i.
The local x axis is roll, local z is swivel, and F maps home axes into camera 1.
"""
from __future__ import annotations

import numpy as np


def rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def world_points(F, pivot, angles, shell, points_local, radius_m, gap_m,
                 red_sign=1):
    """Transform N unit surface vectors; angles can be (3,) or (N,3).

    shell 0 is red, shell 1 green. Both shell centers roll with the yoke;
    each rotates independently around its own local z axis.
    """
    p = np.atleast_2d(np.asarray(points_local, dtype=float))
    q = np.broadcast_to(np.asarray(angles, dtype=float), (len(p), 3))
    h = np.broadcast_to(np.asarray(shell, dtype=int), (len(p),))
    sign = red_sign * (1 - 2 * h)
    beta = q[np.arange(len(p)), 1 + h]
    cb, sb = np.cos(beta), np.sin(beta)
    x = radius_m * (cb * p[:, 0] - sb * p[:, 1])
    y = radius_m * (sb * p[:, 0] + cb * p[:, 1])
    z = radius_m * p[:, 2] + sign * gap_m / 2
    ca, sa = np.cos(q[:, 0]), np.sin(q[:, 0])
    local = np.column_stack((x, ca * y - sa * z, sa * y + ca * z))
    return local @ np.asarray(F).T + np.asarray(pivot)


def project(camera, points):
    """Project metric camera-1 points into ideal undistorted pixels."""
    xyz = np.atleast_2d(points) @ np.asarray(camera['R']).T + camera['t']
    homogeneous = xyz @ np.asarray(camera['K']).T
    # Negative depths remain negative: never silently mirror behind-camera points.
    depth = homogeneous[:, 2]
    depth = np.where(np.abs(depth) < 1e-9, np.copysign(1e-9, depth), depth)
    return homogeneous[:, :2] / depth[:, None]


def initialize_surface_point(camera, uv, F, pivot, angles, shell,
                             radius_m, gap_m, red_sign=1):
    """Nearest visible signed-hemisphere ray hit, with a limb fallback.

    A fallback seeds optimization when the initial axis/pivot guess places a
    pixel outside the sphere. Its correctness is assessed by final pixel errors.
    """
    R = np.asarray(camera['R'], dtype=float)
    origin = -R.T @ np.asarray(camera['t'], dtype=float)
    direction = R.T @ np.linalg.solve(camera['K'], [*uv, 1.])
    direction /= np.linalg.norm(direction)
    sign = red_sign * (1 - 2 * int(shell))
    B = np.asarray(F) @ rotation_x(float(angles[0]))
    S = B @ rotation_z(float(angles[1 + int(shell)]))
    center = np.asarray(pivot) + B @ np.array([0., 0., sign * gap_m / 2])
    offset = origin - center
    b = offset @ direction
    disc = b * b - (offset @ offset - radius_m * radius_m)
    if disc >= 0 and -b - np.sqrt(disc) > 0:
        distance = -b - np.sqrt(disc)
        local = S.T @ (origin + distance * direction - center) / radius_m
        # An imperfect axis seed can put a true edge feature just across the
        # hemisphere boundary. Clamp there; never seed the hidden far surface.
        local[2] = sign * max(sign * local[2], 1e-7)
        return local / np.linalg.norm(local)
    closest = origin + max(-b, 1e-6) * direction
    local = S.T @ (closest - center)
    if np.linalg.norm(local) < 1e-9:
        local = S.T @ (-direction)
    local[2] = sign * max(sign * local[2], 1e-6)
    return local / np.linalg.norm(local)


def polar_to_points(polar, signs):
    """Hemisphere coordinates (azimuth, colatitude in [0,pi/2])."""
    p = np.asarray(polar)
    return np.column_stack((np.cos(p[:, 0]) * np.sin(p[:, 1]),
                            np.sin(p[:, 0]) * np.sin(p[:, 1]),
                            np.asarray(signs) * np.cos(p[:, 1])))


def points_to_polar(points):
    p = np.atleast_2d(points)
    return np.column_stack((np.arctan2(p[:, 1], p[:, 0]),
                            np.arccos(np.clip(abs(p[:, 2]), 0., 1.))))
