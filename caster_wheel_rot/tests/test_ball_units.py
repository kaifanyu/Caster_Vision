"""Stage A: numeric convention and geometry gates (no rendered images)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from common.camera import (
    camera_matrix_from_fov,
    pixel_to_ray,
    undistort_points,
)
from common.estimate import kabsch
from common.rotation import Rx, Rz, decompose_alpha_beta, geodesic_angle
from ball.sphere import (
    fit_circle_points,
    sphere_pose_from_circle,
    unproject_to_sphere,
    viewing_angle,
)


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _wrapped_error(actual: float, expected: float) -> float:
    return float((actual - expected + np.pi) % (2.0 * np.pi) - np.pi)


def test_right_handed_active_axis_conventions() -> None:
    np.testing.assert_allclose(Rx(np.pi / 2.0) @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(Rz(np.pi / 2.0) @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-15)


def test_geodesic_angle_is_the_smallest_relative_angle() -> None:
    assert geodesic_angle(Rx(0.25), Rx(-0.15)) == pytest.approx(0.4, abs=1e-14)
    assert geodesic_angle(np.eye(3), Rz(np.pi)) == pytest.approx(np.pi, abs=1e-14)
    batch = geodesic_angle(np.stack([np.eye(3), np.eye(3)]), np.stack([Rx(0.1), Rz(-0.2)]))
    np.testing.assert_allclose(batch, [0.1, 0.2], atol=1e-14)


def test_decompose_alpha_beta_exact_model_randomized() -> None:
    """Guard the required R = Rx(alpha) @ Rz(beta) multiply order."""

    rng = np.random.default_rng(20260829)
    for alpha, beta in rng.uniform(-np.pi + 0.01, np.pi - 0.01, size=(250, 2)):
        recovered_alpha, recovered_beta, gamma = decompose_alpha_beta(Rx(alpha) @ Rz(beta))
        assert abs(_wrapped_error(recovered_alpha, alpha)) < 1e-9
        assert abs(_wrapped_error(recovered_beta, beta)) < 1e-9
        assert abs(gamma) < 1e-9


def test_pixel_to_ray_uses_positive_camera_z_and_honors_skew() -> None:
    K = np.array([[800.0, 7.0, 320.0], [0.0, 810.0, 240.0], [0.0, 0.0, 1.0]])
    ray = pixel_to_ray([320.0, 240.0], K)
    np.testing.assert_allclose(ray, [0.0, 0.0, 1.0], atol=1e-15)

    point = np.array([0.2, -0.1, 2.0])
    projected = K @ point
    projected = projected[:2] / projected[2]
    expected = point / np.linalg.norm(point)
    np.testing.assert_allclose(pixel_to_ray(projected, K), expected, atol=1e-15)


def test_undistort_points_returns_pixels_and_inverts_opencv_distortion() -> None:
    K = np.array([[900.0, 0.0, 320.0], [0.0, 910.0, 240.0], [0.0, 0.0, 1.0]])
    dist = np.array([0.08, -0.03, 0.001, -0.002, 0.005])
    object_points = np.array(
        [[-0.25, -0.15, 1.0], [0.20, -0.10, 1.0], [0.15, 0.22, 1.0]],
        dtype=float,
    )
    distorted, _ = cv2.projectPoints(
        object_points, np.zeros(3), np.zeros(3), K, dist
    )
    ideal = object_points @ K.T
    ideal = ideal[:, :2] / ideal[:, 2:]
    corrected = undistort_points(distorted.reshape(-1, 2), K, dist)
    np.testing.assert_allclose(corrected, ideal, atol=1e-7)


def test_camera_matrix_fov_fallback_has_expected_geometry() -> None:
    K = camera_matrix_from_fov((720, 1280), 90.0)
    np.testing.assert_allclose(K, [[640.0, 0.0, 640.0], [0.0, 640.0, 360.0], [0.0, 0.0, 1.0]], atol=1e-12)


def test_unproject_to_sphere_round_trip_below_one_nanometer_equivalent() -> None:
    """Stage-A hard gate: project known front-cap directions, then recover."""

    K = np.array([[843.0, 4.0, 319.5], [0.0, 827.0, 241.0], [0.0, 0.0, 1.0]])
    C = np.array([0.35, -0.20, 5.0])
    radius = 1.0
    toward_camera = -C / np.linalg.norm(C)
    tangent_x = np.cross(toward_camera, [0.0, 1.0, 0.0])
    tangent_x /= np.linalg.norm(tangent_x)
    tangent_y = np.cross(toward_camera, tangent_x)

    rng = np.random.default_rng(8)
    theta = rng.uniform(0.0, np.deg2rad(50.0), size=200)
    phi = rng.uniform(-np.pi, np.pi, size=200)
    dirs_true = (
        np.cos(theta)[:, None] * toward_camera
        + np.sin(theta)[:, None]
        * (
            np.cos(phi)[:, None] * tangent_x
            + np.sin(phi)[:, None] * tangent_y
        )
    )
    points = C + radius * dirs_true
    homogeneous = points @ K.T
    uv = homogeneous[:, :2] / homogeneous[:, 2:]

    recovered, valid = unproject_to_sphere(uv, K, C, radius)
    assert np.all(valid)
    error = np.linalg.norm(recovered - dirs_true, axis=1)
    assert float(np.max(error)) < 1e-9


def test_unprojection_marks_missed_rays_invalid_and_nan() -> None:
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    directions, valid = unproject_to_sphere([[1.0e6, 1.0e6]], K, [0.0, 0.0, 5.0], 1.0)
    assert not valid[0]
    assert np.all(np.isnan(directions[0]))


def test_sphere_pose_and_viewing_angle_conventions() -> None:
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    C, radius = sphere_pose_from_circle(320.0, 240.0, 160.0, K)
    np.testing.assert_allclose(C, [0.0, 0.0, np.sqrt(26.0)], atol=1e-14)
    assert radius == 1.0

    center_normal = np.array([0.0, 0.0, -1.0])
    tangent_normal = np.array([np.sqrt(24.0) / 5.0, 0.0, -1.0 / 5.0])
    angles = viewing_angle(
        np.array([center_normal, tangent_normal, [np.nan, 0.0, 1.0], [0.0, 0.0, 0.0]]),
        [0.0, 0.0, 5.0],
        1.0,
    )
    assert angles[0] == pytest.approx(0.0, abs=1e-15)
    assert angles[1] == pytest.approx(np.pi / 2.0, abs=1e-14)
    assert np.isnan(angles[2]) and np.isnan(angles[3])


def test_circle_fit_points_is_exact_on_clean_boundary() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    points = np.column_stack([123.4 + 51.2 * np.cos(theta), 98.7 + 51.2 * np.sin(theta)])
    u0, v0, radius = fit_circle_points(points)
    np.testing.assert_allclose([u0, v0, radius], [123.4, 98.7, 51.2], atol=1e-11)


def test_kabsch_recovers_known_rotation_and_direction_convention() -> None:
    """Stage-A hard guard: the solver maps previous ``a`` forward to ``b``."""

    rng = np.random.default_rng(8675309)
    a = rng.normal(size=(80, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    expected = _axis_angle(np.array([0.31, -0.74, 0.59]), 0.63)
    b = a @ expected.T  # row-array form of b = R @ a

    recovered = kabsch(a, b)
    assert geodesic_angle(recovered, expected) < 1e-6
    np.testing.assert_allclose(a @ recovered.T, b, atol=1e-12)

    # The inverse mapping must be measurably wrong; this fails loudly if the
    # covariance/SVD convention is ever flipped.
    inverse_residual = np.mean(np.linalg.norm(a @ recovered - b, axis=1))
    assert inverse_residual > 0.25
