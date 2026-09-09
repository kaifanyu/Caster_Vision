"""Numeric frame/geometry gates for the swivel-caster implementation."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common.estimate import kabsch
from common.caster_frame import CasterFrame, load_caster_frames, save_caster_frames
from common.rotation import Rz
from swivel.geometry import (
    SwivelGeometry,
    project_camera_points,
    project_sidewall,
    unproject_to_plane,
)
from swivel.roll import (
    estimate_roll_increment,
    image_plane_circular_check,
    prepare_roll_correspondences,
)
from swivel.tag import (
    ArucoTagTracker,
    differentiate_angles_with_gaps,
    generate_marker_image,
    marker_object_points,
    relative_swivel_yaw,
    unwrap_angles_with_gaps,
)


def _side_camera_geometry() -> SwivelGeometry:
    # Camera x=car x, camera y=-car z, camera z=car y.  At psi=0 the
    # signed axle -car-y points toward the camera as -camera-z.
    R_camera_from_car = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    return SwivelGeometry(
        R_camera_from_car=R_camera_from_car,
        t_camera_from_car=np.array([0.0, 0.0, 3.0]),
        swivel_axis_car=np.zeros(3),
        hub_offset_zero_car=np.array([-0.25, 0.0, 0.0]),
        axle_zero_car=np.array([0.0, -1.0, 0.0]),
        wheel_radius=0.45,
        wheel_width=0.08,
    )


def _project(points_camera: np.ndarray, K: np.ndarray) -> np.ndarray:
    uv, valid = project_camera_points(points_camera, K)
    assert np.all(valid)
    return uv


def test_dynamic_trail_and_signed_axle_follow_positive_swivel() -> None:
    geometry = SwivelGeometry(
        R_camera_from_car=np.eye(3),
        t_camera_from_car=np.array([0.2, -0.3, 2.0]),
        swivel_axis_car=np.array([1.0, 2.0, 3.0]),
        hub_offset_zero_car=np.array([-0.2, 0.0, -0.1]),
        axle_zero_car=np.array([0.0, -1.0, 0.0]),
        wheel_radius=0.1,
        wheel_width=0.03,
    )
    np.testing.assert_allclose(geometry.hub_car(0.0), [0.8, 2.0, 2.9], atol=1e-15)
    np.testing.assert_allclose(
        geometry.hub_car(np.pi / 2.0), [1.0, 1.8, 2.9], atol=1e-15
    )
    np.testing.assert_allclose(
        geometry.axle_car(np.pi / 2.0), [1.0, 0.0, 0.0], atol=1e-15
    )
    np.testing.assert_allclose(geometry.heading_zero_car, [1.0, 0.0, 0.0])
    assert geometry.trail == pytest.approx(0.2)

    car_points = np.array([[0.3, -0.4, 0.8], [-0.5, 0.2, 1.1]])
    camera_points = geometry.car_points_to_camera(car_points)
    np.testing.assert_allclose(geometry.camera_points_to_car(camera_points), car_points)
    homogeneous = np.column_stack([car_points, np.ones(len(car_points))])
    via_transform = homogeneous @ geometry.T_camera_from_car.T
    np.testing.assert_allclose(via_transform[:, :3], camera_points)


def test_caster_frame_json_replaces_nonfinite_diagnostics_with_null(tmp_path) -> None:
    path = save_caster_frames(
        tmp_path / "frames.json",
        [
            CasterFrame(
                t=0.0,
                roll_axis_car=np.array([0.0, -1.0, 0.0]),
                omega_roll=0.0,
                omega_spin=0.0,
                r_eff=0.08,
                raw={"failed_residual": np.nan},
            )
        ],
        metadata={"optional_score": np.inf},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["frames"][0]["raw"]["failed_residual"] is None
    assert payload["metadata"]["optional_score"] is None
    loaded, metadata = load_caster_frames(path)
    assert len(loaded) == 1 and metadata["optional_score"] is None


def test_geometry_builds_from_repository_config_transform_direction() -> None:
    R_car_from_camera = Rz(0.3)
    T_car_from_camera = np.eye(4)
    T_car_from_camera[:3, :3] = R_car_from_camera
    T_car_from_camera[:3, 3] = [0.4, -0.2, 0.1]
    geometry = SwivelGeometry.from_config(
        {"T_car_from_cam": T_car_from_camera.tolist()},
        {
            "swivel_axis_car": [0.25, 0.0, 0.16],
            "hub_offset0_car": [-0.05, 0.0, -0.08],
            "axle0_car": [0.0, -1.0, 0.0],
            "wheel_radius_m": 0.079,
            "wheel_width_m": 0.035,
        },
    )
    np.testing.assert_allclose(geometry.T_car_from_camera, T_car_from_camera, atol=1e-14)
    camera_point = np.array([0.2, 0.1, 1.4])
    car_point = (T_car_from_camera @ np.r_[camera_point, 1.0])[:3]
    np.testing.assert_allclose(geometry.camera_points_to_car(camera_point), car_point)


def test_unproject_to_plane_round_trip_and_invalid_rays() -> None:
    K = np.array([[843.0, 4.0, 319.5], [0.0, 827.0, 241.0], [0.0, 0.0, 1.0]])
    Q = np.array([0.12, -0.18, 3.2])
    normal = np.array([0.20, -0.12, -1.0])
    normal /= np.linalg.norm(normal)
    first = np.cross(normal, [0.0, 1.0, 0.0])
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    theta = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    points = Q + 0.35 * (
        np.cos(theta)[:, None] * first + np.sin(theta)[:, None] * second
    )
    uv = _project(points, K)
    recovered, valid = unproject_to_plane(uv, K, Q, normal)
    assert np.all(valid)
    assert float(np.max(np.linalg.norm(recovered - points, axis=1))) < 1e-9

    center_K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    parallel, parallel_valid = unproject_to_plane(
        [[320.0, 240.0]], center_K, [1.0, 0.0, 2.0], [1.0, 0.0, 0.0]
    )
    behind, behind_valid = unproject_to_plane(
        [[320.0, 240.0]], center_K, [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]
    )
    assert not parallel_valid[0] and np.all(np.isnan(parallel[0]))
    assert not behind_valid[0] and np.all(np.isnan(behind[0]))


def test_projected_sidewall_mask_and_edge_on_confidence() -> None:
    geometry = _side_camera_geometry()
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    face_on = project_sidewall(
        geometry,
        0.0,
        K,
        (480, 640),
        inner_radius_fraction=0.2,
        margin_px=1,
    )
    assert face_on.face_sign == 1
    assert face_on.view_confidence > 0.99
    assert int(np.count_nonzero(face_on.mask)) > 20_000
    assert not face_on.mask[240, 253]  # projected hub/inner annulus is removed

    assert geometry.sidewall_view_confidence(np.pi / 2.0) < 1e-12


def _coupled_wheel_pixels(
    geometry: SwivelGeometry,
    K: np.ndarray,
    psi: float,
    phi: float,
    material_radial: np.ndarray,
    face_sign: int = 1,
) -> np.ndarray:
    roll = Rotation.from_rotvec(geometry.axle_zero_car * phi).as_matrix()
    radial_fork = material_radial @ roll.T
    face_offset = face_sign * 0.5 * geometry.wheel_width * geometry.axle_zero_car
    vectors_fork = radial_fork + face_offset
    vectors_car = vectors_fork @ Rz(psi).T
    points_car = geometry.hub_car(psi) + vectors_car
    return _project(geometry.car_points_to_camera(points_car), K)


def test_coupled_swivel_roll_recovery_removes_moving_hub_and_swivel() -> None:
    geometry = _side_camera_geometry()
    K = np.array([[930.0, 3.0, 320.0], [0.0, 910.0, 240.0], [0.0, 0.0, 1.0]])
    first, second = geometry.radial_basis_zero()
    theta = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    radius = np.linspace(0.16, 0.42, len(theta))
    material = radius[:, None] * (
        np.cos(theta)[:, None] * first + np.sin(theta)[:, None] * second
    )
    psi_prev, psi_curr = -0.25, 0.35
    phi_prev, phi_curr = 0.18, 0.61
    uv_prev = _coupled_wheel_pixels(geometry, K, psi_prev, phi_prev, material)
    uv_curr = _coupled_wheel_pixels(geometry, K, psi_curr, phi_curr, material)
    # Deliberate pixel outliers exercise the shared RANSAC/Kabsch path.
    uv_curr[:5] += np.array([18.0, -13.0])

    estimate = estimate_roll_increment(
        uv_prev,
        uv_curr,
        K,
        geometry,
        psi_prev,
        psi_curr,
        face_sign=1,
        min_view_confidence=0.01,
        ransac_iters=300,
        ransac_inlier_deg=0.2,
        min_inliers=12,
        rng=42,
    )
    assert estimate.success, estimate.failure_reason
    assert estimate.method == "ransac_kabsch"
    assert estimate.delta_phi == pytest.approx(phi_curr - phi_prev, abs=2e-5)
    assert estimate.inlier_count >= len(theta) - 6
    assert estimate.inlier_ratio > 0.85
    assert np.rad2deg(estimate.off_axis_residual_rad) < 1e-5
    assert estimate.image_plane_check is not None
    assert estimate.image_plane_check.delta_phi == pytest.approx(
        phi_curr - phi_prev, abs=2e-5
    )
    assert estimate.image_plane_check.inlier_count >= len(theta) - 6
    assert np.rad2deg(abs(estimate.image_plane_disagreement_rad)) < 1e-3
    assert estimate.quality["image_plane_circular_disagreement_deg"] == pytest.approx(
        np.rad2deg(estimate.image_plane_disagreement_rad), abs=1e-12
    )

    # Regression guard: fitting camera-frame centered vectors directly folds
    # the simultaneous swivel into the recovered rotation.
    clean_curr = _coupled_wheel_pixels(geometry, K, psi_curr, phi_curr, material)
    correspondences = prepare_roll_correspondences(
        uv_prev, clean_curr, K, geometry, psi_prev, psi_curr, face_sign=1
    )
    valid = correspondences.geometry_valid_mask
    p0 = correspondences.points_prev_camera[valid] - geometry.hub_center_camera(psi_prev)
    p1 = correspondences.points_curr_camera[valid] - geometry.hub_center_camera(psi_curr)
    naive = kabsch(p0, p1)
    naive_component = Rotation.from_matrix(naive).as_rotvec() @ geometry.axle_camera(psi_curr)
    assert abs(naive_component - (phi_curr - phi_prev)) > 0.01


def test_image_plane_circular_check_corrects_projected_ellipse_independently() -> None:
    geometry = _side_camera_geometry()
    K = np.array([[930.0, 3.0, 320.0], [0.0, 910.0, 240.0], [0.0, 0.0, 1.0]])
    first, second = geometry.radial_basis_zero()
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    radius = np.linspace(0.12, 0.43, len(theta))
    material = radius[:, None] * (
        np.cos(theta)[:, None] * first + np.sin(theta)[:, None] * second
    )
    psi_prev, psi_curr = -0.38, 0.42
    phi_prev, phi_curr = -0.21, 0.36
    uv_prev = _coupled_wheel_pixels(
        geometry, K, psi_prev, phi_prev, material
    )
    uv_curr = _coupled_wheel_pixels(
        geometry, K, psi_curr, phi_curr, material
    )
    uv_curr[:7] += np.array([25.0, -19.0])

    check = image_plane_circular_check(
        uv_prev,
        uv_curr,
        K,
        geometry,
        psi_prev,
        psi_curr,
        face_sign=1,
        inlier_rad=np.deg2rad(0.2),
        min_inliers=12,
    )

    assert check is not None
    assert check.delta_phi == pytest.approx(phi_curr - phi_prev, abs=1e-10)
    assert check.inlier_count >= len(theta) - 8
    assert check.inlier_ratio > 0.85
    assert np.rad2deg(check.mean_inlier_residual_rad) < 1e-8

    # Raw image angles around each projected center retain ellipse/perspective
    # distortion and, unlike the normalized check, do not isolate wheel roll.
    center_prev = _project(
        geometry.car_points_to_camera(geometry.face_center_car(psi_prev, 1))[None, :],
        K,
    )[0]
    center_curr = _project(
        geometry.car_points_to_camera(geometry.face_center_car(psi_curr, 1))[None, :],
        K,
    )[0]
    raw_prev = np.arctan2(
        uv_prev[7:, 1] - center_prev[1], uv_prev[7:, 0] - center_prev[0]
    )
    raw_curr = np.arctan2(
        uv_curr[7:, 1] - center_curr[1], uv_curr[7:, 0] - center_curr[0]
    )
    raw_delta = np.arctan2(
        np.mean(np.sin(raw_curr - raw_prev)),
        np.mean(np.cos(raw_curr - raw_prev)),
    )
    assert abs(raw_delta - (phi_curr - phi_prev)) > np.deg2rad(1.0)


def test_relative_tag_yaw_unwrap_and_derivative_preserve_gaps() -> None:
    zero = Rotation.from_euler("xyz", [0.2, -0.1, 0.4]).as_matrix()
    current = Rz(0.37) @ zero
    assert relative_swivel_yaw(current, zero) == pytest.approx(0.37, abs=1e-14)

    wrapped = np.deg2rad([170.0, -175.0, np.nan, -160.0])
    unwrapped = unwrap_angles_with_gaps(wrapped)
    np.testing.assert_allclose(
        unwrapped[[0, 1, 3]], np.deg2rad([170.0, 185.0, 200.0]), atol=1e-14
    )
    assert np.isnan(unwrapped[2])

    angles = np.array([0.0, 0.1, np.nan, 0.4, 0.5])
    derivative = differentiate_angles_with_gaps(angles, np.arange(5.0))
    np.testing.assert_allclose(derivative[[0, 1, 3, 4]], 0.1, atol=1e-14)
    assert np.isnan(derivative[2])


def _render_tag(
    K: np.ndarray,
    R_camera_from_car: np.ndarray,
    t_camera_from_car: np.ndarray,
    psi: float,
    *,
    marker_length: float,
    marker_id: int,
) -> np.ndarray:
    marker = generate_marker_image("DICT_4X4_50", marker_id, 240)
    R_tag_to_camera = R_camera_from_car @ Rz(psi)
    corners_camera = (
        marker_object_points(marker_length) @ R_tag_to_camera.T + t_camera_from_car
    )
    destination = _project(corners_camera, K).astype(np.float32)
    source = np.array(
        [[0.0, 0.0], [239.0, 0.0], [239.0, 239.0], [0.0, 239.0]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        marker,
        homography,
        (640, 480),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def test_rendered_aruco_marker_detects_and_recovers_relative_yaw() -> None:
    if not hasattr(cv2, "aruco"):
        pytest.skip("OpenCV build has no ArUco support")
    K = np.array([[820.0, 0.0, 320.0], [0.0, 815.0, 240.0], [0.0, 0.0, 1.0]])
    # Downward-looking camera: camera right=-car y, image down=-car x,
    # optical forward=-car z.  A top-mounted tag's +z faces the camera.
    R_camera_from_car = np.array(
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    )
    t_camera_from_car = np.array([0.0, 0.0, 1.0])
    marker_length = 0.14
    marker_id = 7
    tracker = ArucoTagTracker(
        K,
        marker_length,
        marker_id=marker_id,
        dictionary="DICT_4X4_50",
        R_car_from_camera=R_camera_from_car.T,
        R_tag_to_car_zero=np.eye(3),
        expected_normal_camera=R_camera_from_car @ np.array([0.0, 0.0, 1.0]),
        max_reprojection_error_px=1.5,
    )
    first = tracker.track(
        _render_tag(
            K,
            R_camera_from_car,
            t_camera_from_car,
            0.0,
            marker_length=marker_length,
            marker_id=marker_id,
        )
    )
    second = tracker.track(
        _render_tag(
            K,
            R_camera_from_car,
            t_camera_from_car,
            0.42,
            marker_length=marker_length,
            marker_id=marker_id,
        )
    )
    assert first.valid, first.failure_reason
    assert second.valid, second.failure_reason
    assert first.psi == pytest.approx(0.0, abs=np.deg2rad(0.7))
    assert second.psi == pytest.approx(0.42, abs=np.deg2rad(0.7))
    assert max(first.reprojection_error_px, second.reprojection_error_px) < 1.0
