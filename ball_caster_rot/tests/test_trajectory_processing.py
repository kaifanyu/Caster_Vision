"""Offline derivatives respect timing, gaps, and rotation-frame conventions."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ballrot.trajectory import angular_rates, rotation_rates


def _irregular_times() -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.random.default_rng(17).uniform(0.012, 0.028, 90))]


@pytest.mark.parametrize("order", [1, 2, 3])
def test_constant_angular_rate_native_irregular_timestamps_and_endpoints(order: int) -> None:
    times = _irregular_times()
    result = angular_rates(8.0 - 2.4 * times, times, polynomial_order=order)
    np.testing.assert_allclose(result, -2.4, atol=1e-11)


def test_quadratic_angle_exact_derivative_including_one_sided_endpoints() -> None:
    times = _irregular_times()
    result = angular_rates(3.0 + 0.8 * times - 1.5 * times**2, times)
    np.testing.assert_allclose(result, 0.8 - 3.0 * times, atol=1e-11)


@pytest.mark.parametrize("explicit_valid", [False, True])
def test_invalid_observation_splits_rates_without_bridging(explicit_valid: bool) -> None:
    times = np.arange(11) * 0.02
    angles = np.where(np.arange(11) < 5, 2 * times, 100 - 4 * times)
    valid = np.ones(len(times), dtype=bool)
    valid[5] = False
    if not explicit_valid:
        angles[5] = np.nan
    result = angular_rates(angles, times, valid if explicit_valid else None, window_s=1.0)
    np.testing.assert_allclose(result[:5], 2.0, atol=1e-10)
    assert np.isnan(result[5])
    np.testing.assert_allclose(result[6:], -4.0, atol=1e-10)


def test_too_short_segments_or_windows_remain_unresolved() -> None:
    times = np.arange(7) * 0.03
    valid = np.array([True, True, False, True, False, True, True])
    assert np.all(np.isnan(angular_rates(times, times, valid)))
    assert np.all(np.isnan(angular_rates(times, times, window_s=0.01)))
    rotations = Rotation.from_rotvec(times[:, None] * [0.0, 0.0, 1.0]).as_matrix()
    assert np.all(np.isnan(rotation_rates(rotations, times, valid)))


def test_empty_trajectories_and_isolated_observations() -> None:
    assert angular_rates(np.empty(0), np.empty(0)).shape == (0,)
    assert rotation_rates(np.empty((0, 3, 3)), np.empty(0)).shape == (0, 3)
    assert np.isnan(angular_rates([1.0], [0.0])[0])
    assert np.all(np.isnan(rotation_rates(np.eye(3)[None], [0.0])))


def test_local_fit_reduces_derivative_noise_without_changing_native_timing() -> None:
    times = _irregular_times()
    angles = 0.5 * times**2 + np.random.default_rng(42).normal(0.0, 0.004, len(times))
    fitted = angular_rates(angles, times)
    naive = np.gradient(angles, times)
    rms_fit = np.sqrt(np.mean((fitted - times) ** 2))
    rms_naive = np.sqrt(np.mean((naive - times) ** 2))
    assert rms_fit < 0.3 * rms_naive


def test_fixed_axis_rotation_rates_handle_large_total_turns() -> None:
    times = _irregular_times()
    axis = np.array([1.0, 2.0, -3.0]) / np.sqrt(14)
    rotations = Rotation.from_rotvec((0.5 + 5.0 * times)[:, None] * axis).as_matrix()
    result = rotation_rates(rotations, times)
    np.testing.assert_allclose(result, np.broadcast_to(5.0 * axis, result.shape), atol=1e-11)


def test_rotation_quadratic_angle_exact_local_rate() -> None:
    times = _irregular_times()
    rotations = Rotation.from_rotvec((0.2 * times + 0.1 * times**2)[:, None] * [0.0, 1.0, 0.0]).as_matrix()
    result = rotation_rates(rotations, times)
    expected = np.zeros_like(result)
    expected[:, 1] = 0.2 + 0.2 * times
    np.testing.assert_allclose(result, expected, atol=1e-11)


def test_noncommuting_rotations_report_spatial_not_body_angular_velocity() -> None:
    times = np.r_[0.0, np.cumsum(np.random.default_rng(9).uniform(0.006, 0.014, 100))]
    a, b = 0.7, 1.1
    rx = Rotation.from_rotvec(times[:, None] * [a, 0.0, 0.0]).as_matrix()
    rz = Rotation.from_rotvec(times[:, None] * [0.0, 0.0, b]).as_matrix()
    result = rotation_rates(rx @ rz, times, window_s=0.12)
    expected = np.c_[np.full(len(times), a), -b * np.sin(a * times), b * np.cos(a * times)]
    np.testing.assert_allclose(result, expected, atol=0.002)


def test_rotation_rates_rotate_with_camera_coordinates() -> None:
    times = _irregular_times()
    rotations = Rotation.from_rotvec(times[:, None] * [0.0, 0.8, 0.0]).as_matrix()
    camera = Rotation.from_rotvec([0.3, 0.7, -0.4]).as_matrix()
    result = rotation_rates(camera @ rotations @ camera.T, times)
    np.testing.assert_allclose(result, np.broadcast_to(camera @ [0.0, 0.8, 0.0], result.shape), atol=1e-11)


def test_rotation_invalid_gap_does_not_create_spurious_rate() -> None:
    times = np.arange(11) * 0.02
    angles = np.where(np.arange(11) < 5, times, 2.0 - 3.0 * times)
    rotations = Rotation.from_rotvec(angles[:, None] * [0.0, 0.0, 1.0]).as_matrix()
    rotations[5] = np.nan
    result = rotation_rates(rotations, times, window_s=1.0)
    np.testing.assert_allclose(result[:5], np.broadcast_to([0.0, 0.0, 1.0], (5, 3)), atol=1e-10)
    assert np.all(np.isnan(result[5]))
    np.testing.assert_allclose(result[6:], np.broadcast_to([0.0, 0.0, -3.0], (5, 3)), atol=1e-10)


def test_fast_motion_caps_logarithm_support_instead_of_fitting_wrapped_values() -> None:
    times = np.arange(21) * 0.02
    speed = 40.0  # Adjacent steps are 0.8 rad; full window spans multiple turns.
    rotations = Rotation.from_rotvec(times[:, None] * [0.0, 0.0, speed]).as_matrix()
    result = rotation_rates(rotations, times, window_s=0.8)
    assert np.all(np.isnan(result[[0, -1]]))  # Only two nearby poses at each end.
    np.testing.assert_allclose(result[1:-1, 2], speed, atol=1e-10)
    np.testing.assert_allclose(result[1:-1, :2], 0.0, atol=1e-10)


def test_adjacent_motion_outside_logarithm_support_remains_unresolved() -> None:
    times = np.arange(8) * 0.02
    rotations = Rotation.from_rotvec(times[:, None] * [0.0, 100.0, 0.0]).as_matrix()
    assert np.all(np.isnan(rotation_rates(rotations, times, window_s=1.0)))


@pytest.mark.parametrize("times", [[0.0, 0.0, 1.0], [0.0, 1.0, 0.5], [0.0, np.nan, 1.0], [0.0, np.inf, 1.0]])
def test_invalid_timestamps_raise(times: list[float]) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        angular_rates(np.zeros(3), times)
    with pytest.raises(ValueError, match="strictly increasing"):
        rotation_rates(np.repeat(np.eye(3)[None], 3, axis=0), times)


@pytest.mark.parametrize("options", [{"window_s": 0}, {"window_s": -1}, {"window_s": np.nan}, {"polynomial_order": 0}, {"polynomial_order": 1.5}, {"polynomial_order": True}])
def test_invalid_fit_parameters_raise(options: dict) -> None:
    with pytest.raises(ValueError):
        angular_rates(np.arange(4), np.arange(4), **options)
    with pytest.raises(ValueError):
        rotation_rates(np.repeat(np.eye(3)[None], 4, axis=0), np.arange(4), **options)


def test_input_shapes_and_rotation_validity_are_checked() -> None:
    with pytest.raises(ValueError, match="angles"):
        angular_rates(np.zeros((3, 1)), np.arange(3))
    with pytest.raises(ValueError, match="timestamps"):
        angular_rates(np.zeros(3), np.arange(4))
    with pytest.raises(ValueError, match="valid"):
        angular_rates(np.zeros(3), np.arange(3), np.ones(4))
    with pytest.raises(ValueError, match="rotations"):
        rotation_rates(np.eye(3), np.arange(3))
    with pytest.raises(ValueError, match="proper rotation"):
        rotation_rates(np.repeat((2 * np.eye(3))[None], 3, axis=0), np.arange(3))
    # Invalid samples may contain bad finite placeholders; they are never fitted.
    result = rotation_rates(np.repeat((2 * np.eye(3))[None], 3, axis=0), np.arange(3), np.zeros(3, dtype=bool))
    assert np.all(np.isnan(result))
