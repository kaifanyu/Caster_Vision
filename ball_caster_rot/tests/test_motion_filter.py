"""Motion predictions stay separate from visual support and obey uncertainty limits."""
import json

import numpy as np
import pytest

from ballrot.motion_filter import MotionFilterConfig, filter_angles


def test_actual_variable_intervals_preserve_unwrapped_constant_rate():
    times = np.array([0., .03, .07, .15, .18, .27, .29, .42, .50, .66, .75])
    truth = 7*np.pi+1.4*times
    observed = truth[:, None]
    errors = np.full_like(observed, .001)
    original = observed.copy()
    result = filter_angles(times, observed, errors, MotionFilterConfig(
        accel_noise_deg_s2=0., max_prediction_s=1., max_angle_std_deg=90.))
    assert result.updated.all()
    assert not result.rejected.any()
    assert not result.reinitialized.any()
    assert np.max(np.abs(result.angles[:, 0]-truth)) < .001
    assert np.max(np.abs(result.velocities[3:, 0]-1.4)) < .001
    assert np.max(np.abs(result.predicted_angles[3:, 0]-truth[3:])) < .001
    assert np.all(result.angles > 2*np.pi)
    assert np.array_equal(observed, original)
    assert np.isnan(result.predicted_angles[0, 0])
    assert set(result.status[:, 0]) == {"vision"}


def test_fresh_measurements_reduce_large_prior_uncertainty_without_resetting_velocity():
    times = np.arange(11)*.1
    observed = (.5*times)[:, None]
    # The default 180 deg/s initial velocity uncertainty projects to an
    # 18-degree prior at 10 Hz, above the 10-degree rendering cutoff. Fresh
    # measurements must still update it normally and learn the actual rate.
    result = filter_angles(times, observed, np.full_like(observed, .001))
    assert result.updated.all()
    assert not result.reinitialized.any()
    assert np.all(result.status == "vision")
    assert np.max(np.abs(result.velocities[3:, 0]-.5)) < .01
    assert np.max(result.std) < np.deg2rad(10.)


def test_short_gap_is_predicted_long_gap_is_hidden_and_reacquisition_is_explicit():
    times = np.arange(15)*.1
    observed = (.5*times)[:, None]
    observed[4:12] = np.nan
    observed[12:] = 12.+.5*(times[12:]-times[12])[:, None]
    errors = np.where(np.isfinite(observed), .001, np.nan)
    result = filter_angles(times, observed, errors, MotionFilterConfig(
        accel_noise_deg_s2=0., max_prediction_s=.35, max_angle_std_deg=90.))
    assert np.all(result.status[4:7, 0] == "predicted")
    assert np.all(result.status[7:12, 0] == "unresolved")
    assert np.isnan(result.angles[7:12, 0]).all()
    assert np.isnan(result.velocities[7:12, 0]).all()
    assert np.isfinite(result.predicted_angles[7:12, 0]).all()
    assert np.isfinite(result.covariance[7:12, 0]).all()
    assert np.allclose(result.age_s[4:12, 0], times[4:12]-times[3])
    assert result.updated[12, 0] and result.reinitialized[12, 0]
    assert not result.rejected[12, 0]
    assert result.angles[12, 0] == 12.
    assert result.velocities[12, 0] == 0.
    assert result.age_s[12, 0] == 0.
    assert result.predicted_angles[12, 0] < 1.
    assert result.diagnostics["reinitializations"] == [1]
    json.dumps(result.diagnostics, allow_nan=False)


def test_components_are_independent_and_unseen_spin_is_never_inferred():
    times = np.arange(12)*.03
    observed = np.column_stack((times, -2*times, 4*times))
    observed[3:, 1] = np.nan
    observed[:5, 2] = np.nan
    errors = np.where(np.isfinite(observed), .003, np.nan)
    config = MotionFilterConfig(max_prediction_s=.1)
    result = filter_angles(times, observed, errors, config)
    alone = filter_angles(times, observed[:, :1], errors[:, :1], config)
    assert np.array_equal(result.angles[:, 0], alone.angles[:, 0])
    assert np.array_equal(result.covariance[:, 0], alone.covariance[:, 0])
    assert np.all(result.status[:, 0] == "vision")
    assert np.isnan(result.angles[:5, 2]).all()
    assert np.isnan(result.predicted_angles[:5, 2]).all()
    assert np.isnan(result.covariance[:5, 2]).all()
    assert np.isnan(result.age_s[:5, 2]).all()
    assert np.all(result.status[7:, 1] == "unresolved")
    assert result.updated[5, 2]
    assert not result.reinitialized[5, 2]


def test_fresh_outlier_is_rejected_without_replacing_measured_motion():
    times = np.arange(11)*.1
    truth = .2*times
    observed = truth[:, None].copy()
    observed[5, 0] += np.deg2rad(100.)
    result = filter_angles(times, observed, np.full_like(observed, .001), MotionFilterConfig(
        accel_noise_deg_s2=0., initial_velocity_std_deg_s=30., innovation_gate_sigma=3.,
        max_prediction_s=1., max_angle_std_deg=90.))
    assert np.flatnonzero(result.rejected[:, 0]).tolist() == [5]
    assert not result.updated[5, 0]
    assert result.status[5, 0] == "predicted"
    assert result.age_s[5, 0] == pytest.approx(.1)
    assert np.max(np.abs(result.angles[:, 0]-truth)) < .001
    assert not result.reinitialized.any()


def test_uncertainty_limit_can_hide_prediction_before_age_limit():
    observed = np.array([[0.], [np.nan], [np.nan]])
    result = filter_angles([0., .1, .2], observed, [[.001], [np.nan], [np.nan]],
                           MotionFilterConfig(max_prediction_s=1., max_angle_std_deg=1.))
    assert result.status[0, 0] == "vision"
    assert np.all(result.status[1:, 0] == "unresolved")
    assert np.all(result.age_s[1:, 0] < 1.)
    assert np.all(result.std[1:, 0] > np.deg2rad(1.))
    assert np.isnan(result.angles[1:, 0]).all()
    assert np.isfinite(result.predicted_angles[1:, 0]).all()


def test_zero_prediction_horizon_never_renders_missing_measurements():
    result = filter_angles([0., .01], [[1.], [np.nan]], [[.001], [np.nan]],
                           {"max_prediction_s": 0.})
    assert result.status[:, 0].tolist() == ["vision", "unresolved"]
    assert np.isnan(result.angles[1, 0])


def test_acceleration_model_increases_uncertainty_during_a_gap():
    times = np.linspace(0., .3, 16)
    measurements = np.full((len(times), 1), np.nan)
    measurements[0] = 0.
    errors = np.full_like(measurements, .001)
    stationary = filter_angles(times, measurements, errors, MotionFilterConfig(
        accel_noise_deg_s2=0., initial_velocity_std_deg_s=1., max_angle_std_deg=90.))
    accelerating = filter_angles(times, measurements, errors, MotionFilterConfig(
        accel_noise_deg_s2=180., initial_velocity_std_deg_s=1., max_angle_std_deg=90.))
    assert accelerating.std[-1, 0] > stationary.std[-1, 0]*2
    assert np.all(np.diff(accelerating.std[:, 0]) > 0)


def test_joseph_updates_keep_finite_symmetric_positive_semidefinite_covariances():
    rng = np.random.default_rng(143)
    times = np.r_[0., np.cumsum(rng.uniform(.01, .06, 180))]
    truth = np.column_stack((np.sin(times), -.5*np.sin(times/2), .3*times))
    errors = rng.uniform(.005, .025, truth.shape)
    observed = truth+rng.normal(size=truth.shape)*errors
    missing = rng.random(observed.shape) < .15
    missing[0] = False
    observed[missing] = np.nan
    result = filter_angles(times, observed, errors, MotionFilterConfig(max_angle_std_deg=30.))
    assert np.isfinite(result.covariance).all()
    assert np.allclose(result.covariance, result.covariance.swapaxes(-1, -2), atol=1e-14)
    assert np.min(np.linalg.eigvalsh(result.covariance)) >= -1e-14
    assert np.allclose(result.std**2, result.covariance[:, :, 0, 0])
    assert not np.any(result.updated & result.rejected)


def test_all_missing_and_empty_inputs_leave_no_invented_state():
    result = filter_angles([0., .1], np.full((2, 3), np.nan), np.full((2, 3), np.nan))
    assert np.isnan(result.angles).all()
    assert np.isnan(result.covariance).all()
    assert not result.updated.any()
    assert np.all(result.status == "unresolved")
    empty = filter_angles([], np.empty((0, 3)), np.empty((0, 3)))
    assert empty.angles.shape == (0, 3)
    assert empty.covariance.shape == (0, 3, 2, 2)


@pytest.mark.parametrize("timestamps,measurements,std", [
    ([[0.]], [[0.]], [[.1]]),
    ([np.nan], [[0.]], [[.1]]),
    ([0., 0.], [[0.], [1.]], [[.1], [.1]]),
    ([1., 0.], [[0.], [1.]], [[.1], [.1]]),
    ([0.], [0.], [[.1]]),
    ([0.], [[0.], [1.]], [[.1], [.1]]),
    ([0.], np.empty((1, 0)), np.empty((1, 0))),
    ([0.], [[0.]], [.1]),
    ([0.], [[np.inf]], [[.1]]),
    ([0.], [[0.]], [[0.]]),
    ([0.], [[0.]], [[-.1]]),
    ([0.], [[0.]], [[np.nan]]),
    ([0.], [[0.]], [[np.inf]]),
])
def test_invalid_measurement_inputs_are_rejected(timestamps, measurements, std):
    with pytest.raises(ValueError):
        filter_angles(timestamps, measurements, std)


@pytest.mark.parametrize("settings", [
    {"accel_noise_deg_s2": -1.}, {"accel_noise_deg_s2": True},
    {"initial_velocity_std_deg_s": 0.}, {"innovation_gate_sigma": 0.},
    {"max_prediction_s": -.1}, {"max_angle_std_deg": 0.},
    {"max_angle_std_deg": np.inf}, {"unknown": 1}, "invalid",
])
def test_invalid_configuration_is_rejected(settings):
    with pytest.raises(ValueError):
        MotionFilterConfig.from_mapping(settings)
