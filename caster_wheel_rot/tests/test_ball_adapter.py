"""Focused tests for the Approach-A ball-to-common adapter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ball.adapter import adapt_ball_pipeline
from ball.estimate import (
    angular_residuals as ball_angular_residuals,
    kabsch as ball_kabsch,
    ransac_kabsch as ball_ransac_kabsch,
)
from common.estimate import angular_residuals, kabsch, ransac_kabsch


def _camera_increment(
    omega_car: np.ndarray,
    dt: float,
    R_bc: np.ndarray,
    R_ball_to_car: np.ndarray,
) -> np.ndarray:
    """Invert the adapter's two conjugations for a known car-frame omega."""

    R_car = Rotation.from_rotvec(np.asarray(omega_car, dtype=float) * dt).as_matrix()
    return R_bc @ R_ball_to_car.T @ R_car @ R_ball_to_car @ R_bc.T


def _result(
    timestamps: list[float],
    top: list[np.ndarray | None],
    bottom: list[np.ndarray | None],
    *,
    motion: SimpleNamespace | None = None,
    qualities: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        timestamps_s=np.asarray(timestamps, dtype=float),
        top_increments=top,
        bottom_increments=bottom,
        motion=motion,
        qualities=[] if qualities is None else qualities,
    )


def _frames(result: SimpleNamespace, **kwargs):
    defaults = {
        "R_bc": np.eye(3),
        "R_ball_to_car": np.eye(3),
        "r_eff": 0.04,
    }
    defaults.update(kwargs)
    return adapt_ball_pipeline(result, **defaults)


def test_ball_estimator_delegates_shared_kabsch_implementations() -> None:
    assert ball_kabsch is kabsch
    assert ball_angular_residuals is angular_residuals
    assert ball_ransac_kabsch is ransac_kabsch


def test_adapter_conjugates_averages_and_timestamps_at_interval_midpoint() -> None:
    dt = 0.2
    R_bc = Rotation.from_euler("xyz", [0.23, -0.31, 0.17]).as_matrix()
    R_ball_to_car = Rotation.from_euler("z", 0.42).as_matrix()
    top_omega = np.array([2.0, 0.0, 1.0])
    bottom_omega = np.array([4.0, 0.0, 3.0])
    motion = SimpleNamespace(
        alpha=np.array([0.0, 0.2]),
        beta_top=np.array([0.0, 0.4]),
        beta_bottom=np.array([0.0, -0.2]),
    )
    qualities = [
        SimpleNamespace(
            top=SimpleNamespace(inlier_ratio=0.8),
            bottom=SimpleNamespace(inlier_ratio=0.6),
        )
    ]
    result = _result(
        [10.0, 10.0 + dt],
        [_camera_increment(top_omega, dt, R_bc, R_ball_to_car)],
        [_camera_increment(bottom_omega, dt, R_bc, R_ball_to_car)],
        motion=motion,
        qualities=qualities,
    )

    frame = _frames(
        result,
        R_bc=R_bc,
        R_ball_to_car=R_ball_to_car,
        r_eff=0.031,
    )[0]

    assert frame.t == pytest.approx(10.1)
    np.testing.assert_allclose(frame.roll_axis_car, [-1.0, 0.0, 0.0], atol=1e-12)
    assert frame.omega_roll == pytest.approx(3.0, abs=1e-12)
    assert frame.omega_spin == pytest.approx(2.0, abs=1e-12)
    assert frame.r_eff == pytest.approx(0.031)
    assert frame.roll_valid and frame.heading_valid
    assert frame.confidence == pytest.approx(0.7)
    assert frame.raw["alpha"] == pytest.approx(0.1)
    assert frame.raw["beta1"] == pytest.approx(0.2)
    assert frame.raw["beta2"] == pytest.approx(-0.1)
    assert frame.raw["beta1_dot"] == pytest.approx(1.0, abs=1e-12)
    assert frame.raw["beta2_dot"] == pytest.approx(3.0, abs=1e-12)
    assert frame.raw["hemisphere_source"] == "both"


def test_roll_direction_sign_flips_axis_without_changing_speed() -> None:
    dt = 0.1
    increment = Rotation.from_rotvec([0.0, 0.2, 0.0]).as_matrix()
    result = _result([0.0, dt], [increment], [increment])

    default_frame = _frames(result)[0]
    positive_frame = _frames(result, roll_direction_sign=1)[0]

    np.testing.assert_allclose(default_frame.roll_axis_car, -positive_frame.roll_axis_car)
    assert default_frame.omega_roll == pytest.approx(positive_frame.omega_roll)
    assert default_frame.raw["roll_direction_sign"] == -1.0


def test_one_valid_hemisphere_is_usable_at_reduced_confidence() -> None:
    dt = 0.25
    bottom = Rotation.from_rotvec(np.array([0.0, 2.0, -0.5]) * dt).as_matrix()
    quality = SimpleNamespace(
        top=SimpleNamespace(inlier_ratio=0.95),
        bottom=SimpleNamespace(inlier_ratio=0.8),
    )
    result = _result([2.0, 2.25], [None], [bottom], qualities=[quality])

    frame = _frames(result)[0]

    assert frame.roll_valid and frame.heading_valid
    assert frame.confidence == pytest.approx(0.4)
    np.testing.assert_allclose(frame.roll_axis_car, [0.0, -1.0, 0.0], atol=1e-12)
    assert frame.omega_roll == pytest.approx(2.0, abs=1e-12)
    assert frame.omega_spin == pytest.approx(-0.5, abs=1e-12)
    assert frame.raw["top_valid"] is False
    assert frame.raw["bottom_valid"] is True
    assert frame.raw["beta1_dot"] is None
    assert frame.raw["hemisphere_source"] == "bottom"


def test_both_invalid_retains_finite_axis_and_marks_roll_dropout() -> None:
    dt = 0.1
    top = Rotation.from_rotvec([0.2, 0.0, 0.0]).as_matrix()
    result = _result([0.0, dt, 2.0 * dt], [top, None], [None, None])

    first, dropout = _frames(result)

    assert first.roll_valid
    assert dropout.t == pytest.approx(0.15)
    assert not dropout.roll_valid
    assert not dropout.heading_valid
    assert dropout.confidence == 0.0
    assert dropout.omega_roll == 0.0
    assert dropout.omega_spin == 0.0
    np.testing.assert_allclose(dropout.roll_axis_car, first.roll_axis_car)
    assert dropout.raw["hemisphere_source"] == "none"


def test_pipeline_step_valid_mask_can_reject_a_present_increment() -> None:
    result = _result([0.0, 0.1], [np.eye(3)], [np.eye(3)])
    result.top_step_valid = np.array([True, False])
    result.bottom_step_valid = np.array([True, False])

    frame = _frames(result)[0]

    assert not frame.roll_valid
    assert frame.confidence == 0.0
    assert frame.raw["top_valid"] is False
    assert frame.raw["bottom_valid"] is False


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_adapter_rejects_invalid_effective_radius(value: float) -> None:
    result = _result([0.0, 0.1], [np.eye(3)], [np.eye(3)])
    with pytest.raises(ValueError, match="r_eff"):
        _frames(result, r_eff=value)


def test_adapter_rejects_bad_timing_shape_and_direction_sign() -> None:
    result = _result([0.0, 0.1], [], [np.eye(3)])
    with pytest.raises(ValueError, match="increment sequence"):
        _frames(result)

    result = _result([0.0, 0.1], [np.eye(3)], [np.eye(3)])
    with pytest.raises(ValueError, match="roll_direction_sign"):
        _frames(result, roll_direction_sign=0.5)

    result.timestamps_s = np.array([0.1, 0.1])
    with pytest.raises(ValueError, match="strictly increasing"):
        _frames(result)
