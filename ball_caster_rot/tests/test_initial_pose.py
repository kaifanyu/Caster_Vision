"""Regression gates for clip-specific initial roll and saved-angle recovery."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ballrot.config import measurement_frame
from ballrot.integrate import decompose_hemispheres, inter_shell_swivel_axis


def _observed_rotations(
    home: np.ndarray, alpha_deg: np.ndarray, beta_deg: np.ndarray
) -> np.ndarray:
    """Create camera motion from physical poses with nonzero initial phases.

    Build each absolute pose first, then subtract the first pose by a matrix
    product. This is independent of the correction being tested.
    """

    poses = home @ Rotation.from_euler(
        "XZ", np.column_stack([alpha_deg, beta_deg]), degrees=True
    ).as_matrix()
    return poses @ poses[0].T


@pytest.mark.parametrize("initial_roll_deg", [-173.0, -90.0, 37.5, 90.0, 168.0])
def test_initial_roll_recovers_motion_from_arbitrary_initial_shell_phases(
    initial_roll_deg: float,
) -> None:
    rng = np.random.default_rng(9123)
    t = np.linspace(0.0, 1.0, 181)
    delta_alpha_deg = 225.0 * t + 8.0 * np.sin(2.0 * np.pi * t)
    delta_top_deg = -300.0 * t
    delta_bottom_deg = 440.0 * t
    for _ in range(4):
        home = Rotation.random(random_state=rng).as_matrix()
        original_home = home.copy()
        top_phase, bottom_phase = rng.uniform(-180.0, 180.0, 2)
        alpha = initial_roll_deg + delta_alpha_deg
        top = _observed_rotations(home, alpha, top_phase + delta_top_deg)
        bottom = _observed_rotations(home, alpha, bottom_phase + delta_bottom_deg)
        frame = measurement_frame(
            {"R_bc": home, "initial_roll_deg": initial_roll_deg}
        )

        motion = decompose_hemispheres(top, bottom, frame)

        np.testing.assert_allclose(motion.alpha, np.deg2rad(delta_alpha_deg), atol=2e-12)
        np.testing.assert_allclose(motion.alpha_top, motion.alpha_bottom, atol=2e-12)
        np.testing.assert_allclose(motion.beta_top, np.deg2rad(delta_top_deg), atol=2e-12)
        np.testing.assert_allclose(
            motion.beta_bottom, np.deg2rad(delta_bottom_deg), atol=2e-12
        )
        np.testing.assert_allclose(motion.gamma_top, 0.0, atol=2e-12)
        np.testing.assert_allclose(motion.gamma_bottom, 0.0, atol=2e-12)
        np.testing.assert_array_equal(home, original_home)
        # A pose offset must preserve the calibrated roll axis.
        np.testing.assert_allclose(frame[:, 0], home[:, 0], atol=2e-12)
        # Independent differential shell motion observes the initial swivel
        # axis. It need not agree with the home pose's swivel axis.
        axis, diagnostics = inter_shell_swivel_axis(top, bottom)
        assert axis is not None
        assert diagnostics["sample_count"] > 100
        assert abs(float(axis @ frame[:, 2])) == pytest.approx(1.0, abs=2e-12)


def test_zero_initial_roll_preserves_legacy_frame_and_relative_angles() -> None:
    home = Rotation.from_euler("xyz", [31.0, -22.0, 17.0], degrees=True).as_matrix()
    default = measurement_frame({"R_bc": home})
    explicit_zero = measurement_frame({"R_bc": home, "initial_roll_deg": 0.0})
    np.testing.assert_allclose(default, home, atol=1e-15)
    np.testing.assert_allclose(explicit_zero, home, atol=1e-15)
    alpha = np.linspace(0.0, -75.0, 50)
    top = _observed_rotations(home, alpha, np.linspace(61.0, 146.0, 50))
    bottom = _observed_rotations(home, alpha, np.linspace(-23.0, -83.0, 50))
    old = decompose_hemispheres(top, bottom, home)
    new = decompose_hemispheres(top, bottom, default)
    for name in ("alpha", "beta_top", "beta_bottom", "gamma_top", "gamma_bottom"):
        np.testing.assert_allclose(getattr(new, name), getattr(old, name), atol=1e-14)


def test_measurement_frame_does_not_invent_missing_axis_calibration() -> None:
    assert measurement_frame({}) is None
    assert measurement_frame({"R_bc": None, "initial_roll_deg": 90.0}) is None


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_measurement_frame_rejects_nonfinite_initial_roll(value: float) -> None:
    with pytest.raises(ValueError, match="initial_roll_deg"):
        measurement_frame({"R_bc": np.eye(3), "initial_roll_deg": value})


def test_saved_per_shell_euler_angles_recover_camera_rotations_and_keep_gaps() -> None:
    """Historical per-shell angles retain full rotations despite a wrong frame.

    Include a real off-model residual and different per-shell roll so that
    omitting gamma or using the fused alpha cannot accidentally pass.
    """

    t = np.linspace(0.0, 1.0, 91)
    home = Rotation.from_euler("xyz", [29.0, -34.0, 71.0], degrees=True).as_matrix()
    frame = measurement_frame({"R_bc": home, "initial_roll_deg": 83.0})
    top_xyz = np.column_stack([215.0 * t, 6.0 * np.sin(np.pi * t), -280.0 * t])
    bottom_xyz = np.column_stack([210.0 * t, -4.0 * np.sin(np.pi * t), 390.0 * t])
    top = frame @ Rotation.from_euler("XYZ", top_xyz, degrees=True).as_matrix() @ frame.T
    bottom = (
        frame @ Rotation.from_euler("XYZ", bottom_xyz, degrees=True).as_matrix() @ frame.T
    )
    valid_top = np.ones(len(t), dtype=bool)
    valid_bottom = valid_top.copy()
    valid_top[[4, 5, 16]] = False
    valid_bottom[[5, 9, 70]] = False
    old = decompose_hemispheres(
        top, bottom, home, valid_top=valid_top, valid_bottom=valid_bottom
    )

    recovered = []
    for name, expected, valid in (
        ("top", top, valid_top), ("bottom", bottom, valid_bottom)
    ):
        saved_xyz = np.column_stack(
            [getattr(old, f"alpha_{name}"), getattr(old, f"gamma_{name}"), getattr(old, f"beta_{name}")]
        )
        assert np.isnan(saved_xyz[~valid]).all()
        restored = np.repeat(np.eye(3)[None, :, :], len(t), axis=0)
        restored[valid] = home @ Rotation.from_euler("XYZ", saved_xyz[valid]).as_matrix() @ home.T
        np.testing.assert_allclose(restored[valid], expected[valid], atol=3e-12)
        recovered.append(restored)

    corrected = decompose_hemispheres(
        *recovered, frame, valid_top=valid_top, valid_bottom=valid_bottom
    )
    for name, expected_xyz, valid in (
        ("top", top_xyz, valid_top), ("bottom", bottom_xyz, valid_bottom)
    ):
        for index, component in enumerate(("alpha", "gamma", "beta")):
            actual = getattr(corrected, f"{component}_{name}")
            assert np.isnan(actual[~valid]).all()
            np.testing.assert_allclose(
                actual[valid], np.deg2rad(expected_xyz[valid, index]), atol=3e-12
            )
    assert np.isnan(corrected.alpha[5])  # neither shell was measured
    assert np.isfinite(corrected.alpha[4])  # the other shell remains usable
