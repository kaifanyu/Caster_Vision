"""Rotation history metrics must preserve timing, frame conventions and losses."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.plot_motion_history import compute_history


def _accumulate(steps):
    """Construct the saved state, explicitly holding when a solve is absent."""
    states = [np.eye(3)]
    for step in steps:
        states.append(states[-1].copy() if step is None else step @ states[-1])
    return np.asarray(states)


def _payload(times, states, valid=None, frame=None):
    """Serialize full rotations without using the plotting/reorientation helpers."""
    times = np.asarray(times, float)
    frame = np.eye(3) if frame is None else np.asarray(frame, float)
    valid = np.ones(len(times), bool) if valid is None else np.asarray(valid, bool)
    angles = Rotation.from_matrix(frame.T @ states @ frame).as_euler("XYZ")
    frames = {"time_s": times.tolist(), "alpha_rad": angles[:, 0].tolist()}
    for shell in ("top", "bottom"):
        for column, component in enumerate(("alpha", "gamma", "beta")):
            frames[f"{component}_{shell}_rad"] = [
                float(value) if good else None
                for value, good in zip(angles[:, column], valid)
            ]
    quality = []
    for i in range(1, len(times)):
        record = {"frame_index": i}
        for shell in ("top", "bottom"):
            record[shell] = {
                "matched_count": 100, "usable_count": 95, "inlier_count": 90,
                "inlier_ratio": 90 / 95, "mean_residual_deg": .1,
                "median_residual_deg": .08, "median_fb_error_px": .2,
                "success": bool(valid[i]),
            }
        quality.append(record)
    return {
        "metadata": {"R_bc": frame.tolist(), "frame_count": len(times), "fps": 30},
        "frames": frames, "quality": quality,
    }


def test_known_spatial_rotation_uses_each_nonuniform_time_interval():
    times = np.array([0, .04, .17, .25])
    spatial_rate_deg_s = np.array([12., -7., 9.])
    states = Rotation.from_rotvec(np.deg2rad(times[:, None] * spatial_rate_deg_s)).as_matrix()
    history = compute_history(_payload(times, states))
    shell = history["shells"]["top"]
    speed = np.linalg.norm(spatial_rate_deg_s)
    np.testing.assert_allclose(history["dt_s"], [.04, .13, .08], atol=1e-14)
    np.testing.assert_allclose(history["mid_time_s"], [.02, .105, .21], atol=1e-14)
    np.testing.assert_allclose(
        shell["omega_camera_deg_s"], np.tile(spatial_rate_deg_s, (3, 1)), atol=1e-11
    )
    np.testing.assert_allclose(shell["speed_deg_s"], speed, atol=1e-11)
    np.testing.assert_allclose(shell["step_deg"], speed * np.diff(times), atol=1e-11)
    np.testing.assert_allclose(shell["net_deg"], speed * times, atol=1e-11)
    np.testing.assert_allclose(shell["observed_arc_deg"], speed * times, atol=1e-11)


def test_failed_interval_is_missing_but_later_local_step_remains_recoverable():
    steps = [
        Rotation.from_euler("x", 10, degrees=True).as_matrix(),
        None,
        Rotation.from_euler("y", 20, degrees=True).as_matrix(),
        Rotation.from_euler("z", 5, degrees=True).as_matrix(),
    ]
    valid = [True, True, False, True, True]
    states = _accumulate(steps)
    shell = compute_history(
        _payload([0, .1, .4, .5, .7], states, valid)
    )["shells"]["top"]
    np.testing.assert_array_equal(shell["valid_step"], valid)
    np.testing.assert_array_equal(shell["incomplete_history"], [False, False, True, True, True])
    np.testing.assert_allclose(shell["R_camera"][2], states[1], atol=1e-12)
    assert np.isnan(shell["net_deg"][2])
    assert np.isnan(shell["step_deg"][1])
    assert np.isnan(shell["speed_deg_s"][1])
    assert np.all(np.isnan(shell["omega_camera_deg_s"][1]))
    np.testing.assert_allclose(shell["step_deg"][[0, 2, 3]], [10, 20, 5], atol=1e-12)
    np.testing.assert_allclose(shell["omega_camera_deg_s"][2:], [[0, 200, 0], [0, 0, 25]], atol=1e-11)
    np.testing.assert_allclose(shell["observed_arc_deg"], [0, 10, 10, 30, 35], atol=1e-12)
    assert shell["status"].tolist() == [0, 2, 0, 0]


def test_noncommuting_steps_are_resolved_in_spatial_camera_axes():
    increments = Rotation.from_rotvec(np.deg2rad([[30, 0, 0], [0, 40, 0], [0, 0, -20]])).as_matrix()
    states = _accumulate(increments)
    times = np.array([0, .1, .35, .85])
    shell = compute_history(_payload(times, states))["shells"]["top"]
    expected_rates = np.array([[300, 0, 0], [0, 160, 0], [0, 0, -40]])
    np.testing.assert_allclose(shell["omega_camera_deg_s"], expected_rates, atol=1e-11)
    np.testing.assert_allclose(shell["R_camera"][-1], increments[2] @ increments[1] @ increments[0], atol=1e-12)
    # A body-relative product has a different second axis for these rotations.
    wrong_second = Rotation.from_matrix(states[1].T @ states[2]).as_rotvec()
    assert abs(wrong_second[2]) > .1


def test_changing_saved_ball_axes_preserves_all_camera_rotation_metrics():
    times = [0, .1, .25, .4]
    increments = Rotation.from_rotvec(np.deg2rad([[20, 3, -4], [-2, -15, 6], [7, 12, -8]])).as_matrix()
    states = _accumulate(increments)
    arbitrary_frame = Rotation.from_euler("xyz", [34, -26, 71], degrees=True).as_matrix()
    original = compute_history(_payload(times, states))["shells"]["top"]
    reexpressed = compute_history(_payload(times, states, frame=arbitrary_frame))["shells"]["top"]
    for key in ("R_camera", "net_deg", "step_deg", "observed_arc_deg", "speed_deg_s", "omega_camera_deg_s"):
        np.testing.assert_allclose(reexpressed[key], original[key], atol=1e-11)


def test_camera_frame_conjugation_rotates_vectors_and_preserves_scalar_metrics():
    times = [0, .1, .25, .4]
    increments = Rotation.from_rotvec(np.deg2rad([[20, 3, -4], [-2, -15, 6], [7, 12, -8]])).as_matrix()
    states = _accumulate(increments)
    change = Rotation.from_euler("xyz", [29, -38, 17], degrees=True).as_matrix()
    original = compute_history(_payload(times, states))["shells"]["top"]
    transformed = compute_history(_payload(times, change @ states @ change.T))["shells"]["top"]
    for key in ("net_deg", "step_deg", "observed_arc_deg", "speed_deg_s"):
        np.testing.assert_allclose(transformed[key], original[key], atol=1e-11)
    np.testing.assert_allclose(
        transformed["omega_camera_deg_s"], original["omega_camera_deg_s"] @ change.T, atol=1e-11
    )


def test_temporal_recovery_keeps_absolute_pose_without_inventing_a_gap_increment():
    times = [0.0, 0.1, 0.2, 0.3, 0.4]
    states = Rotation.from_euler("z", [0, 10, 20, 30, 40], degrees=True).as_matrix()
    pose_valid = [True, True, False, True, True]
    payload = _payload(times, states, valid=pose_valid)
    payload["temporal"] = {"config": {"enabled": True}}
    for name in ("top", "bottom"):
        payload["frames"][f"valid_{name}"] = pose_valid
        payload["quality"][1][name]["success"] = True  # rejected temporal pose
        payload["quality"][2][name]["success"] = False  # recovered absolute pose
    history = compute_history(payload)
    for shell in history["shells"].values():
        np.testing.assert_array_equal(shell["valid_pose"], pose_valid)
        np.testing.assert_array_equal(shell["valid_step"], [True, True, False, False, True])
        np.testing.assert_allclose(shell["R_camera"][3:], states[3:], atol=1e-12)
        np.testing.assert_allclose(shell["net_deg"][[0, 1, 3, 4]], [0, 10, 30, 40], atol=1e-12)
        assert np.all(np.isnan(shell["step_deg"][[1, 2]]))
        assert np.all(np.isnan(shell["omega_camera_deg_s"][[1, 2]]))
        np.testing.assert_allclose(shell["step_deg"][[0, 3]], [10, 10], atol=1e-12)
        np.testing.assert_allclose(shell["observed_arc_deg"], [0, 10, 10, 10, 20], atol=1e-12)
        assert shell["status"].tolist() == [0, 2, 2, 0]


def test_explicit_valid_flags_without_temporal_diagnostics_preserve_legacy_increments():
    steps = [Rotation.from_euler("x", 10, degrees=True).as_matrix(), None,
             Rotation.from_euler("z", 20, degrees=True).as_matrix()]
    flags = [True, True, False, True]
    payload = _payload([0, .1, .2, .3], _accumulate(steps), flags)
    for name in ("top", "bottom"):
        payload["frames"][f"valid_{name}"] = flags
    shell = compute_history(payload)["shells"]["top"]
    np.testing.assert_array_equal(shell["valid_step"], flags)
    np.testing.assert_allclose(shell["step_deg"][[0, 2]], [10, 20], atol=1e-12)
