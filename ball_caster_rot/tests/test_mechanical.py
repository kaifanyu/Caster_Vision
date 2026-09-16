"""Mechanics are enforced while pixels, not animation plausibility, gate poses."""

from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ballrot.mechanical import MechanicalConfig, refine_mechanical_trajectory
from ballrot.offline import OfflineConfig, SurfaceObservation
from ballrot.rotation import Rx, Rz


K = np.array([[700., 0., 320.], [0., 700., 240.], [0., 0., 1.]])
C = np.array([0., 0., 4.])
F = Rotation.from_euler("XYZ", [90., 8., 15.], degrees=True).as_matrix() @ Rx(np.deg2rad(17))


def scene(count=10, landmarks=45, noise=.025, outliers=False, top_sign=1):
    rng = np.random.default_rng(884)
    alpha = np.linspace(0, .16, count)
    beta = {"top": np.linspace(0, .18, count), "bottom": np.linspace(0, -.2, count)}
    truth, initial, observations = {}, {}, {}
    for name, sign in (("top", top_sign), ("bottom", -top_sign)):
        x = rng.uniform(-.48, .48, landmarks)
        z = sign * rng.uniform(.2, .58, landmarks)
        points = np.column_stack((x, -np.sqrt(1 - x*x - z*z), z))
        truth[name] = np.array([F @ Rx(a) @ Rz(b) @ F.T for a, b in zip(alpha, beta[name])])
        initial[name] = []
        observations[name] = []
        for index, (a, b) in enumerate(zip(alpha, beta[name])):
            phase = index / (count - 1)
            perturbed = Rx(a + sign * phase * .014) @ Rotation.from_rotvec([0, sign * .012 * phase, 0]).as_matrix() @ Rz(b + sign * phase * .009)
            initial[name].append(F @ perturbed @ F.T)
            normals = (F @ Rx(a) @ Rz(b) @ points.T).T
            homogeneous = (C + normals) @ K.T
            uv = homogeneous[:, :2] / homogeneous[:, 2:]
            uv += rng.normal(0, noise, uv.shape)
            if outliers and index not in (0, count - 1):
                uv[:4] += rng.normal(0, 10, (4, 2))
            observations[name].extend(SurfaceObservation(index, point, pixel) for point, pixel in enumerate(uv))
        initial[name] = np.array(initial[name])
    valid = {name: np.ones(count, bool) for name in truth}
    return observations, truth, initial, valid, alpha, beta


def run(data, initial, valid, *, mechanical=None, top_sign=1, **options):
    return refine_mechanical_trajectory(data, initial, valid, K, C, F,
                                        config=mechanical or MechanicalConfig(enabled=True, gap_fraction=.1),
                                        offline_config=OfflineConfig(enabled=True, max_nfev=80, **options),
                                        top_shell_sign=top_sign)


def angular_error(first, second):
    return np.rad2deg(Rotation.from_matrix(first @ second.transpose(0, 2, 1)).magnitude())


def assert_mechanics(result):
    relative = {name: F.T @ poses @ F for name, poses in result.rotations.items()}
    assert np.allclose(relative["top"][:, :, 2], relative["bottom"][:, :, 2], atol=1e-12)
    for poses in result.rotations.values():
        assert np.allclose(poses @ poses.transpose(0, 2, 1), np.eye(3), atol=1e-12)
        assert np.allclose(np.linalg.det(poses), 1, atol=1e-12)
        assert np.allclose(poses[0], np.eye(3), atol=1e-12)
    for rotations in relative.values():
        assert np.max(np.abs(Rotation.from_matrix(rotations).as_euler("XYZ")[:, 1])) < 1e-10


def test_joint_pixels_recover_shared_roll_and_independent_counter_rotation():
    data, truth, initial, valid, alpha, beta = scene()
    result = run(data, initial, valid)
    assert result.diagnostics["status"] == "completed", result.diagnostics
    assert result.valid["top"].all(), result.diagnostics
    assert result.valid["bottom"].all(), result.diagnostics
    assert_mechanics(result)
    assert np.max(np.abs(result.alpha - alpha)) < np.deg2rad(.04)
    for name in truth:
        assert np.max(angular_error(result.rotations[name], truth[name])) < .05
        assert np.max(np.abs(result.beta[name] - beta[name])) < np.deg2rad(.05)
        assert np.median(angular_error(result.rotations[name], truth[name])) < np.median(angular_error(initial[name], truth[name])) / 10
    assert result.diagnostics["after"]["median_px"] < result.diagnostics["before"]["median_px"] / 8
    assert result.diagnostics["minimum_signed_landmark_z"] >= .1 - 1e-10
    assert result.diagnostics["gap_distance_sphere_units"] == pytest.approx(.2)
    json.dumps(result.diagnostics, allow_nan=False)


def test_missing_shell_spin_stays_unknown_while_shared_roll_is_measured():
    data, truth, initial, valid, alpha, _ = scene()
    data["bottom"] = [entry for entry in data["bottom"] if entry.frame_index != 5]
    valid["bottom"][5] = False
    result = run(data, initial, valid)
    assert result.valid["top"][5]
    assert not result.valid["bottom"][5]
    assert np.isfinite(result.alpha[5])
    assert abs(result.alpha[5] - alpha[5]) < np.deg2rad(.05)
    assert np.isnan(result.beta["bottom"][5])
    assert_mechanics(result)


def test_observed_invalid_pose_recovers_but_blank_image_does_not():
    data, truth, initial, valid, _, _ = scene()
    for name in ("top", "bottom"):
        valid[name][[4, 6]] = False
        initial[name][4] = initial[name][3]
        data[name] = [entry for entry in data[name] if entry.frame_index != 6]
    result = run(data, initial, valid)
    for name in ("top", "bottom"):
        assert result.valid[name][4], result.diagnostics
        assert angular_error(result.rotations[name], truth[name])[4] < .06
        assert not result.valid[name][6]
        assert result.diagnostics["frames"][name][4]["status"] == "recovered"
    assert np.isnan(result.alpha[6])


def test_robust_pixel_loss_handles_some_bad_matches():
    data, truth, initial, valid, _, _ = scene(outliers=True)
    result = run(data, initial, valid)
    for name in truth:
        assert result.valid[name].all(), result.diagnostics
        assert np.median(angular_error(result.rotations[name], truth[name])) < .2
    assert_mechanics(result)


def test_bad_frame_does_not_become_valid_by_mechanical_projection():
    data, _, initial, valid, _, _ = scene()
    rng = np.random.default_rng(578)
    for name in data:
        data[name] = [replace(entry, uv=entry.uv + rng.normal(0, 10, 2))
                      if entry.frame_index == 5 else entry for entry in data[name]]
    result = run(data, initial, valid)
    assert result.diagnostics["status"] == "completed", result.diagnostics
    for name in data:
        assert not result.valid[name][5]
        assert result.valid[name][[0, 1, 2, 7, 8, 9]].all(), result.diagnostics
    assert_mechanics(result)


def test_unobserved_shell_is_never_inferred_from_other_shell_spin():
    data, _, initial, valid, _, _ = scene()
    data["bottom"] = []
    result = run(data, initial, valid)
    assert result.valid["top"].all()
    assert not result.valid["bottom"].any()
    assert np.isnan(result.beta["bottom"]).all()
    assert np.isfinite(result.alpha).all()
    assert_mechanics(result)


def test_disconnected_untrusted_tracks_do_not_acquire_a_spin_reference():
    data, _, initial, valid, _, _ = scene()
    data["bottom"] = [replace(entry, track_id=entry.track_id + (1000 if entry.frame_index >= 5 else 0))
                      for entry in data["bottom"]]
    valid["bottom"][5:] = False
    result = run(data, initial, valid)
    assert not result.valid["bottom"][5:].any()
    assert result.valid["top"].all()


def test_reacquired_shell_does_not_pin_roll_measured_by_continuous_other_shell():
    data, truth, initial, valid, alpha, beta = scene(noise=0)
    data["bottom"] = [replace(entry, track_id=entry.track_id + (1000 if entry.frame_index >= 5 else 0))
                      for entry in data["bottom"]]
    initial["bottom"][5] = F @ Rx(alpha[5] + np.deg2rad(4)) @ Rz(beta["bottom"][5]) @ F.T
    result = run(data, initial, valid)
    assert result.diagnostics["roll_anchor_frames"] == [0]
    assert result.valid["top"].all(), result.diagnostics
    assert result.valid["bottom"].all(), result.diagnostics
    assert np.max(np.abs(result.alpha - alpha)) < np.deg2rad(.001)
    assert np.max(angular_error(result.rotations["top"], truth["top"])) < .001


def test_flipping_shell_assignment_requires_matching_cap_geometry():
    data, truth, initial, valid, _, _ = scene(top_sign=-1, landmarks=100)
    result = run(data, initial, valid, top_sign=-1)
    assert result.valid["top"].all(), result.diagnostics
    assert result.valid["bottom"].all(), result.diagnostics
    for name in truth:
        assert np.median(angular_error(result.rotations[name], truth[name])) < .06


def test_no_evidence_yields_only_invalid_constrained_placeholders():
    data, _, initial, valid, _, _ = scene()
    result = run({name: [] for name in data}, initial, valid)
    assert not result.valid["top"].any()
    assert not result.valid["bottom"].any()
    assert np.isnan(result.alpha).all()
    assert_mechanics(result)


def test_iteration_exhaustion_is_not_silently_accepted():
    data, _, initial, valid, _, _ = scene()
    result = refine_mechanical_trajectory(data, initial, valid, K, C, F,
                                          config=MechanicalConfig(enabled=True),
                                          offline_config=OfflineConfig(max_nfev=1))
    assert result.diagnostics["status"] == "solver_did_not_converge"
    assert not result.valid["top"].any()
    assert_mechanics(result)


def test_disabled_mode_preserves_original_results():
    data, _, initial, valid, _, _ = scene()
    result = run(data, initial, valid, mechanical=MechanicalConfig(enabled=False))
    for name in data:
        assert np.array_equal(result.rotations[name], initial[name])
        assert np.array_equal(result.valid[name], valid[name])


@pytest.mark.parametrize("settings", [{"enabled": 1}, {"gap_fraction": -.1}, {"gap_fraction": 1},
                                    {"gap_fraction": np.nan}, {"gap_fraction": True},
                                    {"gap_fraction": "0.1"}, {"unknown": 3}])
def test_invalid_config_is_rejected(settings):
    with pytest.raises(ValueError):
        MechanicalConfig.from_mapping(settings)


def test_invalid_geometry_and_trajectory_inputs_are_rejected():
    data, _, initial, valid, _, _ = scene()
    for kwargs in ({"F": np.diag([1, 1, -1])}, {"C": [0, 0, .5]}, {"radius": 0},
                   {"top_shell_sign": 0}, {"initial_valid": {"top": np.ones(2), "bottom": valid["bottom"]}}):
        arguments = dict(observations=data, initial_rotations=initial, initial_valid=valid,
                         K=K, C=C, F=F, config=MechanicalConfig(enabled=True))
        arguments.update(kwargs)
        with pytest.raises(ValueError):
            refine_mechanical_trajectory(**arguments)
