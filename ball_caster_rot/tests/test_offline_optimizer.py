"""Independent reprojection evidence corrects poses without inventing gaps."""

from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ballrot.offline import OfflineConfig, SurfaceObservation, refine_trajectory, _select_observations


K = np.array([[700., 0., 320.], [0., 700., 240.], [0., 0., 1.]])
C = np.array([0., 0., 4.])


def scene(count=12, landmarks=50, noise=0.06, outliers=False):
    rng = np.random.default_rng(138)
    xy = rng.uniform(-0.43, 0.43, (landmarks, 2))
    points = np.column_stack((xy, -np.sqrt(1 - np.sum(xy**2, axis=1))))
    truth = []
    observations = []
    for frame in range(count):
        phase = frame / max(count-1, 1)
        pose = Rotation.from_rotvec([0.08*phase, 0.06*np.sin(phase*2), -0.12*phase]).as_matrix()
        truth.append(pose)
        xyz = C + points @ pose.T
        uvw = xyz @ K.T
        uv = uvw[:, :2] / uvw[:, 2:]
        uv += rng.normal(0, noise, uv.shape)
        if outliers and frame not in (0, count - 1):
            uv[:5] += rng.normal(0, 12, (5, 2))
        observations.extend(SurfaceObservation(frame, i, pixel) for i, pixel in enumerate(uv))
    truth = np.array(truth)
    bias = Rotation.from_rotvec(np.linspace(0, 1, count)[:, None] * np.deg2rad([1.5, -1., .7])).as_matrix()
    initial = bias @ truth
    return observations, truth, initial


def angle_errors(poses, truth):
    return np.rad2deg(Rotation.from_matrix(poses @ truth.transpose(0, 2, 1)).magnitude())


def run(observations, initial, valid=None, **settings):
    if valid is None:
        valid = np.ones(len(initial), dtype=bool)
    config = OfflineConfig(enabled=True, **settings)
    return refine_trajectory(observations, initial, valid, K, C, config=config)


def test_batch_reprojection_reduces_noncommuting_pose_bias_against_truth():
    observations, truth, initial = scene()
    result = run(observations, initial)
    assert result.diagnostics["optimized_components"] == 1, result.diagnostics
    assert np.median(angle_errors(result.rotations, truth)) < 0.06
    assert np.median(angle_errors(result.rotations, truth)) < np.median(angle_errors(initial, truth)) / 10
    assert np.array_equal(result.rotations[0], initial[0])
    report = result.diagnostics["components"][0]
    assert report["after"]["median_px"] < report["before"]["median_px"] / 5
    assert result.valid.all()
    json.dumps(result.diagnostics, allow_nan=False)


def test_robust_fit_resists_inconsistent_pixel_outliers():
    observations, truth, initial = scene(outliers=True)
    result = run(observations, initial, max_nfev=60)
    assert result.diagnostics["optimized_components"] == 1, result.diagnostics
    assert np.median(angle_errors(result.rotations, truth)) < 0.15
    assert min(frame.get("inlier_fraction", 1) for frame in result.diagnostics["frames"]) < 1


def test_missing_pose_recovers_only_with_real_cross_frame_observations():
    observations, truth, initial = scene()
    valid = np.ones(len(initial), dtype=bool)
    valid[5] = False
    valid[7] = False
    initial[[5, 7]] = initial[[4, 6]]
    observations = [entry for entry in observations if entry.frame_index != 7]
    result = run(observations, initial, valid)
    assert result.valid[5]
    assert not result.valid[7]
    assert angle_errors(result.rotations, truth)[5] < .08
    assert np.array_equal(result.rotations[7], initial[7])
    assert result.diagnostics["frames"][5]["status"] == "recovered"
    assert result.diagnostics["frames"][7]["status"] == "unresolved"


def test_disconnected_untrusted_component_does_not_acquire_global_orientation():
    observations, truth, initial = scene(count=10)
    observations = [replace(entry, track_id=entry.track_id + (1000 if entry.frame_index >= 5 else 0))
                    for entry in observations]
    valid = np.array([True]*5 + [False]*5)
    result = run(observations, initial, valid)
    assert result.diagnostics["optimized_components"] == 1
    assert len(result.diagnostics["excluded_invalid_frames"]) == 5
    assert not result.valid[5:].any()
    assert np.array_equal(result.rotations[5:], initial[5:])


def test_unsupported_invalid_tail_does_not_block_connected_trusted_pose_refinement():
    observations, truth, initial = scene(count=6, landmarks=45)
    # Frame 2 sees both generations of landmarks and connects the entire
    # observation graph. New points after it have only one trusted view.
    observations = ([entry for entry in observations if entry.frame_index <= 2]
                    + [replace(entry, track_id=entry.track_id + 1000)
                       for entry in observations if entry.frame_index >= 2])
    valid = np.array([True]*3 + [False]*3)
    result = run(observations, initial, valid)
    assert result.diagnostics["optimized_components"] == 1
    assert np.median(angle_errors(result.rotations, truth)[:3]) < .05
    assert np.array_equal(result.rotations[3:], initial[3:])
    assert not result.valid[3:].any()
    assert {entry["frame_index"] for entry in result.diagnostics["excluded_invalid_frames"]} == {3, 4, 5}
    assert all(result.diagnostics["frames"][frame]["reason"] ==
               "invalid_frame_without_independent_landmark_support" for frame in (3, 4, 5))


def test_disabling_recovery_still_refines_supported_valid_poses():
    observations, truth, initial = scene()
    valid = np.ones(len(initial), dtype=bool)
    valid[5] = False
    result = run(observations, initial, valid, recover_invalid_frames=False)
    assert result.diagnostics["optimized_components"] == 1
    assert np.median(angle_errors(result.rotations, truth)[valid]) < .05
    assert not result.valid[5]
    assert np.array_equal(result.rotations[5], initial[5])
    assert result.diagnostics["frames"][5]["reason"] == "invalid_frame_recovery_disabled"


def test_invalid_frame_with_concentrated_support_does_not_block_good_poses():
    observations, truth, initial = scene()
    observations = [replace(entry, uv=np.array([320., 240.]) +
                            (entry.uv - [320., 240.]) * .001)
                    if entry.frame_index == 5 else entry for entry in observations]
    valid = np.ones(len(initial), dtype=bool)
    valid[5] = False
    result = run(observations, initial, valid)
    assert result.diagnostics["optimized_components"] == 1
    assert np.median(angle_errors(result.rotations, truth)[valid]) < .05
    assert not result.valid[5]
    assert result.diagnostics["frames"][5]["reason"] == "invalid_frame_without_spread_landmark_support"


def test_one_shared_point_does_not_establish_a_complete_orientation_reference():
    observations, _, initial = scene(count=10)
    observations = [replace(entry, track_id=entry.track_id + (
        1000 if entry.frame_index >= 5 and entry.track_id != 0 else 0))
        for entry in observations]
    valid = np.array([True]*5 + [False]*5)
    result = run(observations, initial, valid)
    assert result.diagnostics["optimized_components"] == 1
    assert not result.valid[5:].any()
    assert np.array_equal(result.rotations[5:], initial[5:])


def test_outlier_bridges_cannot_establish_global_orientation_between_good_groups():
    original, _, initial = scene(count=6, landmarks=70)
    rng = np.random.default_rng(851)
    observations = []
    for entry in original:
        # Each half has many mutually consistent local landmarks, but no true
        # material identities crossing the group boundary.
        observations.append(replace(entry, track_id=entry.track_id +
                                    (100 if entry.frame_index >= 3 else 0)))
        if entry.track_id < 15:
            # Wrong persistent associations form a well-spread apparent bridge
            # before fitting; their inconsistent pixels become final outliers.
            observations.append(replace(entry, track_id=1000+entry.track_id,
                                        uv=entry.uv + rng.uniform(-12., 12., 2)))
    result = run(observations, initial, max_nfev=100, max_pose_change_deg=30.)
    assert result.diagnostics["optimized_components"] == 0
    report = result.diagnostics["components"][0]
    assert report["reason"] == "disconnected_inlier_graph", report
    assert len(report["inlier_graph_components"]) >= 2
    assert all(frame["inlier_fraction"] >= .7 for frame in report["frames"])
    assert np.array_equal(result.rotations, initial)


@pytest.mark.parametrize("settings,reason", [
    ({"max_nfev": 1}, "solver_did_not_converge"),
    ({"max_pose_change_deg": .1}, "pose_correction_exceeds_limit"),
])
def test_rejected_fit_preserves_input_poses(settings, reason):
    observations, _, initial = scene()
    result = run(observations, initial, **settings)
    assert np.array_equal(result.rotations, initial)
    assert result.diagnostics["components"][0]["reason"] == reason


def test_concentrated_features_are_not_accepted_as_reliable_pose_evidence():
    observations, _, initial = scene()
    observations = [replace(entry, uv=np.array([320., 240.]) + (entry.uv - [320., 240.]) * .001)
                    for entry in observations]
    result = run(observations, initial)
    assert result.diagnostics["optimized_components"] == 0
    assert np.array_equal(result.rotations, initial)


def test_disabled_empty_and_short_inputs_preserve_data():
    observations, _, initial = scene(count=2)
    valid = np.ones(2, dtype=bool)
    disabled = refine_trajectory(observations, initial, valid, K, C)
    assert disabled.diagnostics["status"] == "disabled"
    short = run(observations, initial)
    assert short.diagnostics["status"] == "insufficient_observations"
    empty = run([], np.empty((0, 3, 3)))
    assert empty.rotations.shape == (0, 3, 3)
    assert np.array_equal(short.rotations, initial)


def test_duplicate_observations_use_highest_weight_and_selection_is_bounded():
    observations, _, initial = scene(landmarks=90)
    duplicates = [replace(entry, uv=entry.uv + .1, weight=.5) for entry in observations]
    result = run(observations + duplicates, initial, max_tracks_per_frame=35)
    assert result.diagnostics["selected_observation_count"] <= 35 * len(initial)
    assert result.diagnostics["optimized_components"] == 1, result.diagnostics


def test_long_chained_tracks_cannot_crowd_out_short_fixed_image_templates():
    observations = []
    x, y = np.meshgrid(np.linspace(220., 420., 8), np.linspace(140., 340., 8))
    pixels = np.column_stack((x.ravel(), y.ravel()))
    for frame in range(10):
        for track, uv in enumerate(pixels):
            observations.append(SurfaceObservation(frame, track, uv, weight=1.))
            if frame < 3:
                observations.append(SurfaceObservation(frame, track + 1000, uv, weight=2.))
    selected = _select_observations(observations, OfflineConfig(max_tracks_per_frame=40))
    first_image = [entry for entry in selected if entry.frame_index == 0]
    assert len(first_image) == 40
    direct = [entry for entry in first_image if entry.track_id >= 1000]
    adjacent = [entry for entry in first_image if entry.track_id < 1000]
    assert len(direct) == 20
    assert len(adjacent) == 20
    # Both sources span the image; adjacent tracks connect the direct-template
    # images to later frames having only ordinary forward observations.
    for group in (direct, adjacent):
        assert np.ptp(np.array([entry.uv for entry in group]), axis=0).min() > 150.
    assert len([entry for entry in selected if entry.frame_index == 9]) == 40


def test_source_quota_fills_unused_capacity_without_exceeding_image_budget():
    observations = []
    pixels = np.array([(x, y) for x in range(200, 401, 20) for y in range(150, 351, 20)], dtype=float)
    for frame in range(3):
        for track, uv in enumerate(pixels):
            observations.append(SurfaceObservation(frame, track, uv, weight=2.))
        for track, uv in enumerate(pixels[:5]):
            observations.append(SurfaceObservation(frame, 1000+track, uv, weight=1.))
    selected = _select_observations(observations, OfflineConfig(max_tracks_per_frame=40))
    for frame in range(3):
        image = [entry for entry in selected if entry.frame_index == frame]
        assert len(image) == 40
        assert sum(entry.weight == 1. for entry in image) == 5
        assert sum(entry.weight == 2. for entry in image) == 35


@pytest.mark.parametrize("settings", [
    {"enabled": 1}, {"min_track_length": 2}, {"max_nfev": 0},
    {"robust_loss_scale_px": np.nan}, {"min_inlier_fraction": 1.1},
    {"max_tracks_per_frame": 5}, {"max_pose_change_deg": 100},
    {"recover_invalid_frames": 0}, {"unknown": 0},
    {"backward_window_frames": 2, "backward_stride": 3},
    {"rate_polynomial_order": 4}, {"rate_window_s": 0},
])
def test_offline_config_rejects_invalid_values(settings):
    with pytest.raises(ValueError):
        OfflineConfig.from_mapping(settings)


def test_observation_and_trajectory_validation():
    with pytest.raises(ValueError):
        SurfaceObservation(0, 1, np.array([np.nan, 2]))
    with pytest.raises(ValueError):
        run([SurfaceObservation(1, 1, np.array([320., 240.]))], np.eye(3)[None])
    with pytest.raises(ValueError):
        run([], np.zeros((2, 3, 3)))
