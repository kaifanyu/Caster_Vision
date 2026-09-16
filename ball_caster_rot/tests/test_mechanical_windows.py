"""Window fits preserve a measured gauge and never invent a hidden shell spin."""
from dataclasses import replace
import json

import numpy as np
import pytest

from ballrot.mechanical import MechanicalConfig, refine_mechanical_trajectory
from ballrot.offline import OfflineConfig
from test_mechanical import K, C, F, scene, angular_error, assert_mechanics


def fit(data, initial, valid, *, window=9, overlap=3):
    return refine_mechanical_trajectory(
        data, initial, valid, K, C, F,
        config=MechanicalConfig(enabled=True, gap_fraction=.1,
                                window_frames=window, overlap_frames=overlap),
        offline_config=OfflineConfig(enabled=True, max_nfev=80))


def test_nonidentity_calibration_and_nonzero_overlap_angles_keep_global_pose():
    data, truth, initial, valid, alpha, beta = scene(count=24, noise=.01)
    result = fit(data, initial, valid)
    assert len(result.diagnostics["windows"]) >= 3
    assert result.diagnostics["status"] == "completed", result.diagnostics
    assert_mechanics(result)
    for name in data:
        assert result.valid[name].all(), result.diagnostics
        assert np.max(angular_error(result.rotations[name], truth[name])) < .07
        assert np.max(np.abs(result.beta[name] - beta[name])) < np.deg2rad(.07)
    assert np.max(np.abs(result.alpha-alpha)) < np.deg2rad(.07)
    assert result.diagnostics["windows"][1]["start_frame"] > 0
    json.dumps(result.diagnostics, allow_nan=False)


def test_bad_middle_images_do_not_erase_earlier_poses_or_observed_later_motion():
    data, truth, initial, valid, _, _ = scene(count=30, noise=.01)
    rng = np.random.default_rng(894)
    for name in data:
        data[name] = [replace(entry, uv=entry.uv+rng.normal(0, 18, 2))
                      if entry.frame_index in (10, 11) else entry for entry in data[name]]
    result = fit(data, initial, valid, window=10, overlap=4)
    for name in data:
        assert result.valid[name][:10].all(), result.diagnostics
        assert not result.valid[name][10:12].any()
        assert result.valid[name][12:].all(), result.diagnostics
        assert np.max(angular_error(result.rotations[name], truth[name])[result.valid[name]]) < .15
    assert_mechanics(result)


def test_disconnected_late_images_do_not_start_a_new_global_reference():
    data, _, initial, valid, _, _ = scene(count=30)
    for name in data:
        data[name] = [replace(entry, track_id=entry.track_id+(1000 if entry.frame_index >= 22 else 0))
                      for entry in data[name] if entry.frame_index < 8 or entry.frame_index >= 22]
    result = fit(data, initial, valid, window=10, overlap=3)
    for name in data:
        assert result.valid[name][:8].all(), result.diagnostics
        assert not result.valid[name][8:].any()
        assert result.diagnostics["gaps"][name][0]["start_frame"] == 8
    assert result.diagnostics["status"] == "stopped_at_unresolved_interval"
    # One successful window, then at most 10/5/3-frame retries of the gap.
    assert len(result.diagnostics["windows"]) <= 4


def test_one_shot_observation_iterators_are_materialized_before_validation():
    data, _, initial, valid, _, _ = scene(count=6)
    expected = fit(data, initial, valid, window=6, overlap=2)
    actual = fit({name: iter(entries) for name, entries in data.items()},
                 initial, valid, window=6, overlap=2)
    for name in data:
        assert np.array_equal(actual.valid[name], expected.valid[name])
        assert np.allclose(actual.rotations[name], expected.rotations[name])


def test_lost_shell_reference_cannot_be_replaced_by_other_shell_roll():
    data, _, initial, valid, _, _ = scene(count=24)
    data["bottom"] = [replace(entry, track_id=entry.track_id+(1000 if entry.frame_index >= 12 else 0))
                      for entry in data["bottom"] if entry.frame_index < 8 or entry.frame_index >= 12]
    result = fit(data, initial, valid)
    assert result.valid["top"].all(), result.diagnostics
    assert result.valid["bottom"][:8].all()
    assert not result.valid["bottom"][8:].any()
    assert np.isnan(result.beta["bottom"][8:]).all()
    assert np.isfinite(result.alpha).all()
    assert_mechanics(result)


def test_window_recovery_requires_observations_and_preserves_missing_shell_flags():
    data, truth, initial, valid, _, _ = scene(count=24)
    for name in data:
        valid[name][[10, 12]] = False
        initial[name][10] = initial[name][9]
        data[name] = [entry for entry in data[name] if entry.frame_index != 12]
    result = fit(data, initial, valid)
    for name in data:
        assert result.valid[name][10], result.diagnostics
        assert result.diagnostics["frames"][name][10]["status"] == "recovered"
        assert angular_error(result.rotations[name], truth[name])[10] < .1
        assert not result.valid[name][12]
    assert_mechanics(result)


def test_no_measured_first_image_cannot_acquire_origin_from_raw_later_poses():
    data, _, initial, valid, _, _ = scene(count=12)
    data = {name: [entry for entry in entries if entry.frame_index != 0]
            for name, entries in data.items()}
    result = fit(data, initial, valid)
    assert not result.valid["top"].any()
    assert not result.valid["bottom"].any()


@pytest.mark.parametrize("settings", [
    {"window_frames": True}, {"window_frames": -1}, {"window_frames": 2, "overlap_frames": 1},
    {"overlap_frames": 3}, {"window_frames": 10}, {"window_frames": 10, "overlap_frames": 10},
    {"window_frames": 10, "overlap_frames": 1.5},
])
def test_window_config_rejects_ambiguous_or_invalid_overlap(settings):
    with pytest.raises(ValueError):
        MechanicalConfig.from_mapping(settings)
