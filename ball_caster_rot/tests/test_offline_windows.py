"""Bounded image fits preserve a global rotation gauge and isolate bad data."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ballrot.offline import OfflineConfig, SurfaceObservation, refine_trajectory


K = np.array([[700., 0., 320.], [0., 700., 240.], [0., 0., 1.]])
C = np.array([0., 0., 4.])


def scene(count=30):
    rng = np.random.default_rng(895)
    xy = rng.uniform(-.44, .44, (35, 2))
    points = np.column_stack((xy, -np.sqrt(1 - np.sum(xy**2, axis=1))))
    phase = np.linspace(0, 1, count)
    truth = Rotation.from_rotvec(np.column_stack((.08 * phase, .05*np.sin(phase*2), -.1*phase))).as_matrix()
    initial = Rotation.from_rotvec(phase[:, None] * np.deg2rad([1.1, -.8, .5])).as_matrix() @ truth
    observations = []
    for index, pose in enumerate(truth):
        xyz = C + points @ pose.T
        uvw = xyz @ K.T
        uv = uvw[:, :2] / uvw[:, 2:]
        observations.extend(SurfaceObservation(index, track, pixel)
                            for track, pixel in enumerate(uv))
    return observations, truth, initial


def error(first, second):
    return np.rad2deg(Rotation.from_matrix(first @ second.transpose(0, 2, 1)).magnitude())


def fit(observations, initial, valid=None, **settings):
    if valid is None:
        valid = np.ones(len(initial), dtype=bool)
    config = OfflineConfig(enabled=True, window_frames=14, window_overlap_frames=5,
                           max_nfev=60, **settings)
    return refine_trajectory(observations, initial, valid, K, C, config=config)


def test_windows_correct_bias_and_keep_accepted_overlap_fixed():
    observations, truth, initial = scene()
    result = fit(observations, initial)
    prefix = fit([entry for entry in observations if entry.frame_index < 14], initial[:14])
    assert result.valid.all()
    assert np.max(error(result.rotations, truth)) < .005
    np.testing.assert_allclose(result.rotations[:14], prefix.rotations, atol=1e-14)
    assert result.diagnostics["mode"] == "overlapping_windows"
    assert len(result.diagnostics["windows"]) >= 3
    assert result.diagnostics["windows"][1]["fixed_overlap_frames"] == list(range(9, 14))
    assert all(component["max_original_trusted_correction_deg"] <= 5.
               for component in result.diagnostics["components"] if component["accepted"])


def test_corrupted_image_does_not_reject_the_entire_sequence():
    observations, truth, initial = scene(count=36)
    rng = np.random.default_rng(912)
    observations = [replace(entry, uv=entry.uv + rng.normal(0., 20., 2))
                    if entry.frame_index == 16 else entry for entry in observations]
    original = initial.copy()
    result = fit(observations, initial)
    clean = np.arange(36) != 16
    assert np.count_nonzero([entry["status"] in ("refined", "anchor")
                            for entry in result.diagnostics["frames"]]) >= 30
    assert np.max(error(result.rotations[clean], truth[clean])) < .02
    np.testing.assert_array_equal(result.rotations[16], initial[16])
    assert result.valid[16]  # Retains the caller's valid forward estimate.
    assert result.diagnostics["frames"][16]["status"] == "rejected"
    assert any(16 in window["pruned_observation_frames"]
               for window in result.diagnostics["windows"])
    np.testing.assert_array_equal(initial, original)


def test_nonidentity_global_reference_is_preserved_in_every_window():
    observations, truth, initial = scene()
    gauge = Rotation.from_euler("XYZ", [23., -18., 11.], degrees=True).as_matrix()
    truth = truth @ gauge
    initial = initial @ gauge
    result = fit(observations, initial)
    np.testing.assert_allclose(result.rotations[0], gauge, atol=1e-14)
    assert np.max(error(result.rotations, truth)) < .005
    assert np.linalg.norm(result.rotations[-1] - np.eye(3)) > .2


def test_blank_gap_and_disconnected_untrusted_tail_are_never_fabricated():
    observations, _, initial = scene()
    observations = [replace(entry, track_id=entry.track_id + (1000 if entry.frame_index >= 20 else 0))
                    for entry in observations if entry.frame_index != 12]
    valid = np.ones(30, dtype=bool)
    valid[12] = False
    valid[20:] = False
    result = fit(observations, initial, valid)
    assert not result.valid[12]
    assert not result.valid[20:].any()
    np.testing.assert_array_equal(result.rotations[12], initial[12])
    np.testing.assert_array_equal(result.rotations[20:], initial[20:])
    assert result.diagnostics["recovered_frames"] == 0


def test_graph_recovery_counts_and_retains_measured_initialization_when_fit_rejects():
    observations, truth, initial = scene(count=10)
    valid = np.zeros(10, dtype=bool)
    valid[:2] = True
    initial[:] = truth  # Isolate graph provenance from initial drift.
    config = OfflineConfig(enabled=True, graph_recovery_enabled=True,
                           window_frames=8, window_overlap_frames=3, max_nfev=1)
    result = refine_trajectory(observations, initial, valid, K, C, config=config)
    assert result.valid.all(), result.diagnostics
    assert result.diagnostics["graph_recovery"]["recovered_frames"] == 8
    assert result.diagnostics["recovered_frames"] == 8
    assert np.max(error(result.rotations, truth)) < .001
    assert all(entry.get("pose_source", "").startswith("image_graph")
               for entry in result.diagnostics["frames"][2:])
    assert not valid[2:].any()


@pytest.mark.parametrize("settings", [
    {"window_frames": -1}, {"window_frames": True},
    {"window_frames": 2}, {"window_frames": 12, "window_overlap_frames": 12},
    {"window_frames": 12, "window_overlap_frames": 1},
    {"window_overlap_frames": -1}, {"graph_recovery_enabled": 1},
    {"graph_recovery_max_step_deg": 0}, {"graph_recovery_max_step_deg": 90},
])
def test_window_and_graph_options_are_validated(settings):
    with pytest.raises(ValueError):
        OfflineConfig(**settings)

