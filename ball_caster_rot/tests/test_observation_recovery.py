"""Image-connected recovery must not invent orientation across unseen gaps."""

from dataclasses import replace
import json

import numpy as np
from scipy.spatial.transform import Rotation

from ballrot.offline import OfflineConfig, SurfaceObservation
from ballrot.recovery import recover_observation_graph


K = np.array([[700., 0., 320.], [0., 700., 240.], [0., 0., 1.]])
C = np.array([0., 0., 4.])


def scene(count=14):
    rng = np.random.default_rng(72)
    xy = rng.uniform(-.38, .38, (40, 2))
    points = np.column_stack((xy, -np.sqrt(1 - np.sum(xy**2, axis=1))))
    poses = Rotation.from_euler('xyz', [[.4*i, -.3*i, .6*i] for i in range(count)], degrees=True).as_matrix()
    # A nonidentity global reference catches incorrect left/right composition.
    reference = Rotation.from_euler('xyz', [12, -8, 20], degrees=True).as_matrix()
    truth = poses @ reference
    observations = []
    # Short overlapping templates replace old ones throughout the sequence.
    for start in range(0, count - 2, 2):
        for frame in range(start, min(count, start + 5)):
            xyz = C + points @ poses[frame].T
            uvw = xyz @ K.T
            uv = uvw[:, :2] / uvw[:, 2:]
            observations.extend(SurfaceObservation(frame, start*100 + j, pixel, 2.)
                                for j, pixel in enumerate(uv))
    initial = np.repeat(truth[1:2], count, axis=0)
    initial[:2] = truth[:2]
    valid = np.zeros(count, bool)
    valid[:2] = True
    return observations, truth, initial, valid


def run(obs, poses, valid, **changes):
    settings = dict(enabled=True, graph_recovery_enabled=True, max_tracks_per_frame=120,
                    backward_window_frames=6, max_reprojection_px=.5, min_inlier_fraction=.85)
    settings.update(changes)
    return recover_observation_graph(obs, poses, valid, K, C, config=OfflineConfig(**settings))


def test_overlapping_templates_restore_a_long_tail_in_the_original_gauge():
    obs, truth, poses, valid = scene()
    result = run(obs, poses, valid)
    assert result.valid.all(), result.diagnostics
    np.testing.assert_allclose(result.rotations, truth, atol=1e-7)
    assert result.diagnostics['recovered_frames'] == len(truth) - 2
    assert all(len(frame['anchors']) >= 2 for frame in result.diagnostics['frames'])
    np.testing.assert_array_equal(valid, [True, True] + [False] * (len(truth) - 2))
    json.dumps(result.diagnostics, allow_nan=False)


def test_a_missing_image_stays_missing_but_observed_cross_gap_matches_can_recover_after_it():
    obs, truth, poses, valid = scene()
    obs = [entry for entry in obs if entry.frame_index != 5]
    result = run(obs, poses, valid)
    assert not result.valid[5]
    assert result.valid[6:].all(), result.diagnostics
    np.testing.assert_allclose(result.rotations[result.valid], truth[result.valid], atol=1e-7)
    np.testing.assert_array_equal(result.rotations[5], poses[5])


def test_disconnected_later_tracks_cannot_receive_a_global_reference():
    obs, _, poses, valid = scene()
    obs = [replace(entry, track_id=entry.track_id + (100000 if entry.frame_index >= 6 else 0))
           for entry in obs]
    result = run(obs, poses, valid)
    assert result.valid[:6].all()
    assert not result.valid[6:].any()
    np.testing.assert_array_equal(result.rotations[6:], poses[6:])


def test_one_trusted_image_is_insufficient_to_bootstrap_recovery():
    obs, _, poses, valid = scene()
    valid[1] = False
    result = run(obs, poses, valid)
    np.testing.assert_array_equal(result.valid, valid)


def test_inconsistent_trusted_references_are_not_averaged_into_measurements():
    obs, _, poses, valid = scene()
    poses[1] = Rotation.from_euler('y', 12, degrees=True).as_matrix() @ poses[1]
    result = run(obs, poses, valid)
    np.testing.assert_array_equal(result.valid, valid)


def test_recovery_off_preserves_inputs():
    obs, _, poses, valid = scene()
    result = run(obs, poses, valid, recover_invalid_frames=False)
    assert result.diagnostics['status'] == 'disabled'
    np.testing.assert_array_equal(result.valid, valid)
    np.testing.assert_array_equal(result.rotations, poses)
