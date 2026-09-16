"""Reverse observations must retain actual image evidence and material IDs."""

from types import SimpleNamespace

import numpy as np

from ballrot.estimate import solve_hemisphere_increment
from ballrot.offline import OfflineConfig
from ballrot.offline_observations import OfflineObservationCollector, save_observations
from ballrot.rotation import Rx, Rz
from ballrot.temporal import TemporalConfig
from ballrot.track import KLTConfig, TrackMatches


K = np.array([[400., 0., 128.], [0., 400., 128.], [0., 0., 1.]])
C = np.array([0., 0., 4.])
RADIUS_PX = 400. / np.sqrt(15.)
SOLVER = {"ransac_iters": 30, "ransac_inlier_deg": 1., "min_inliers": 8,
          "limb_cull_deg": 65.}


def points(pose=np.eye(3)):
    x, y = np.meshgrid(np.linspace(-.43, .43, 7), np.linspace(-.43, .43, 7))
    directions = np.column_stack((x.ravel(), y.ravel(),
                                  -np.sqrt(1 - x.ravel()**2 - y.ravel()**2)))
    homogeneous = (C + directions @ pose.T) @ K.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def pair(first, second, *, indices=None, ids=None):
    if indices is None:
        indices = np.arange(len(first))
    matches = TrackMatches(first[indices], second[indices], np.zeros(len(indices)),
                           len(first), len(first), len(first), source_indices=indices,
                           track_ids=None if ids is None else ids[indices])
    estimate = solve_hemisphere_increment(matches.uv_prev, matches.uv_curr, K, C,
                                          rng=3, **SOLVER)
    assert estimate.success
    return matches, estimate


def collector(**settings):
    config = OfflineConfig(enabled=True, **settings)
    return OfflineObservationCollector(config, K, C, 1., RADIUS_PX, KLTConfig(),
                                       SOLVER, TemporalConfig(enabled=True))


def push(coll, index, pose=np.eye(3), *, ids=None, valid=True, sharpness=50., accepted=True):
    uv = points(pose)
    if ids is None:
        ids = 100 + np.arange(len(uv))
    gray = np.full((256, 256), index, dtype=np.uint8)
    masks = {name: np.ones(gray.shape, dtype=bool) for name in ("top", "bottom")}
    features = {name: (uv.copy(), ids.copy()) for name in masks}
    poses = {name: pose.copy() for name in masks}
    records = {name: {"valid": valid, "adjacent": {"accepted": accepted},
                       "sharpness": sharpness, "sharpness_ratio": 1.} for name in masks}
    coll.push(index, gray, masks, features, poses, records)
    return gray, masks, features, poses, records


def test_adjacent_pixels_require_accepted_estimate_and_persistent_ids():
    coll = collector()
    uv = points()
    ids = np.arange(len(uv)) + 50
    matches, estimate = pair(uv, points(Rx(.02)), ids=ids)
    coll.add_adjacent("top", 1, matches, estimate, {"adjacent": {"accepted": False}})
    assert not coll.lists()["top"]
    coll.add_adjacent("top", 1, matches, estimate, {"adjacent": {"accepted": True}})
    assert len(coll.lists()["top"]) == 2*len(uv)
    assert {coll.landmark_sources[entry.track_id] for entry in coll.lists()["top"]} == {
        ("adjacent", 0, int(identifier)) for identifier in ids}


def test_anchor_source_indices_keep_template_identity_separate_from_chained_pixels():
    coll = collector()
    uv = points()
    ids = 200 + np.arange(len(uv))
    matches, estimate = pair(uv, points(Rx(.02)), ids=ids)
    coll.add_adjacent("top", 1, matches, estimate, {"adjacent": {"accepted": True}})
    indices = np.arange(len(uv))[::-2]
    direct, fit = pair(uv, points(Rx(.025)), indices=indices)
    key = SimpleNamespace(index=0, ids=ids)
    tracker = SimpleNamespace(last_anchor_observations=[(key, direct, fit)])
    coll.add_anchors("top", 1, tracker)
    stored = {(entry.frame_index, entry.track_id): entry for entry in coll.lists()["top"]}
    lookup = {source: identifier for identifier, source in coll.landmark_sources.items()}
    for match_index, source in enumerate(indices):
        value = stored[(1, lookup[("anchor", 0, int(ids[source]))])]
        assert value.weight == 2.
        assert np.array_equal(value.uv, direct.uv_curr[match_index])
        # The original chained point is separate evidence, not a material
        # coordinate overwritten by a different source-image observation.
        chained = stored[(1, lookup[("adjacent", 0, int(ids[source]))])]
        assert chained.weight == 1.
        assert np.array_equal(chained.uv, matches.uv_curr[source])


def test_later_keyframe_snapshot_cannot_redefine_an_earlier_template_point():
    coll = collector()
    ids = np.array([7])
    uv = lambda x, y: np.array([[float(x), float(y)]])
    coll._pair("top", 0, 3, uv(20, 30), uv(21, 31), ids, 2., family="anchor")
    # LK drift means the same persistent ID at frame 3 now seeds a different
    # material pixel. Its new template must remain a separate landmark.
    coll._pair("top", 3, 6, uv(24, 34), uv(25, 35), ids, 2., family="anchor")
    coll._pair("top", 0, 6, uv(20, 30), uv(22, 32), ids, 2., family="anchor")
    stored = {(entry.frame_index, coll.landmark_sources[entry.track_id]): entry
              for entry in coll.lists()["top"]}
    assert np.array_equal(stored[(3, ("anchor", 0, 7))].uv, [21., 31.])
    assert np.array_equal(stored[(3, ("anchor", 3, 7))].uv, [24., 34.])
    assert np.array_equal(stored[(0, ("anchor", 0, 7))].uv, [20., 30.])
    assert len({entry.track_id for entry in coll.lists()["top"]}) == 2


def test_reverse_matching_preserves_new_ids_and_maps_subset_indices(monkeypatch):
    coll = collector()
    push(coll, 0)
    pose = Rz(.03) @ Rx(.04)
    uv_current = points(pose)
    current_ids = 1000 + np.arange(len(uv_current))  # newly detected current landmarks
    indices = np.arange(len(uv_current))[::-2]
    calls = []

    def exact_image_observation(gray, past, uv, **kwargs):
        calls.append(kwargs["initial_uv_curr"].copy())
        # The sphere-pose projection seeds the earlier image correctly even
        # though these new IDs did not exist in the forward tracker then.
        assert np.allclose(kwargs["initial_uv_curr"], points(), atol=1e-4)
        return pair(uv, points(), indices=indices)[0]

    monkeypatch.setattr("ballrot.offline_observations.track_features", exact_image_observation)
    push(coll, 3, pose, ids=current_ids)
    assert len(calls) == 2
    for name in ("top", "bottom"):
        observations = coll.lists()[name]
        assert {coll.landmark_sources[entry.track_id] for entry in observations} == {
            ("reverse", 3, int(identifier)) for identifier in current_ids[indices]}
        assert {entry.frame_index for entry in observations} == {0, 3}
    assert all(item["accepted"] for item in coll.reverse_diagnostics)


def test_reverse_coherent_wrong_pattern_is_rejected_against_trusted_poses(monkeypatch):
    coll = collector()
    push(coll, 0)
    wrong_pixels = points(Rz(np.deg2rad(9.)))

    def wrong_pattern(gray, past, uv, **kwargs):
        return pair(uv, wrong_pixels)[0]

    monkeypatch.setattr("ballrot.offline_observations.track_features", wrong_pattern)
    push(coll, 3)
    assert all(not item["accepted"] for item in coll.reverse_diagnostics)
    assert all(item["forward_disagreement_deg"] > 8. for item in coll.reverse_diagnostics)
    assert not coll.lists()["top"]


def test_reverse_does_not_constrain_evidence_to_held_unresolved_pose(monkeypatch):
    coll = collector()
    push(coll, 0, valid=False)
    recovered_pixels = points(Rz(np.deg2rad(9.)))
    monkeypatch.setattr("ballrot.offline_observations.track_features",
                        lambda gray, past, uv, **kwargs: pair(uv, recovered_pixels)[0])
    push(coll, 3)
    assert all(item["accepted"] for item in coll.reverse_diagnostics)
    assert coll.lists()["top"]


def test_reverse_does_not_invent_pixels_in_heavily_blurred_target(monkeypatch):
    coll = collector()
    push(coll, 0, sharpness=.2)
    def unexpected_call(*args, **kwargs):
        raise AssertionError("blurred image should not supply observations")
    monkeypatch.setattr("ballrot.offline_observations.track_features", unexpected_call)
    push(coll, 3)
    assert not coll.reverse_diagnostics
    assert not coll.lists()["top"]


def test_buffer_is_bounded_and_does_not_alias_caller_arrays():
    coll = collector(backward_window_frames=3)
    gray, masks, features, poses, records = push(coll, 0, accepted=False)
    original = coll.buffer[-1].points["top"][0].copy()
    gray[:] = 255
    masks["top"][:] = False
    features["top"][0][:] = 0
    poses["top"][:] = 0
    records["top"]["sharpness"] = 0
    saved = coll.buffer[-1]
    assert np.all(saved.gray == 0)
    assert saved.masks["top"].all()
    assert np.array_equal(saved.points["top"][0], original)
    assert np.array_equal(saved.predictions["top"], np.eye(3))
    assert saved.records["top"]["sharpness"] == 50.
    for index in range(1, 8):
        push(coll, index, accepted=False)
    assert [frame.index for frame in coll.buffer] == [5, 6, 7]


def test_numeric_observation_archive_supports_empty_shell_and_round_trip(tmp_path):
    coll = collector()
    uv = points()
    ids = np.arange(len(uv))
    matches, estimate = pair(uv, points(Rx(.02)), ids=ids)
    coll.add_adjacent("top", 1, matches, estimate, {"adjacent": {"accepted": True}})
    poses = {name: np.repeat(np.eye(3)[None], 2, axis=0) for name in ("top", "bottom")}
    valid = {name: np.ones(2, dtype=bool) for name in poses}
    destination = save_observations(tmp_path / "observations.npz", coll.lists(),
                                    timestamps=np.array([0., .04]), K=K, center=C, radius=1.,
                                    initial_rotations=poses, initial_valid=valid,
                                    refined_rotations=poses, refined_valid=valid)
    with np.load(destination, allow_pickle=False) as data:
        assert data["bottom_uv"].shape == (0, 2)
        assert len(data["top_uv"]) == 2 * len(uv)
        assert np.array_equal(data["K"], K)
        assert np.array_equal(data["top_initial_rotations"], poses["top"])
