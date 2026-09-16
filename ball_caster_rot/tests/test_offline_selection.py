"""A bounded feature budget must preserve nearby-template image connections."""
from collections import Counter

import numpy as np

from ballrot.offline import OfflineConfig, SurfaceObservation, _select_observations


def test_known_material_points_can_be_reobserved_in_one_image():
    points = np.array([(x, y) for x in range(4) for y in range(4)], dtype=float)
    observations = [SurfaceObservation(12, i, point) for i, point in enumerate(points)]
    config = OfflineConfig()
    assert _select_observations(observations, config) == []
    selected = _select_observations(observations, config, known_track_ids=set(range(16)))
    assert len(selected) == 16
    assert {e.track_id for e in selected} == set(range(16))


def test_known_map_connections_receive_budget_with_many_new_templates():
    points = np.array([(x*20., y*20.) for x in range(6) for y in range(6)])
    observations = [SurfaceObservation(frame, i, point, 1.)
                    for frame in range(3) for i, point in enumerate(points)]
    observations += [SurfaceObservation(frame, 100+j, point, 2.)
                     for frame in range(3) for j, point in enumerate(points)]
    chosen = _select_observations(observations, OfflineConfig(max_tracks_per_frame=24),
                                  known_track_ids=set(range(36)))
    for frame in range(3):
        local = [e for e in chosen if e.frame_index == frame]
        assert len(local) <= 24
        assert sum(e.track_id < 36 for e in local) >= 12


def test_direct_map_priority_does_not_spend_the_adjacent_reservation():
    points = np.array([(x*20., y*20.) for x in range(10) for y in range(10)])
    observations = [SurfaceObservation(frame, i, point, 1.)
                    for frame in range(3) for i, point in enumerate(points[:50])]
    observations += [SurfaceObservation(frame, 100+i, point, 2.)
                     for frame in range(3) for i, point in enumerate(points)]
    chosen = _select_observations(observations, OfflineConfig(max_tracks_per_frame=100),
                                  known_track_ids=set(range(100, 200)))
    for frame in range(3):
        local = [e for e in chosen if e.frame_index == frame]
        assert len(local) <= 100
        assert sum(e.weight <= 1. for e in local) >= 24
        assert sum(e.track_id >= 100 for e in local) >= 24


def interleaved_tracks():
    pixels = np.array([(x, y) for x in np.linspace(200, 400, 8)
                       for y in np.linspace(100, 300, 8)])
    observations = []
    # Persistent tracks connect all images. Long direct tracks are available at
    # every third image; shorter direct templates also observe intervening ones.
    for track, pixel in enumerate(pixels):
        observations.extend(SurfaceObservation(frame, track, pixel, 1.) for frame in range(10))
        observations.extend(SurfaceObservation(frame, 1000+track, pixel, 2.)
                            for frame in (0, 3, 6, 9))
        for group, frames in enumerate(((0, 1, 3), (0, 2, 3), (3, 4, 6),
                                         (3, 5, 6), (6, 7, 9), (6, 8, 9))):
            observations.extend(SurfaceObservation(frame, 2000+100*group+track, pixel, 2.)
                                for frame in frames)
    return observations


def test_short_template_tracks_keep_observations_between_keyframes():
    observations = interleaved_tracks()
    config = OfflineConfig(max_tracks_per_frame=100)
    selected = _select_observations(observations, config)
    counts = Counter(entry.frame_index for entry in selected)
    tracks = Counter(entry.track_id for entry in selected)
    assert max(counts.values()) <= 100
    assert min(tracks.values()) >= config.min_track_length
    for frame in range(10):
        local = [entry for entry in selected if entry.frame_index == frame]
        assert len(local) >= config.min_observations_per_frame
        assert sum(entry.weight == 1. for entry in local) >= 24
        direct = [entry for entry in local if entry.weight > 1.]
        assert len(direct) >= config.min_observations_per_frame, (frame, len(direct))
        assert np.ptp(np.array([entry.uv for entry in direct]), axis=0).min() > 140.


def test_selection_is_order_independent_and_keeps_original_pixels():
    observations = interleaved_tracks()
    config = OfflineConfig(max_tracks_per_frame=60)
    selected = _select_observations(observations, config)
    shuffled = observations.copy()
    np.random.default_rng(82).shuffle(shuffled)
    repeated = _select_observations(shuffled, config)
    assert [(entry.frame_index, entry.track_id) for entry in selected] == [
        (entry.frame_index, entry.track_id) for entry in repeated]
    originals = {(entry.frame_index, entry.track_id): entry for entry in observations}
    assert all(entry is originals[(entry.frame_index, entry.track_id)] for entry in selected)


def test_no_budget_pressure_preserves_every_supported_observation():
    observations = [SurfaceObservation(frame, track, np.array([track, track%3], dtype=float))
                    for frame in range(4) for track in range(20)]
    # A two-image identity is not a valid landmark, even with free capacity.
    observations += [SurfaceObservation(frame, 100, np.array([5., 10.])) for frame in (0, 1)]
    selected = _select_observations(observations, OfflineConfig(max_tracks_per_frame=30))
    assert len(selected) == 80
    assert all(entry.track_id != 100 for entry in selected)
