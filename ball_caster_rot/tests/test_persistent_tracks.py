"""Persistent feature identities and independent keyframe image matching."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ballrot.track import (
    KLTConfig,
    PersistentKLTTracker,
    TrackMatches,
    detect_features,
    track_features,
)


def _texture() -> np.ndarray:
    rng = np.random.default_rng(31415)
    image = rng.integers(0, 256, size=(160, 240), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0.7)


def _translated(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return cv2.warpAffine(
        image,
        np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32),
        (image.shape[1], image.shape[0]),
    )


def _masks(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    top = np.zeros(shape, dtype=np.uint8)
    bottom = np.zeros(shape, dtype=np.uint8)
    top[20:70, 24:-24] = 255
    bottom[90:-20, 24:-24] = 255
    return {"top": top, "bottom": bottom}


def _config(**kwargs: object) -> KLTConfig:
    return KLTConfig(max_corners=60, min_distance_px=8.0, **kwargs)


def test_persistent_ids_follow_translated_pixels_and_are_unique_across_shells() -> None:
    first = _texture()
    second = _translated(first, 1.0, 1.0)
    third = _translated(first, 2.0, 2.0)
    masks = _masks(first.shape)
    tracker = PersistentKLTTracker(_config())
    tracker.initialize(first, masks)
    original = {name: tracker.points(name) for name in masks}
    assert len(set(original["top"][1]) & set(original["bottom"][1])) == 0

    pair_one = tracker.track_pair(first, second, masks, masks)
    pair_two = tracker.track_pair(second, third, masks, masks)
    for name in masks:
        initial_uv, initial_ids = original[name]
        first_matches, second_matches = pair_one[name], pair_two[name]
        common = np.intersect1d(first_matches.track_ids, second_matches.track_ids)
        assert len(common) >= 40
        initial_by_id = dict(zip(initial_ids, initial_uv))
        first_by_id = dict(zip(first_matches.track_ids, first_matches.uv_curr))
        second_by_id = dict(zip(second_matches.track_ids, second_matches.uv_curr))
        for track_id in common:
            np.testing.assert_allclose(
                second_by_id[track_id] - first_by_id[track_id], [1.0, 1.0], atol=0.05
            )
            np.testing.assert_allclose(
                second_by_id[track_id] - initial_by_id[track_id], [2.0, 2.0], atol=0.05
            )
        points, ids = tracker.points(name)
        assert len(ids) <= tracker.config.max_corners
        points[:] = -99
        ids[:] = -1
        fresh_points, fresh_ids = tracker.points(name)
        assert np.all(fresh_points >= 0) and np.all(fresh_ids >= 0)


def test_lost_tracks_are_reseeded_without_reusing_identities() -> None:
    first = _texture()
    masks = _masks(first.shape)
    hidden = {"top": np.zeros_like(masks["top"]), "bottom": masks["bottom"]}
    tracker = PersistentKLTTracker(_config())
    tracker.initialize(first, masks)
    original_ids = tracker.points("top")[1]
    lost = tracker.track_pair(first, first, masks, hidden)
    assert lost.top.count == 0
    assert lost.top.track_ids.shape == (0,)
    assert len(tracker.points("top")[1]) == 0

    exposed = tracker.track_pair(first, first, hidden, masks)
    assert exposed.top.count == 0
    new_points, new_ids = tracker.points("top")
    assert len(new_ids) == tracker.config.max_corners
    assert not np.any(np.isin(new_ids, original_ids))
    recovered = tracker.track_pair(first, first, masks, masks)
    np.testing.assert_array_equal(recovered.top.track_ids, new_ids)
    np.testing.assert_allclose(recovered.top.uv_prev, new_points)

    before_reset = np.concatenate([tracker.points(name)[1] for name in masks])
    tracker.initialize(first, masks)
    assert tracker.points("top")[1].min() > before_reset.max()


def test_pruning_rejects_tested_outliers_but_preserves_untested_reseeds() -> None:
    first = _texture()
    masks = _masks(first.shape)
    tracker = PersistentKLTTracker(_config())
    tracker.initialize(first, masks)
    changed_masks = {name: mask.copy() for name, mask in masks.items()}
    changed_masks["top"][:, :100] = 0
    matched = tracker.track_pair(first, first, masks, changed_masks).top
    current_uv, current_ids = tracker.points("top")
    untested = current_ids[~np.isin(current_ids, matched.track_ids)]
    assert len(untested) > 0
    allowed = matched.track_ids[::2]
    tracker.prune("top", allowed)
    after_uv, after_ids = tracker.points("top")
    assert set(after_ids) == set(allowed) | set(untested)
    assert len(after_uv) < len(current_uv)
    next_pair = tracker.track_pair(first, first, changed_masks, changed_masks)
    assert set(allowed).issubset(next_pair.top.track_ids)
    assert not np.any(np.isin(next_pair.top.track_ids, matched.track_ids[1::2]))


def test_initial_flow_aligns_images_independently_and_preserves_source_indices() -> None:
    first = _texture()
    displacement = np.array([17.0, -9.0])
    second = _translated(first, *displacement)
    mask = np.zeros(first.shape, dtype=np.uint8)
    mask[35:-35, 35:-40] = 255
    current_mask = np.zeros_like(mask)
    current_mask[20:-20, 125:-20] = 255
    config = _config(max_level=0, klt_win=15)
    points = detect_features(first, mask, config)
    # Offset guesses deliberately: returned pixels must come from LK alignment.
    guesses = points + displacement + np.array([0.4, -0.3])
    original_guesses = guesses.copy()
    result = track_features(
        first,
        second,
        points,
        prev_mask=mask,
        curr_mask=current_mask,
        initial_uv_curr=guesses,
        config=config,
    )
    assert result.count >= 15
    assert result.count < len(points)
    np.testing.assert_allclose(result.uv_prev, points[result.source_indices])
    np.testing.assert_allclose(
        result.uv_curr - result.uv_prev,
        np.broadcast_to(displacement, result.uv_curr.shape),
        atol=0.03,
    )
    np.testing.assert_array_equal(guesses, original_guesses)
    assert np.all(result.uv_curr[:, 0] >= 124.5)
    assert np.max(result.fb_error) < 0.03


def test_track_matches_ids_are_optional_and_checked_when_supplied() -> None:
    kwargs = dict(
        uv_prev=np.array([[5.0, 6.0]]),
        uv_curr=np.array([[6.0, 7.0]]),
        fb_error=np.array([0.1]),
        detected_count=1,
        forward_count=1,
        backward_count=1,
    )
    assert TrackMatches(**kwargs).track_ids is None
    np.testing.assert_array_equal(TrackMatches(**kwargs, track_ids=np.array([3])).track_ids, [3])
    for invalid in (np.array([1.2]), np.array([-1]), np.array([1, 2]), np.array([[1]])):
        with pytest.raises(ValueError, match="track_ids"):
            TrackMatches(**kwargs, track_ids=invalid)
    with pytest.raises(ValueError, match="unique"):
        TrackMatches(
            uv_prev=np.ones((2, 2)),
            uv_curr=np.ones((2, 2)),
            fb_error=np.zeros(2),
            detected_count=2,
            forward_count=2,
            backward_count=2,
            track_ids=np.array([3, 3]),
        )


def test_initial_flow_rejects_bad_shape_or_nonfinite_guesses() -> None:
    frame = _texture()
    points = np.array([[40.0, 40.0], [60.0, 60.0]])
    for invalid in (np.ones((1, 2)), np.array([[40.0, 40.0], [np.nan, 60.0]])):
        with pytest.raises(ValueError, match="initial_uv_curr"):
            track_features(frame, frame, points, initial_uv_curr=invalid)
