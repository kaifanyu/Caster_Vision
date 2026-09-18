"""Recovery and drift-correction gates for nearby image anchors."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from ballrot.estimate import solve_hemisphere_increment
from ballrot.rotation import Rx, Rz, geodesic_angle
from ballrot.segment import SegmentationMasks
from ballrot.temporal import (
    TemporalConfig,
    TemporalRotationTracker,
    estimate_quality,
    safe_tracking_masks,
    temporal_report,
)
from ballrot.track import KLTConfig, TrackMatches


K = np.array([[400.0, 0.0, 128.0], [0.0, 400.0, 128.0], [0.0, 0.0, 1.0]])
CENTER = np.array([0.0, 0.0, 4.0])
RADIUS_PX = 400.0 / np.sqrt(15.0)
SOLVER = {"ransac_iters": 30, "ransac_inlier_deg": 1.0, "min_inliers": 8,
          "limb_cull_deg": 65.0}


def _directions() -> np.ndarray:
    x, y = np.meshgrid(np.linspace(-0.45, 0.45, 8), np.linspace(-0.45, 0.45, 8))
    x, y = x.ravel(), y.ravel()
    return np.column_stack((x, y, -np.sqrt(1.0 - x*x - y*y)))


def _ry(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _pixels(pose: np.ndarray) -> np.ndarray:
    points = CENTER + _directions() @ pose.T
    homogeneous = points @ K.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def _pair(previous: np.ndarray, current: np.ndarray):
    uv_prev, uv_curr = _pixels(previous), _pixels(current)
    count = len(uv_prev)
    matches = TrackMatches(uv_prev, uv_curr, np.zeros(count), count, count, count)
    estimate = solve_hemisphere_increment(uv_prev, uv_curr, K, CENTER, 1.0,
                                         rng=1, **SOLVER)
    assert estimate.success
    return matches, estimate


def _tracker(**config_overrides):
    config = TemporalConfig(enabled=True, **config_overrides)
    tracker = TemporalRotationTracker(K, CENTER, 1.0, RADIUS_PX, KLTConfig(), SOLVER, config)
    gray = np.random.default_rng(2).integers(0, 255, (256, 256), dtype=np.uint8)
    mask = np.ones(gray.shape, dtype=bool)
    uv, ids = _pixels(np.eye(3)), np.arange(len(_directions()))
    tracker.initialize(gray, mask, uv, ids)
    assert len(tracker.keyframes) == 1
    return tracker, gray, mask, ids


def _direct_observer(tracker, truth, *, offsets=None):
    """Independent, exact pixel observations, including projection/unprojection."""
    offsets = offsets or {}

    def observe(key, gray, mask, prediction, uv, ids):
        pose = Rz(np.deg2rad(offsets.get(key.index, 0.0))) @ truth
        matches, estimate = _pair(key.pose, pose)
        quality = estimate_quality(matches, estimate, RADIUS_PX, tracker.config)
        quality["keyframe"] = key.index
        return matches, estimate, quality

    return observe


@pytest.mark.parametrize("settings", [
    {"enabled": 1}, {"correction_interval": 0}, {"boundary_margin_px": -1},
    {"min_inliers": 2}, {"min_inlier_ratio": 1.1}, {"max_residual_deg": np.nan},
    {"keyframe_interval": 46, "max_keyframe_age": 45}, {"unknown_setting": True},
])
def test_temporal_config_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        TemporalConfig.from_mapping(settings)


@pytest.mark.parametrize("layout", ["collinear", "concentrated"])
def test_quality_rejects_geometrically_weak_feature_distribution(layout):
    matches, estimate = _pair(np.eye(3), Rz(0.01))
    if layout == "collinear":
        pixels = np.column_stack((np.linspace(70.0, 185.0, matches.count),
                                  np.full(matches.count, 128.0)))
    else:
        pixels = 128.0 + 0.01 * (matches.uv_curr - 128.0)
    matches = replace(matches, uv_curr=pixels)
    quality = estimate_quality(matches, estimate, RADIUS_PX, TemporalConfig())
    assert not quality["accepted"]
    assert "features too concentrated or collinear" in quality["reasons"]


def test_quality_rejects_high_residual_and_poor_match_consensus():
    matches, estimate = _pair(np.eye(3), Rz(0.01))
    good = estimate_quality(matches, estimate, RADIUS_PX, TemporalConfig())
    assert good["accepted"]
    noisy = replace(estimate, residual_rad=np.full(matches.count, np.deg2rad(0.7)))
    assert "high rotation residual" in estimate_quality(
        matches, noisy, RADIUS_PX, TemporalConfig())["reasons"]
    inliers = np.arange(matches.count) < matches.count // 2
    weak = replace(estimate, inlier_mask=inliers)
    assert "low inlier agreement" in estimate_quality(
        matches, weak, RADIUS_PX, TemporalConfig())["reasons"]


def test_boundary_margin_excludes_occluder_and_silhouette_without_mutating_masks():
    ball = np.zeros((32, 32), dtype=bool)
    ball[2:30, 2:30] = True
    yoke = np.zeros_like(ball)
    yoke[8:12, 12:16] = True
    top = ball.copy()
    top[16:] = False
    top[yoke] = False
    bottom = ball.copy()
    bottom[:16] = False
    original = SegmentationMasks(ball, top, bottom, yoke)
    safe = safe_tracking_masks(original, 2)
    assert safe.top[5, 5]
    assert not safe.top[2, 5]  # silhouette border
    assert not safe.top[7, 13]  # adjacent to occluder
    assert not safe.top[15, 5]  # hemisphere separation
    assert original.top[7, 13]
    np.testing.assert_array_equal(safe.yoke, original.yoke)
    np.testing.assert_array_equal(safe.ball, original.ball)
    assert safe_tracking_masks(original, 0) is original


def test_keyframe_pixels_correct_biased_increment_chain(monkeypatch):
    tracker, gray, mask, ids = _tracker(correction_interval=3)
    uncorrected = np.eye(3)
    for index in range(1, 4):
        previous = Rz(np.deg2rad(index - 1))
        truth = Rz(np.deg2rad(index))
        matches, estimate = _pair(previous, truth)
        biased = replace(estimate, R=Rz(np.deg2rad(1.5)))
        uncorrected = biased.R @ uncorrected
        monkeypatch.setattr(tracker, "_observe", _direct_observer(tracker, truth))
        pose, valid, record = tracker.update(index, gray, mask, matches, biased,
                                              _pixels(truth), ids)
        assert valid
    assert record["status"] == "keyframe"
    assert record["used_keyframes"] == [0]
    assert np.rad2deg(geodesic_angle(uncorrected, truth)) == pytest.approx(1.5)
    assert np.rad2deg(geodesic_angle(pose, truth)) < 1e-5
    assert record["correction_deg"] == pytest.approx(1.5, abs=1e-5)


def test_multiple_noncommuting_keyframes_refine_in_the_original_camera_frame(monkeypatch):
    tracker, gray, mask, ids = _tracker(correction_interval=2, keyframe_interval=1)
    first_pose = Rx(0.06) @ _ry(-0.04) @ Rz(0.02)
    matches, estimate = _pair(np.eye(3), first_pose)
    pose, valid, _ = tracker.update(1, gray, mask, matches, estimate,
                                    _pixels(first_pose), ids)
    assert valid
    assert [key.index for key in tracker.keyframes] == [0, 1]

    # A noncommuting camera-space increment makes right/left composition and
    # the conversion of keyframe directions back to frame zero observable.
    increment = _ry(0.05) @ Rx(-0.03) @ Rz(0.025)
    truth = increment @ first_pose
    matches, estimate = _pair(first_pose, truth)
    biased = replace(estimate, R=Rx(0.012) @ estimate.R)
    monkeypatch.setattr(tracker, "_observe", _direct_observer(tracker, truth))
    pose, valid, record = tracker.update(2, gray, mask, matches, biased,
                                         _pixels(truth), ids)
    assert valid and record["status"] == "keyframe"
    assert record["used_keyframes"] == [0, 1]
    assert np.rad2deg(geodesic_angle(pose, truth)) < 1e-5
    assert np.rad2deg(geodesic_angle(first_pose @ increment, truth)) > 0.1


def test_unresolved_interval_requires_anchor_and_recovery_preserves_all_motion(monkeypatch):
    tracker, gray, mask, ids = _tracker(correction_interval=20)
    matches, estimate = _pair(np.eye(3), Rz(np.deg2rad(1.0)))
    pose, valid, _ = tracker.update(1, gray, mask, matches, estimate,
                                    _pixels(Rz(np.deg2rad(1.0))), ids)
    assert valid
    last_good = pose.copy()

    # One blank frame loses the observation. A later sharp pair alone has no
    # information about the rotation that occurred during that missing frame.
    failure = replace(estimate, R=None, inlier_mask=np.zeros(matches.count, dtype=bool),
                      failure_reason="no texture")
    pose, valid, record = tracker.update(2, np.zeros_like(gray), mask, matches, failure,
                                         _pixels(Rz(np.deg2rad(2.0))), ids)
    assert not valid and record["status"] == "unresolved"
    np.testing.assert_allclose(pose, last_good)

    def unavailable(key, gray, mask, prediction, uv, ids):
        return matches, failure, {"accepted": False, "reasons": ["anchor hidden"],
                                  "keyframe": key.index}

    monkeypatch.setattr(tracker, "_observe", unavailable)
    pair3, estimate3 = _pair(Rz(np.deg2rad(2.0)), Rz(np.deg2rad(3.0)))
    pose, valid, record = tracker.update(3, gray, mask, pair3, estimate3,
                                         _pixels(Rz(np.deg2rad(3.0))), ids)
    assert not valid and record["status"] == "unresolved"
    np.testing.assert_allclose(pose, last_good)

    truth = Rz(np.deg2rad(4.0))
    monkeypatch.setattr(tracker, "_observe", _direct_observer(tracker, truth))
    pair4, estimate4 = _pair(Rz(np.deg2rad(3.0)), truth)
    pose, valid, record = tracker.update(4, gray, mask, pair4, estimate4,
                                         _pixels(truth), ids)
    assert valid and record["status"] == "recovered"
    assert np.rad2deg(geodesic_angle(pose, truth)) < 1e-5
    assert np.rad2deg(geodesic_angle(pose, last_good)) == pytest.approx(3.0, abs=1e-5)
    report = temporal_report(tracker.config, {"top": tracker})
    assert report["summary"]["top"]["unresolved_frames"] == [2, 3]
    assert report["summary"]["top"]["status_counts"]["recovered"] == 1


def test_gradually_changing_contrast_keeps_refreshing_nearby_keyframes(monkeypatch):
    tracker, gray, mask, ids = _tracker(correction_interval=3, keyframe_interval=4,
                                      max_keyframe_age=12)
    identity = np.eye(3)
    matches, estimate = _pair(identity, identity)
    monkeypatch.setattr(tracker, "_observe", _direct_observer(tracker, identity))
    for index in range(1, 49):
        # The visible texture remains sharp while its contrast slowly falls.
        # A reference tied only to old anchors can prevent every later refresh.
        contrast = np.rint(128.0 + 0.98**index * (gray.astype(float) - 128.0)).astype(np.uint8)
        _, valid, record = tracker.update(index, contrast, mask, matches, estimate,
                                          _pixels(identity), ids)
        assert valid
        assert "blur or insufficient texture" not in record["adjacent"]["reasons"]
    assert tracker.keyframes[-1].index >= 44
    assert all(index >= 36 for index in record["keyframes"])
    assert record["sharpness"] < 0.2 * tracker.history[1]["sharpness"]


def test_weak_motion_prediction_seeds_search_but_cannot_resume_pose_without_anchor(monkeypatch):
    tracker, gray, mask, ids = _tracker(correction_interval=20)
    matches, estimate = _pair(np.eye(3), Rz(np.deg2rad(1.0)))
    last_good, valid, _ = tracker.update(1, gray, mask, matches, estimate,
                                         _pixels(Rz(np.deg2rad(1.0))), ids)
    assert valid
    weak = replace(estimate, residual_rad=np.full(matches.count, np.deg2rad(0.9)))
    search_predictions = []

    def unavailable(key, gray, mask, prediction, uv, ids):
        search_predictions.append(prediction.copy())
        return matches, weak, {"accepted": False, "reasons": ["anchor hidden"],
                               "keyframe": key.index}

    monkeypatch.setattr(tracker, "_observe", unavailable)
    for index in (2, 3, 4):
        step_estimate = weak if index < 4 else estimate
        pose, valid, record = tracker.update(index, gray, mask, matches, step_estimate,
                                              _pixels(Rz(np.deg2rad(index))), ids)
        assert not valid and record["status"] == "unresolved"
        np.testing.assert_allclose(pose, last_good)
        assert np.rad2deg(geodesic_angle(search_predictions[-1],
                                        Rz(np.deg2rad(index)))) < 1e-5

    # The anchor corrects provisional motion, including its accumulated bias.
    # Accepted pose and the next search prediction must both restart there.
    recovered = Rz(np.deg2rad(5.5))
    monkeypatch.setattr(tracker, "_observe", _direct_observer(tracker, recovered))
    pose, valid, record = tracker.update(5, gray, mask, matches, estimate,
                                         _pixels(recovered), ids)
    assert valid and record["status"] == "recovered"
    assert np.rad2deg(geodesic_angle(pose, recovered)) < 1e-5
    pose, valid, record = tracker.update(6, gray, mask, matches, estimate,
                                         _pixels(Rz(np.deg2rad(6.5))), ids)
    assert valid and record["status"] == "increment"
    assert np.rad2deg(geodesic_angle(pose, Rz(np.deg2rad(6.5)))) < 1e-5


def test_last_good_image_bridges_gap_when_scheduled_anchor_no_longer_matches(monkeypatch):
    tracker, gray, mask, ids = _tracker(correction_interval=3, keyframe_interval=8)
    for index in range(1, 6):
        truth = Rz(np.deg2rad(index))
        matches, estimate = _pair(Rz(np.deg2rad(index - 1)), truth)
        monkeypatch.setattr(tracker, "_observe", _direct_observer(tracker, truth))
        _, valid, record = tracker.update(index, gray, mask, matches, estimate,
                                          _pixels(truth), ids)
        assert valid
        if index == 3:
            # Routine drift correction still uses the older scheduled anchor;
            # the recent recovery snapshot must not dominate that correction.
            assert record["used_keyframes"] == [0]
    assert [key.index for key in tracker.keyframes] == [0]
    assert tracker.recovery_keyframe.index == 5

    truth = Rz(np.deg2rad(6.0))
    matches, estimate = _pair(Rz(np.deg2rad(5.0)), truth)
    _, valid, record = tracker.update(6, np.zeros_like(gray), mask, matches, estimate,
                                      _pixels(truth), ids)
    assert not valid and record["status"] == "unresolved"
    assert record["recovery_keyframe"] == 5

    truth = Rz(np.deg2rad(7.0))
    direct = _direct_observer(tracker, truth)

    def near_image_only(key, gray, mask, prediction, uv, ids):
        observation = direct(key, gray, mask, prediction, uv, ids)
        if key.index == 0:
            observation[2]["accepted"] = False
            observation[2]["reasons"].append("old markings no longer visible")
        return observation

    monkeypatch.setattr(tracker, "_observe", near_image_only)
    matches, estimate = _pair(Rz(np.deg2rad(6.0)), truth)
    pose, valid, record = tracker.update(7, gray, mask, matches, estimate,
                                         _pixels(truth), ids)
    assert valid and record["status"] == "recovered"
    assert record["used_keyframes"] == [5]
    assert [item["keyframe"] for item in record["anchor_attempts"]] == [0, 5]
    assert np.rad2deg(geodesic_angle(pose, truth)) < 1e-5
    assert record["recovery_keyframe"] == 7


def test_accepted_lower_contrast_image_refreshes_rescue_without_promoting_anchor():
    tracker, gray, mask, ids = _tracker(correction_interval=3, keyframe_interval=1)
    truth = Rz(np.deg2rad(1.0))
    matches, estimate = _pair(np.eye(3), truth)
    lower_contrast = np.rint(128.0 + 0.885 * (gray.astype(float) - 128.0)).astype(np.uint8)
    pose, valid, record = tracker.update(1, lower_contrast, mask, matches, estimate,
                                         _pixels(truth), ids)
    assert tracker.config.min_sharpness_ratio < record["sharpness_ratio"]
    assert record["sharpness_ratio"] < tracker.config.keyframe_sharpness_ratio
    assert valid and record["status"] == "increment"
    assert [key.index for key in tracker.keyframes] == [0]
    assert tracker.recovery_keyframe.index == 1
    assert record["recovery_keyframe"] == 1
    np.testing.assert_array_equal(tracker.recovery_keyframe.gray, lower_contrast)
    assert np.rad2deg(geodesic_angle(tracker.recovery_keyframe.pose, truth)) < 1e-5
    assert np.rad2deg(geodesic_angle(pose, truth)) < 1e-5


def test_conflicting_keyframes_do_not_replace_valid_increment(monkeypatch):
    tracker, gray, mask, ids = _tracker(correction_interval=2, keyframe_interval=1,
                                      max_anchor_disagreement_deg=1.0)
    pose1 = Rz(np.deg2rad(1.0))
    matches1, estimate1 = _pair(np.eye(3), pose1)
    tracker.update(1, gray, mask, matches1, estimate1, _pixels(pose1), ids)
    assert [key.index for key in tracker.keyframes] == [0, 1]
    truth = Rz(np.deg2rad(2.0))
    monkeypatch.setattr(tracker, "_observe",
                        _direct_observer(tracker, truth, offsets={1: 2.0}))
    matches2, estimate2 = _pair(pose1, truth)
    pose, valid, record = tracker.update(2, gray, mask, matches2, estimate2,
                                         _pixels(truth), ids)
    assert valid and record["status"] == "increment"
    assert record["anchor_rejection"] == "keyframes disagree"
    assert record["used_keyframes"] == []
    assert np.rad2deg(geodesic_angle(pose, truth)) < 1e-5
    assert [key.index for key in tracker.keyframes] == [0, 1]


def test_large_keyframe_jump_does_not_replace_valid_increment(monkeypatch):
    tracker, gray, mask, ids = _tracker(correction_interval=1, keyframe_interval=1,
                                      max_correction_deg=2.0)
    truth = Rz(np.deg2rad(1.0))
    monkeypatch.setattr(tracker, "_observe",
                        _direct_observer(tracker, truth, offsets={0: 8.0}))
    matches, estimate = _pair(np.eye(3), truth)
    pose, valid, record = tracker.update(1, gray, mask, matches, estimate,
                                         _pixels(truth), ids)
    assert valid and record["status"] == "increment"
    assert "correction exceeds limit" in record["anchor_attempts"][0]["reasons"]
    assert np.rad2deg(geodesic_angle(pose, truth)) < 1e-5
    assert [key.index for key in tracker.keyframes] == [0]


def test_old_keyframes_cannot_restore_a_lost_pose(monkeypatch):
    tracker, gray, mask, ids = _tracker(correction_interval=1, keyframe_interval=2,
                                      max_keyframe_age=2)
    tracker.valid = False
    calls = []
    monkeypatch.setattr(tracker, "_observe", lambda *args: calls.append(args))
    truth = Rz(np.deg2rad(3.0))
    matches, estimate = _pair(Rz(np.deg2rad(2.0)), truth)
    _, valid, record = tracker.update(3, gray, mask, matches, estimate, _pixels(truth), ids)
    assert not valid and record["status"] == "unresolved"
    assert calls == []


@pytest.mark.parametrize("missing_frame", [None, 3])
def test_textured_sphere_pipeline_tracks_and_recovers_from_image_anchors(missing_frame):
    """Image-level gate: an axial sphere rotation is a known 2-D image rotation."""
    from tests.temporal_adapter import run_pipeline

    rng = np.random.default_rng(9)
    texture = rng.integers(0, 255, (256, 256), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0.6)
    yy, xx = np.indices(texture.shape)
    texture[(xx-128)**2 + (yy-128)**2 > RADIUS_PX**2] = 0
    frames = [cv2.cvtColor(cv2.warpAffine(
        texture, cv2.getRotationMatrix2D((128, 128), -0.7 * index, 1), (256, 256)),
        cv2.COLOR_GRAY2BGR) for index in range(7)]
    if missing_frame is not None:
        frames[missing_frame] = np.zeros_like(frames[missing_frame])
    result = run_pipeline(
        frames, K=K, circle=(128.0, 128.0, RADIUS_PX), R_bc=np.eye(3),
        segment_config={"mode": "equator", "equator": {"deadband_px": 3}},
        track_config={"max_corners": 160, "min_distance_px": 6, "quality": 0.01},
        estimate_config={"ransac_iters": 40, "min_inliers": 8},
        temporal_config={"enabled": True, "correction_interval": 2,
                         "keyframe_interval": 3, "min_inliers": 15,
                         "boundary_margin_px": 3},
    )
    expected = Rz(np.deg2rad(4.2))
    expected_valid = np.ones(len(frames), dtype=bool)
    if missing_frame is not None:
        expected_valid[missing_frame] = False
        assert result.top_increments[missing_frame - 1] is None
        assert result.top_increments[missing_frame] is None
        assert np.isnan(result.motion.alpha_top[missing_frame])
    np.testing.assert_array_equal(result.top_step_valid, expected_valid)
    np.testing.assert_array_equal(result.bottom_step_valid, expected_valid)
    assert np.rad2deg(geodesic_angle(result.top_absolute[-1], expected)) < 0.15
    assert np.rad2deg(geodesic_angle(result.bottom_absolute[-1], expected)) < 0.15
    for name in ("top", "bottom"):
        histories = result.temporal["frames"][name]
        assert any(record["status"] == "keyframe" for record in histories)
        first_ids = result.matches[0][name].track_ids
        last_ids = result.matches[-1][name].track_ids
        if missing_frame is None:
            assert len(np.intersect1d(first_ids, last_ids)) >= 15
        else:
            assert histories[missing_frame + 1]["status"] == "recovered"
            assert len(np.intersect1d(first_ids, last_ids)) == 0
