"""Optional timestamped recovery uses predictions only to find visual anchors."""
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ballrot.motion_recovery import RotationMotionPrior, ViewReferenceBank, verify_patch_matches
from ballrot.rotation import Rz, geodesic_angle
from ballrot.temporal import TemporalConfig, TemporalRotationTracker, estimate_quality
from ballrot.track import KLTConfig, TrackMatches
from tests.test_temporal_port import K, CENTER, RADIUS_PX, SOLVER, _directions, _pixels, _pair


def test_so3_prediction_uses_native_intervals_and_expires_without_measurements():
    rate = np.array([.2, -.3, .4])
    prior = RotationMotionPrior(.5)
    for time in (0., .02, .11, .2):
        prior.observe(time, Rotation.from_rotvec(rate*time).as_matrix())
    truth = Rotation.from_rotvec(rate*.27).as_matrix()
    assert geodesic_angle(prior.predict(.27), truth) < 1e-7
    assert prior.predict(.71) is None
    prior.observe(.9, Rotation.from_rotvec(rate*.9).as_matrix())
    assert len(prior.history) == 1
    assert prior.predict(.95) is None
    for time in np.arange(1., 2., .05):
        prior.observe(time, Rotation.from_rotvec(rate*time).as_matrix())
    assert len(prior.history) == 5
    other = RotationMotionPrior(.5)
    assert not other.history


def test_reference_bank_is_bounded_diverse_and_expires_by_time():
    bank = ViewReferenceBank(4, 2., 8.)
    other = ViewReferenceBank(4, 2., 8.)
    for index in range(30):
        key = SimpleNamespace(index=index, timestamp_s=index*.05,
                              pose=Rz(np.deg2rad(index*9)), sharpness=10.,
                              gray=np.full((16, 16), index, np.uint8))
        bank.add(key)
        assert len(bank.frames) <= 4
    assert len(bank.frames) == 4
    assert len({key.index for key in bank.frames}) == 4
    assert not other.frames
    bank.expire(3.6)
    assert bank.frames == []


def _enabled_tracker(**options):
    cfg = TemporalConfig(enabled=True, motion_recovery_enabled=True, **options)
    tracker = TemporalRotationTracker(K, CENTER, 1., RADIUS_PX, KLTConfig(), SOLVER, cfg)
    gray = np.random.default_rng(42).integers(0, 255, (256, 256), np.uint8)
    mask = np.ones(gray.shape, bool)
    ids = np.arange(len(_directions()))
    tracker.initialize(gray, mask, _pixels(np.eye(3)), ids, timestamp_s=0.)
    return tracker, gray, mask, ids


def test_motion_seed_can_find_a_fresh_anchor_after_bad_increment_prediction(monkeypatch):
    tracker, gray, mask, ids = _enabled_tracker(correction_interval=20)
    first = Rz(np.deg2rad(1.))
    matches, estimate = _pair(np.eye(3), first)
    assert tracker.update(1, gray, mask, matches, estimate, _pixels(first), ids, timestamp_s=.1)[1]
    failure = replace(estimate, R=None, inlier_mask=np.zeros(matches.count, bool), failure_reason="occluded")
    assert not tracker.update(2, np.zeros_like(gray), mask, matches, failure,
                              _pixels(first), ids, timestamp_s=.2)[1]
    truth = Rz(np.deg2rad(3.))

    def observe(key, image, mask, prediction, uv, ids):
        if geodesic_angle(prediction, truth) < np.deg2rad(.05):
            direct, fit = _pair(key.pose, truth)
            quality = estimate_quality(direct, fit, RADIUS_PX, tracker.config)
            quality["keyframe"] = key.index
            return direct, fit, quality
        return matches, failure, {"accepted": False, "reasons": ["search missed actual pixels"], "keyframe": key.index}

    monkeypatch.setattr(tracker, "_observe", observe)
    biased = replace(estimate, R=Rz(np.deg2rad(.1)))
    pose, valid, record = tracker.update(3, gray, mask, matches, biased, _pixels(truth), ids, timestamp_s=.3)
    assert valid and record["status"] == "recovered"
    assert geodesic_angle(pose, truth) < 1e-7
    assert any(attempt["accepted"] and attempt["prediction_hypothesis"] == "angular_velocity"
               for attempt in record["anchor_attempts"])


def _image_run(recovery, *, bank_age=1.):
    from tests.temporal_adapter import run_pipeline
    times = np.array([0., .035, .08, .12, .17, .22, .27, .31, .38])
    texture = np.random.default_rng(9).integers(0, 255, (256, 256), np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), .6)
    yy, xx = np.indices(texture.shape)
    texture[(xx-128)**2+(yy-128)**2 > RADIUS_PX**2] = 0
    frames = []
    for index, time in enumerate(times):
        gray = cv2.warpAffine(texture, cv2.getRotationMatrix2D((128, 128), -10*time, 1.), (256, 256))
        if index in (3, 4, 5):
            gray[:] = 0
        frames.append((cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), float(time)))
    return run_pipeline(
        frames, K=K, circle=(128., 128., RADIUS_PX), R_bc=np.eye(3),
        segment_config={"mode": "equator", "equator": {"deadband_px": 3}},
        track_config={"max_corners": 160, "min_distance_px": 6, "quality": .01},
        estimate_config={"ransac_iters": 40, "min_inliers": 8},
        temporal_config={"enabled": True, "correction_interval": 2, "keyframe_interval": 2,
                         "max_keyframe_age": 2, "min_inliers": 15,
                         "motion_recovery_enabled": recovery, "recovery_bank_age_s": bank_age})


def test_real_image_bank_recovers_past_old_frame_age_without_accepting_blank_predictions():
    baseline = _image_run(False)
    result = _image_run(True)
    assert not baseline.top_step_valid[6:].any()
    assert not baseline.bottom_step_valid[6:].any()
    for name in ("top", "bottom"):
        valid = getattr(result, f"{name}_step_valid")
        assert not valid[3:6].any()
        assert valid[6:].all()
        history = result.temporal["frames"][name]
        assert history[6]["status"] == "recovered"
        assert history[3]["motion_recovery"]["motion_prediction_available"]
        assert not history[3]["motion_recovery"]["prediction_is_measurement"]
        assert history[6]["used_keyframes"]
        assert len(history[6]["anchor_attempts"]) <= 12
        assert geodesic_angle(getattr(result, f"{name}_absolute")[-1], Rz(np.deg2rad(3.8))) < np.deg2rad(.15)


def test_expired_bank_cannot_restore_pose_from_motion_prediction_alone():
    result = _image_run(True, bank_age=.1)
    assert not result.top_step_valid[3:].any()
    assert not result.bottom_step_valid[3:].any()
    assert result.temporal["frames"]["top"][6]["motion_recovery"]["motion_prediction_available"]
    assert result.temporal["frames"]["top"][6]["used_keyframes"] == []


def test_timestamp_validation_and_optional_mode_defaults():
    tracker, gray, mask, ids = _enabled_tracker()
    matches, estimate = _pair(np.eye(3), np.eye(3))
    for timestamp in (None, np.nan, 0., -.1):
        with pytest.raises(ValueError):
            tracker.update(1, gray, mask, matches, estimate, _pixels(np.eye(3)), ids, timestamp_s=timestamp)
    assert TemporalConfig().motion_recovery_enabled is False


def test_appearance_verification_rejects_unrelated_pixels_and_duplicate_support():
    source = np.random.default_rng(18).integers(0, 255, (128, 128), np.uint8)
    uv = np.array([[40., 40.], [40., 40.], [80., 80.]])
    matches = TrackMatches(uv, uv.copy(), np.array([.4, .2, 1.1]),
                           detected_count=18, forward_count=3, backward_count=3,
                           source_indices=np.array([9, 12, 17]), track_ids=np.array([100, 101, 102]))
    verified = verify_patch_matches(matches, source, source.copy(), .8)
    assert verified.count == 1  # one pixel contributes one independent point
    assert verified.source_indices.tolist() == [12]
    assert verified.track_ids.tolist() == [101]
    unrelated = np.random.default_rng(31).integers(0, 255, source.shape, np.uint8)
    assert verify_patch_matches(matches, source, unrelated, .8).count == 0
    assert verify_patch_matches(matches, source, np.zeros_like(source), .8).count == 0


@pytest.mark.parametrize("values", [
    {"motion_recovery_enabled": 1}, {"recovery_bank_size": 0}, {"recovery_max_hypotheses": 0},
    {"recovery_bank_age_s": 0}, {"motion_prediction_horizon_s": np.nan}, {"recovery_patch_similarity": 1.1},
])
def test_motion_recovery_configuration_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        TemporalConfig.from_mapping(values)
