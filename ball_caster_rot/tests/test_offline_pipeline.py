"""Image-level checks for the complete forward and offline trajectory paths."""

import cv2
import numpy as np
import pytest

from ballrot.pipeline import run_pipeline
from ballrot.rotation import Rz, geodesic_angle


K = np.array([[400., 0., 128.], [0., 400., 128.], [0., 0., 1.]])
RADIUS_PX = 400. / np.sqrt(15.)


def textured_frames(count=10, missing=None):
    # A camera-axis rotation of a centered sphere is exactly an image-plane
    # rotation. This provides image truth without mocking tracking or geometry.
    rng = np.random.default_rng(9)
    texture = rng.integers(0, 255, (256, 256), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), .6)
    yy, xx = np.indices(texture.shape)
    texture[(xx-128)**2 + (yy-128)**2 > RADIUS_PX**2] = 0
    frames = [cv2.cvtColor(cv2.warpAffine(
        texture, cv2.getRotationMatrix2D((128, 128), -.7*index, 1.), (256, 256)),
        cv2.COLOR_GRAY2BGR) for index in range(count)]
    if missing is not None:
        frames[missing] = np.zeros_like(frames[missing])
    return frames


def run(frames, *, offline=True, temporal=True):
    return run_pipeline(
        frames, K=K, circle=(128., 128., RADIUS_PX), R_bc=np.eye(3),
        segment_config={"mode": "equator", "equator": {"deadband_px": 3}},
        track_config={"max_corners": 160, "min_distance_px": 6, "quality": .01},
        estimate_config={"ransac_iters": 40, "min_inliers": 8},
        temporal_config={"enabled": temporal, "correction_interval": 2,
                         "keyframe_interval": 3, "min_inliers": 15,
                         "boundary_margin_px": 3},
        offline_config={"enabled": offline, "max_tracks_per_frame": 70},
    )


def test_offline_pipeline_uses_reverse_images_and_preserves_forward_evidence():
    frames = textured_frames()
    baseline = run(frames, offline=False)
    refined = run(frames)
    assert baseline.offline is None
    assert refined.offline is not None
    assert any(entry["accepted"] for entry in refined.offline["reverse_observations"])
    expected = Rz(np.deg2rad(.7*(len(frames)-1)))
    for name in ("top", "bottom"):
        np.testing.assert_allclose(refined.initial_rotations[name],
                                   getattr(baseline, f"{name}_absolute"), atol=1e-12)
        np.testing.assert_array_equal(refined.initial_valid[name],
                                      getattr(baseline, f"{name}_step_valid"))
        assert refined.offline["summary"][name]["optimized_components"] >= 1
        assert refined.offline["summary"][name]["refined_frames"] >= len(frames)-2
        assert refined.offline_observations[name]
        assert max(entry.weight for entry in refined.offline_observations[name]) == 2.
        components = refined.offline["shells"][name]["components"]
        accepted = [component for component in components if component["accepted"]]
        assert all(component["robust_cost_after"] <= component["robust_cost_before"]
                   for component in accepted)
        assert np.rad2deg(geodesic_angle(getattr(refined, f"{name}_absolute")[-1], expected)) < .15
        for before, after in zip(baseline.matches, refined.matches):
            np.testing.assert_array_equal(before[name].track_ids, after[name].track_ids)
            np.testing.assert_allclose(before[name].uv_curr, after[name].uv_curr)
        for before, after in zip(baseline.estimates, refined.estimates):
            np.testing.assert_allclose(before[name].R, after[name].R)


def test_offline_pipeline_retains_unobserved_gap_and_does_not_bridge_increments():
    missing = 4
    frames = textured_frames(missing=missing)
    result = run(frames)
    expected_valid = np.ones(len(frames), dtype=bool)
    expected_valid[missing] = False
    expected = Rz(np.deg2rad(.7*(len(frames)-1)))
    for name in ("top", "bottom"):
        np.testing.assert_array_equal(getattr(result, f"{name}_step_valid"), expected_valid)
        assert getattr(result, f"{name}_increments")[missing-1] is None
        assert getattr(result, f"{name}_increments")[missing] is None
        assert np.isnan(getattr(result.motion, f"alpha_{name}")[missing])
        assert missing not in {entry.frame_index for entry in result.offline_observations[name]}
        assert result.offline["shells"][name]["frames"][missing]["status"] == "unresolved"
        assert np.rad2deg(geodesic_angle(getattr(result, f"{name}_absolute")[-1], expected)) < .15
        assert result.offline["summary"][name]["optimized_components"] >= 1


@pytest.mark.parametrize("timestamps", [[0., 0.], [1., .5], [0., np.nan]])
def test_offline_pipeline_rejects_invalid_time_instead_of_inventing_sampling(timestamps):
    frames = textured_frames(count=2)
    with pytest.raises(ValueError, match="strictly increasing timestamps"):
        run(list(zip(frames, timestamps)))


def test_offline_pipeline_keeps_native_irregular_timestamps():
    frames = textured_frames(count=4)
    times = np.array([.012, .041, .079, .119])
    result = run(list(zip(frames, times)))
    np.testing.assert_array_equal(result.timestamps_s, times)


def test_offline_pipeline_requires_persistent_identity_tracking():
    with pytest.raises(ValueError, match="persistent feature identities"):
        run(textured_frames(count=1), temporal=False)


def test_offline_csv_distinguishes_refinement_from_forward_fallback(tmp_path):
    import csv
    from ballrot.diagnostics import write_run_outputs

    result = run(textured_frames())
    paths = write_run_outputs(tmp_path, result.timestamps_s, result.motion,
                              result.qualities, result.top_absolute, result.bottom_absolute,
                              temporal=result.temporal, offline=result.offline)
    with paths["csv"].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for name in ("top", "bottom"):
        statuses = [entry["status"] for entry in result.offline["shells"][name]["frames"]]
        assert [row[f"{name}_offline_status"] for row in rows] == statuses
        assert "refined" in statuses
