"""Image truth for timestamped recovery after ordinary reference views expire."""

import cv2
import numpy as np

from ballrot.pipeline import run_pipeline
from ballrot.rotation import Rz, geodesic_angle


def test_recovery_bank_reconnects_real_pixels_after_a_long_blank_gap():
    times = np.array([0., .03, .08, .14, .20, .30, .42, .56, .60])
    radius = 400. / np.sqrt(15.)
    camera = np.array([[400., 0., 128.], [0., 400., 128.], [0., 0., 1.]])
    texture = np.random.default_rng(9).integers(0, 255, (256, 256), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), .6)
    yy, xx = np.indices(texture.shape)
    texture[(xx - 128)**2 + (yy - 128)**2 > radius**2] = 0
    frames = [cv2.cvtColor(cv2.warpAffine(
        texture, cv2.getRotationMatrix2D((128, 128), -20 * time, 1), (256, 256)),
        cv2.COLOR_GRAY2BGR) for time in times]
    for index in range(3, 7):
        frames[index][:] = 0

    def run(recovery):
        return run_pipeline(
            list(zip(frames, times)), K=camera, circle=(128., 128., radius), R_bc=np.eye(3),
            segment_config={"mode": "equator", "equator": {"deadband_px": 3}},
            track_config={"max_corners": 160, "min_distance_px": 6, "quality": .01},
            estimate_config={"ransac_iters": 40, "min_inliers": 8},
            temporal_config={
                "enabled": True, "correction_interval": 2, "keyframe_interval": 2,
                "max_keyframe_age": 2, "min_inliers": 15, "boundary_margin_px": 3,
                "motion_recovery_enabled": recovery, "recovery_bank_age_s": 2.,
                "motion_prediction_horizon_s": 1.,
            },
        )

    baseline, recovered = run(False), run(True)
    np.testing.assert_array_equal(recovered.timestamps_s, times)
    for shell in ("top", "bottom"):
        assert not getattr(baseline, shell + "_step_valid")[-1]
        valid = getattr(recovered, shell + "_step_valid")
        assert not valid[3:7].any()  # prediction alone cannot create a measurement
        assert valid[-1]
        expected = Rz(np.deg2rad(20 * times[-1]))
        assert np.rad2deg(geodesic_angle(getattr(recovered, shell + "_absolute")[-1], expected)) < .25
        history = recovered.temporal["frames"][shell]
        recovery_rows = [row for row in history[7:] if row["status"] == "recovered"]
        assert recovery_rows
        assert any(row["used_keyframes"] for row in recovery_rows)
