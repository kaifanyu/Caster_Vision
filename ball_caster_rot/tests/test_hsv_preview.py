"""The sampler must preview what will actually be tracked and written."""

import cv2
import numpy as np

from scripts.inspect_hsv import _preview


def fixture():
    hsv = np.zeros((90, 90, 3), dtype=np.uint8)
    hsv[:] = [0, 0, 210]
    hsv[20:35, 25:40] = [3, 180, 190]
    hsv[50:65, 50:65] = [75, 150, 160]
    hsv[20:35, 36:40] = [0, 0, 20]  # dark occluder alongside red mark
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    config = {"segment": {
        "mode": "color", "top_hsv": {"lo": [175, 100, 80], "hi": [10, 255, 255]},
        "bottom_hsv": {"lo": [60, 100, 80], "hi": [90, 255, 255]},
        "yoke_hsv": {"lo": [0, 0, 0], "hi": [179, 255, 35]},
        "grow_px": 6, "yoke_dilate_px": 2, "morphology_px": 1,
    }, "temporal": {"enabled": True, "boundary_margin_px": 2}}
    return frame, hsv, config


def test_preview_shows_saved_ranges_growth_and_yoke_before_new_samples():
    frame, hsv, config = fixture()
    preview = _preview(frame, hsv, {name: [] for name in ("top", "bottom", "yoke")},
                       (45, 45, 43), config=config)
    # White beside a colored mark is admitted by growth, as in tracking.
    assert not np.array_equal(preview[56, 48], frame[56, 48])
    # Feature-center erosion removes the extreme grown boundary.
    np.testing.assert_array_equal(preview[56, 43], frame[56, 43])
    # The occluder is red, rather than the cyan shell tint.
    assert preview[26, 37, 2] > preview[26, 37, 0]


def test_preview_uses_requested_sv_margin_instead_of_default_30():
    frame, hsv, config = fixture()
    config["segment"].update(grow_px=0, yoke_dilate_px=0)
    config["temporal"]["enabled"] = False
    samples = {"top": [], "bottom": [np.array([[75, 150, 185]], dtype=np.uint8)], "yoke": []}
    tight = _preview(frame, hsv, samples, (45, 45, 43), config=config, sv_margin=5)
    broad = _preview(frame, hsv, samples, (45, 45, 43), config=config, sv_margin=30)
    np.testing.assert_array_equal(tight[55, 55], frame[55, 55])
    assert not np.array_equal(broad[55, 55], frame[55, 55])


def test_preview_honors_hue_margin_and_supports_sampling_before_circle_fit():
    frame, hsv, _ = fixture()
    samples = {"top": [], "bottom": [np.array([[70, 150, 160]], dtype=np.uint8)], "yoke": []}
    tight = _preview(frame, hsv, samples, None, hue_margin=1, sv_margin=0)
    broad = _preview(frame, hsv, samples, None, hue_margin=6, sv_margin=0)
    np.testing.assert_array_equal(tight[55, 55], frame[55, 55])
    assert not np.array_equal(broad[55, 55], frame[55, 55])
