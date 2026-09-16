"""Paint-supported masks reject moving unpainted interior surfaces."""

import cv2
import numpy as np
import pytest

from ballrot.pipeline import _segment
from ballrot.segment import PaintSupportConfig, segment_frame


TOP = {"lo": [175, 60, 0], "hi": [10, 255, 255]}
BOTTOM = {"lo": [60, 8, 0], "hi": [95, 255, 255]}
CIRCLE = (128, 96, 112)
OPTIONS = dict(top_hsv=TOP, bottom_hsv=BOTTOM, grow_px=17,
               separation_px=3, morph_kernel=3)


def scene(shift=0):
    hsv = np.full((192, 256, 3), (0, 0, 210), dtype=np.uint8)
    # Colored markings on the exterior; a dim green-gray exposed interior
    # falsely satisfies the permissive bottom hue/saturation range.
    hsv[45:66, 54+shift:75+shift] = (0, 190, 180)
    hsv[110:131, 54+shift:75+shift] = (75, 150, 140)
    hsv[108:135, 140+shift:163+shift] = (75, 12, 100)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@pytest.mark.parametrize("shift", [-20, 0, 20])
def test_rejects_moving_gray_interior_but_keeps_painted_surface(shift):
    image = scene(shift)
    old = segment_frame(image, CIRCLE, **OPTIONS)
    new = segment_frame(image, CIRCLE, **OPTIONS,
                        paint_support={"enabled": True, "max_distance_px": 8})
    assert old.bottom[120, 150+shift]
    assert not new.bottom[120, 150+shift]
    assert new.top[55, 64+shift]
    assert new.bottom[120, 64+shift]
    # A useful near-paint band survives, but unrestricted growth into a
    # farther unmarked surface does not.
    assert new.bottom[120, 79+shift]
    assert old.bottom[120, 88+shift]
    assert not new.bottom[120, 88+shift]
    assert not np.any(new.top & new.bottom)


def test_dark_saturated_interior_cannot_seed_surface_support():
    image = scene()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[108:135, 140:163] = (75, 180, 20)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    old = segment_frame(image, CIRCLE, **OPTIONS)
    new = segment_frame(image, CIRCLE, **OPTIONS,
                        paint_support={"enabled": True})
    assert old.bottom[120, 150]
    assert not new.bottom[120, 150]
    assert new.bottom[120, 64]


def test_no_paint_support_does_not_fall_back_to_gray_shell_region():
    hsv = np.full((192, 256, 3), (75, 12, 100), dtype=np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    result = segment_frame(image, CIRCLE, **OPTIONS,
                           paint_support={"enabled": True})
    assert not result.top.any()
    assert not result.bottom.any()


def test_disabled_filter_preserves_all_legacy_masks():
    image = scene()
    baseline = segment_frame(image, CIRCLE, **OPTIONS)
    disabled = segment_frame(image, CIRCLE, **OPTIONS, paint_support={
        "enabled": False, "min_saturation": 250, "min_value": 250,
        "max_distance_px": 0,
    })
    for name in baseline.as_dict():
        np.testing.assert_array_equal(baseline[name], disabled[name])


def test_pipeline_passes_paint_support_configuration():
    result = _segment(scene(), CIRCLE, {
        "mode": "color", "top_hsv": TOP, "bottom_hsv": BOTTOM,
        "grow_px": 17, "paint_support": {"enabled": True},
    })
    assert not result.bottom[120, 150]
    assert result.bottom[120, 64]


@pytest.mark.parametrize("bad", [
    False, {"enabled": "false"}, {"enabled": 1}, {"unknown": 1},
    {"min_saturation": -1}, {"min_saturation": 256},
    {"min_saturation": 30.5}, {"min_saturation": True},
    {"min_value": -1}, {"min_value": 256},
    {"max_distance_px": -1}, {"max_distance_px": float("nan")},
    {"max_distance_px": float("inf")}, {"max_distance_px": "8"},
    {"max_distance_px": True},
])
def test_invalid_configuration_is_rejected(bad):
    with pytest.raises(ValueError, match="paint_support"):
        PaintSupportConfig.from_mapping(bad)


def test_requires_color_segmentation():
    with pytest.raises(ValueError, match="requires color"):
        segment_frame(scene(), CIRCLE, mode="equator",
                      paint_support={"enabled": True})


def test_support_respects_explicit_zero_growth():
    options = {**OPTIONS, "grow_px": 0}
    result = segment_frame(scene(), CIRCLE, **options,
                           paint_support={"enabled": True})
    assert result.bottom[120, 64]
    assert not result.bottom[120, 79]
