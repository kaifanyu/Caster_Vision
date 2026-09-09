"""Segmentation masks for the projected ball caster.

The functions in this module deliberately return boolean masks.  OpenCV's
``uint8`` (0/255) convention is handled only at the API boundary in
``track.py``; keeping masks boolean here prevents accidental bitwise-mask
bugs in the rest of the pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class SegmentationMasks:
    """The four masks needed by the tracking pipeline.

    ``top`` and ``bottom`` are always disjoint and already exclude both the
    yoke and pixels outside ``ball``.
    """

    ball: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    yoke: np.ndarray

    def __post_init__(self) -> None:
        shape = np.asarray(self.ball).shape
        if len(shape) != 2:
            raise ValueError("segmentation masks must be two-dimensional")
        for name in ("ball", "top", "bottom", "yoke"):
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(
                    f"mask {name!r} has shape {value.shape}, expected {shape}"
                )
            object.__setattr__(self, name, value.astype(bool, copy=False))

        if np.any(self.top & self.bottom):
            raise ValueError("top and bottom masks must be disjoint")
        if np.any((self.top | self.bottom | self.yoke) & ~self.ball):
            raise ValueError("top, bottom, and yoke masks must lie inside ball")
        if np.any((self.top | self.bottom) & self.yoke):
            raise ValueError("top and bottom masks must exclude the yoke")

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in {"ball", "top", "bottom", "yoke"}:
            raise KeyError(name)
        return getattr(self, name)

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return a shallow dictionary view of the masks."""

        return {
            "ball": self.ball,
            "top": self.top,
            "bottom": self.bottom,
            "yoke": self.yoke,
        }

    def as_uint8(self, name: str) -> np.ndarray:
        """Return one mask in the 0/255 representation expected by OpenCV."""

        return self[name].astype(np.uint8) * 255


def _image_shape(image_or_shape: np.ndarray | Sequence[int]) -> tuple[int, int]:
    if isinstance(image_or_shape, np.ndarray):
        shape = image_or_shape.shape
    else:
        shape = tuple(int(v) for v in image_or_shape)
    if len(shape) < 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"invalid image shape: {shape}")
    return int(shape[0]), int(shape[1])


def _parse_circle(
    circle: Sequence[float] | Mapping[str, Any],
) -> tuple[float, float, float]:
    if isinstance(circle, Mapping):
        try:
            u0 = circle["u0"]
            v0 = circle["v0"]
            radius = circle.get("r_px", circle.get("radius"))
        except KeyError as exc:
            raise ValueError("circle mapping needs u0, v0, and r_px") from exc
        if radius is None:
            raise ValueError("circle mapping needs r_px (or radius)")
    else:
        if len(circle) != 3:
            raise ValueError("circle must be (u0, v0, r_px)")
        u0, v0, radius = circle

    values = np.asarray([u0, v0, radius], dtype=float)
    if not np.all(np.isfinite(values)) or values[2] <= 0:
        raise ValueError(f"invalid projected circle: {values.tolist()}")
    return float(values[0]), float(values[1]), float(values[2])


def make_ball_mask(
    image_or_shape: np.ndarray | Sequence[int],
    circle: Sequence[float] | Mapping[str, Any],
    *,
    margin_px: float = 0.0,
) -> np.ndarray:
    """Return the fixed projected-circle mask.

    Positive ``margin_px`` shrinks the usable ball region.  This is useful
    when the fitted silhouette has a one- or two-pixel uncertainty.
    """

    height, width = _image_shape(image_or_shape)
    u0, v0, radius = _parse_circle(circle)
    usable_radius = radius - float(margin_px)
    if not np.isfinite(usable_radius) or usable_radius <= 0:
        raise ValueError("margin_px leaves no usable ball area")
    yy, xx = np.ogrid[:height, :width]
    return (xx - u0) ** 2 + (yy - v0) ** 2 <= usable_radius**2


def _parse_hsv_range(
    hsv_range: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(hsv_range, Mapping):
        if "lo" not in hsv_range or "hi" not in hsv_range:
            raise ValueError(f"{name} must contain 'lo' and 'hi'")
        lo, hi = hsv_range["lo"], hsv_range["hi"]
    else:
        if len(hsv_range) != 2:
            raise ValueError(f"{name} must be (lo, hi)")
        lo, hi = hsv_range

    lo_array = np.asarray(lo, dtype=float)
    hi_array = np.asarray(hi, dtype=float)
    if lo_array.shape != (3,) or hi_array.shape != (3,):
        raise ValueError(f"{name} bounds must each contain H, S, and V")
    if not np.all(np.isfinite(lo_array)) or not np.all(np.isfinite(hi_array)):
        raise ValueError(f"{name} contains non-finite bounds")
    if np.any(lo_array < 0) or np.any(hi_array < 0):
        raise ValueError(f"{name} bounds cannot be negative")
    if lo_array[0] > 179 or hi_array[0] > 179:
        raise ValueError(f"{name} hue bounds must be in OpenCV's 0..179 range")
    if np.any(lo_array[1:] > 255) or np.any(hi_array[1:] > 255):
        raise ValueError(f"{name} saturation/value bounds must be in 0..255")
    if np.any(lo_array[1:] > hi_array[1:]):
        raise ValueError(f"{name} may wrap hue, but not saturation or value")
    return np.rint(lo_array).astype(np.uint8), np.rint(hi_array).astype(np.uint8)


def threshold_hsv(
    hsv_image: np.ndarray,
    hsv_range: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
    *,
    name: str = "HSV range",
) -> np.ndarray:
    """Threshold an OpenCV HSV image, including hue ranges crossing red.

    A hue interval such as ``lo=[170,...], hi=[10,...]`` wraps through zero.
    """

    hsv = np.asarray(hsv_image)
    if hsv.ndim != 3 or hsv.shape[2] != 3:
        raise ValueError("hsv_image must have shape (height, width, 3)")
    if hsv.dtype != np.uint8:
        raise ValueError("hsv_image must use OpenCV uint8 HSV values")
    lo, hi = _parse_hsv_range(hsv_range, name)
    if lo[0] <= hi[0]:
        return cv2.inRange(hsv, lo, hi).astype(bool)

    high_hue_lo = lo.copy()
    high_hue_hi = hi.copy()
    high_hue_hi[0] = 179
    low_hue_lo = lo.copy()
    low_hue_lo[0] = 0
    return (
        cv2.inRange(hsv, high_hue_lo, high_hue_hi).astype(bool)
        | cv2.inRange(hsv, low_hue_lo, hi).astype(bool)
    )


def clean_mask(
    mask: np.ndarray,
    *,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """Apply a light morphological opening then closing."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    kernel_size = int(kernel_size)
    iterations = int(iterations)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if iterations < 0:
        raise ValueError("iterations cannot be negative")
    if kernel_size <= 1 or iterations == 0:
        return binary.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    image = binary.astype(np.uint8) * 255
    image = cv2.morphologyEx(
        image, cv2.MORPH_OPEN, kernel, iterations=iterations
    )
    image = cv2.morphologyEx(
        image, cv2.MORPH_CLOSE, kernel, iterations=iterations
    )
    return image.astype(bool)


def _dilate_mask(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """Grow a boolean mask by an elliptical structuring element."""

    binary = np.asarray(mask, dtype=bool)
    radius = float(radius_px)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("dilation radius must be finite and non-negative")
    size = int(round(radius))
    if size < 1 or not binary.any():
        return binary.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * size + 1, 2 * size + 1)
    )
    return cv2.dilate(binary.astype(np.uint8), kernel).astype(bool)


def _equator_parameters(
    equator: Mapping[str, float] | None,
    *,
    u0: float,
    v0: float,
    default_band_px: float,
) -> tuple[float, float, float]:
    equator = {} if equator is None else equator
    if not isinstance(equator, Mapping):
        raise ValueError("equator must be a mapping")
    y = equator.get(
        "y",
        equator.get(
            "v", equator.get("v0", v0 + float(equator.get("offset_px", 0.0)))
        ),
    )
    slope = equator.get("slope", equator.get("slope_dv_du"))
    if slope is None:
        slope = np.tan(np.deg2rad(float(equator.get("angle_deg", 0.0))))
    band = equator.get(
        "band_px", equator.get("deadband_px", default_band_px)
    )
    values = np.asarray([y, slope, band], dtype=float)
    if not np.all(np.isfinite(values)) or values[2] < 0:
        raise ValueError("invalid equator y/slope/band_px")
    # u0 is intentionally accepted here to document the line's pivot.
    _ = u0
    return float(values[0]), float(values[1]), float(values[2])


def segment_frame(
    frame_bgr: np.ndarray,
    circle: Sequence[float] | Mapping[str, Any],
    *,
    mode: str = "color",
    top_hsv: Mapping[str, Sequence[float]] | Sequence[Sequence[float]] | None = None,
    bottom_hsv: Mapping[str, Sequence[float]]
    | Sequence[Sequence[float]]
    | None = None,
    yoke_hsv: Mapping[str, Sequence[float]]
    | Sequence[Sequence[float]]
    | None = None,
    equator: Mapping[str, float] | None = None,
    equator_band_px: float = 0.0,
    ball_margin_px: float = 0.0,
    morph_kernel: int = 3,
    morph_iterations: int = 1,
    grow_px: float = 0.0,
    separation_px: float = 0.0,
    yoke_dilate_px: float = 0.0,
    color_wins_over_yoke: bool = False,
) -> SegmentationMasks:
    """Build ball, hemisphere, and yoke masks for one BGR frame.

    In ``color`` mode, ``top_hsv`` and ``bottom_hsv`` are required.  In
    ``equator`` mode, the ball is split by

    ``v = equator_y + slope * (u - circle_u0)``.

    A non-zero ``equator_band_px`` creates an intentionally unassigned strip
    around that line, preventing features on the physical seam from being
    attributed to either hemisphere.

    ``grow_px`` dilates each colour mask so the untextured surface *between*
    speckles also becomes trackable.  Sparse, high-contrast markings (printed
    triangles, stickers) otherwise restrict feature detection to the marks
    themselves and starve KLT.  ``separation_px`` is an extra safety band: a
    grown region is discarded wherever it comes within that distance of the
    other hemisphere, so growth can never carry a feature across the seam.
    ``yoke_dilate_px`` widens the yoke exclusion after thresholding, and
    ``color_wins_over_yoke`` keeps pixels that matched a speckle colour even
    when they also fall inside the yoke range.
    """

    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame_bgr must have shape (height, width, 3)")
    if frame.dtype != np.uint8:
        raise ValueError("frame_bgr must be a uint8 OpenCV BGR image")

    mode = str(mode).strip().lower()
    if mode not in {"color", "equator"}:
        raise ValueError("segment mode must be 'color' or 'equator'")

    u0, v0, _ = _parse_circle(circle)
    ball = make_ball_mask(frame, circle, margin_px=ball_margin_px)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    yoke_enabled = not (
        isinstance(yoke_hsv, Mapping)
        and not bool(yoke_hsv.get("enabled", True))
    )
    if yoke_hsv is None or not yoke_enabled:
        yoke = np.zeros(ball.shape, dtype=bool)
    else:
        yoke = clean_mask(
            threshold_hsv(hsv, yoke_hsv, name="yoke_hsv"),
            kernel_size=morph_kernel,
            iterations=morph_iterations,
        )
        yoke = _dilate_mask(yoke, yoke_dilate_px) & ball

    if mode == "color":
        if top_hsv is None or bottom_hsv is None:
            raise ValueError(
                "color segmentation requires both top_hsv and bottom_hsv"
            )
        top = clean_mask(
            threshold_hsv(hsv, top_hsv, name="top_hsv"),
            kernel_size=morph_kernel,
            iterations=morph_iterations,
        )
        bottom = clean_mask(
            threshold_hsv(hsv, bottom_hsv, name="bottom_hsv"),
            kernel_size=morph_kernel,
            iterations=morph_iterations,
        )

        # Overlapping thresholds are ambiguous; silently choosing one would
        # contaminate a hemisphere solve, so reject overlap from both.
        overlap = top & bottom
        top &= ball & ~overlap
        bottom &= ball & ~overlap
        if color_wins_over_yoke:
            # A pixel that matched a speckle colour is speckle, not yoke; the
            # yoke range is a catch-all for dark pixels and otherwise eats
            # shaded marks near the limb.
            yoke &= ~(top | bottom)
        usable_ball = ball & ~yoke

        if grow_px > 0:
            grown_top = _dilate_mask(top, grow_px)
            grown_bottom = _dilate_mask(bottom, grow_px)
            # Grow only into surface that the *other* hemisphere's grown
            # region (plus separation_px) does not claim, so no feature can
            # be seeded on the far side of the seam.
            reach_top = _dilate_mask(grown_top, separation_px)
            reach_bottom = _dilate_mask(grown_bottom, separation_px)
            top = grown_top & ~reach_bottom
            bottom = grown_bottom & ~reach_top
        top &= usable_ball
        bottom &= usable_ball
    else:
        usable_ball = ball & ~yoke
        equator_y, slope, band = _equator_parameters(
            equator,
            u0=u0,
            v0=v0,
            default_band_px=float(equator_band_px),
        )
        yy, xx = np.indices(ball.shape, dtype=float)
        line_y = equator_y + slope * (xx - u0)
        half_band = 0.5 * band
        top = usable_ball & (yy < line_y - half_band)
        bottom = usable_ball & (yy > line_y + half_band)

    return SegmentationMasks(ball=ball, top=top, bottom=bottom, yoke=yoke)


def assign_points_to_hemispheres(
    uv: np.ndarray,
    masks: SegmentationMasks,
) -> np.ndarray:
    """Label pixels as top (0), bottom (1), or invalid/ambiguous (-1)."""

    points = np.asarray(uv, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("uv must have shape (N, 2)")
    labels = np.full(len(points), -1, dtype=np.int8)
    finite = np.all(np.isfinite(points), axis=1)
    xy = np.zeros((len(points), 2), dtype=np.int64)
    xy[finite] = np.rint(points[finite]).astype(np.int64)
    height, width = masks.ball.shape
    inside = (
        finite
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    indices = np.flatnonzero(inside)
    x = xy[indices, 0]
    y = xy[indices, 1]
    labels[indices[masks.top[y, x]]] = 0
    labels[indices[masks.bottom[y, x]]] = 1
    return labels


# Short alias used by some pipeline callers.
build_masks = segment_frame
