#!/usr/bin/env python
"""Interactively sample top, bottom, and yoke colors and suggest HSV ranges.

Samples accumulate across as many frames as you visit.  Speckle appearance
changes a great deal with pose and shading -- a hemisphere that is oblique and
shadowed in frame 0 is bright and frontal 200 frames later -- so ranges fitted
from a single frame systematically under-segment the rest of the clip.  Step
through the clip with the navigation keys and sample each class wherever it
looks different.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from ballrot.camera import undistort_image
from ballrot.config import (
    camera_matrix,
    configured_circle,
    distortion_coefficients,
    load_config,
    resolve_from_config,
    update_yaml,
)
from ballrot.io_frames import FrameSource
from ballrot.pipeline import _segment
from ballrot.segment import threshold_hsv
from ballrot.temporal import TemporalConfig, safe_tracking_masks

CLASSES = ("top", "bottom", "yoke")


def _range(
    samples: list[np.ndarray],
    *,
    dark_object: bool = False,
    hue_margin: float = 6.0,
    tail_pct: float = 2.0,
    sv_margin: int = 30,
) -> dict[str, list[int]]:
    """Fit an HSV box around the pooled samples of one class.

    Hue bounds come from the circular sample spread plus ``hue_margin``
    rather than a fixed half-width, so a class whose hue genuinely varies
    (shadowed versus lit paint) is not clipped.  Saturation and value use
    percentile tails plus ``sv_margin`` in both directions; a one-sided
    value cap is the single most common cause of missing detections when
    the class was only ever sampled in shade.
    """

    values = np.concatenate(samples, axis=0).astype(float)
    lo_s = max(0, int(np.percentile(values[:, 1], tail_pct)) - sv_margin)
    hi_s = min(255, int(np.percentile(values[:, 1], 100 - tail_pct)) + sv_margin)
    lo_v = max(0, int(np.percentile(values[:, 2], tail_pct)) - sv_margin)
    hi_v = min(255, int(np.percentile(values[:, 2], 100 - tail_pct)) + sv_margin)
    if dark_object:
        # The yoke is defined by darkness, not hue.  Keep the upper value
        # bound tight: every pixel it claims is removed from both
        # hemispheres before tracking.
        return {"lo": [0, 0, 0], "hi": [179, hi_s, hi_v]}

    # Work on the doubled hue angle so red near 0/179 averages correctly.
    hues = values[:, 0] * (2 * np.pi / 180.0)
    center = np.arctan2(np.mean(np.sin(hues)), np.mean(np.cos(hues)))
    offsets = np.rad2deg(np.angle(np.exp(1j * (hues - center)))) / 2.0
    lo_off = float(np.percentile(offsets, tail_pct)) - hue_margin
    hi_off = float(np.percentile(offsets, 100 - tail_pct)) + hue_margin
    center_h = (np.rad2deg(center) / 2.0) % 180.0
    lo_h = int(round((center_h + lo_off) % 180))
    hi_h = int(round((center_h + hi_off) % 180))
    return {"lo": [lo_h, lo_s, lo_v], "hi": [hi_h, hi_s, hi_v]}


def _preview(
    frame: np.ndarray,
    hsv: np.ndarray,
    samples: dict[str, list[np.ndarray]],
    circle: tuple[float, float, float] | None,
    *,
    config: dict | None = None,
    hue_margin: float = 6.0,
    sv_margin: int = 30,
) -> np.ndarray:
    """Preview the masks tracking will use with the proposed saved ranges.

    Unsampled classes retain their configured ranges. With a fitted circle,
    include growth, shell separation, yoke exclusion and the tracking margin.
    Without enough geometry/ranges, show only the available color thresholds.
    """

    display = frame.copy()
    colors = {"top": (255, 255, 0), "bottom": (255, 0, 255), "yoke": (0, 0, 255)}
    settings = dict((config or {}).get("segment", {}))
    for name in CLASSES:
        if samples[name]:
            settings[f"{name}_hsv"] = _range(
                samples[name], dark_object=name == "yoke",
                hue_margin=hue_margin, sv_margin=sv_margin)
    if circle is not None and all(settings.get(f"{name}_hsv") for name in ("top", "bottom")):
        masks = _segment(frame, circle, settings)
        temporal = TemporalConfig.from_mapping((config or {}).get("temporal"))
        if temporal.enabled:
            masks = safe_tracking_masks(masks, temporal.boundary_margin_px)
        tint = display.copy()
        for name in CLASSES:
            tint[masks[name]] = colors[name]
        return cv2.addWeighted(display, 0.55, tint, 0.45, 0)
    inside = np.ones(frame.shape[:2], dtype=bool)
    if circle is not None:
        yy, xx = np.ogrid[: frame.shape[0], : frame.shape[1]]
        inside = (xx - circle[0]) ** 2 + (yy - circle[1]) ** 2 <= circle[2] ** 2
    tint = display.copy()
    for name in CLASSES:
        fitted = settings.get(f"{name}_hsv")
        if not fitted or not fitted.get("enabled", True):
            continue
        tint[threshold_hsv(hsv, fitted) & inside] = colors[name]
    return cv2.addWeighted(display, 0.55, tint, 0.45, 0)


def _collect_frames(
    source: FrameSource,
    config: dict,
    wanted_count: int,
) -> list[tuple[int, np.ndarray]]:
    """Decode a set of evenly spaced, undistorted frames spanning the clip."""

    total = source.frame_count
    if total:
        wanted = set(
            np.linspace(0, total - 1, min(wanted_count, total))
            .round()
            .astype(int)
            .tolist()
        )
    else:
        wanted = None
    frames: list[tuple[int, np.ndarray]] = []
    K = None
    dist = None
    for record in source:
        if K is None:
            K, _ = camera_matrix(config.get("camera", {}), record.image.shape)
            dist = distortion_coefficients(config.get("camera", {}))
        if wanted is None or record.index in wanted:
            frames.append((record.index, undistort_image(record.image, K, dist)))
        if wanted is None and len(frames) >= wanted_count:
            break
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--clip", type=Path, help="Override config input.path")
    parser.add_argument(
        "--write", action="store_true", help="Write suggested ranges to config"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=12,
        help="Evenly spaced frames made available for sampling (default: 12)",
    )
    parser.add_argument(
        "--patch",
        type=int,
        default=2,
        help="Half-width in pixels of the square sampled per click (default: 2)",
    )
    parser.add_argument(
        "--hue-margin",
        type=float,
        default=6.0,
        help="Extra hue half-width beyond the sampled spread (default: 6)",
    )
    parser.add_argument(
        "--sv-margin",
        type=int,
        default=30,
        help="Extra saturation/value margin on both sides (default: 30)",
    )
    args = parser.parse_args()
    if args.frames < 1:
        raise SystemExit("--frames must be at least 1")
    config, config_path = load_config(args.config)
    path = args.clip or resolve_from_config(config_path, config["input"]["path"])
    source = FrameSource(path, input_type="auto")
    frames = _collect_frames(source, config, args.frames)
    if not frames:
        raise SystemExit("input contains no decodable frames")
    hsv_frames = [cv2.cvtColor(image, cv2.COLOR_BGR2HSV) for _, image in frames]
    circle = configured_circle(config.get("circle", {}))

    samples: dict[str, list[np.ndarray]] = {name: [] for name in CLASSES}
    per_frame: dict[str, list[int]] = {name: [] for name in CLASSES}
    active = ["top"]
    position = [0]
    show_preview = [True]
    window = "HSV sampler"

    def on_click(event, x, y, _flags, _userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        hsv = hsv_frames[position[0]]
        half = max(0, int(args.patch))
        y0, y1 = max(0, y - half), min(hsv.shape[0], y + half + 1)
        x0, x1 = max(0, x - half), min(hsv.shape[1], x + half + 1)
        samples[active[0]].append(hsv[y0:y1, x0:x1].reshape(-1, 3).copy())
        per_frame[active[0]].append(position[0])

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_click)
    print(f"Loaded {len(frames)} frames spanning the clip.")
    print("Keys: T/B/Y class | N or . next frame | P or , previous | 0 first")
    print("      SPACE toggle mask preview | R reset active class | U undo last")
    print("      ENTER finish | Q/Esc cancel")
    print("Preview includes tracking-mask growth, separation, yoke exclusion and boundary margin when circle/ranges are set.")
    print("Unsampled classes show their saved ranges. Sample only painted marks for T/B; avoid white rims, holes and the yoke.")
    while True:
        index, frame = frames[position[0]]
        hsv = hsv_frames[position[0]]
        display = (
            _preview(frame, hsv, samples, circle, config=config,
                     hue_margin=args.hue_margin, sv_margin=args.sv_margin)
            if show_preview[0] else frame.copy()
        )
        if circle is not None:
            cv2.circle(
                display,
                (int(round(circle[0])), int(round(circle[1]))),
                int(round(circle[2])),
                (0, 255, 0),
                1,
            )
        counts = "  ".join(
            f"{name}={len(samples[name])} clicks/"
            f"{len(set(per_frame[name]))} frames"
            for name in CLASSES
        )
        lines = (
            f"sampling {active[0].upper()}   frame {index}  "
            f"({position[0] + 1}/{len(frames)})",
            counts,
            ("tracking masks ON (SPACE)" if circle is not None
             and all(config.get("segment", {}).get(f"{name}_hsv") or samples[name]
                     for name in ("top", "bottom")) else "HSV colors only (circle/ranges unset)")
            if show_preview[0] else "preview OFF (SPACE)",
        )
        for row, text in enumerate(lines):
            cv2.putText(
                display,
                text,
                (10, 30 + 28 * row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("t"), ord("T")):
            active[0] = "top"
        elif key in (ord("b"), ord("B")):
            active[0] = "bottom"
        elif key in (ord("y"), ord("Y")):
            active[0] = "yoke"
        elif key in (ord("n"), ord("N"), ord(".")):
            position[0] = min(position[0] + 1, len(frames) - 1)
        elif key in (ord("p"), ord("P"), ord(",")):
            position[0] = max(position[0] - 1, 0)
        elif key == ord("0"):
            position[0] = 0
        elif key == ord(" "):
            show_preview[0] = not show_preview[0]
        elif key in (ord("r"), ord("R")):
            samples[active[0]].clear()
            per_frame[active[0]].clear()
        elif key in (ord("u"), ord("U")):
            if samples[active[0]]:
                samples[active[0]].pop()
                per_frame[active[0]].pop()
        elif key in (13, 10):
            break
        elif key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(window)
            return 1
    cv2.destroyWindow(window)

    missing = [name for name in CLASSES if not samples[name]]
    if missing:
        raise SystemExit("No samples collected for: " + ", ".join(missing))
    thin = [name for name in ("top", "bottom") if len(set(per_frame[name])) < 3]
    if thin:
        print(
            "WARNING: "
            + ", ".join(thin)
            + " was sampled on fewer than three frames. Ranges fitted that way "
            "usually miss the same class under different lighting; re-run and "
            "sample each hemisphere bright, shaded, frontal, and near the limb."
        )
    suggested = {
        f"{name}_hsv": _range(
            samples[name],
            dark_object=name == "yoke",
            hue_margin=args.hue_margin,
            sv_margin=args.sv_margin,
        )
        for name in CLASSES
    }
    for name in CLASSES:
        key = f"{name}_hsv"
        print(
            f"{key}: {suggested[key]}   "
            f"({len(samples[name])} clicks over "
            f"{len(set(per_frame[name]))} frames)"
        )
    if args.write:
        suggested["yoke_hsv"]["enabled"] = True
        update_yaml(config_path, {"segment": suggested})
        print(f"Updated: {config_path}")
    else:
        print("Re-run with --write to store these ranges after reviewing them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
