#!/usr/bin/env python
"""Interactively sample ball top, bottom, and yoke HSV ranges."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from common.camera import undistort_image
from common.config import (
    camera_matrix,
    distortion_coefficients,
    load_config,
    resolve_from_config,
    update_yaml,
)
from common.io_frames import FrameSource


def _range(samples: list[np.ndarray], *, dark_object: bool = False) -> dict[str, list[int]]:
    """Suggest an OpenCV HSV range from sampled pixel patches."""

    if not samples:
        raise ValueError("at least one non-empty HSV sample is required")
    values = np.concatenate(samples, axis=0).astype(float)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("HSV samples must contain rows of H, S, V")
    if not np.all(np.isfinite(values)):
        raise ValueError("HSV samples must be finite")
    if dark_object:
        return {
            "lo": [0, 0, 0],
            "hi": [
                179,
                min(255, int(np.percentile(values[:, 1], 95)) + 35),
                min(255, int(np.percentile(values[:, 2], 95)) + 25),
            ],
        }
    hues = values[:, 0] * (2 * np.pi / 180.0)
    mean_sine, mean_cosine = np.mean(np.sin(hues)), np.mean(np.cos(hues))
    if np.hypot(mean_sine, mean_cosine) < 1e-6:
        raise ValueError("sampled hues are too dispersed to define one color range")
    center_h = np.arctan2(mean_sine, mean_cosine)
    center_h = (center_h % (2 * np.pi)) * 180.0 / (2 * np.pi)
    hue_delta = 10
    lo_h = int(round((center_h - hue_delta) % 180))
    hi_h = int(round((center_h + hue_delta) % 180))
    lo_s = max(0, int(np.percentile(values[:, 1], 5)) - 25)
    hi_s = min(255, int(np.percentile(values[:, 1], 95)) + 25)
    lo_v = max(0, int(np.percentile(values[:, 2], 5)) - 25)
    hi_v = min(255, int(np.percentile(values[:, 2], 95)) + 25)
    return {"lo": [lo_h, lo_s, lo_v], "hi": [hi_h, hi_s, hi_v]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--clip", type=Path, help="Override config input.ball_clip")
    parser.add_argument(
        "--write", action="store_true", help="Write suggested ranges to ball.segment"
    )
    return parser


def _clip_path(args: argparse.Namespace, config: dict, config_path: Path) -> Path:
    if args.clip is not None:
        return args.clip.expanduser().resolve()
    value = (config.get("input", {}) or {}).get("ball_clip")
    if value is None:
        raise ValueError("input.ball_clip is missing and --clip was not supplied")
    return resolve_from_config(config_path, value)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config, config_path = load_config(args.config)
        input_config = config.get("input", {}) or {}
        source = FrameSource(
            _clip_path(args, config, config_path),
            input_type=str(input_config.get("type", "auto")),
            max_frames=1,
            image_fps=input_config.get("fps_override"),
        )
        record = next(iter(source))
        K, _ = camera_matrix(config.get("camera", {}) or {}, record.image.shape)
        frame = undistort_image(
            record.image,
            K,
            distortion_coefficients(config.get("camera", {}) or {}),
        )
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        samples: dict[str, list[np.ndarray]] = {"top": [], "bottom": [], "yoke": []}
        active = ["top"]
        window = "HSV sampler: T/B/Y select; click patches; Enter prints; R resets; Q exits"

        def on_click(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            y0, y1 = max(0, y - 2), min(hsv.shape[0], y + 3)
            x0, x1 = max(0, x - 2), min(hsv.shape[1], x + 3)
            patch = hsv[y0:y1, x0:x1].reshape(-1, 3)
            if len(patch):
                samples[active[0]].append(patch.copy())

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, on_click)
        try:
            while True:
                display = frame.copy()
                counts = ", ".join(f"{key}={len(value)}" for key, value in samples.items())
                cv2.putText(
                    display,
                    f"sampling {active[0]} | {counts}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
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
                elif key in (ord("r"), ord("R")):
                    samples[active[0]].clear()
                elif key in (13, 10):
                    break
                elif key in (27, ord("q"), ord("Q")):
                    return 1
        finally:
            cv2.destroyWindow(window)

        missing = [name for name, values in samples.items() if not values]
        if missing:
            raise ValueError("no samples collected for: " + ", ".join(missing))
        suggested = {
            f"{name}_hsv": _range(values, dark_object=name == "yoke")
            for name, values in samples.items()
        }
        for name, value in suggested.items():
            print(f"{name}: {value}")
        if args.write:
            suggested["yoke_hsv"]["enabled"] = True
            update_yaml(config_path, {"ball": {"segment": suggested}})
            print(f"Updated ball.segment in: {config_path}")
        else:
            print("Re-run with --write to store these ranges after reviewing them.")
        return 0
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())

