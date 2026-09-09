#!/usr/bin/env python
"""Fit the ball caster's fixed projected circle and update ``ball.circle``."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from ball.calibration import (
    circle_from_three_points,
    detect_circle_auto,
    draw_circle_preview,
)
from common.camera import undistort_image
from common.config import (
    camera_matrix,
    distortion_coefficients,
    load_config,
    resolve_from_config,
    update_yaml,
)


def _first_frame(path: Path) -> "cv2.typing.MatLike":
    """Decode one BGR frame from an image, video, directory, or glob."""

    if path.is_file():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return image
        capture = cv2.VideoCapture(str(path))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if ok and frame is not None:
            return frame
        raise ValueError(f"could not decode the first frame of {path}")
    candidates: list[Path] = []
    if path.is_dir():
        for suffix in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            candidates.extend(path.glob(suffix))
    else:
        candidates = [Path(item) for item in glob.glob(str(path))]
    for candidate in sorted(candidates):
        frame = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
        if frame is not None:
            return frame
    raise ValueError(f"no readable image/video at {path}")


def _interactive_three_points(frame: "cv2.typing.MatLike") -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    window = "Ball circle: click 3 silhouette points; R resets; Enter accepts"

    def callback(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
            points.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, callback)
    try:
        while True:
            display = frame.copy()
            for index, point in enumerate(points):
                center = (int(round(point[0])), int(round(point[1])))
                cv2.circle(display, center, 5, (0, 255, 255), -1)
                cv2.putText(
                    display,
                    str(index + 1),
                    (center[0] + 7, center[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("r"), ord("R")):
                points.clear()
            elif key in (13, 10) and len(points) == 3:
                break
            elif key in (27, ord("q"), ord("Q")):
                raise KeyboardInterrupt("circle selection cancelled")
    finally:
        cv2.destroyWindow(window)
    return points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument(
        "--clip",
        type=Path,
        help="Video/image input (defaults to config input.ball_clip)",
    )
    parser.add_argument(
        "--manual", action="store_true", help="Click three ball-silhouette points"
    )
    parser.add_argument(
        "--point",
        nargs=2,
        type=float,
        action="append",
        metavar=("U", "V"),
        help="Noninteractive silhouette point; supply exactly three times",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="Preview path (default: <output.dir>/ball/circle_preview.png)",
    )
    return parser


def _input_path(args: argparse.Namespace, config: dict, config_path: Path) -> Path:
    if args.clip is not None:
        return args.clip.expanduser().resolve()
    value = (config.get("input", {}) or {}).get("ball_clip")
    if value is None:
        raise ValueError("input.ball_clip is missing and --clip was not supplied")
    return resolve_from_config(config_path, value)


def _preview_path(args: argparse.Namespace, config: dict, config_path: Path) -> Path:
    if args.preview is not None:
        return args.preview.expanduser().resolve()
    output = (config.get("output", {}) or {}).get("dir", "out")
    return resolve_from_config(config_path, output) / "ball" / "circle_preview.png"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.manual and args.point:
            raise ValueError("--manual and --point are mutually exclusive")
        config, config_path = load_config(args.config)
        preview_path = _preview_path(args, config, config_path)
        if preview_path == config_path:
            raise ValueError("--preview must not overwrite the calibration config")
        frame = _first_frame(_input_path(args, config, config_path))
        K, approximate = camera_matrix(config.get("camera", {}) or {}, frame.shape)
        dist = distortion_coefficients(config.get("camera", {}) or {})
        frame = undistort_image(frame, K, dist)

        if args.point:
            if len(args.point) != 3:
                raise ValueError("--point must be supplied exactly three times")
            circle = circle_from_three_points(args.point)
            source = "three supplied points"
        elif args.manual:
            circle = circle_from_three_points(_interactive_three_points(frame))
            source = "three clicked points"
        else:
            u0, v0, radius, diagnostics = detect_circle_auto(frame)
            circle = (u0, v0, radius)
            source = f"automatic {diagnostics['detector']} + edge refinement"

        update_yaml(
            config_path,
            {
                "ball": {
                    "circle": {
                        "u0": float(circle[0]),
                        "v0": float(circle[1]),
                        "r_px": float(circle[2]),
                    }
                }
            },
        )
        preview = draw_circle_preview(
            frame, circle, preview_path, label=source
        )
        print(f"Circle source: {source}")
        print(f"u0={circle[0]:.4f}, v0={circle[1]:.4f}, r_px={circle[2]:.4f}")
        print(f"Updated ball.circle in: {config_path}")
        print(f"Preview: {preview}")
        if approximate:
            print("WARNING: camera.K is approximate; calibrate intrinsics before measurement.")
        print("Inspect the preview; use --manual if the green ring misses the silhouette.")
        return 0
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
