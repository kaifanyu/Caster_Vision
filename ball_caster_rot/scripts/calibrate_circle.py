#!/usr/bin/env python
"""Fit the caster ball's fixed projected circle and update a YAML config."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from ballrot.calibration import (
    circle_from_three_points,
    detect_circle_auto,
    draw_circle_preview,
)
from ballrot.camera import undistort_image
from ballrot.config import (
    camera_matrix,
    distortion_coefficients,
    load_config,
    resolve_from_config,
    update_yaml,
)


def _first_frame(path: Path) -> "cv2.typing.MatLike":
    if path.is_file():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return image
        capture = cv2.VideoCapture(str(path))
        ok, frame = capture.read()
        capture.release()
        if ok and frame is not None:
            return frame
        raise ValueError(f"could not decode the first frame of {path}")
    candidates = []
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


def _interactive_three_points(frame):
    points: list[tuple[float, float]] = []
    window = "Ball circle: click 3 silhouette points; R resets; Enter accepts"

    def callback(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
            points.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, callback)
    while True:
        display = frame.copy()
        for index, point in enumerate(points):
            cv2.circle(display, (int(point[0]), int(point[1])), 5, (0, 255, 255), -1)
            cv2.putText(
                display,
                str(index + 1),
                (int(point[0]) + 7, int(point[1]) - 7),
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
            cv2.destroyWindow(window)
            raise KeyboardInterrupt("circle selection cancelled")
    cv2.destroyWindow(window)
    return points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument(
        "--clip", type=Path, help="Video/image input (defaults to config input.path)"
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
        "--preview", type=Path, default=PROJECT_ROOT / "out" / "circle_preview.png"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config, config_path = load_config(args.config)
    input_value = args.clip or resolve_from_config(config_path, config["input"]["path"])
    input_path = Path(input_value).expanduser().resolve()
    frame = _first_frame(input_path)
    K, approximate = camera_matrix(config.get("camera", {}), frame.shape)
    dist = distortion_coefficients(config.get("camera", {}))
    frame = undistort_image(frame, K, dist)

    if args.point:
        if len(args.point) != 3:
            raise SystemExit("--point must be supplied exactly three times")
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
        {"circle": {"u0": circle[0], "v0": circle[1], "r_px": circle[2]}},
    )
    preview = draw_circle_preview(frame, circle, args.preview, label=source)
    print(f"Circle source: {source}")
    print(f"u0={circle[0]:.4f}, v0={circle[1]:.4f}, r_px={circle[2]:.4f}")
    print(f"Updated: {config_path}")
    print(f"Preview: {preview}")
    if approximate:
        print("WARNING: camera.K is still approximate; calibrate intrinsics next.")
    print("Inspect the preview. Re-run with --manual if the green ring misses the silhouette.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

