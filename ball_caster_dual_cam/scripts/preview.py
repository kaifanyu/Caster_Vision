#!/usr/bin/env python3
"""Preview undistorted circle, color masks and detected corners before tracking."""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ballrot.camera import undistort_image
from ballrot.sphere import fit_circle_points
from ballrot.track import KLTConfig, detect_features
from dualcam.config import CAMERA_NAMES, DEFAULT_CONFIG, load_config, load_intrinsics, write_yaml
from dualcam.session import SelectedVideo
from dualcam.tracking import choose_circle, masks_for


def pick_circle(frame):
    scale = min(1., 1100/frame.shape[1])
    points = []
    title = "Click 3 outer silhouette points; ENTER accepts; R resets; ESC cancels"
    def mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
            points.append((x/scale, y/scale))
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, mouse)
    try:
        while True:
            image = cv2.resize(frame, None, fx=scale, fy=scale)
            for point in points:
                cv2.circle(image, tuple(np.rint(np.array(point)*scale).astype(int)), 4, (0, 255, 255), -1)
            cv2.imshow(title, image)
            key = cv2.waitKey(20) & 0xff
            if key == 27:
                raise ValueError("Circle selection canceled")
            if key == ord('r'):
                points.clear()
            if key in (10, 13) and len(points) == 3:
                return fit_circle_points(points)
    finally:
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--camera", choices=CAMERA_NAMES, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--session", type=Path)
    source.add_argument("--image", type=Path)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--output", type=Path, required=True, help="PNG diagnostic image")
    circle = p.add_mutually_exclusive_group()
    circle.add_argument("--circle", nargs=3, type=float, metavar=("U", "V", "R"))
    circle.add_argument("--pick-circle", action="store_true", help="GUI: three reviewed outer silhouette points")
    args = p.parse_args()
    cfg = load_config(args.config)
    cam = cfg["cameras"][args.camera]
    intrinsic = load_intrinsics(cam)
    if args.frame < 0:
        raise ValueError("frame must be nonnegative")
    if args.session:
        video = SelectedVideo(args.session/f"{args.camera}.avi", intrinsic["image_size"])
        try:
            frame = video.read(args.frame)
        finally:
            video.close()
    else:
        frame = cv2.imread(str(args.image))
        if frame is None or [frame.shape[1], frame.shape[0]] != intrinsic["image_size"]:
            raise ValueError("Image must be readable and match the calibrated image_size")
    frame = undistort_image(frame, intrinsic["K"], intrinsic["dist"])
    if args.circle:
        cam["circle"] = args.circle
    if args.pick_circle:
        cam["circle"] = pick_circle(frame)
    circle = choose_circle(frame, cam)
    masks = masks_for(frame, circle, cam)
    overlay = frame.copy()
    counts = {}
    for name, color in (("top", (0, 0, 255)), ("bottom", (0, 255, 0))):
        selected = masks[name]
        overlay[selected] = (0.65*overlay[selected]+0.35*np.array(color)).astype(np.uint8)
        points = detect_features(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), selected,
                                 KLTConfig.from_mapping(cfg["tracking"]))
        counts[name] = len(points)
        for uv in points:
            cv2.circle(overlay, tuple(np.rint(uv).astype(int)), 3, (255, 255, 0), -1)
    cv2.circle(overlay, tuple(np.rint(circle[:2]).astype(int)), int(round(circle[2])), (0, 255, 255), 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), overlay):
        raise OSError(f"Could not save {args.output}")
    write_yaml(args.output.with_suffix(".yaml"), {"camera": args.camera, "circle": list(circle),
               "coordinates": "undistorted pixels", "red_corners": counts["top"], "green_corners": counts["bottom"]})
    print(f"Saved {args.output}; red corners={counts['top']}, green corners={counts['bottom']}")
    print(f"Reviewed circle suggestion for cameras.{args.camera}.circle: {list(map(float, circle))}")
    print("Review masks across poses: exclude yoke, inner discs, rims and background. Corner count alone is not accuracy.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, cv2.error) as error:
        raise SystemExit(f"Preview failed: {error}")
