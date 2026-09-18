#!/usr/bin/env python3
"""Preview undistorted circle, color masks and detected corners before tracking."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ballrot.camera import undistort_image
from ballrot.sphere import fit_circle_points
from ballrot.track import KLTConfig, detect_features
from dualcam.config import CAMERA_NAMES, DEFAULT_CONFIG, load_config, load_intrinsics, write_yaml
from dualcam.session import SelectedVideo
from dualcam.tracking import choose_circle, masks_for


def save_circle(config_path, camera, circle):
    """Replace only this camera's circle, preserving comments and relative paths."""
    path = Path(config_path).expanduser().resolve()
    source = path.read_bytes().decode("utf-8")
    document = yaml.compose(source)

    def field(node, name):
        if not isinstance(node, yaml.MappingNode):
            raise ValueError(f"Expected a YAML mapping containing {name}")
        matches = [value for key, value in node.value if key.value == name]
        if len(matches) > 1:
            raise ValueError(f"Duplicate YAML field: {name}")
        return matches[0] if matches else None

    camera_node = field(field(document, "cameras"), camera)
    circle_node = field(camera_node, "circle")
    replacement = json.dumps(list(map(float, circle)), allow_nan=False)
    newline = "\r\n" if "\r\n" in source else "\n"
    if circle_node is None:
        # A missing circle previously enabled automatic detection. Insert a saved
        # value at the first camera key without rewriting the rest of the YAML.
        first_key = camera_node.value[0][0]
        start = end = first_key.start_mark.index
        separator = ", " if camera_node.flow_style else newline + " " * first_key.start_mark.column
        replacement = "circle: " + replacement + separator
    else:
        start, end = circle_node.start_mark.index, circle_node.end_mark.index
        if isinstance(circle_node, yaml.SequenceNode) and not circle_node.flow_style:
            replacement += newline + " " * circle_node.end_mark.column
    updated = source[:start] + replacement + source[end:]
    # Validate the edited document before replacing the file. Do not serialize
    # load_config() here: it contains resolved machine-specific paths.
    expected = yaml.safe_load(source)
    expected["cameras"][camera]["circle"] = list(map(float, circle))
    if yaml.safe_load(updated) != expected:
        raise ValueError("Could not update only the selected camera's circle in this YAML layout")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(updated)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--camera", choices=CAMERA_NAMES, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--session", type=Path)
    source.add_argument("--image", type=Path)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--output", type=Path, required=True, help="PNG diagnostic image")
    circle = p.add_mutually_exclusive_group()
    circle.add_argument("--circle", nargs=3, type=float, metavar=("U", "V", "R"),
                        help="Set and save this camera's circle in --config")
    circle.add_argument("--pick-circle", action="store_true",
                        help="GUI: click three outer silhouette points; ENTER accepts and saves to --config")
    p.add_argument("--no-save", action="store_true",
                   help="Preview a picked/explicit circle without updating --config")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    config_path = Path(cfg["_config_path"])
    if config_path in (args.output.resolve(), args.output.with_suffix(".yaml").resolve()):
        raise ValueError("Preview output paths must not overwrite the rig configuration")
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
    if (args.pick_circle or args.circle is not None) and not args.no_save:
        save_circle(config_path, args.camera, circle)
        print(f"Updated {config_path}: cameras.{args.camera}.circle = {list(map(float, circle))}")
    else:
        print(f"Preview circle for cameras.{args.camera}.circle: {list(map(float, circle))}; config unchanged")
    print("Review masks across poses: exclude yoke, inner discs, rims and background. Corner count alone is not accuracy.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, cv2.error) as error:
        raise SystemExit(f"Preview failed: {error}")
