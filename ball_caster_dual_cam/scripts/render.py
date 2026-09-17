#!/usr/bin/env python3
"""Render calibrated roll and moving swivel axes on both undistorted videos."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dualcam.config import CAMERA_NAMES, DEFAULT_CONFIG, load_config, load_rig
from dualcam.model import project, rotation_x
from dualcam.session import SelectedVideo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    cameras, intrinsics, _ = load_rig(cfg)
    result = json.loads(args.results.read_text())
    from dualcam.workflow import calibration_hashes
    if result.get("calibration_hashes") != calibration_hashes(cfg):
        raise ValueError("Results refer to different intrinsics/stereo; use the original calibration files")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new or empty rendering output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    fit = result["fit"]
    F, C = np.asarray(fit["F"]), np.asarray(fit["pivot"])
    data = fit["datasets"][0]
    q = np.asarray(data["angles"], float)
    valid = np.asarray(data["valid"], bool)
    length = float(result["geometry"]["radius_m"])
    for ci, name in enumerate(CAMERA_NAMES):
        info = intrinsics[ci]
        source = SelectedVideo(Path(result["session"])/f"{name}.avi", info["image_size"])
        times = np.asarray(data["times"], float)
        fps = 1/np.median(np.diff(times))
        writer = cv2.VideoWriter(str(args.output/f"{name}_axes.avi"), cv2.VideoWriter_fourcc(*'MJPG'),
                                 float(fps), tuple(info["image_size"]))
        if not writer.isOpened():
            source.close()
            raise OSError("Could not open diagnostic video writer")
        try:
            for i, pair in enumerate(result["pairs"]):
                raw = source.read(pair[ci])
                image = cv2.undistort(raw, info["K"], info["dist"])
                if valid[i, 0]:
                    directions = [F[:, 0], (F @ rotation_x(q[i, 0]))[:, 2]]
                    for direction, color in zip(directions, ((255, 180, 0), (0, 220, 255))):
                        xyz = np.stack([C-length*direction, C+length*direction])
                        camera_xyz = xyz @ cameras[ci]["R"].T+cameras[ci]["t"]
                        if np.all(camera_xyz[:, 2] > 0):
                            pixels = project(cameras[ci], xyz)
                            if np.isfinite(pixels).all() and np.max(np.abs(pixels)) < 1e6:
                                a, b = np.rint(pixels).astype(int)
                                cv2.arrowedLine(image, tuple(a), tuple(b), color, 3, tipLength=.1)
                status = "roll measured" if valid[i, 0] else "ROLL UNRESOLVED"
                cv2.putText(image, f"{name}  t={times[i]:.3f}s  {status}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
                cv2.putText(image, "blue: roll axis   yellow: swivel axis   diagnostic playback uses median FPS", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
                writer.write(image)
        finally:
            source.close()
            writer.release()
    print(f"Saved both diagnostic overlays to {args.output}. Use results.csv timestamps for measurement.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, cv2.error) as error:
        raise SystemExit(f"Render failed: {error}")
