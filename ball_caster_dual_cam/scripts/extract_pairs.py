#!/usr/bin/env python3
"""Extract matched raw checkerboard images and capture profiles from a recording."""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dualcam.config import CAMERA_NAMES, DEFAULT_CONFIG, load_config, write_json, write_yaml
from dualcam.session import SelectedVideo, load_session


def extract(config, session_path, output, every_seconds=2.):
    if not np.isfinite(every_seconds) or every_seconds <= 0:
        raise ValueError("every-seconds must be positive and finite")
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose a new or empty image directory: {output}")
    session = load_session(session_path, config)
    if session["metadata"].get("mode") != "checkerboard":
        raise ValueError("Use a session recorded with --mode checkerboard")
    selected = []
    next_time = 0.
    for i, stamp in enumerate(session["times"]):
        if stamp >= next_time:
            selected.append(i)
            next_time = stamp + every_seconds
    sources = {}
    try:
        for ci, name in enumerate(CAMERA_NAMES):
            camera = config["cameras"][name]
            folder = output/name
            folder.mkdir(parents=True, exist_ok=True)
            source = sources[name] = SelectedVideo(session["path"]/f"{name}.avi",
                [camera["capture"]["width"], camera["capture"]["height"]])
            recorded = session["metadata"]["cameras"][name]
            for j, index in enumerate(selected):
                frame = source.read(int(session["pairs"][index, ci]))
                if not cv2.imwrite(str(folder/f"pair_{j:04d}.png"), frame):
                    raise OSError(f"Could not save image in {folder}")
            write_yaml(folder/"capture_profile.yaml", {
                "capture_profile": recorded["requested"], "device": recorded.get("device"),
                "actual_controls": recorded.get("controls"), "session": str(session["path"]),
                "coordinate_system": "raw unrotated, uncropped camera pixels"})
    finally:
        for source in sources.values():
            source.close()
    write_json(output/"pairs.json", {"session": str(session["path"]),
               "selected_pairs": session["pairs"][selected], "times": session["times"][selected],
               "timing": session["report"],
               "note": "Keep only SHARP STATIONARY board poses. Remove rejected pair filenames from BOTH folders."})
    return len(selected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--every-seconds", type=float, default=2.)
    args = p.parse_args()
    count = extract(load_config(args.config), args.session, args.output, args.every_seconds)
    print(f"Saved {count} corresponding image pairs in {args.output}")
    print("Inspect images: retain at least 10 sharp stationary board poses with varied positions and tilts.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, cv2.error) as error:
        raise SystemExit(f"Extraction failed: {error}")
