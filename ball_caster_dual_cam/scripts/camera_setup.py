#!/usr/bin/env python3
"""Inspect webcam capabilities, or explicitly apply and verify the rig profile."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dualcam.capture import (CAMERAS, CameraError, V4L2Camera, configure_open_capture,
                             load_capture_config, warmup_capture)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/rig.yaml")
    parser.add_argument("--camera", choices=(*CAMERAS, "all"), default="all")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--inspect", action="store_true", help="Read-only capabilities (default).")
    actions.add_argument("--apply", action="store_true", help="Write and read back the configured profile.")
    parser.add_argument("--output", type=Path, help="Optional JSON capabilities/profile report.")
    parser.add_argument("--focus", type=int, help="Explicit C920 focus trial; requires --camera c920 --apply.")
    parser.add_argument("--preview", action="store_true", help="Preview one camera after --apply; q exits.")
    parser.add_argument("--preview-seconds", type=float, default=20)
    args = parser.parse_args()
    if args.focus is not None and (args.camera != "c920" or not args.apply):
        parser.error("--focus requires --camera c920 --apply; Brio 101 has fixed focus.")
    if args.preview and (not args.apply or args.camera == "all" or args.preview_seconds <= 0):
        parser.error("--preview requires --apply, one --camera, and a positive --preview-seconds.")
    config, _ = load_capture_config(args.config)
    if args.focus is not None:
        config["cameras"]["c920"]["capture"]["focus"] = args.focus
    names = CAMERAS if args.camera == "all" else (args.camera,)
    cameras = {name: V4L2Camera(name, config["cameras"][name]) for name in names}
    reports, failed = {}, False
    if args.apply:
        # Check every selected device before changing any of them.
        prepared = {name: camera.prepare() for name, camera in cameras.items()}
        for name, camera in cameras.items():
            if args.preview:
                cap = cv2.VideoCapture(camera.device, cv2.CAP_V4L2)
                try:
                    reports[name] = configure_open_capture(cap, camera, prepared[name])
                    reports[name]["warmup_controls"] = warmup_capture(cap, camera, prepared[name][0])
                    deadline = time.monotonic() + args.preview_seconds
                    while time.monotonic() < deadline:
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            raise CameraError(f"{name}: preview read failed.")
                        # Display original pixels; GUI window may be resized by the user.
                        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                        cv2.imshow(name, frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    reports[name]["final_controls"] = camera.verify_controls(prepared[name][0])
                finally:
                    cap.release()
                    cv2.destroyAllWindows()
            else:
                reports[name] = camera.apply(prepared=prepared[name])
            print(f"{name}: profile applied and read back successfully.")
            print(json.dumps(reports[name], indent=2))
    else:
        for name, camera in cameras.items():
            try:
                reports[name] = camera.inspect()
                print(f"{name}: {camera.device}\n{reports[name]['controls_raw']}\n{reports[name]['formats_raw']}")
            except CameraError as error:
                reports[name] = {"device": camera.device, "error": str(error)}
                print(str(error), file=sys.stderr)
                failed = True
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2) + "\n")
    if args.focus is not None:
        print(f"Trial focus={args.focus}; save the chosen value in cameras.c920.capture.focus before calibration.")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CameraError, OSError, ValueError, cv2.error) as error:
        raise SystemExit(f"Camera setup failed: {error}")
