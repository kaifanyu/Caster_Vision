#!/usr/bin/env python3
"""Record both cameras with verified profiles and independent host timestamps."""
import argparse
from pathlib import Path
import signal
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dualcam.capture import CameraError, load_capture_config, record_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/rig.yaml")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are refused.")
    parser.add_argument("--duration", type=float, required=True, help="Requested duration in seconds.")
    parser.add_argument("--mode", choices=("roll", "swivel", "motion", "checkerboard"), required=True)
    args = parser.parse_args()
    config, config_path = load_capture_config(args.config)
    stop = threading.Event()
    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, lambda *_: stop.set())
    try:
        print("Opening both cameras and checking profiles. Ctrl+C stops and finalizes the session.", flush=True)
        session = record_session(config, config_path, args.output, args.duration, args.mode, stop)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    print(f"Saved {session['status']} session: {args.output}")
    for name, stats in session["stats"].items():
        fps = stats["observed_fps"]
        label = f"{fps:.2f}" if fps is not None else "unavailable"
        print(f"  {name}: {stats['frames']} frames, observed {label} fps, "
              f"{stats['long_intervals']} long intervals")
        for warning in session["cameras"][name]["warnings"]:
            print(f"  {name}: {warning}")
    print("CSV receive timestamps are approximate; verify inter-camera timing before moving stereo fits.")


if __name__ == "__main__":
    try:
        main()
    except (CameraError, OSError, ValueError) as error:
        raise SystemExit(f"Recording failed: {error}")
