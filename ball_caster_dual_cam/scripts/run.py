#!/usr/bin/env python3
"""Fit common roll and independent red/green spins from a paired motion session."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dualcam.config import DEFAULT_CONFIG
from dualcam.workflow import run_motion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--session", required=True, help="Paired --mode motion recording directory")
    parser.add_argument("--output", required=True, help="New/empty results directory")
    parser.add_argument("--initial-roll-deg", type=float, required=True, help="KNOWN absolute roll at the first paired frame, relative to calibrated home; use 0 only when held at home")
    parser.add_argument("--max-frames", type=int, default=180)
    args = parser.parse_args(argv)
    try:
        report = run_motion(args.config, args.session, args.output,
                            initial_roll_deg=args.initial_roll_deg, max_frames=args.max_frames)
    except (OSError, ValueError) as exc:
        print(f"Motion fit failed: {exc}", file=sys.stderr)
        return 2
    if not report["success"]:
        reasons = "; ".join(report["fit"]["diagnostics"].get("reasons", []))
        print(f"Motion fit rejected: {reasons}\nDiagnostics: {Path(args.output) / 'results.json'}", file=sys.stderr)
        return 2
    print(f"Results: {Path(args.output) / 'results.json'}\nAngles and validity flags: {Path(args.output) / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
