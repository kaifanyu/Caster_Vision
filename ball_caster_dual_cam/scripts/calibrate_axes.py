#!/usr/bin/env python3
"""Refine shared axes/pivot from pure roll and swivel recordings starting at home."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dualcam.config import DEFAULT_CONFIG
from dualcam.workflow import calibrate_axes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--roll", required=True, help="Paired --mode roll session; first paired frame at known home")
    parser.add_argument("--swivel", required=True, help="Paired --mode swivel session; first paired frame at the SAME known home")
    parser.add_argument("--output", required=True, help="New/empty diagnostic output directory; accepted axes go to axes.path in the rig config")
    parser.add_argument("--max-frames", type=int, default=180, help="Maximum paired frames per clip, starting at home")
    parser.add_argument("--roll-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--swivel-sign", type=int, choices=(-1, 1), default=1)
    args = parser.parse_args(argv)
    try:
        report = calibrate_axes(args.config, args.roll, args.swivel, args.output,
                                max_frames=args.max_frames, roll_sign=args.roll_sign,
                                swivel_sign=args.swivel_sign)
    except (OSError, ValueError) as exc:
        print(f"Axis calibration failed: {exc}", file=sys.stderr)
        return 2
    if not report["success"]:
        reasons = "; ".join(report["fit"]["diagnostics"].get("reasons", []))
        print(f"Axis calibration rejected: {reasons}\nDiagnostics: {Path(args.output) / 'report.json'}\nExisting accepted axes were not replaced.", file=sys.stderr)
        return 2
    print(f"Accepted axes: {report['accepted_axes_path']}\nDiagnostics: {Path(args.output) / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
