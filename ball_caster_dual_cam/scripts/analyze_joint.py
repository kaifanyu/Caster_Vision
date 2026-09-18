#!/usr/bin/env python3
"""Fit one full offline metric trajectory to both native camera recordings."""
import argparse
from pathlib import Path
import sys
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dualcam.config import DEFAULT_CONFIG
from dualcam.offline_workflow import run_joint, finalize_saved_joint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--session', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache', help='Checked native-observation cache directory')
    parser.add_argument('--initial-roll-deg', type=float, default=8., help='Approximate signed roll relative to calibrated home; refined with a 3-degree prior')
    parser.add_argument('--no-auto-initial', action='store_true', help='Skip initial two-view rim check')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--max-nfev', type=int, default=220)
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument('--resume-from', help='Reuse a saved joint results.json for conditional spin recovery and final refinement')
    continuation.add_argument('--finalize-from', help='Validate and re-export a saved bundle with current phase diagnostics, without refitting')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args(argv)
    if args.max_nfev < 1: parser.error('--max-nfev must be positive')
    cv2.setNumThreads(2)
    if args.finalize_from:
        result = finalize_saved_joint(args.config, args.session, args.finalize_from, args.output,
                                       cache=args.cache, render=args.render)
    else:
        result = run_joint(args.config, args.session, args.output, cache=args.cache,
                       initial_roll_deg=args.initial_roll_deg, max_frames=args.max_frames,
                       max_nfev=args.max_nfev, auto_initial=not args.no_auto_initial, render=args.render,
                       resume_from=args.resume_from)
    print(f'Results: {Path(args.output)/"results.json"}')
    print(f'Fitted initial roll: {result["initial_roll_deg"]:.3f} deg')
    print(result['summary']['coverage'])
    return 0 if result['success'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
