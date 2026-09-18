#!/usr/bin/env python3
"""Track arbitrary two-camera motion using native times, keyframes and one Kalman state."""
import argparse
from pathlib import Path
import sys
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dualcam.config import DEFAULT_CONFIG, load_config
from dualcam.fused_workflow import run_fused_motion, render_fused


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    p.add_argument('--session', help='Recorded motion session directory')
    p.add_argument('--c920-video', help='External AVI/MP4 in original C920 pixels')
    p.add_argument('--brio-video', help='External AVI/MP4 in original Brio pixels')
    p.add_argument('--c920-timestamps', help='Original frame_index,timestamp_s CSV on the shared clock')
    p.add_argument('--brio-timestamps', help='Original frame_index,timestamp_s CSV on the shared clock')
    p.add_argument('--initial-roll-deg', type=float, required=True,
                   help='Known starting roll relative to calibration; 0 only at home in both initial views')
    p.add_argument('--max-frames', type=int, help='Optional limit per camera; default processes all native frames')
    p.add_argument('--output', required=True, help='New or empty result directory')
    p.add_argument('--render', action='store_true', help='Also create both shared-state axis overlays')
    p.add_argument('--measurement-source', choices=('rotation', 'metric'), default='rotation',
                   help='rotation: image/keyframe rotations (approximate sphere); metric: stricter fixed-landmark pixel fit')
    p.add_argument('--opencv-threads', type=int, default=2, help='Bound OpenCV worker count (default 2)')
    args = p.parse_args(argv)
    if args.opencv_threads < 1:
        p.error('--opencv-threads must be positive')
    cv2.setNumThreads(args.opencv_threads)
    external = (args.c920_video, args.brio_video, args.c920_timestamps, args.brio_timestamps)
    if bool(args.session) == any(x is not None for x in external) or (not args.session and not all(external)):
        p.error('Use --session OR all four external video/timestamp arguments')
    try:
        result = run_fused_motion(args.config, args.output, initial_roll_deg=args.initial_roll_deg,
                                  session=args.session, max_frames=args.max_frames,
                                  measurement_source=args.measurement_source,
                                  videos=None if args.session else external[:2],
                                  timestamps=None if args.session else external[2:])
        if args.render:
            render_fused(result, load_config(args.config), Path(args.output)/'overlays')
    except (OSError, ValueError) as exc:
        print(f'Motion tracking failed: {exc}', file=sys.stderr)
        return 2
    print(f'Results: {Path(args.output)/"results.json"}\nTrajectory: {Path(args.output)/"results.csv"}')
    print(f'Coverage (per native event): {result["summary"]["coverage"]}')
    print('Inspect vision/predicted/unresolved labels. Completion does not independently validate physical accuracy.')
    return 0 if result['success'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
