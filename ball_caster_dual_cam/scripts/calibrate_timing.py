#!/usr/bin/env python3
"""Measure relative camera latency from irregular shared light pulses, or inspect pairing."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dualcam.config import DEFAULT_CONFIG, load_config, write_json, write_yaml
from dualcam.session import SelectedVideo, load_session
from dualcam.timing import brightness_signal, estimate_offset


def pick_roi(directory, name):
    video = SelectedVideo(directory / f'{name}.avi')
    try:
        frame = video.read(0)
    finally:
        video.close()
    scale = min(1., 1100/frame.shape[1])
    title = f'{name}: select SAME light patch; ENTER accepts; C cancels'
    try:
        rectangle = cv2.selectROI(title, cv2.resize(frame, None, fx=scale, fy=scale), fromCenter=False)
    finally:
        cv2.destroyAllWindows()
    if not rectangle[2] or not rectangle[3]:
        raise ValueError('ROI selection canceled')
    return [int(round(v/scale)) for v in rectangle]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New/empty diagnostic directory')
    parser.add_argument('--pairing-only', action='store_true', help='Inspect timestamp pairing using current config; no light recording required')
    parser.add_argument('--pick-rois', action='store_true', help='Select the same light patch in both raw images')
    parser.add_argument('--c920-roi', nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'))
    parser.add_argument('--brio101-roi', nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'))
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError('Output must be a new or empty directory')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'kind': 'camera_timing', 'session': str(args.session.resolve()), 'success': False}
    try:
        if args.pairing_only:
            session = load_session(args.session, cfg)
            report.update(success=True, kind='pairing_inspection', pairing=session['report'])
            print(json.dumps(session['report'], indent=2))
        else:
            metadata = json.loads((args.session/'session.json').read_text())
            if metadata.get('status') not in ('complete', 'interrupted'):
                raise ValueError('Timing session must be finalized successfully')
            if metadata.get('mode') != 'timing':
                raise ValueError('Use a dedicated recording made with record.py --mode timing')
            signals = []
            rois = {}
            for name in ('c920', 'brio101'):
                profile = metadata.get('cameras', {}).get(name, {}).get('requested')
                if profile != cfg['cameras'][name]['capture']:
                    raise ValueError(f'{name}: timing recording capture profile differs from the current config; use its original config')
                roi = pick_roi(args.session, name) if args.pick_rois else getattr(args, name+'_roi')
                if roi is None:
                    raise ValueError('Use --pick-rois or provide both --c920-roi and --brio101-roi')
                rois[name] = roi
                times, values = brightness_signal(args.session, name, roi, [profile['width'], profile['height']])
                signals.append((times, values))
                with (args.output/f'{name}_brightness.csv').open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['timestamp_s', 'mean_brightness'])
                    writer.writerows(zip(times, values))
            report.update(estimate_offset(*signals[0], *signals[1]), rois=rois,
                          recorded_capture_profiles=metadata.get('cameras', {}),
                          original_timing=metadata.get('timing', {}))
            if report['success']:
                write_yaml(args.output/'timing_suggestion.yaml', {'timing': {
                    'brio_offset_s': report['brio_offset_s'],
                    'max_pair_skew_ms': cfg['timing'].get('max_pair_skew_ms', 12.),
                    'verified': False}})
            print(f"Measured brio_offset_s: {report['brio_offset_s']:+.6f} s; "
                  f"edge residual p95: {report['residual_p95_ms']:.1f} ms; "
                  f"estimated delay change: {report['estimated_delay_change_ms']:.1f} ms")
            print('Repeat with an independent pulse recording. Review report.json before copying the offset into rig.yaml.')
            for reason in report['reasons']:
                print('Rejected: '+reason)
    except Exception as exc:
        report.update(success=False, error=str(exc))
        raise
    finally:
        write_json(args.output/'report.json', report)
    return 0 if report['success'] else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, cv2.error) as error:
        raise SystemExit(f'Timing check failed: {error}')
