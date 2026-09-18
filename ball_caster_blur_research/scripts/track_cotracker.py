"""Run official CoTracker3 offline on native camera streams using the local GPU."""
from pathlib import Path
import argparse
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blurtrack import ROOT, BASELINE
from blurtrack.observations import collect_cotracker
from blurtrack.cotracker_backend import OfflineCoTracker
from dualcam.config import load_config
from dualcam.native_observations import collect_native_tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', default=str(BASELINE/'data/run'))
    parser.add_argument('--native-cache', default=str(BASELINE/'out/joint_native_cache'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default=str(BASELINE/'config/rig.yaml'))
    parser.add_argument('--repo', default=str(ROOT/'third_party/co-tracker'))
    parser.add_argument('--checkpoint', default=str(ROOT/'checkpoints/scaled_offline.pth'))
    parser.add_argument('--cameras', choices=('both', 'c920', 'brio101'), default='both')
    parser.add_argument('--window', type=int, default=60)
    parser.add_argument('--stride', type=int, default=40)
    parser.add_argument('--points', type=int, default=32)
    parser.add_argument('--side', type=int, default=640)
    parser.add_argument('--start', type=float)
    parser.add_argument('--end', type=float)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    import cv2, torch
    cv2.setNumThreads(2); torch.set_num_threads(4)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA required for this experiment but unavailable')
    cfg = load_config(args.config)
    native = collect_native_tracks(cfg, args.session, output=args.native_cache)
    backend = OfflineCoTracker(args.repo, args.checkpoint, device=args.device)
    ids = (0, 1) if args.cameras == 'both' else (('c920', 'brio101').index(args.cameras),)
    tracked = collect_cotracker(cfg, native, args.output, backend, length=args.window, stride=args.stride,
        points_per_shell=args.points, resized_side=args.side, camera_ids=ids, start_s=args.start, end_s=args.end)
    print(f'Finished: {len(tracked["observations"]["uv"])} observations in {args.output}', flush=True)


if __name__ == '__main__': main()
