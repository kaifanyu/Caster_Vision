"""Render existing, bounded physics-fit candidates over both native videos."""
from pathlib import Path
import argparse
import json
import os
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blurtrack.physics_axis_video import render_physics_axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fps', type=float, default=30.)
    parser.add_argument('--slow-motion', type=float, default=.1)
    parser.add_argument('--loops', type=int, default=2)
    args = parser.parse_args()
    result = render_physics_axes(args.payload, args.output, args.fps, args.slow_motion, args.loops)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
