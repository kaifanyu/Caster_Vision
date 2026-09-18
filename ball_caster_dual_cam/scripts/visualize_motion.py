#!/usr/bin/env python3
"""Create an interactive offline 3D viewer from one saved fused trajectory."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dualcam.visualization import write_motion_viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True,
                        help='Existing fused_motion results.json from track_motion.py')
    parser.add_argument('--output', type=Path, required=True,
                        help='New or empty standalone .html file')
    args = parser.parse_args()
    output = write_motion_viewer(args.results, args.output)
    print(f'3D viewer: {output}')
    print('Open this HTML file in a browser. No server, videos, or network connection is needed.')
    print('The viewer preserves vision/predicted/unresolved labels; it does not repair missing motion.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, TypeError) as error:
        raise SystemExit(f'3D viewer export failed: {error}')
