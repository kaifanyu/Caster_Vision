"""Write the local physics-axis comparison viewer and optional video montage."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blurtrack import ROOT
from blurtrack.physics_preview import write_preview


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/physics_axis_preview')
    parser.add_argument('--render',action='store_true')
    args=parser.parse_args()
    payload=write_preview(args.output)
    print(f'Wrote {len(payload["cases"])} local comparisons: {args.output / "orientation_3d.html"}',flush=True)
    if args.render:
        from blurtrack.physics_axis_video import render_physics_axes
        render_physics_axes(args.output/'preview_data.json',args.output)


if __name__=='__main__':main()
