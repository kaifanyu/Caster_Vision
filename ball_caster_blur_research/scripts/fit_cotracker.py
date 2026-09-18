"""Fit CoTracker/KLT observations to the shared gap and independent shell spins."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blurtrack import ROOT, BASELINE
from blurtrack.metric import fit_experiment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-cache',default=str(BASELINE/'out/joint_native_cache'))
    p.add_argument('--tracks',default=str(ROOT/'experiments/cotracker_native'))
    p.add_argument('--baseline',default=str(BASELINE/'out/orientation_final/results.json'))
    p.add_argument('--output',required=True)
    p.add_argument('--source',choices=('cotracker','klt','hybrid'),default='cotracker')
    p.add_argument('--cameras',choices=('both','c920','brio101'),default='both')
    p.add_argument('--max-nfev',type=int,default=120)
    p.add_argument('--acceleration-std',type=float,default=500.)
    p.add_argument('--holdout',action='store_true')
    p.add_argument('--refit-timing',action='store_true')
    p.add_argument('--render',action='store_true')
    args = p.parse_args()
    ids = (0,1) if args.cameras=='both' else (('c920','brio101').index(args.cameras),)
    r = fit_experiment(args.native_cache,args.tracks,args.baseline,args.output,source=args.source,
        camera_ids=ids,max_nfev=args.max_nfev,holdout=args.holdout,render=args.render,
        freeze_timing=not args.refit_timing,acceleration_std_deg_s2=args.acceleration_std)
    print('Coverage:',r['summary']['coverage'])
    print('Results:',str(Path(args.output)/'results.json'))


if __name__ == '__main__': main()
