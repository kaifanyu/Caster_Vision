"""Recompute rate diagnostics and exports from a saved physical fit, without refitting."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blurtrack import BASELINE
from blurtrack.rate_analysis import annotate_rates
from blurtrack.exports import write_artifacts
from dualcam.config import load_config, load_rig
from dualcam.native_observations import load_native_tracks
from dualcam.spin_initialization import reseed_spins


def refresh(output, render=False):
    output = Path(output)
    report = json.loads((output/'results.json').read_text())
    with np.load(output/'bundle.npz',allow_pickle=False) as saved:
        bundle = {k:saved[k] for k in saved.files}
    evidence_path = output/'strict_rate_evidence.npz'
    if not evidence_path.exists():
        cfg=load_config(BASELINE/'config/rig.yaml'); cameras,_,_=load_rig(cfg)
        native=load_native_tracks(BASELINE/'out/joint_native_cache')
        ids=report['experiment']['camera_ids']
        shifts=report['experiment'].get('fixed_clock_shifts_s',[0.,0.])
        native={**native,'times':[native['times'][ci]+shifts[ci] if ci in ids else np.empty(0) for ci in range(2)]}
        with np.load(output/'seed.npz',allow_pickle=False) as saved:
            seed={k:saved[k] for k in saved.files}
        seed['events']=[(float(t),int(ci),int(fi)) for t,ci,fi in seed['events']]
        inlier=bundle['observation_inlier']
        obs={k:bundle['observations_'+k][inlier] for k in ('camera','shell','frame','uv')}
        obs['track']=bundle['landmark_keys'][bundle['observation_landmark'][inlier],2]
        obs['weight']=np.ones(inlier.sum())
        strict={**native,'observations':obs}
        conditioned,_=reseed_spins(strict,seed,cameras,np.asarray(report['F']),np.asarray(report['pivot']),
            cfg['geometry'],bundle['knots'],bundle['angles'],float(bundle['additional_brio_offset_s']))
        evidence=conditioned['spin_evidence']
        np.savez_compressed(evidence_path,**{k:v for k,v in evidence.items() if isinstance(v,np.ndarray)})
    else:
        with np.load(evidence_path,allow_pickle=False) as saved:
            evidence={k:saved[k] for k in saved.files}
    report=annotate_rates(report,bundle['knots'],bundle['angles'],evidence)
    write_artifacts(output,report,render=render)
    print(output,report['rate_analysis']['coverage'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('outputs',nargs='+');p.add_argument('--render',action='store_true')
    a=p.parse_args()
    for path in a.outputs: refresh(path,a.render)
