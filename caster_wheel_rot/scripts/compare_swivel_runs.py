#!/usr/bin/env python
"""Compare accepted coverage and held-out reference-dot agreement of real runs.

The reference is a separate visual cue, not external ground truth: it shares
the camera/plane calibration and can itself be biased or occluded. For a fair
paired error comparison both methods are scored on the SAME frame intervals.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import cv2
import numpy as np
import yaml

from swivel.geometry import SwivelGeometry
from swivel.reference import detect_reference_phase


def reference_track(video, result, reference_config):
    metadata=result['metadata']
    g=SwivelGeometry.from_config(metadata['camera'],metadata['swivel_calibration']['geometry'])
    K=np.array(metadata['camera']['K']);dist=np.array(metadata['camera']['dist'])
    phase=np.full(len(result['psi_rad']),np.nan)
    radius=np.full(len(phase),np.nan)
    cap=cv2.VideoCapture(str(video));maps=None;index=0
    try:
        while index<len(phase):
            ok,image=cap.read()
            if not ok:break
            if result['psi_valid'][index]:
                if maps is None:maps=cv2.initUndistortRectifyMap(K,dist,None,K,(image.shape[1],image.shape[0]),cv2.CV_32FC1)
                image=cv2.remap(image,*maps,cv2.INTER_LINEAR)
                psi=result['psi_rad'][index]
                if g.sidewall_view_confidence(psi) >= .35:
                    observation=detect_reference_phase(image,psi,g,K,config=reference_config)
                    if observation.valid:
                        phase[index]=observation.phase_wrapped;radius[index]=observation.radial_fraction
            index+=1
    finally:cap.release()
    delta=(np.diff(phase)+np.pi)%(2*np.pi)-np.pi
    good=np.isfinite(delta) & (np.abs(delta)<np.deg2rad(20)) & (np.abs(np.diff(radius))<.12)
    return delta,good


def run_comparison(before_dir,after_dir,config,output,names):
    cfg=yaml.safe_load(Path(config).read_text())
    reference=dict(cfg['swivel']['reference_dot'])
    reference.update({'min_radial_fraction':.35,'max_radial_fraction':1.05})
    rows=[];series={}
    for name in names:
        a=json.loads((Path(before_dir)/name/'results.json').read_text())
        b=json.loads((Path(after_dir)/name/'results.json').read_text())
        if a['timestamps_s']!=b['timestamps_s']:
            raise ValueError(f'{name}: runs have different frame timestamps')
        ca=np.array([q['roll_valid'] for q in a['interval_quality']],bool)
        cb=np.array([q['roll_valid'] for q in b['interval_quality']],bool)
        da=np.diff(a['phi_rad'])
        db=np.array([q.get('delta_phi_rad',np.nan) if q.get('delta_phi_rad') is not None else np.nan for q in b['interval_quality']],float)
        # Use one reference geometry/pose track for both errors; neither method
        # gets its own target or a cherry-picked easier subset.
        ref,rgood=reference_track(b['metadata']['input'],b,reference)
        shared=ca & cb & rgood
        wrap=lambda x:np.rad2deg((x+np.pi)%(2*np.pi)-np.pi)
        ea,eb=wrap(da-ref),wrap(db-ref)
        def median(values,mask):return float(np.median(np.abs(values[mask]))) if mask.any() else None
        row={'clip':name,'frames':len(a['timestamps_s']),
            'before_tag_coverage':float(np.mean(a['psi_valid'])),
            'after_tag_coverage':float(np.mean(b['psi_valid'])),
            'before_roll_coverage':float(ca.mean()),'after_roll_coverage':float(cb.mean()),
            'shared_reference_intervals':int(shared.sum()),
            'before_reference_median_abs_error_deg':median(ea,shared),
            'after_reference_median_abs_error_deg':median(eb,shared),
            'after_reference_intervals':int((cb&rgood).sum()),
            'after_all_reference_median_abs_error_deg':median(eb,cb&rgood),
            'after_accumulated_phi_complete':bool(b['phi_valid'][-1]),
            'before_observed_roll_deg':float(np.rad2deg(a['phi_rad'][-1]-a['phi_rad'][0])),
            'after_observed_roll_deg':float(np.rad2deg(b['phi_rad'][-1]-b['phi_rad'][0]))}
        rows.append(row)
        series[name]={'timestamps_s':a['timestamps_s'][:-1],
                      'before_deg':np.rad2deg(da).tolist(),
                      'after_deg':[float(x) if np.isfinite(x) else None for x in np.rad2deg(db)],
                      'reference_deg':[float(x) if ok else None for x,ok in zip(np.rad2deg(ref),rgood)],
                      'shared_reference_intervals':shared.tolist()}
        print(json.dumps(row),flush=True)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    note='Coverage is acceptance, not accuracy. Reference-dot agreement shares calibration and is not external ground truth. The sweep was used to calibrate the tag relationship; roll/clip were not. No reference-dot correction was enabled in these benchmark runs.'
    (output/'comparison.json').write_text(json.dumps({'note':note,'runs':rows},indent=2),encoding='utf-8')
    (output/'reference_series.json').write_text(json.dumps(series,indent=2),encoding='utf-8')
    lines=['Real clip tracking comparison','',note,'',
        '| Clip | Tag before / after | Roll before / after | Paired reference error before / after | Paired intervals |',
        '| --- | --- | --- | --- | ---: |']
    for r in rows:
        err=lambda key:'unavailable' if r[key] is None else f'{r[key]:.3f} deg'
        lines.append(f"| {r['clip']} | {r['before_tag_coverage']:.1%} / {r['after_tag_coverage']:.1%} | {r['before_roll_coverage']:.1%} / {r['after_roll_coverage']:.1%} | {err('before_reference_median_abs_error_deg')} / {err('after_reference_median_abs_error_deg')} | {r['shared_reference_intervals']} |")
    (output/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(names),1,figsize=(12,3.3*len(names)),squeeze=False)
    for ax,name in zip(axes[:,0],names):
        s=series[name]
        for key,label in [('before_deg','Before'),('after_deg','After'),('reference_deg','Green reference (visible only)')]:
            ax.plot(s['timestamps_s'],np.asarray(s[key],float),label=label,linewidth=1,alpha=.8)
        ax.set(title=name,xlabel='Time (s)',ylabel='Roll increment (deg/frame)');ax.legend(loc='upper right')
    fig.tight_layout();fig.savefig(output/'roll_increment_comparison.png',dpi=150);plt.close(fig)
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--clips',nargs='+',default=['roll','clip','sweep','psi_zero'])
    args=p.parse_args(argv);cv2.setNumThreads(2)
    run_comparison(args.before,args.after,args.config,args.output,args.clips)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
