"""Test measured texture/exposure fits on existing native webcam recordings."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from blurtrack import ROOT,BASELINE
from blurtrack.blur_physics import BlurFitProblem,ray_shell_normals,color_signal
from blurtrack.observed_occlusion import observed_shell_mask,occlusion_metadata
from dualcam.config import load_config,load_rig,load_intrinsics,write_json,CAMERA_NAMES
from dualcam.workflow import load_axes,calibration_hashes
from dualcam.native_observations import load_native_tracks
from dualcam.session import SelectedVideo


def linear_rgb(bgr):
    x=bgr[...,::-1].astype(np.float32)/255
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)


def verify_atlas(atlas_dir,cfg,baseline,F,pivot):
    """Refuse stale measured texture rather than silently changing its gauge."""
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest=json.loads((Path(atlas_dir)/'atlas_manifest.json').read_text())
    report=json.loads((baseline/'results.json').read_text())
    if manifest['baseline_sha256']!=digest(baseline/'results.json'):
        raise ValueError('Texture atlas baseline hash differs; rebuild the atlas')
    if manifest.get('baseline_bundle_sha256',digest(baseline/'bundle.npz'))!=digest(baseline/'bundle.npz'):
        raise ValueError('Texture atlas trajectory bundle hash differs')
    if manifest['calibration_hashes']!=calibration_hashes(cfg) or manifest['geometry']!=cfg['geometry']:
        raise ValueError('Texture atlas calibration/geometry differs')
    if not np.allclose(report['F'],F) or not np.allclose(report['pivot'],pivot):
        raise ValueError('Current axes differ from texture atlas material coordinates')
    if manifest.get('axes_sha256',digest(cfg['axes']['path']))!=digest(cfg['axes']['path']):
        raise ValueError('Texture atlas axes hash differs')
    for entry in manifest['atlases']:
        path=Path(atlas_dir)/f'atlas_{entry["camera"]}_{entry["shell"]}.npz'
        if digest(path)!=entry['sha256']:raise ValueError(f'Texture atlas hash mismatch: {path}')
    return manifest


def prepare(center,shell,atlas_dir,side=96,frame_count=5,quadrature=9,exposure_scale=1.,camera_ids=(0,1),device='cuda'):
    cfg=load_config(BASELINE/'config/rig.yaml'); cameras,_,_=load_rig(cfg); axes=load_axes(cfg)
    F,pivot=axes['R_bc'],axes['pivot_c920_m']; geometry=cfg['geometry']
    native=load_native_tracks(BASELINE/'out/joint_native_cache')
    baseline=ROOT/'experiments/hybrid_dual'
    atlas_manifest=verify_atlas(atlas_dir,cfg,baseline,F,pivot)
    with np.load(baseline/'bundle.npz',allow_pickle=False) as saved:
        knots,angles=saved['knots'],saved['angles']; offset=float(saved['additional_brio_offset_s'])
    nodes,weights=np.polynomial.legendre.leggauss(quadrature); weights/=2
    atlas=[]
    for name in CAMERA_NAMES:
        path=Path(atlas_dir)/f'atlas_{name}_{("red","green")[shell]}.npz'
        with np.load(path,allow_pickle=False) as saved:atlas.append({k:saved[k].copy() for k in saved.files})
    normals=[];geo_valid=[];times=[];obs=[];masks=[];ids=[];train=[];frames=[];bounds=[]
    for ci in camera_ids:
        info=load_intrinsics(cfg['cameras'][CAMERA_NAMES[ci]])
        camtimes=np.asarray(native['times'][ci])+offset*(ci==1)
        middle=int(np.argmin(abs(camtimes-center))); half=frame_count//2
        indices=np.arange(max(0,middle-half),min(len(camtimes),middle+half+1))
        circle=native['circles'][ci]; cx,cy,r=circle
        x0=max(0,int(cx-r*1.02));x1=min(info['image_size'][0],int(cx+r*1.02))
        y0=max(0,int(cy-r*1.02));y1=min(info['image_size'][1],int(cy+r*1.02))
        yy,xx=np.indices((side,side));uv=np.column_stack((x0+(xx.ravel()+.5)*(x1-x0)/side-.5,y0+(yy.ravel()+.5)*(y1-y0)/side-.5))
        mapx,mapy=cv2.initUndistortRectifyMap(info['K'],info['dist'],None,info['K'],tuple(info['image_size']),cv2.CV_32FC1)
        video=SelectedVideo(native['videos'][ci],info['image_size'])
        exposure=cfg['cameras'][CAMERA_NAMES[ci]]['capture']['exposure_us']*1e-6*exposure_scale
        for fi in indices:
            raw=cv2.remap(video.read(int(fi)),mapx,mapy,cv2.INTER_LINEAR)
            rgb=linear_rgb(raw)
            # Sample linear light at the exact native pixel centers used by rays.
            measured=cv2.remap(rgb,uv[:,0].astype(np.float32).reshape(side,side),uv[:,1].astype(np.float32).reshape(side,side),cv2.INTER_LINEAR).reshape(-1,3)
            sampletime=camtimes[fi]+nodes*exposure/2
            ns=[];vs=[]
            for t in sampletime:
                n,v,_=ray_shell_normals(cameras[ci],uv,F,pivot,np.interp(t,knots,angles[:,0]),shell,geometry)
                ns.append(n);vs.append(v)
            # Dark yoke/background and clipped sensor values cannot be texture.
            bright=measured.mean(axis=1)
            pixel=(bright>.022)&(measured.max(axis=1)<.99)&np.asarray(vs).all(axis=0)
            pixel&=observed_shell_mask(ci,uv,(x0,y0,x1,y1))
            signal=color_signal(measured,shell)
            other=color_signal(measured,1-shell)
            pixel&=(other<.15) # reject confidently opposite-colored surfaces
            normals.append(ns);geo_valid.append(vs);times.append(sampletime)
            obs.append(signal);masks.append(pixel);ids.append(ci)
            train.append(((xx//6+yy//6)%4!=0).ravel())
            frames.append(dict(camera=CAMERA_NAMES[ci],camera_id=ci,frame=int(fi),time_s=float(camtimes[fi]),
                               exposure_s=exposure,observed_rgb=measured.reshape(side,side,3)))
            bounds.append([x0,y0,x1,y1])
        video.close()
    arrays=dict(normals=np.asarray(normals,np.float32),geometry_valid=np.asarray(geo_valid,bool),
                sample_times=np.asarray(times),sample_weights=weights,camera_ids=np.asarray(ids),observed=np.asarray(obs),
                pixel_valid=np.asarray(masks),train_mask=np.asarray(train))
    problem=BlurFitProblem(atlas,**arrays,reference_time_s=center,shell=shell,device=device)
    beta=float(np.interp(center,knots,angles[:,shell+1]))
    omega=float((np.interp(center+.05,knots,angles[:,shell+1])-np.interp(center-.05,knots,angles[:,shell+1]))/.1)
    metadata=dict(center_s=center,shell=('red','green')[shell],baseline_parameters_rad=[beta,omega],
        frames=[{k:v for k,v in f.items() if k!='observed_rgb'} for f in frames],pixel_bounds=bounds,side=side,
        quadrature=quadrature,exposure_scale=exposure_scale,camera_ids=list(camera_ids),
        nominal_exposures_s=[.0156,.008],additional_brio_offset_s=offset,
        fixed_geometry=geometry,reference_texture='Earlier measured frames in the frozen hybrid material coordinate system; fixed per camera; see atlas manifest for reference blur',
        observed_occlusion=occlusion_metadata(),
        atlas_directory=str(Path(atlas_dir).resolve()),
        atlas_manifest=atlas_manifest,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/'blurtrack/blur_physics.py',baseline/'bundle.npz',BASELINE/'config/rig.yaml']},
        time_model='Corrected host receive times treated as exposure centers; physical exposure midpoint/readout unverified',
        image_model='Approximate sRGB inverse response; linear RGB contrast; per-frame affine gain/offset; no rolling-shutter model',
        roll_policy='Fixed frozen hybrid trajectory roll throughout each exposure; roll uncertainty is not estimated',
        trajectory_initializer=str(baseline/'results.json'),
        accuracy_validated=False,applied_to_final_trajectory=False)
    return problem,metadata,frames,arrays


def seeds_for(problem,baseline,full=True):
    offsets=np.deg2rad(np.arange(-180,180,15) if full else [-15,0,15])
    speeds=np.deg2rad(np.arange(-2400,2401,100)) if full else np.asarray([baseline[1]-2,baseline[1],baseline[1]+2])
    scores=[]
    for delta in offsets:
        for speed in speeds:
            p=np.array([baseline[0]+delta,speed]);s=problem.score(p)
            scores.append(dict(parameters_rad=p.tolist(),**s))
    scores.sort(key=lambda s:s['objective'])
    selected=[np.asarray(baseline)]
    for score in scores:
        p=np.asarray(score['parameters_rad'])
        if score['coverage']<.3:continue
        if all(abs((p[0]-x[0]+np.pi)%(2*np.pi)-np.pi)>.15 or abs(p[1]-x[1])>1.5 for x in selected):
            selected.append(p)
        if len(selected)>=8:break
    return selected,scores


def save_contact(output,problem,metadata,frames,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if result.get('best_parameters_rad') is None:return
    baseline=metadata['baseline_parameters_rad'];best=result['best_parameters_rad']
    bp,bv=problem.predict(baseline);fp,fv=problem.predict(best)
    common=bv&fv&problem.pixel_valid.cpu().numpy()
    result['baseline_common_pixel_score']=problem.score(baseline,common)
    result['fitted_common_pixel_score']=problem.score(best,common)
    result['baseline_comparison_pixels']=int(common.sum())
    # Display each hypothesis on its OWN known texture support. This is useful
    # diagnostic imagery, but never a numerical comparison on unequal pixels.
    fit_score=problem.score(best);base_score=problem.score(baseline)
    eligible=problem.pixel_valid.cpu().numpy()
    result['camera_support']={CAMERA_NAMES[ci]:{
        'eligible_pixels':int(eligible[problem.camera_ids==ci].sum()),
        'candidate_pixels':int((fv&eligible)[problem.camera_ids==ci].sum()),
        'prior_pixels':int((bv&eligible)[problem.camera_ids==ci].sum()),
        'prior_candidate_common_pixels':int(common[problem.camera_ids==ci].sum())}
        for ci in metadata['camera_ids']}
    side=metadata['side'];which=[]
    for ci in metadata['camera_ids']:
        candidates=[i for i,f in enumerate(frames) if f['camera_id']==ci]
        which.append(min(candidates,key=lambda i:abs(frames[i]['time_s']-metadata['center_s'])))
    fig,axes=plt.subplots(len(which),5,figsize=(15,3.4*len(which)),squeeze=False)
    for row,i in enumerate(which):
        rgb=frames[i]['observed_rgb'];display=np.where(rgb<=.0031308,12.92*rgb,1.055*np.maximum(rgb,0)**(1/2.4)-.055)
        axes[row,0].imshow(np.clip(display,0,1));axes[row,0].set_title(f'{frames[i]["camera"]} raw {frames[i]["time_s"]:.3f}s')
        observed=problem.observed[i].cpu().numpy().reshape(side,side)
        old=base_score['gain'][i]*bp[i]+base_score['offset'][i]
        new=fit_score['gain'][i]*fp[i]+fit_score['offset'][i]
        use=(fv[i]&eligible[i]).reshape(side,side)
        for column,image,title,support in ((1,observed,'Measured linear contrast',eligible[i].reshape(side,side)),(2,old.reshape(side,side),'Prior on known texture only',(bv[i]&eligible[i]).reshape(side,side)),(3,new.reshape(side,side),'Candidate on known texture only',use)):
            axes[row,column].imshow(np.ma.masked_where(~support,image),cmap='magma',vmin=-.04,vmax=.35)
            axes[row,column].set_title(title)
        axes[row,4].imshow(np.ma.masked_where(~use,(new.reshape(side,side)-observed)),cmap='coolwarm',vmin=-.12,vmax=.12)
        axes[row,4].set_title('Fitted minus observed')
        for ax in axes[row]:ax.set_axis_off()
    fig.suptitle(f'{metadata["shell"]} at {metadata["center_s"]:.3f}s: illustrative hypothesis {np.rad2deg(best[1]):.1f} deg/s; {result["status"]}\nOwn support shown; {int(common.sum())} pixels shared with prior; {result.get("common_comparison_pixels",0)} shared across hypotheses. No validated real rate.',fontsize=11)
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=140);plt.close(fig)
    np.savez_compressed(output/'prediction.npz',observed=problem.observed.cpu().numpy(),baseline=bp,fitted=fp,common=common)


def gate_export(result):
    """Do not export an arbitrarily ranked hypothesis as a recovered rate."""
    result['recovered_rate_deg_s']=None
    if not result.get('common_comparison_available',False) and result.get('best_parameters_rad') is not None:
        result['illustrative_parameters_rad']=result['best_parameters_rad']
        result['best_parameters_rad']=None
        result['rejection_reason']='Too few identical measured pixels to compare competing motion hypotheses; no recovered rate.'
    elif result.get('status')!='candidate':
        if result.get('best_parameters_rad') is not None:
            result['illustrative_parameters_rad']=result['best_parameters_rad']
            result['best_parameters_rad']=None
        result['rejection_reason']='Motion hypotheses or texture evidence remain ambiguous; no recovered rate.'
    else:
        result['rejection_reason']='Photometric candidate only; no independent real angular reference.'


def run(args):
    if args.frames<1 or args.frames%2!=1:raise ValueError('--frames must be a positive odd count')
    if args.side<16 or args.quadrature<1 or args.exposure_scale<=0:raise ValueError('Invalid image side, quadrature or exposure scale')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise ValueError('Use a new output folder for a new experiment')
    torch.set_num_threads(4);cv2.setNumThreads(2)
    start=time.perf_counter()
    ids=(0,1) if args.cameras=='both' else (('c920','brio101').index(args.cameras),)
    problem,meta,frames,arrays=prepare(args.center,('red','green').index(args.shell),args.atlas,
        side=args.side,frame_count=args.frames,quadrature=args.quadrature,exposure_scale=args.exposure_scale,camera_ids=ids,device=args.device)
    print(f'Prepared {len(frames)} frames, {int(problem.pixel_valid.sum())} eligible pixels on {args.device}',flush=True)
    np.savez_compressed(out/'inputs.npz',**arrays)
    if args.initial:
        initial=np.deg2rad(args.initial);seeds,grid=seeds_for(problem,initial,full=False)
    else:seeds,grid=seeds_for(problem,meta['baseline_parameters_rad'])
    write_json(out/'coarse_grid.json',grid)
    print('Fitting seeds:',np.rad2deg(seeds).tolist(),flush=True)
    result=problem.fit(seeds,phase_radius_deg=20,speed_radius_deg_s=220,maxiter=args.iterations)
    save_contact(out,problem,meta,frames,result)
    gate_export(result)
    result.update(metadata=meta,total_runtime_s=time.perf_counter()-start)
    if torch.cuda.is_available() and args.device=='cuda':
        result['gpu']=dict(name=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(out/'results.json',result)
    if result.get('best_parameters_rad'):
        import csv
        with (out/'candidate_rates.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['time_s','camera','frame','candidate_phase_deg','candidate_rate_deg_s','status','turn_count_valid'])
            phase,speed=result['best_parameters_rad']
            for f in meta['frames']:
                writer.writerow([f['time_s'],f['camera'],f['frame'],np.rad2deg(phase+speed*(f['time_s']-args.center)),np.rad2deg(speed),'unvalidated_'+result['status'],False])
    print(json.dumps({k:result[k] for k in ('status','best_parameters_rad','common_comparison_pixels','runtime_s') if k in result}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--center',type=float,default=19.2);p.add_argument('--shell',choices=('red','green'),default='red')
    p.add_argument('--atlas',default=str(ROOT/'experiments/blur_physics/atlas'));p.add_argument('--output',required=True)
    p.add_argument('--side',type=int,default=96);p.add_argument('--frames',type=int,default=5)
    p.add_argument('--quadrature',type=int,default=9);p.add_argument('--exposure-scale',type=float,default=1.)
    p.add_argument('--cameras',choices=('both','c920','brio101'),default='both');p.add_argument('--device',default='cuda')
    p.add_argument('--iterations',type=int,default=60);p.add_argument('--initial',nargs=2,type=float,metavar=('PHASE_DEG','RATE_DEG_S'))
    run(p.parse_args())
