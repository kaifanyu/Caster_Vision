"""Joint separated-hemisphere fit for learned and conventional observations."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from . import BASELINE
from .observations import hybrid_observations
from scipy.spatial import cKDTree
from dualcam.config import load_config, load_rig, write_json
from dualcam.workflow import load_axes, _output_directory, calibration_hashes
from dualcam.offline import NativeBundle, event_table
from dualcam.model import initialize_surface_point
from dualcam.native_observations import load_native_tracks
from dualcam.offline_workflow import summarize_solution
from dualcam.offline_finalization import finalize_phase_estimates, write_offline_csvs
from dualcam.visualization import write_motion_viewer


def exclude_reference_tracks(obs, reference, radius=6.):
    """Drop an entire candidate identity if it overlaps any reference sample.

    Spatial proximity is only a conservative cross-tracker identity proxy;
    evaluation still reports the shared full-data initialization caveat.
    """
    identities = np.column_stack([obs[k] for k in ('camera','shell','track')])
    keys, inverse = np.unique(identities, axis=0, return_inverse=True)
    contaminated = np.zeros(len(keys), bool)
    for ci in (0, 1):
        for h in (0, 1):
            a = (obs['camera']==ci)&(obs['shell']==h)
            b = (reference['camera']==ci)&(reference['shell']==h)
            for fi in np.unique(reference['frame'][b]):
                rows = np.flatnonzero(a & (obs['frame']==fi))
                refs = reference['uv'][b & (reference['frame']==fi)]
                if len(rows):
                    hit = cKDTree(refs).query(obs['uv'][rows])[0] <= radius
                    contaminated[inverse[rows[hit]]] = True
    keep = ~contaminated[inverse]
    return {k:v[keep] for k,v in obs.items()}


def initialize_metric(data, cameras, F, pivot, geometry, knots, q, offset, initial_prior_deg=8.):
    """Use accepted trajectory solely as a nonlinear initialization, not a factor."""
    obs = data['observations']; events = event_table(data)
    keys, landmark = np.unique(np.column_stack([obs[k] for k in ('camera','shell','track')]), axis=0, return_inverse=True)
    times = np.asarray([data['times'][ci][fi]+offset*(ci==1) for ci,fi in zip(obs['camera'], obs['frame'])])
    poses = np.column_stack([np.interp(times,knots,q[:,k]) for k in range(3)])
    points = np.full((len(keys),3),np.nan)
    for li in range(len(keys)):
        rows = np.flatnonzero(landmark==li)
        # Strongest observation initializes a material coordinate, whose final
        # correctness must be established independently by image residuals.
        i = rows[np.argmax(obs['weight'][rows])]; ci,h = int(obs['camera'][i]),int(obs['shell'][i])
        points[li] = initialize_surface_point(cameras[ci],obs['uv'][i],F,pivot,poses[i],h,
            geometry['radius_m'],geometry['gap_m'],geometry.get('red_shell_sign',1))
    event_times = np.array([e[0] for e in events])
    return {'events': events, 'angles': np.column_stack([np.interp(event_times,knots,q[:,k]) for k in range(3)]),
            'points':points,'keys':keys,'landmark':landmark,
            'initial_brio_offset_s':offset,'initial_roll_prior_rad':np.deg2rad(initial_prior_deg),
            'per_camera':[{},{}],'quality':[None]*len(events),'components':[[] for _ in events]}


def refine(data, seed, cameras, F, pivot, geometry, *, max_nfev=120, freeze_timing=True,
           acceleration_std_deg_s2=500., progress=print):
    problem = NativeBundle(data,seed,cameras,F,pivot,geometry,acceleration_std_deg_s2=acceleration_std_deg_s2)
    x = problem.x0.copy(); active = np.arange(len(x))
    if freeze_timing: active = active[active != problem.timing_id]
    def expand(z):
        full = x.copy(); full[active] = z; return full
    scale = np.ones(len(x)); scale[problem.timing_id] = .03
    stages = []; start = time.perf_counter()
    for stage in range(2):
        if progress: progress(f'Metric fit {stage+1}/2: {len(problem.times)} observations, {len(problem.keys)} landmarks, {len(problem.knots)} knots',flush=True)
        result = least_squares(lambda z:problem.evaluate(expand(z)),x[active],
            jac=lambda z:problem.evaluate(expand(z),True)[:,active],
            bounds=(problem.lower[active],problem.upper[active]),loss='soft_l1',f_scale=1.5,
            x_scale=scale[active],max_nfev=max_nfev,ftol=2e-5,xtol=1e-7,gtol=1e-6,
            tr_options={'atol':1e-6,'btol':1e-6,'maxiter':300})
        x[active] = result.x
        errors = np.linalg.norm(problem.pixels(x)[0],axis=1); visible = problem.visibility(x)
        stages.append({'nfev':result.nfev,'cost':result.cost,'converged':result.success,
                       'message':result.message,'median_error_px':float(np.median(errors)),
                       'fraction_within_3px':float(np.mean((errors<=3)&visible)),
                       'initial_roll_deg':float(np.rad2deg(problem.unpack(x)[0][0,0])),
                       'additional_brio_offset_s':float(x[problem.timing_id]),
                       'elapsed_s':time.perf_counter()-start})
        if progress: progress(str(stages[-1]),flush=True)
        problem.pixel_scale = np.where((errors<=5)&visible,1.,.03)
    return problem,x,stages


def fit_experiment(native_path, learned_path, baseline_report, output, *, source='cotracker',
                   camera_ids=(0,1), max_nfev=120, holdout=False, render=False,
                   config_path=None, freeze_timing=True, acceleration_std_deg_s2=500.,progress=print):
    from .evaluation import landmark_split, subset_data, score_heldout
    if render and len(camera_ids)!=2:
        raise ValueError('Combined stereo replay requires both cameras; monocular fits still export their 3D viewer')
    cfg = load_config(config_path or BASELINE/'config/rig.yaml')
    cameras,_,_ = load_rig(cfg); axes = load_axes(cfg); F,pivot = axes['R_bc'],axes['pivot_c920_m']
    native = load_native_tracks(native_path)
    learned = load_native_tracks(learned_path) if source != 'klt' and learned_path else None
    if source != 'klt' and learned is None:
        raise ValueError('Learned observations are required for this source')
    if learned is not None and learned['provenance']['base_native_provenance'] != native['provenance']:
        raise ValueError('Learned and conventional tracks have different input provenance')
    previous = json.loads(Path(baseline_report).read_text())
    if previous.get('calibration_hashes') != calibration_hashes(cfg):
        raise ValueError('Current calibration differs from the baseline initializer')
    if not np.allclose(previous['F'], F) or not np.allclose(previous['pivot'], pivot):
        raise ValueError('Baseline rig axes or pivot differ from the current configuration')
    if previous.get('geometry') != cfg['geometry']:
        raise ValueError('Baseline radius/gap configuration differs from the current configuration')
    if previous.get('observation_cache_provenance') != native['provenance']:
        raise ValueError('Baseline initializer belongs to different native recordings')
    with np.load(Path(baseline_report).parent/'bundle.npz',allow_pickle=False) as saved:
        knots,q,offset = saved['knots'].copy(),saved['angles'].copy(),float(saved['additional_brio_offset_s'])
    if source == 'klt': data = {**native,'observations':{k:v.copy() for k,v in native['observations'].items()}}
    elif source == 'cotracker': data = {**native,'report':learned['report'],'observations':{k:v.copy() for k,v in learned['observations'].items()}}
    elif source == 'hybrid': data = {**native,'report':{'klt':native['report'],'cotracker':learned['report']},'observations':hybrid_observations(native['observations'],learned['observations'])}
    else: raise ValueError('source must be klt, cotracker, or hybrid')
    # The same conventional material tracks form the reference for every branch.
    split = landmark_split(native)
    reference = subset_data(native,split['holdout_mask'])
    if holdout:
        if source == 'klt': data = subset_data(data,split['train_mask'])
        else: data['observations'] = exclude_reference_tracks(data['observations'],reference['observations'])
    mask = np.isin(data['observations']['camera'],camera_ids)
    data['observations'] = {k:v[mask] for k,v in data['observations'].items()}
    from .camera_ablation import prepare_camera_data
    if len(camera_ids)==1 and not freeze_timing:
        raise ValueError('A single-camera ablation cannot refit inter-camera timing')
    data,seed_q,fit_offset,clock_shifts = prepare_camera_data(data,camera_ids,knots,q,offset)
    seed = initialize_metric(data,cameras,F,pivot,cfg['geometry'],knots,seed_q,fit_offset,
                             previous.get('supplied_initial_roll_deg',8.))
    out = _output_directory(output)
    problem,x,stages = refine(data,seed,cameras,F,pivot,cfg['geometry'],max_nfev=max_nfev,
        freeze_timing=freeze_timing,acceleration_std_deg_s2=acceleration_std_deg_s2,progress=progress)
    angles,polar = problem.unpack(x)
    frames,summary,inlier = summarize_solution(problem,x,data,seed,F,pivot)
    summary.update(optimization_stages=stages,optimizer_converged=bool(stages[-1]['converged']),
                   native_tracking=data['report'],acceleration_std_deg_s2=acceleration_std_deg_s2)
    report = {k:copy.deepcopy(previous[k]) for k in ('supplied_initial_roll_deg',) if k in previous}
    report.update({'kind':'fused_motion','method':source+'_separated_hemisphere_bundle','measurement_source':source,
              'status':'completed','frames':frames,'summary':summary,'config':cfg,'geometry':cfg['geometry'],
              'initial_roll_deg':float(np.rad2deg(angles[0,0])), 'F':F,'pivot':pivot,
              'timestamp_origin_s':native['timestamp_origin_s']+summary['output_clock_origin_shift_s'],
              'experiment':{'source':source,'camera_ids':list(camera_ids),'heldout_reference_excluded':holdout,
                'timing_frozen':freeze_timing,'baseline_initialization':str(Path(baseline_report).resolve()),
                'fixed_clock_shifts_s':clock_shifts,
                'phase_gauge':'first active observed image' if len(camera_ids)==1 else 'common initial native event',
                'effective_original_brio_offset_s':float(x[problem.timing_id])+clock_shifts[1],
                'initialization_scope':'Common baseline initialization only; no baseline trajectory residuals. Sensor ablations are conditional refinements, not independent from-scratch reconstructions.',
                'learned_provenance':None if learned is None else learned['provenance'],
                'projection_model':'Two separated radius-100mm hemispheres;20mm gap;shared roll and independent shell spins; original calibrated projections.',
                'fit_device':'CPU SciPy analytic sparse metric bundle; neural observations computed on GPU'},
              'calibration_hashes':calibration_hashes(cfg),'videos':native['videos'],'use_video_manifest':True})
    np.savez_compressed(out/'bundle.npz',knots=problem.knots,angles=angles,surface_polar=polar,
        optimized_parameters=x,additional_brio_offset_s=x[problem.timing_id],landmark_keys=problem.keys,
        observations_camera=problem.obs['camera'],observations_shell=problem.obs['shell'],
        observations_frame=problem.obs['frame'],observations_uv=problem.obs['uv'],
        observation_landmark=problem.landmark,observation_inlier=inlier,original_observation_rows=problem.original_rows)
    np.savez_compressed(out/'seed.npz',**{k:v for k,v in seed.items() if k not in ('per_camera','quality','components')})
    write_json(out/'fit_diagnostics.json',summary)
    report = finalize_phase_estimates(report,data,seed,cameras,F,pivot,cfg['geometry'],problem.knots,
                                      angles,float(x[problem.timing_id]),out)
    report['measurement_source'] = source+'_metric_with_phase_estimates'
    # Rate support uses only final accepted metric pixels; inferred hidden tracks
    # and rejected neural points cannot validate a derivative.
    from dualcam.spin_initialization import reseed_spins
    strict_data = {**data,'observations':{k:v[inlier] for k,v in problem.obs.items()}}
    conditioned,rate_diagnostics = reseed_spins(strict_data,seed,cameras,F,pivot,cfg['geometry'],
        problem.knots,angles,float(x[problem.timing_id]))
    np.savez_compressed(out/'strict_rate_evidence.npz', **{k:v for k,v in conditioned['spin_evidence'].items() if isinstance(v,np.ndarray)})
    from .rate_analysis import annotate_rates
    report = annotate_rates(report,problem.knots,angles,conditioned['spin_evidence'])
    report['rate_diagnostics'] = rate_diagnostics
    if progress: progress('Scoring held-out conventional material tracks.',flush=True)
    evaluation = score_heldout(native,cameras,F,pivot,cfg['geometry'],problem.knots,angles,
        float(x[problem.timing_id])+clock_shifts[1],heldout_keys=split['heldout_keys'],trajectory_excludes_heldout=holdout)
    evaluation['initialization_caveat'] = ('The common nonlinear initializer came from the accepted full-data fit. '
        'Whole tracks overlapping held-out reference pixels are excluded from this optimization when requested. '
        'Spatial overlap is a conservative identity proxy across algorithms. This is a '
        'conditional consistency screen, not a fully independent generalization or physical accuracy test.')
    write_json(out/'evaluation.json',evaluation)
    report['evaluation'] = evaluation
    for frame in report['frames']:
        ci = ('c920','brio101').index(frame['camera'])
        frame['receive_time_s'] = float(native['times'][ci][frame['source_frame']])
        frame['camera_time_correction_s'] = clock_shifts[ci]+float(x[problem.timing_id])*(ci==1)
    from .exports import write_artifacts
    write_artifacts(out,report,render=render)
    return report
