"""Reproducible offline workflow and evidence-based trajectory exports."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .config import CAMERA_NAMES, load_config, load_rig, write_json
from .fused_workflow import write_fused_csv
from .initial_pose import fit_initial_pose
from .model import rotation_x, rotation_z, polar_to_points
from .native_observations import collect_native_tracks
from .offline import seed_trajectory, refine_native
from .roll_initialization import reseed_roll
from .workflow import load_axes, calibration_hashes, calibration_matches, _output_directory, _record_provenance


def _anchored_knots(problem, inlier, component):
    """Accepted material tracks must connect fitted knots to the initial gauge."""
    n = len(problem.knots); parent = np.arange(n+len(problem.keys))
    def root(a):
        while a != parent[a]:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    selected = inlier.copy()
    if component: selected &= problem.obs['shell'] == component-1
    for i in np.flatnonzero(selected):
        landmark = n+problem.landmark[i]
        for knot, weight in ((problem.left[i], 1-problem.fraction[i]),
                             (problem.left[i]+1, problem.fraction[i])):
            if weight > 1e-8: parent[root(int(knot))] = root(int(landmark))
    anchor = root(0)
    return np.array([root(i) == anchor for i in range(n)])


def summarize_solution(problem, x, data, seed, F, pivot):
    offset = float(x[problem.timing_id])
    events = sorted((t+offset*(ci == 1), ci, fi) for t, ci, fi in seed['events'])
    times = np.asarray([e[0] for e in events]); origin = float(times[0])
    obs = problem.obs
    q = problem.angles_at(x, times)
    problem.left, problem.fraction = problem.observation_interpolation(x)
    errors = np.linalg.norm(problem.pixels(x)[0], axis=1)
    inlier = (errors <= 3.) & problem.visibility(x)
    # Two lucky endpoint pixels do not establish a persistent material feature.
    counts = np.bincount(problem.landmark[inlier], minlength=len(problem.keys))
    inlier &= counts[problem.landmark] >= 3
    event_index = {(ci, fi): ei for ei, (_, ci, fi) in enumerate(events)}
    observation_event = np.asarray([event_index[ci, fi] for ci, fi in zip(obs['camera'], obs['frame'])])
    support = np.zeros((len(events), 3), bool)
    support_counts = np.zeros((len(events), 3), int)
    _, polar = problem.unpack(x)
    points = polar_to_points(polar, problem.signs)
    for ei in range(len(events)):
        selected = inlier & (observation_event == ei)
        for component in range(3):
            use = selected if component == 0 else selected & (obs['shell'] == component-1)
            p = points[problem.landmark[use]]
            support_counts[ei, component] = len(p)
            if len(p) >= 6:
                spread = np.linalg.svd(p-p.mean(axis=0), compute_uv=False)[1]/np.sqrt(len(p))
                support[ei, component] = spread >= .025
    left, fraction = problem.interpolation(times)
    anchored = np.column_stack([_anchored_knots(problem, inlier, k) for k in range(3)])
    connected = anchored[left] & anchored[left+1]
    # At a knot, an unconnected neighbor has zero contribution.
    connected[fraction < 1e-8] = anchored[left[fraction < 1e-8]]
    connected[fraction > 1-1e-8] = anchored[left[fraction > 1-1e-8]+1]
    camera_indices = np.array([e[1] for e in events])
    distance = np.full((len(events), 2, 3), np.inf)
    for ci in range(2):
        for k in range(3):
            supported_times = times[support[:, k] & (camera_indices == ci)]
            if not len(supported_times): continue
            ids = np.clip(np.searchsorted(supported_times, times), 0, len(supported_times)-1)
            distance[:, ci, k] = np.minimum(abs(times-supported_times[ids]),
                                            abs(times-supported_times[np.maximum(0, ids-1)]))
    age = distance.min(axis=1)
    near = (age <= .055) & connected
    predicted = (age <= .15) & connected & ~near
    usable = near | predicted
    statuses = np.full(q.shape, 'unresolved', dtype='<U12')
    statuses[near] = 'vision'; statuses[predicted] = 'predicted'
    statuses[0] = 'home'; usable[0] = True
    turns = np.ones(3, bool)
    unique_times, unique_ids, inverse = np.unique(times, return_index=True, return_inverse=True)
    rate = np.gradient(q[unique_ids], unique_times, axis=0)[inverse]
    frames = []
    for ei, (time, ci, fi) in enumerate(events):
        turns &= usable[ei]
        values = np.where(usable[ei], q[ei], np.nan)
        R = F @ rotation_x(values[0]) if np.isfinite(values[0]) else None
        shell_quaternions = []
        for h in range(2):
            shell_quaternions.append(Rotation.from_matrix(R @ rotation_z(values[1+h])).as_quat()
                                     if R is not None and np.isfinite(values[1+h]) else None)
        used_cameras = [[CAMERA_NAMES[c] for c in range(2) if distance[ei, c, k] <= .055]
                        for k in range(3)]
        frames.append({'time_s': time-origin, 'receive_time_s': float(data['times'][ci][fi]),
                       'camera_time_correction_s': offset if ci == 1 else 0.,
                       'camera': CAMERA_NAMES[ci], 'source_frame': fi,
                       'angles': values, 'relative_angles': values-np.array([q[0, 0], 0., 0.]),
                       'angular_velocity': np.where(usable[ei], rate[ei], np.nan),
                       'status': statuses[ei], 'std_rad': np.full(3, np.nan), 'age_s': age[ei],
                       'turn_count_valid': turns.copy(), 'support_cameras': used_cameras,
                       'native_inlier_counts': support_counts[ei],
                       'visual_components': np.flatnonzero(support[ei]).tolist(),
                       'caster_quaternion_xyzw': None if R is None else Rotation.from_matrix(R).as_quat(),
                       'shell_quaternions_xyzw': shell_quaternions,
                       'swivel_axis_c920': None if R is None else R[:, 2],
                       'pivot_c920_m': pivot, 'offline_refined': True})
        frames[-1]['shell_centers_c920_m'] = (None if R is None else
            [pivot+R[:, 2]*problem.geometry.get('red_shell_sign', 1)*(1-2*h)*problem.geometry['gap_m']/2
             for h in range(2)])
    cameras = {}
    for ci, name in enumerate(CAMERA_NAMES):
        selected = obs['camera'] == ci; good = inlier & selected
        cameras[name] = {'observations': int(selected.sum()), 'inliers': int(good.sum()),
                         'inlier_fraction': float(good.sum()/max(1, selected.sum())),
                         'inlier_rmse_px': float(np.sqrt(np.mean(errors[good]**2))) if good.any() else None,
                         'median_error_px': float(np.median(errors[selected])) if selected.any() else None,
                         'visual_updates': int(np.any(support[camera_indices == ci], axis=1).sum())}
    recovered = {}
    for ci, name in enumerate(CAMERA_NAMES):
        recovered[name] = {label: int(np.count_nonzero(near[:, k] & (distance[:, ci, k] > .055)
                                                     & (distance[:, 1-ci, k] <= .055)))
                           for k, label in enumerate(('roll', 'red', 'green'))}
    return frames, {'native_images': len(events), 'retained_observations': len(obs['uv']),
                    'additional_brio_offset_s': offset, 'output_clock_origin_shift_s': origin,
                    'accepted_observations': int(inlier.sum()), 'per_camera': cameras,
                    'coverage': {label: dict(Counter(statuses[:, k])) for k, label in enumerate(('roll', 'red', 'green'))},
                    'supported_by_other_camera_when_this_camera_has_no_nearby_support': recovered,
                    'accuracy_validated': False, 'accepted_axis_calibration_modified': False,
                    'turn_count_valid_at_end': turns,
                    'inlier_rmse_px': float(np.sqrt(np.mean(errors[inlier]**2))) if inlier.any() else None}, inlier


def run_joint(config_path, session, output, *, initial_roll_deg=8., cache=None,
              max_frames=None, max_nfev=220, auto_initial=True, render=False, progress=print,
              resume_from=None):
    out = _output_directory(output)
    report = {'kind': 'fused_motion', 'method': 'native_time_joint_metric_bundle',
              'success': False, 'status': 'failed'}
    try:
        if not np.isfinite(initial_roll_deg): raise ValueError('initial_roll_deg must be finite')
        cfg = load_config(config_path); cameras, infos, _ = load_rig(cfg); axes = load_axes(cfg)
        F, pivot = axes['R_bc'], axes['pivot_c920_m']; geometry = cfg['geometry']
        _record_provenance(report, cfg, infos)
        data = collect_native_tracks(cfg, session, output=cache, max_frames=max_frames, progress=progress)
        if abs(data['times'][0][0]-data['times'][1][0]) > .1:
            raise ValueError('Initial camera images are over 100ms apart; a common starting pose is not established')
        if resume_from is None:
            initial = (fit_initial_pose(cameras, data['initial_frames'], cfg, F, pivot, initial_roll_deg)
                       if auto_initial else {'accepted': False, 'initial_roll_deg': float(initial_roll_deg),
                                             'method': 'supplied_approximate_pose'})
            if progress: progress(f'Initial roll seed {initial["initial_roll_deg"]:.3f} deg; rim fit accepted={initial["accepted"]}', flush=True)
            seed = seed_trajectory(data, cameras, F, pivot, geometry, initial['initial_roll_deg'], progress,
                                   offline_hypotheses=True)
            seed, roll_seed_report = reseed_roll(data, seed, cameras, F, pivot, geometry, initial['initial_roll_deg'])
            problem, x, initial_stages = refine_native(data, seed, cameras, F, pivot, geometry, max_nfev=max_nfev, progress=progress)
            initial_frames, initial_summary, _ = summarize_solution(problem, x, data, seed, F, pivot)
            initial_knots, initial_angles = problem.knots, problem.unpack(x)[0]
            initial_offset = float(x[problem.timing_id])
        else:
            source_path = Path(resume_from).expanduser().resolve()
            previous = json.loads(source_path.read_text(encoding='utf-8'))
            if (previous.get('method') != report['method'] or not calibration_matches(cfg, previous.get('calibration_hashes'))
                    or previous.get('observation_cache_provenance') != data['provenance']
                    or not np.allclose(previous.get('F'), F) or not np.allclose(previous.get('pivot'), pivot)):
                raise ValueError('Saved refinement inputs or calibration differ from current native observations')
            with np.load(source_path.parent/'seed.npz', allow_pickle=False) as saved:
                seed = {k: saved[k].copy() for k in ('angles', 'points', 'landmark', 'keys')}
                seed['events'] = [(float(t), int(ci), int(fi)) for t, ci, fi in saved['events']]
                for key in ('initial_brio_offset_s', 'initial_roll_prior_rad'):
                    if key in saved: seed[key] = float(saved[key])
            diag = json.loads((source_path.parent/'seed_diagnostics.json').read_text(encoding='utf-8'))
            seed.update(per_camera=diag['per_camera'], quality=diag['quality'], components=[[] for _ in seed['events']])
            with np.load(source_path.parent/'bundle.npz', allow_pickle=False) as saved:
                initial_knots, initial_angles = saved['knots'].copy(), saved['angles'].copy()
                initial_offset = float(saved['additional_brio_offset_s'])
            initial = previous['initial_pose']; roll_seed_report = previous['roll_initialization']
            initial_summary, initial_frames = previous['summary'], previous['frames']
            initial_stages = initial_summary['optimization_stages']
            initial_roll_deg = previous['supplied_initial_roll_deg']
            report['refined_from'] = str(source_path)
        write_json(out/'initial_pose.json', initial)
        write_json(out/'roll_initialization.json', roll_seed_report)
        # With common roll established, ray/sphere geometry supplies direct
        # relative azimuth constraints for both independently spinning shells.
        # Reinitialize spins before the final unrestricted metric bundle solve.
        roll_coverage = np.mean([f['status'][0] in ('home', 'vision') for f in initial_frames])
        if roll_coverage >= .8:
            from .spin_initialization import reseed_spins
            if progress: progress('Recovering independent shell phase from metric surface azimuth tracks.', flush=True)
            seed, spin_report = reseed_spins(data, seed, cameras, F, pivot, geometry,
                                             initial_knots, initial_angles, initial_offset)
            seed['initial_roll_prior_rad'] = np.deg2rad(initial['initial_roll_deg'])
            write_json(out/'spin_initialization.json', spin_report)
            report['spin_initialization'] = spin_report
        elif resume_from is not None:
            raise ValueError('Saved trajectory lacks enough supported roll to condition shell-spin recovery')
        np.savez_compressed(out/'seed.npz', angles=seed['angles'], points=seed['points'],
                            landmark=seed['landmark'], keys=seed['keys'], events=np.asarray(seed['events']),
                            initial_brio_offset_s=seed.get('initial_brio_offset_s', 0.),
                            initial_roll_prior_rad=seed.get('initial_roll_prior_rad', np.deg2rad(initial['initial_roll_deg'])))
        write_json(out/'seed_diagnostics.json', {'per_camera': seed['per_camera'], 'quality': seed['quality']})
        if roll_coverage >= .8:
            problem, x, stages = refine_native(data, seed, cameras, F, pivot, geometry, max_nfev=max_nfev, progress=progress)
        else:
            stages = initial_stages
        frames, summary, inlier = summarize_solution(problem, x, data, seed, F, pivot)
        q, polar = problem.unpack(x)
        np.savez_compressed(out/'bundle.npz', knots=problem.knots, angles=q, surface_polar=polar,
                            additional_brio_offset_s=x[problem.timing_id], optimized_parameters=x,
                            landmark_keys=problem.keys, observations_camera=problem.obs['camera'],
                            observations_frame=problem.obs['frame'], observations_shell=problem.obs['shell'],
                            observations_uv=problem.obs['uv'], observation_landmark=problem.landmark,
                            observation_inlier=inlier, original_observation_rows=problem.original_rows)
        summary['optimization_stages'] = stages
        summary['initial_joint_fit'] = {k: initial_summary[k] for k in
                                       ('coverage', 'inlier_rmse_px', 'additional_brio_offset_s')}
        summary['initial_optimization_stages'] = initial_stages
        summary['optimizer_converged'] = bool(stages[-1]['converged'])
        summary['supported_event_fraction'] = {label: float(np.mean([f['status'][k] in ('home', 'vision') for f in frames]))
                                                for k, label in enumerate(('roll', 'red', 'green'))}
        summary['initial_roll_change_deg'] = float(np.rad2deg(q[0, 0])-initial['initial_roll_deg'])
        summary['native_tracking'] = data['report']
        report.update(status='completed', success=bool(stages[-1]['converged'] and
                      min(summary['supported_event_fraction'].values()) >= .5),
                      measurement_source='joint_metric', config=cfg, geometry=geometry,
                      calibration_hashes=calibration_hashes(cfg), axes_sha256=axes['sha256'],
                      F=F, pivot=pivot, videos=data['videos'], use_video_manifest=True,
                      timestamp_origin_s=data['timestamp_origin_s']+summary['output_clock_origin_shift_s'], initial_pose=initial,
                      roll_initialization=roll_seed_report,
                      initial_roll_deg=float(np.rad2deg(q[0, 0])), supplied_initial_roll_deg=float(initial_roll_deg),
                      summary=summary, frames=frames,
                      observation_cache_provenance=data['provenance'],
                      timing_policy='Every native image retained. A constant additional Brio time correction (bounded +/-80ms, 50ms prior) is jointly fit in pixel space. CSV times include this correction and share one origin. This is motion-based alignment conditional on geometry, not independently measured exposure synchronization.',
                      orientation_reference='R_c920_from_caster = F @ Rx(roll); red/green poses additionally use Rz(beta). Roll is relative to calibrated home; each beta is relative to its initial phase. relative_angles subtracts the fitted starting roll.',
                      uncertainty_scope='Pixel residuals are fit consistency, not angular accuracy. std_rad is null: a marginal covariance including landmark, calibration and camera-time uncertainty has not been calculated.',
                      support_policy='vision means accepted, spatially distributed pixels within 55ms in either view, connected to the initial pose by material tracks; predicted extends to 150ms; unresolved remains null. Camera-local support is exported separately.',
                      translation_policy='Assembly pivot is fixed in the mounted C920 frame; shell centers move with roll. Robot/world translation is not observable from these mounted views alone.')
        from .offline_finalization import finalize_phase_estimates, write_offline_csvs
        report = finalize_phase_estimates(report, data, seed, cameras, F, pivot, geometry,
                                          problem.knots, q, float(x[problem.timing_id]), out)
        frames = report['frames']
        write_json(out/'results.json', report)
        write_offline_csvs(out, frames)
        from .visualization import write_motion_viewer
        write_motion_viewer(out/'results.json', out/'orientation_3d.html')
        if render:
            from .offline_render import render_offline
            render_offline(report, cfg, out/'replay')
        return report
    except Exception as exc:
        report.update(success=False, status='failed', error=str(exc))
        write_json(out/'results.json', report)
        raise


def finalize_saved_joint(config_path, session, source, output, *, cache=None, render=False, progress=print):
    """Re-export an existing converged bundle with current phase diagnostics.

    This avoids repeating nonlinear optimization when only export/evidence logic
    changed. Exact input provenance and saved parameter interpretation are checked.
    """
    import shutil
    from .offline import NativeBundle
    from .offline_finalization import finalize_phase_estimates, write_offline_csvs
    from .visualization import write_motion_viewer
    source = Path(source).expanduser().resolve()
    report = json.loads(source.read_text(encoding='utf-8'))
    cfg = load_config(config_path); cameras, _, _ = load_rig(cfg); axes = load_axes(cfg)
    F, pivot = axes['R_bc'], axes['pivot_c920_m']
    data = collect_native_tracks(cfg, session, output=cache, progress=progress)
    if (report.get('method') != 'native_time_joint_metric_bundle'
            or not calibration_matches(cfg, report.get('calibration_hashes'))
            or report.get('observation_cache_provenance') != data['provenance']
            or not np.allclose(report.get('F'), F) or not np.allclose(report.get('pivot'), pivot)):
        raise ValueError('Saved refinement inputs or calibration differ from current native observations')
    with np.load(source.parent/'seed.npz', allow_pickle=False) as saved:
        seed = {k: saved[k].copy() for k in ('angles', 'points', 'landmark', 'keys')}
        seed['events'] = [(float(t), int(ci), int(fi)) for t, ci, fi in saved['events']]
        for key in ('initial_brio_offset_s', 'initial_roll_prior_rad'):
            if key in saved: seed[key] = float(saved[key])
    diag = json.loads((source.parent/'seed_diagnostics.json').read_text(encoding='utf-8'))
    seed.update(per_camera=diag['per_camera'], quality=diag['quality'])
    problem = NativeBundle(data, seed, cameras, F, pivot, cfg['geometry'])
    with np.load(source.parent/'bundle.npz', allow_pickle=False) as saved:
        x = saved['optimized_parameters'].copy()
        np.testing.assert_allclose(problem.knots, saved['knots'], atol=1e-12)
        np.testing.assert_allclose(problem.unpack(x)[0], saved['angles'], atol=1e-12)
    frames, summary, _ = summarize_solution(problem, x, data, seed, F, pivot)
    report['frames'] = frames
    report['summary'].update(summary)
    report['finalized_from'] = str(source)
    out = _output_directory(output)
    for filename in ('seed.npz', 'seed_diagnostics.json', 'bundle.npz', 'initial_pose.json',
                     'roll_initialization.json', 'spin_initialization.json'):
        if (source.parent/filename).is_file(): shutil.copy2(source.parent/filename, out/filename)
    report = finalize_phase_estimates(report, data, seed, cameras, F, pivot, cfg['geometry'],
                                      problem.knots, problem.unpack(x)[0], float(x[problem.timing_id]), out)
    write_json(out/'results.json', report)
    write_offline_csvs(out, report['frames'])
    write_motion_viewer(out/'results.json', out/'orientation_3d.html')
    if render:
        from .offline_render import render_offline
        render_offline(report, cfg, out/'replay')
    return report
