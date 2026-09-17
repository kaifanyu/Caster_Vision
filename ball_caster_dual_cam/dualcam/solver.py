"""Offline joint two-view bundle fit of caster axes, motion and surface tracks.

The landmark coordinates are nuisance variables constrained to the real signed
hemispheres. They are local to one camera/clip: cross-camera feature matching is
not required. Shared frame angles couple the two cameras in image space.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsmr
from scipy.spatial.transform import Rotation

from .model import (initialize_surface_point, points_to_polar, polar_to_points,
                    project, world_points)


def _components(frame, track, n, selected, min_track_frames):
    """Frames joined by accepted tracks and connected to known home frame 0."""
    parent = np.arange(n)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    seen = np.zeros(n, dtype=bool)
    for t in np.unique(track[selected]):
        fs = np.unique(frame[selected & (track == t)])
        if len(fs) < min_track_frames:
            continue
        seen[fs] = True
        first = root(int(fs[0]))
        for f in fs[1:]:
            parent[root(int(f))] = first
    return np.array([seen[f] and root(f) == root(0) for f in range(n)])


def _support(data, accepted, min_points, min_track_frames):
    obs, n = data['obs'], len(data['q0'])
    valid = np.zeros((n, 3), dtype=bool)
    counts = np.bincount(obs['frame'][accepted], minlength=n)
    connected = _components(obs['frame'], obs['landmark'], n,
                            accepted, min_track_frames)
    valid[:, 0] = connected & (counts >= min_points)
    for h in (0, 1):
        selected = accepted & (obs['shell'] == h)
        counts = np.bincount(obs['frame'][selected], minlength=n)
        connected = _components(obs['frame'], obs['landmark'], n,
                                selected, min_track_frames)
        valid[:, h + 1] = connected & (counts >= min_points)
    return valid


def _prepare(dataset, cameras, F, pivot, radius, gap, red_sign, min_track_frames):
    times = np.asarray(dataset['times'], dtype=float)
    if times.ndim != 1 or len(times) < min_track_frames or not np.all(np.isfinite(times)):
        raise ValueError('A clip needs finite one-dimensional times and at least three frames.')
    if np.any(np.diff(times) <= 0):
        raise ValueError('Synchronized frame times must be strictly increasing.')
    q = np.asarray(dataset.get('initial_angles', np.zeros((len(times), 3))), dtype=float).copy()
    if q.shape != (len(times), 3) or not np.all(np.isfinite(q)):
        raise ValueError('initial_angles must be finite (Nframe, 3) radians.')
    mode = dataset.get('mode', 'motion')
    if mode not in ('roll', 'swivel', 'motion'):
        raise ValueError('Clip mode must be roll, swivel, or motion.')
    if mode == 'roll':
        q[:, 1:] = q[0, 1:]
    if mode == 'swivel':
        q[:, 0] = float(dataset.get('initial_roll_rad', q[0, 0]))
    source = dataset['observations']
    obs = {}
    for key in ('camera', 'shell', 'frame', 'track'):
        if key not in source:
            raise ValueError('Missing observation index: ' + key)
        indices = np.asarray(source[key])
        if indices.dtype.kind not in 'iu':
            raise ValueError('Observation ' + key + ' must contain integer indices.')
        obs[key] = indices.astype(int)
    if np.any(obs['track'] < 0):
        raise ValueError('Observation track IDs must be nonnegative and camera/shell-local.')
    obs['uv'] = np.asarray(source['uv'], dtype=float)
    m = len(obs['frame'])
    obs['weight'] = np.asarray(source.get('weight', np.ones(m)), dtype=float)
    if any(obs[k].shape != (m,) for k in ('camera', 'shell', 'frame', 'track', 'weight')) or obs['uv'].shape != (m, 2):
        raise ValueError('Observation arrays must have consistent lengths, uv shape (N,2).')
    if (np.any((obs['camera'] < 0) | (obs['camera'] >= len(cameras)))
            or np.any((obs['shell'] < 0) | (obs['shell'] > 1))
            or np.any((obs['frame'] < 0) | (obs['frame'] >= len(times)))):
        raise ValueError('Observation camera/shell/frame index out of range.')
    if not np.all(np.isfinite(obs['uv'])) or not np.all(np.isfinite(obs['weight'])) or np.any(obs['weight'] <= 0):
        raise ValueError('Pixels and positive weights must be finite.')
    keys = np.column_stack([obs[k] for k in ('camera', 'shell', 'track')])
    if m == 0:
        raise ValueError('Clip has no observations.')
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    keep = np.zeros(m, dtype=bool)
    for t in np.unique(inverse):
        ids = np.flatnonzero(inverse == t)
        if len(np.unique(obs['frame'][ids])) != len(ids):
            raise ValueError('A camera/shell/track may have only one observation per frame.')
        if len(ids) >= min_track_frames:
            keep[ids] = True
    if not np.any(keep):
        raise ValueError('No surface tracks span enough frames (minimum three).')
    _, landmarks = np.unique(keys[keep], axis=0, return_inverse=True)
    obs = {k: v[keep] for k, v in obs.items()}
    obs['landmark'] = landmarks
    local, signs = [], []
    for l in range(landmarks.max() + 1):
        ids = np.flatnonzero(landmarks == l)
        i = ids[np.argmin(obs['frame'][ids])]
        h, c, f = obs['shell'][i], obs['camera'][i], obs['frame'][i]
        local.append(initialize_surface_point(cameras[c], obs['uv'][i], F, pivot,
                                              q[f], h, radius, gap, red_sign))
        signs.append(red_sign * (1 - 2 * h))
    return {'obs': obs, 'q0': q, 'times': times, 'mode': mode,
            'polar0': points_to_polar(local), 'signs': np.asarray(signs),
            'original_size': m, 'kept_indices': np.flatnonzero(keep)}


def fit_joint(datasets, cameras, F_init, pivot_init, radius_m, gap_m, *,
              calibrate_axes=False, refine_pivot=False, options=None):
    """Fit synchronized clips, using metric fixed extrinsics and ideal pixels.

    cameras: [{K:3x3,R:3x3,t:3}], camera 0 must be the identity reference.
    datasets: [{times:(T,), initial_angles:(T,3), mode, observations:{camera,
      shell (0=red,1=green), frame, track, uv (N,2), weight (optional)}}].
    Every clip's first angles are a known pose, not an arbitrary optimizer guess.
    Tracks need >=3 frames and a temporal path to frame 0 for valid output.
    Axis calibration additionally requires home-start roll AND swivel clips.
    All returned angles are radians. Unsupported estimates are NaN. Callers must
    check success before persisting F/pivot as a usable calibration.
    """
    options = {} if options is None else dict(options)
    defaults = {'min_track_frames': 3, 'min_points_per_frame': 3,
                'robust_px': 1.5, 'outlier_px': 5., 'max_nfev': 250,
                'max_inlier_rmse_px': 2.5, 'min_inlier_fraction': .6,
                'min_calibration_excursion_deg': 5., 'red_sign': 1,
                'axis_rank_rtol': 1e-5, 'min_surface_spread': .025}
    unknown = set(options) - set(defaults) - {'verbose', 'ftol', 'xtol', 'gtol'}
    if unknown:
        raise ValueError('Unknown solver options: ' + ', '.join(sorted(unknown)))
    cfg = defaults | options
    for key in ('min_track_frames', 'min_points_per_frame'):
        if not isinstance(cfg[key], (int, np.integer)) or cfg[key] < 3:
            raise ValueError(key + ' must be an integer of at least three.')
    for key in ('robust_px', 'outlier_px', 'max_inlier_rmse_px',
                'min_calibration_excursion_deg', 'axis_rank_rtol', 'min_surface_spread'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(key + ' must be positive and finite.')
    if not 0 < cfg['min_inlier_fraction'] <= 1:
        raise ValueError('min_inlier_fraction must lie in (0,1].')
    red_sign = cfg['red_sign']
    if red_sign not in (-1, 1) or radius_m <= 0 or gap_m < 0:
        raise ValueError('Need positive radius, nonnegative gap, red_sign +/-1.')
    F_init = np.asarray(F_init, dtype=float)
    pivot_init = np.asarray(pivot_init, dtype=float)
    if (F_init.shape != (3, 3) or not np.all(np.isfinite(F_init))
            or not np.allclose(F_init.T @ F_init, np.eye(3), atol=1e-5)
            or np.linalg.det(F_init) < .99999):
        raise ValueError('F_init must be a proper rotation with columns [roll,y,swivel].')
    if pivot_init.shape != (3,) or not np.all(np.isfinite(pivot_init)):
        raise ValueError('pivot_init must be a finite metric camera-1 position.')
    cameras = [{k: np.asarray(c[k], dtype=float) for k in ('K', 'R', 't')} for c in cameras]
    if len(cameras) != 2:
        raise ValueError('Exactly two calibrated cameras are required.')
    for c in cameras:
        if (c['K'].shape != (3, 3) or c['R'].shape != (3, 3) or c['t'].shape != (3,)
                or not all(np.all(np.isfinite(v)) for v in c.values())
                or not np.allclose(c['R'].T @ c['R'], np.eye(3), atol=1e-5)
                or np.linalg.det(c['R']) < .99999
                or c['K'][0, 0] <= 0 or c['K'][1, 1] <= 0):
            raise ValueError('Invalid calibrated camera matrix or rigid transform.')
    if not np.allclose(cameras[0]['R'], np.eye(3)) or not np.allclose(cameras[0]['t'], 0):
        raise ValueError('Camera 0 must define the reference frame (R=I,t=0).')
    if calibrate_axes:
        if not {'roll', 'swivel'} <= {d.get('mode') for d in datasets}:
            raise ValueError('Axis calibration requires both pure-roll and pure-swivel clips.')
        for d in datasets:
            initial = np.asarray(d.get('initial_angles', np.zeros((len(d['times']), 3))))
            if not np.allclose(initial[0], 0, atol=1e-8) or abs(d.get('initial_roll_rad', 0.)) > 1e-8:
                raise ValueError('Calibration clips must begin at the same explicitly known home (all angles zero).')
    prepared = [_prepare(d, cameras, F_init, pivot_init, radius_m, gap_m,
                         red_sign, cfg['min_track_frames']) for d in datasets]
    if not prepared:
        raise ValueError('At least one clip is required.')
    # Guessed poses in an unanchored track segment must not influence axes.
    for d in prepared:
        while True:
            obs = d['obs']
            valid = _support(d, np.ones(len(obs['frame']), bool),
                             cfg['min_points_per_frame'], cfg['min_track_frames'])
            keep = valid[obs['frame'], 0]
            if d['mode'] != 'roll':
                keep &= valid[obs['frame'], 1 + obs['shell']]
            for landmark in np.unique(obs['landmark']):
                selected = obs['landmark'] == landmark
                if np.count_nonzero(keep & selected) < cfg['min_track_frames']:
                    keep[selected] = False
            if keep.all():
                break
            if not keep.any():
                raise ValueError('No sufficiently supported tracks connect this clip to its known home frame.')
            d['obs'] = {k: v[keep] for k, v in obs.items()}
            d['kept_indices'] = d['kept_indices'][keep]
        used, inverse = np.unique(d['obs']['landmark'], return_inverse=True)
        d['obs']['landmark'] = inverse
        d['polar0'] = d['polar0'][used]
        d['signs'] = d['signs'][used]
    x0, lower, upper = [], [], []
    axes_slice, pivot_slice = None, None
    if calibrate_axes:
        axes_slice = slice(len(x0), len(x0) + 3)
        x0.extend([0.] * 3); lower.extend([-np.pi] * 3); upper.extend([np.pi] * 3)
    if refine_pivot:
        pivot_slice = slice(len(x0), len(x0) + 3)
        x0.extend(pivot_init); lower.extend([-np.inf] * 3); upper.extend([np.inf] * 3)
    for d in prepared:
        valid = _support(d, np.ones(len(d['obs']['frame']), bool),
                         cfg['min_points_per_frame'], cfg['min_track_frames'])
        active = valid.copy()
        active[0] = False  # fixed known home resolves all angle/landmark gauges
        if d['mode'] == 'roll':
            active[:, 1:] = False
        elif d['mode'] == 'swivel':
            active[:, 0] = False
        d['angle_ids'] = np.full(d['q0'].shape, -1, dtype=int)
        d['angle_ids'][active] = np.arange(len(x0), len(x0) + active.sum())
        d['active'] = active
        x0.extend(d['q0'][active]); lower.extend([-np.inf] * active.sum()); upper.extend([np.inf] * active.sum())
        d['point_slice'] = slice(len(x0), len(x0) + d['polar0'].size)
        polar = d['polar0'].copy()
        polar[:, 1] = np.clip(polar[:, 1], 1e-7, np.pi / 2 - 1e-7)
        x0.extend(polar.ravel())
        lower.extend(np.tile([-np.inf, 0.], len(polar)))
        upper.extend(np.tile([np.inf, np.pi / 2], len(polar)))
    x0 = np.asarray(x0)
    nres = sum(len(d['obs']['frame']) * 2 for d in prepared)
    sparsity = lil_matrix((nres, len(x0)), dtype=int)
    row = 0
    for d in prepared:
        obs = d['obs']
        for i, f in enumerate(obs['frame']):
            rows = slice(row + 2 * i, row + 2 * i + 2)
            if axes_slice is not None:
                sparsity[rows, axes_slice] = 1
            if pivot_slice is not None:
                sparsity[rows, pivot_slice] = 1
            for component in (0, int(obs['shell'][i]) + 1):
                index = d['angle_ids'][f, component]
                if index >= 0:
                    sparsity[rows, index] = 1
            start = d['point_slice'].start + 2 * obs['landmark'][i]
            sparsity[rows, start:start + 2] = 1
        row += len(obs['frame']) * 2

    def unpack(x, d):
        q = d['q0'].copy()
        q[d['active']] = x[d['angle_ids'][d['active']]]
        points = polar_to_points(x[d['point_slice']].reshape(-1, 2), d['signs'])
        return q, points

    def geometry(x):
        F = Rotation.from_rotvec(x[axes_slice]).as_matrix() @ F_init if calibrate_axes else F_init
        C = x[pivot_slice] if refine_pivot else pivot_init
        return F, C

    def pixel_errors(x, d, F, C):
        obs = d['obs']
        q, points = unpack(x, d)
        xyz = world_points(F, C, q[obs['frame']], obs['shell'],
                           points[obs['landmark']], radius_m, gap_m, red_sign)
        prediction = np.empty_like(obs['uv'])
        depth = np.empty(len(xyz))
        for camera_index, camera in enumerate(cameras):
            selected = obs['camera'] == camera_index
            prediction[selected] = project(camera, xyz[selected])
            depth[selected] = (xyz[selected] @ camera['R'].T + camera['t'])[:, 2]
        return prediction - obs['uv'], depth

    def residual(x):
        F, C = geometry(x)
        return np.concatenate([(pixel_errors(x, d, F, C)[0]
                                * np.sqrt(d['obs']['weight'])[:, None]).ravel()
                               for d in prepared])

    result = least_squares(residual, x0, bounds=(lower, upper),
                           jac_sparsity=sparsity.tocsr(), x_scale='jac',
                           loss='soft_l1', f_scale=cfg['robust_px'],
                           max_nfev=cfg['max_nfev'],
                           ftol=cfg.get('ftol', 1e-7), xtol=cfg.get('xtol', 1e-7),
                           gtol=cfg.get('gtol', 1e-7), verbose=cfg.get('verbose', 0))
    F, C = geometry(result.x)
    reasons = []
    if not result.success:
        reasons.append('Optimizer did not converge: ' + result.message)
    outputs, camera_stats = [], {}
    for d in prepared:
        errors, depth = pixel_errors(result.x, d, F, C)
        norms = np.linalg.norm(errors, axis=1)
        accepted = (norms <= cfg['outlier_px']) & (depth > 0)
        q, points = unpack(result.x, d)
        obs = d['obs']
        xyz = world_points(F, C, q[obs['frame']], obs['shell'],
                           points[obs['landmark']], radius_m, gap_m, red_sign)
        centers = world_points(F, C, q[obs['frame']], obs['shell'],
                               np.zeros_like(xyz), radius_m, gap_m, red_sign)
        for ci, camera in enumerate(cameras):
            selected = obs['camera'] == ci
            camera_center = -camera['R'].T @ camera['t']
            facing = np.sum((xyz[selected] - centers[selected]) *
                            (camera_center - xyz[selected]), axis=1) >= 0
            accepted[selected] &= facing
        # Count alone cannot distinguish a useful distributed pattern from
        # duplicate corners on one tiny patch. Require two-dimensional spread
        # of fitted metric surface points, normalized by the measured radius.
        # Pool both views in camera 1: complementary surface coverage counts.
        normalized_xyz = (xyz - C) / radius_m
        # A poorly observed frame must not be the sole bridge carrying the
        # absolute phase into a later disconnected set of tracks. Reassess
        # spatial coverage after each pruning pass as well as connectivity.
        while True:
            spatial_support = np.zeros_like(q, dtype=bool)
            for frame in np.unique(obs['frame'][accepted]):
                for component in range(3):
                    selected = accepted & (obs['frame'] == frame)
                    if component:
                        selected &= obs['shell'] == component - 1
                    locations = normalized_xyz[selected]
                    if len(locations) < cfg['min_points_per_frame']:
                        continue
                    centered = locations - locations.mean(axis=0)
                    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(locations))
                    spatial_support[frame, component] = (
                        eigenvalues[-2] >= cfg['min_surface_spread'] ** 2)
            valid = _support(d, accepted, cfg['min_points_per_frame'], cfg['min_track_frames'])
            valid &= spatial_support
            supported_observation = valid[obs['frame'], 0]
            if d['mode'] != 'roll':
                supported_observation &= valid[obs['frame'], 1 + obs['shell']]
            updated = accepted & supported_observation
            if np.array_equal(updated, accepted):
                break
            accepted = updated
        d['accepted'] = accepted
        original_inlier = np.zeros(d['original_size'], dtype=bool)
        original_inlier[d['kept_indices']] = accepted
        stats = {}
        for ci in range(len(cameras)):
            selected = d['obs']['camera'] == ci
            included = selected & accepted
            s = {'observations': int(selected.sum()), 'inliers': int(included.sum()),
                 'inlier_rmse_px': float(np.sqrt(np.mean(norms[included] ** 2))) if included.any() else None,
                 'all_rmse_px': float(np.sqrt(np.mean(norms[selected] ** 2))) if selected.any() else None}
            stats[str(ci)] = s
            aggregate = camera_stats.setdefault(str(ci), {'observations': 0, 'inliers': 0, 'squared_error': 0.})
            aggregate['observations'] += s['observations']
            aggregate['inliers'] += s['inliers']
            aggregate['squared_error'] += float((norms[included] ** 2).sum())
        outputs.append({'times': d['times'], 'angles': q, 'valid': valid,
                        'mode': d['mode'], 'observation_inlier': original_inlier,
                        'per_camera': stats})
    for ci, s in camera_stats.items():
        s['inlier_fraction'] = s['inliers'] / max(s['observations'], 1)
        s['inlier_rmse_px'] = np.sqrt(s.pop('squared_error') / s['inliers']) if s['inliers'] else None
        if calibrate_axes and s['observations'] < 12:
            reasons.append(f'Camera {ci} has insufficient calibration observations.')
        if s['observations'] and (s['inlier_fraction'] < cfg['min_inlier_fraction']
                or s['inlier_rmse_px'] is None or s['inlier_rmse_px'] > cfg['max_inlier_rmse_px']):
            reasons.append(f'Camera {ci} reprojection/inlier quality is insufficient.')
    axis_singular_values = None
    if calibrate_axes:
        excursions = {'roll': [], 'swivel': []}
        for out in outputs:
            components = (0,) if out['mode'] == 'roll' else (1, 2) if out['mode'] == 'swivel' else ()
            for component in components:
                values = out['angles'][out['valid'][:, component], component]
                if len(values) >= 3:
                    excursions[out['mode']].append(float(np.ptp(values)))
        for mode, values in excursions.items():
            if not values or max(values) < np.deg2rad(cfg['min_calibration_excursion_deg']):
                reasons.append(f'Insufficient {mode} calibration excursion connected to home.')
        # Eliminate landmark/angle nuisance directions before assessing 3-axis rank.
        accepted_rows = np.concatenate([np.repeat(d['accepted'], 2) for d in prepared])
        J = result.jac.tocsr()[accepted_rows]
        if J.shape[0] < 3:
            axis_singular_values = np.zeros(3)
        else:
            nuisance = J[:, 3:]
            reduced = []
            for k in range(3):
                column = J[:, k].toarray().ravel()
                solution = lsmr(nuisance, column, atol=1e-8, btol=1e-8,
                                maxiter=max(500, nuisance.shape[1] * 2))[0]
                reduced.append(column - nuisance @ solution)
            axis_singular_values = np.linalg.svd(np.column_stack(reduced), compute_uv=False)
        if (axis_singular_values[0] < 1e-3
                or axis_singular_values[-1] < cfg['axis_rank_rtol'] * axis_singular_values[0]):
            reasons.append('Axis calibration is rank deficient after accounting for unknown landmarks and motion.')
    def has_observed_motion(out):
        components = [0] if out['mode'] == 'roll' else [1, 2] if out['mode'] == 'swivel' else [0, 1, 2]
        return out['valid'][1:, components].any()

    if not any(has_observed_motion(o) for o in outputs):
        reasons.append('No motion samples retain distributed surface tracks connected to the known home frame.')
    success = not reasons
    for out in outputs:
        if not success:
            out['valid'][:] = False
        out['angles'] = np.where(out['valid'], out['angles'], np.nan)
    return {'success': success, 'F': F, 'pivot': C, 'datasets': outputs,
            'diagnostics': {'reasons': reasons, 'optimizer_message': result.message,
                            'nfev': result.nfev, 'cost': float(result.cost),
                            'per_camera': camera_stats,
                            'axis_singular_values': axis_singular_values}}
