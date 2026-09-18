"""Track-level held-out image consistency, never a ground-truth accuracy score.

Pose fitting must exclude whole held-out reference landmarks. This module only
fits two nuisance coordinates per held-out material point, from its first up to
three usable observations; it never adjusts trajectory or camera parameters.
Use the SAME reference observations and held-out keys when comparing branches.
CoTracker and KLT integer track IDs do not establish shared material identities.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


CAMERAS = ('c920', 'brio101')
SHELLS = ('red', 'green')
COMPONENTS = ('roll', 'red', 'green')
STRICT = frozenset(('home', 'vision'))


def _observations(data):
    obs = data['observations']
    out = {key: np.asarray(obs[key]) for key in ('camera', 'shell', 'frame', 'track', 'uv')}
    n = len(out['camera'])
    for key in ('camera', 'shell', 'frame', 'track'):
        a = out[key]
        if a.shape != (n,) or not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
            raise ValueError(f'observations.{key} must be an integer vector')
        out[key] = a.astype(np.int64)
    if (out['uv'].shape != (n, 2) or not np.isfinite(out['uv']).all()
            or np.any(~np.isin(out['camera'], (0, 1))) or np.any(~np.isin(out['shell'], (0, 1)))
            or np.any(out['frame'] < 0) or np.any(out['track'] < 0)):
        raise ValueError('Need finite pixel pairs and valid camera/shell/frame/track IDs')
    out['uv'] = out['uv'].astype(float)
    out['weight'] = np.asarray(obs.get('weight', np.ones(n)), float)
    if out['weight'].shape != (n,) or not np.isfinite(out['weight']).all() or np.any(out['weight'] < 0):
        raise ValueError('Observation weights must be finite and nonnegative')
    return out


def _keys(obs):
    return np.column_stack([obs[key] for key in ('camera', 'shell', 'track')])


def landmark_split(data, holdout_fraction=.2, split_seed='caster-v1', heldout_keys=None):
    """Deterministic camera/shell-stratified whole-landmark split.

    Within each stratum, select ceil(fraction * tracks), retaining at least one
    training track. A one-track stratum cannot provide both train and holdout.
    Returned masks index ORIGINAL observation rows, including any duplicates.
    Explicit keys reuse a split across branches on a common reference dataset.
    """
    if not np.isfinite(holdout_fraction) or not 0 < holdout_fraction < 1:
        raise ValueError('holdout_fraction must be between zero and one')
    obs = _observations(data)
    unique = [tuple(map(int, key)) for key in np.unique(_keys(obs), axis=0)]
    all_keys = set(unique)
    if heldout_keys is None:
        selected = set()
        for ci in range(2):
            for shell in range(2):
                group = [key for key in unique if key[:2] == (ci, shell)]
                group.sort(key=lambda key: sha256(f'{split_seed}:{key[0]}:{key[1]}:{key[2]}'.encode()).digest())
                count = min(max(0, len(group) - 1), int(np.ceil(len(group) * holdout_fraction)))
                selected.update(group[:count])
    else:
        selected = set()
        for key in heldout_keys:
            if len(key) != 3 or any(isinstance(v, bool) or int(v) != v for v in key):
                raise ValueError('Held-out keys must be integer (camera, shell, track) triples')
            selected.add(tuple(map(int, key)))
        if selected - all_keys:
            raise ValueError('Held-out keys are absent from this reference observation dataset')
    holdout = np.array([tuple(key) in selected for key in _keys(obs)], bool)
    strata = {}
    for ci, name in enumerate(CAMERAS):
        for shell, label in enumerate(SHELLS):
            group = [key for key in unique if key[:2] == (ci, shell)]
            count = sum(key in selected for key in group)
            strata[f'{name}/{label}'] = {'tracks': len(group), 'heldout_tracks': count,
                                         'training_tracks': len(group) - count}
    return {'train_mask': ~holdout, 'holdout_mask': holdout,
            'heldout_keys': [list(key) for key in sorted(selected)],
            'training_keys': [list(key) for key in sorted(all_keys - selected)],
            'split_seed': str(split_seed), 'requested_holdout_fraction': float(holdout_fraction),
            'unit': 'whole camera-local material track', 'strata': strata}


def subset_data(data, mask):
    """Return independent observation arrays while preserving dataset metadata."""
    mask = np.asarray(mask, bool)
    count = len(data['observations']['camera'])
    if mask.shape != (count,):
        raise ValueError('Observation mask has the wrong shape')
    return {**data, 'observations': {key: np.asarray(value)[mask].copy()
                                    for key, value in data['observations'].items()}}


def _rx(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def _rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _sample(knots, angles, times):
    result = np.column_stack([np.interp(times, knots, angles[:, k]) for k in range(3)])
    result[(times < knots[0]) | (times > knots[-1])] = np.nan
    return result


def _world(point, q, shell, F, pivot, geometry):
    sign = geometry.get('red_shell_sign', 1) * (1 - 2 * shell)
    beta = q[:, shell + 1]
    cb, sb = np.cos(beta), np.sin(beta)
    radius = geometry['radius_m']
    x = radius * (cb * point[0] - sb * point[1])
    y = radius * (sb * point[0] + cb * point[1])
    z = radius * point[2] + sign * geometry['gap_m'] / 2
    ca, sa = np.cos(q[:, 0]), np.sin(q[:, 0])
    local = np.column_stack((x, ca * y - sa * z, sa * y + ca * z))
    normal = np.column_stack((x, ca * y - sa * radius * point[2],
                              sa * y + ca * radius * point[2])) / radius
    return local @ F.T + pivot, normal @ F.T


def _project(camera, xyz):
    image_xyz = xyz @ camera['R'].T + camera['t']
    homogeneous = image_xyz @ camera['K'].T
    return homogeneous[:, :2] / np.maximum(homogeneous[:, 2:3], 1e-9), image_xyz[:, 2]


def _polar(point, sign):
    return np.array([np.arctan2(point[1], point[0]), np.arccos(np.clip(sign * point[2], 0., 1.))])


def _point(polar, sign):
    azimuth, theta = polar
    return np.array([np.cos(azimuth) * np.sin(theta), np.sin(azimuth) * np.sin(theta),
                     sign * np.cos(theta)])


def _ray_seed(camera, uv, q, shell, F, pivot, geometry, min_incidence):
    if not np.isfinite(q).all():
        return None
    eye = -camera['R'].T @ camera['t']
    direction = camera['R'].T @ np.linalg.solve(camera['K'], np.r_[uv, 1.])
    direction /= np.linalg.norm(direction)
    sign = geometry.get('red_shell_sign', 1) * (1 - 2 * shell)
    B = F @ _rx(q[0])
    S = B @ _rz(q[shell + 1])
    center = pivot + B @ np.array([0., 0., sign * geometry['gap_m'] / 2])
    delta = eye - center
    b = delta @ direction
    disc = b * b - delta @ delta + geometry['radius_m'] ** 2
    if disc < 0:
        return None
    distance = -b - np.sqrt(disc)
    if distance <= 0:
        return None
    normal = (eye + distance * direction - center) / geometry['radius_m']
    point = S.T @ normal
    if sign * point[2] < 0 or -normal @ direction < min_incidence:
        return None
    return point / np.linalg.norm(point)


def _visibility(camera, xyz, normals, q, shell, F, pivot, geometry, min_incidence):
    """Surface incidence and other-half occlusion; the external yoke is unknown."""
    eye = -camera['R'].T @ camera['t']
    ray = xyz - eye
    distances = np.linalg.norm(ray, axis=1)
    ray /= np.maximum(distances[:, None], 1e-12)
    depth = (xyz @ camera['R'].T + camera['t'])[:, 2]
    incidence = -np.einsum('ij,ij->i', normals, ray)
    reasons = np.full(len(xyz), 'scored', dtype='<U32')
    reasons[incidence < min_incidence] = 'backfacing_or_grazing'
    reasons[depth <= 0] = 'behind_camera'
    sign = -geometry.get('red_shell_sign', 1) * (1 - 2 * shell)
    radius, gap = geometry['radius_m'], geometry['gap_m']
    ca, sa = np.cos(q[:, 0]), np.sin(q[:, 0])
    centers = np.column_stack((np.zeros(len(q)), -sa * sign * gap / 2,
                               ca * sign * gap / 2)) @ F.T + pivot
    delta = eye - centers
    b = np.einsum('ij,ij->i', ray, delta)
    disc = b * b - np.einsum('ij,ij->i', delta, delta) + radius * radius
    roots = np.sqrt(np.maximum(disc, 0.))
    for distance in (-b - roots, -b + roots):
        hit_home = (eye + distance[:, None] * ray - centers) @ F
        local_z = -sa * hit_home[:, 1] + ca * hit_home[:, 2]
        hidden = ((disc >= 0) & (distance > 0) & (distance < distances - radius * .002)
                  & (sign * local_z >= -radius * .0001))
        reasons[hidden & (reasons == 'scored')] = 'occluded_by_other_shell'
    reasons[~np.isfinite(q).all(axis=1)] = 'outside_or_unknown_trajectory'
    return reasons


def _stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return {'count': int(len(values)), 'median_px': float(np.median(values)) if len(values) else None,
            'p95_px': float(np.percentile(values, 95)) if len(values) else None,
            'rmse_px': float(np.sqrt(np.mean(values ** 2))) if len(values) else None,
            'max_px': float(values.max()) if len(values) else None}


def _aggregate(samples):
    errors = [item['error_px'] for item in samples if item['status'] == 'scored']
    counts = Counter(item['status'] for item in samples)
    return {**_stats(errors), 'candidate_observations': len(samples), 'status_counts': dict(counts),
            'scored_fraction': len(errors) / len(samples) if samples else None,
            'fraction_within_3px_including_failures': sum(value <= 3. for value in errors) / len(samples)
                                                       if samples else None}


def _windows(windows):
    if windows is None:
        return []
    result = []
    for index, item in enumerate(windows):
        if isinstance(item, dict):
            name, start, end = item.get('name', f'window_{index}'), item['start_s'], item['end_s']
        else:
            start, end = item
            name = f'{start:g}-{end:g}s'
        if not np.isfinite([start, end]).all() or end < start:
            raise ValueError('Windows require finite start_s <= end_s')
        result.append((str(name), float(start), float(end)))
    return result


def score_heldout(data, cameras, F, pivot, geo, knots, angles, offset=0., *,
                  heldout_keys=None, holdout_fraction=.2, split_seed='caster-v1',
                  initialization_observations=3, min_incidence=.05,
                  max_initial_rmse_px=5., trajectory_excludes_heldout=False,
                  reference_name='common_reference_tracks', windows=None,
                  include_observations=False):
    """Score later pixels of reference tracks absent from pose fitting.

    ``knots, angles`` are unwrapped physical-time samples on the ORIGINAL data
    clock. Brio samples are evaluated at data.times[1][frame] + offset exactly
    once. The function does not re-zero this clock or extrapolate trajectories.
    Initialization uses only the first 1..3 usable ray/sphere observations per
    held-out track. All subsequent visible pixel errors, including outliers, are
    retained. Invisible/failed cases remain in the candidate denominator.

    The exclusion flag is a caller assertion, not independently verifiable.
    A previously fitted full-data baseline must leave it False. Such scores are
    descriptive fit consistency, not a leakage-free validation result.
    """
    if isinstance(initialization_observations, bool) or initialization_observations not in (1, 2, 3):
        raise ValueError('initialization_observations must be 1, 2, or 3')
    if not np.isfinite([offset, min_incidence, max_initial_rmse_px]).all() or not 0 <= min_incidence < 1 or max_initial_rmse_px <= 0:
        raise ValueError('Invalid timing, incidence, or initialization error threshold')
    obs = _observations(data)
    F, pivot = np.asarray(F, float), np.asarray(pivot, float)
    knots, angles = np.asarray(knots, float), np.asarray(angles, float)
    if (F.shape != (3, 3) or pivot.shape != (3,) or not np.isfinite(F).all()
            or not np.isfinite(pivot).all() or knots.ndim != 1 or len(knots) < 2
            or not np.isfinite(knots).all() or np.any(np.diff(knots) <= 0)
            or angles.shape != (len(knots), 3)):
        raise ValueError('Invalid geometry or trajectory arrays')
    if geo['radius_m'] <= 0 or not 0 <= geo['gap_m'] < 2 * geo['radius_m']:
        raise ValueError('Need positive radius and a valid physical gap')
    cameras = [{key: np.asarray(camera[key], float) for key in ('R', 't', 'K')} for camera in cameras]
    if len(cameras) != 2:
        raise ValueError('Need both calibrated cameras, including for mono-fit evaluation')
    times = [np.asarray(t, float) for t in data['times']]
    for ci in range(2):
        if (times[ci].ndim != 1 or not np.isfinite(times[ci]).all()
                or np.any(np.diff(times[ci]) <= 0)
                or np.any(obs['frame'][obs['camera'] == ci] >= len(times[ci]))):
            raise ValueError('Invalid native camera timestamps or source frame indices')
    split = landmark_split(data, holdout_fraction, split_seed, heldout_keys)
    groups = {}
    duplicate_count = 0
    unique_rows = {}
    for row in np.flatnonzero(split['holdout_mask']):
        key = tuple(int(obs[k][row]) for k in ('camera', 'shell', 'track'))
        identity = (*key, int(obs['frame'][row]))
        if identity in unique_rows:
            other = unique_rows[identity]
            if not np.allclose(obs['uv'][row], obs['uv'][other], atol=1e-6, rtol=0):
                raise ValueError('Conflicting pixels share a camera/shell/track/frame identity')
            duplicate_count += 1
            continue
        unique_rows[identity] = row
        groups.setdefault(key, []).append(row)
    samples, tracks = [], []
    for key in sorted(groups):
        ci, shell, track = key
        rows = np.array(sorted(groups[key], key=lambda row: obs['frame'][row]), int)
        corrected = times[ci][obs['frame'][rows]] + offset * (ci == 1)
        q = _sample(knots, angles, corrected)
        initial, candidates = [], []
        for local, row in enumerate(rows):
            if obs['weight'][row] <= 0:
                continue
            seed = _ray_seed(cameras[ci], obs['uv'][row], q[local], shell, F, pivot, geo, min_incidence)
            if seed is not None:
                initial.append(local)
                candidates.append(seed)
                if len(initial) >= initialization_observations:
                    break
        details = {'key': list(key), 'source_observations': len(rows),
                   'initialization_frames': [int(obs['frame'][rows[i]]) for i in initial],
                   'initialization_observations': len(initial)}
        point, reason = None, 'no_valid_initial_observations'
        if initial:
            sign = geo.get('red_shell_sign', 1) * (1 - 2 * shell)
            average = np.mean(candidates, axis=0)
            average /= np.linalg.norm(average)
            polar0 = _polar(average, sign)
            polar0[1] = np.clip(polar0[1], 1e-6, np.pi / 2 - 1e-6)
            initial_q, uv = q[initial], obs['uv'][rows[initial]]
            weight = np.sqrt(obs['weight'][rows[initial]])

            def residual(polar):
                xyz, _ = _world(_point(polar, sign), initial_q, shell, F, pivot, geo)
                return ((_project(cameras[ci], xyz)[0] - uv) * weight[:, None]).ravel()

            try:
                fitted = least_squares(residual, polar0, bounds=([-np.inf, 1e-7], [np.inf, np.pi / 2 - 1e-7]),
                                       loss='soft_l1', f_scale=2., max_nfev=80)
                point = _point(fitted.x, sign)
                xyz, normals = _world(point, initial_q, shell, F, pivot, geo)
                initial_error = np.linalg.norm(_project(cameras[ci], xyz)[0] - uv, axis=1)
                details['initialization_rmse_px'] = float(np.sqrt(np.mean(initial_error ** 2)))
                if not fitted.success:
                    reason = 'landmark_optimizer_failed'
                elif details['initialization_rmse_px'] > max_initial_rmse_px:
                    reason = 'initialization_residual_too_large'
                elif np.any(_visibility(cameras[ci], xyz, normals, initial_q, shell, F, pivot, geo, min_incidence) != 'scored'):
                    reason = 'initialization_not_visible'
                else:
                    reason = 'initialized'
                if reason != 'initialized':
                    point = None
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                point, reason = None, 'landmark_optimizer_failed'
                details['error'] = str(exc)
        details['status'] = reason
        track_samples = []
        evaluate = [i for i in range(len(rows)) if i not in initial]
        if point is not None:
            xyz, normals = _world(point, q, shell, F, pivot, geo)
            uv, _ = _project(cameras[ci], xyz)
            errors = np.linalg.norm(uv - obs['uv'][rows], axis=1)
            visibility = _visibility(cameras[ci], xyz, normals, q, shell, F, pivot, geo, min_incidence)
        for local in evaluate:
            row = rows[local]
            status = 'landmark_init_failed' if point is None else str(visibility[local])
            if point is not None and local <= max(initial):
                status = 'before_initialization'
            if obs['weight'][row] <= 0:
                status = 'zero_weight_observation'
            item = {'camera': CAMERAS[ci], 'shell': SHELLS[shell], 'track': track,
                    'source_frame': int(obs['frame'][row]), 'time_s': float(corrected[local]),
                    'status': status, 'error_px': float(errors[local]) if status == 'scored' else None}
            track_samples.append(item)
        details.update(_aggregate(track_samples))
        tracks.append(details)
        samples.extend(track_samples)
    result = {'metric': 'heldout_later_pixel_reprojection_consistency', 'accuracy_validated': False,
              'reference_name': str(reference_name),
              'trajectory_excludes_heldout_asserted': bool(trajectory_excludes_heldout),
              'validation_scope': ('whole-reference-track holdout, conditional on fixed geometry/trajectory'
                                   if trajectory_excludes_heldout else
                                   'descriptive consistency only; pose-fit exclusion is unverified or false'),
              'warnings': ['Trackers share source images; correlated pixel errors are possible.',
                           'Per-track point initialization uses early reference pixels; later pixels only are scored.',
                           'Model-invisible and failed tracks reduce scored coverage; do not compare median error alone.'],
              'clock': 'original data-relative camera time plus additional Brio offset; no re-zeroing',
              'additional_brio_offset_s': float(offset),
              'heldout_keys': split['heldout_keys'], 'split_seed': str(split_seed),
              'split_strata': split['strata'], 'heldout_tracks': len(tracks),
              'duplicate_observations_removed': duplicate_count,
              'initialization_observations': sum(t['initialization_observations'] for t in tracks),
              'track_status_counts': dict(Counter(t['status'] for t in tracks)),
              'overall': _aggregate(samples),
              'per_camera': {name: _aggregate([s for s in samples if s['camera'] == name]) for name in CAMERAS},
              'per_shell': {name: _aggregate([s for s in samples if s['shell'] == name]) for name in SHELLS},
              'per_camera_shell': {f'{camera}/{shell}': _aggregate([s for s in samples if s['camera'] == camera and s['shell'] == shell])
                                   for camera in CAMERAS for shell in SHELLS},
              'tracks': tracks,
              'windows': {name: {'start_s': start, 'end_s': end,
                                  **_aggregate([s for s in samples if start <= s['time_s'] <= end])}
                           for name, start, end in _windows(windows)}}
    if include_observations:
        result['observations'] = samples
    return result


def _read_report(value):
    return json.loads(Path(value).read_text(encoding='utf-8')) if isinstance(value, (str, Path)) else value


def _angle_metrics(frames, offset, start, end, max_gap_s):
    selected = [frame for frame in frames if start <= frame['time_s'] + offset <= end]
    result = {}
    unique = {frame['time_s'] + offset: frame for frame in selected}
    times = np.array(sorted(unique))
    for k, name in enumerate(COMPONENTS):
        counts = Counter(str(frame['status'][k]) for frame in selected)
        strict = [str(frame['status'][k]) in STRICT and frame['angles'][k] is not None
                  and np.isfinite(frame['angles'][k]) for frame in selected]
        rates = []
        for i in range(1, len(times)):
            a, b = unique[times[i - 1]], unique[times[i]]
            dt = times[i] - times[i - 1]
            if (1e-8 < dt <= max_gap_s and a['status'][k] in STRICT and b['status'][k] in STRICT
                    and a['angles'][k] is not None and b['angles'][k] is not None):
                rate = np.rad2deg((b['angles'][k] - a['angles'][k]) / dt)
                if np.isfinite(rate):
                    rates.append(float(rate))
        result[name] = {'events': len(selected), 'status_counts': dict(counts),
                        'strict_supported_events': int(sum(strict)),
                        'strict_supported_fraction': sum(strict) / len(selected) if selected else None,
                        'strict_rate_intervals': len(rates),
                        'strict_rate_min_deg_s': min(rates) if rates else None,
                        'strict_rate_max_deg_s': max(rates) if rates else None,
                        'strict_rate_p95_abs_deg_s': float(np.percentile(np.abs(rates), 95)) if rates else None}
    return result


def _aligned_samples(frames, offset, grid, max_gap_s):
    unique = {float(frame['time_s']) + offset: frame for frame in frames}
    times = np.array(sorted(unique))
    q = np.array([unique[time]['angles'] for time in times], float)
    statuses = np.array([unique[time]['status'] for time in times])
    q[statuses == 'unresolved'] = np.nan
    values = np.full((len(grid), 3), np.nan)
    strict = np.zeros((len(grid), 3), bool)
    for n, time in enumerate(grid):
        left = np.searchsorted(times, time, side='right') - 1
        if left >= 0 and np.isclose(times[left], time, atol=1e-9, rtol=0):
            values[n] = q[left]
            strict[n] = np.isin(statuses[left], tuple(STRICT)) & np.isfinite(q[left])
        elif 0 <= left < len(times) - 1 and times[left + 1] - times[left] <= max_gap_s:
            weight = (time - times[left]) / (times[left + 1] - times[left])
            values[n] = (1 - weight) * q[left] + weight * q[left + 1]
            strict[n] = np.isin(statuses[left], tuple(STRICT)) & np.isin(statuses[left + 1], tuple(STRICT)) & np.isfinite(values[n])
    return values, strict


def compare_reports(branches, windows=None, *, max_gap_s=.15):
    """Compare support and wrapped disagreement, without calling either truth.

    ``branches`` maps names to fused-motion report dictionaries or JSON paths.
    When all reports retain timestamp_origin_s, align their physical clocks
    before sampling. Otherwise explicitly compare their relative output clocks.
    Native event counts are coverage diagnostics, not independent sample counts.
    """
    if not branches or not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError('Need branches and a positive finite max_gap_s')
    reports = {str(name): _read_report(value) for name, value in branches.items()}
    if any(not report.get('frames') for report in reports.values()):
        raise ValueError('Each branch needs native trajectory frames')
    origins = [report.get('timestamp_origin_s') for report in reports.values()]
    absolute = all(origin is not None and np.isfinite(origin) for origin in origins)
    common_origin = float(min(origins)) if absolute else None
    offsets = {name: float(report['timestamp_origin_s'] - common_origin) if absolute else 0.
               for name, report in reports.items()}
    all_times = np.array(sorted(set(float(frame['time_s']) + offsets[name]
                                    for name, report in reports.items() for frame in report['frames'])))
    requested_windows = [('full', float(all_times[0]), float(all_times[-1])), *_windows(windows)]
    branch_metrics = {name: {label: {'start_s': start, 'end_s': end,
                                     'components': _angle_metrics(report['frames'], offsets[name], start, end, max_gap_s)}
                             for label, start, end in requested_windows}
                      for name, report in reports.items()}
    aligned = {name: _aligned_samples(report['frames'], offsets[name], all_times, max_gap_s)
               for name, report in reports.items()}
    pairs = {}
    names = list(reports)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            first, second = names[a], names[b]
            qa, sa = aligned[first]
            qb, sb = aligned[second]
            disagreement = np.abs(np.rad2deg(np.arctan2(np.sin(qa - qb), np.cos(qa - qb))))
            pair_windows = {}
            for label, start, end in requested_windows:
                within = (all_times >= start) & (all_times <= end)
                components = {}
                for k, name in enumerate(COMPONENTS):
                    scopes = {}
                    for scope, valid in (('strict_only', sa[:, k] & sb[:, k]),
                                          ('all_finite_estimates', np.isfinite(qa[:, k]) & np.isfinite(qb[:, k]))):
                        values = disagreement[within & valid, k]
                        scopes[scope] = {'samples': len(values),
                                         'median_abs_wrapped_deg': float(np.median(values)) if len(values) else None,
                                         'p95_abs_wrapped_deg': float(np.percentile(values, 95)) if len(values) else None,
                                         'max_abs_wrapped_deg': float(values.max()) if len(values) else None}
                    components[name] = scopes
                pair_windows[label] = {'start_s': start, 'end_s': end, 'components': components}
            pairs[f'{first} vs {second}'] = pair_windows
    return {'accuracy_validated': False, 'metric': 'branch_support_and_disagreement',
            'time_alignment': 'shared absolute recording clock' if absolute else 'relative output clocks; common exposure origin unverified',
            'common_timestamp_origin_s': common_origin, 'branch_origin_offsets_s': offsets,
            'strict_statuses': sorted(STRICT), 'max_interpolation_gap_s': float(max_gap_s),
            'branches': branch_metrics, 'pairwise': pairs,
            'note': 'Agreement is not accuracy. Phase-estimated and predicted states are excluded from strict coverage/rate metrics; wrapped disagreements cannot establish whole turns.'}
