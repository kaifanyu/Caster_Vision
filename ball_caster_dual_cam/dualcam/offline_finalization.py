"""A continuous best estimate with measured and prior-dependent phase separated."""
from __future__ import annotations

from collections import Counter
import copy

import numpy as np
from scipy.spatial.transform import Rotation

from .config import CAMERA_NAMES, write_json
from .model import rotation_x, rotation_z


def apply_phase_estimates(frames, summary, evidence, F, knots, fitted_angles, max_bridge_s=.15):
    """Retain the metric bundle curve, explicitly labeling short phase bridges.

    Angular evidence diagnoses observability; it does not replace the pixel fit.
    A brief support gap can be interpolated by the batch motion prior, but every
    subsequent accumulated phase depending on that bridge stays phase_estimated.
    Long gaps remain unresolved. Original strictly supported outputs are retained.
    """
    frames, summary = copy.deepcopy(frames), copy.deepcopy(summary)
    index = {(int(c), int(f)): i for i, (c, f) in enumerate(zip(evidence['event_camera'], evidence['event_frame']))}
    ids = np.array([index[(CAMERA_NAMES.index(f['camera']), f['source_frame'])] for f in frames])
    times = np.asarray(evidence['event_times_s'])[ids]
    if np.any(np.diff(times) < 0):
        raise ValueError('Evidence must follow corrected native time')
    np.testing.assert_allclose(times-times[0], [f['time_s'] for f in frames], atol=1e-9)
    q = np.column_stack([np.interp(times, knots, fitted_angles[:, k]) for k in range(3)])
    local = np.asarray(evidence['event_locally_supported'])[ids].copy()
    connected = np.asarray(evidence['event_phase_connected'])[ids]
    roll_usable = np.array([f['status'][0] in ('home', 'vision', 'predicted') for f in frames])
    local &= roll_usable[:, None]
    bridge = np.zeros_like(local)
    allowed = np.ones_like(local)
    gaps = []
    for h, name in enumerate(('red', 'green')):
        padded = np.r_[True, local[:, h], True]
        starts = np.flatnonzero(padded[:-1] & ~padded[1:])
        ends = np.flatnonzero(~padded[:-1] & padded[1:])
        for start, end in zip(starts, ends):
            before, after = max(0, start-1), min(len(times)-1, end)
            duration = float(times[after]-times[before])
            bounded = start > 0 and end < len(times) and duration <= max_bridge_s
            gaps.append({'shell': name, 'start_s': float(times[before]-times[0]),
                         'end_s': float(times[after]-times[0]), 'duration_s': duration,
                         'short_bridge_estimated': bounded})
            if bounded: bridge[start:end, h] = True
            elif start == 0 and duration <= max_bridge_s:
                bridge[start:end, h] = True
            elif end == len(times) and duration <= .05:
                bridge[start:end, h] = True
            else:
                allowed[start:, h] = False
    can_estimate = (local | bridge) & allowed & np.logical_and.accumulate(roll_usable)[:, None]
    summary['strict_metric_bundle_coverage'] = copy.deepcopy(summary['coverage'])
    summary['phase_bridge_gaps'] = gaps
    summary['phase_bridge_max_s'] = max_bridge_s
    summary['phase_estimate_policy'] = ('Final angles remain the unrestricted joint pixel-bundle solution. '
        'Conditional angular edges check local spin evidence and phase observability. Phase depending on a short '
        'motion-prior bridge is explicitly phase_estimated; whole-turn validity remains false. Long gaps stay unresolved.')
    camera_support = np.asarray(evidence['event_support_cameras'])[ids]
    turn_valid = np.ones(2, bool)
    for i, frame in enumerate(frames):
        frame['strict_angles'] = np.asarray(frame['angles'], float).copy()
        frame['strict_status'] = list(frame['status'])
        frame['status'] = list(frame['status'])
        frame['angles'] = np.asarray(frame['angles'], float)
        frame['turn_count_valid'] = list(frame['turn_count_valid'])
        frame['angular_track_counts'] = np.asarray(evidence['event_track_counts'])[ids[i]]
        frame['angular_normal_spread'] = np.asarray(evidence['event_normal_spread'])[ids[i]]
        frame['angular_phase_connected'] = connected[i]
        frame['component_measurement_source'] = ['joint_metric_pixels']*3
        for h in (0, 1):
            k = h+1
            strict = frame['status'][k] in ('home', 'vision') and (connected[i, h] or i == 0)
            if not strict: turn_valid[h] = False
            if not strict:
                frame['strict_angles'][k] = np.nan
                frame['strict_status'][k] = 'unresolved'
            if not strict and can_estimate[i, h]:
                frame['angles'][k] = q[i, k]
                frame['status'][k] = 'phase_estimated'
                frame['component_measurement_source'][k] = 'joint_metric_pixels_with_prior_dependent_phase'
                frame['support_cameras'][k] = [name for ci, name in enumerate(CAMERA_NAMES) if camera_support[i, h, ci]]
            elif not strict and not can_estimate[i, h]:
                frame['angles'][k] = np.nan
                frame['status'][k] = 'unresolved'
            frame['turn_count_valid'][k] = bool(frame['turn_count_valid'][k] and turn_valid[h])
        frame['relative_angles'] = frame['angles']-[q[0, 0], 0., 0.]
        R = F @ rotation_x(frame['angles'][0]) if np.isfinite(frame['angles'][0]) else None
        frame['shell_quaternions_xyzw'] = [Rotation.from_matrix(R @ rotation_z(frame['angles'][h+1])).as_quat()
            if R is not None and np.isfinite(frame['angles'][h+1]) else None for h in (0, 1)]
    final = np.array([f['angles'] for f in frames])
    unique, uid, inverse = np.unique(times, return_index=True, return_inverse=True)
    rate = np.gradient(final[uid], unique, axis=0)[inverse]
    for frame, velocity in zip(frames, rate): frame['angular_velocity'] = velocity
    strict_angles = np.asarray([f['strict_angles'] for f in frames])
    strict_rate = np.gradient(strict_angles[uid], unique, axis=0)[inverse]
    for frame, velocity in zip(frames, strict_rate): frame['strict_angular_velocity'] = velocity
    summary['coverage'] = {label: dict(Counter(f['status'][k] for f in frames))
                           for k, label in enumerate(('roll', 'red', 'green'))}
    summary['supported_event_fraction'] = {label: float(np.mean([f['status'][k] in ('home', 'vision') for f in frames]))
                                           for k, label in enumerate(('roll', 'red', 'green'))}
    summary['estimated_event_fraction'] = {label: float(np.mean(np.isfinite(final[:, k])))
                                           for k, label in enumerate(('roll', 'red', 'green'))}
    summary['turn_count_valid_at_end'] = frames[-1]['turn_count_valid']
    return frames, summary


def finalize_phase_estimates(report, data, seed, cameras, F, pivot, geometry, knots, angles, offset, output):
    from .spin_initialization import reseed_spins
    conditioned, diagnostics = reseed_spins(data, seed, cameras, F, pivot, geometry, knots, angles, offset)
    evidence = conditioned['spin_evidence']
    report['frames'], report['summary'] = apply_phase_estimates(report['frames'], report['summary'], evidence, F, knots, angles)
    report['conditional_spin_check'] = diagnostics
    report['measurement_source'] = 'joint_metric_with_phase_estimates'
    report['support_policy'] = report['summary']['phase_estimate_policy']
    report['success'] = bool(report['summary'].get('optimizer_converged') and
                             min(report['summary']['estimated_event_fraction'].values()) >= .5)
    report['accuracy_validated'] = False
    write_json(output/'conditional_spin_check.json', diagnostics)
    np.savez_compressed(output/'phase_evidence.npz', **{key: value for key, value in evidence.items() if isinstance(value, np.ndarray)})
    return report


def write_offline_csvs(output, frames):
    from .fused_workflow import write_fused_csv
    write_fused_csv(output/'results.csv', frames)
    strict = [{**f, 'angles': f['strict_angles'], 'status': f['strict_status'],
                'angular_velocity': f['strict_angular_velocity']} for f in frames]
    write_fused_csv(output/'strict_results.csv', strict)
