"""Visualize local physical-blur hypotheses without stitching a new trajectory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import ROOT
from dualcam.model import rotation_x, rotation_z


CASES = [
    ('red_control', 'Red control - 8.0 s', 'final_red8_control'),
    ('green_fast', 'Green fast swivel - 11.35 s', 'final_green11_atlas12'),
    ('red_fast', 'Red fast swivel - 19.2 s', 'final_red19_atlas12'),
    ('red_late', 'Red later window - 21.2 s', 'final_red21_atlas12'),
    ('red_strict', 'Red strict texture - unresolved', 'final_red19_strict'),
    ('green_strict', 'Green strict texture - unresolved', 'final_green11_strict'),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def material_axes(F, pivot, geometry, angles, shell_index):
    """Shell center and right-handed material frame, in calibrated coordinates."""
    if shell_index not in (0, 1):
        raise ValueError('shell_index must be 0 or 1')
    B = np.asarray(F) @ rotation_x(float(angles[0]))
    sign = geometry.get('red_shell_sign', 1) * (1 - 2*shell_index)
    center = np.asarray(pivot) + B[:, 2] * sign * geometry['gap_m']/2
    return center, B @ rotation_z(float(angles[shell_index+1]))


def sample_state(payload, case, time_s):
    """Return separate prior/trial states; no local candidate outside support."""
    t = float(time_s)
    if not np.isfinite(t):
        raise ValueError('time_s must be finite')
    baseline = payload['baseline']
    knots, angles = np.asarray(baseline['times']), np.asarray(baseline['angles'])
    prior = np.array([np.interp(t, knots, angles[:, k]) for k in range(3)])
    active = (case.get('status') == 'candidate' and case.get('parameters_rad') is not None
              and case['start_s'] <= t <= case['end_s'])
    trial = None
    if active:
        trial = prior.copy()
        phase, rate = case['parameters_rad']
        trial[case['shell_index']+1] = phase + rate*(t-case['center_s'])
    return {'baseline':prior, 'trial':trial, 'trial_active':active}


def native_trial_allowed(case, camera, source_frame):
    """Image overlays must refer to one of the exact frames fitted in this case."""
    return (case.get('status') == 'candidate' and case.get('parameters_rad') is not None
            and any(f['camera'] == camera and f.get('frame', f.get('source_frame')) == source_frame
                    for f in case['frames']))


def build_payload(research_root=ROOT):
    root = Path(research_root)
    source = root/'experiments/hybrid_dual'
    report_path = source/'results.json'
    bundle_path = source/'bundle.npz'
    report = json.loads(report_path.read_text())
    native_times = {(f['camera'], f['source_frame']):f['time_s'] for f in report['frames']}
    with np.load(bundle_path, allow_pickle=False) as saved:
        knots, angles = saved['knots'].copy(), saved['angles'].copy()
    cases = []
    source_hashes = {str(report_path):sha256(report_path), str(bundle_path):sha256(bundle_path)}
    for identifier, label, folder in CASES:
        path = root/'experiments/blur_physics'/folder/'results.json'
        result = json.loads(path.read_text())
        meta = result['metadata']
        atlas = meta['atlas_manifest']
        if atlas['baseline_sha256'] != sha256(report_path) or atlas['baseline_bundle_sha256'] != sha256(bundle_path):
            raise ValueError(f'Physical fit and viewer use different material gauges: {path}')
        if meta['fixed_geometry'] != report['geometry']:
            raise ValueError(f'Physical fit geometry differs: {path}')
        frames = meta['frames']
        for frame in frames:
            key = (frame['camera'], frame['frame'])
            if key not in native_times or not np.isclose(native_times[key], frame['time_s'], atol=1e-9, rtol=0):
                raise ValueError(f'Fit frame timing differs from saved native recording: {path}, {key}')
        times = [f['time_s'] for f in frames]
        prior = result.get('baseline_common_pixel_score', {}).get('test', {})
        fitted = result.get('fitted_common_pixel_score', {}).get('test', {})
        has_candidate = result['status'] == 'candidate' and result.get('best_parameters_rad') is not None
        old, new = prior.get('rmse'), fitted.get('rmse')
        decision = ('Unresolved: competing hypotheses cannot be compared reliably' if not has_candidate else
                    'Not accepted: held-out error worsened' if old is not None and new is not None and new > old else
                    'Local photometric candidate only; angular accuracy is unverified')
        case = dict(id=identifier, label=label, shell_index=('red','green').index(meta['shell']),
                    center_s=meta['center_s'], start_s=min(times), end_s=max(times),
                    parameters_rad=result.get('best_parameters_rad') if has_candidate else None,
                    prior_parameters_rad=meta['baseline_parameters_rad'],
                    status=result['status'], heldout_decision=decision, source_result=str(path),
                    source_sha256=sha256(path), frames=frames, prior_rmse=old, candidate_rmse=new,
                    heldout_pixels=prior.get('pixels', 0), accuracy_validated=False, turn_count_valid=False,
                    include_in_video=has_candidate)
        cases.append(case)
        source_hashes[str(path)] = sha256(path)
    return dict(schema_version=1, kind='local_physics_hypothesis_preview',
                F=report['F'], pivot=report['pivot'], geometry=report['geometry'],
                config=report['config'], videos=report['videos'], initial_roll_deg=report['initial_roll_deg'],
                calibration_hashes=report['calibration_hashes'],
                baseline={'times':knots.tolist(), 'angles':angles.tolist(),
                          'native':[{k:f[k] for k in ('camera','time_s','source_frame')} for f in report['frames']]},
                cases=cases, source_hashes=source_hashes,
                default_case='red_fast', applied_to_final_trajectory=False,
                model='R_shell = F Rx(alpha(t)) Rz(beta_shell(t)); displaced hemisphere centers retained',
                policy='Separate local hypotheses only. No stitching, phase blending, rate extrapolation or whole-turn certification. '
                       'The frozen hybrid trajectory supplies roll and the other shell. Display interpolation does not add image observations.',
                axis_legend={'X':'Fixed carrier roll axis', 'Z':'Rolled swivel axis (shared by both shells)',
                             'red X/Y':'Axes attached to red material', 'green X/Y':'Axes attached to green material'})


def write_preview(output, research_root=ROOT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = build_payload(research_root)
    template = Path(__file__).with_name('physics_preview.html').read_text(encoding='utf-8')
    text = json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('</', '<\/')
    metadata_path = output/'render_metadata.json'
    segments = json.loads(metadata_path.read_text()).get('segments', []) if metadata_path.exists() else []
    (output/'preview_data.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    (output/'orientation_3d.html').write_text(template.replace('__PHYSICS_PREVIEW_DATA__', text).replace(
        '__VIDEO_SEGMENTS__', json.dumps(segments, allow_nan=False)), encoding='utf-8')
    return payload
