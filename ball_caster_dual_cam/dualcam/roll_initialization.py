"""Image-label roll hypotheses used ONLY to initialize metric bundle fitting.

A separating line between red and green paint is an approximate geometric cue,
not an angular measurement: paint density, perspective and yoke occlusion bias
its normal. The pixel bundle must independently assess every returned hypothesis.
"""
from __future__ import annotations

import copy
import cv2
import numpy as np

from .model import initialize_surface_point


def _wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def _projected_axis(camera, F, pivot):
    xyz = np.asarray(camera['R']) @ pivot + camera['t']
    homogeneous = np.asarray(camera['K']) @ xyz
    if homogeneous[2] <= 0:
        raise ValueError('Pivot must be in front of both cameras')
    uv = homogeneous[:2] / homogeneous[2]
    J = (camera['K'][:2] - uv[:, None] * camera['K'][2]) / homogeneous[2]
    basis = J @ camera['R'] @ F
    matrix = np.column_stack((-basis[:, 1], basis[:, 2]))
    if np.linalg.cond(matrix) > 20:
        raise ValueError('Projected roll direction is poorly observable in this view')
    return matrix


def _separator_direction(red, green):
    """Return a signed maximum-margin normal and conservative support details."""
    if min(len(red), len(green)) < 8:
        return None, {'reason': 'fewer_than_eight_points_per_shell'}
    uv = np.vstack((red, green)); labels = np.r_[np.ones(len(red)), -np.ones(len(green))].astype(np.int32)
    scaled = ((uv - np.mean(uv, axis=0)) / 100.).astype(np.float32)
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR); svm.setType(cv2.ml.SVM_C_SVC); svm.setC(10.)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 1000, 1e-6))
    svm.train(scaled, cv2.ml.ROW_SAMPLE, labels)
    _, alpha, indices = svm.getDecisionFunction(0)
    normal = (alpha.ravel() @ svm.getSupportVectors()[indices.ravel()]).ravel()
    length = np.linalg.norm(normal)
    if not np.isfinite(length) or length < 1e-8:
        return None, {'reason': 'degenerate_separator'}
    normal /= length
    if normal @ (red.mean(axis=0) - green.mean(axis=0)) < 0:
        normal *= -1
    error = float(np.mean(svm.predict(scaled)[1].ravel() != labels))
    gap = float(np.percentile(red @ normal, 10) - np.percentile(green @ normal, 90))
    centroid_distance = float(np.linalg.norm(red.mean(axis=0) - green.mean(axis=0)))
    report = {'reason': 'accepted', 'misclassified_fraction': error,
              'separation_px': gap, 'centroid_distance_px': centroid_distance,
              'counts': [len(red), len(green)]}
    if error > .05 or gap < 8. or centroid_distance < 30.:
        report['reason'] = 'weak_or_overlapping_shell_separation'
        return None, report
    report['weight'] = float(min(1., min(len(red), len(green)) / 30.) * min(1., gap / 40.))
    return normal, report


def _limited_interpolation(times, values, query, max_gap_s=.15):
    """Interpolate between observations only; never bridge a long missing span."""
    times = np.asarray(times); values = np.asarray(values); query = np.asarray(query)
    result = np.full(query.shape, np.nan)
    if not len(times):
        return result
    right = np.searchsorted(times, query, side='left')
    exact = (right < len(times)) & (times[np.minimum(right, len(times)-1)] == query)
    result[exact] = values[right[exact]]
    between = (~exact) & (right > 0) & (right < len(times))
    ii = np.flatnonzero(between)
    if len(ii):
        lo, hi = right[ii]-1, right[ii]
        ii = ii[(times[hi]-times[lo]) <= max_gap_s]
        lo, hi = right[ii]-1, right[ii]
        f = (query[ii]-times[lo])/(times[hi]-times[lo])
        result[ii] = (1-f)*values[lo]+f*values[hi]
    return result


def _initial_hold_bias(times, angles, initial_roll, duration_s=2.5):
    selected = np.isfinite(angles) & (times <= times[0]+duration_s)
    if selected.sum() < 12:
        return 0., {'applied': False, 'reason': 'too_few_initial_hold_cues'}
    deviations = _wrap(angles[selected]-initial_roll)
    bias = float(np.median(deviations))
    spread = float(np.percentile(deviations, 90)-np.percentile(deviations, 10))
    if abs(bias) > np.deg2rad(30) or spread > np.deg2rad(8):
        return 0., {'applied': False, 'reason': 'hold_moves_or_cue_disagrees_with_initial_pose',
                    'candidate_bias_deg': float(np.rad2deg(bias)), 'p90_p10_deg': float(np.rad2deg(spread))}
    return bias, {'applied': True, 'bias_deg': float(np.rad2deg(bias)),
                  'p90_p10_deg': float(np.rad2deg(spread)), 'samples': int(selected.sum())}


def reseed_roll(data, seed, cameras, F, pivot, geo, initial_roll_deg):
    """Return a new seed with image-supported roll corrections and new landmarks.

    Whole-turn selection uses temporal continuity, with the existing seed only
    selecting the branch after a gap over 150 ms. Gaps are never invented visual
    support. Spins stay unchanged; all material points are reinitialized from
    their first observed pixel and candidate pose for subsequent bundle fitting.
    """
    if not np.isfinite(initial_roll_deg):
        raise ValueError('initial_roll_deg must be finite')
    F, pivot = np.asarray(F), np.asarray(pivot)
    obs = data['observations']; events = seed['events']
    event_times = np.asarray([e[0] for e in events])
    old_angles = np.asarray(seed['angles'])
    if old_angles.shape != (len(events), 3):
        raise ValueError('Seed angles must match the event table')
    initial = np.deg2rad(initial_roll_deg)
    identities = np.column_stack([obs[k] for k in ('camera', 'shell', 'track')])
    _, identity = np.unique(identities, axis=0, return_inverse=True)
    counts = np.bincount(identity)
    long_track = counts[identity] >= 3
    per_camera, camera_angles, camera_weights = [], [], []
    for ci, camera in enumerate(cameras):
        native_times = np.asarray(data['times'][ci])
        angles = np.full(len(native_times), np.nan); weights = np.zeros(len(native_times))
        details = []
        try:
            matrix = _projected_axis(camera, F, pivot)
        except ValueError as exc:
            per_camera.append({'camera': ci, 'reason': str(exc), 'valid_frames': 0})
            camera_angles.append(np.full(len(events), np.nan)); camera_weights.append(np.zeros(len(events)))
            continue
        for fi in range(len(native_times)):
            take = (obs['camera'] == ci) & (obs['frame'] == fi) & long_track
            uv, shells = obs['uv'][take], obs['shell'][take]
            normal, report = _separator_direction(uv[shells == 0], uv[shells == 1])
            details.append(report)
            if normal is None:
                continue
            coefficients = np.linalg.solve(matrix, normal * geo.get('red_shell_sign', 1))
            angles[fi] = np.arctan2(coefficients[0], coefficients[1])
            weights[fi] = report['weight']
        bias, hold = _initial_hold_bias(native_times, angles, initial)
        valid = np.flatnonzero(np.isfinite(angles))
        angles[valid] = _wrap(angles[valid]-bias)
        # Unwrap each contiguous supported segment; use causal branch only after
        # a gap, where this approximate cue cannot establish whole revolutions.
        previous = None
        for fi in valid:
            if previous is None or native_times[fi]-native_times[previous] > .15:
                reference = np.interp(native_times[fi], event_times, old_angles[:, 0])
            else:
                reference = angles[previous]
            angles[fi] = reference + _wrap(angles[fi]-reference)
            previous = fi
        # A centered five-observation median suppresses point-set changes. It
        # must not reach across a missing interval or manufacture new support.
        filtered = angles.copy()
        for j, fi in enumerate(valid):
            neighbors = valid[max(0, j-2):j+3]
            neighbors = neighbors[abs(native_times[neighbors]-native_times[fi]) <= .075]
            filtered[fi] = np.median(angles[neighbors])
        interpolated = _limited_interpolation(native_times[valid], filtered[valid], event_times)
        iw = _limited_interpolation(native_times[valid], weights[valid], event_times)
        camera_angles.append(interpolated); camera_weights.append(np.nan_to_num(iw))
        per_camera.append({'camera': ci, 'condition_number': float(np.linalg.cond(matrix)),
                           'valid_frames': len(valid), 'total_frames': len(native_times),
                           'initial_hold_alignment': hold, 'native_roll_rad': angles,
                           'native_quality': details})
    camera_angles = np.asarray(camera_angles); camera_weights = np.asarray(camera_weights)
    joint = np.full(len(events), np.nan); disagreement = np.full(len(events), np.nan)
    views = np.sum(np.isfinite(camera_angles), axis=0)
    for i in range(len(events)):
        take = np.isfinite(camera_angles[:, i]) & (camera_weights[:, i] > 0)
        if not np.any(take):
            continue
        alpha, weight = camera_angles[take, i], camera_weights[take, i]
        if len(alpha) == 2:
            disagreement[i] = abs(_wrap(alpha[1]-alpha[0]))
            if disagreement[i] > np.deg2rad(35):
                # A disagreeing weak view cannot drag the joint hypothesis.
                if max(weight) < 2*min(weight):
                    continue
                keep = np.argmax(weight); alpha, weight = alpha[keep:keep+1], weight[keep:keep+1]
        wrapped = np.arctan2(np.sum(weight*np.sin(alpha)), np.sum(weight*np.cos(alpha)))
        reference = joint[i-1] if i and np.isfinite(joint[i-1]) and event_times[i]-event_times[i-1] <= .15 else old_angles[i, 0]
        joint[i] = reference + _wrap(wrapped-reference)
    candidate = old_angles.copy(); usable = np.isfinite(joint)
    delta = np.zeros(len(events)); delta[usable] = _wrap(joint[usable]-old_angles[usable, 0])
    # Preserve the precise causal seed where cues agree within a few degrees;
    # progressively replace roll once the approximate image cue shows drift.
    strength = np.clip((abs(delta)-np.deg2rad(3))/np.deg2rad(7), 0., 1.)
    candidate[:, 0] += strength*delta
    candidate[0, 0] = initial
    updated = copy.deepcopy(seed); updated['angles'] = candidate
    points = np.full_like(np.asarray(seed['points']), np.nan)
    first_row = {}
    for row, landmark in enumerate(seed['landmark']):
        if int(landmark) not in first_row:
            first_row[int(landmark)] = row
    for landmark, row in first_row.items():
        ci, fi, shell = (int(obs[k][row]) for k in ('camera', 'frame', 'shell'))
        t = data['times'][ci][fi]
        q = np.array([np.interp(t, event_times, candidate[:, k]) for k in range(3)])
        points[landmark] = initialize_surface_point(cameras[ci], obs['uv'][row], F, pivot, q,
                                                    shell, geo['radius_m'], geo['gap_m'], geo.get('red_shell_sign', 1))
    updated['points'] = points
    report = {'kind': 'approximate_image_roll_initialization',
              'usage': 'Initialization hypothesis only; never a final angle measurement or covariance update.',
              'per_camera': per_camera, 'joint_roll_rad': joint, 'joint_support_views': views,
              'joint_disagreement_deg': np.rad2deg(disagreement),
              'correction_deg': np.rad2deg(candidate[:, 0]-old_angles[:, 0]),
              'supported_events': int(usable.sum()), 'total_events': len(events),
              'changed_events_over_1deg': int(np.sum(abs(candidate[:, 0]-old_angles[:, 0]) > np.deg2rad(1))),
              'max_abs_correction_deg': float(np.rad2deg(abs(candidate[:, 0]-old_angles[:, 0]).max())),
              'min_track_frames': 3, 'max_interpolated_gap_s': .15,
              'landmarks_reinitialized': len(first_row)}
    return updated, report
