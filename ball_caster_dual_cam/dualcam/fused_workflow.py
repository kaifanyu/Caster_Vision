"""Native-video inputs, a shared fused trajectory, and diagnostic overlays."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import copy
import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .config import CAMERA_NAMES, load_config, load_rig, write_json
from .fusion import FusionConfig, IncrementProposal, KeyframeView, SharedMotionFilter, fit_visual_pose
from .model import project, rotation_x, rotation_z
from .session import SelectedVideo, load_session, read_timestamps
from .tracking import choose_circle, masks_for
from .workflow import _output_directory, _record_provenance, calibration_hashes, load_axes


def motion_inputs(cfg, *, session=None, videos=None, timestamps=None, max_frames=None):
    if max_frames is not None and (isinstance(max_frames, bool) or int(max_frames) != max_frames or max_frames < 3):
        raise ValueError('max_frames must be an integer >= 3')
    if session is not None:
        if videos is not None or timestamps is not None:
            raise ValueError('Choose a recorded session OR two videos with timestamps')
        inputs = load_session(session, cfg, max_frames=max_frames, native=True)
        if inputs['metadata'].get('mode') != 'motion':
            raise ValueError('Native motion tracking needs a motion session')
    else:
        if videos is None or timestamps is None or len(videos) != 2 or len(timestamps) != 2:
            raise ValueError('Provide a session or both videos and their original timestamp CSVs')
        inputs = {'path': None, 'metadata': {'mode': 'motion'},
                  'video_paths': [Path(p).expanduser().resolve() for p in videos],
                  'raw_times': [read_timestamps(p)[:max_frames] for p in timestamps]}
    times = [np.array(t, float) for t in inputs['raw_times']]
    if any(len(t) < 3 for t in times):
        raise ValueError('Each camera needs at least three timestamped frames')
    times[1] += cfg['timing'].get('brio_offset_s', 0.)
    if max(times[0][0], times[1][0]) >= min(times[0][-1], times[1][-1]):
        raise ValueError('The camera recordings do not overlap on the corrected clock')
    # Each camera must see the supplied start pose. Reject a delayed second
    # recording rather than silently seeding its map at an unrelated pose.
    if abs(times[0][0]-times[1][0]) > .1:
        raise ValueError('Camera starts differ by >100 ms; trim both with original timestamps to a known common start')
    origin = min(t[0] for t in times)
    events = sorted((float(t-origin), ci, i) for ci, values in enumerate(times)
                    for i, t in enumerate(values))
    return {**inputs, 'events': events, 'timestamp_origin_s': float(origin)}


def run_fused_motion(config_path, output, *, initial_roll_deg, session=None,
                     videos=None, timestamps=None, max_frames=None, progress=print,
                     measurement_source='rotation'):
    out = _output_directory(output)
    report = {'kind': 'fused_motion', 'success': False, 'status': 'failed'}
    try:
        if not np.isfinite(initial_roll_deg):
            raise ValueError('initial_roll_deg must be the known starting roll angle')
        if measurement_source not in ('rotation', 'metric'):
            raise ValueError('measurement_source must be rotation or metric')
        cfg = load_config(config_path); cameras, infos, _ = load_rig(cfg); axes = load_axes(cfg)
        options = FusionConfig.from_mapping(cfg.get('fusion'))
        source = motion_inputs(cfg, session=session, videos=videos, timestamps=timestamps, max_frames=max_frames)
        _record_provenance(report, cfg, infos)
        if session is None:
            report['warnings'].append('External video capture settings are not verified by a session manifest; current rig settings are assumed. Timestamp CSVs must use the same clock and original decoded frame order.')
        report.update(config=cfg, fusion_config=asdict(options), geometry=cfg['geometry'],
                      measurement_source=measurement_source,
                      calibration_hashes=calibration_hashes(cfg), axes_sha256=axes['sha256'],
                      F=axes['R_bc'], pivot=axes['pivot_c920_m'],
                      videos=source['video_paths'], initial_roll_deg=float(initial_roll_deg),
                      use_video_manifest=session is not None,
                      timestamp_origin_s=source['timestamp_origin_s'],
                      timing_policy='Each native image updates one common state at recorded time plus configured camera offset; no pairing discard.',
                      orientation_reference='R_c920_from_caster = F @ Rx(roll); shell rotations additionally use Rz(shell spin). Shell spins start at zero in this recording.',
                      uncertainty_scope='Conditional on fixed calibration and mapped landmarks; does not include camera latency, calibration bias or correlated map drift.',
                      frames=[])
        F, pivot = axes['R_bc'], axes['pivot_c920_m']; geo = cfg['geometry']
        q0 = np.deg2rad([initial_roll_deg, 0., 0.])
        state = SharedMotionFilter(q0, source['events'][0][0], options)
        views = [KeyframeView(c, F, pivot, geo['radius_m'], geo['gap_m'], cfg.get('tracking'),
                              options, geo.get('red_shell_sign', 1)) for c in cameras]
        streams = []; maps = []; circles = [None, None]
        proposals = [None, None]
        counters = [Counter(), Counter()]
        try:
            for path, info in zip(source['video_paths'], infos):
                streams.append(SelectedVideo(path, info['image_size'], use_manifest=session is not None))
                maps.append(cv2.initUndistortRectifyMap(info['K'], info['dist'], None, info['K'],
                                                       tuple(info['image_size']), cv2.CV_32FC1))
            for event_index, (time, ci, frame_index) in enumerate(source['events']):
                seed = state.predict(time)
                frame = cv2.remap(streams[ci].read(frame_index), *maps[ci], cv2.INTER_LINEAR)
                if circles[ci] is None:
                    circles[ci] = choose_circle(frame, cfg['cameras'][CAMERA_NAMES[ci]])
                    proposals[ci] = IncrementProposal(cameras[ci], F, circles[ci], q0, cfg.get('tracking'), cfg.get('temporal'))
                masks = masks_for(frame, circles[ci], cfg['cameras'][CAMERA_NAMES[ci]])
                image_seed = proposals[ci].observe(frame, masks, frame_index, time)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                measured = []; visual = None; key_matches = 0; filter_accepted = False
                if frame_index == 0:
                    views[ci].promote(gray, masks, q0, frame_index)
                    counters[ci]['home_frames'] += 1
                elif measurement_source == 'rotation':
                    visual = proposals[ci].measurement(seed, options.max_off_mechanism_deg,
                                                        options.max_shell_roll_disagreement_deg,
                                                        options.rotation_measurement_std_deg)
                    if visual is not None:
                        ids = visual['components']
                        filter_accepted = state.update(ids, visual['angles'][ids], visual['covariance'])
                        if filter_accepted:
                            measured = ids.tolist()
                            counters[ci]['visual_updates'] += 1
                            for record in proposals[ci].records:
                                if record is not None:
                                    counters[ci]['rotation_'+record['status']] += 1
                        else:
                            counters[ci]['innovation_rejections'] += 1
                    else:
                        counters[ci]['rotation_unresolved_or_inconsistent'] += 1
                else:
                    # A large acceleration can outrun a constant-rate search.
                    # Prefer the fresh native-image hypothesis for alignment;
                    # it does not itself update the fused state.
                    observation = views[ci].observe(gray, masks, image_seed)
                    if observation is not None:
                        key_matches = observation['keyframe_matches']
                        visual = fit_visual_pose(cameras[ci], F, pivot, geo['radius_m'], geo['gap_m'],
                                                 observation['points'], observation['shells'], observation['uv'],
                                                 image_seed, options, geo.get('red_shell_sign', 1))
                        if visual['success']:
                            ids = visual['components']
                            filter_accepted = state.update(ids, visual['angles'][ids], visual['covariance'])
                            if filter_accepted:
                                measured = ids.tolist()
                                views[ci].promote(gray, masks, visual['angles'], frame_index,
                                                  observation, visual['inlier'], ids)
                                proposals[ci].anchor(visual['angles'])
                                counters[ci]['visual_updates'] += 1
                                counters[ci]['keyframe_matches'] += key_matches
                            else:
                                counters[ci]['innovation_rejections'] += 1
                        else:
                            counters[ci][visual['reason']] += 1
                    else:
                        counters[ci]['no_matches'] += 1
                snapshot = state.snapshot(measured)
                if event_index == 0:
                    snapshot['status'][:] = 'home'
                # Matrices/quaternions use one shared roll state, in C920's frame.
                q = snapshot['angles']
                R = F @ rotation_x(q[0]) if np.isfinite(q[0]) else None
                quaternion = Rotation.from_matrix(R).as_quat() if R is not None else None
                axes_direction = R[:, 2] if R is not None else None
                entry = {'time_s': time, 'camera': CAMERA_NAMES[ci], 'source_frame': frame_index,
                         **snapshot, 'caster_quaternion_xyzw': quaternion, 'swivel_axis_c920': axes_direction,
                         'visual_components': measured, 'keyframe_matches': key_matches,
                         'rotation_tracking': [None if record is None else {
                             'status': record['status'], 'adjacent': record['adjacent'],
                             'used_keyframes': record['used_keyframes'], 'anchor_rejection': record['anchor_rejection']}
                             for record in proposals[ci].records],
                         'rotation_gamma_deg': np.rad2deg(proposals[ci].gamma),
                         'rotation_shell_valid': proposals[ci].valid.copy(),
                         'rotation_candidate_angles': image_seed,
                         'visual': None if visual is None else {k: v for k, v in visual.items()
                                   if k != 'inlier'},
                         'filter_update_accepted': filter_accepted}
                report['frames'].append(entry)
                if progress and (event_index+1) % 100 == 0:
                    progress(f'Fused {event_index+1}/{len(source["events"])} native images; '
                             f'visual updates c920={counters[0]["visual_updates"]}, '
                             f'brio101={counters[1]["visual_updates"]}', flush=True)
        finally:
            for stream in streams:
                stream.close()
        statuses = np.array([f['status'] for f in report['frames']])
        coverage = {component: dict(Counter(statuses[:, i])) for i, component in enumerate(('roll', 'red', 'green'))}
        report.update(success=bool(any(c['visual_updates'] for c in counters)), status='completed',
                      summary={'native_images': len(source['events']), 'per_camera': dict(zip(CAMERA_NAMES, counters)),
                               'coverage': coverage, 'accepted_axis_calibration_modified': False,
                               'accuracy_validated': False,
                               'note': 'Completion is not full-clip measurement acceptance. Inspect vision/predicted/unresolved and turn_count_valid per component.'})
        write_json(out/'results.json', report)
        write_fused_csv(out/'results.csv', report['frames'])
        return report
    except Exception as exc:
        report.update(success=False, status='failed', error=str(exc))
        write_json(out/'results.json', report)
        raise


def write_fused_csv(path, frames):
    fields = ['time_s', 'camera', 'source_frame']
    for label in ('roll', 'red', 'green'):
        fields.extend([f'{label}_deg', f'{label}_deg_s', f'{label}_status', f'{label}_std_deg', f'{label}_turn_count_valid'])
    fields += ['caster_qx', 'caster_qy', 'caster_qz', 'caster_qw', 'swivel_x', 'swivel_y', 'swivel_z']
    with Path(path).open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(fields)
        for f in frames:
            row = [f['time_s'], f['camera'], f['source_frame']]
            for i in range(3):
                row += [np.rad2deg(f['angles'][i]), np.rad2deg(f['angular_velocity'][i]), f['status'][i],
                        np.rad2deg(f['std_rad'][i]), bool(f['turn_count_valid'][i])]
            row += list(f['caster_quaternion_xyzw']) if f['caster_quaternion_xyzw'] is not None else [None]*4
            row += list(f['swivel_axis_c920']) if f['swivel_axis_c920'] is not None else [None]*3
            writer.writerow(['' if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)) else v for v in row])


def refilter_rotation_report(original, options):
    """Tune the shared filter without decoding images or changing visual gates.

    Reuses only saved, accepted approximate-rotation measurements. Native image
    evidence and keyframe decisions remain exactly as originally recorded.
    """
    if original.get('kind') != 'fused_motion' or original.get('measurement_source') != 'rotation':
        raise ValueError('Refiltering requires a fused rotation-source result')
    if not original.get('frames'):
        raise ValueError('No saved native events to refilter')
    report = copy.deepcopy(original)
    report['fusion_config'] = asdict(options)
    report['refiltered_without_new_image_evidence'] = True
    state = SharedMotionFilter(np.deg2rad([report['initial_roll_deg'],0.,0.]), report['frames'][0]['time_s'], options)
    F = np.asarray(report['F'])
    camera_updates = Counter()
    for i, event in enumerate(report['frames']):
        predicted = state.predict(event['time_s'])
        visual = event.get('visual'); measured = []; accepted = False
        if visual is not None and visual.get('success') and visual.get('model') == 'enclosing_sphere_rotation':
            ids = np.asarray(visual['components'], int); q = np.asarray(visual['angles'], float)
            q = predicted+np.arctan2(np.sin(q-predicted),np.cos(q-predicted))
            accepted = state.update(ids,q[ids],np.eye(len(ids))*np.deg2rad(options.rotation_measurement_std_deg)**2)
            if accepted:
                measured=ids.tolist(); camera_updates[event['camera']]+=1
        snapshot=state.snapshot(measured)
        if i==0: snapshot['status'][:]='home'
        event.update(snapshot,visual_components=measured,filter_update_accepted=accepted)
        R=F @ rotation_x(snapshot['angles'][0]) if np.isfinite(snapshot['angles'][0]) else None
        event['caster_quaternion_xyzw']=Rotation.from_matrix(R).as_quat() if R is not None else None
        event['swivel_axis_c920']=R[:,2] if R is not None else None
    statuses=np.array([f['status'] for f in report['frames']])
    report['summary']['coverage']={name:dict(Counter(statuses[:,i])) for i,name in enumerate(('roll','red','green'))}
    report['summary']['refilter_visual_updates']=dict(camera_updates)
    report['summary']['original_per_camera']=copy.deepcopy(original['summary']['per_camera'])
    report['summary']['per_camera']={name:{'visual_updates':camera_updates[name]} for name in CAMERA_NAMES}
    report['success']=bool(sum(camera_updates.values()))
    return report


def render_fused(report, cfg, output):
    out = _output_directory(output)
    cameras, infos, _ = load_rig(cfg)
    from .workflow import calibration_matches
    if not calibration_matches(cfg, report.get('calibration_hashes')):
        raise ValueError('Results refer to different intrinsics/stereo')
    F, pivot = np.array(report['F']), np.array(report['pivot'])
    for ci, name in enumerate(CAMERA_NAMES):
        frames = [f for f in report['frames'] if f['camera'] == name]
        times = np.array([f['time_s'] for f in frames]); info = infos[ci]
        mapx,mapy=cv2.initUndistortRectifyMap(info['K'],info['dist'],None,info['K'],
                                             tuple(info['image_size']),cv2.CV_32FC1)
        source = SelectedVideo(Path(report['videos'][ci]), info['image_size'],
                               use_manifest=report.get('use_video_manifest', True))
        writer = cv2.VideoWriter(str(out/f'{name}_axes.avi'), cv2.VideoWriter_fourcc(*'MJPG'),
                                 float(1/np.median(np.diff(times))), tuple(info['image_size']))
        try:
            if not writer.isOpened():
                raise OSError('Could not open overlay video writer')
            for f in frames:
                image = cv2.remap(source.read(f['source_frame']),mapx,mapy,cv2.INTER_LINEAR)
                q = np.array(f['angles'], dtype=float); status = f['status'][0]
                if np.isfinite(q[0]) and status != 'unresolved':
                    directions = [F[:, 0], (F @ rotation_x(q[0]))[:, 2]]
                    colors = [(255, 180, 0), (0, 220, 255)] if status in ('vision', 'home') else [(0, 140, 255)]*2
                    for direction, color in zip(directions, colors):
                        xyz = pivot+report['geometry']['radius_m']*np.array([-direction, direction])
                        pixels = project(cameras[ci], xyz)
                        depth = (xyz @ cameras[ci]['R'].T+cameras[ci]['t'])[:, 2]
                        if (depth > 0).all() and np.isfinite(pixels).all() and abs(pixels).max() < 1e6:
                            a, b = np.rint(pixels).astype(int)
                            cv2.arrowedLine(image, tuple(a), tuple(b), color, 3, tipLength=.1)
                text = f'{name}  t={f["time_s"]:.3f}s  roll: {status}  red: {f["status"][1]}  green: {f["status"][2]}'
                cv2.putText(image, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
                cv2.putText(image, 'Shared two-camera state | orange: predicted | diagnostic playback; use CSV times',
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
                if not f['turn_count_valid'][0]:
                    cv2.putText(image, 'ROLL TURN COUNT UNVERIFIED AFTER GAP', (20, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, .65, (0,140,255), 2)
                writer.write(image)
        finally:
            source.close(); writer.release()
