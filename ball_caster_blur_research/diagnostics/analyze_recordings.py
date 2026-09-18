"""Read-only blur audit of the existing dual-camera recordings and estimates.

Uses only the baseline's existing NumPy/OpenCV/PyYAML dependencies. All files
written by default stay in this new research project's diagnostics/output.
Example (from this directory): python analyze_recordings.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


CAMERAS = ('c920', 'brio101')
SHELLS = ('red', 'green')
WINDOWS = {'static_reference': (0.5, 2.5), 'reversal_11s': (11.0, 11.6),
           'phase_break_19s': (18.7, 19.5), 'phase_break_21s': (21.0, 21.4)}
CONTACT_TIMES = {'11s': (1., 11., 11.2, 11.4, 11.6),
                 '19s': (1., 18.7, 19., 19.2, 19.5),
                 '21s': (1., 21.1, 21.2, 21.3, 21.4)}


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'samples': 0, 'median': None, 'p95': None, 'maximum': None}
    return {'samples': len(values), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'maximum': float(np.max(values))}


def read_times(path):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    indices = np.array([int(r['frame_index']) for r in rows])
    times = np.array([float(r['timestamp_s']) for r in rows])
    if not np.array_equal(indices, np.arange(len(rows))) or np.any(np.diff(times) <= 0):
        raise ValueError(f'Invalid native frame order/timestamps: {path}')
    return times


def observed_exposure(camera_metadata):
    stages = {}
    for stage in ('controls', 'controls_before_stream', 'warmup_controls', 'final_controls'):
        controls = camera_metadata.get(stage, {})
        raw = controls.get('exposure_time_absolute', controls.get('exposure_absolute'))
        if raw is not None:
            stages[stage] = {'exposure_absolute_units': raw, 'exposure_us': int(raw)*100,
                             'auto_exposure_mode': controls.get('auto_exposure')}
    if not stages:
        raise ValueError('No actual camera exposure-control readback in session metadata')
    values = [r['exposure_us'] for r in stages.values()]
    exposure_us = stages.get('warmup_controls', stages.get('controls', list(stages.values())[0]))['exposure_us']
    return {'exposure_us': exposure_us, 'exposure_s': exposure_us/1e6,
            'readbacks_agree': len(set(values)) == 1, 'readback_stages': stages,
            'requested_exposure_us': camera_metadata.get('requested', {}).get('exposure_us'),
            'unit_conversion_source': 'baseline dualcam/capture.py maps exposure_us // 100 into V4L2 exposure_time_absolute',
            'interpretation': 'Read-back integration setting; individual exposure start/midpoint timestamps were not recorded.'}


def native_motion(cache, times, origin, exposures):
    """Surviving adjacent native KLT displacements, independent of angle fit."""
    with np.load(cache, allow_pickle=False) as archive:
        obs = {k: archive[k].copy() for k in ('camera', 'shell', 'frame', 'track', 'uv', 'weight')}
    order = np.lexsort((obs['frame'], obs['track'], obs['shell'], obs['camera']))
    first, second = order[:-1], order[1:]
    same = np.ones(len(first), bool)
    for key in ('camera', 'shell', 'track'):
        same &= obs[key][first] == obs[key][second]
    same &= obs['frame'][second] == obs['frame'][first] + 1
    first, second = first[same], second[same]
    ci = obs['camera'][first]
    t0 = np.array([times[c][f] for c, f in zip(ci, obs['frame'][first])])
    t1 = np.array([times[c][f] for c, f in zip(ci, obs['frame'][second])])
    speed = np.linalg.norm(obs['uv'][second]-obs['uv'][first], axis=1)/(t1-t0)
    exposure = np.array([exposures[CAMERAS[c]]['exposure_s'] for c in ci])
    return {'camera': ci, 'shell': obs['shell'][first], 'first_frame': obs['frame'][first],
            'second_frame': obs['frame'][second], 'midpoint_s': (t0+t1)/2-origin,
            'interval_s': t1-t0, 'speed_px_s': speed, 'blur_length_px': speed*exposure,
            'minimum_fb_weight': np.minimum(obs['weight'][first], obs['weight'][second])}


def native_statistics(edges, times, origin):
    frames = []
    for ci, name in enumerate(CAMERAS):
        for frame in range(1, len(times[ci])):
            for shell, shell_name in enumerate(SHELLS):
                take = (edges['camera'] == ci) & (edges['shell'] == shell) & (edges['second_frame'] == frame)
                speed, blur = quantiles(edges['speed_px_s'][take]), quantiles(edges['blur_length_px'][take])
                frames.append({'camera': name, 'shell': shell_name, 'source_frame': frame,
                               'receive_time_s': float(times[ci][frame]-origin),
                               'interval_ms': float(1000*(times[ci][frame]-times[ci][frame-1])),
                               'surviving_track_edges': speed['samples'], 'speed_median_px_s': speed['median'],
                               'speed_p95_px_s': speed['p95'], 'blur_median_px': blur['median'], 'blur_p95_px': blur['p95']})
    summaries = {}
    for window, (lo, hi) in {'all': (-np.inf, np.inf), **WINDOWS}.items():
        summaries[window] = {}
        for ci, name in enumerate(CAMERAS):
            summaries[window][name] = {}
            for shell, shell_name in enumerate(SHELLS):
                take = ((edges['camera'] == ci) & (edges['shell'] == shell)
                        & (edges['midpoint_s'] >= lo) & (edges['midpoint_s'] <= hi))
                summaries[window][name][shell_name] = {
                    'speed_px_s': quantiles(edges['speed_px_s'][take]),
                    'estimated_blur_length_px': quantiles(edges['blur_length_px'][take]),
                    'sample_intervals_over_50ms': int(np.sum(edges['interval_s'][take] > .05))}
    return frames, summaries


def estimated_rate_summary(results):
    output = {}
    frames = results['frames']
    for component, name in enumerate(('roll', 'red_spin', 'green_spin')):
        entries = []
        for frame in frames:
            omega = frame.get('angular_velocity', [None]*3)[component]
            if omega is None or not np.isfinite(omega):
                continue
            entries.append({'time_s': frame['time_s'], 'receive_time_s': frame.get('receive_time_s'),
                            'camera': frame['camera'], 'source_frame': frame['source_frame'],
                            'rate_deg_s': float(np.rad2deg(omega)), 'status': frame['status'][component],
                            'strict_status': frame.get('strict_status', frame['status'])[component],
                            'turn_count_valid': frame['turn_count_valid'][component]})
        groups = {'all_estimates': entries,
                  'strict_supported': [r for r in entries if r['strict_status'] in ('home', 'vision')],
                  'phase_estimated': [r for r in entries if r['status'] == 'phase_estimated']}
        output[name] = {}
        for group, rows in groups.items():
            output[name][group] = {'absolute_rate_deg_s': quantiles([abs(r['rate_deg_s']) for r in rows]),
                                   'peak_sample': max(rows, key=lambda r: abs(r['rate_deg_s'])) if rows else None}
    return {'warning': 'Existing model-derived rate estimates, NOT ground truth. Groups use existing PHASE status, not a dedicated velocity-observability test. phase_estimated can retain identifiable local angular velocity despite an unknown accumulated phase or turn count. Strict phase support also does not prove angular accuracy.',
            'components': output}


def shell_masks(frame, circle, camera_config):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    ball = (xx-circle[0])**2+(yy-circle[1])**2 < (.94*circle[2])**2
    masks = {'ball_roi': ball}
    for shell in SHELLS:
        spec = camera_config['segment'][shell+'_hsv']; low, high = spec['lo'], spec['hi']
        hue = ((hsv[:, :, 0] >= low[0]) & (hsv[:, :, 0] <= high[0])) if low[0] <= high[0] else ((hsv[:, :, 0] >= low[0]) | (hsv[:, :, 0] <= high[0]))
        paint = hue & (hsv[:, :, 1] >= low[1]) & (hsv[:, :, 1] <= high[1]) & (hsv[:, :, 2] >= low[2]) & (hsv[:, :, 2] <= high[2]) & ball
        masks[shell] = (cv2.dilate(paint.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ball
    return masks


def sharpness_metrics(frame, circle, config):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    result = {}
    for name, mask in shell_masks(frame, circle, config).items():
        count = int(mask.sum())
        result[name] = {'mask_pixels': count,
                        'laplacian_variance': float(np.var(laplacian[mask])) if count >= 64 else None,
                        'mean_gradient_energy': float(np.mean((gx*gx+gy*gy)[mask])) if count >= 64 else None,
                        'intensity_std': float(np.std(gray[mask])) if count >= 64 else None}
    return result


def annotated(image, lines):
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 24*len(lines)+4), (12, 12, 12), -1)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (7, 20+24*i), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def square_crop(frame, cx, cy, half):
    size = 2*int(half); result = np.full((size, size, 3), 20, np.uint8)
    x0, y0 = int(round(cx))-int(half), int(round(cy))-int(half)
    sx0, sy0 = max(0, x0), max(0, y0); sx1, sy1 = min(frame.shape[1], x0+size), min(frame.shape[0], y0+size)
    result[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = frame[sy0:sy1, sx0:sx1]
    return result


def image_audit(baseline, output, config, times, origin, exposures, frame_motion):
    rows, captured = [], {}
    motion_lookup = {(r['camera'], r['source_frame'], r['shell']): r for r in frame_motion}
    for ci, name in enumerate(CAMERAS):
        camera_config = config['cameras'][name]; circle = np.array(camera_config['circle'], float)
        calibration = yaml.safe_load((baseline/'config'/camera_config['intrinsics']).resolve().read_text())
        K, dist = np.asarray(calibration['K'], float), np.asarray(calibration['dist'], float)
        mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K, tuple(calibration['image_size']), cv2.CV_32FC1)
        relative = times[ci]-origin
        selected = set(np.flatnonzero((relative >= .5) & (relative <= 2.5))[::8].tolist())
        for label, (lo, hi) in WINDOWS.items():
            if label != 'static_reference':
                selected.update(np.flatnonzero((relative >= lo) & (relative <= hi)).tolist())
        for sequence in CONTACT_TIMES.values():
            selected.update(int(np.argmin(abs(relative-t))) for t in sequence)
        video = cv2.VideoCapture(str(baseline/'data/run'/f'{name}.avi'), cv2.CAP_OPENCV_MJPEG)
        if not video.isOpened():
            raise ValueError(f'Cannot read source {name} AVI')
        try:
            for fi in sorted(selected):
                video.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, raw = video.read()
                if not ok:
                    raise ValueError(f'Missing {name} frame {fi}')
                frame = cv2.remap(raw, mapx, mapy, cv2.INTER_LINEAR)
                for region, metrics in sharpness_metrics(frame, circle, camera_config).items():
                    rows.append({'camera': name, 'source_frame': fi, 'receive_time_s': float(relative[fi]),
                                 'region': region, **metrics})
                contact_needed = any(fi == int(np.argmin(abs(relative-t))) for seq in CONTACT_TIMES.values() for t in seq)
                if contact_needed:
                    tile = cv2.resize(square_crop(frame, *circle[:2], int(circle[2])), (320, 320), interpolation=cv2.INTER_AREA)
                    blur = [motion_lookup.get((name, fi, s), {}).get('blur_p95_px') for s in SHELLS]
                    blur_text = '/'.join('NA' if b is None else f'{b:.1f}' for b in blur)
                    tile = annotated(tile, [f'{name} #{fi} t={relative[fi]:.3f}s', f'exposure {exposures[name]["exposure_us"]/1000:g} ms', f'KLT blur p95 R/G: {blur_text} px'])
                    patch = square_crop(frame, circle[0]-.42*circle[2], circle[1], 120)
                    native = np.full((260, 320, 3), 20, np.uint8); native[20:260, 40:280] = patch
                    native = annotated(native, ['Native 240x240 fixed texture crop'])
                    captured[name, fi] = np.vstack((tile, native))
        finally:
            video.release()
        print(f'{name}: audited {len(selected)} native images', flush=True)
    for label, sequence in CONTACT_TIMES.items():
        panels = []
        for ci, name in enumerate(CAMERAS):
            indices = [int(np.argmin(abs(times[ci]-origin-t))) for t in sequence]
            panels.append(np.hstack([captured[name, fi] for fi in indices]))
        cv2.imwrite(str(output/f'contact_{label}.jpg'), np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 94])
    summaries = {}
    for label, (lo, hi) in WINDOWS.items():
        summaries[label] = {}
        for name in CAMERAS:
            summaries[label][name] = {}
            for region in ('ball_roi', *SHELLS):
                selected = [r for r in rows if r['camera'] == name and r['region'] == region and lo <= r['receive_time_s'] <= hi]
                summaries[label][name][region] = {'laplacian_variance': quantiles([r['laplacian_variance'] for r in selected]),
                                                  'gradient_energy': quantiles([r['mean_gradient_energy'] for r in selected])}
    for label in WINDOWS:
        for name in CAMERAS:
            for region in ('ball_roi', *SHELLS):
                base = summaries['static_reference'][name][region]['laplacian_variance']['median']
                now = summaries[label][name][region]['laplacian_variance']['median']
                summaries[label][name][region]['laplacian_median_ratio_to_static'] = now/base if now is not None and base else None
    return rows, summaries


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path(__file__).resolve().parents[2]/'ball_caster_dual_cam')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'output')
    args = parser.parse_args(); baseline, output = args.baseline.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True); cv2.setNumThreads(2)
    session_path = baseline/'data/run/session.json'; config_path = baseline/'config/rig.yaml'
    result_path = baseline/'out/orientation_final/results.json'; cache_path = baseline/'out/joint_native_cache/native_tracks.npz'
    session = json.loads(session_path.read_text()); config = yaml.safe_load(config_path.read_text())
    results = json.loads(result_path.read_text())
    angular_check_path = baseline/'out/orientation_final/conditional_spin_check.json'
    angular_check = json.loads(angular_check_path.read_text())
    times = [read_times(baseline/'data/run'/f'{name}_timestamps.csv') for name in CAMERAS]
    origin = min(t[0] for t in times)
    exposures = {name: observed_exposure(session['cameras'][name]) for name in CAMERAS}
    source_paths = [session_path, config_path, result_path, cache_path, angular_check_path]
    camera_reports = {}
    for ci, name in enumerate(CAMERAS):
        video_path = baseline/'data/run'/f'{name}.avi'; timestamp_path = baseline/'data/run'/f'{name}_timestamps.csv'
        source_paths.extend((video_path, timestamp_path))
        video = cv2.VideoCapture(str(video_path), cv2.CAP_OPENCV_MJPEG)
        shape = [int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT)); nominal = video.get(cv2.CAP_PROP_FPS); video.release()
        dt = np.diff(times[ci]); circle = config['cameras'][name]['circle']
        camera_reports[name] = {'video_frames': count, 'timestamp_frames': len(times[ci]), 'image_size': shape,
                                'nominal_video_fps': nominal, 'observed_fps': (len(times[ci])-1)/(times[ci][-1]-times[ci][0]),
                                'interval_ms': quantiles(1000*dt), 'intervals_over_50ms': int(np.sum(dt > .05)),
                                'undistorted_configured_circle_xy_radius_px': circle,
                                'configured_ball_diameter_px': 2*circle[2], 'circle_scope': 'Existing reviewed enclosing circle, not a new measured physical silhouette.',
                                'actual_recorded_exposure': exposures[name],
                                'session_drop_estimate': session.get('stats', {}).get(name, {})}
    edges = native_motion(cache_path, times, origin, exposures)
    frame_motion, motion_summary = native_statistics(edges, times, origin)
    sharpness_rows, sharpness_summary = image_audit(baseline, output, config, times, origin, exposures, frame_motion)
    write_csv(output/'native_motion_by_frame.csv', frame_motion)
    write_csv(output/'targeted_sharpness.csv', sharpness_rows)
    report = {'baseline': baseline, 'read_only_input_files': {str(p): {'sha256': file_hash(p), 'bytes': p.stat().st_size} for p in source_paths},
              'versions': {'opencv': cv2.__version__, 'numpy': np.__version__},
              'timestamp_origin_s': origin, 'diagnostic_clock': 'Raw host receive timestamps minus common earliest timestamp; video nominal FPS is not used.',
              'cameras': camera_reports, 'native_pixel_motion': motion_summary,
              'sharpness_proxy': sharpness_summary, 'estimated_angular_rates': estimated_rate_summary(results),
              'ambiguity': {'strict_supported_event_fraction': results['summary'].get('supported_event_fraction'),
                            'locally_supported_angular_event_fraction': angular_check.get('angular_evidence', {}).get('locally_supported_event_fraction'),
                            'local_angular_support_gaps': angular_check.get('angular_evidence', {}).get('local_support_gaps'),
                            'display_coverage': results['summary'].get('coverage'),
                            'phase_bridge_gaps': results['summary'].get('phase_bridge_gaps'),
                            'turn_count_valid_at_end': results['summary'].get('turn_count_valid_at_end'),
                            'fitted_brio_time_offset_s': results['summary'].get('additional_brio_offset_s'),
                            'accuracy_validated': results.get('accuracy_validated', False)},
              'limitations': [
                  'Blur length is surviving native KLT pixel displacement / receive-time interval * read-back exposure. It assumes locally constant image velocity and exposure duration matching its control readback.',
                  'Native pixel velocity is measured in undistorted original-K pixels. Raw-image distortion, temporal acceleration, shutter/readout behavior, sensor latency and USB buffering introduce uncertainty.',
                  'Failed or occluded tracks are absent: displacement and blur summaries are survivor-biased and can underestimate the hardest motion. Host receive gaps are not measured exposure gaps.',
                  'Laplacian and gradient statistics are sharpness proxies, not blur-kernel or MTF measurements. Shell pose, paint density, mask thresholding, illumination and noise confound comparisons; absolute cross-camera values are not equivalent.',
                  'Orientation-derived rates are model estimates. phase_estimated flags accumulated phase/turn ambiguity and does not imply all subsequent local velocities are unobserved. A constant unknown phase cancels in a derivative. Existing local angular support is a useful proxy, not an independent validation of rate accuracy.',
                  'A subsequent estimator should export separate rate_status, phase_valid and turn_count_valid fields, and test derivative observability within each locally connected component independently of its absolute phase.',
                  'Contact-sheet full-ball panels are resized for context. The fixed 240x240 texture patches preserve native pixel scale; JPEG sheet compression is for inspection only.'],
              'reproduce': f'python "{Path(__file__).resolve()}" --baseline "{baseline}" --output "{output}"'}
    write_json(output/'blur_diagnostics.json', report)
    print(json.dumps(serializable({'cameras': camera_reports,
                                  'target_motion': {k: v for k, v in motion_summary.items() if k != 'all'},
                                  'estimated_rates': report['estimated_angular_rates']}), indent=2), flush=True)


if __name__ == '__main__':
    main()
