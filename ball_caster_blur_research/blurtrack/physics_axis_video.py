"""Research-only, timestamp-faithful previews of bounded physics-fit trials.

The movie draws a hypothesis; it does not certify a recovered trajectory. Only
the fitted shell changes, and only inside the candidate's actual fit window.
Camera overlays use the displayed native image's corrected timestamp.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from dualcam.config import CAMERA_NAMES, load_rig, write_json
from dualcam.model import project, rotation_x, rotation_z
from dualcam.offline_render import (RenderState, draw_mesh, draw_simulation,
                                    _virtual_camera, nearest_frame)
from dualcam.session import SelectedVideo
from dualcam.workflow import calibration_hashes, load_axes


SIZE = (1600, 900)
CAMERA_SIZE = (776, 424)
BACKGROUND = (22, 19, 17)
PANEL = (35, 31, 28)
WHITE = (235, 238, 242)
MUTED = (168, 175, 180)
AMBER = (83, 180, 252)
SHELL_COLORS = ((93, 99, 246), (143, 222, 102))


def angles_at(payload, case, time_s):
    """Return (prior, trial, applied), without extrapolating local trial rates."""
    times = np.asarray(payload['baseline']['times'], float)
    angles = np.asarray(payload['baseline']['angles'], float)
    if not times[0] <= time_s <= times[-1]:
        return np.full(3, np.nan), np.full(3, np.nan), False
    prior = np.array([np.interp(time_s, times, angles[:, k]) for k in range(3)])
    trial = prior.copy()
    parameters = case.get('parameters_rad')
    applied = (case.get('status') == 'candidate' and parameters is not None and case['start_s'] - 1e-10 <= time_s
               <= case['end_s'] + 1e-10)
    if applied:
        phase, speed = np.asarray(parameters, float)
        trial[1 + int(case['shell_index'])] = phase + speed * (time_s - case['center_s'])
    return prior, trial, bool(applied)


def material_axes(F, pivot, geometry, angles):
    """World points [shell center, material X tip, material Y tip], per shell.

    These axes rotate with beta. Carrier X/Z alone cannot show swivel rotation.
    The signed shell centers remain separated by the measured rim-plane gap.
    """
    F, pivot, q = np.asarray(F), np.asarray(pivot), np.asarray(angles)
    B = F @ rotation_x(q[0])
    result = []
    for shell in (0, 1):
        sign = geometry.get('red_shell_sign', 1) * (1 - 2 * shell)
        center = pivot + B @ np.array([0., 0., sign * geometry['gap_m'] / 2])
        S = B @ rotation_z(q[shell + 1])
        length = float(geometry['radius_m']) * 1.20
        result.append(np.array([center, center + length * S[:, 0],
                                center + length * S[:, 1]]))
    return result


def _text(image, value, xy, size=.56, color=WHITE, width=1):
    cv2.putText(image, str(value), xy, cv2.FONT_HERSHEY_SIMPLEX, size,
                color, width, cv2.LINE_AA)


def _state(q):
    return RenderState(np.asarray(q), ('vision',) * 3, np.full(3, np.nan),
                       np.zeros(3, dtype=bool))


def draw_material_axes(image, camera, F, pivot, geometry, q, *, ghost=False,
                       only_shell=None):
    for shell, xyz in enumerate(material_axes(F, pivot, geometry, q)):
        if only_shell is not None and shell != only_shell:
            continue
        depth = (xyz @ np.asarray(camera['R']).T + camera['t'])[:, 2]
        uv = project(camera, xyz)
        if not (np.isfinite(uv).all() and np.all(depth > 0) and np.max(np.abs(uv)) < 1e6):
            continue
        uv = np.rint(uv).astype(int)
        for axis in (1, 2):
            color = (155, 155, 155) if ghost else SHELL_COLORS[shell]
            if axis == 2 and not ghost:
                color = tuple(int(.55 * c + .45 * 255) for c in color)
            cv2.arrowedLine(image, tuple(uv[0]), tuple(uv[axis]), color,
                            1 if ghost else 3, cv2.LINE_AA, tipLength=.09)
            if not ghost:
                label = ('R' if shell == 0 else 'G') + ('x' if axis == 1 else 'y')
                _text(image, label, tuple(uv[axis] + [5, -3]), .48, color, 2)
    return image


def _camera_id(frame):
    value = frame.get('camera_id', frame.get('camera'))
    return CAMERA_NAMES.index(value) if isinstance(value, str) else int(value)


def _source_index(frame):
    return int(frame.get('source_frame', frame.get('frame')))


def _case_frames(payload, case, ci):
    events = case.get('frames', payload['baseline']['native'])
    frames = [f for f in events if _camera_id(f) == ci
              and case['start_s'] - 1e-10 <= float(f['time_s']) <= case['end_s'] + 1e-10]
    if not frames:
        raise ValueError(f'{case["id"]}: no supported native frames for {CAMERA_NAMES[ci]}')
    return sorted(frames, key=lambda f: f['time_s'])


def _crop_camera(camera, info, pivot, geometry):
    """Fixed calibrated crop enclosing both separated shells and material axes."""
    extent = 1.55 * geometry['radius_m'] + geometry['gap_m'] / 2
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                        for z in (-1, 1)], float) * extent + pivot
    uv = project(camera, corners)
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    center = (lo + hi) / 2
    width, height = hi - lo
    aspect = CAMERA_SIZE[0] / CAMERA_SIZE[1]
    width = max(width, height * aspect)
    height = width / aspect
    x0, y0 = np.floor(center - [width / 2, height / 2]).astype(int)
    x1, y1 = np.ceil(center + [width / 2, height / 2]).astype(int)
    image_width, image_height = info['image_size']
    x0, x1 = max(0, x0), min(image_width, x1)
    y0, y1 = max(0, y0), min(image_height, y1)
    sx, sy = CAMERA_SIZE[0] / (x1 - x0), CAMERA_SIZE[1] / (y1 - y0)
    A = np.array([[sx, 0, -sx * x0], [0, sy, -sy * y0], [0, 0, 1.]])
    return {**camera, 'K': A @ np.asarray(camera['K'])}, (x0, y0, x1, y1)


def _render_frame(payload, case, time_s, frames, cached, cameras, crops,
                  slow_motion, loop, title=False):
    F, pivot = np.asarray(payload['F']), np.asarray(payload['pivot'])
    geometry = payload['geometry']
    prior, trial, applied = angles_at(payload, case, time_s)
    canvas = np.full((SIZE[1], SIZE[0], 3), BACKGROUND, np.uint8)
    _text(canvas, 'PHYSICS TRIAL / SIMULATED BODY AXES', (18, 38), .95, WHITE, 2)
    _text(canvas, 'UNVALIDATED local hypothesis | held-out test does not establish accurate recovery',
          (18, 68), .55, AMBER, 1)
    for ci, name in enumerate(CAMERA_NAMES):
        native_times = np.array([f['time_s'] for f in frames[ci]])
        source = frames[ci][nearest_frame(native_times, time_s)]
        source_time, source_index = float(source['time_s']), _source_index(source)
        q_prior, q_trial, source_applied = angles_at(payload, case, source_time)
        panel = cached[(ci, source_index)].copy()
        draw_mesh(panel, cameras[ci], F, pivot, geometry, _state(q_trial))
        draw_material_axes(panel, cameras[ci], F, pivot, geometry, q_prior,
                           ghost=True, only_shell=int(case['shell_index']))
        draw_material_axes(panel, cameras[ci], F, pivot, geometry, q_trial)
        cv2.rectangle(panel, (0, 0), (CAMERA_SIZE[0], 49), BACKGROUND, -1)
        _text(panel, f'CAM{ci} / {name.upper()}  |  native image {source_index}', (12, 21), .55)
        _text(panel, f'image t={source_time:.4f}s | overlay at this image time | instantaneous pose',
              (12, 42), .41, MUTED)
        cv2.rectangle(panel, (0, CAMERA_SIZE[1] - 31), CAMERA_SIZE, BACKGROUND, -1)
        _text(panel, 'TRIAL shell + fixed baseline context' if source_applied else 'Baseline context only',
              (12, CAMERA_SIZE[1] - 10), .49, AMBER)
        left = 16 + ci * 792
        canvas[86:86 + CAMERA_SIZE[1], left:left + CAMERA_SIZE[0]] = panel
    sim_size = (544, 302)
    simulation = draw_simulation(*sim_size, F, pivot, geometry, _state(trial))
    virtual = _virtual_camera(F, pivot, *sim_size)
    draw_material_axes(simulation, virtual, F, pivot, geometry, prior,
                       ghost=True, only_shell=int(case['shell_index']))
    draw_material_axes(simulation, virtual, F, pivot, geometry, trial)
    canvas[556:858, 16:560] = simulation
    _text(canvas, '3D MODEL / SHELL MATERIAL AXES', (26, 544), .56, WHITE, 2)
    _text(canvas, 'Gray: prior trial-shell axes | bright: current model', (27, 878), .44, MUTED)
    x = 585
    _text(canvas, case.get('label', case['id']), (x, 550), .78, WHITE, 2)
    shell = ('RED', 'GREEN')[int(case['shell_index'])]
    _text(canvas, f'{shell} local window {case["start_s"]:.4f} - {case["end_s"]:.4f}s',
          (x, 584), .58, SHELL_COLORS[int(case['shell_index'])], 2)
    speed = np.rad2deg(case['parameters_rad'][1])
    _text(canvas, f'Candidate angular speed: {speed:+.1f} deg/s', (x, 618), .62)
    _text(canvas, 'Status: ' + case.get('heldout_decision', case.get('status', 'unvalidated')),
          (x, 649), .49, AMBER)
    _text(canvas, f'Model time {time_s:.4f}s | {slow_motion:g}x slow motion | loop {loop + 1}',
          (x, 682), .56)
    _text(canvas, f'Carrier roll: {np.rad2deg(trial[0]):+.2f} deg (baseline)', (x, 715), .54)
    _text(canvas, 'Rx/Ry and Gx/Gy rotate with each shell; X/Z are carrier axes.',
          (x, 748), .46, MUTED)
    _text(canvas, 'Other shell + carrier retain baseline. Whole turns remain unverified.',
          (x, 776), .46, MUTED)
    _text(canvas, 'Slow-motion poses interpolate a model; repeated frames are not new observations.',
          (x, 804), .43, MUTED)
    _text(canvas, f'Physical radius {geometry["radius_m"] * 1000:.0f} mm | gap {geometry["gap_m"] * 1000:.0f} mm',
          (x, 833), .5, MUTED)
    if title:
        cv2.rectangle(canvas, (385, 348), (1215, 461), BACKGROUND, -1)
        _text(canvas, case.get('label', case['id']), (410, 394), .90, WHITE, 2)
        _text(canvas, 'Local fit preview - candidate, not validated recovery', (410, 433), .63, AMBER, 2)
    return canvas


def _ffmpeg():
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def render_physics_axes(payload_path, output_dir, fps=30., slow_motion=.1, loops=2):
    """Render bounded cases into one annotated H.264 movie and per-case posters."""
    if not 1 <= fps <= 120 or not 0 < slow_motion <= 1 or int(loops) != loops or loops < 1:
        raise ValueError('Expected fps 1..120, slow_motion in (0,1], positive integer loops')
    payload_path, output = Path(payload_path).resolve(), Path(output_dir).resolve()
    payload = json.loads(payload_path.read_text(encoding='utf-8'))
    cases = [case for case in payload['cases']
             if case.get('include_in_video', True) and case.get('status') == 'candidate'
             and case.get('parameters_rad') is not None]
    if not cases:
        raise ValueError('No finite local candidate is available to render')
    if calibration_hashes(payload['config']) != payload['calibration_hashes']:
        raise ValueError('Current calibration files differ from the saved physics preview')
    axes = load_axes(payload['config'])
    if not (np.allclose(axes['R_bc'], payload['F'], atol=1e-10, rtol=0)
            and np.allclose(axes['pivot_c920_m'], payload['pivot'], atol=1e-10, rtol=0)):
        raise ValueError('Current calibrated axes differ from the saved physics preview')
    times = np.asarray(payload['baseline']['times'], float)
    angles = np.asarray(payload['baseline']['angles'], float)
    if (times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0)
            or angles.shape != (len(times), 3)):
        raise ValueError('Baseline must contain increasing knots and three-angle rows')
    for case in cases:
        if not (times[0] <= case['start_s'] < case['end_s'] <= times[-1]
                and case['start_s'] <= case['center_s'] <= case['end_s']
                and len(case['parameters_rad']) == 2
                and np.isfinite(case['parameters_rad']).all()):
            raise ValueError(f'Invalid local support/parameters: {case["id"]}')
    output.mkdir(parents=True, exist_ok=True)
    movie, raw = output / 'physics_axes.mp4', output / 'physics_axes.rendering.mp4'
    for path in (movie, raw, output / 'render_metadata.json'):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite preview output: {path}')
    cameras, infos, _ = load_rig(payload['config'])
    cropped, crops, cache = [], [], {}
    all_frames = {case['id']: [_case_frames(payload, case, ci) for ci in (0, 1)] for case in cases}
    for ci in (0, 1):
        camera, crop = _crop_camera(cameras[ci], infos[ci], np.asarray(payload['pivot']), payload['geometry'])
        cropped.append(camera)
        crops.append(crop)
        info = infos[ci]
        remap = cv2.initUndistortRectifyMap(info['K'], info['dist'], None, info['K'],
                                          tuple(info['image_size']), cv2.CV_32FC1)
        indices = sorted({_source_index(frame) for frames in all_frames.values() for frame in frames[ci]})
        stream = SelectedVideo(payload['videos'][ci], info['image_size'])
        try:
            for frame_index in indices:
                frame = cv2.remap(stream.read(frame_index), *remap, cv2.INTER_LINEAR)
                x0, y0, x1, y1 = crop
                cache[(ci, frame_index)] = cv2.resize(frame[y0:y1, x0:x1], CAMERA_SIZE, interpolation=cv2.INTER_AREA)
        finally:
            stream.close()
        print(f'Decoded {len(indices)} native {CAMERA_NAMES[ci]} images', flush=True)
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), float(fps), SIZE)
    if not writer.isOpened():
        raise OSError('Could not open MPEG-4 writer')
    schedule, posters, frame_count = [], [], 0
    try:
        for case in cases:
            frames = all_frames[case['id']]
            poster = _render_frame(payload, case, case['center_s'], frames, cache, cropped, crops,
                                   slow_motion, 0)
            poster_path = output / f'poster_{case["id"]}.jpg'
            if not cv2.imwrite(str(poster_path), poster, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise OSError(f'Could not write {poster_path}')
            posters.append(str(poster_path))
            title = _render_frame(payload, case, case['center_s'], frames, cache, cropped, crops,
                                  slow_motion, 0, title=True)
            count = int(round(.8 * fps))
            schedule.append(dict(case_id=case['id'], video_start_s=frame_count / fps,
                                 video_end_s=(frame_count + count) / fps, source_start_s=case['center_s'],
                                 source_end_s=case['center_s'], speed=0., kind='title', loop=None))
            for _ in range(count):
                writer.write(title)
            frame_count += count
            for loop in range(loops):
                count = max(2, int(np.ceil((case['end_s'] - case['start_s']) / slow_motion * fps)) + 1)
                source_times = np.linspace(case['start_s'], case['end_s'], count)
                schedule.append(dict(case_id=case['id'], video_start_s=frame_count / fps,
                                     video_end_s=(frame_count + count) / fps, source_start_s=case['start_s'],
                                     source_end_s=case['end_s'], speed=slow_motion,
                                     effective_speed_between_frame_centers=(case['end_s'] - case['start_s']) * fps / (count - 1),
                                     kind='motion', loop=loop))
                for time_s in source_times:
                    writer.write(_render_frame(payload, case, float(time_s), frames, cache, cropped,
                                                crops, slow_motion, loop))
                frame_count += count
            print(f'Rendered {case["id"]} ({frame_count} montage frames)', flush=True)
    finally:
        writer.release()
    executable = _ffmpeg()
    codec = 'mpeg4'
    if executable:
        process = subprocess.run([executable, '-nostdin', '-hide_banner', '-loglevel', 'error',
                                  '-i', str(raw), '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
                                  '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', str(movie)],
                                 capture_output=True, text=True)
        if process.returncode == 0 and movie.exists() and movie.stat().st_size:
            raw.unlink()
            codec = 'h264'
        else:
            if movie.exists():
                movie.unlink()
            raw.replace(movie)
    else:
        raw.replace(movie)
    probe = cv2.VideoCapture(str(movie))
    decoded_count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_size = [int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)), int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    decoded_fps = float(probe.get(cv2.CAP_PROP_FPS))
    probe.release()
    if decoded_count != frame_count or decoded_size != list(SIZE):
        raise RuntimeError('Written preview frame count/dimensions failed verification')
    result = dict(movie=str(movie), posters=posters, fps=decoded_fps, frames=frame_count,
                  size=list(SIZE), codec=codec, duration_s=frame_count / fps,
                  payload_sha256=hashlib.sha256(payload_path.read_bytes()).hexdigest(),
                  segments=schedule, camera_crops_native_pixels=crops,
                  native_images_decoded=len(cache), cases=[case['id'] for case in cases],
                  camera_frame_policy='Nearest exact fit-window native image; overlay at its corrected timestamp.',
                  state_policy='Only selected shell replaced within local support; carrier and other shell remain baseline.',
                  axes_policy='Material X/Y axes use F Rx(alpha) Rz(beta_shell) about separated shell centers.',
                  interpolation_policy='Slow-motion poses are model interpolation; repeated video frames are not new observations.',
                  accuracy_validated=False, turn_count_valid=False, applied_to_global_trajectory=False)
    write_json(output / 'render_metadata.json', result)
    return result
