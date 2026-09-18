"""Timestamp-faithful two-camera replay of the saved common caster state.

Camera panels use the estimate at the displayed image's corrected timestamp.
The synthetic view uses the output timestamp. Neither frame selection nor
interpolation changes the saved estimate; unknown states and long gaps stay
unknown. The wire markings illustrate fitted phase, not measured surface marks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from .config import CAMERA_NAMES, load_rig, write_json
from .model import project, rotation_x, rotation_z, world_points
from .session import SelectedVideo
from .visualization import build_viewer_payload


SIZE = (1600, 900)
BACKGROUND = (22, 19, 17)
PANEL = (35, 31, 28)
WHITE = (236, 239, 242)
MUTED = (168, 175, 180)
SHELL_COLORS = ((93, 99, 246), (143, 222, 102))
STATUS_COLORS = {'home': (224, 218, 137), 'vision': (174, 221, 135),
                 'predicted': (83, 180, 252), 'phase_estimated': (91, 191, 255),
                 'unresolved': (138, 144, 152)}
STATUS_LABELS = {'phase_estimated': 'phase estimate'}


@dataclass
class RenderState:
    angles: np.ndarray
    status: tuple[str, str, str]
    std_rad: np.ndarray
    turn_count_valid: np.ndarray


class TrajectorySampler:
    """Sample unwrapped saved angles without bridging unresolved observations."""

    def __init__(self, frames, max_gap_s=.15):
        if not np.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError('max_gap_s must be positive and finite')
        self.max_gap_s = float(max_gap_s)
        # At simultaneous camera events the last state has seen both images.
        unique = {}
        for frame in frames:
            unique[float(frame['time_s'])] = frame
        if not unique:
            raise ValueError('No trajectory to render')
        self.times = np.array(sorted(unique))
        self.frames = [unique[t] for t in self.times]
        self.angles = np.array([f['angles'] for f in self.frames], float)
        self.status = np.array([f['status'] for f in self.frames])
        self.angles[self.status == 'unresolved'] = np.nan
        self.std = np.array([f.get('std_rad', [None] * 3) for f in self.frames], float)
        self.turns = np.array([f.get('turn_count_valid', [False] * 3)
                               for f in self.frames], bool)
        self.turns[np.isin(self.status, ('unresolved', 'phase_estimated'))] = False

    @staticmethod
    def unknown():
        return RenderState(np.full(3, np.nan), ('unresolved',) * 3,
                           np.full(3, np.nan), np.zeros(3, bool))

    def at(self, time_s):
        j = int(np.searchsorted(self.times, time_s, side='right')) - 1
        if j >= 0 and np.isclose(time_s, self.times[j], atol=1e-9, rtol=0):
            return RenderState(self.angles[j].copy(), tuple(self.status[j]),
                               self.std[j].copy(), self.turns[j].copy())
        if j < 0 or j + 1 >= len(self.times):
            return self.unknown()
        span = self.times[j + 1] - self.times[j]
        if span > self.max_gap_s + 1e-10:
            return self.unknown()
        weight = (time_s - self.times[j]) / span
        valid = np.isfinite(self.angles[j]) & np.isfinite(self.angles[j + 1])
        angles = np.full(3, np.nan)
        # States are already unwrapped: shortest-path interpolation would erase
        # legitimate high-speed revolutions in an offline estimate.
        angles[valid] = (1 - weight) * self.angles[j, valid] + weight * self.angles[j + 1, valid]
        statuses = []
        for k in range(3):
            endpoints = (self.status[j, k], self.status[j + 1, k])
            statuses.append('unresolved' if not valid[k] else
                            'phase_estimated' if 'phase_estimated' in endpoints else
                            'predicted' if 'predicted' in endpoints else 'vision')
        std = np.maximum(self.std[j], self.std[j + 1])
        std[~valid] = np.nan
        return RenderState(angles, tuple(statuses), std, self.turns[j] & self.turns[j + 1] & valid)


def nearest_frame(times, time_s):
    """Closest image on the corrected camera clock; ties use the earlier image."""
    j = int(np.searchsorted(times, time_s))
    candidates = [k for k in (j - 1, j) if 0 <= k < len(times)]
    return min(candidates, key=lambda k: (abs(times[k] - time_s), k))


def hemisphere_curves(sign):
    """Unit hemisphere rim, latitude and longitude curves, with a phase stripe."""
    azimuth = np.linspace(0, 2 * np.pi, 97)
    curves = []
    for latitude in (0., 25., 50., 72.):
        lat = np.deg2rad(latitude)
        curve = np.column_stack((np.cos(lat) * np.cos(azimuth),
                                 np.cos(lat) * np.sin(azimuth),
                                 np.full(len(azimuth), sign * np.sin(lat))))
        curves.append((curve, 'rim' if latitude == 0 else 'latitude'))
    lat = np.linspace(0, np.pi / 2, 37)
    for phase in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        curve = np.column_stack((np.cos(lat) * np.cos(phase),
                                 np.cos(lat) * np.sin(phase), sign * np.sin(lat)))
        curves.append((curve, 'phase' if phase == 0 else 'longitude'))
    return curves


def _camera_origin(camera):
    return -np.asarray(camera['R']).T @ np.asarray(camera['t'])


def visible_surface(camera, xyz, normals, centers, B, signs, radius):
    """Cull rear-facing points and ray hits occluded by either signed shell."""
    origin = _camera_origin(camera)
    to_point = xyz - origin
    distance = np.linalg.norm(to_point, axis=1)
    direction = to_point / np.maximum(distance[:, None], 1e-12)
    depth = (xyz @ np.asarray(camera['R']).T + camera['t'])[:, 2]
    visible = (np.einsum('ij,ij->i', normals, -direction) >= -0.015) & (depth > 1e-6)
    for center, sign in zip(centers, signs):
        offset = origin - center
        b = direction @ offset
        disc = b * b - (offset @ offset - radius * radius)
        roots = np.sqrt(np.maximum(disc, 0))
        for depth_hit in (-b - roots, -b + roots):
            hit_local_z = ((origin + depth_hit[:, None] * direction - center) @ B)[:, 2]
            occluded = ((disc >= 0) & (depth_hit > 1e-6)
                        & (depth_hit < distance - radius * .002)
                        & (sign * hit_local_z >= -radius * .0001))
            visible &= ~occluded
    return visible


def _draw_visible_curve(image, pixels, visible, color, thickness):
    valid = visible & np.isfinite(pixels).all(axis=1) & (np.abs(pixels).max(axis=1) < 1e6)
    padded = np.r_[False, valid, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    for start, end in zip(starts, ends):
        if end - start >= 2:
            points = np.rint(pixels[start:end]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [points], False, color, thickness, cv2.LINE_AA)


def draw_mesh(image, camera, F, pivot, geometry, state, *, thickness=1):
    """Project the physical separated hemispheres into ideal camera pixels."""
    q = state.angles.copy()
    if not np.isfinite(q[0]):
        return image
    radius, gap = geometry['radius_m'], geometry['gap_m']
    signs = [geometry.get('red_shell_sign', 1), -geometry.get('red_shell_sign', 1)]
    B = F @ rotation_x(q[0])
    centers = [pivot + B @ [0, 0, sign * gap / 2] for sign in signs]
    for shell, sign in enumerate(signs):
        known_phase = np.isfinite(q[shell + 1])
        angles = q.copy()
        angles[~np.isfinite(angles)] = 0.
        S = B @ rotation_z(angles[shell + 1])
        color = SHELL_COLORS[shell] if known_phase else MUTED
        if state.status[shell + 1] == 'predicted':
            color = tuple(int(c * .8) for c in color)
        for curve, kind in hemisphere_curves(sign):
            # Axially symmetric geometry is still known when shell phase is not.
            if not known_phase and kind in ('longitude', 'phase'):
                continue
            xyz = world_points(F, pivot, angles, shell, curve, radius, gap,
                               geometry.get('red_shell_sign', 1))
            visible = visible_surface(camera, xyz, curve @ S.T, centers, B, signs, radius)
            stroke = WHITE if kind == 'phase' else color
            if state.status[shell + 1] == 'phase_estimated' and kind in ('longitude', 'phase'):
                stroke = STATUS_COLORS['phase_estimated']
            width = thickness + (kind in ('rim', 'phase'))
            _draw_visible_curve(image, project(camera, xyz), visible, stroke, width)
    _draw_axes(image, camera, F, pivot, q[0], radius, thickness + 1)
    return image


def _draw_axes(image, camera, F, pivot, roll, radius, thickness):
    B = F @ rotation_x(roll)
    for direction, color, label in ((F[:, 0], (245, 184, 61), 'X'),
                                     (B[:, 2], (77, 225, 250), 'Z')):
        xyz = pivot + np.array([-1.3, 1.3])[:, None] * radius * direction
        depth = (xyz @ np.asarray(camera['R']).T + camera['t'])[:, 2]
        pixels = project(camera, xyz)
        if np.all(depth > 0) and np.isfinite(pixels).all() and np.abs(pixels).max() < 1e6:
            start, end = np.rint(pixels).astype(int)
            cv2.arrowedLine(image, tuple(start), tuple(end), color, thickness,
                            cv2.LINE_AA, tipLength=.065)
            cv2.putText(image, label, tuple(end + [7, -5]), cv2.FONT_HERSHEY_SIMPLEX,
                        .55 * thickness, color, max(1, thickness // 2), cv2.LINE_AA)


def _virtual_camera(F, pivot, width, height):
    eye = np.array([.34, -.52, .32])
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    Rhome = np.array([right, down, forward])
    R = Rhome @ F.T
    distance = np.linalg.norm(eye)
    focal = height * 2.5
    return {'R': R, 't': np.array([0., 0., distance]) - R @ pivot,
            'K': np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.]])}


def draw_simulation(width, height, F, pivot, geometry, state):
    image = np.full((height, width, 3), PANEL, np.uint8)
    if not np.isfinite(state.angles[0]):
        _text(image, 'Orientation unresolved', (width // 2 - 135, height // 2), .75, MUTED)
        return image
    camera = _virtual_camera(F, pivot, width, height)
    q = np.nan_to_num(state.angles, nan=0.)
    B = F @ rotation_x(q[0])
    radius, gap = geometry['radius_m'], geometry['gap_m']
    faces = []
    light = np.array([-.4, -.6, -1.])
    light /= np.linalg.norm(light)
    origin = _camera_origin(camera)
    for shell in (0, 1):
        sign = geometry.get('red_shell_sign', 1) * (1 - 2 * shell)
        longitude = np.linspace(0, 2 * np.pi, 25)
        latitude = np.linspace(0, np.pi / 2, 11)
        az, lat = np.meshgrid(longitude, latitude)
        local = np.stack((np.cos(lat) * np.cos(az), np.cos(lat) * np.sin(az),
                          sign * np.sin(lat)), axis=-1)
        xyz = world_points(F, pivot, q, shell, local.reshape(-1, 3), radius, gap,
                           geometry.get('red_shell_sign', 1)).reshape(local.shape)
        S = B @ rotation_z(q[shell + 1])
        base = np.array(SHELL_COLORS[shell] if np.isfinite(state.angles[shell + 1]) else MUTED)
        for row in range(len(latitude) - 1):
            for col in range(len(longitude) - 1):
                vertices = xyz[[row, row, row + 1, row + 1], [col, col + 1, col + 1, col]]
                center = vertices.mean(axis=0)
                normal = S @ local[row:row + 2, col:col + 2].mean(axis=(0, 1))
                if normal @ (origin - center) < 0:
                    continue
                depth = (camera['R'] @ center + camera['t'])[2]
                shade = .30 + .64 * max(0., (camera['R'] @ normal) @ light)
                color = tuple(int(v) for v in np.clip(base * shade, 0, 255))
                faces.append((depth, np.rint(project(camera, vertices)).astype(np.int32), color))
    for _, polygon, color in sorted(faces, key=lambda item: item[0], reverse=True):
        cv2.fillConvexPoly(image, polygon, color, cv2.LINE_AA)
    draw_mesh(image, camera, F, pivot, geometry, state)
    return image


def _text(image, text, location, scale=.6, color=WHITE, thickness=1):
    cv2.putText(image, str(text), location, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _timeline(sampler, width=502, height=275):
    image = np.full((height, width, 3), PANEL, np.uint8)
    start, end = sampler.times[[0, -1]]
    duration = max(end - start, 1e-9)
    x = 66 + (width - 84) * (sampler.times - start) / duration
    for k, (label, color) in enumerate(zip(('ROLL', 'RED', 'GREEN'),
                                          ((77, 225, 250), *SHELL_COLORS))):
        y0 = 18 + k * 82
        _text(image, label, (10, y0 + 29), .45, color)
        values = np.rad2deg(sampler.angles[:, k])
        finite = np.isfinite(values)
        if not finite.any():
            _text(image, 'unresolved', (90, y0 + 29), .48, MUTED)
            continue
        lo, hi = np.min(values[finite]), np.max(values[finite])
        span = max(hi - lo, 1.)
        y = y0 + 57 - (values - lo) / span * 45
        _text(image, f'{hi:.0f}', (68, y0 + 9), .34, MUTED)
        _text(image, f'{lo:.0f}', (68, y0 + 70), .34, MUTED)
        for i in range(1, len(values)):
            if finite[i - 1:i + 1].all() and sampler.times[i] - sampler.times[i - 1] <= sampler.max_gap_s:
                stroke = (STATUS_COLORS['phase_estimated'] if
                          'phase_estimated' in sampler.status[i - 1:i + 1, k] else color)
                cv2.line(image, (int(x[i - 1]), int(y[i - 1])), (int(x[i]), int(y[i])),
                         stroke, 1, cv2.LINE_AA)
    _text(image, f'{start:.1f}s', (65, height - 5), .4, MUTED)
    _text(image, f'{end:.1f}s', (width - 53, height - 5), .4, MUTED)
    return image


def _camera_panel(image, name, source_time, playback_time, source_frame, state):
    image = cv2.resize(image, (776, 436), interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, 0), (776, 43), (19, 19, 19), -1)
    label = 'CAM0 / C920' if name == 'c920' else 'CAM1 / BRIO 101'
    _text(image, label, (15, 28), .72, WHITE, 2)
    _text(image, f'image {source_frame} | t={source_time:.3f}s | dt={1000 * (source_time - playback_time):+.0f}ms',
          (276, 27), .50)
    cv2.rectangle(image, (0, 401), (776, 436), (19, 19, 19), -1)
    for k, label in enumerate(('ROLL', 'RED', 'GREEN')):
        shown = STATUS_LABELS.get(state.status[k], state.status[k])
        _text(image, f'{label}: {shown}', (15 + k * 252, 424), .51,
              STATUS_COLORS[state.status[k]])
    return image


def _draw_details(canvas, state, time_s, duration, initial_roll, geometry):
    x = 590
    _text(canvas, 'ONE SHARED TRAJECTORY', (x, 592), .67, WHITE, 2)
    for k, label in enumerate(('Roll / home', 'Red / start', 'Green / start')):
        y = 637 + k * 62
        value = state.angles[k]
        string = 'unresolved' if not np.isfinite(value) else f'{np.rad2deg(value):+8.2f} deg'
        color = (77, 225, 250) if k == 0 else SHELL_COLORS[k - 1]
        _text(canvas, label, (x, y), .52, color)
        _text(canvas, string, (x + 153, y), .70, WHITE, 2)
        std = state.std_rad[k]
        uncertainty = f' +/- {np.rad2deg(std):.2f} deg' if np.isfinite(std) else ''
        shown = STATUS_LABELS.get(state.status[k], state.status[k])
        _text(canvas, shown + uncertainty, (x + 155, y + 21), .41,
              STATUS_COLORS[state.status[k]])
        if state.status[k] == 'phase_estimated':
            _text(canvas, 'Phase depends on gap prior', (x + 155, y + 36), .35,
                  STATUS_COLORS['phase_estimated'])
        elif np.isfinite(value) and not state.turn_count_valid[k]:
            _text(canvas, 'Full-turn count unverified', (x + 155, y + 36), .35,
                  STATUS_COLORS['predicted'])
    _text(canvas, f'Initial roll: {initial_roll:+.2f} deg from calibrated home', (x, 811), .43, MUTED)
    _text(canvas, f'Radius {geometry["radius_m"] * 1000:.0f} mm | rim-plane gap {geometry["gap_m"] * 1000:.0f} mm',
          (x, 835), .43, MUTED)
    _text(canvas, f't = {time_s:6.3f} / {duration:.3f} s', (1248, 51), .70, WHITE, 2)
    cv2.rectangle(canvas, (16, 873), (1584, 879), (66, 62, 58), -1)
    cv2.rectangle(canvas, (16, 873), (16 + int(1568 * time_s / max(duration, 1e-9)), 879),
                  (174, 221, 135), -1)


def render_offline(report, cfg, output, fps=30, *, progress=print, max_gap_s=.15,
                   transcode=True):
    """Write combined_tracking.mp4 and five PNG stills, bounded in video memory.

    Source images are decoded once in increasing native frame order. A camera
    image is selected by closest corrected timestamp, and its overlay is sampled
    at that image's time, avoiding the erroneous same-frame-index stereo pairing.
    ``output`` may be an existing analysis directory; this function refuses to
    overwrite its own nonempty movie. Returns the paths and replay diagnostics.
    """
    if isinstance(fps, bool) or not np.isfinite(fps) or not 1 <= fps <= 120:
        raise ValueError('fps must be finite and between 1 and 120')
    payload = build_viewer_payload(report)
    sampler = TrajectorySampler(payload['frames'], max_gap_s=max_gap_s)
    cameras, infos, _ = load_rig(cfg)
    from .workflow import calibration_matches
    if report.get('calibration_hashes') and not calibration_matches(cfg, report['calibration_hashes']):
        raise ValueError('Results refer to different intrinsics/stereo')
    out = Path(output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    movie = out / 'combined_tracking.mp4'
    raw_movie = out / 'combined_tracking.rendering.mp4'
    for path in (movie, raw_movie):
        if path.exists() and path.stat().st_size:
            raise ValueError(f'Refusing to overwrite nonempty replay: {path}')
    F, pivot = np.asarray(payload['F']), np.asarray(payload['pivot'])
    geometry = report['geometry']
    per_camera = [[frame for frame in payload['frames'] if frame['camera'] == name]
                  for name in CAMERA_NAMES]
    if any(not frames for frames in per_camera):
        raise ValueError('Replay requires native events from both cameras')
    times = [np.array([f['time_s'] for f in frames]) for frames in per_camera]
    duration = float(sampler.times[-1])
    count = int(np.floor(duration * fps + 1e-8)) + 1
    still_indices = set(np.rint(np.linspace(0, count - 1, min(5, count))).astype(int))
    stills = []
    trace = _timeline(sampler)
    streams, maps = [], []
    cached_frames, cached_indices = [None, None], [-1, -1]
    writer = None
    try:
        for ci, info in enumerate(infos):
            streams.append(SelectedVideo(report['videos'][ci], info['image_size'],
                                          use_manifest=report.get('use_video_manifest', True)))
            maps.append(cv2.initUndistortRectifyMap(info['K'], info['dist'], None,
                                                   info['K'], tuple(info['image_size']), cv2.CV_32FC1))
        writer = cv2.VideoWriter(str(raw_movie), cv2.VideoWriter_fourcc(*'mp4v'), float(fps), SIZE)
        if not writer.isOpened():
            raise OSError('Could not open MPEG-4 replay writer')
        for index in range(count):
            time_s = index / fps
            state = sampler.at(time_s)
            canvas = np.full((SIZE[1], SIZE[0], 3), BACKGROUND, np.uint8)
            _text(canvas, 'CASTER  /  STEREO ORIENTATION', (19, 45), 1.02, WHITE, 2)
            _text(canvas, 'Calibrated geometry | two native camera clocks | one fused estimate',
                  (20, 72), .51, MUTED)
            for ci, name in enumerate(CAMERA_NAMES):
                selected = nearest_frame(times[ci], time_s)
                frame = per_camera[ci][selected]
                native_time, source_index = frame['time_s'], frame['source_frame']
                if cached_indices[ci] != source_index:
                    decoded = streams[ci].read(source_index)
                    cached_frames[ci] = cv2.remap(decoded, *maps[ci], cv2.INTER_LINEAR)
                    cached_indices[ci] = source_index
                source_state = sampler.at(native_time)
                if abs(native_time - time_s) > max_gap_s:
                    panel = np.full((436, 776, 3), PANEL, np.uint8)
                    _text(panel, f'{name.upper()} / no nearby source image', (48, 224), .8, MUTED)
                else:
                    overlaid = draw_mesh(cached_frames[ci].copy(), cameras[ci], F, pivot,
                                         geometry, source_state, thickness=2)
                    panel = _camera_panel(overlaid, name, native_time, time_s, source_index, source_state)
                canvas[93:529, 16 + ci * 792:792 + ci * 792] = panel
            cv2.rectangle(canvas, (16, 547), (570, 858), PANEL, -1)
            simulated = draw_simulation(554, 263, F, pivot, geometry, state)
            canvas[586:849, 16:570] = simulated
            _text(canvas, '3D ESTIMATE / CALIBRATED HOME FRAME', (32, 574), .54, WHITE, 2)
            _text(canvas, 'Meridian: relative phase | amber: gap-prior estimate',
                  (34, 843), .40, MUTED)
            cv2.rectangle(canvas, (578, 547), (1070, 858), PANEL, -1)
            cv2.rectangle(canvas, (1082, 547), (1584, 858), PANEL, -1)
            _text(canvas, 'ANGLE HISTORY / DEG', (1096, 577), .58, WHITE, 2)
            canvas[583:858, 1082:1584] = trace
            cursor = 1082 + 66 + int((502 - 84) * (time_s - sampler.times[0]) /
                                    max(duration - sampler.times[0], 1e-9))
            if 1082 <= cursor < 1584:
                cv2.line(canvas, (cursor, 588), (cursor, 837), WHITE, 1, cv2.LINE_AA)
            _draw_details(canvas, state, time_s, duration, payload['initial_roll_deg'], geometry)
            writer.write(canvas)
            if index in still_indices:
                still = out / f'tracking_{index:05d}.png'
                if not cv2.imwrite(str(still), canvas):
                    raise OSError(f'Could not write {still}')
                stills.append(still)
            if progress and (index == 0 or (index + 1) % 150 == 0):
                progress(f'Rendered {index + 1}/{count} stereo replay frames', flush=True)
    finally:
        for stream in streams:
            stream.close()
        if writer is not None:
            writer.release()
    codec = 'mpeg4'
    executable = shutil.which('ffmpeg') if transcode else None
    if executable:
        process = subprocess.run([executable, '-nostdin', '-hide_banner', '-loglevel', 'error',
                                  '-y', '-i', str(raw_movie), '-c:v', 'libx264', '-preset', 'fast',
                                  '-crf', '21', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                                  '-an', str(movie)], capture_output=True, text=True)
        if process.returncode == 0 and movie.exists() and movie.stat().st_size:
            raw_movie.unlink()
            codec = 'h264'
        else:
            if movie.exists():
                movie.unlink()
            raw_movie.replace(movie)
    else:
        raw_movie.replace(movie)
    result = {'movie': movie, 'stills': stills, 'fps': float(fps), 'frames': count,
              'size': SIZE, 'codec': codec, 'source_duration_s': duration,
              'max_interpolation_gap_s': max_gap_s,
              'camera_frame_policy': 'Nearest corrected native timestamp; overlay at image time.',
              'state_policy': 'Unwrapped linear interpolation only between finite resolved endpoints within max gap.',
              'phase_markings': 'Illustrative reference meridians; shell phase is relative to recording start.',
              'phase_estimated': 'Amber phase retains a prior-dependent phase after a tracking gap; later measured changes do not establish accumulated phase or whole turns.',
              'accuracy_note': 'A rendering of the estimate, not independent accuracy validation.'}
    write_json(out / 'replay.json', result)
    return result
