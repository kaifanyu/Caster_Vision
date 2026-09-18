"""Export a saved fused trajectory as a self-contained, offline 3D viewer.

Only the saved result is read: viewing an old fit never reloads today's camera
calibration or changes the measurements, prediction horizon, or validity flags.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from numbers import Real
import os
from pathlib import Path

import numpy as np


CAMERAS = ('c920', 'brio101')
COMPONENTS = ('roll', 'red', 'green')
STATUSES = frozenset(('home', 'vision', 'predicted', 'phase_estimated', 'unresolved'))
TEMPLATE = Path(__file__).with_name('motion_viewer.html')


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _triple(value, name):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3:
        raise ValueError(f'{name} must contain exactly three components')
    return value


def _optional_nonnegative(value, name):
    if value is None:
        return None
    if isinstance(value, Real) and not isinstance(value, bool) and not math.isfinite(value):
        return None
    number = _number(value, name)
    if number < 0:
        raise ValueError(f'{name} must be nonnegative')
    return number


def build_viewer_payload(report, source_results=None):
    """Validate and simplify the shared state without inventing motion.

    Angles are radians in the original event order. Unknown components become
    JSON null, even if a source report contains a stale numeric placeholder.
    Equal timestamps from different cameras are retained as separate events.
    """
    if not isinstance(report, dict) or report.get('kind') != 'fused_motion':
        raise ValueError('The 3D viewer requires a fused_motion results.json')
    try:
        F = np.asarray(report.get('F'), dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError('F must be a finite proper 3 x 3 rotation matrix') from exc
    if (F.shape != (3, 3) or not np.isfinite(F).all()
            or not np.allclose(F.T @ F, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(F), 1., atol=1e-6, rtol=0)):
        raise ValueError('F must be a finite proper 3 x 3 rotation matrix')
    pivot = [_number(value, 'pivot') for value in _triple(report.get('pivot'), 'pivot')]
    geometry = report.get('geometry')
    if not isinstance(geometry, dict):
        raise ValueError('Saved results must contain their geometry')
    radius = _number(geometry.get('radius_m'), 'radius_m')
    gap = _number(geometry.get('gap_m'), 'gap_m')
    if radius <= 0 or not 0 <= gap < 2 * radius:
        raise ValueError('radius_m must be positive and gap_m must be within [0, 2 * radius_m)')
    sign = geometry.get('red_shell_sign', 1)
    if isinstance(sign, bool) or sign not in (-1, 1):
        raise ValueError('red_shell_sign must be +1 or -1')
    initial_roll = _number(report.get('initial_roll_deg'), 'initial_roll_deg')
    original_frames = report.get('frames')
    if not isinstance(original_frames, list) or not original_frames:
        raise ValueError('Saved results have no native motion events to display')
    frames = []
    previous_time = -math.inf
    camera_previous = {}
    for index, event in enumerate(original_frames):
        prefix = f'Frame {index}'
        if not isinstance(event, dict):
            raise ValueError(f'{prefix} must be an event object')
        time = _number(event.get('time_s'), f'{prefix} time_s')
        camera = event.get('camera')
        source_frame = event.get('source_frame')
        if time < 0 or time < previous_time:
            raise ValueError('Event times must be nonnegative and nondecreasing')
        if camera not in CAMERAS:
            raise ValueError(f'{prefix} camera must be c920 or brio101')
        if (isinstance(source_frame, bool) or not isinstance(source_frame, (int, np.integer))
                or source_frame < 0):
            raise ValueError(f'{prefix} source_frame must be a nonnegative integer')
        if camera in camera_previous:
            previous_index, previous_camera_time = camera_previous[camera]
            if source_frame <= previous_index or time <= previous_camera_time:
                raise ValueError('Each camera must have strictly increasing frame indices and timestamps')
        previous_time = time
        camera_previous[camera] = (source_frame, time)
        status = list(_triple(event.get('status'), f'{prefix} status'))
        if any(not isinstance(value, str) or value not in STATUSES for value in status):
            raise ValueError(f'{prefix} has an unknown component status')
        raw_angles = _triple(event.get('angles'), f'{prefix} angles')
        angles = [None if state == 'unresolved' else _number(raw_angles[i], f'{prefix} {COMPONENTS[i]} angle')
                  for i, state in enumerate(status)]
        std = [_optional_nonnegative(value, f'{prefix} std_rad') for value in
               _triple(event.get('std_rad', [None] * 3), f'{prefix} std_rad')]
        age = [_optional_nonnegative(value, f'{prefix} age_s') for value in
               _triple(event.get('age_s', [None] * 3), f'{prefix} age_s')]
        turns = _triple(event.get('turn_count_valid', [False] * 3), f'{prefix} turn_count_valid')
        if any(not isinstance(value, (bool, np.bool_)) for value in turns):
            raise ValueError(f'{prefix} turn_count_valid must contain booleans')
        frames.append({'time_s': time, 'camera': camera, 'source_frame': int(source_frame),
                       'angles': angles, 'status': status, 'std_rad': std, 'age_s': age,
                       'turn_count_valid': [bool(value) and status[i] not in ('unresolved', 'phase_estimated')
                                            for i, value in enumerate(turns)]})

    source = report.get('measurement_source', 'unknown')
    if not isinstance(source, str):
        raise ValueError('measurement_source must be a string')
    notes = [
        'One shared state combines the two camera streams at their native timestamps.',
        'This is a visualization of the saved estimate, not independent physical accuracy validation.',
        'Predicted states lack a fresh image measurement; unresolved components are hidden. Playback does not fill missing source samples.',
        'phase_estimated retains a prior-dependent starting phase after a tracking gap. Later relative changes may be measured, but accumulated phase and full-turn count are not established by those changes.',
        'Shell spins are relative to the start of this recording; reference markings are illustrative.',
        'A recovered orientation does not establish missed full revolutions when turn_count_valid is false.',
    ]
    if source == 'rotation':
        notes.append('Measurements use the approximate enclosing-sphere rotation model, not a validated metric surface fit.')
    if report.get('status') != 'completed':
        notes.append('The source processing run is not marked completed; inspect its original report.')
    for key in ('orientation_reference', 'uncertainty_scope'):
        if isinstance(report.get(key), str):
            notes.append(report[key])
    warnings = report.get('warnings', [])
    if isinstance(warnings, list):
        notes.extend(str(warning) for warning in warnings)
    original_summary = report.get('summary', {})
    if not isinstance(original_summary, dict):
        raise ValueError('summary must be an object')
    per_camera = original_summary.get('per_camera', {})
    if not isinstance(per_camera, dict):
        raise ValueError('summary.per_camera must be an object')
    return {
        'title': 'Combined caster orientation',
        'source_results': str(source_results) if source_results is not None else None,
        'measurement_source': source, 'F': F.tolist(), 'pivot': pivot,
        'radius_m': radius, 'gap_m': gap, 'red_shell_sign': int(sign),
        'initial_roll_deg': initial_roll, 'frames': frames,
        'summary': {'native_images': len(frames), 'per_camera': deepcopy(per_camera),
                    'coverage': {name: dict(Counter(frame['status'][i] for frame in frames))
                                 for i, name in enumerate(COMPONENTS)}},
        'notes': notes,
    }


def write_motion_viewer(results_path, output_path):
    """Write one portable HTML file from saved results, refusing nonempty output."""
    source = Path(results_path).expanduser().resolve()
    output = Path(output_path).expanduser().absolute()
    if output.suffix.lower() not in ('.html', '.htm'):
        raise ValueError('Choose an .html output file')
    if output.is_symlink() or (output.exists() and (not output.is_file() or output.stat().st_size)):
        raise ValueError('Choose a new or empty HTML output file')
    report = json.loads(source.read_text(encoding='utf-8'))
    payload = build_viewer_payload(report, source_results=source)
    encoded = json.dumps(payload, allow_nan=False, ensure_ascii=False, separators=(',', ':'))
    # A literal </script> must never terminate the embedded data block. Escaping
    # line separators also makes the payload safe in older JavaScript engines.
    encoded = encoded.replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    template = TEMPLATE.read_text(encoding='utf-8')
    if template.count('__MOTION_DATA__') != 1:
        raise ValueError('Motion viewer template must contain one data placeholder')
    html = template.replace('__MOTION_DATA__', encoded)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Recheck after opening so a concurrently written nonempty file is not
    # truncated. Refuse symlinks even if the destination changes after stat().
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o644)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        if os.fstat(stream.fileno()).st_size:
            raise ValueError('Choose a new or empty HTML output file')
        stream.write(html)
    return output
