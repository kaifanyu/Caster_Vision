"""Tabular exports with distinct local-rate and accumulated-phase validity.

CSV empty fields mean unavailable. ``strict`` rates and vector ``valid`` flags
require measurement-supported (``vision``) rates; a short predicted interval
never becomes a strict measurement. A valid local rate does not validate the
accumulated phase or its integer turn count.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path


_COMPONENTS = ('roll', 'red_spin', 'green_spin')
_RATE_STATUSES = {'vision', 'predicted', 'unresolved'}


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _fields():
    fields = ['time_s', 'timestamp_s', 'receive_time_s', 'camera', 'source_frame',
              'rate_method', 'rate_grid_spacing_s']
    for component in _COMPONENTS:
        fields.extend(component+'_'+name for name in (
            'angular_velocity_rad_s', 'angular_velocity_deg_s',
            'strict_angular_velocity_rad_s', 'strict_angular_velocity_deg_s',
            'rate_status', 'rate_measured', 'phase_status', 'turn_count_valid',
            'rate_support_start_s', 'rate_support_end_s', 'rate_source',
            'rate_relative_null_projection'))
    for shell in ('red', 'green'):
        fields.extend(shell+'_angular_velocity_rig_'+axis+'_rad_s' for axis in 'xyz')
        fields.extend(shell+'_strict_angular_velocity_rig_'+axis+'_rad_s' for axis in 'xyz')
        fields.extend((shell+'_angular_velocity_rig_status', shell+'_angular_velocity_rig_valid'))
    return fields


def _triple(frame, key, default):
    value = frame.get(key, default)
    if value is None or len(value) != 3:
        raise ValueError(f'{key} must contain roll, red spin and green spin entries')
    return value


def export_rates_csv(output_path, report):
    """Write annotated native-event rates and return the output ``Path``.

    Call after :func:`blurtrack.rate_analysis.annotate_rates`. ``time_s`` and
    support bounds use the common relative trajectory clock. ``timestamp_s``
    adds the report's ``timestamp_origin_s`` when present; this is the source
    recording clock, not a claim of UTC or exposure-midpoint timing. Component
    rates are supplied in rad/s and deg/s; rig vectors use rad/s in the frame
    documented by ``report['rate_analysis']['rig_vector_frame']``.

    Existing poses, phase flags and turn-count validity are not inferred from
    rate support. A missing turn-count flag remains blank, never assumed true.
    """
    path = Path(output_path)
    metadata = report.get('rate_analysis', {})
    origin = _finite(report.get('timestamp_origin_s'))
    rows = []
    for frame in report.get('frames', []):
        statuses = _triple(frame, 'rate_status', None)
        if any(status not in _RATE_STATUSES for status in statuses):
            raise ValueError('rate_status must use vision, predicted or unresolved; run annotate_rates first')
        rates = _triple(frame, 'angular_velocity', [None]*3)
        phase = _triple(frame, 'phase_status', frame.get('status', ['unresolved']*3))
        turns = _triple(frame, 'turn_count_valid', [None]*3)
        supports = _triple(frame, 'rate_support_interval_s', [None]*3)
        sources = _triple(frame, 'rate_source', [None]*3)
        projections = _triple(frame, 'rate_relative_null_projection', [None]*3)
        time = _finite(frame.get('time_s'))
        row = {'time_s': time, 'timestamp_s': origin+time if origin is not None and time is not None else None,
               'receive_time_s': _finite(frame.get('receive_time_s')),
               'camera': frame.get('camera'), 'source_frame': frame.get('source_frame'),
               'rate_method': metadata.get('method'),
               'rate_grid_spacing_s': _finite(metadata.get('grid_spacing_s'))}
        measured = []
        for i, component in enumerate(_COMPONENTS):
            rate = _finite(rates[i]) if statuses[i] != 'unresolved' else None
            is_measured = statuses[i] == 'vision' and rate is not None
            measured.append(is_measured)
            strict = rate if is_measured else None
            support = supports[i] if is_measured else None
            if support is not None and len(support) != 2:
                raise ValueError('Each rate_support_interval_s entry must be null or [start,end]')
            values = {
                'angular_velocity_rad_s': rate,
                'angular_velocity_deg_s': math.degrees(rate) if rate is not None else None,
                'strict_angular_velocity_rad_s': strict,
                'strict_angular_velocity_deg_s': math.degrees(strict) if strict is not None else None,
                'rate_status': statuses[i], 'rate_measured': is_measured,
                'phase_status': phase[i], 'turn_count_valid': turns[i],
                'rate_support_start_s': _finite(support[0]) if support is not None else None,
                'rate_support_end_s': _finite(support[1]) if support is not None else None,
                'rate_source': sources[i], 'rate_relative_null_projection': _finite(projections[i])}
            row.update({component+'_'+key: value for key, value in values.items()})
        for i, shell in ((1, 'red'), (2, 'green')):
            vector_status = frame.get('angular_velocity_rig_'+shell+'_status', 'unresolved')
            if vector_status not in _RATE_STATUSES:
                raise ValueError('Rig rate status must use vision, predicted or unresolved')
            vector = frame.get('angular_velocity_rig_'+shell)
            if vector is not None and len(vector) != 3:
                raise ValueError('Rig angular velocity must be null or a three-component vector')
            vector = [_finite(v) for v in vector] if vector is not None else [None]*3
            available = (vector_status != 'unresolved' and statuses[0] != 'unresolved'
                         and statuses[i] != 'unresolved' and _finite(rates[0]) is not None
                         and _finite(rates[i]) is not None and all(v is not None for v in vector))
            vector_valid = available and vector_status == 'vision' and measured[0] and measured[i]
            # Inconsistent upstream vector/component labels cannot create a
            # falsely strict vector. Export the conservative combined status.
            vector_status = 'vision' if vector_valid else 'predicted' if available else 'unresolved'
            for axis, value in zip('xyz', vector):
                row[shell+'_angular_velocity_rig_'+axis+'_rad_s'] = value if available else None
                row[shell+'_strict_angular_velocity_rig_'+axis+'_rad_s'] = value if vector_valid else None
            row[shell+'_angular_velocity_rig_status'] = vector_status
            row[shell+'_angular_velocity_rig_valid'] = bool(vector_valid)
        rows.append(row)
    # Validate before opening the file so a malformed report cannot replace a
    # previously completed export with a partly written one.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=_fields())
        writer.writeheader()
        writer.writerows(rows)
    return path
def write_artifacts(output, report, *, render=False):
    """Export a saved fit, normalizing JSON nulls for the legacy CSV writer."""
    import copy
    import tempfile
    import numpy as np
    from pathlib import Path
    from dualcam.config import write_json
    from dualcam.offline_finalization import write_offline_csvs
    from dualcam.visualization import write_motion_viewer

    output = Path(output)
    report['kind'] = 'fused_motion'
    write_json(output/'results.json', report)
    export_rates_csv(output/'angular_rates.csv', report)
    frames = copy.deepcopy(report['frames'])
    for frame in frames:
        for key in ('angles', 'angular_velocity', 'strict_angles', 'strict_angular_velocity', 'std_rad'):
            frame[key] = np.asarray(frame[key], dtype=float)
    write_offline_csvs(output, frames)
    with tempfile.TemporaryDirectory(prefix='viewer_',dir=output) as temporary:
        viewer = Path(temporary)/'orientation_3d.html'
        write_motion_viewer(output/'results.json', viewer)
        viewer.replace(output/'orientation_3d.html')
    if render:
        from dualcam.offline_render import render_offline
        render_offline(report, report['config'], output/'replay')
