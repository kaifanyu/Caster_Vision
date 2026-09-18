"""Relative receive-to-image delay from the SAME irregular light pulses.

This estimates a constant relative latency, not hardware exposure synchronization.
Signals must start and end dark and contain the complete same pulse sequence.
"""
from __future__ import annotations

import cv2
import numpy as np

from .session import SelectedVideo, read_timestamps


def brightness_signal(directory, name, roi, image_size=None):
    times = read_timestamps(directory / f'{name}_timestamps.csv')
    video = SelectedVideo(directory / f'{name}.avi', image_size)
    x, y, w, h = map(int, roi)
    values = []
    try:
        for i in range(len(times)):
            frame = video.read(i)
            if x < 0 or y < 0 or w < 3 or h < 3 or x+w > frame.shape[1] or y+h > frame.shape[0]:
                raise ValueError(f'{name}: ROI must be a rectangle inside the raw image, at least 3x3 pixels')
            values.append(float(cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY).mean()))
    finally:
        video.close()
    return times, np.asarray(values)


def light_edges(times, brightness):
    times, values = np.asarray(times, float), np.asarray(brightness, float)
    if (times.ndim != 1 or values.shape != times.shape or len(times) < 20
            or not np.isfinite(times).all() or not np.isfinite(values).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('Need matching finite brightness samples and increasing timestamps')
    low, high = np.percentile(values, [10, 90])
    if high-low < 20:
        raise ValueError('Light contrast is too small; select a tighter ROI around the shared illuminated patch')
    normalized = (values-low)/(high-low)
    if np.median(normalized[:5]) > .25 or np.median(normalized[-5:]) > .25:
        raise ValueError('Start and end with the light OFF in both cameras')
    # Hysteresis suppresses noise; report the bracket around the 50% crossing.
    edges = []
    state, last_edge_index = False, 0
    for i in range(1, len(times)):
        crossed = normalized[i] >= .75 if not state else normalized[i] <= .25
        if not crossed:
            continue
        sign = 1 if not state else -1
        candidates = [j for j in range(last_edge_index+1, i+1)
                      if (normalized[j-1] < .5 <= normalized[j] if sign == 1
                          else normalized[j-1] > .5 >= normalized[j])]
        if not candidates:
            raise ValueError('Ambiguous light transition; use a clean, steady light with distinct ON/OFF holds')
        j = candidates[-1]
        fraction = (.5-normalized[j-1])/(normalized[j]-normalized[j-1])
        edges.append({'time_s': float(times[j-1]+fraction*(times[j]-times[j-1])),
                      'direction': sign, 'bracket_s': [float(times[j-1]), float(times[j])]})
        state = not state
        last_edge_index = i
    if state or len(edges) < 12:
        raise ValueError('Need at least six complete ON/OFF pulses, with light OFF before and after')
    edge_times = np.array([e['time_s'] for e in edges])
    if (edge_times[0]-times[0] < .5 or times[-1]-edge_times[-1] < .5
            or np.min(np.diff(edge_times)) < .2):
        raise ValueError('Hold each light state for at least 0.3 s and leave >=0.5 s dark at both ends')
    if np.max([e['bracket_s'][1]-e['bracket_s'][0] for e in edges]) > .12:
        raise ValueError('Dropped frames obscure a light transition; repeat the timing recording')
    return {'edges': edges, 'low_brightness': float(low), 'high_brightness': float(high),
            'median_frame_interval_ms': float(1000*np.median(np.diff(times)))}


def estimate_offset(first_times, first_brightness, second_times, second_brightness):
    signals = [light_edges(first_times, first_brightness), light_edges(second_times, second_brightness)]
    left, right = [s['edges'] for s in signals]
    if len(left) != len(right) or [e['direction'] for e in left] != [e['direction'] for e in right]:
        raise ValueError('Cameras detected different pulse sequences; check ROIs and repeat with slower, complete pulses')
    a, b = [np.array([e['time_s'] for e in edges]) for edges in (left, right)]
    if np.ptp(a) < 8:
        raise ValueError('Spread the pulses over at least eight seconds to check relative delay stability')
    intervals = np.diff(a)
    if np.std(intervals) < .06 or max(np.std(np.diff(a[::2])), np.std(np.diff(a[1::2]))) < .06:
        raise ValueError('Use irregular ON/OFF durations, not a periodic blinking signal')
    offsets = a-b  # aligned_brio = recorded_brio + offset
    offset = float(np.median(offsets))
    if abs(offset) > .5:
        raise ValueError('Measured offset exceeds 0.5 seconds; inspect buffering and pulse correspondence')
    residual = offsets-offset
    drift = float(np.polyfit(a-a.mean(), offsets, 1)[0])
    drift_span = abs(drift)*np.ptp(a)
    p95 = float(np.percentile(abs(residual), 95))
    direction_bias = abs(float(np.median(offsets[::2])-np.median(offsets[1::2])))
    # Generous consistency checks, not an accuracy guarantee for moving geometry.
    reasons = []
    if p95 > .025:
        reasons.append('Pulse delay varies by more than 25 ms (95th percentile); a constant offset is inadequate')
    if drift_span > .020:
        reasons.append('Estimated delay changes by more than 20 ms across the recording; repeat and inspect drift/buffering')
    if direction_bias > .020:
        reasons.append('ON and OFF edges disagree by more than 20 ms; inspect exposure, clipping and the selected light patch')
    brackets = [[l['bracket_s'][0]-r['bracket_s'][1], l['bracket_s'][1]-r['bracket_s'][0]]
                for l, r in zip(left, right)]
    return {'success': not reasons, 'reasons': reasons,
            'brio_offset_s': offset, 'convention': 'aligned_brio_time = recorded_brio_time + brio_offset_s',
            'matched_edges': len(a), 'pulse_span_s': float(np.ptp(a)),
            'per_edge_offset_s': offsets.tolist(), 'per_edge_offset_bracket_s': brackets,
            'residual_p95_ms': p95*1000, 'offset_mad_ms': float(np.median(abs(residual))*1000),
            'estimated_delay_change_ms': float(drift_span*1000),
            'on_off_offset_difference_ms': direction_bias*1000,
            'signals': signals,
            'limitation': 'Single ROI and constant relative image/receive latency only. '
                          'Frame sampling, exposure integration, rolling shutter, drift and buffering remain. '
                          'Repeat in an independent recording before marking timing verified.'}
