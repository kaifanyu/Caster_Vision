"""Read-only visual audit of CoTracker windows on their native source frames.

Colors describe collector candidates, not fitted physical measurements. Raw
frames are undistorted with their original intrinsics, never neural-restored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import BASELINE
from .observations import crop_bounds
from dualcam.config import CAMERA_NAMES, load_intrinsics, write_json
from dualcam.native_observations import load_native_tracks
from dualcam.session import SelectedVideo


SHELL_COLORS = ((80, 90, 255), (100, 235, 105))


def _window_catalog(directory):
    result = {name: [] for name in CAMERA_NAMES}
    for path in sorted(Path(directory).glob('*.json')):
        if not path.with_suffix('.npz').is_file():
            continue
        details = json.loads(path.read_text(encoding='utf-8'))
        name = details.get('camera')
        if name in result and 'source_start' in details and 'source_end' in details:
            result[name].append((path.with_suffix('.npz'), details))
    return result


def _select_window(entries, source_frame):
    """Use the window that places the requested frame closest to its center."""
    choices = [(path, details) for path, details in entries
               if details['source_start'] <= source_frame <= details['source_end']]
    if not choices:
        return None
    return min(choices, key=lambda pair: (
        abs(source_frame - (pair[1]['source_start'] + pair[1]['source_end']) / 2),
        -(pair[1]['source_end'] - pair[1]['source_start']), str(pair[0])))


def _text(frame, text, x, y, scale=.55, color=(228, 233, 238)):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _annotate(frame, saved, local_frame, details, *, label_queries=False):
    canvas = frame.copy()
    xy = saved['tracks_native'][local_frame]
    accepted = saved['accepted'][local_frame].astype(bool)
    shells = saved['shells'].astype(int)
    visibility = saved['visibility'][local_frame]
    threshold = details.get('provenance', {}).get('settings', {}).get('visibility_threshold', .8)
    query_frame = saved['queries_crop'][:, 0].astype(int) == local_frame
    visible_rejected = ~accepted & (visibility >= threshold)
    finite = np.isfinite(xy).all(axis=1)
    inside = finite & (xy[:, 0] >= 0) & (xy[:, 0] < frame.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < frame.shape[0])
    for index in np.flatnonzero(inside & visible_rejected):
        center = tuple(np.rint(xy[index]).astype(int))
        cv2.drawMarker(canvas, center, (145, 145, 145), cv2.MARKER_TILTED_CROSS, 8, 1, cv2.LINE_AA)
    for index in np.flatnonzero(inside & accepted):
        center = tuple(np.rint(xy[index]).astype(int))
        cv2.circle(canvas, center, 4, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 3, SHELL_COLORS[shells[index]], -1, cv2.LINE_AA)
        if query_frame[index]:
            cv2.circle(canvas, center, 7, (250, 250, 250), 1, cv2.LINE_AA)
        if label_queries:
            _text(canvas, f'q{index}', center[0] + 5, center[1] - 5, .35, SHELL_COLORS[shells[index]])
    return canvas, {
        'accepted_candidates': int(accepted.sum()),
        'accepted_red': int((accepted & (shells == 0)).sum()),
        'accepted_green': int((accepted & (shells == 1)).sum()),
        'accepted_query_anchors': int((accepted & query_frame).sum()),
        'rejected_model_visible': int(visible_rejected.sum()),
        'accepted_outside_image': int((accepted & ~inside).sum()),
        'total_queries': len(shells),
    }


def _panel(frame, bounds, title, subtitle, width=800, height=510):
    panel = np.full((height, width, 3), (27, 22, 19), np.uint8)
    _text(panel, title, 18, 28, .64)
    _text(panel, subtitle, 18, 51, .44, (190, 199, 207))
    if frame is not None:
        x0, y0, x1, y1 = bounds
        crop = frame[y0:y1, x0:x1]
        scale = min((width - 24) / crop.shape[1], (height - 73) / crop.shape[0])
        size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
        resized = cv2.resize(crop, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        x, y = (width - size[0]) // 2, 65 + (height - 73 - size[1]) // 2
        panel[y:y + size[1], x:x + size[0]] = resized
    return panel


def render_track_contacts(nativecache, trackingdir, out, *,
                          times_s=(1., 11.2, 19.2, 21.2), label_queries=False,
                          additional_brio_offset_s=0.):
    """Save an overlay contact sheet and raw/overlay pairs at selected times.

    ``nativecache`` is the original conventional native_tracks.npz (or loaded
    dictionary). Source times retain its data-relative recording clock; an
    optional extra Brio offset shifts only frame selection and displayed time.
    Missing in-progress windows are shown explicitly and never substituted by
    distant images. Call again after collection finishes to fill missing panels.
    """
    native = load_native_tracks(nativecache) if isinstance(nativecache, (str, Path)) else nativecache
    cfg = native['provenance']['configuration']
    targets = np.asarray(times_s, float)
    if targets.ndim != 1 or not len(targets) or not np.isfinite(targets).all() or not np.isfinite(additional_brio_offset_s):
        raise ValueError('Need finite target times and Brio offset')
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    catalog = _window_catalog(trackingdir)
    panels, records = {}, []
    for ci, name in enumerate(CAMERA_NAMES):
        info = load_intrinsics(cfg['cameras'][name])
        mapx, mapy = cv2.initUndistortRectifyMap(info['K'], info['dist'], None, info['K'],
                                               tuple(info['image_size']), cv2.CV_32FC1)
        corrected_times = np.asarray(native['times'][ci]) + additional_brio_offset_s * (ci == 1)
        indices = [int(np.argmin(np.abs(corrected_times - target))) for target in targets]
        video = SelectedVideo(native['videos'][ci], info['image_size'])
        frames = {}
        try:
            for source_frame in sorted(set(indices)):
                frames[source_frame] = cv2.remap(video.read(source_frame), mapx, mapy, cv2.INTER_LINEAR)
        finally:
            video.close()
        for ti, (target, source_frame) in enumerate(zip(targets, indices)):
            frame = frames[source_frame]
            actual = float(corrected_times[source_frame])
            record = {'camera': name, 'requested_time_s': float(target), 'source_frame': source_frame,
                      'source_time_s': float(native['times'][ci][source_frame]), 'display_time_s': actual,
                      'selection_error_s': actual - float(target)}
            selected = _select_window(catalog[name], source_frame)
            bounds = crop_bounds(native['circles'][ci], info['image_size'])
            heading = f'{name.upper()}   t={actual:.3f}s   source frame {source_frame}'
            raw = _panel(frame, bounds, heading, 'Raw recorded image, undistorted with original intrinsics')
            if selected is None:
                annotated = _panel(frame, bounds, heading, 'NO TRACKING WINDOW AVAILABLE FOR THIS FRAME')
                record.update(status='missing_window', window=None)
            else:
                path, details = selected
                with np.load(path, allow_pickle=False) as saved:
                    locations = np.flatnonzero(saved['source_indices'] == source_frame)
                    if len(locations) != 1:
                        raise ValueError(f'{path}: expected exactly one matching native source index')
                    overlay, counts = _annotate(frame, saved, int(locations[0]), details, label_queries=label_queries)
                record.update(status='rendered', window=str(path.resolve()), counts=counts)
                record['window_source_range'] = [details['source_start'], details['source_end']]
                subtitle = (f'Candidates R/G={counts["accepted_red"]}/{counts["accepted_green"]}; '
                            f'query anchors={counts["accepted_query_anchors"]}; rejected-visible={counts["rejected_model_visible"]}')
                annotated = _panel(overlay, bounds, heading, subtitle)
            panels[(ti, ci)] = (raw, annotated)
            records.append(record)
    banner = np.full((90, 1600, 3), (27, 22, 19), np.uint8)
    _text(banner, 'CoTracker image audit | red / green = collector candidates; gray x = rejected model-visible', 20, 28, .65)
    _text(banner, 'White ring = detected query anchor. Candidate colors do not establish physical evidence or material identity.', 20, 53, .56)
    _text(banner, 'Native recorded images, calibrated undistortion. No learned deblurring. Frame times use the native data clock.', 20, 76, .50)
    contact_rows = [banner]
    artifacts = []
    for ti, target in enumerate(targets):
        contact_rows.append(np.hstack([panels[(ti, ci)][1] for ci in range(2)]))
        pair = np.vstack([banner, *[np.hstack(panels[(ti, ci)]) for ci in range(2)]])
        name = f'tracks_{ti:02d}_{target:06.3f}s.jpg'
        if not cv2.imwrite(str(output / name), pair, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise OSError(f'Cannot write {output / name}')
        artifacts.append(str((output / name).resolve()))
    contact_path = output / 'track_contacts.jpg'
    if not cv2.imwrite(str(contact_path), np.vstack(contact_rows), [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError(f'Cannot write {contact_path}')
    report = {'contact_sheet': str(contact_path.resolve()), 'raw_overlay_pairs': artifacts,
              'requested_times_s': targets.tolist(), 'additional_brio_offset_s': float(additional_brio_offset_s),
              'candidate_policy': 'Window candidates before spatial deduplication and physical metric rejection; query anchors are detector observations.',
              'records': records}
    write_json(output / 'track_preview.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-cache', type=Path, default=BASELINE / 'out/joint_native_cache/native_tracks.npz')
    parser.add_argument('--tracking-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--times', nargs='+', type=float, default=[1., 11.2, 19.2, 21.2])
    parser.add_argument('--label-queries', action='store_true')
    parser.add_argument('--brio-offset', type=float, default=0.)
    args = parser.parse_args()
    result = render_track_contacts(args.native_cache, args.tracking_dir, args.output, times_s=args.times,
                                   label_queries=args.label_queries, additional_brio_offset_s=args.brio_offset)
    print(json.dumps(result, indent=2))
