"""CoTracker native-frame observations with explicit pixel and identity mapping."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import ROOT
from dualcam.config import CAMERA_NAMES, load_intrinsics, write_json
from dualcam.native_observations import save_native_tracks, load_native_tracks
from dualcam.session import SelectedVideo
from dualcam.tracking import masks_for


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def crop_bounds(circle, size, padding=1.10):
    cx, cy, radius = circle; width, height = size
    return (max(0, int(np.floor(cx-radius*padding))), max(0, int(np.floor(cy-radius*padding))),
            min(width, int(np.ceil(cx+radius*padding))), min(height, int(np.ceil(cy+radius*padding))))


def crop_to_native(points, bounds, resized_shape):
    """Invert OpenCV resize's pixel-center convention, then add crop origin."""
    x0, y0, x1, y1 = bounds; height, width = resized_shape
    return (np.asarray(points)+.5)*[(x1-x0)/width, (y1-y0)/height]-.5+[x0, y0]


def native_to_crop(points, bounds, resized_shape):
    x0, y0, x1, y1 = bounds; height, width = resized_shape
    return (np.asarray(points)-[x0, y0]+.5)*[width/(x1-x0), height/(y1-y0)]-.5


def window_ranges(n, length=60, stride=40):
    if length < 3 or stride < 1 or stride >= length:
        raise ValueError('Need 1 <= stride < length and length >= 3')
    start = 0
    while start < n:
        end = min(n, start+length)
        if end-start >= 3: yield start, end
        if end == n: break
        start += stride


def spatial_deduplicate(obs, radius=5., preferred=None):
    """Keep one observation per nearby image feature; never merge identities."""
    groups = np.column_stack([obs[k] for k in ('camera', 'shell', 'frame')])
    _, inverse = np.unique(groups, axis=0, return_inverse=True)
    keep = np.zeros(len(inverse), bool)
    priority = np.asarray(obs['weight']).copy()
    if preferred is not None: priority += 10*np.asarray(preferred)
    for g in range(inverse.max()+1 if len(inverse) else 0):
        rows = np.flatnonzero(inverse == g)
        rows = rows[np.argsort(-priority[rows], kind='stable')]
        chosen = []
        for row in rows:
            if not chosen or np.min(np.linalg.norm(obs['uv'][chosen]-obs['uv'][row], axis=1)) >= radius:
                chosen.append(row); keep[row] = True
    return {key: np.asarray(value)[keep] for key, value in obs.items()}


def exclude_reference_neighborhood(obs, reference, radius=6.):
    """Exclude reference-material neighborhoods for a held-out pixel audit."""
    keep = np.ones(len(obs['uv']), bool)
    for ci in (0, 1):
        for h in (0, 1):
            a = (obs['camera'] == ci) & (obs['shell'] == h)
            b = (reference['camera'] == ci) & (reference['shell'] == h)
            for fi in np.unique(reference['frame'][b]):
                rows = np.flatnonzero(a & (obs['frame'] == fi))
                refs = reference['uv'][b & (reference['frame'] == fi)]
                if len(rows): keep[rows] &= cKDTree(refs).query(obs['uv'][rows])[0] > radius
    return {k: v[keep] for k, v in obs.items()}


def hybrid_observations(klt, learned, distance_px=5.):
    """Prefer established KLT measurements where both see the same feature."""
    learned = {k: v.copy() for k, v in learned.items()}
    learned['track'] += int(klt['track'].max(initial=0))+1
    obs = {k: np.concatenate((klt[k], learned[k])) for k in klt}
    preferred = np.r_[np.ones(len(klt['uv']), bool), np.zeros(len(learned['uv']), bool)]
    return spatial_deduplicate(obs, distance_px, preferred)


def collect_cotracker(cfg, native, output, backend, *, length=60, stride=40,
                      points_per_shell=32, resized_side=640, camera_ids=(0, 1),
                      visibility_threshold=.8, confidence_threshold=.5, start_s=None, end_s=None,
                      progress=print):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(['git', '-C', str(backend.repo), 'rev-parse', 'HEAD'],
                               capture_output=True, text=True, check=True).stdout.strip()
    settings = dict(length=length, stride=stride, points_per_shell=points_per_shell,
                    resized_side=resized_side, camera_ids=list(camera_ids),
                    visibility_threshold=visibility_threshold, confidence_threshold=confidence_threshold,
                    start_s=start_s, end_s=end_s)
    provenance = {'cache_version': 1, 'base_native_provenance': native['provenance'],
                  'tracker': 'cotracker3_scaled_offline', 'revision': revision,
                  'checkpoint_sha256': sha256(backend.checkpoint), 'settings': settings,
                  'collector_sha256': sha256(__file__),
                  'backend_sha256': sha256(Path(__file__).with_name('cotracker_backend.py'))}
    cache = output/'native_tracks.npz'
    if cache.exists():
        return load_native_tracks(cache, expected_provenance=provenance)
    observations = []; reports = []; track_base = 0
    for ci in camera_ids:
        name = CAMERA_NAMES[ci]; info = load_intrinsics(cfg['cameras'][name])
        video = SelectedVideo(native['videos'][ci], info['image_size'])
        mapx, mapy = cv2.initUndistortRectifyMap(info['K'], info['dist'], None, info['K'],
                                               tuple(info['image_size']), cv2.CV_32FC1)
        bounds = crop_bounds(native['circles'][ci], info['image_size'])
        frame_cache = {}
        selected = np.flatnonzero(((native['times'][ci] >= start_s) if start_s is not None else True)
                                  & ((native['times'][ci] <= end_s) if end_s is not None else np.ones(len(native['times'][ci]), bool)))
        if not len(selected): continue
        for wi, (lo, hi) in enumerate(window_ranges(len(selected), length, stride)):
            indices = selected[lo:hi]; prefix = output/f'{name}_{indices[0]:05d}_{indices[-1]:05d}'
            chunk_file = prefix.with_suffix('.npz'); chunk_report = prefix.with_suffix('.json')
            if chunk_file.exists() and chunk_report.exists():
                details = json.loads(chunk_report.read_text())
                if details['provenance'] != provenance: raise ValueError('Stale window cache; choose a new output directory')
                with np.load(chunk_file, allow_pickle=False) as saved:
                    rows = saved['accepted_rows']; observations.extend(rows.tolist())
                    track_base = max(track_base, int(saved['next_track_id']))
                reports.append(details); continue
            frames = []; masks = []; grays = []
            frame_cache = {fi:value for fi,value in frame_cache.items() if fi >= indices[0]}
            for fi in indices:
                if int(fi) not in frame_cache:
                    frame = cv2.remap(video.read(int(fi)), mapx, mapy, cv2.INTER_LINEAR)
                    mask = masks_for(frame, native['circles'][ci], cfg['cameras'][name])
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    x0, y0, x1, y1 = bounds
                    rgb = cv2.cvtColor(cv2.resize(frame[y0:y1, x0:x1], (resized_side, resized_side), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                    frame_cache[int(fi)] = (rgb, mask, gray)
                rgb, mask, gray = frame_cache[int(fi)]
                frames.append(rgb); masks.append(mask); grays.append(gray)
            queries = []; shells = []; anchors = []
            # Queries at actual detected corners, including later sharp context.
            for anchor in sorted(set((0, len(indices)//2, len(indices)-1))):
                for h, label in enumerate(('top', 'bottom')):
                    corners = cv2.goodFeaturesToTrack(grays[anchor], maxCorners=points_per_shell,
                        qualityLevel=.005, minDistance=12., mask=masks[anchor][label].astype(np.uint8)*255,
                        blockSize=5)
                    if corners is None: continue
                    corners = corners[:, 0]
                    transformed = native_to_crop(corners, bounds, (resized_side, resized_side))
                    for point, native_point in zip(transformed, corners):
                        if not (0 <= point[0] < resized_side and 0 <= point[1] < resized_side): continue
                        queries.append([anchor, *point]); shells.append(h); anchors.append(native_point)
            if not queries:
                if progress: progress(f'{name} window {wi}: no query corners', flush=True)
                continue
            if progress: progress(f'{name} CoTracker offline frames {indices[0]}-{indices[-1]}, {len(queries)} points', flush=True)
            prediction = backend.predict(np.asarray(frames), np.asarray(queries, np.float32), backward=True)
            tracks = crop_to_native(prediction['tracks'], bounds, (resized_side, resized_side))
            vis, conf = prediction['visibility'], prediction['confidence']
            raw_ok = (vis >= visibility_threshold) & (conf >= confidence_threshold) & np.isfinite(tracks).all(axis=2)
            accepted = raw_ok.copy(); mask_ok = np.zeros_like(raw_ok)
            for ti in range(len(indices)):
                pixel = np.rint(np.nan_to_num(tracks[ti], nan=-1e6)).astype(int)
                inside = (pixel[:, 0] >= 0) & (pixel[:, 0] < info['image_size'][0]) & (pixel[:, 1] >= 0) & (pixel[:, 1] < info['image_size'][1])
                for h, label in enumerate(('top', 'bottom')):
                    use = np.flatnonzero(inside & (np.asarray(shells) == h))
                    mask_ok[ti, use] = masks[ti][label][pixel[use, 1], pixel[use, 0]]
            accepted &= mask_ok
            for qi, query in enumerate(queries):
                anchor = int(query[0])
                # The detector's query is a raw-image observation, not an
                # independent success of the neural tracker at its own query.
                tracks[anchor, qi] = anchors[qi]
                accepted[anchor, qi] = True
            rows = []
            for qi, h in enumerate(shells):
                keep = np.flatnonzero(accepted[:, qi])
                if len(keep) < 4: continue
                for ti in keep:
                    weight = float(np.clip(vis[ti, qi]*conf[ti, qi], .05, 1.))
                    if ti == int(queries[qi][0]): weight = 1.
                    rows.append([ci, h, int(indices[ti]), track_base+qi, *tracks[ti, qi], weight])
            track_base += len(queries)
            details = {'camera': name, 'source_start': int(indices[0]), 'source_end': int(indices[-1]),
                       'query_count': len(queries), 'accepted_observations': len(rows),
                       'raw_visible_confident': int(raw_ok.sum()), 'mask_rejections': int((raw_ok & ~mask_ok).sum()),
                       'crop_bounds': bounds, 'resized_side': resized_side, 'provenance': provenance,
                       'inference': prediction.get('metrics', {})}
            np.savez_compressed(chunk_file, tracks_native=tracks, visibility=vis, confidence=conf,
                                accepted=accepted, queries_crop=np.asarray(queries), shells=shells,
                                source_indices=indices, accepted_rows=np.asarray(rows).reshape(-1, 7), next_track_id=track_base)
            write_json(chunk_report, details); reports.append(details); observations.extend(rows)
            if progress: progress(f'  accepted {len(rows)} native observations; {details["inference"]}', flush=True)
        video.close()
    if not observations: raise ValueError('No accepted learned observations')
    rows = np.asarray(observations)
    obs = {k: rows[:, i].astype(np.int64) for i, k in enumerate(('camera', 'shell', 'frame', 'track'))}
    obs.update(uv=rows[:, 4:6], weight=rows[:, 6])
    original_count = len(rows); obs = spatial_deduplicate(obs)
    result = {**native, 'observations': obs, 'provenance': provenance,
              'report': {'tracker': 'CoTracker3 offline', 'windows': reports,
                         'before_spatial_deduplication': original_count, 'observations': len(obs['uv']),
                         'native_images': sum(map(len, native['times'])),
                         'measurement_policy': 'Raw model scores + painted-shell masks; query pixels come from the corner detector. Final metric residual/visibility/observability gates determine accepted physical evidence. No window identities are silently stitched.'}}
    save_native_tracks(cache, result)
    write_json(output/'tracking_summary.json', result['report'])
    return result
