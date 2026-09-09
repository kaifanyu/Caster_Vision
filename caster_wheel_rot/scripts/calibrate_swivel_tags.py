#!/usr/bin/env python
"""Calibrate multiple fork tags from overlapping observations in a slow sweep.

Writes a separate complete config by default. Only the explicitly requested
IDs are used. Re-run after moving any marker; this is not per-run auto learning.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from common.rotation import Rz
from swivel.geometry import SwivelGeometry
from swivel.tag import ArucoTagTracker, _DetectorBackend, aruco_dictionary, detector_parameters


def robust_rotation_mean(matrices, *, min_samples=8, max_spread_deg=8.):
    matrices = np.asarray(matrices, dtype=float)
    if len(matrices) < min_samples:
        raise ValueError(f"only {len(matrices)} overlapping poses; need {min_samples}")
    rotations = Rotation.from_matrix(matrices)
    # Start from a medoid so mirror branches do not skew the initial mean.
    distances = np.array([(r.inv() * rotations).magnitude() for r in rotations])
    center = rotations[int(np.argmin(np.median(distances, axis=1)))]
    keep = np.ones(len(matrices), bool)
    for _ in range(4):
        errors = np.rad2deg((center.inv()*rotations).magnitude())
        keep = errors <= max_spread_deg
        if keep.sum() < min_samples:
            raise ValueError("marker mount/pose observations do not give enough consistent rotations")
        center = rotations[keep].mean()
    errors = np.rad2deg((center.inv()*rotations).magnitude())
    keep = errors <= max_spread_deg
    if keep.sum() < min_samples or keep.mean() < .5:
        raise ValueError("fewer than half of the marker observations agree; inspect rigid mounting")
    return center.as_matrix(), {"samples":len(matrices), "retained":int(keep.sum()),
        "median_error_deg":float(np.median(errors[keep])),
        "p95_error_deg":float(np.percentile(errors[keep],95)),
        "rejected":int((~keep).sum())}


def observe_clip(path, cfg, ids):
    g = SwivelGeometry.from_mapping(cfg)
    K, dist = np.array(cfg['camera']['K']), np.array(cfg['camera']['dist'])
    marker = cfg['swivel']['marker']
    backend = _DetectorBackend(aruco_dictionary(marker['dictionary']), detector_parameters())
    trackers = {i:ArucoTagTracker(K, marker['size_m'], marker_id=i,
        R_car_from_camera=g.R_car_from_camera,
        expected_normal_camera=g.R_camera_from_car @ np.array([0.,0.,1.]),
        max_reprojection_error_px=marker.get('max_reprojection_error_px',2.)) for i in ids}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open {path}")
    maps = None
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if maps is None:
                maps = cv2.initUndistortRectifyMap(K,dist,None,K,(image.shape[1],image.shape[0]),cv2.CV_32FC1)
            image = cv2.remap(image,*maps,cv2.INTER_LINEAR)
            corners, found, _ = backend.detect(image)
            lookup = {int(i):c for i,c in zip([] if found is None else found.ravel(), corners)}
            yield {i:tr.track_corners(lookup.get(i)) for i,tr in trackers.items()}
    finally:
        cap.release()


def calibrate(cfg, clips, ids, *, zero_clip=None, min_samples=8, max_spread_deg=8.):
    primary = int(cfg['swivel']['marker'].get('id',0))
    if primary not in ids or len(set(ids)) != len(ids):
        raise ValueError("IDs must be unique and include the primary configured marker")
    zero = cfg['swivel']['marker'].get('psi0_rad')
    if zero is None:
        raise ValueError("calibrate psi-zero first")
    primary_zero = Rz(float(zero))
    report = {'primary_id':primary, 'clips':[str(Path(p).resolve()) for p in clips], 'markers':{}}
    if zero_clip is not None:
        poses = [o[primary].R_tag_to_car for o in observe_clip(zero_clip,cfg,[primary]) if o[primary].valid]
        primary_zero, zero_report = robust_rotation_mean(poses,min_samples=min_samples,max_spread_deg=2.)
        report['primary_zero'] = {'clip':str(Path(zero_clip).resolve()), **zero_report}
    relative = {i:[] for i in ids if i != primary}
    for clip in clips:
        for observations in observe_clip(clip,cfg,ids):
            a = observations[primary]
            if not a.valid:
                continue
            for i in relative:
                b = observations[i]
                if b.valid:
                    relative[i].append(a.R_tag_to_camera.T @ b.R_tag_to_camera)
    zeros = {str(primary):primary_zero.tolist()}
    for i, matrices in relative.items():
        rel, stats = robust_rotation_mean(matrices,min_samples=min_samples,max_spread_deg=max_spread_deg)
        zeros[str(i)] = (primary_zero @ rel).tolist()
        report['markers'][str(i)] = stats
    report['zero_rotations'] = zeros
    return zeros, report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'config.yaml')
    p.add_argument('--clip',type=Path,action='append',required=True)
    p.add_argument('--zero-clip',type=Path)
    p.add_argument('--ids',type=int,nargs='+',required=True)
    p.add_argument('--output-config',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--min-samples',type=int,default=8)
    p.add_argument('--max-spread-deg',type=float,default=8.)
    args = p.parse_args(argv)
    cv2.setNumThreads(2)
    cfg = yaml.safe_load(args.config.read_text())
    zeros, report = calibrate(cfg,args.clip,args.ids,zero_clip=args.zero_clip,
                             min_samples=args.min_samples,max_spread_deg=args.max_spread_deg)
    cfg['swivel']['marker']['zero_rotations'] = zeros
    cfg['swivel']['marker']['calibration_report'] = str(args.report.resolve())
    # Preserve path meaning when the complete config is saved elsewhere.
    for key in ['swivel_clip','ball_clip','car_track']:
        if cfg.get('input',{}).get(key):
            cfg['input'][key] = str((args.config.resolve().parent/cfg['input'][key]).resolve())
    for path, content in [(args.output_config,yaml.safe_dump(cfg,sort_keys=False)),
                          (args.report,json.dumps(report,indent=2)+'\n')]:
        path.parent.mkdir(parents=True,exist_ok=True)
        temp = path.with_suffix(path.suffix+'.tmp')
        temp.write_text(content,encoding='utf-8'); temp.replace(path)
    print(json.dumps(report,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
