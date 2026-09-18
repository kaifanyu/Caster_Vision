"""Frozen measured shell appearance for exposure-integrated image fitting.

Each camera gets its own atlas because its color response is different. Atlas
pixels are measured samples, not generated completions. The appearance and
material gauge are conditional on the supplied trajectory and lighting.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import cv2
import numpy as np

from . import BASELINE
from .observations import sha256
from .observed_occlusion import observed_shell_mask, occlusion_metadata
from dualcam.config import CAMERA_NAMES, load_config, load_rig, write_json
from dualcam.model import rotation_x, rotation_z, world_points, project
from dualcam.native_observations import load_native_tracks
from dualcam.session import SelectedVideo
from dualcam.workflow import load_axes, calibration_hashes


def srgb_to_linear(rgb):
    """Approximate transfer inversion; camera ISP response is not calibrated."""
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0., 1.)
    return np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - .055)


def material_grid(height=192, width=512, sign=1):
    """Cell-centered phi [0,2pi), theta [0,pi/2], including signed local z."""
    if height < 2 or width < 4 or sign not in (-1, 1):
        raise ValueError('Invalid hemisphere atlas dimensions or sign')
    phi, theta = np.meshgrid((np.arange(width) + .5) * 2 * np.pi / width,
                             (np.arange(height) + .5) * np.pi / (2 * height))
    return np.stack((np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta),
                     sign*np.cos(theta)), axis=-1)


def atlas_coordinates(normals, height, width, sign=1):
    """Continuous array coordinates; horizontal interpolation must wrap."""
    n = np.asarray(normals)
    phi = np.mod(np.arctan2(n[..., 1], n[..., 0]), 2*np.pi)
    theta = np.arccos(np.clip(sign*n[..., 2], 0, 1))
    return np.stack((phi*width/(2*np.pi)-.5, theta*height/(np.pi/2)-.5), axis=-1)


def visible_projection(camera, F, pivot, angles, shell, normals, geometry,
                       min_incidence=.30, rim_margin_deg=5.):
    """Project own front surface and reject nearer hits on the other shell."""
    shape = normals.shape[:-1]; points = normals.reshape(-1, 3)
    radius, gap = geometry['radius_m'], geometry['gap_m']
    red_sign = geometry.get('red_shell_sign', 1); sign = red_sign*(1-2*shell)
    B = np.asarray(F) @ rotation_x(float(angles[0]))
    S = B @ rotation_z(float(angles[1+shell]))
    xyz = world_points(F, pivot, angles, shell, points, radius, gap, red_sign)
    origin = -np.asarray(camera['R']).T @ np.asarray(camera['t'])
    towards = origin - xyz; distance = np.linalg.norm(towards, axis=1)
    incidence = np.sum((points @ S.T) * towards / distance[:, None], axis=1)
    valid = (incidence >= min_incidence) & (sign*points[:, 2] > np.sin(np.deg2rad(rim_margin_deg)))
    # A signed sphere's front root can lie outside its hemisphere. Test both
    # roots before declaring it an occluder; roots behind the target do not count.
    other_sign = -sign
    other_center = np.asarray(pivot) + B @ np.array([0., 0., other_sign*gap/2])
    direction = -towards / distance[:, None]; offset = origin - other_center
    b = direction @ offset; disc = b*b - (offset@offset - radius*radius)
    root = np.sqrt(np.maximum(disc, 0.))
    for hit_distance in (-b-root, -b+root):
        hit = origin + hit_distance[:, None]*direction - other_center
        other_local_z = hit @ B[:, 2]
        occulted = (disc >= 0) & (hit_distance > 0) & (hit_distance < distance-1e-5) & (other_sign*other_local_z >= 0)
        valid &= ~occulted
    depth = (xyz @ np.asarray(camera['R']).T + camera['t'])[:, 2]
    valid &= depth > 0
    return project(camera, xyz).reshape(*shape, 2), valid.reshape(shape), incidence.reshape(shape)


def robust_fuse(samples, weights, *, min_count=2, color_tolerance=.18):
    """Weighted median reference followed by bounded weighted averaging.

    No spatial inpainting: unobserved cells remain invalid. Large cross-frame
    color disagreement is removed before averaging to suppress occlusion and
    mismapped edge contamination, although calibration bias remains possible.
    """
    values = np.asarray(samples, np.float32); weights = np.asarray(weights, np.float32)
    if values.shape[:-1] != weights.shape or values.shape[-1] != 3:
        raise ValueError('Expected samples [N,H,W,3], weights [N,H,W]')
    weights = np.where(np.isfinite(values).all(axis=-1) & np.isfinite(weights) & (weights>0), weights, 0)
    values = np.nan_to_num(values)
    median = np.zeros(values.shape[1:], np.float32)
    half = weights.sum(axis=0)/2
    for channel in range(3):
        order = np.argsort(values[..., channel], axis=0)
        sorted_values = np.take_along_axis(values[..., channel], order, axis=0)
        sorted_weights = np.take_along_axis(weights, order, axis=0)
        index = np.argmax(np.cumsum(sorted_weights, axis=0) >= half[None], axis=0)
        median[..., channel] = np.take_along_axis(sorted_values, index[None], axis=0)[0]
    accepted = (weights > 0) & (np.linalg.norm(values - median[None], axis=-1) <= color_tolerance)
    effective = weights*accepted; total = effective.sum(axis=0)
    count = accepted.sum(axis=0).astype(np.uint16)
    rgb = (values*effective[..., None]).sum(axis=0)/np.maximum(total[..., None], 1e-12)
    valid = count >= min_count
    rgb[~valid] = 0
    return {'rgb':rgb.astype(np.float32), 'valid':valid, 'weight':total, 'count':count,
            'spread':np.sqrt(((values-rgb[None])**2*effective[..., None]).sum(axis=(0,3))/np.maximum(total, 1e-12))}


def reference_candidates(frames, camera_name, shell, cutoff_s, exposure_s, radius_px,
                         max_blur_px=6., count=48, excluded_intervals=((7.7,8.3),)):
    """Slow, supported native images distributed across time and material view."""
    chosen = [f for f in frames if f['camera'] == camera_name and f['time_s'] < cutoff_s
              and f['status'][0] in ('home', 'vision') and f['status'][1+shell] in ('home', 'vision')
              and not any(lo<=f['time_s']<=hi for lo,hi in excluded_intervals)]
    if not chosen: raise ValueError('No supported reference frames before cutoff')
    times = np.asarray([f['time_s'] for f in chosen]); angles = np.asarray([f['angles'] for f in chosen])
    velocity = np.gradient(angles, times, axis=0)
    predicted = (abs(velocity[:, 0]) + abs(velocity[:, 1+shell]))*exposure_s*radius_px
    eligible = np.flatnonzero(predicted <= max_blur_px)
    if not len(eligible): raise ValueError('No sharp-enough reference candidates under motion bound')
    # Choose one low-motion frame per equal temporal bin, then fill with low
    # motion frames while maintaining a minimum time separation.
    selected = []
    for lo, hi in zip(np.linspace(0, cutoff_s, count+1)[:-1], np.linspace(0, cutoff_s, count+1)[1:]):
        ids = eligible[(times[eligible]>=lo) & (times[eligible]<hi)]
        if len(ids): selected.append(int(ids[np.argmin(predicted[ids])]))
    for i in eligible[np.argsort(predicted[eligible])]:
        if len(selected) >= count: break
        if not selected or np.min(abs(times[selected]-times[i])) > .10: selected.append(int(i))
    return [{**chosen[i], 'predicted_blur_px':float(predicted[i])} for i in sorted(set(selected))]


def _sample_reference(image_bgr, camera, F, pivot, q, shell, grid, geometry, predicted_blur,
                      camera_name=None):
    uv, valid, incidence = visible_projection(camera,F,pivot,q,shell,grid,geometry)
    height,width = image_bgr.shape[:2]
    valid &= (uv[...,0]>=2)&(uv[...,0]<width-3)&(uv[...,1]>=2)&(uv[...,1]<height-3)
    if camera_name is not None:valid &= observed_shell_mask(camera_name,uv)
    mapped = cv2.remap(image_bgr, uv[...,0].astype(np.float32), uv[...,1].astype(np.float32), cv2.INTER_LINEAR)
    rgb = mapped[...,::-1].astype(np.float32)/255.
    hsv = cv2.cvtColor(mapped,cv2.COLOR_BGR2HSV)
    # White substrate and correct paint are retained. Dark yoke, very saturated
    # wrong-shell paint and channel clipping do not enter the texture estimate.
    wrong_paint = ((hsv[...,0]>=30)&(hsv[...,0]<=105)) if shell==0 else ((hsv[...,0]<=20)|(hsv[...,0]>=160))
    valid &= (rgb.max(axis=-1)>.28)&(rgb.max(axis=-1)<253/255.)&~(wrong_paint&(hsv[...,1]>80))
    gray = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray,cv2.CV_32F,ksize=3)
    sampled_lap = cv2.remap(abs(lap),uv[...,0].astype(np.float32),uv[...,1].astype(np.float32),cv2.INTER_LINEAR)
    sharpness = float(np.quantile(sampled_lap[valid],.8)) if np.any(valid) else 0.
    # Frame sharpness moderates but cannot substitute for exposure-smear bound.
    quality = np.clip(sharpness/30.,.25,2.)/(1+predicted_blur**2)
    weights = valid.astype(np.float32)*np.maximum(incidence,0.)**2*quality
    return srgb_to_linear(rgb),weights,{'sharpness_laplacian_q80':sharpness,'quality_weight':float(quality),
                                       'visible_fraction':float(np.mean(valid))}


def select_diverse_samples(weights, count, views=None):
    """Greedy diminishing coverage: favors new reliable material areas."""
    selected=[]; accumulated=np.zeros(weights.shape[1:],np.float32)
    # Frame-level blur quality belongs in photometric fusion, while reference
    # selection must not let repeated perfect home views crowd out newly seen
    # material. Normalize quality here and retain incidence across each image.
    scale=np.max(weights,axis=(1,2))
    coverage=weights/np.maximum(scale[:,None,None],1e-8)
    for _ in range(min(count,len(weights))):
        scores=(coverage/(.05+accumulated[None])).sum(axis=(1,2))
        scores[selected]=-np.inf
        if views is not None and len(selected)>=4:
            difference=np.angle(np.exp(1j*(np.asarray(views)[:,None,:]-np.asarray(views)[selected][None,:,:])))
            repeated=np.sum(np.max(abs(difference),axis=2)<np.deg2rad(3.),axis=1)>=4
            scores[repeated]=-np.inf
        index=int(np.argmax(scores))
        if not np.isfinite(scores[index]) or scores[index]<=0: break
        selected.append(index); accumulated+=coverage[index]
    return sorted(selected)


def build_atlases(native_path, baseline_path, output, *, config_path=None,
                   height=192, width=512, max_references=24, max_blur_px=6.,
                   cutoffs=(18.5,10.5), progress=print):
    cfg=load_config(config_path or BASELINE/'config/rig.yaml')
    cameras,intrinsics,_=load_rig(cfg); axes=load_axes(cfg)
    F,pivot=axes['R_bc'],axes['pivot_c920_m']; geometry=cfg['geometry']
    native=load_native_tracks(native_path)
    baseline_path=Path(baseline_path); report=json.loads(baseline_path.read_text())
    if report['calibration_hashes'] != calibration_hashes(cfg) or not np.allclose(report['F'],F) or not np.allclose(report['pivot'],pivot):
        raise ValueError('Atlas trajectory and current geometry calibration differ')
    if report['geometry'] != geometry or [Path(p).resolve() for p in report['videos']] != [Path(p).resolve() for p in native['videos']]:
        raise ValueError('Atlas trajectory and native recordings/geometry differ')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    summary={'method':'frozen_measured_per_camera_texture','baseline':str(baseline_path.resolve()),
             'baseline_sha256':sha256(baseline_path),'source_sha256':sha256(__file__),
             'baseline_bundle_sha256':sha256(baseline_path.with_name('bundle.npz')),
             'axes_sha256':sha256(cfg['axes']['path']),'F':F,'pivot':pivot,
             'calibration_hashes':calibration_hashes(cfg),'geometry':geometry,
             'native_provenance':native['provenance'],'config_snapshot':cfg,
             'observed_occlusion':occlusion_metadata(),'atlases':[]}
    for ci,name in enumerate(CAMERA_NAMES):
        exposure_s=cfg['cameras'][name]['capture']['exposure_us']/1e6
        candidates=[reference_candidates(report['frames'],name,h,cutoffs[h],exposure_s,native['circles'][ci][2],
                                          max_blur_px=max_blur_px,count=max_references*2) for h in (0,1)]
        by_index={int(f['source_frame']):[] for group in candidates for f in group}
        for h,group in enumerate(candidates):
            for f in group:by_index[int(f['source_frame'])].append((h,f))
        info=intrinsics[ci];video=SelectedVideo(native['videos'][ci],info['image_size'])
        mapx,mapy=cv2.initUndistortRectifyMap(info['K'],info['dist'],None,info['K'],tuple(info['image_size']),cv2.CV_32FC1)
        grids=[material_grid(height,width,geometry.get('red_shell_sign',1)*(1-2*h)) for h in (0,1)]
        samples=[[],[]];weights=[[],[]];details=[[],[]]
        try:
            for frame_index in sorted(by_index):
                image=cv2.remap(video.read(frame_index),mapx,mapy,cv2.INTER_LINEAR)
                for h,f in by_index[frame_index]:
                    rgb,w,quality=_sample_reference(image,cameras[ci],F,pivot,np.asarray(f['angles']),h,grids[h],geometry,f['predicted_blur_px'],name)
                    samples[h].append(rgb);weights[h].append(w)
                    details[h].append({'source_frame':frame_index,'time_s':f['time_s'],'angles':f['angles'],
                                       'undistorted_frame_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
                                       'predicted_blur_px':f['predicted_blur_px'],**quality})
        finally:video.close()
        for h,label in enumerate(('red','green')):
            views=np.asarray([f['angles'] for f in details[h]])[:,[0,1+h]]
            selected=select_diverse_samples(np.asarray(weights[h]),max_references,views)
            fused=robust_fuse(np.asarray(samples[h])[selected],np.asarray(weights[h])[selected])
            metadata={'camera':name,'camera_id':ci,'shell':label,'shell_id':h,'sign':geometry.get('red_shell_sign',1)*(1-2*h),
                'height':height,'width':width,'grid':'phi=(x+.5)*2pi/W; theta=(y+.5)*pi/(2H); n_z=sign*cos(theta)',
                'references':[details[h][i] for i in selected],'cutoff_s':cutoffs[h],
                'excluded_reference_intervals_s':[[7.7,8.3]],
                'max_predicted_blur_px':max_blur_px,'exposure_s_assumed':exposure_s,'incidence_min':.30,
                'rim_margin_deg':5.,'min_reference_count':2,'fusion_color_tolerance_linear_rgb':.18,
                'photometry':'Approximate sRGB transfer inversion; uncalibrated ISP; fixed per-camera texture, no inpainting.',
                'gauge':'Conditional on supplied trajectory; texture registration is not an independent orientation ground truth.',
                'baseline_sha256':summary['baseline_sha256'],'source_sha256':summary['source_sha256'],
                'baseline_bundle_sha256':summary['baseline_bundle_sha256'],
                'axes_sha256':summary['axes_sha256'],
                'observed_occlusion':summary['observed_occlusion'],
                'valid_fraction':float(np.mean(fused['valid']))}
            path=out/f'atlas_{name}_{label}.npz'
            np.savez_compressed(path,**fused,metadata_json=np.array(json.dumps(metadata)))
            preview=(linear_to_srgb(fused['rgb'])*255).astype(np.uint8)
            preview[~fused['valid']]=[25,25,25]
            cv2.imwrite(str(path.with_suffix('.png')),preview[...,::-1])
            summary['atlases'].append({**metadata,'path':str(path.resolve()),'sha256':sha256(path)})
            if progress:progress(f'{name} {label}: {len(selected)} references, {100*metadata["valid_fraction"]:.1f}% measured atlas coverage',flush=True)
    write_json(out/'atlas_manifest.json',summary)
    return summary
