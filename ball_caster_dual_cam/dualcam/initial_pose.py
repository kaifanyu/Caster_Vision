"""Estimate a run's starting roll from the two separated-shell rim curves.

The rim is an absolute orientation cue; the painted triangles alone have an
unknown initial material phase. Inputs are already undistorted BGR images in
each calibration's original pixel coordinates. This only changes the run's
initial angle, never the calibrated mechanical frame or the metric geometry.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

from .config import CAMERA_NAMES
from .model import project, world_points
from .tracking import choose_circle, masks_for


def _edge_maps(image, camera_config):
    """Orientation-conditioned distances to unpainted physical image edges."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    direction = np.mod(np.arctan2(gy, gx), np.pi)
    edges = cv2.Canny(gray, 35, 100) != 0
    # Triangle borders are material features, not shell rims. Retain neutral
    # plastic edges and discard a small band around segmented paint.
    masks = masks_for(image, choose_circle(image, camera_config), camera_config)
    paint = (masks['top'] | masks['bottom']).astype(np.uint8)
    paint = cv2.dilate(paint, np.ones((7, 7), np.uint8)) != 0
    edges &= ~paint
    maps = []
    bins = 12
    for angle in np.arange(bins)*np.pi/bins:
        difference = abs(np.arctan2(np.sin(2*(direction-angle)),
                                   np.cos(2*(direction-angle))))/2
        selected = edges & (difference <= np.pi/8)
        maps.append(cv2.distanceTransform((~selected).astype(np.uint8),
                                         cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
    return np.asarray(maps), int(edges.sum())


def _rim_pixels(camera, F, pivot, angle, shell, radius, gap, red_sign, count=180):
    phi = np.arange(count)*2*np.pi/count
    points = np.column_stack((np.cos(phi), np.sin(phi), np.zeros(count)))
    q = np.array([angle, 0., 0.])
    xyz = world_points(F, pivot, q, shell, points, radius, gap, red_sign)
    centers = world_points(F, pivot, q, shell, np.zeros_like(points), radius, gap, red_sign)
    eye = -np.asarray(camera['R']).T @ np.asarray(camera['t'])
    facing = np.sum((xyz-centers)*(eye-xyz), axis=1) > 0
    uv = project(camera, xyz)
    tangent = np.roll(uv, -1, axis=0)-np.roll(uv, 1, axis=0)
    # Edge maps use the image gradient (the curve's normal), modulo pi.
    normal_angle = np.mod(np.arctan2(tangent[:, 1], tangent[:, 0])+np.pi/2, np.pi)
    depth = (xyz @ np.asarray(camera['R']).T+camera['t'])[:, 2]
    return uv[facing & (depth > 0)], normal_angle[facing & (depth > 0)]


def _sample_distances(maps, pixels, normal_angles):
    h, w = maps.shape[1:]
    inside = ((pixels[:, 0] >= 1) & (pixels[:, 0] < w-2)
              & (pixels[:, 1] >= 1) & (pixels[:, 1] < h-2))
    if not inside.any():
        return np.empty(0)
    uv = pixels[inside]
    bins = np.rint(normal_angles[inside]*len(maps)/np.pi).astype(int) % len(maps)
    x, y = np.floor(uv).astype(int).T
    dx, dy = (uv-np.floor(uv)).T
    return ((1-dx)*(1-dy)*maps[bins, y, x]+dx*(1-dy)*maps[bins, y, x+1]
            +(1-dx)*dy*maps[bins, y+1, x]+dx*dy*maps[bins, y+1, x+1])


def fit_initial_pose(cameras, frames, cfg, F, pivot, initial_roll_deg=8.,
                     search_half_width_deg=25.):
    """Fit one starting roll from both views; unknown shell phases remain zero.

    ``frames`` is ``[c920_images, brio_images]``; each entry can be one BGR
    image or a sequence from a short stationary start interval. Returns a
    JSON-compatible dict. ``initial_roll_deg`` in the result is the fitted
    angle only when ``accepted``; otherwise it is the explicitly supplied
    seed. ``candidate_roll_deg`` and rejection reasons remain inspectable.

    Occluding yokes and weak edges are handled by trimming the worst 35% of
    each shell's rim samples. Per-camera agreement and profile conditioning
    gates prevent featureless/ambiguous images from manufacturing a reference.
    Reported profile width is a fit diagnostic, not calibrated uncertainty.
    """
    if len(cameras) != 2 or len(frames) != 2:
        raise ValueError('Initial pose needs exactly two camera/image groups')
    if not np.isfinite(initial_roll_deg) or not np.isfinite(search_half_width_deg) or search_half_width_deg <= 0:
        raise ValueError('Initial roll and positive search width must be finite')
    F, pivot = np.asarray(F, float), np.asarray(pivot, float)
    geo = cfg['geometry']
    radius, gap = float(geo['radius_m']), float(geo['gap_m'])
    red_sign = int(geo.get('red_shell_sign', 1))
    groups = []
    edge_counts = []
    for ci, group in enumerate(frames):
        images = [group] if isinstance(group, np.ndarray) and group.ndim == 3 else list(group)
        if not images:
            raise ValueError('Each camera needs at least one initial image')
        maps = []
        counts = []
        for image in images:
            if np.asarray(image).ndim != 3 or image.shape[2] != 3:
                raise ValueError('Initial images must be undistorted BGR arrays')
            distance, count = _edge_maps(image, cfg['cameras'][CAMERA_NAMES[ci]])
            maps.append(distance)
            counts.append(count)
        groups.append(maps)
        edge_counts.append(counts)

    def evaluate(angle_deg, camera_id=None, details=False):
        values = []
        rows = []
        for ci in range(2) if camera_id is None else [camera_id]:
            for shell in (0, 1):
                uv, normal = _rim_pixels(cameras[ci], F, pivot, np.deg2rad(angle_deg),
                                         shell, radius, gap, red_sign)
                distances = [_sample_distances(m, uv, normal) for m in groups[ci]]
                distance = np.median(np.asarray(distances), axis=0) if distances and len(distances[0]) else np.empty(0)
                if len(distance) < 20:
                    cost, support, median = 25., 0., 25.
                else:
                    retained = np.sort(distance)[:max(20, int(np.ceil(.65*len(distance))))]
                    cost = float(np.mean(np.minimum(retained, 25.)))
                    support = float(np.mean(distance <= 6.))
                    median = float(np.median(distance))
                values.append(cost)
                rows.append({'camera': CAMERA_NAMES[ci], 'shell': 'red' if shell == 0 else 'green',
                             'trimmed_mean_distance_px': cost, 'median_distance_px': median,
                             'fraction_within_6px': support, 'samples': len(distance)})
        return rows if details else float(np.mean(values))

    lo, hi = initial_roll_deg-search_half_width_deg, initial_roll_deg+search_half_width_deg
    grid = np.linspace(lo, hi, int(np.ceil((hi-lo)*4))+1)
    costs = np.array([evaluate(a) for a in grid])
    index = int(np.argmin(costs))
    bounds = (grid[max(0, index-2)], grid[min(len(grid)-1, index+2)])
    fitted = minimize_scalar(evaluate, bounds=bounds, method='bounded', options={'xatol': .005})
    angle, cost = float(fitted.x), float(fitted.fun)
    per_camera = []
    for ci in range(2):
        profile = np.array([evaluate(a, ci) for a in grid])
        j = int(np.argmin(profile))
        per_camera.append({'camera': CAMERA_NAMES[ci], 'candidate_roll_deg': float(grid[j]),
                           'trimmed_mean_distance_px': float(profile[j]),
                           'edge_pixels': edge_counts[ci]})
    near = grid[costs <= cost+1.]
    width = float(np.ptp(near)) if len(near) else 0.
    separated = abs(grid-angle) >= 5.
    competing_margin = float(np.min(costs[separated])-cost) if separated.any() else 0.
    rows = evaluate(angle, details=True)
    reasons = []
    if min(angle-lo, hi-angle) < .75:
        reasons.append('minimum lies at the search boundary')
    if cost > 6.:
        reasons.append('rim reprojection distances exceed six pixels')
    if competing_margin < .5 or width > 8.:
        reasons.append('initial roll is weakly constrained or has a competing minimum')
    if abs(per_camera[0]['candidate_roll_deg']-per_camera[1]['candidate_roll_deg']) > 8.:
        reasons.append('camera-specific initial roll estimates disagree by over eight degrees')
    if any(row['samples'] < 20 or row['fraction_within_6px'] < .25 for row in rows):
        reasons.append('insufficient visible rim edge support')
    return {'accepted': not reasons, 'initial_roll_deg': angle if not reasons else float(initial_roll_deg),
            'candidate_roll_deg': angle, 'supplied_seed_deg': float(initial_roll_deg),
            'search_bounds_deg': [float(lo), float(hi)],
            'method': 'two_view_front_rim_oriented_edge_fit',
            'trimmed_mean_distance_px': cost, 'profile_width_at_plus_1px_deg': width,
            'competing_minimum_margin_px': competing_margin, 'per_camera': per_camera,
            'rim_support': rows, 'rejection_reasons': reasons,
            'profile': {'roll_deg': grid.tolist(), 'cost_px': costs.tolist()},
            'note': 'Fixed calibrated frame/pivot and measured radius/gap. Rim fit assumes stationary initial images; profile width is not an absolute accuracy estimate. Shell spins remain relative to their initial paint phases.'}
