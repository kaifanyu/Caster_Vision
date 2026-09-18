"""Bounded motion search seeds and accepted-image references for visual recovery."""
from __future__ import annotations

from collections import deque
from dataclasses import replace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .appearance import _patch_similarity
from .rotation import geodesic_angle
from .sphere import unproject_to_sphere


class RotationMotionPrior:
    """Short SO(3) extrapolation from accepted poses and their real timestamps.

    A prediction is only an image-search seed. No prediction is an observation,
    and a sufficiently long gap resets the rate history when vision returns.
    """

    def __init__(self, horizon_s):
        self.horizon_s = float(horizon_s)
        self.history = deque(maxlen=5)

    def observe(self, time_s, pose):
        time_s = float(time_s)
        if self.history:
            dt = time_s-self.history[-1][0]
            if dt <= 0:
                raise ValueError("motion-prior observations need increasing timestamps")
            if dt > self.horizon_s:
                self.history.clear()
        self.history.append((time_s, np.asarray(pose, float).copy()))

    def age(self, time_s):
        return None if not self.history else float(time_s-self.history[-1][0])

    def predict(self, time_s):
        if len(self.history) < 2 or not 0 <= self.age(time_s) <= self.horizon_s:
            return None
        entries = list(self.history)
        rotation_sum = np.zeros(3)
        elapsed = 0.
        for (before_t, before), (after_t, after) in zip(entries, entries[1:]):
            rotation_sum += Rotation.from_matrix(after @ before.T).as_rotvec()
            elapsed += after_t-before_t
        rate = rotation_sum/elapsed
        return Rotation.from_rotvec(rate*self.age(time_s)).as_matrix() @ entries[-1][1]


class ViewReferenceBank:
    """Keep a small set of accepted, sharp, different camera orientations."""

    def __init__(self, size, max_age_s, minimum_view_deg):
        self.size, self.max_age_s = int(size), float(max_age_s)
        self.minimum_view = np.deg2rad(minimum_view_deg)
        self.frames = []

    def expire(self, time_s):
        self.frames = [key for key in self.frames if key.timestamp_s is not None
                       and 0 <= time_s-key.timestamp_s <= self.max_age_s]

    def add(self, key):
        if key is None or key.timestamp_s is None:
            return
        self.expire(key.timestamp_s)
        if any(old.index == key.index for old in self.frames):
            return
        distance = [geodesic_angle(key.pose, old.pose) for old in self.frames]
        if distance and min(distance) < self.minimum_view:
            nearest = int(np.argmin(distance))
            if key.sharpness >= .8*self.frames[nearest].sharpness:
                self.frames[nearest] = key
            return
        self.frames.append(key)
        if len(self.frames) > self.size:
            # Retain the newest observation and evict a redundant older view.
            def redundancy(index):
                nearest = min(geodesic_angle(self.frames[index].pose, other.pose)
                              for j, other in enumerate(self.frames) if j != index)
                return nearest, self.frames[index].sharpness, self.frames[index].index
            discard = min(range(len(self.frames)-1), key=redundancy)
            self.frames.pop(discard)


def sphere_patch_affines(source_uv, K, center, radius, relative):
    """Local image deformation of a small material patch under a search pose."""
    samples = np.asarray(source_uv)[:, None, :] + np.array([[0., 0.], [1., 0.], [0., 1.]])
    directions, valid = unproject_to_sphere(samples, K, center, radius)
    xyz = center + radius * (directions @ np.asarray(relative).T)
    homogeneous = xyz @ np.asarray(K).T
    projected = homogeneous[..., :2] / homogeneous[..., 2:]
    affines = np.swapaxes(projected[:, 1:]-projected[:, :1], 1, 2)
    affines[~valid.all(axis=1)] = np.nan
    return affines


def _warped_patch_similarity(source, target, source_uv, target_uv, affine, size=15):
    """Compare measured centers after the predicted local surface deformation."""
    if not np.all(np.isfinite(affine)):
        return None
    half = size//2
    if not (half+1 <= source_uv[0] < source.shape[1]-half-1
            and half+1 <= source_uv[1] < source.shape[0]-half-1):
        return None
    yy, xx = np.mgrid[-half:half+1, -half:half+1]
    offsets = np.stack([xx, yy], axis=-1)
    pixels = offsets @ affine.T + target_uv
    if (np.any(pixels[..., 0] < 0) or np.any(pixels[..., 0] >= target.shape[1]-1)
            or np.any(pixels[..., 1] < 0) or np.any(pixels[..., 1] >= target.shape[0]-1)):
        return None
    a = cv2.getRectSubPix(source, (size, size), tuple(map(float, source_uv))).astype(float)
    b = cv2.remap(target, pixels[..., 0].astype(np.float32), pixels[..., 1].astype(np.float32),
                  interpolation=cv2.INTER_LINEAR).astype(float)
    if min(a.std(), b.std()) < 5.:
        return None
    a -= a.mean()
    b -= b.mean()
    return float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b)))


def verify_patch_matches(matches, source_gray, target_gray, minimum_similarity, *, affines=None):
    """Filter measured LK pixels by appearance and distinct target corners.

    Recovery deliberately adds a <=1 px forward/back check even if ordinary
    tracking was configured more loosely. A pose may warp the patch comparison,
    but its center always remains the pixel actually found by optical flow.
    """
    candidates = []
    for index, (source, target, fb) in enumerate(zip(matches.uv_prev, matches.uv_curr, matches.fb_error)):
        if fb > 1.:
            continue
        similarity = (_patch_similarity(source_gray, target_gray, source, target) if affines is None else
                      _warped_patch_similarity(source_gray, target_gray, source, target, affines[index]))
        if similarity is not None and similarity >= minimum_similarity:
            candidates.append((-similarity, float(fb), index))
    chosen, occupied = [], []
    for _, _, index in sorted(candidates):
        pixel = matches.uv_curr[index]
        if occupied and np.any(np.linalg.norm(np.asarray(occupied)-pixel, axis=1) <= 2.):
            continue
        chosen.append(index)
        occupied.append(pixel)
    chosen = np.asarray(sorted(chosen), int)
    return replace(matches, uv_prev=matches.uv_prev[chosen], uv_curr=matches.uv_curr[chosen],
                   fb_error=matches.fb_error[chosen], source_indices=matches.source_indices[chosen],
                   track_ids=None if matches.track_ids is None else matches.track_ids[chosen])


__all__ = ["RotationMotionPrior", "ViewReferenceBank", "sphere_patch_affines", "verify_patch_matches"]
