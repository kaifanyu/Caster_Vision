"""Appearance-verified re-observations of already measured material landmarks.

Projected pixels initialize image matching only. Every returned pixel comes
from forward/backward LK and an independent patch comparison; the mechanical
estimator still applies its ordinary geometric and image-support gates.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import logging
import tempfile

import cv2
import numpy as np

from .camera import undistort_image, validate_camera_matrix
from .io_frames import FrameSource
from .offline import SurfaceObservation
from .rotation import Rx, Rz
from .shell_geometry import surface_camera
from .track import KLTConfig, track_features

logger = logging.getLogger(__name__)


def _patch_similarity(first, second, source_uv, target_uv, size=15):
    half = size//2+1
    for image, point in ((first, source_uv), (second, target_uv)):
        if not (half <= point[0] < image.shape[1]-half and half <= point[1] < image.shape[0]-half):
            return None
    a = cv2.getRectSubPix(first, (size, size), tuple(map(float, source_uv))).astype(float)
    b = cv2.getRectSubPix(second, (size, size), tuple(map(float, target_uv))).astype(float)
    if min(a.std(), b.std()) < 5.:
        return None
    a -= a.mean()
    b -= b.mean()
    return float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b)))


def _match_images(source, target, source_uv, guesses, *, source_mask=None,
                  target_mask=None, min_similarity=.85, max_prediction_error_px=32.):
    """Return source indices, observed pixels and appearance diagnostics."""
    matches = track_features(
        source, target, source_uv, initial_uv_curr=guesses,
        prev_mask=source_mask, curr_mask=target_mask,
        config=KLTConfig(klt_win=21, max_level=3, fwd_bwd_err_px=1., criteria_count=40))
    accepted = []
    if matches.source_indices is None:
        return accepted
    for index, uv, fb_error in zip(matches.source_indices, matches.uv_curr, matches.fb_error):
        prediction_error = float(np.linalg.norm(uv-guesses[index]))
        if prediction_error > max_prediction_error_px:
            continue
        similarity = _patch_similarity(source, target, source_uv[index], uv)
        if similarity is None or similarity < min_similarity:
            continue
        accepted.append((int(index), uv.copy(), {"forward_backward_error_px": float(fb_error),
                         "patch_similarity": similarity, "prediction_error_px": prediction_error}))
    return accepted


def _distributed(indices, pixels, limit):
    """Round-robin spatial cells instead of taking the first painted cluster."""
    indices = list(indices)
    if len(indices) <= limit:
        return indices
    points = np.asarray([pixels[index] for index in indices])
    bins = np.minimum(3, ((points-points.min(axis=0))/np.maximum(np.ptp(points, axis=0), 1.)*4).astype(int))
    cells = {}
    for index, cell in zip(indices, bins):
        cells.setdefault(tuple(cell), []).append(index)
    chosen = []
    depth = 0
    while len(chosen) < limit:
        current = [values[depth] for values in cells.values() if len(values) > depth]
        if not current:
            break
        chosen.extend(current[:limit-len(chosen)])
        depth += 1
    return chosen


class ImageLandmarkRematcher:
    """Bounded image rematching with an exact sequential decode and disk cache.

    Call ``close`` (or use a context manager) when the fit is finished. The
    original clip is never changed. Temporary pixels use an eight-frame RAM
    cache and a disk spool, so random source-frame access never seeks a video.
    """

    def __init__(self, clip_path, K, dist, C, F, radius, gap_fraction, geometry,
                 top_shell_sign, *, segmentation_config=None, circle=None,
                 max_frames=None, max_template_age=120, max_templates=3,
                 max_tracks_per_frame=100):
        self.K = validate_camera_matrix(K)
        self.dist = dist
        self.C, self.F = np.asarray(C, float), np.asarray(F, float)
        self.radius, self.gap_fraction = float(radius), float(gap_fraction)
        self.geometry, self.top_shell_sign = geometry, top_shell_sign
        if segmentation_config is not None and circle is None:
            raise ValueError("image rematching segmentation requires the calibrated image circle")
        for name, value in (("max_template_age", max_template_age), ("max_templates", max_templates),
                            ("max_tracks_per_frame", max_tracks_per_frame)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_template_age = int(max_template_age)
        self.max_templates = int(max_templates)
        self.max_tracks_per_frame = int(max_tracks_per_frame)
        self.segmentation_config, self.circle = segmentation_config, circle
        self.source = FrameSource(clip_path, max_frames=max_frames)
        self._iterator = iter(self.source.frames())
        width, height = self.source.size
        self._shape = (height, width)
        self._channels = 2 if segmentation_config is not None else 1
        self._stride = height*width*self._channels
        self._temporary = tempfile.TemporaryDirectory(prefix="ballrot-rematch-")
        self._spool = open(Path(self._temporary.name)/"frames.uint8", "w+b")
        self._decoded = 0
        self._cache = OrderedDict()
        self._closed = False
        self.additions = {"top": [], "bottom": []}
        self.provenance = []
        self._added = {"top": set(), "bottom": set()}
        self._calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if not self._closed:
            self._iterator.close()
            self._cache.clear()
            self._spool.close()
            self._temporary.cleanup()
            self._closed = True

    @property
    def diagnostics(self):
        return {"enabled": True, "method": "accepted_landmark_forward_backward_lk_patch_verification",
                "max_template_age_frames": self.max_template_age, "max_templates_per_target": self.max_templates,
                "max_tracks_per_frame": self.max_tracks_per_frame, "forward_backward_limit_px": 1.,
                "minimum_patch_similarity": .85, "shell_masks_enabled": self.segmentation_config is not None,
                "minimum_target_separation_px": 2.,
                "spatial_collision_rejections": {name: sum(call["spatial_collision_rejections"][name]
                                                            for call in self._calls) for name in self.additions},
                "decoded_frames": self._decoded,
                "added_observation_counts": {name: len(entries) for name, entries in self.additions.items()},
                "limitations": ["A single matching template does not uniquely identify repeated texture; "
                                "geometric consensus and the normal mechanical gates remain required."],
                "calls": list(self._calls)}

    def _frame(self, index):
        if self._closed:
            raise RuntimeError("image rematcher is closed")
        if index < 0:
            raise ValueError("frame index must be nonnegative")
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        while self._decoded <= index:
            try:
                bgr = next(self._iterator)
            except StopIteration as exc:
                raise ValueError(f"rematching clip ended before saved frame {index}") from exc
            image = undistort_image(bgr, self.K, self.dist)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            planes = [gray]
            if self.segmentation_config is not None:
                from .pipeline import _segment
                masks = _segment(image, self.circle, self.segmentation_config)
                labels = masks.top.astype(np.uint8)+2*masks.bottom.astype(np.uint8)
                planes.append(labels)
            self._spool.seek(0, 2)
            self._spool.write(np.stack(planes).tobytes())
            self._decoded += 1
        self._spool.seek(index*self._stride)
        stored = np.frombuffer(self._spool.read(self._stride), np.uint8).reshape((self._channels, *self._shape))
        result = (stored[0], stored[1] if self._channels > 1 else None)
        self._cache[index] = result
        while len(self._cache) > 8:
            self._cache.popitem(last=False)
        return result

    def _templates(self, name, landmarks, measured, reports):
        templates = {}
        for index in np.flatnonzero(measured[name]):
            entry = reports[name][int(index)]
            if entry.get("reason") != "accepted" or entry.get("status") not in ("refined", "recovered"):
                continue
            points = {}
            for observation in entry.get("reprojection", []):
                identifier = observation.get("track_id")
                if identifier not in landmarks[name] or not observation.get("inlier") or not observation.get("visible"):
                    continue
                pixel = np.asarray(observation["observed_uv"], float)
                if pixel.shape == (2,) and np.all(np.isfinite(pixel)):
                    points[int(identifier)] = pixel
            if points:
                templates[int(index)] = points
        return templates

    def propose(self, start, end, landmarks, alpha, beta, measured, by_frame, reports):
        """Return new measured pixels for existing map IDs, never new landmarks."""
        result = {"top": [], "bottom": []}
        attempted = {"top": 0, "bottom": 0}
        collisions = {"top": 0, "bottom": 0}
        source_frames = {"top": set(), "bottom": set()}
        for name, sign, label in (("top", self.top_shell_sign, 1), ("bottom", -self.top_shell_sign, 2)):
            if not landmarks[name]:
                continue
            templates = self._templates(name, landmarks, measured, reports)
            for target in range(start, end):
                if measured[name][target]:
                    continue
                sources = sorted((index for index in templates
                                  if 0 < abs(index-target) <= self.max_template_age),
                                 key=lambda index: (abs(index-target), -len(templates[index]), index))[:self.max_templates]
                if not sources:
                    continue
                existing = {entry.track_id for entry in by_frame[name].get(target, [])}
                existing.update(identifier for frame, identifier in self._added[name] if frame == target)
                orientation = self.F @ Rx(float(alpha[target])) @ Rz(float(beta[name][target]))
                target_gray, target_labels = self._frame(target)
                candidates = {}
                for source in sources:
                    points = templates[source]
                    identifiers = [identifier for identifier in points if identifier not in existing]
                    if not identifiers:
                        continue
                    material = np.asarray([landmarks[name][identifier] for identifier in identifiers])
                    normals = material @ orientation.T
                    xyz = surface_camera(material, orientation, self.C, self.radius, sign,
                                         self.gap_fraction, self.geometry)
                    projected = xyz @ self.K.T
                    guesses = projected[:, :2]/np.maximum(projected[:, 2:], 1e-8)
                    visible = (xyz[:, 2] > 0) & (-np.einsum("ij,ij->i", normals, xyz)/np.linalg.norm(xyz, axis=1)
                                                >= np.cos(np.deg2rad(65.)))
                    h, w = target_gray.shape
                    visible &= ((guesses[:, 0] >= 9) & (guesses[:, 0] < w-9)
                                & (guesses[:, 1] >= 9) & (guesses[:, 1] < h-9))
                    usable = _distributed(np.flatnonzero(visible), guesses, self.max_tracks_per_frame)
                    if not usable:
                        continue
                    identifiers = [identifiers[index] for index in usable]
                    guesses = guesses[usable]
                    uv = np.asarray([points[identifier] for identifier in identifiers])
                    source_gray, source_labels = self._frame(source)
                    matched = _match_images(source_gray, target_gray, uv, guesses,
                                            source_mask=None if source_labels is None else source_labels == label,
                                            target_mask=None if target_labels is None else target_labels == label)
                    attempted[name] += len(uv)
                    source_frames[name].add(source)
                    for index, pixel, details in matched:
                        candidates.setdefault(identifiers[index], []).append((pixel, source, details))
                accepted = {}
                for identifier, matches in candidates.items():
                    pixels = np.asarray([match[0] for match in matches])
                    # Independent accepted templates must agree on identity.
                    if np.max(np.linalg.norm(pixels[:, None]-pixels[None], axis=2)) > 2.:
                        continue
                    accepted[identifier] = max(matches, key=lambda match: match[2]["patch_similarity"])
                # Distinct template-family IDs can describe the same physical
                # corner. They must not manufacture independent pose support.
                ranked = sorted(accepted, key=lambda identifier: (
                    -accepted[identifier][2]["patch_similarity"],
                    accepted[identifier][2]["forward_backward_error_px"], identifier))
                occupied = [entry.uv for entry in by_frame[name].get(target, [])]
                occupied.extend(entry.uv for entry in self.additions[name] if entry.frame_index == target)
                distinct = []
                for identifier in ranked:
                    pixel = accepted[identifier][0]
                    if occupied and np.any(np.linalg.norm(np.asarray(occupied)-pixel, axis=1) <= 2.):
                        collisions[name] += 1
                        continue
                    distinct.append(identifier)
                    occupied.append(pixel)
                selected = _distributed(distinct, {key: value[0] for key, value in accepted.items()},
                                        self.max_tracks_per_frame)
                for identifier in selected:
                    pixel, source, details = accepted[identifier]
                    observation = SurfaceObservation(target, identifier, pixel, 2.)
                    result[name].append(observation)
                    self.additions[name].append(observation)
                    self._added[name].add((target, identifier))
                    self.provenance.append({"shell": name, "frame_index": target, "track_id": identifier,
                                            "source_frame": source, "source_kind": "accepted_reprojection_inlier",
                                            **details})
        self._calls.append({"start_frame": int(start), "end_frame_exclusive": int(end),
                            "attempted_tracks": attempted,
                            "spatial_collision_rejections": collisions,
                            "source_frames": {name: sorted(frames) for name, frames in source_frames.items()},
                            "added_observations": {name: len(entries) for name, entries in result.items()}})
        logger.info("Image rematching %d:%d: proposed top=%d bottom=%d; spatial collisions top=%d bottom=%d",
                    start, end, len(result["top"]), len(result["bottom"]), collisions["top"], collisions["bottom"])
        return result


__all__ = ["ImageLandmarkRematcher"]
