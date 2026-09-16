"""Bounded image buffering and observation export for offline pose fitting."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .estimate import prepare_sphere_correspondences, solve_hemisphere_increment
from .offline import OfflineConfig, SurfaceObservation
from .rotation import geodesic_angle
from .temporal import estimate_quality
from .track import track_features


def _distributed_indices(uv: np.ndarray, ids: np.ndarray, limit: int) -> np.ndarray:
    """Keep stable identities within spatial bins, with a bounded image budget."""
    if len(uv) <= limit:
        return np.arange(len(uv))
    lower = np.min(uv, axis=0)
    extent = np.maximum(np.ptp(uv, axis=0), 1.0)
    cells = np.minimum(((uv - lower) / extent * 8).astype(int), 7)
    buckets = {}
    for index in np.argsort(ids, kind="stable"):
        cell = tuple(cells[index])
        buckets.setdefault(cell, []).append(int(index))
    chosen = []
    depth = 0
    while len(chosen) < limit:
        added = [values[depth] for values in buckets.values() if len(values) > depth]
        if not added:
            break
        chosen.extend(added[:limit - len(chosen)])
        depth += 1
    return np.asarray(chosen, dtype=int)


@dataclass
class BufferedFrame:
    index: int
    gray: np.ndarray
    masks: object
    points: dict
    predictions: dict
    records: dict


class OfflineObservationCollector:
    """Collect pixels only; reverse matching never modifies the forward tracker."""

    def __init__(self, config: OfflineConfig, K, center, radius, radius_px, klt,
                 solver_options, temporal_config, *, seed=701):
        self.config = config
        self.K, self.center, self.radius, self.radius_px = K, center, radius, radius_px
        self.klt, self.solver_options, self.temporal_config = klt, solver_options, temporal_config
        self.random = np.random.default_rng(seed)
        self.observations = {name: {} for name in ("top", "bottom")}
        self.buffer = deque(maxlen=config.backward_window_frames)
        self.reverse_diagnostics = []
        self.limit = 2 * config.max_tracks_per_frame
        self._landmark_lookup = {}
        self.landmark_sources = {}

    def _landmark_ids(self, family, source_frame, original_ids):
        """A source-image pixel defines a material point, not its old LK ID.

        Chained LK can drift within a painted mark. A later keyframe snapshot
        must not redefine the world point recorded by an earlier template.
        """
        output = []
        for identifier in original_ids:
            key = (family, int(source_frame), int(identifier))
            if key not in self._landmark_lookup:
                index = len(self._landmark_lookup)
                self._landmark_lookup[key] = index
                self.landmark_sources[index] = key
            output.append(self._landmark_lookup[key])
        return np.asarray(output, dtype=np.int64)

    def _pair(self, name, first_index, second_index, uv_first, uv_second, ids, weight,
              *, family="adjacent"):
        chosen = _distributed_indices(uv_second, ids, self.limit)
        # Forward IDs remain persistent. Direct observations are tied to the
        # exact source template shared by all their target-image measurements.
        landmark_ids = self._landmark_ids(family, 0 if family == "adjacent" else first_index,
                                          np.asarray(ids)[chosen])
        destination = self.observations[name]
        for index, identifier in zip(chosen, landmark_ids):
            identifier = int(identifier)
            for frame, uv in ((first_index, uv_first[index]), (second_index, uv_second[index])):
                key = (frame, identifier)
                previous = destination.get(key)
                if previous is None or weight > previous.weight:
                    destination[key] = SurfaceObservation(frame, identifier, uv.copy(), weight)

    def add_adjacent(self, name, index, matches, estimate, record):
        if record["adjacent"]["accepted"] and matches.track_ids is not None:
            keep = estimate.inlier_mask
            self._pair(name, index - 1, index, matches.uv_prev[keep], matches.uv_curr[keep],
                       matches.track_ids[keep], 1.0)

    def add_anchors(self, name, index, tracker):
        for key, matches, estimate in tracker.last_anchor_observations:
            keep = estimate.inlier_mask
            ids = key.ids[matches.source_indices[keep]]
            self._pair(name, key.index, index, matches.uv_prev[keep], matches.uv_curr[keep], ids, 2.0,
                       family="anchor")

    def push(self, index, gray, masks, points, predictions, records):
        """Use a later good image to re-observe selected earlier buffered images."""
        if (self.config.backward_window_frames > 0
                and index % self.config.backward_stride == 0):
            for name in ("top", "bottom"):
                if not records[name].get("adjacent", {}).get("accepted", False):
                    continue
                uv, ids = points[name]
                chosen = _distributed_indices(uv, ids, self.limit)
                uv, ids = uv[chosen], ids[chosen]
                geometry = prepare_sphere_correspondences(
                    uv, uv, self.K, self.center, self.radius,
                    limb_cull_deg=self.solver_options.get("limb_cull_deg", 65.0))
                uv, ids = uv[geometry.valid_mask], ids[geometry.valid_mask]
                directions = geometry.dirs_prev[geometry.valid_mask]
                if len(uv) < self.temporal_config.min_inliers:
                    continue
                for past in self.buffer:
                    age = index - past.index
                    if age < 2 or (past.records[name].get("valid", False)
                                   and age > self.config.backward_stride + 1
                                   and age % self.config.backward_stride != 0):
                        continue
                    record = past.records[name]
                    # A matching algorithm cannot restore texture absent from a
                    # heavily blurred image. Keep that sample unobserved.
                    if record.get("sharpness", 1.0) < 1.0 or record.get("sharpness_ratio", 1.0) < self.temporal_config.min_sharpness_ratio:
                        continue
                    relative = past.predictions[name] @ predictions[name].T
                    xyz = self.center + self.radius * (directions @ relative.T)
                    projected = xyz @ self.K.T
                    guesses = (projected[:, :2] / projected[:, 2:]).astype(np.float32)
                    old_uv, old_ids = past.points[name]
                    known = dict(zip(map(int, old_ids), old_uv))
                    for j, identifier in enumerate(ids):
                        if int(identifier) in known:
                            guesses[j] = known[int(identifier)]
                    matches = track_features(gray, past.gray, uv, prev_mask=masks[name],
                                             curr_mask=past.masks[name], config=self.klt,
                                             initial_uv_curr=guesses)
                    estimate = solve_hemisphere_increment(
                        matches.uv_prev, matches.uv_curr, self.K, self.center, self.radius,
                        rng=self.random, **self.solver_options)
                    quality = estimate_quality(matches, estimate, self.radius_px, self.temporal_config)
                    if estimate.success:
                        angle = float(np.rad2deg(geodesic_angle(estimate.R, np.eye(3))))
                        quality["rotation_deg"] = angle
                        if angle > self.temporal_config.max_keyframe_rotation_deg:
                            quality["accepted"] = False
                            quality["reasons"].append("reverse view is too far away")
                        # Repeated markings can produce a coherent but wrong
                        # rotation. Two trusted forward poses provide an
                        # independent bounded consistency check; a held pose
                        # inside an unresolved interval does not provide one.
                        disagreement = float(np.rad2deg(geodesic_angle(estimate.R, relative)))
                        quality["forward_disagreement_deg"] = disagreement
                        if (record.get("valid", False) and records[name].get("valid", False)
                                and disagreement > self.temporal_config.max_correction_deg):
                            quality["accepted"] = False
                            quality["reasons"].append("reverse observation disagrees with trusted forward poses")
                    self.reverse_diagnostics.append({"shell": name, "source_frame": index,
                                                     "target_frame": past.index, **quality})
                    if quality["accepted"]:
                        keep = estimate.inlier_mask
                        self._pair(name, index, past.index, matches.uv_prev[keep], matches.uv_curr[keep],
                                   ids[matches.source_indices[keep]], 2.0, family="reverse")
        if self.config.backward_window_frames > 0:
            # The caller may reuse arrays on the next frame. Keep independent
            # snapshots, and retain only masks/quality fields actually needed
            # by reverse matching rather than the full segmentation/history.
            buffered_masks = {name: np.asarray(masks[name], dtype=bool).copy()
                              for name in ("top", "bottom")}
            buffered_points = {name: (uv.copy(), ids.copy())
                               for name, (uv, ids) in points.items()}
            buffered_records = {name: {key: record[key] for key in
                                ("valid", "sharpness", "sharpness_ratio") if key in record}
                                for name, record in records.items()}
            self.buffer.append(BufferedFrame(index, gray.copy(), buffered_masks, buffered_points,
                                              {name: value.copy() for name, value in predictions.items()},
                                              buffered_records))

    def lists(self):
        return {name: list(observations.values()) for name, observations in self.observations.items()}


def save_observations(path: str | Path, observations: dict, *, timestamps, K, center,
                      radius, initial_rotations: dict, initial_valid: dict,
                      refined_rotations: dict, refined_valid: dict,
                      landmark_sources: dict | None = None,
                      unconstrained_rotations: dict | None = None,
                      unconstrained_valid: dict | None = None) -> Path:
    """Save numeric audit arrays; pixel coordinates refer to undistorted images."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (unconstrained_rotations is None) != (unconstrained_valid is None):
        raise ValueError("unconstrained rotations and validity must be supplied together")
    arrays = {"schema_version": np.array(3 if unconstrained_rotations is not None else 2), "timestamps_s": np.asarray(timestamps),
              "K": K, "sphere_center": center, "sphere_radius": np.array(radius),
              "coordinate_system": np.array("undistorted_pixels; camera_relative_rotations")}
    if landmark_sources is not None:
        identifiers = sorted(landmark_sources)
        arrays.update({
            "landmark_id": np.asarray(identifiers, dtype=np.int64),
            "landmark_family": np.asarray([landmark_sources[i][0] for i in identifiers], dtype="U8"),
            "landmark_source_frame": np.asarray([landmark_sources[i][1] for i in identifiers], dtype=np.int64),
            "landmark_source_track_id": np.asarray([landmark_sources[i][2] for i in identifiers], dtype=np.int64),
        })
    for name, values in observations.items():
        arrays.update({
            f"{name}_frame_index": np.array([item.frame_index for item in values], dtype=np.int64),
            f"{name}_track_id": np.array([item.track_id for item in values], dtype=np.int64),
            f"{name}_uv": np.asarray([item.uv for item in values], dtype=float).reshape(-1, 2),
            f"{name}_weight": np.array([item.weight for item in values]),
            f"{name}_initial_rotations": initial_rotations[name],
            f"{name}_initial_valid": initial_valid[name],
            f"{name}_refined_rotations": refined_rotations[name],
            f"{name}_refined_valid": refined_valid[name],
        })
        if unconstrained_rotations is not None:
            arrays[f"{name}_unconstrained_rotations"] = unconstrained_rotations[name]
            arrays[f"{name}_unconstrained_valid"] = unconstrained_valid[name]
    np.savez_compressed(destination, **arrays)
    return destination
