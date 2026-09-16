"""Local image anchors for rotation tracking, with explicit loss/recovery states.

Each keyframe retains its original pixels. Direct LK observations against those
pixels constrain the current absolute orientation, independently of the chain
of adjacent-frame rotations. This is a bounded local correction, not global
bundle adjustment or a guarantee against long-term drift.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import cv2
import numpy as np

from .estimate import (
    HemisphereEstimate, angular_residuals, kabsch,
    prepare_sphere_correspondences, solve_hemisphere_increment,
)
from .rotation import geodesic_angle
from .segment import SegmentationMasks
from .track import KLTConfig, TrackMatches, track_features


@dataclass(frozen=True)
class TemporalConfig:
    enabled: bool = False
    correction_interval: int = 3
    keyframe_interval: int = 8
    max_keyframes: int = 3
    max_keyframe_age: int = 45
    min_inliers: int = 20
    min_inlier_ratio: float = 0.70
    max_residual_deg: float = 0.50
    min_spread_fraction: float = 0.025
    min_sharpness_ratio: float = 0.30
    keyframe_sharpness_ratio: float = 0.80
    max_keyframe_rotation_deg: float = 30.0
    max_correction_deg: float = 5.0
    max_anchor_disagreement_deg: float = 3.0
    boundary_margin_px: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("temporal.enabled must be boolean")
        for name in ("correction_interval", "keyframe_interval", "max_keyframes",
                     "max_keyframe_age", "min_inliers", "boundary_margin_px"):
            value = getattr(self, name)
            floor = 0 if name == "boundary_margin_px" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < floor:
                raise ValueError(f"temporal.{name} must be an integer >= {floor}")
        if self.min_inliers < 3 or self.keyframe_interval > self.max_keyframe_age:
            raise ValueError("temporal needs >=3 inliers and keyframe_interval <= max_keyframe_age")
        for name in ("min_inlier_ratio", "min_spread_fraction", "min_sharpness_ratio",
                     "keyframe_sharpness_ratio", "max_residual_deg",
                     "max_keyframe_rotation_deg", "max_correction_deg",
                     "max_anchor_disagreement_deg"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"temporal.{name} must be finite and positive")
        for name in ("min_inlier_ratio", "min_spread_fraction", "min_sharpness_ratio",
                     "keyframe_sharpness_ratio"):
            if getattr(self, name) > 1:
                raise ValueError(f"temporal.{name} must be <= 1")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> TemporalConfig:
        values = dict(values or {})
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown temporal settings: {sorted(unknown)}")
        return cls(**values)


def safe_tracking_masks(masks: SegmentationMasks, margin_px: int) -> SegmentationMasks:
    """Inset feature centers from mask boundaries, including detected occluders.

    This is a configurable center margin; it does not claim to isolate every
    coarser pyramid window or to detect occlusion missing from segmentation.
    """
    if margin_px == 0:
        return masks
    kernel = np.ones((2 * margin_px + 1, 2 * margin_px + 1), np.uint8)
    inset = lambda mask: cv2.erode(mask.astype(np.uint8), kernel,
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return SegmentationMasks(masks.ball, inset(masks.top), inset(masks.bottom), masks.yoke)


def image_sharpness(gray: np.ndarray, mask: np.ndarray) -> float:
    pixels = cv2.Laplacian(gray, cv2.CV_32F)[np.asarray(mask, dtype=bool)]
    return float(np.var(pixels)) if len(pixels) else 0.0


def estimate_quality(matches: TrackMatches, estimate: HemisphereEstimate,
                     radius_px: float, config: TemporalConfig) -> dict[str, Any]:
    """Assess accepted correspondences, not the rejected pre-filter LK set."""
    inlier_uv = matches.uv_curr[estimate.inlier_mask]
    spread = 0.0
    if len(inlier_uv) >= 3:
        centered = inlier_uv - np.mean(inlier_uv, axis=0)
        spread = float(np.linalg.svd(centered, compute_uv=False)[-1]
                       / np.sqrt(len(centered)) / radius_px)
    residual = estimate.mean_inlier_residual_deg
    reasons = []
    if not estimate.success:
        reasons.append(estimate.failure_reason or "rotation solve failed")
    if estimate.inlier_count < config.min_inliers:
        reasons.append("too few inliers")
    if estimate.inlier_ratio < config.min_inlier_ratio:
        reasons.append("low inlier agreement")
    if not np.isfinite(residual) or residual > config.max_residual_deg:
        reasons.append("high rotation residual")
    if spread < config.min_spread_fraction:
        reasons.append("features too concentrated or collinear")
    return {
        "accepted": not reasons, "reasons": reasons,
        "matches": matches.count, "inliers": estimate.inlier_count,
        "inlier_ratio": estimate.inlier_ratio, "mean_residual_deg": residual,
        "spread_fraction": spread,
        "accepted_fb_median_px": float(np.median(matches.fb_error)) if matches.count else None,
    }


@dataclass
class Keyframe:
    index: int
    gray: np.ndarray
    mask: np.ndarray
    uv: np.ndarray
    ids: np.ndarray
    directions: np.ndarray
    pose: np.ndarray
    sharpness: float


class TemporalRotationTracker:
    """Track one shell's absolute orientation relative to frame zero."""

    def __init__(self, K: np.ndarray, center: np.ndarray, radius: float,
                 radius_px: float, klt: KLTConfig, solver_options: Mapping[str, Any],
                 config: TemporalConfig, *, seed: int = 7):
        self.K, self.center, self.radius = K, center, radius
        self.radius_px, self.klt, self.config = radius_px, klt, config
        self.solver_options = dict(solver_options)
        self.rng = np.random.default_rng(seed)
        self.pose = np.eye(3)
        self.prediction = np.eye(3)
        self.valid = True
        self.recent_sharpness: deque[float] = deque(maxlen=config.keyframe_interval)
        self.keyframes: list[Keyframe] = []
        self.recovery_keyframe: Keyframe | None = None
        self.history: list[dict[str, Any]] = []
        # Consumed immediately by optional offline processing; images are not
        # retained in the trajectory history.
        self.last_anchor_observations: list[tuple[Keyframe, TrackMatches, HemisphereEstimate]] = []

    def _make_keyframe(self, index, gray, mask, uv, ids, sharpness) -> Keyframe | None:
        geometry = prepare_sphere_correspondences(
            uv, uv, self.K, self.center, self.radius,
            limb_cull_deg=self.solver_options.get("limb_cull_deg", 65.0))
        keep = geometry.valid_mask
        if np.count_nonzero(keep) < self.config.min_inliers:
            return None
        return Keyframe(index, gray.copy(), mask.copy(), uv[keep].copy(),
                        ids[keep].copy(), geometry.dirs_prev[keep].copy(),
                        self.pose.copy(), sharpness)

    def _add_keyframe(self, index, gray, mask, uv, ids, sharpness) -> None:
        key = self._make_keyframe(index, gray, mask, uv, ids, sharpness)
        if key is None:
            return
        self.keyframes.append(key)
        self.keyframes = self.keyframes[-self.config.max_keyframes:]

    def initialize(self, gray, mask, uv, ids) -> None:
        sharpness = image_sharpness(gray, mask)
        self.recent_sharpness.append(sharpness)
        self._add_keyframe(0, gray, mask, uv, ids, sharpness)
        self.recovery_keyframe = self.keyframes[-1] if self.keyframes else None
        self.history.append({"frame_index": 0, "status": "initial", "valid": True,
                             "keyframes": [key.index for key in self.keyframes]})

    def _observe(self, key: Keyframe, gray, mask, prediction, uv, ids):
        # Surviving feature identities provide a good search seed. Lost tracks
        # can still be recovered using the sphere's predicted image projection.
        relative = prediction @ key.pose.T
        directions = key.directions @ relative.T
        xyz = self.center + self.radius * directions
        homogeneous = xyz @ self.K.T
        guesses = (homogeneous[:, :2] / homogeneous[:, 2:]).astype(np.float32)
        known = {int(identifier): point for identifier, point in zip(ids, uv)}
        for i, identifier in enumerate(key.ids):
            if int(identifier) in known:
                guesses[i] = known[int(identifier)]
        direct = track_features(key.gray, gray, key.uv, prev_mask=key.mask,
                                curr_mask=mask, config=self.klt, initial_uv_curr=guesses)
        estimate = solve_hemisphere_increment(
            direct.uv_prev, direct.uv_curr, self.K, self.center, self.radius,
            rng=self.rng, **self.solver_options)
        quality = estimate_quality(direct, estimate, self.radius_px, self.config)
        quality["keyframe"] = key.index
        if estimate.success:
            angle = float(np.rad2deg(geodesic_angle(estimate.R, np.eye(3))))
            quality["rotation_deg"] = angle
            if angle > self.config.max_keyframe_rotation_deg:
                quality["reasons"].append("keyframe view is too far away")
                quality["accepted"] = False
        return direct, estimate, quality

    def update(self, index: int, gray: np.ndarray, mask: np.ndarray,
               matches: TrackMatches, estimate: HemisphereEstimate,
               uv: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, bool, dict[str, Any]]:
        cfg = self.config
        self.last_anchor_observations = []
        previous_valid = self.valid
        sharpness = image_sharpness(gray, mask)
        # Use local image history: texture and lighting change as a shell turns.
        # Comparing forever with the initial sharp image would label a different
        # visible marking distribution as blur and prevent new anchors entirely.
        reference = float(np.median(self.recent_sharpness)) if self.recent_sharpness else sharpness
        sharpness_ratio = sharpness / max(reference, 1.0)
        self.recent_sharpness.append(sharpness)
        sharp = sharpness >= 1.0 and sharpness_ratio >= cfg.min_sharpness_ratio
        quality = estimate_quality(matches, estimate, self.radius_px, cfg)
        if not sharp:
            quality["accepted"] = False
            quality["reasons"].append("blur or insufficient texture")
        adjacent_good = quality["accepted"]
        # Weak increments may seed an image search through a gap, but never
        # become accepted motion without a fresh keyframe observation.
        if estimate.R is not None:
            self.prediction = estimate.R @ self.prediction
        prediction = self.prediction
        candidate = estimate.R @ self.pose if adjacent_good and previous_valid else None
        attempts, observations = [], []
        contested = False
        correct = index % cfg.correction_interval == 0 or not adjacent_good or not previous_valid
        if correct and sharp:
            anchors = list(self.keyframes)
            # Retain the last trustworthy sharp frame between scheduled
            # keyframes. It gives a short direct bridge across a sudden blur
            # burst without replacing the older drift-correction references.
            rescue = self.recovery_keyframe
            if ((not adjacent_good or not previous_valid) and rescue is not None
                    and all(key.index != rescue.index for key in anchors)):
                anchors.append(rescue)
            for key in anchors:
                if index - key.index > cfg.max_keyframe_age:
                    continue
                direct, anchor_estimate, anchor_quality = self._observe(
                    key, gray, mask, prediction, uv, ids)
                attempts.append(anchor_quality)
                if anchor_quality["accepted"]:
                    pose = anchor_estimate.R @ key.pose
                    if candidate is not None:
                        correction = float(np.rad2deg(geodesic_angle(pose, prediction)))
                        if correction > cfg.max_correction_deg:
                            contested = True
                            anchor_quality["accepted"] = False
                            anchor_quality["reasons"].append("correction exceeds limit")
                            continue
                    observations.append((key, direct, anchor_estimate, pose))
        # All retained views must agree before they jointly refine a pose.
        disagreement = max((float(np.rad2deg(geodesic_angle(a[3], b[3])))
                            for a in observations for b in observations), default=0.0)
        rejection = None
        anchored = None
        if disagreement > cfg.max_anchor_disagreement_deg:
            rejection = "keyframes disagree"
            contested = True
        elif observations:
            world, current = [], []
            # Equal observation counts prevent a richly textured recent frame
            # from overwhelming the older anchor. These fits are correlated;
            # their point count is not an independent confidence probability.
            count = min(item[2].inlier_count for item in observations)
            for key, direct, anchor_estimate, _ in observations:
                geometry = prepare_sphere_correspondences(
                    direct.uv_prev, direct.uv_curr, self.K, self.center, self.radius,
                    limb_cull_deg=self.solver_options.get("limb_cull_deg", 65.0))
                indices = np.flatnonzero(anchor_estimate.inlier_mask)
                indices = indices[np.linspace(0, len(indices) - 1, count).astype(int)]
                world.append(geometry.dirs_prev[indices] @ key.pose)
                current.append(geometry.dirs_curr[indices])
            a, b = np.concatenate(world), np.concatenate(current)
            fitted = kabsch(a, b)
            residual = float(np.rad2deg(np.mean(angular_residuals(a, b, fitted))))
            if residual <= cfg.max_residual_deg:
                anchored = fitted
            else:
                rejection = "joint keyframe residual exceeds limit"
                contested = True
            if (anchored is not None and candidate is not None
                    and np.rad2deg(geodesic_angle(anchored, prediction)) > cfg.max_correction_deg):
                anchored = None
                rejection = "joint correction exceeds limit"
                contested = True

        status = "unresolved"
        correction_deg = None
        if anchored is not None:
            self.last_anchor_observations = [(key, direct, result)
                                             for key, direct, result, _ in observations]
            correction_deg = float(np.rad2deg(geodesic_angle(anchored, prediction)))
            candidate = anchored
            status = "keyframe" if adjacent_good and previous_valid else "recovered"
        elif candidate is not None:
            status = "increment"
        self.valid = candidate is not None
        if self.valid:
            self.pose = candidate
            self.prediction = candidate.copy()
        # An unresolved pose is only a display placeholder. Never use a good
        # adjacent step to resume the global chain after a gap: require an anchor.
        # Do not turn a disputed adjacent estimate into a new reference and
        # eventually evict the older anchors that exposed the disagreement.
        if self.valid and not contested and sharp:
            if (sharpness_ratio >= cfg.keyframe_sharpness_ratio
                    and (not self.keyframes or index - self.keyframes[-1].index >= cfg.keyframe_interval)):
                self._add_keyframe(index, gray, mask, uv, ids, sharpness)
            # The rescue image is the latest accepted observation. Keeping a
            # slightly sharper but older frame here can unnecessarily enlarge
            # the gap just when rapid motion makes nearby overlap essential.
            self.recovery_keyframe = (self.keyframes[-1]
                                      if self.keyframes and self.keyframes[-1].index == index
                                      else self._make_keyframe(index, gray, mask, uv, ids, sharpness))
        record = {
            "frame_index": index, "status": status, "valid": self.valid,
            "adjacent": quality, "sharpness": sharpness, "sharpness_ratio": sharpness_ratio,
            "anchor_attempts": attempts, "anchor_rejection": rejection,
            "anchor_disagreement_deg": disagreement,
            "used_keyframes": [item[0].index for item in observations] if anchored is not None else [],
            "correction_deg": correction_deg, "keyframes": [key.index for key in self.keyframes],
            "keyframe_promotion_blocked": contested,
            "recovery_keyframe": self.recovery_keyframe.index if self.recovery_keyframe else None,
        }
        self.history.append(record)
        return self.pose.copy(), self.valid, record


def temporal_report(config: TemporalConfig, trackers: Mapping[str, TemporalRotationTracker]) -> dict[str, Any]:
    summary = {}
    for name, tracker in trackers.items():
        frames = tracker.history[1:]
        corrections = [record["correction_deg"] for record in frames if record["correction_deg"] is not None]
        summary[name] = {
            "status_counts": dict(Counter(record["status"] for record in frames)),
            "unresolved_frames": [record["frame_index"] for record in frames if not record["valid"]],
            "max_correction_deg": max(corrections, default=0.0),
        }
    return {"config": asdict(config), "summary": summary,
            "frames": {name: tracker.history for name, tracker in trackers.items()}}
