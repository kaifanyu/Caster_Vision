"""Sparse KLT speckle tracking with forward-backward validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class KLTConfig:
    """Parameters for detection and pyramidal Lucas-Kanade tracking."""

    max_corners: int = 400
    quality: float = 0.01
    min_distance_px: float = 6.0
    klt_win: int = 21
    max_level: int = 3
    fwd_bwd_err_px: float = 1.0
    block_size: int = 3
    criteria_count: int = 30
    criteria_epsilon: float = 0.01
    min_eig_threshold: float = 1e-4

    def __post_init__(self) -> None:
        if self.max_corners <= 0:
            raise ValueError("max_corners must be positive")
        if not (0.0 < self.quality <= 1.0):
            raise ValueError("quality must lie in (0, 1]")
        if self.min_distance_px < 0:
            raise ValueError("min_distance_px cannot be negative")
        if self.klt_win <= 1 or self.klt_win % 2 == 0:
            raise ValueError("klt_win must be an odd integer greater than one")
        if self.max_level < 0:
            raise ValueError("max_level cannot be negative")
        if self.fwd_bwd_err_px < 0:
            raise ValueError("fwd_bwd_err_px cannot be negative")
        if self.block_size <= 1:
            raise ValueError("block_size must be greater than one")
        if self.criteria_count <= 0 or self.criteria_epsilon <= 0:
            raise ValueError("KLT termination criteria must be positive")
        if self.min_eig_threshold < 0:
            raise ValueError("min_eig_threshold cannot be negative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "KLTConfig":
        """Build from the ``track:`` config section, ignoring unrelated keys."""

        if values is None:
            return cls()
        values = dict(values)
        if "pyramid_levels" in values and "max_level" not in values:
            values["max_level"] = values["pyramid_levels"]
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass(frozen=True)
class TrackMatches:
    """Accepted correspondences for one hemisphere and one frame pair."""

    uv_prev: np.ndarray
    uv_curr: np.ndarray
    fb_error: np.ndarray
    detected_count: int = 0
    forward_count: int = 0
    backward_count: int = 0
    fb_error_all: np.ndarray | None = None
    source_indices: np.ndarray | None = None
    track_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        uv_prev = np.asarray(self.uv_prev, dtype=np.float64)
        uv_curr = np.asarray(self.uv_curr, dtype=np.float64)
        fb_error = np.asarray(self.fb_error, dtype=np.float64)
        if uv_prev.ndim != 2 or uv_prev.shape[1:] != (2,):
            raise ValueError("uv_prev must have shape (N, 2)")
        if uv_curr.shape != uv_prev.shape:
            raise ValueError("uv_curr must match uv_prev's shape")
        if fb_error.shape != (len(uv_prev),):
            raise ValueError("fb_error must have shape (N,)")
        if not np.all(np.isfinite(uv_prev)) or not np.all(np.isfinite(uv_curr)):
            raise ValueError("accepted tracked pixels must be finite")
        if not np.all(np.isfinite(fb_error)) or np.any(fb_error < 0):
            raise ValueError("accepted forward-backward errors must be finite and >= 0")

        fb_error_all = (
            np.empty(0, dtype=np.float64)
            if self.fb_error_all is None
            else np.asarray(self.fb_error_all, dtype=np.float64)
        )
        if (
            fb_error_all.ndim != 1
            or not np.all(np.isfinite(fb_error_all))
            or np.any(fb_error_all < 0)
        ):
            raise ValueError("fb_error_all must be a finite, non-negative 1-D array")
        source_indices = (
            np.arange(len(uv_prev), dtype=np.int64)
            if self.source_indices is None
            else np.asarray(self.source_indices, dtype=np.int64)
        )
        if source_indices.shape != (len(uv_prev),):
            raise ValueError("source_indices must have shape (N,)")
        track_ids = None
        if self.track_ids is not None:
            supplied_ids = np.asarray(self.track_ids)
            if supplied_ids.shape != (len(uv_prev),) or supplied_ids.dtype.kind not in "iu":
                raise ValueError("track_ids must be an integer array with shape (N,)")
            if np.any(supplied_ids < 0) or len(np.unique(supplied_ids)) != len(supplied_ids):
                raise ValueError("track_ids must be non-negative and unique")
            track_ids = supplied_ids.astype(np.int64, copy=True)

        counts = (self.detected_count, self.forward_count, self.backward_count)
        if any(int(value) < 0 for value in counts):
            raise ValueError("tracking counts cannot be negative")
        if not (
            len(uv_prev)
            <= int(self.backward_count)
            <= int(self.forward_count)
            <= int(self.detected_count)
        ):
            raise ValueError("tracking counts are inconsistent")

        object.__setattr__(self, "uv_prev", uv_prev)
        object.__setattr__(self, "uv_curr", uv_curr)
        object.__setattr__(self, "fb_error", fb_error)
        object.__setattr__(self, "fb_error_all", fb_error_all)
        object.__setattr__(self, "source_indices", source_indices)
        object.__setattr__(self, "track_ids", track_ids)

    @property
    def count(self) -> int:
        return len(self.uv_prev)

    @property
    def retention_ratio(self) -> float:
        return self.count / self.detected_count if self.detected_count else 0.0

    @classmethod
    def empty(cls, detected_count: int = 0) -> "TrackMatches":
        empty_uv = np.empty((0, 2), dtype=np.float64)
        return cls(
            uv_prev=empty_uv,
            uv_curr=empty_uv.copy(),
            fb_error=np.empty(0, dtype=np.float64),
            detected_count=detected_count,
            forward_count=0,
            backward_count=0,
        )


@dataclass(frozen=True)
class FrameMatches:
    """Top and bottom matches between two consecutive frames."""

    top: TrackMatches
    bottom: TrackMatches

    def __getitem__(self, hemisphere: str) -> TrackMatches:
        if hemisphere not in {"top", "bottom"}:
            raise KeyError(hemisphere)
        return getattr(self, hemisphere)

    def as_dict(self) -> dict[str, TrackMatches]:
        return {"top": self.top, "bottom": self.bottom}


def _as_gray(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.dtype != np.uint8:
        raise ValueError("KLT frames must be uint8 images")
    if image.ndim == 2:
        return np.ascontiguousarray(image)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("frame must be grayscale, BGR, or BGRA")


def _as_cv_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    binary = np.asarray(mask)
    if binary.shape != shape:
        raise ValueError(f"mask shape {binary.shape} does not match frame {shape}")
    return np.ascontiguousarray(binary.astype(bool).astype(np.uint8) * 255)


def _points_in_mask(uv: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    points = np.asarray(uv, dtype=float)
    if mask is None:
        return np.all(np.isfinite(points), axis=1)
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    finite = np.all(np.isfinite(points), axis=1)
    xy = np.zeros((len(points), 2), dtype=np.int64)
    xy[finite] = np.rint(points[finite]).astype(np.int64)
    inside = (
        finite
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    result = np.zeros(len(points), dtype=bool)
    indices = np.flatnonzero(inside)
    result[indices] = binary[xy[indices, 1], xy[indices, 0]]
    return result


def detect_features(
    gray: np.ndarray,
    mask: np.ndarray,
    config: KLTConfig | Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Detect Shi-Tomasi corners and return an ``(N, 2)`` float array."""

    cfg = config if isinstance(config, KLTConfig) else KLTConfig.from_mapping(config)
    image = _as_gray(gray)
    cv_mask = _as_cv_mask(mask, image.shape)
    corners = cv2.goodFeaturesToTrack(
        image,
        maxCorners=cfg.max_corners,
        qualityLevel=cfg.quality,
        minDistance=cfg.min_distance_px,
        mask=cv_mask,
        blockSize=cfg.block_size,
        useHarrisDetector=False,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    return np.ascontiguousarray(corners.reshape(-1, 2), dtype=np.float32)


def track_features(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    uv_prev: np.ndarray,
    *,
    prev_mask: np.ndarray | None = None,
    curr_mask: np.ndarray | None = None,
    config: KLTConfig | Mapping[str, Any] | None = None,
    initial_uv_curr: np.ndarray | None = None,
) -> TrackMatches:
    """Track pixels with optional current-image guesses and round-trip gates.

    Guesses initialize an independent image alignment; they do not replace
    the tracked pixels. ``source_indices`` always indexes ``uv_prev``.
    """

    cfg = config if isinstance(config, KLTConfig) else KLTConfig.from_mapping(config)
    previous = _as_gray(prev_gray)
    current = _as_gray(curr_gray)
    if current.shape != previous.shape:
        raise ValueError("consecutive frames must have the same dimensions")
    if prev_mask is not None and np.asarray(prev_mask).shape != previous.shape:
        raise ValueError("prev_mask does not match the frame dimensions")
    if curr_mask is not None and np.asarray(curr_mask).shape != current.shape:
        raise ValueError("curr_mask does not match the frame dimensions")

    points = np.asarray(uv_prev, dtype=np.float32)
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ValueError("uv_prev must have shape (N, 2)")
    guesses = None
    if initial_uv_curr is not None:
        guesses = np.array(initial_uv_curr, dtype=np.float32, copy=True, order="C")
        if guesses.shape != points.shape or not np.all(np.isfinite(guesses)):
            raise ValueError("initial_uv_curr must contain finite pixels with shape (N, 2)")
    detected_count = len(points)
    if detected_count == 0:
        return TrackMatches.empty()
    if not np.all(np.isfinite(points)):
        raise ValueError("uv_prev must contain only finite pixels")

    lk_params = {
        "winSize": (cfg.klt_win, cfg.klt_win),
        "maxLevel": cfg.max_level,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            cfg.criteria_count,
            cfg.criteria_epsilon,
        ),
        "minEigThreshold": cfg.min_eig_threshold,
    }
    if guesses is not None:
        lk_params["flags"] = cv2.OPTFLOW_USE_INITIAL_FLOW
    forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points.reshape(-1, 1, 2),
        None if guesses is None else guesses.reshape(-1, 1, 2),
        **lk_params,
    )
    if forward is None or status_forward is None:
        return TrackMatches.empty(detected_count=detected_count)

    forward = forward.reshape(-1, 2)
    status_forward = status_forward.reshape(-1).astype(bool)
    status_forward &= np.all(np.isfinite(forward), axis=1)
    forward_indices = np.flatnonzero(status_forward)
    forward_count = len(forward_indices)
    if forward_count == 0:
        return TrackMatches(
            np.empty((0, 2)),
            np.empty((0, 2)),
            np.empty(0),
            detected_count=detected_count,
            forward_count=0,
            backward_count=0,
        )

    backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        current,
        previous,
        forward[forward_indices].reshape(-1, 1, 2),
        None if guesses is None else points[forward_indices].reshape(-1, 1, 2).copy(),
        **lk_params,
    )
    if backward is None or status_backward is None:
        return TrackMatches(
            np.empty((0, 2)),
            np.empty((0, 2)),
            np.empty(0),
            detected_count=detected_count,
            forward_count=forward_count,
            backward_count=0,
        )

    backward = backward.reshape(-1, 2)
    status_backward = status_backward.reshape(-1).astype(bool)
    status_backward &= np.all(np.isfinite(backward), axis=1)
    both_local = np.flatnonzero(status_backward)
    backward_count = len(both_local)
    if backward_count == 0:
        return TrackMatches(
            np.empty((0, 2)),
            np.empty((0, 2)),
            np.empty(0),
            detected_count=detected_count,
            forward_count=forward_count,
            backward_count=0,
        )

    source_indices = forward_indices[both_local]
    prior = points[source_indices].astype(np.float64)
    next_points = forward[source_indices].astype(np.float64)
    round_trip = backward[both_local].astype(np.float64)
    fb_error_all = np.linalg.norm(round_trip - prior, axis=1)

    accepted = np.isfinite(fb_error_all) & (fb_error_all <= cfg.fwd_bwd_err_px)
    accepted &= _points_in_mask(prior, prev_mask)
    accepted &= _points_in_mask(next_points, curr_mask)

    return TrackMatches(
        uv_prev=prior[accepted],
        uv_curr=next_points[accepted],
        fb_error=fb_error_all[accepted],
        detected_count=detected_count,
        forward_count=forward_count,
        backward_count=backward_count,
        fb_error_all=fb_error_all,
        source_indices=source_indices[accepted],
    )


def _hemisphere_mask(masks: Any, name: str) -> np.ndarray:
    if isinstance(masks, Mapping):
        if name not in masks:
            raise KeyError(f"masks do not contain {name!r}")
        return np.asarray(masks[name])
    if hasattr(masks, name):
        return np.asarray(getattr(masks, name))
    raise TypeError("masks must be a SegmentationMasks-like object or mapping")


class KLTTracker:
    """Stateless-per-pair tracker that re-seeds features on every frame.

    Re-detecting on each previous frame is intentional: it continually
    replaces speckles entering the central, well-conditioned part of the
    visible sphere and avoids carrying stale tracks near the limb.
    """

    def __init__(self, config: KLTConfig | Mapping[str, Any] | None = None) -> None:
        self.config = (
            config if isinstance(config, KLTConfig) else KLTConfig.from_mapping(config)
        )

    def track_hemisphere(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_mask: np.ndarray,
        curr_mask: np.ndarray | None = None,
    ) -> TrackMatches:
        previous = _as_gray(prev_frame)
        current = _as_gray(curr_frame)
        features = detect_features(previous, prev_mask, self.config)
        return track_features(
            previous,
            current,
            features,
            prev_mask=prev_mask,
            curr_mask=curr_mask,
            config=self.config,
        )

    def track_pair(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_masks: Any,
        curr_masks: Any | None = None,
    ) -> FrameMatches:
        """Detect and track top and bottom speckles for one frame pair."""

        top_current_mask = (
            None if curr_masks is None else _hemisphere_mask(curr_masks, "top")
        )
        bottom_current_mask = (
            None if curr_masks is None else _hemisphere_mask(curr_masks, "bottom")
        )
        top = self.track_hemisphere(
            prev_frame,
            curr_frame,
            _hemisphere_mask(prev_masks, "top"),
            top_current_mask,
        )
        bottom = self.track_hemisphere(
            prev_frame,
            curr_frame,
            _hemisphere_mask(prev_masks, "bottom"),
            bottom_current_mask,
        )
        return FrameMatches(top=top, bottom=bottom)


class PersistentKLTTracker:
    """Carry feature identities forward and replenish only vacant locations.

    Call ``initialize`` on the first image, then ``track_pair`` in image order.
    IDs are unique across both shells for this tracker's lifetime, including
    after reinitialization. ``points`` includes newly seeded features on the
    current image, while returned matches contain only observed tracks.
    """

    def __init__(self, config: KLTConfig | Mapping[str, Any] | None = None) -> None:
        self.config = (
            config if isinstance(config, KLTConfig) else KLTConfig.from_mapping(config)
        )
        self._next_id = 0
        self._shape: tuple[int, int] | None = None
        self._points: dict[str, np.ndarray] = {}
        self._ids: dict[str, np.ndarray] = {}
        self._last_tested: dict[str, np.ndarray] = {}

    @staticmethod
    def _check_name(name: str) -> None:
        if name not in {"top", "bottom"}:
            raise KeyError(name)

    def initialize(self, frame: np.ndarray, masks: Any) -> None:
        gray = _as_gray(frame)
        self._shape = gray.shape
        for name in ("top", "bottom"):
            self._points[name] = np.empty((0, 2), dtype=np.float64)
            self._ids[name] = np.empty(0, dtype=np.int64)
            self._last_tested[name] = np.empty(0, dtype=np.int64)
            self._replenish(name, gray, _hemisphere_mask(masks, name))

    def points(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return independent copies of current-image pixels and their IDs."""

        self._check_name(name)
        if self._shape is None:
            raise RuntimeError("initialize the persistent tracker before reading points")
        return self._points[name].copy(), self._ids[name].copy()

    def prune(self, name: str, allowed_ids: np.ndarray) -> None:
        """Drop rejected tracks, preserving reseeds not tested in the last pair.

        Pass the RANSAC inlier IDs from the last returned matches. Newly seeded
        current-image features have no rotation estimate yet and are retained.
        Vacancies are replenished on the next call to ``track_pair``.
        """

        self._check_name(name)
        if self._shape is None:
            raise RuntimeError("initialize the persistent tracker before pruning")
        keep = ~np.isin(self._ids[name], self._last_tested[name]) | np.isin(
            self._ids[name], np.asarray(allowed_ids, dtype=np.int64)
        )
        self._points[name] = self._points[name][keep]
        self._ids[name] = self._ids[name][keep]

    def _replenish(self, name: str, gray: np.ndarray, mask: np.ndarray) -> None:
        detection_mask = _as_cv_mask(mask, gray.shape)
        existing = self._points[name]
        keep = _points_in_mask(existing, detection_mask)
        existing = existing[keep]
        self._ids[name] = self._ids[name][keep]
        self._points[name] = existing
        remaining = self.config.max_corners - len(existing)
        if remaining <= 0:
            return
        exclusion_radius = int(np.ceil(self.config.min_distance_px))
        for point in existing:
            cv2.circle(
                detection_mask,
                tuple(np.rint(point).astype(int)),
                exclusion_radius,
                0,
                thickness=-1,
            )
        candidates = detect_features(
            gray, detection_mask, replace(self.config, max_corners=remaining)
        )
        # Mask rasterization rounds subpixel locations; retain the exact spacing
        # condition too so a newly detected point cannot duplicate a live track.
        if len(existing) and len(candidates):
            squared_distance = np.sum(
                (candidates[:, None, :] - existing[None, :, :]) ** 2, axis=2
            )
            candidates = candidates[
                np.all(squared_distance > self.config.min_distance_px**2, axis=1)
            ]
        count = len(candidates)
        if count:
            new_ids = np.arange(self._next_id, self._next_id + count, dtype=np.int64)
            self._next_id += count
            self._points[name] = np.concatenate([existing, candidates], axis=0)
            self._ids[name] = np.concatenate([self._ids[name], new_ids])

    def track_pair(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_masks: Any,
        curr_masks: Any | None = None,
    ) -> FrameMatches:
        """Advance both shells; initialize automatically on the first pair.

        If omitted, current masks are assumed unchanged from previous masks.
        """

        previous = _as_gray(prev_frame)
        current = _as_gray(curr_frame)
        if current.shape != previous.shape:
            raise ValueError("consecutive frames must have the same dimensions")
        if self._shape is None:
            self.initialize(previous, prev_masks)
        if self._shape != previous.shape:
            raise ValueError("frame dimensions changed; reinitialize the persistent tracker")
        current_masks = prev_masks if curr_masks is None else curr_masks
        tracked: dict[str, TrackMatches] = {}
        for name in ("top", "bottom"):
            previous_mask = _hemisphere_mask(prev_masks, name)
            current_mask = _hemisphere_mask(current_masks, name)
            self._replenish(name, previous, previous_mask)
            result = track_features(
                previous,
                current,
                self._points[name],
                prev_mask=previous_mask,
                curr_mask=current_mask,
                config=self.config,
            )
            ids = self._ids[name][result.source_indices]
            tracked[name] = replace(result, track_ids=ids)
            self._points[name] = result.uv_curr.copy()
            self._ids[name] = ids.copy()
            self._last_tested[name] = ids.copy()
            self._replenish(name, current, current_mask)
        return FrameMatches(top=tracked["top"], bottom=tracked["bottom"])


def track_frame_pair(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    prev_masks: Any,
    curr_masks: Any | None = None,
    *,
    config: KLTConfig | Mapping[str, Any] | None = None,
) -> FrameMatches:
    """Convenience wrapper around :class:`KLTTracker`."""

    return KLTTracker(config).track_pair(
        prev_frame, curr_frame, prev_masks, curr_masks
    )
