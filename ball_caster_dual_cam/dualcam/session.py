"""Receive-time session alignment. This is not hardware synchronization."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .config import CAMERA_NAMES


def read_timestamps(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty timestamps: {path}")
    indices = np.array([int(r["frame_index"]) for r in rows])
    times = np.array([float(r["timestamp_s"]) for r in rows])
    if not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError(f"{path}: video frame indices must be consecutive from zero")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"{path}: timestamps must be finite and strictly increasing")
    return times


def pair_timestamps(first, second, *, second_offset_s=0., max_skew_s=.012):
    """Nearest, monotone, one-to-one pairs; reject excessive time separation."""
    first, second = np.asarray(first, float), np.asarray(second, float)
    for values in (first, second):
        if values.ndim != 1 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
            raise ValueError("Camera times must be strictly increasing finite arrays")
    if not np.isfinite(second_offset_s) or not np.isfinite(max_skew_s) or max_skew_s <= 0:
        raise ValueError("Time offset must be finite and pairing tolerance positive")
    adjusted = second + second_offset_s
    pairs = []
    last = -1
    for i, stamp in enumerate(first):
        j = int(np.searchsorted(adjusted, stamp))
        candidates = [k for k in (j-1, j) if last < k < len(adjusted)]
        if not candidates:
            continue
        closest = min(candidates, key=lambda k: abs(adjusted[k]-stamp))
        if abs(adjusted[closest]-stamp) <= max_skew_s:
            pairs.append((i, closest))
            last = closest
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if not len(pairs):
        return pairs, np.empty(0), np.empty(0)
    skew = adjusted[pairs[:, 1]] - first[pairs[:, 0]]
    return pairs, (first[pairs[:, 0]]+adjusted[pairs[:, 1]])/2, skew


def load_session(path, cfg, *, max_frames=None, check_profile=True):
    directory = Path(path).expanduser().resolve()
    metadata_path = directory / "session.json"
    if not metadata_path.is_file():
        raise ValueError(f"Missing {metadata_path}; use scripts/record.py to retain capture provenance")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") not in ("complete", "interrupted"):
        raise ValueError(f"Session is not finalized successfully; inspect {metadata_path} before analysis")
    old = metadata.get("config_snapshot", {}).get("cameras", {})
    if check_profile:
        for name in CAMERA_NAMES:
            # Never reinterpret a recorded resolution/focus/exposure as a new one.
            recorded = metadata.get("cameras", {}).get(name, {}).get("requested")
            if not isinstance(recorded, dict):
                recorded = old.get(name, {}).get("capture")
            if recorded is None:
                raise ValueError(f"{name}: session is missing its requested capture profile")
            requested = cfg["cameras"][name]["capture"]
            changed = [k for k in sorted(set(recorded) | set(requested))
                       if k not in recorded or k not in requested or recorded[k] != requested[k]]
            if changed:
                raise ValueError(f"{name}: recording profile differs from current rig: {changed}; use its original config")
    raw = [read_timestamps(directory/f"{name}_timestamps.csv") for name in CAMERA_NAMES]
    timing = cfg["timing"]
    pairs, times, skew = pair_timestamps(*raw, second_offset_s=timing.get("brio_offset_s", 0.),
                                         max_skew_s=timing.get("max_pair_skew_ms", 12.)/1000)
    if max_frames is not None:
        if max_frames < 3:
            raise ValueError("max_frames must be at least 3")
        pairs, times, skew = pairs[:max_frames], times[:max_frames], skew[:max_frames]
    if len(pairs) < 3:
        raise ValueError("Fewer than three synchronized frame pairs; check timestamp offset and tolerance")
    return {"path": directory, "metadata": metadata, "pairs": pairs,
            "times": times-times[0], "skew_s": skew,
            "report": {"paired_frames": len(pairs), "source_frames": [len(t) for t in raw],
                       "max_pair_skew_ms": float(1000*np.max(np.abs(skew))),
                       "median_pair_skew_ms": float(1000*np.median(np.abs(skew))),
                       "brio_offset_s": timing.get("brio_offset_s", 0.),
                       "timing_verified": bool(timing.get("verified", False)),
                       "timestamp_source": "host receive time; exposure synchronization is not guaranteed"}}


class SelectedVideo:
    """Read selected frames sequentially without inaccurate compressed-video seeks."""
    def __init__(self, path, image_size=None):
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        self.index = -1
        self.image_size = image_size
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.path}")

    def read(self, index):
        if index <= self.index:
            raise ValueError("Selected video indices must strictly increase")
        frame = None
        while self.index < index:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise ValueError(f"{self.path}: cannot read recorded frame {self.index+1}")
            self.index += 1
        if self.image_size is not None and [frame.shape[1], frame.shape[0]] != list(self.image_size):
            raise ValueError(f"{self.path}: frame shape differs from calibrated image_size")
        return frame

    def close(self):
        self.cap.release()
