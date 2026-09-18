"""Receive-time session alignment. This is not hardware synchronization."""
from __future__ import annotations

import csv
import json
from numbers import Integral
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
                       "max_pair_interval_s": float(np.max(np.diff(times))),
                       "pair_gaps_over_100ms": [{"from_s": float(times[i]-times[0]),
                                                "to_s": float(times[i+1]-times[0]),
                                                "duration_s": float(times[i+1]-times[i])}
                                               for i in np.flatnonzero(np.diff(times) > .1001)],
                       "max_pair_skew_ms": float(1000*np.max(np.abs(skew))),
                       "median_pair_skew_ms": float(1000*np.median(np.abs(skew))),
                       "brio_offset_s": timing.get("brio_offset_s", 0.),
                       "timing_verified": bool(timing.get("verified", False)),
                       "timestamp_source": "host receive time; exposure synchronization is not guaranteed"}}


class SelectedVideo:
    """Read global frame indices, following only explicitly declared AVI segments.

    Older sessions without ``video_segments`` remain single-file videos. Segment
    paths come from the adjacent session.json camera entry, never a filename glob.
    """
    def __init__(self, path, image_size=None):
        self.path = Path(path)
        self.index = -1
        self.image_size = image_size
        self.cap = None
        self._segments = self._load_segments()
        self._segment_index = -1
        if self._segments is None:
            self._open(self.path)
        else:
            self._advance_segment()

    def _load_segments(self):
        manifest = self.path.parent / "session.json"
        if self.path.stem not in CAMERA_NAMES or not manifest.is_file():
            return None
        try:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            cameras = metadata.get("cameras", {})
            camera = cameras.get(self.path.stem, {})
            if "video_segments" not in camera:
                return None
            entries = camera["video_segments"]
        except (AttributeError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid video segment metadata in {manifest}") from exc
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{manifest}: video_segments must be a nonempty list")
        segments, seen, next_frame = [], set(), 0
        directory = self.path.parent.resolve()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"{manifest}: each video segment must be an object")
            name = entry.get("path")
            if (not isinstance(name, str) or not name or name in (".", "..")
                    or "/" in name or "\\" in name or Path(name).is_absolute()):
                raise ValueError(f"{manifest}: video segment path must be a relative basename")
            segment_path = self.path.parent / name
            resolved = segment_path.resolve()
            if resolved.parent != directory:
                raise ValueError(f"{manifest}: video segment path escapes the session directory")
            if resolved in seen:
                raise ValueError(f"{manifest}: duplicate video segment path: {name}")
            seen.add(resolved)
            start, count = entry.get("start_frame"), entry.get("frame_count")
            if type(start) is not int or type(count) is not int or start < 0 or count < 0:
                raise ValueError(f"{manifest}: segment start_frame and frame_count must be nonnegative integers")
            if start != next_frame:
                raise ValueError(f"{manifest}: video segment start_frame values must be contiguous from zero")
            if not segments and name != self.path.name:
                raise ValueError(f"{manifest}: first video segment must match requested video {self.path.name}")
            if not segment_path.is_file():
                raise ValueError(f"Missing video segment: {segment_path}")
            segments.append({"path": segment_path, "start_frame": start, "frame_count": count})
            next_frame += count
        self._frame_count = next_frame
        return segments

    def _open(self, path):
        if self.cap is not None:
            self.cap.release()
        # Brio JPEGs contain a short APP0 metadata stub that older FFmpeg
        # versions complain about on every frame. OpenCV's native MJPEG AVI
        # reader accepts it without modifying the JPEGs or suppressing errors.
        # Other codecs/builds retain the usual backend fallback.
        if Path(path).suffix.lower() == ".avi":
            self.cap = cv2.VideoCapture(str(path), cv2.CAP_OPENCV_MJPEG)
            if self.cap.isOpened():
                return
            self.cap.release()
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            raise ValueError(f"Cannot open video: {path}")

    def _advance_segment(self):
        self._segment_index += 1
        while self._segment_index < len(self._segments):
            segment = self._segments[self._segment_index]
            if segment["frame_count"]:
                self._open(segment["path"])
                return
            self._segment_index += 1

    def read(self, index):
        if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
            raise ValueError("Selected video index must be a nonnegative integer")
        if index <= self.index:
            raise ValueError("Selected video indices must strictly increase")
        if self._segments is not None and index >= self._frame_count:
            raise ValueError(f"{self.path}: recorded frame {index} is outside declared video segments")
        frame = None
        while self.index < index:
            source = self.path
            if self._segments is not None:
                segment = self._segments[self._segment_index]
                if self.index + 1 >= segment["start_frame"] + segment["frame_count"]:
                    self._advance_segment()
                    segment = self._segments[self._segment_index]
                source = segment["path"]
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise ValueError(f"{source}: cannot read recorded frame {self.index+1} (premature EOF or decode failure)")
            self.index += 1
        if self.image_size is not None and [frame.shape[1], frame.shape[0]] != list(self.image_size):
            raise ValueError(f"{self.path}: frame shape differs from calibrated image_size")
        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()
