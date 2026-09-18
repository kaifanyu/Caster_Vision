"""Native-time, camera-local pixel tracks for a joint offline trajectory fit.

No frame pairing or enclosing-sphere rotation fit is performed here. Every
observation retains its camera's original frame index and receive timestamp.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from ballrot.track import KLTConfig, PersistentKLTTracker
from .config import CAMERA_NAMES, jsonable, load_intrinsics
from .session import SelectedVideo, load_session
from .tracking import choose_circle, masks_for


CACHE_VERSION = 1
OBSERVATION_KEYS = ("camera", "shell", "frame", "track", "uv", "weight")


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_path(output):
    path = Path(output).expanduser().resolve()
    return path if path.suffix == ".npz" else path / "native_tracks.npz"


def _provenance(cfg, source, max_frames, tracking):
    """Fingerprint content, not modification times, including segmented AVIs."""
    root = source["path"]
    paths = {root / "session.json"}
    for name, video in zip(CAMERA_NAMES, source["video_paths"]):
        paths.update((video, root / f"{name}_timestamps.csv",
                      Path(cfg["cameras"][name]["intrinsics"])))
        for segment in source["metadata"].get("cameras", {}).get(name, {}).get("video_segments", []):
            filename = segment.get("path")
            if not isinstance(filename, str) or Path(filename).name != filename or "/" in filename or "\\" in filename:
                raise ValueError("Video segment must be a relative basename")
            paths.add(root / filename)
    for key in ("stereo", "axes"):
        if cfg.get(key, {}).get("path"):
            paths.add(Path(cfg[key]["path"]))
    for module in (Path(__file__), Path(__file__).with_name("tracking.py"),
                   Path(__file__).parents[1] / "ballrot" / "track.py",
                   Path(__file__).parents[1] / "ballrot" / "segment.py"):
        paths.add(module)
    return {"cache_version": CACHE_VERSION, "max_frames": max_frames,
            "configuration": jsonable(cfg), "tracking": asdict(tracking),
            "file_sha256": {str(p.resolve()): _file_hash(p) for p in sorted(paths)}}


def save_native_tracks(path, tracked):
    """Save numeric arrays and JSON metadata; loading never requires pickle."""
    path = _cache_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(tracked["observations"])
    for ci in range(2):
        arrays[f"times_{ci}"] = tracked["times"][ci]
        arrays[f"raw_times_{ci}"] = tracked["raw_times"][ci]
        arrays[f"initial_frames_{ci}"] = np.asarray(tracked["initial_frames"][ci], dtype=np.uint8)
    arrays["circles"] = tracked["circles"]
    metadata = {key: value for key, value in tracked.items()
                if key not in ("observations", "times", "raw_times", "circles", "initial_frames")}
    arrays["metadata_json"] = np.asarray(json.dumps(jsonable(metadata), sort_keys=True, allow_nan=False))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    return path


def load_native_tracks(path, *, expected_provenance=None):
    """Load a cache, rejecting stale provenance when supplied by the caller."""
    with np.load(_cache_path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if expected_provenance is not None and metadata.get("provenance") != expected_provenance:
            raise ValueError("Native observation cache provenance differs from current inputs")
        if metadata.get("provenance", {}).get("cache_version") != CACHE_VERSION:
            raise ValueError("Unsupported native observation cache version")
        result = {**metadata,
                  "observations": {key: data[key].copy() for key in OBSERVATION_KEYS},
                  "times": [data[f"times_{ci}"].copy() for ci in range(2)],
                  "raw_times": [data[f"raw_times_{ci}"].copy() for ci in range(2)],
                  "initial_frames": [data[f"initial_frames_{ci}"].copy() for ci in range(2)],
                  "circles": data["circles"].copy()}
    result["videos"] = [Path(p) for p in result["videos"]]
    return result


def collect_native_tracks(cfg, session, output=None, max_frames=None, progress=print):
    """Track both shells in all native frames, optionally reusing a checked cache.

    Frame indices in ``observations`` index the corresponding camera's ``times``
    array. The configured Brio offset is added before subtracting a COMMON
    origin; independent subtraction of each camera's first timestamp would
    silently discard its measured initial delay.
    """
    source = load_session(session, cfg, max_frames=max_frames, native=True)
    infos = [load_intrinsics(cfg["cameras"][name]) for name in CAMERA_NAMES]
    options = dict(cfg.get("tracking", {}))
    options.setdefault("max_corners", 100)
    tracking = KLTConfig.from_mapping(options)
    provenance = _provenance(cfg, source, max_frames, tracking)
    if output is not None and _cache_path(output).is_file():
        try:
            cached = load_native_tracks(output, expected_provenance=provenance)
        except ValueError:
            if progress:
                progress("Native track cache is stale; extracting current inputs.")
        else:
            if progress:
                progress(f"Loaded native tracks: {_cache_path(output)}")
            return cached

    raw_times = [np.asarray(t, dtype=float).copy() for t in source["raw_times"]]
    if any(len(t) < 3 for t in raw_times):
        raise ValueError("Each camera needs at least three timestamped frames")
    times = [t.copy() for t in raw_times]
    times[1] += float(cfg["timing"].get("brio_offset_s", 0.))
    if max(t[0] for t in times) >= min(t[-1] for t in times):
        raise ValueError("Camera recordings do not overlap on the corrected clock")
    origin = float(min(t[0] for t in times))
    times = [t - origin for t in times]
    rows, circles, initial_frames, summaries = [], [], [], []

    for ci, name in enumerate(CAMERA_NAMES):
        info = infos[ci]
        video = SelectedVideo(source["video_paths"][ci], info["image_size"])
        mapx, mapy = cv2.initUndistortRectifyMap(info["K"], info["dist"], None, info["K"],
                                              tuple(info["image_size"]), cv2.CV_32FC1)
        tracker = PersistentKLTTracker(tracking)
        previous = previous_masks = None
        collected, first_frames = {}, []
        support = np.zeros((len(times[ci]), 2), dtype=int)
        try:
            for fi in range(len(times[ci])):
                frame = cv2.remap(video.read(fi), mapx, mapy, cv2.INTER_LINEAR)
                if fi < 3:
                    first_frames.append(frame.copy())
                if previous is None:
                    circle = choose_circle(frame, cfg["cameras"][name])
                    circles.append(circle)
                masks = masks_for(frame, circle, cfg["cameras"][name])
                if previous is None:
                    tracker.initialize(frame, masks)
                else:
                    matches = tracker.track_pair(previous, frame, previous_masks, masks)
                    for shell, label in enumerate(("top", "bottom")):
                        match = matches[label]
                        support[fi, shell] = match.count
                        if not match.count:
                            continue
                        for identifier, uv0, uv1, error in zip(match.track_ids, match.uv_prev,
                                                              match.uv_curr, match.fb_error):
                            weight = 1. / np.sqrt(1. + float(error) ** 2)
                            for endpoint, uv in ((fi - 1, uv0), (fi, uv1)):
                                key = (shell, endpoint, int(identifier))
                                # A middle endpoint occurs in two adjacent pairs.
                                # Keep one pixel with the more conservative weight.
                                if key in collected:
                                    weight_at_endpoint = min(weight, collected[key][-1])
                                else:
                                    weight_at_endpoint = weight
                                collected[key] = (ci, shell, endpoint, int(identifier), uv.copy(), weight_at_endpoint)
                previous, previous_masks = frame, masks
                if progress and ((fi + 1) % 60 == 0 or fi + 1 == len(times[ci])):
                    progress(f"{name}: native pixel tracks {fi + 1}/{len(times[ci])}")
        finally:
            video.close()
        rows.extend(collected.values())
        initial_frames.append(np.asarray(first_frames))
        summaries.append({"camera": name, "frames": len(times[ci]), "observations": len(collected),
                          "tracks_per_shell": [len({k[2] for k in collected if k[0] == h}) for h in range(2)],
                          "median_pair_support_per_shell": np.median(support[1:], axis=0).tolist(),
                          "pairs_below_eight_tracks_per_shell": (support[1:] < 8).sum(axis=0).tolist()})

    if not rows:
        raise ValueError("No surviving native tracks; review circles, masks, focus and exposure")
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    observations = {key: np.asarray(values, dtype=np.int64 if key in OBSERVATION_KEYS[:4] else float)
                    for key, values in zip(OBSERVATION_KEYS, zip(*rows))}
    result = {"times": times, "raw_times": raw_times, "observations": observations,
              "videos": source["video_paths"], "circles": np.asarray(circles),
              "initial_frames": initial_frames, "timestamp_origin_s": origin,
              "session": str(source["path"]), "session_mode": source["metadata"].get("mode"),
              "metadata": source["metadata"], "provenance": provenance,
              "report": {"per_camera": summaries,
                         "frame_index_convention": "source native frame index within each camera",
                         "timing_policy": "all frames, common origin, configured Brio offset; no pairing discard",
                         "timestamp_source": "host receive time; exposure timing is not independently verified",
                         "tracking_policy": "camera-local persistent KLT, painted-shell masks, forward/backward validation"}}
    if output is not None:
        save_native_tracks(output, result)
    return result
