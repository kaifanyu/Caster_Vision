"""Prepare fair camera subsets without retaining the inactive event timeline."""
from __future__ import annotations

import copy

import numpy as np


def prepare_camera_data(data, camera_ids, knots, q, offset):
    """Return (copied_data, copied_q, fit_offset, clock_shifts).

    ``clock_shifts[ci]`` is the additive change applied to that camera's working
    times; recorded raw timestamps and provenance are never changed. A mono
    Brio fit absorbs the already-estimated offset into its working timeline and
    has no remaining relative-camera timing parameter. Its two shell phases
    are re-gauged at the first active physical frame time, matching the first
    knot of the active-only event table. Roll and the absolute working time
    origin are preserved. Home interpolation follows the initializer's
    ``numpy.interp`` convention, including endpoint clamping.
    """
    cameras = tuple(camera_ids)
    if not cameras or len(set(cameras)) != len(cameras) or any(ci not in (0, 1) for ci in cameras):
        raise ValueError("camera_ids must select camera 0, camera 1, or both exactly once")
    knots = np.asarray(knots, dtype=float)
    angles = np.array(q, dtype=float, copy=True)
    if knots.ndim != 1 or len(knots) < 2 or angles.shape != (len(knots), 3):
        raise ValueError("Need increasing knots and q with shape [n_knots,3]")
    if not np.isfinite(knots).all() or np.any(np.diff(knots) <= 0) or not np.isfinite(angles).all():
        raise ValueError("Knots and initial angles must be finite; knots must increase")
    if not np.isfinite(offset):
        raise ValueError("offset must be finite")
    working = copy.deepcopy(data)
    if len(working["times"]) != 2:
        raise ValueError("Expected two camera timestamp arrays")
    for ci in cameras:
        times = np.asarray(working["times"][ci], dtype=float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"Active camera {ci} needs nonempty, finite, increasing timestamps")
    obs = working["observations"]
    keep = np.isin(obs["camera"], cameras)
    if not keep.any():
        raise ValueError("Selected cameras have no observations")
    for ci in cameras:
        frame = np.asarray(obs["frame"])[keep & (np.asarray(obs["camera"]) == ci)]
        if np.any(frame != np.floor(frame)) or np.any(frame < 0) or np.any(frame >= len(working["times"][ci])):
            raise ValueError(f"Camera {ci} observation frame is outside its active timeline")
    working["observations"] = {key: np.asarray(value)[keep].copy() for key, value in obs.items()}
    clock_shifts = np.zeros(2, dtype=float)
    if len(cameras) == 2:
        return working, angles, float(offset), clock_shifts
    active = cameras[0]
    # Do not compact frame indices: every retained observation continues to
    # index its original camera's complete source-frame timestamp array.
    working["times"] = [np.asarray(t, dtype=float).copy() if ci == active else np.empty(0, dtype=float)
                        for ci, t in enumerate(working["times"])]
    if active == 1:
        clock_shifts[1] = float(offset)
        working["times"][1] += float(offset)
    first_physical_time = working["times"][active][0]
    for spin in (1, 2):
        angles[:, spin] -= np.interp(first_physical_time, knots, angles[:, spin])
    return working, angles, 0., clock_shifts
