"""Local offline trajectory derivatives using native observation timestamps.

These fits reduce derivative noise without replacing the measured poses. Missing
observations split the trajectory: no rate is inferred through an invalid gap.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _inputs(
    timestamps: np.ndarray,
    count: int,
    valid: np.ndarray | None,
    window_s: float,
    polynomial_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(timestamps, dtype=float)
    if times.shape != (count,):
        raise ValueError("timestamps must have shape (N,) matching the observations")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError("window_s must be positive and finite")
    if (
        isinstance(polynomial_order, (bool, np.bool_))
        or not isinstance(polynomial_order, (int, np.integer))
        or polynomial_order < 1
    ):
        raise ValueError("polynomial_order must be a positive integer")
    usable = np.ones(count, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if usable.shape != (count,):
        raise ValueError("valid must have shape (N,)")
    return times, usable.copy()


def _segments(valid: np.ndarray):
    """Yield half-open bounds of contiguous valid observations."""

    transitions = np.diff(np.r_[False, valid, False].astype(np.int8))
    yield from zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1))


def _support(times: np.ndarray, index: int, window_s: float) -> np.ndarray:
    # Shift the full-width window inward at a segment endpoint. It never reaches
    # across an invalid sample because ``times`` contains only this segment.
    lower = max(float(times[0]), min(float(times[index]) - window_s / 2, float(times[-1]) - window_s))
    upper = lower + window_s
    # Floating-point roundoff should not drop a point exactly at a boundary.
    tolerance = 16 * np.finfo(float).eps * max(1.0, abs(lower), abs(upper))
    return np.flatnonzero((times >= lower - tolerance) & (times <= upper + tolerance))


def _derivative(
    offsets: np.ndarray, values: np.ndarray, window_s: float, polynomial_order: int
) -> np.ndarray | None:
    if len(offsets) < polynomial_order + 1:
        return None
    # Scaling time keeps short/high-rate sequences numerically well conditioned.
    scale = float(np.max(np.abs(offsets)))
    if scale <= 0:
        return None
    normalized = offsets / scale
    design = np.vander(normalized, N=polynomial_order + 1, increasing=True)
    root_weight = np.exp(-0.25 * (offsets / (window_s / 2)) ** 2)
    weighted_design = design * root_weight[:, None]
    weighted_values = values * root_weight[:, None]
    coefficients, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)
    if rank < polynomial_order + 1:
        return None
    return coefficients[1] / scale


def angular_rates(
    angles: np.ndarray,
    timestamps: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    window_s: float = 0.25,
    polynomial_order: int = 2,
) -> np.ndarray:
    """Estimate angle derivatives in rad/s by weighted local polynomial fits.

    ``window_s`` is the complete temporal span, centered where possible and
    shifted inward at segment ends. Angles must already be unwrapped if they
    represent a periodic coordinate. Nonfinite angles and false ``valid`` entries
    break the fit into independent segments. Fewer than ``polynomial_order + 1``
    observations in the local window produce NaN, including isolated samples.
    The function neither fills missing observations nor resamples time.
    """

    values = np.asarray(angles, dtype=float)
    if values.ndim != 1:
        raise ValueError("angles must have shape (N,)")
    times, usable = _inputs(timestamps, len(values), valid, window_s, polynomial_order)
    usable &= np.isfinite(values)
    result = np.full(len(values), np.nan)
    for start, stop in _segments(usable):
        segment_times = times[start:stop]
        segment_values = values[start:stop]
        for local_index in range(stop - start):
            support = _support(segment_times, local_index, window_s)
            derivative = _derivative(
                segment_times[support] - segment_times[local_index],
                segment_values[support, None] - segment_values[local_index],
                window_s,
                polynomial_order,
            )
            if derivative is not None:
                result[start + local_index] = derivative[0]
    return result


def rotation_rates(
    rotations: np.ndarray,
    timestamps: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    window_s: float = 0.25,
    polynomial_order: int = 2,
) -> np.ndarray:
    """Estimate spatial angular velocity in the rotations' fixed frame, rad/s.

    Active matrices map initial surface directions into current camera-frame
    directions. Thus the fitted local coordinate is ``log(R_j @ R_i.T)`` and
    the output is camera-frame angular velocity, not Euler-coordinate rates or
    body-frame velocity. Fitting approximates the trajectory near each pose.

    Window/validity rules match :func:`angular_rates`. Rotational path length
    from the center to a contributing observation is additionally limited to
    90 degrees; distant points cannot wrap back into a small principal logarithm.
    This can leave fast or undersampled motion without enough support (NaN).
    Motion exceeding 180 degrees between observations cannot be disambiguated
    from rotation matrices alone. Invalid observations may contain NaN matrices;
    finite matrices marked valid must be proper rotations.
    """

    matrices = np.asarray(rotations, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape (N, 3, 3)")
    times, usable = _inputs(timestamps, len(matrices), valid, window_s, polynomial_order)
    usable &= np.all(np.isfinite(matrices), axis=(1, 2))
    selected = matrices[usable]
    if len(selected) and (
        not np.allclose(np.swapaxes(selected, 1, 2) @ selected, np.eye(3), atol=1e-6, rtol=0)
        or not np.allclose(np.linalg.det(selected), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError("valid rotations must be proper rotation matrices")
    result = np.full((len(matrices), 3), np.nan)
    for start, stop in _segments(usable):
        segment_times = times[start:stop]
        segment_rotations = matrices[start:stop]
        if stop - start < polynomial_order + 1:
            continue
        steps = segment_rotations[1:] @ np.swapaxes(segment_rotations[:-1], 1, 2)
        path_length = np.r_[0.0, np.cumsum(Rotation.from_matrix(steps).magnitude())]
        for local_index in range(stop - start):
            support = _support(segment_times, local_index, window_s)
            support = support[np.abs(path_length[support] - path_length[local_index]) <= np.pi / 2]
            if len(support) < polynomial_order + 1:
                continue
            relative = segment_rotations[support] @ segment_rotations[local_index].T
            coordinates = Rotation.from_matrix(relative).as_rotvec()
            derivative = _derivative(
                segment_times[support] - segment_times[local_index],
                coordinates,
                window_s,
                polynomial_order,
            )
            if derivative is not None:
                result[start + local_index] = derivative
    return result
