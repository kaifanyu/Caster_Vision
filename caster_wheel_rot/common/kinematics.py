"""Angular-velocity splitting and car-track interpolation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.signal import correlate, correlation_lags, savgol_filter
from scipy.spatial.transform import Rotation


def split_omega(
    R_incr: np.ndarray,
    dt: float,
    z: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> tuple[np.ndarray, float, float]:
    """Split an active incremental rotation into horizontal and vertical parts."""

    matrix = np.asarray(R_incr, dtype=float)
    vertical = np.asarray(z, dtype=float).reshape(3)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("R_incr must be a finite 3x3 matrix")
    if not np.isfinite(dt) or float(dt) <= 0.0:
        raise ValueError("dt must be positive and finite")
    norm_z = float(np.linalg.norm(vertical))
    if norm_z <= 0.0:
        raise ValueError("z must be non-zero")
    vertical /= norm_z
    omega = Rotation.from_matrix(matrix).as_rotvec() / float(dt)
    omega_spin = float(omega @ vertical)
    horizontal = omega - omega_spin * vertical
    magnitude = float(np.linalg.norm(horizontal))
    axis = horizontal / magnitude if magnitude > 1e-9 else np.array([1.0, 0.0, 0.0])
    return axis, magnitude, omega_spin


def contact_velocity_car(
    v_car_world: np.ndarray,
    omega_car: float,
    theta_car: float,
    r_arm_car: np.ndarray,
) -> np.ndarray:
    """Transfer planar chassis velocity to a fixed caster reference point."""

    velocity = np.asarray(v_car_world, dtype=float).reshape(2)
    arm = np.asarray(r_arm_car, dtype=float).reshape(2)
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(arm)):
        raise ValueError("velocity and r_arm_car must be finite")
    if not np.isfinite(omega_car) or not np.isfinite(theta_car):
        raise ValueError("omega_car and theta_car must be finite")
    c, s = np.cos(theta_car), np.sin(theta_car)
    car_to_world = np.array([[c, -s], [s, c]])
    r_world = car_to_world @ arm
    v_contact_world = velocity + float(omega_car) * np.array([-r_world[1], r_world[0]])
    return car_to_world.T @ v_contact_world


@dataclass(frozen=True)
class CarTrack:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    vx_world: np.ndarray
    vy_world: np.ndarray
    omega: np.ndarray

    def __post_init__(self) -> None:
        arrays = [np.asarray(getattr(self, name), dtype=float).reshape(-1) for name in (
            "t", "x", "y", "theta", "vx_world", "vy_world", "omega"
        )]
        length = len(arrays[0])
        if length < 2 or any(len(array) != length for array in arrays):
            raise ValueError("all car-track arrays must share length >= 2")
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("car-track arrays must be finite")
        if np.any(np.diff(arrays[0]) <= 0.0):
            raise ValueError("car-track timestamps must be strictly increasing")
        for name, array in zip(("t", "x", "y", "theta", "vx_world", "vy_world", "omega"), arrays):
            object.__setattr__(self, name, array)

    def interpolate(self, timestamps: Iterable[float]) -> dict[str, np.ndarray]:
        query = np.asarray(list(timestamps), dtype=float)
        if query.ndim != 1 or not np.all(np.isfinite(query)):
            raise ValueError("timestamps must be a finite 1-D sequence")
        result = {
            name: np.interp(query, self.t, getattr(self, name))
            for name in ("x", "y", "theta", "vx_world", "vy_world", "omega")
        }
        result["t"] = query
        result["in_range"] = (query >= self.t[0]) & (query <= self.t[-1])
        return result


@dataclass(frozen=True)
class SyncEstimate:
    """Clock offset estimate; add ``offset_s`` to raw caster timestamps."""

    offset_s: float
    peak_correlation: float
    ambiguity_ratio: float
    overlap_s: float


def estimate_clock_offset(
    caster_t: Iterable[float],
    caster_signal: Iterable[float],
    car_t: Iterable[float],
    car_signal: Iterable[float],
    *,
    max_offset_s: float = 1.0,
) -> SyncEstimate:
    """Estimate a timestamp offset from the same event observed by both sensors.

    This must be a genuinely common signal/event.  Using physical caster
    response against chassis demand conflates clock offset with mechanical lag.
    Positive output means the raw caster timestamps must be shifted later.
    """

    ct = np.asarray(list(caster_t), dtype=float)
    cs = np.asarray(list(caster_signal), dtype=float)
    rt = np.asarray(list(car_t), dtype=float)
    rs = np.asarray(list(car_signal), dtype=float)
    if len(ct) < 3 or len(rt) < 3 or cs.shape != ct.shape or rs.shape != rt.shape:
        raise ValueError("each timestamp/signal pair must have matching length >= 3")
    if not all(np.all(np.isfinite(value)) for value in (ct, cs, rt, rs)):
        raise ValueError("sync inputs must be finite")
    if np.any(np.diff(ct) <= 0.0) or np.any(np.diff(rt) <= 0.0):
        raise ValueError("sync timestamps must be strictly increasing")
    if not np.isfinite(max_offset_s) or max_offset_s <= 0.0:
        raise ValueError("max_offset_s must be positive")
    dt = float(min(np.median(np.diff(ct)), np.median(np.diff(rt))))
    start = max(ct[0] - max_offset_s, rt[0])
    end = min(ct[-1] + max_offset_s, rt[-1])
    if end - start < 4.0 * dt:
        raise ValueError("insufficient overlapping duration for synchronization")
    grid = np.arange(start, end + 0.25 * dt, dt)
    caster_uniform = np.interp(grid, ct, cs)
    car_uniform = np.interp(grid, rt, rs)
    caster_uniform = caster_uniform - np.mean(caster_uniform)
    car_uniform = car_uniform - np.mean(car_uniform)
    norm = float(np.linalg.norm(caster_uniform) * np.linalg.norm(car_uniform))
    if norm <= np.finfo(float).eps:
        raise ValueError("sync signals must have nonzero variation")
    corr = correlate(car_uniform, caster_uniform, mode="full") / norm
    lags = correlation_lags(len(car_uniform), len(caster_uniform), mode="full")
    allowed = np.abs(lags * dt) <= max_offset_s
    indices = np.flatnonzero(allowed)
    best_index = int(indices[np.argmax(corr[allowed])])
    peak = float(corr[best_index])
    suppressed = corr[allowed].copy()
    local_index = int(np.flatnonzero(indices == best_index)[0])
    guard = max(1, int(round(0.05 / dt)))
    suppressed[max(0, local_index - guard) : local_index + guard + 1] = -np.inf
    second = float(np.max(suppressed)) if np.any(np.isfinite(suppressed)) else 0.0
    ambiguity = peak / max(abs(second), 1e-12)
    return SyncEstimate(
        offset_s=float(lags[best_index] * dt),
        peak_correlation=peak,
        ambiguity_ratio=float(ambiguity),
        overlap_s=float(end - start),
    )


def _odd_window(requested: int, length: int, polyorder: int) -> int | None:
    window = int(requested)
    if window < 3 or length < 3:
        return None
    if window % 2 == 0:
        window += 1
    largest = length if length % 2 == 1 else length - 1
    window = min(window, largest)
    return window if window > polyorder else None


def _column(row: dict[str, str], aliases: tuple[str, ...], *, required: bool) -> float | None:
    lowered = {key.strip().casefold(): value for key, value in row.items() if key is not None}
    for alias in aliases:
        if alias.casefold() in lowered and str(lowered[alias.casefold()]).strip() != "":
            return float(lowered[alias.casefold()])
    if required:
        raise ValueError(f"CSV is missing one of the required columns: {aliases}")
    return None


def load_car_track(
    path: str | Path,
    *,
    time_offset_s: float = 0.0,
    smooth_window: int = 9,
    polyorder: int = 2,
    theta_unit: str = "rad",
) -> CarTrack:
    """Load ``t,x,y,theta`` CSV and derive missing world velocities.

    Accepted optional velocity names include ``vx``/``v_x``/``vx_world``,
    ``vy``/``v_y``/``vy_world``, and ``omega``/``omega_car``/``theta_dot``.
    ``time_offset_s`` is added to the CSV timestamps.
    """

    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("car-track CSV must contain at least two data rows")
    t = np.array([_column(row, ("t", "time", "timestamp", "timestamp_s"), required=True) for row in rows], dtype=float)
    x = np.array([_column(row, ("x", "x_world"), required=True) for row in rows], dtype=float)
    y = np.array([_column(row, ("y", "y_world"), required=True) for row in rows], dtype=float)
    theta = np.array([_column(row, ("theta", "theta_car", "heading", "yaw"), required=True) for row in rows], dtype=float)
    if theta_unit.strip().casefold() in {"deg", "degree", "degrees"}:
        theta = np.deg2rad(theta)
    elif theta_unit.strip().casefold() not in {"rad", "radian", "radians"}:
        raise ValueError("theta_unit must be rad or deg")
    t += float(time_offset_s)
    order = np.argsort(t)
    t, x, y, theta = t[order], x[order], y[order], theta[order]
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("car-track timestamps must be unique")
    theta = np.unwrap(theta)
    window = _odd_window(smooth_window, len(t), polyorder)
    x_s = savgol_filter(x, window, polyorder, mode="interp") if window else x
    y_s = savgol_filter(y, window, polyorder, mode="interp") if window else y
    theta_s = savgol_filter(theta, window, polyorder, mode="interp") if window else theta

    optional = []
    for aliases in (("vx_world", "vx", "v_x"), ("vy_world", "vy", "v_y"), ("omega", "omega_car", "theta_dot", "yaw_rate")):
        values = [_column(row, aliases, required=False) for row in rows]
        optional.append(None if any(value is None for value in values) else np.asarray(values, dtype=float)[order])
    vx = optional[0] if optional[0] is not None else np.gradient(x_s, t, edge_order=2 if len(t) >= 3 else 1)
    vy = optional[1] if optional[1] is not None else np.gradient(y_s, t, edge_order=2 if len(t) >= 3 else 1)
    omega = optional[2] if optional[2] is not None else np.gradient(theta_s, t, edge_order=2 if len(t) >= 3 else 1)
    return CarTrack(t, x_s, y_s, theta_s, vx, vy, omega)
