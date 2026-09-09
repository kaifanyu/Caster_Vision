"""Device-independent caster kinematics and comparison metrics.

The only device-facing input is :class:`common.caster_frame.CasterFrame`.
Adapters are responsible for calibrating the sign of ``roll_axis_car`` so
that ``cross(+z, roll_axis_car)`` points along positive rolling travel, as
specified by the project convention.  An adapter may provide
``raw['rolling_heading_car']`` when that heading remains observable while the
roll rate is zero or temporarily unavailable.

All summary integrals are *gap safe*: an interval contributes only when both
of its endpoints are valid and its duration is not an acquisition gap.  This
avoids silently integrating across a masked roll dropout or missing frames.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import detrend, savgol_filter, welch

from common.caster_frame import CasterFrame
from common.kinematics import CarTrack, contact_velocity_car


def signed_angle(a: np.ndarray, b: np.ndarray) -> float | np.ndarray:
    """Return the signed planar angle from ``a`` to ``b`` in radians.

    The leading dimensions are broadcast by NumPy.  Metrics code masks
    zero-speed vectors before calling this helper; as with ``atan2(0, 0)``, a
    direct call with two zero vectors returns zero.
    """

    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    if first.shape[-1:] != (2,) or second.shape[-1:] != (2,):
        raise ValueError("a and b must end in shape (2,)")
    cross = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    dot = first[..., 0] * second[..., 0] + first[..., 1] * second[..., 1]
    result = np.arctan2(cross, dot)
    return float(result) if np.ndim(result) == 0 else result


@dataclass(frozen=True)
class MetricConfig:
    """Numerical gates and analysis settings for :func:`compute_metrics`."""

    min_contact_speed_mps: float = 1e-4
    min_longitudinal_speed_mps: float = 1e-4
    slip_epsilon_mps: float = 1e-6
    min_confidence: float = 0.0
    max_gap_factor: float = 2.5
    max_gap_s: float | None = None
    lag_max_s: float = 1.0
    lag_min_samples: int = 12
    settling_threshold_deg: float = 5.0
    settling_dwell_s: float = 0.25
    step_time_s: float | None = None
    shimmy_min_frequency_hz: float = 0.25
    shimmy_max_frequency_hz: float | None = None
    shimmy_nperseg: int = 256
    shimmy_min_samples: int = 16
    shimmy_start_s: float | None = None
    shimmy_end_s: float | None = None

    def __post_init__(self) -> None:
        positive = (
            "min_contact_speed_mps",
            "min_longitudinal_speed_mps",
            "slip_epsilon_mps",
            "max_gap_factor",
            "lag_max_s",
            "settling_threshold_deg",
            "settling_dwell_s",
            "shimmy_min_frequency_hz",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_gap_factor <= 0.0:
            raise ValueError("max_gap_factor must be positive")
        if self.max_gap_s is not None and (
            not np.isfinite(self.max_gap_s) or self.max_gap_s <= 0.0
        ):
            raise ValueError("max_gap_s must be positive and finite when set")
        if not np.isfinite(self.min_confidence) or not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must lie in [0, 1]")
        if self.lag_min_samples < 4:
            raise ValueError("lag_min_samples must be at least four")
        if self.shimmy_min_samples < 4:
            raise ValueError("shimmy_min_samples must be at least four")
        if self.shimmy_nperseg < 4:
            raise ValueError("shimmy_nperseg must be at least four")
        if self.shimmy_max_frequency_hz is not None and (
            not np.isfinite(self.shimmy_max_frequency_hz)
            or self.shimmy_max_frequency_hz <= self.shimmy_min_frequency_hz
        ):
            raise ValueError(
                "shimmy_max_frequency_hz must be finite and above shimmy_min_frequency_hz"
            )
        for name in ("step_time_s", "shimmy_start_s", "shimmy_end_s"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when set")
        if (
            self.shimmy_start_s is not None
            and self.shimmy_end_s is not None
            and self.shimmy_end_s <= self.shimmy_start_s
        ):
            raise ValueError("shimmy_end_s must be later than shimmy_start_s")

    @classmethod
    def from_value(cls, value: "MetricConfig | Mapping[str, Any] | None") -> "MetricConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("config must be MetricConfig, a mapping, or None")
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in allowed})


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json_array(value: np.ndarray) -> list[Any]:
    array = np.asarray(value)
    if array.ndim == 0:
        return [_json_scalar(array.item())]
    if array.ndim == 1:
        return [_json_scalar(item) for item in array]
    return [_json_array(item) for item in array]


@dataclass(frozen=True)
class MetricSeries:
    """Per-sample values and explicit validity masks."""

    t: np.ndarray
    rolling_heading_car: np.ndarray
    heading_angle_rad: np.ndarray
    demand_heading_angle_rad: np.ndarray
    contact_velocity_car: np.ndarray
    contact_speed_mps: np.ndarray
    v_long_mps: np.ndarray
    v_lat_car_mps: np.ndarray
    scrub_speed_mps: np.ndarray
    v_roll_mps: np.ndarray
    alignment_error_rad: np.ndarray
    slip_ratio: np.ndarray
    confidence: np.ndarray
    track_valid: np.ndarray
    heading_valid: np.ndarray
    roll_valid: np.ndarray
    alignment_valid: np.ndarray
    slip_valid: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            item.name: _json_array(np.asarray(getattr(self, item.name)))
            for item in fields(self)
        }


@dataclass(frozen=True)
class MetricSummary:
    """Scalar comparison results.

    ``rolling_efficiency`` is the independent longitudinal-distance fraction
    ``integral(abs(v_long)) / integral(abs(v_contact))``.  It is deliberately
    not defined as ``1 - scrub_fraction`` because Euclidean longitudinal and
    lateral components do not add that way.
    """

    mean_abs_alignment_rad: float
    mean_abs_alignment_deg: float
    total_scrub_m: float
    scrub_fraction: float
    rolling_efficiency: float
    mean_slip: float
    mean_abs_slip: float
    swivel_lag_s: float
    settling_time_s: float
    shimmy_frequency_hz: float
    shimmy_amplitude_deg: float
    sample_count: int
    track_valid_count: int
    heading_valid_count: int
    roll_valid_count: int
    alignment_valid_count: int
    slip_valid_count: int
    alignment_valid_duration_s: float
    scrub_valid_duration_s: float
    slip_valid_duration_s: float

    def to_dict(self) -> dict[str, Any]:
        return {key: _json_scalar(value) for key, value in asdict(self).items()}

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass(frozen=True)
class MetricResult:
    series: MetricSeries
    summary: MetricSummary

    def to_dict(self) -> dict[str, Any]:
        return {"series": self.series.to_dict(), "summary": self.summary.to_dict()}


def _vector(raw: Mapping[str, Any], key: str, length: int) -> np.ndarray | None:
    if key not in raw:
        return None
    vector = np.asarray(raw[key], dtype=float).reshape(-1)
    allowed_shapes = {(length,)} | ({(3,)} if length == 2 else set())
    if vector.shape not in allowed_shapes:
        raise ValueError(f"raw[{key!r}] must contain a {length}-vector")
    if length == 2 and vector.shape == (3,):
        vector = vector[:2]
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"raw[{key!r}] must contain a finite {length}-vector")
    return vector


def _heading(frame: CasterFrame) -> np.ndarray:
    override = _vector(frame.raw, "rolling_heading_car", 2)
    if override is None:
        # Required project convention.  Adapters calibrate the directed axis
        # sign on a known forward run.
        override = np.cross(np.array([0.0, 0.0, 1.0]), frame.roll_axis_car)[:2]
    magnitude = float(np.linalg.norm(override))
    if magnitude <= 1e-12:
        raise ValueError("rolling heading must be non-zero")
    return override / magnitude


def _max_gap(t: np.ndarray, config: MetricConfig) -> float:
    if config.max_gap_s is not None:
        return float(config.max_gap_s)
    steps = np.diff(t)
    return float(config.max_gap_factor * np.median(steps))


def _integral(
    t: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    max_gap_s: float,
) -> float:
    """Trapezoidal integral without joining disconnected valid samples."""

    if len(t) < 2:
        return 0.0
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    dt = np.diff(t)
    pairs = valid[:-1] & valid[1:] & (dt > 0.0) & (dt <= max_gap_s)
    if not np.any(pairs):
        return 0.0
    trapezoids = 0.5 * (values[:-1] + values[1:]) * dt
    return float(np.sum(trapezoids[pairs]))


def _duration(t: np.ndarray, valid: np.ndarray, max_gap_s: float) -> float:
    return _integral(t, np.ones(len(t), dtype=float), valid, max_gap_s)


def _weighted_mean(
    t: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    max_gap_s: float,
) -> float:
    duration = _duration(t, valid, max_gap_s)
    if duration <= 0.0:
        return float("nan")
    return _integral(t, values, valid, max_gap_s) / duration


def _runs(t: np.ndarray, valid: np.ndarray, max_gap_s: float) -> list[np.ndarray]:
    indices = np.flatnonzero(valid)
    if not len(indices):
        return []
    output: list[np.ndarray] = []
    start = 0
    for local in range(1, len(indices)):
        previous, current = indices[local - 1], indices[local]
        if current != previous + 1 or t[current] - t[previous] > max_gap_s:
            output.append(indices[start:local])
            start = local
    output.append(indices[start:])
    return output


def _longest_run(t: np.ndarray, valid: np.ndarray, max_gap_s: float) -> np.ndarray:
    runs = _runs(t, valid, max_gap_s)
    if not runs:
        return np.empty(0, dtype=int)
    return max(runs, key=lambda item: (t[item[-1]] - t[item[0]], len(item)))


def _smooth_angle_rate(angle: np.ndarray, t: np.ndarray) -> np.ndarray:
    unwrapped = np.unwrap(np.asarray(angle, dtype=float))
    count = len(unwrapped)
    if count >= 7:
        window = min(9, count if count % 2 else count - 1)
        if window >= 5:
            unwrapped = savgol_filter(unwrapped, window, 2, mode="interp")
    return np.gradient(unwrapped, t, edge_order=2 if count >= 3 else 1)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    first = first - np.mean(first)
    second = second - np.mean(second)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(first @ second / denominator) if denominator > 1e-15 else float("nan")


def _response_lag(
    t: np.ndarray,
    demand_angle: np.ndarray,
    response_angle: np.ndarray,
    valid: np.ndarray,
    max_gap_s: float,
    config: MetricConfig,
) -> float:
    run = _longest_run(t, valid, max_gap_s)
    if len(run) < config.lag_min_samples:
        return float("nan")
    sample_t = t[run]
    dt = float(np.median(np.diff(sample_t)))
    if not np.isfinite(dt) or dt <= 0.0:
        return float("nan")
    uniform_t = np.arange(sample_t[0], sample_t[-1] + 0.25 * dt, dt)
    if len(uniform_t) < config.lag_min_samples:
        return float("nan")
    demand = np.interp(uniform_t, sample_t, np.unwrap(demand_angle[run]))
    response = np.interp(uniform_t, sample_t, np.unwrap(response_angle[run]))
    demand_rate = _smooth_angle_rate(demand, uniform_t)
    response_rate = _smooth_angle_rate(response, uniform_t)
    max_shift = min(int(round(config.lag_max_s / dt)), len(uniform_t) - 4)
    minimum_overlap = min(config.lag_min_samples, len(uniform_t))
    candidates: list[tuple[float, int]] = []
    for shift in range(-max_shift, max_shift + 1):
        if shift > 0:
            first, second = demand_rate[:-shift], response_rate[shift:]
        elif shift < 0:
            first, second = demand_rate[-shift:], response_rate[:shift]
        else:
            first, second = demand_rate, response_rate
        if len(first) < minimum_overlap:
            continue
        score = _correlation(first, second)
        if np.isfinite(score):
            candidates.append((score, shift))
    if not candidates:
        return float("nan")
    _, best_shift = max(candidates, key=lambda item: item[0])
    # Positive means response(t + lag) best matches demand(t): response lags.
    return float(best_shift * dt)


def _settling_time(
    t: np.ndarray,
    alignment: np.ndarray,
    valid: np.ndarray,
    max_gap_s: float,
    config: MetricConfig,
    step_time_s: float | None,
) -> float:
    if step_time_s is None:
        return float("nan")
    threshold = np.deg2rad(config.settling_threshold_deg)
    settled = valid & (t >= step_time_s) & (np.abs(alignment) <= threshold)
    for run in _runs(t, settled, max_gap_s):
        if not len(run):
            continue
        duration = float(t[run[-1]] - t[run[0]])
        if duration + 1e-12 >= config.settling_dwell_s:
            return float(max(0.0, t[run[0]] - step_time_s))
    return float("nan")


def _shimmy(
    t: np.ndarray,
    heading_angle: np.ndarray,
    valid: np.ndarray,
    max_gap_s: float,
    config: MetricConfig,
    steady_mask: np.ndarray | None,
) -> tuple[float, float]:
    selected = valid.copy()
    if config.shimmy_start_s is not None:
        selected &= t >= config.shimmy_start_s
    if config.shimmy_end_s is not None:
        selected &= t <= config.shimmy_end_s
    if steady_mask is not None:
        mask = np.asarray(steady_mask, dtype=bool)
        if mask.shape != t.shape:
            raise ValueError("steady_mask must have one value per CasterFrame")
        selected &= mask
    run = _longest_run(t, selected, max_gap_s)
    if len(run) < config.shimmy_min_samples:
        return float("nan"), float("nan")
    sample_t = t[run]
    dt = float(np.median(np.diff(sample_t)))
    uniform_t = np.arange(sample_t[0], sample_t[-1] + 0.25 * dt, dt)
    if len(uniform_t) < config.shimmy_min_samples:
        return float("nan"), float("nan")
    angle = np.interp(uniform_t, sample_t, np.unwrap(heading_angle[run]))
    angle = detrend(angle, type="linear")
    nperseg = min(config.shimmy_nperseg, len(angle))
    frequencies, density = welch(
        angle,
        fs=1.0 / dt,
        nperseg=nperseg,
        detrend=False,
        scaling="density",
    )
    frequency_mask = frequencies >= config.shimmy_min_frequency_hz
    if config.shimmy_max_frequency_hz is not None:
        frequency_mask &= frequencies <= config.shimmy_max_frequency_hz
    candidates = np.flatnonzero(frequency_mask)
    if not len(candidates):
        return float("nan"), float("nan")
    peak = int(candidates[np.argmax(density[candidates])])
    df = float(frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 0.0
    lo, hi = max(0, peak - 1), min(len(density), peak + 2)
    band_power = float(np.sum(density[lo:hi]) * df)
    amplitude_rad = np.sqrt(max(0.0, 2.0 * band_power))
    return float(frequencies[peak]), float(np.rad2deg(amplitude_rad))


def compute_metrics(
    frames: Sequence[CasterFrame],
    car_track: CarTrack,
    r_arm_car: np.ndarray | Sequence[float] = (0.0, 0.0),
    *,
    config: MetricConfig | Mapping[str, Any] | None = None,
    step_time_s: float | None = None,
    steady_mask: np.ndarray | None = None,
) -> MetricResult:
    """Compute common contact-kinematics metrics for either caster device.

    ``raw['contact_offset_car']`` may override ``r_arm_car`` per sample, and
    ``raw['contact_velocity_rel_car']`` may add motion of that point relative
    to the chassis (for example, the trail term of a swivelling fork).  These
    keys are geometry-neutral; this module never branches on device type.
    """

    cfg = MetricConfig.from_value(config)
    samples = list(frames)
    if len(samples) < 2:
        raise ValueError("at least two CasterFrame samples are required")
    if not all(isinstance(frame, CasterFrame) for frame in samples):
        raise TypeError("frames must contain only CasterFrame values")
    t = np.asarray([frame.t for frame in samples], dtype=float)
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("CasterFrame timestamps must be strictly increasing")
    default_arm = np.asarray(r_arm_car, dtype=float).reshape(-1)
    if default_arm.shape != (2,) or not np.all(np.isfinite(default_arm)):
        raise ValueError("r_arm_car must be a finite 2-vector")

    interpolated = car_track.interpolate(t)
    track_valid = np.asarray(interpolated["in_range"], dtype=bool)
    headings = np.asarray([_heading(frame) for frame in samples])
    heading_angle = np.arctan2(headings[:, 1], headings[:, 0])
    confidence = np.asarray([frame.confidence for frame in samples], dtype=float)
    # Confidence zero always means unusable.  Positive values exactly on a
    # caller-selected threshold remain valid.
    confidence_valid = (confidence > 0.0) & (confidence >= cfg.min_confidence)
    heading_valid = (
        track_valid
        & confidence_valid
        & np.asarray([frame.heading_valid for frame in samples], dtype=bool)
    )
    roll_valid = (
        track_valid
        & confidence_valid
        & np.asarray([frame.roll_valid for frame in samples], dtype=bool)
    )

    contact = np.empty((len(samples), 2), dtype=float)
    for index, frame in enumerate(samples):
        arm = _vector(frame.raw, "contact_offset_car", 2)
        relative = _vector(frame.raw, "contact_velocity_rel_car", 2)
        contact[index] = contact_velocity_car(
            np.array([interpolated["vx_world"][index], interpolated["vy_world"][index]]),
            interpolated["omega"][index],
            interpolated["theta"][index],
            default_arm if arm is None else arm,
        )
        if relative is not None:
            contact[index] += relative

    contact_speed = np.linalg.norm(contact, axis=1)
    v_long = np.einsum("ij,ij->i", contact, headings)
    v_lat = contact - v_long[:, None] * headings
    scrub_speed = np.linalg.norm(v_lat, axis=1)
    v_roll = np.asarray([frame.omega_roll * frame.r_eff for frame in samples])
    demand_heading = np.arctan2(contact[:, 1], contact[:, 0])
    alignment_valid = heading_valid & (contact_speed >= cfg.min_contact_speed_mps)
    alignment = np.full(len(samples), np.nan, dtype=float)
    alignment[alignment_valid] = signed_angle(
        headings[alignment_valid], contact[alignment_valid]
    )
    slip_valid = (
        roll_valid
        & heading_valid
        & (np.abs(v_long) >= cfg.min_longitudinal_speed_mps)
    )
    slip = np.full(len(samples), np.nan, dtype=float)
    denominator = np.maximum(np.abs(v_long[slip_valid]), cfg.slip_epsilon_mps)
    slip[slip_valid] = (v_long[slip_valid] - v_roll[slip_valid]) / denominator

    max_gap_s = _max_gap(t, cfg)
    alignment_duration = _duration(t, alignment_valid, max_gap_s)
    scrub_valid = heading_valid
    scrub_duration = _duration(t, scrub_valid, max_gap_s)
    slip_duration = _duration(t, slip_valid, max_gap_s)
    mean_abs_alignment = _weighted_mean(
        t, np.abs(alignment), alignment_valid, max_gap_s
    )
    total_scrub = (
        _integral(t, scrub_speed, scrub_valid, max_gap_s)
        if scrub_duration > 0.0
        else float("nan")
    )
    contact_distance = (
        _integral(t, contact_speed, scrub_valid, max_gap_s)
        if scrub_duration > 0.0
        else float("nan")
    )
    longitudinal_distance = (
        _integral(t, np.abs(v_long), scrub_valid, max_gap_s)
        if scrub_duration > 0.0
        else float("nan")
    )
    scrub_fraction = (
        total_scrub / contact_distance if contact_distance > 0.0 else float("nan")
    )
    rolling_efficiency = (
        longitudinal_distance / contact_distance
        if contact_distance > 0.0
        else float("nan")
    )
    mean_slip = _weighted_mean(t, slip, slip_valid, max_gap_s)
    mean_abs_slip = _weighted_mean(t, np.abs(slip), slip_valid, max_gap_s)
    lag = _response_lag(
        t,
        demand_heading,
        heading_angle,
        alignment_valid,
        max_gap_s,
        cfg,
    )
    requested_step = cfg.step_time_s if step_time_s is None else step_time_s
    settling = _settling_time(
        t, alignment, alignment_valid, max_gap_s, cfg, requested_step
    )
    shimmy_frequency, shimmy_amplitude = _shimmy(
        t, heading_angle, heading_valid, max_gap_s, cfg, steady_mask
    )

    series = MetricSeries(
        t=t,
        rolling_heading_car=headings,
        heading_angle_rad=heading_angle,
        demand_heading_angle_rad=demand_heading,
        contact_velocity_car=contact,
        contact_speed_mps=contact_speed,
        v_long_mps=v_long,
        v_lat_car_mps=v_lat,
        scrub_speed_mps=scrub_speed,
        v_roll_mps=v_roll,
        alignment_error_rad=alignment,
        slip_ratio=slip,
        confidence=confidence,
        track_valid=track_valid,
        heading_valid=heading_valid,
        roll_valid=roll_valid,
        alignment_valid=alignment_valid,
        slip_valid=slip_valid,
    )
    summary = MetricSummary(
        mean_abs_alignment_rad=mean_abs_alignment,
        mean_abs_alignment_deg=float(np.rad2deg(mean_abs_alignment)),
        total_scrub_m=total_scrub,
        scrub_fraction=scrub_fraction,
        rolling_efficiency=rolling_efficiency,
        mean_slip=mean_slip,
        mean_abs_slip=mean_abs_slip,
        swivel_lag_s=lag,
        settling_time_s=settling,
        shimmy_frequency_hz=shimmy_frequency,
        shimmy_amplitude_deg=shimmy_amplitude,
        sample_count=len(samples),
        track_valid_count=int(np.count_nonzero(track_valid)),
        heading_valid_count=int(np.count_nonzero(heading_valid)),
        roll_valid_count=int(np.count_nonzero(roll_valid)),
        alignment_valid_count=int(np.count_nonzero(alignment_valid)),
        slip_valid_count=int(np.count_nonzero(slip_valid)),
        alignment_valid_duration_s=alignment_duration,
        scrub_valid_duration_s=scrub_duration,
        slip_valid_duration_s=slip_duration,
    )
    return MetricResult(series=series, summary=summary)


__all__ = [
    "MetricConfig",
    "MetricResult",
    "MetricSeries",
    "MetricSummary",
    "compute_metrics",
    "signed_angle",
]
