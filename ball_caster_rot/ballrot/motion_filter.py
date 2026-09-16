"""Conditional angle/rate estimates from independent visual angle measurements.

This filter supplies a motion prediction, not additional image evidence. Angles
arrive already unwrapped; it neither wraps innovations nor establishes a turn
count across an unobserved interval. Reported uncertainty is conditional on the
supplied measurement errors and the configured motion model.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from numbers import Real
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MotionFilterConfig:
    """Angles use degrees here only; filter inputs and outputs use radians.

    Acceleration noise is a standard deviation in degrees/s², treated as a
    constant random acceleration within each sampling interval. It is not a
    continuous-time noise spectral density.
    """

    accel_noise_deg_s2: float = 180.
    initial_velocity_std_deg_s: float = 180.
    innovation_gate_sigma: float = 5.
    max_prediction_s: float = .35
    max_angle_std_deg: float = 10.

    def __post_init__(self):
        nonnegative = {"accel_noise_deg_s2", "max_prediction_s"}
        for item in fields(self):
            value = getattr(self, item.name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or value < 0
                    or (item.name not in nonnegative and value == 0)):
                relation = "nonnegative" if item.name in nonnegative else "positive"
                raise ValueError(f"motion_filter.{item.name} must be finite and {relation}")
            object.__setattr__(self, item.name, float(value))

    @classmethod
    def from_mapping(cls, values=None):
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise ValueError("motion filter settings must be a mapping")
        unknown = set(values)-{item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown motion filter settings: {sorted(map(str, unknown))}")
        return cls(**values)


@dataclass
class MotionFilterResult:
    """Per-component estimates, with no transfer of evidence between components.

    ``angles`` and ``velocities`` are NaN when status is unresolved. After
    initialization, ``predicted_angles`` retains the prior angle even when it
    is too stale to render. ``std`` and ``covariance`` retain posterior/internal
    uncertainty at such times. Before initialization they are NaN, as is age.
    Covariance orders its state as [angle, angular velocity]. ``updated`` marks
    accepted visual measurements; ``reinitialized`` marks a reset after a stale
    time gap and explicitly starts a new velocity segment.
    """

    angles: np.ndarray
    velocities: np.ndarray
    std: np.ndarray
    covariance: np.ndarray
    predicted_angles: np.ndarray
    updated: np.ndarray
    rejected: np.ndarray
    status: np.ndarray
    age_s: np.ndarray
    reinitialized: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


def filter_angles(timestamps, measurements, measurement_std, config=None):
    """Filter N×D unwrapped angle measurements using actual timestamp intervals.

    NaN measurements are absent. Their standard deviations are ignored; each
    present measurement requires a finite positive standard deviation in
    radians. Supplied finite measurements are assumed to have passed the
    caller's image-quality checks. A stale filter may accept one as a new
    segment without claiming its old prediction or gap turn count was correct.
    """
    cfg = config if isinstance(config, MotionFilterConfig) else MotionFilterConfig.from_mapping(config)
    times = np.asarray(timestamps, dtype=float)
    observed = np.asarray(measurements, dtype=float)
    errors = np.asarray(measurement_std, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("timestamps must be a finite one-dimensional array")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if observed.ndim != 2 or observed.shape[0] != len(times) or observed.shape[1] == 0:
        raise ValueError("measurements must have shape (N, D), with D >= 1")
    if errors.shape != observed.shape:
        raise ValueError("measurement_std must have the same (N, D) shape as measurements")
    if np.any(np.isinf(observed)):
        raise ValueError("measurements must contain finite angles or NaN for absent observations")
    present = np.isfinite(observed)
    if np.any(~np.isfinite(errors[present])) or np.any(errors[present] <= 0):
        raise ValueError("present measurements require finite positive standard deviations")

    n, dimension = observed.shape
    angles = np.full((n, dimension), np.nan)
    velocities = np.full_like(angles, np.nan)
    uncertainty = np.full_like(angles, np.nan)
    prior_angles = np.full_like(angles, np.nan)
    age_s = np.full_like(angles, np.nan)
    covariance = np.full((n, dimension, 2, 2), np.nan)
    updated = np.zeros((n, dimension), bool)
    rejected = np.zeros_like(updated)
    reinitialized = np.zeros_like(updated)
    status = np.full((n, dimension), "unresolved", dtype="U10")

    accel_variance = np.deg2rad(cfg.accel_noise_deg_s2)**2
    initial_velocity_variance = np.deg2rad(cfg.initial_velocity_std_deg_s)**2
    std_limit = np.deg2rad(cfg.max_angle_std_deg)
    state = np.zeros((dimension, 2))
    state_covariance = np.zeros((dimension, 2, 2))
    initialized = np.zeros(dimension, bool)
    last_observed = np.full(dimension, np.nan)
    initialization_counts = np.zeros(dimension, int)
    for index, time in enumerate(times):
        dt = float(time-times[index-1]) if index else 0.
        transition = np.array([[1., dt], [0., 1.]])
        acceleration_effect = np.array([.5*dt*dt, dt])
        process_covariance = accel_variance*np.outer(acceleration_effect, acceleration_effect)
        for component in range(dimension):
            stale = False
            if initialized[component]:
                state[component] = transition @ state[component]
                state_covariance[component] = (
                    transition @ state_covariance[component] @ transition.T + process_covariance)
                prior_angles[index, component] = state[component, 0]
                # Rendering uncertainty is not a reset condition: a fresh
                # visual update can reduce a broad prior covariance. Resetting
                # here on prior std would prevent learning velocity at lower
                # sample rates even with uninterrupted good observations.
                stale = time-last_observed[component] > cfg.max_prediction_s

            if present[index, component]:
                variance = errors[index, component]**2
                if not initialized[component] or stale:
                    reinitialized[index, component] = initialized[component]
                    initialization_counts[component] += 1
                    state[component] = (observed[index, component], 0.)
                    state_covariance[component] = np.diag((variance, initial_velocity_variance))
                    initialized[component] = True
                    updated[index, component] = True
                else:
                    innovation = observed[index, component]-state[component, 0]
                    innovation_variance = state_covariance[component, 0, 0]+variance
                    if abs(innovation) > cfg.innovation_gate_sigma*np.sqrt(innovation_variance):
                        rejected[index, component] = True
                    else:
                        gain = state_covariance[component, :, 0]/innovation_variance
                        state[component] += gain*innovation
                        residual_transform = np.eye(2)
                        residual_transform[:, 0] -= gain
                        state_covariance[component] = (
                            residual_transform @ state_covariance[component] @ residual_transform.T
                            + variance*np.outer(gain, gain))
                        updated[index, component] = True
                if updated[index, component]:
                    last_observed[component] = time

            if not initialized[component]:
                continue
            state_covariance[component] = .5*(state_covariance[component]+state_covariance[component].T)
            if (not np.all(np.isfinite(state[component]))
                    or not np.all(np.isfinite(state_covariance[component]))):
                raise ValueError("motion filter overflowed; check timestamp, angle and uncertainty scales")
            covariance[index, component] = state_covariance[component]
            uncertainty[index, component] = np.sqrt(max(0., state_covariance[component, 0, 0]))
            age_s[index, component] = float(time-last_observed[component])
            renderable = (age_s[index, component] <= cfg.max_prediction_s
                          and uncertainty[index, component] <= std_limit)
            if renderable:
                angles[index, component], velocities[index, component] = state[component]
                status[index, component] = "vision" if updated[index, component] else "predicted"

    diagnostics = {
        "method": "independent_angle_velocity_kalman_filter",
        "config": asdict(cfg), "angle_unit": "radians", "velocity_unit": "radians_per_second",
        "process_noise": "independent constant random acceleration within each timestamp interval",
        "uncertainty_note": "Conditional, tuned uncertainty; not a calibrated absolute accuracy guarantee.",
        "reinitialization_note": "Stale reacquisition resets velocity; turns inside an unobserved gap remain unverified.",
        "accepted_updates": updated.sum(axis=0).astype(int).tolist(),
        "rejected_measurements": rejected.sum(axis=0).astype(int).tolist(),
        "initializations": initialization_counts.astype(int).tolist(),
        "reinitializations": reinitialized.sum(axis=0).astype(int).tolist(),
    }
    return MotionFilterResult(angles, velocities, uncertainty, covariance, prior_angles,
                              updated, rejected, status, age_s, reinitialized, diagnostics)


__all__ = ["MotionFilterConfig", "MotionFilterResult", "filter_angles"]
