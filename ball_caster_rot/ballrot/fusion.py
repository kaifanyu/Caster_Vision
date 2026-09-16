"""Conservative angle observations for an experimental temporal filter.

The saved visual pose is the only measurement. Its mechanical refit is not
another independent sensor. Noise values here are tunable heuristics, not
calibrated estimates of tracking drift or geometry error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


COORDINATES = ("alpha", "beta_top", "beta_bottom")


@dataclass
class AngleObservations:
    timestamps: np.ndarray
    angles: np.ndarray
    std: np.ndarray
    diagnostics: dict[str, Any]


def _array(frames: Mapping, key: str, count: int, *, boolean=False) -> np.ndarray:
    values = np.asarray(frames[key], dtype=None if boolean else float)
    if values.shape != (count,):
        raise ValueError(f"{key} must have {count} entries")
    if boolean and values.dtype.kind != "b":
        raise ValueError(f"{key} must contain booleans")
    return values


def visual_angle_observations(
    payload: Mapping[str, Any], *, source: str = "unconstrained",
    base_std_deg: float = 1.5, max_tilt_deg: float = 5.0,
    max_roll_disagreement_deg: float = 10.0,
) -> AngleObservations:
    """Read radians, retain validity/gaps, and form ONE common-roll observation.

    Angles retain the source's unwrapped branches, including complete turns.
    Conflicting branches/rolls are rejected rather than selected using the
    filter prediction. Shell spins remain independent observations.
    """
    parameters = np.array([base_std_deg, max_tilt_deg, max_roll_disagreement_deg])
    if not np.all(np.isfinite(parameters)) or np.any(parameters <= 0):
        raise ValueError("measurement noise and quality limits must be positive and finite")
    times = np.asarray(payload["frames"]["time_s"], dtype=float)
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be a nonempty, finite, strictly increasing vector")
    count = len(times)
    angles = np.full((count, 3), np.nan)
    std = np.full_like(angles, np.nan)
    sigma = np.deg2rad(base_std_deg)
    diagnostics: dict[str, Any] = {"measurement_source": source}

    if source == "mechanical":
        if not payload.get("mechanical", {}).get("enabled", False):
            raise ValueError("mechanical measurements require a mechanical results file")
        frames = payload["frames"]
        top = _array(frames, "valid_top", count, boolean=True)
        bottom = _array(frames, "valid_bottom", count, boolean=True)
        for column, (name, valid) in enumerate(zip(COORDINATES, (top | bottom, top, bottom))):
            values = _array(frames, name + "_rad", count)
            if np.any(valid & ~np.isfinite(values)):
                raise ValueError(f"valid {name} measurement is nonfinite")
            angles[valid, column] = values[valid]
            std[valid, column] = sigma
        diagnostics["source_valid_top"] = top.tolist()
        diagnostics["source_valid_bottom"] = bottom.tolist()
    elif source == "unconstrained":
        raw = payload.get("unconstrained")
        if not raw or raw.get("angle_unit") != "radians":
            raise ValueError("unconstrained measurements require saved poses with angle_unit=radians")
        frames = raw["frames"]
        shell_data = []
        for column, shell in enumerate(("top", "bottom"), start=1):
            valid = _array(frames, "valid_" + shell, count, boolean=True)
            alpha = _array(frames, "alpha_" + shell, count)
            beta = _array(frames, "beta_" + shell, count)
            gamma = _array(frames, "gamma_" + shell, count)
            if np.any(valid & ~(np.isfinite(alpha) & np.isfinite(beta) & np.isfinite(gamma))):
                raise ValueError(f"valid {shell} source pose has nonfinite angles")
            usable = valid & (np.abs(gamma) <= np.deg2rad(max_tilt_deg))
            shell_std = np.sqrt(sigma ** 2 + gamma ** 2)
            angles[usable, column] = beta[usable]
            std[usable, column] = shell_std[usable]
            shell_data.append((alpha, gamma, usable))
            diagnostics["source_valid_" + shell] = valid.tolist()
            diagnostics["tilt_rejected_" + shell] = (valid & ~usable).tolist()
        (alpha_t, gamma_t, top), (alpha_b, gamma_b, bottom) = shell_data
        for alpha, gamma, only in ((alpha_t, gamma_t, top & ~bottom), (alpha_b, gamma_b, bottom & ~top)):
            angles[only, 0] = alpha[only]
            std[only, 0] = np.sqrt(sigma ** 2 + gamma[only] ** 2)
        both = top & bottom
        difference = np.where(both, np.abs(alpha_t - alpha_b), np.nan)
        conflict = both & (difference > np.deg2rad(max_roll_disagreement_deg))
        agree = both & ~conflict
        angles[agree, 0] = (alpha_t[agree] + alpha_b[agree]) / 2
        # A second shell does not halve shared geometry/tracking uncertainty.
        std[agree, 0] = np.sqrt(sigma ** 2 + np.maximum(gamma_t[agree] ** 2, gamma_b[agree] ** 2)
                                + (difference[agree] / 2) ** 2)
        diagnostics["roll_conflict"] = conflict.tolist()
        diagnostics["roll_disagreement_rad"] = difference.tolist()
    else:
        raise ValueError("measurement source must be unconstrained or mechanical")
    diagnostics["measurement_count"] = dict(zip(COORDINATES, np.isfinite(angles).sum(axis=0).tolist()))
    return AngleObservations(times, angles, std, diagnostics)
