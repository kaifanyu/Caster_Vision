"""Adapt Approach-A ball rotations to the device-independent frame seam."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from common.caster_frame import CasterFrame
from common.kinematics import split_omega
from common.rotation import is_rotation_matrix


_HORIZONTAL_EPS = 1e-9


def _member(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _rotation_matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if not is_rotation_matrix(matrix, atol=1e-6):
        raise ValueError(f"{name} must be a finite, proper 3x3 rotation matrix")
    return matrix


def _quality_confidence(
    result: Any,
    interval_index: int,
    hemisphere: str,
    valid: bool,
) -> float:
    """Return a bounded per-side confidence, with absent diagnostics neutral."""

    if not valid:
        return 0.0
    qualities = _member(result, "qualities", None)
    if qualities is None or interval_index >= len(qualities):
        return 1.0
    item = _member(qualities[interval_index], hemisphere, None)
    if item is None:
        return 1.0
    ratio = _member(item, "inlier_ratio", None)
    if ratio is None:
        return 1.0
    ratio = float(ratio)
    return float(np.clip(ratio, 0.0, 1.0)) if np.isfinite(ratio) else 0.0


def _increment_valid(
    result: Any,
    hemisphere: str,
    interval_index: int,
    increment: Any,
    frame_count: int,
) -> bool:
    """Combine the increment sentinel with Approach-A's optional step mask."""

    if increment is None:
        return False
    mask = _member(result, f"{hemisphere}_step_valid", None)
    if mask is None:
        return True
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if len(values) == frame_count:
        return bool(values[interval_index + 1])
    if len(values) == frame_count - 1:
        return bool(values[interval_index])
    raise ValueError(
        f"result.{hemisphere}_step_valid must have one value per frame or interval"
    )


def _midpoint_motion_value(result: Any, name: str, interval_index: int) -> float | None:
    """Read one unwrapped Approach-A angle at the interval midpoint."""

    motion = _member(result, "motion", None)
    values = _member(motion, name, None) if motion is not None else None
    if values is None:
        return None
    array = np.asarray(values, dtype=float).reshape(-1)
    right = interval_index + 1
    if right >= len(array):
        return None
    pair = array[interval_index : right + 1]
    if len(pair) != 2 or not np.all(np.isfinite(pair)):
        return None
    return float(np.mean(pair))


def _camera_to_car_increment(
    increment_camera: Any,
    R_bc: np.ndarray,
    R_ball_to_car: np.ndarray,
    name: str,
) -> np.ndarray:
    """Conjugate a camera-coordinate increment through ball into car axes."""

    increment = _rotation_matrix(increment_camera, name)
    increment_ball = R_bc.T @ increment @ R_bc
    return R_ball_to_car @ increment_ball @ R_ball_to_car.T


def _nominal_roll_axis(
    R_ball_to_car: np.ndarray, roll_direction_sign: float
) -> np.ndarray:
    axis = float(roll_direction_sign) * (R_ball_to_car @ np.array([1.0, 0.0, 0.0]))
    axis = axis.copy()
    axis[2] = 0.0
    norm = float(np.linalg.norm(axis))
    return axis / norm if norm > _HORIZONTAL_EPS else np.array([1.0, 0.0, 0.0])


def adapt_ball_pipeline(
    result: Any,
    *,
    R_bc: np.ndarray,
    R_ball_to_car: np.ndarray,
    r_eff: float,
    roll_direction_sign: float = -1.0,
) -> list[CasterFrame]:
    """Convert a :class:`ball.pipeline.PipelineResult` to ``CasterFrame`` records.

    One output record represents each source interval and is timestamped at
    its midpoint.  Approach-A increments live in camera coordinates.  ``R_bc``
    maps ball vectors to camera coordinates, while ``R_ball_to_car`` maps ball
    vectors to car coordinates, so an increment is transformed as::

        R_car = R_ball_to_car @ (R_bc.T @ R_camera @ R_bc) @ R_ball_to_car.T

    The horizontal angular-velocity vectors from all valid hemispheres are
    averaged before their magnitude and direction are computed.  A single
    valid hemisphere therefore still supplies a usable rolling estimate; an
    interval with neither side valid is retained with finite zero rates and
    ``raw['roll_valid'] == False`` so downstream metrics can mask it.

    ``roll_direction_sign`` resolves the rolling-axis line ambiguity.  The
    default ``-1`` makes the common ``z × roll_axis`` heading convention agree
    with the documented positive direction for the Approach-A rig.
    """

    ball_to_camera = _rotation_matrix(R_bc, "R_bc")
    ball_to_car = _rotation_matrix(R_ball_to_car, "R_ball_to_car")
    radius = float(r_eff)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("r_eff must be finite and positive")
    direction_sign = float(roll_direction_sign)
    if direction_sign not in (-1.0, 1.0):
        raise ValueError("roll_direction_sign must be +1 or -1")

    timestamps = np.asarray(_member(result, "timestamps_s"), dtype=float)
    if timestamps.ndim != 1 or len(timestamps) < 2:
        raise ValueError("result.timestamps_s must be a one-dimensional array of length >= 2")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("result.timestamps_s must be finite and strictly increasing")

    top_increments = _member(result, "top_increments")
    bottom_increments = _member(result, "bottom_increments")
    if not isinstance(top_increments, Sequence) or not isinstance(
        bottom_increments, Sequence
    ):
        raise TypeError("result increments must be sequences")
    interval_count = len(timestamps) - 1
    if len(top_increments) != interval_count or len(bottom_increments) != interval_count:
        raise ValueError("each increment sequence must have len(timestamps_s) - 1 entries")

    frames: list[CasterFrame] = []
    last_axis = _nominal_roll_axis(ball_to_car, direction_sign)
    for index in range(interval_count):
        start = float(timestamps[index])
        end = float(timestamps[index + 1])
        dt = end - start
        top_valid = _increment_valid(
            result, "top", index, top_increments[index], len(timestamps)
        )
        bottom_valid = _increment_valid(
            result, "bottom", index, bottom_increments[index], len(timestamps)
        )

        horizontal_parts: list[np.ndarray] = []
        spins: list[float] = []
        top_roll_rate: float | None = None
        bottom_roll_rate: float | None = None
        top_spin: float | None = None
        bottom_spin: float | None = None

        if top_valid:
            top_car = _camera_to_car_increment(
                top_increments[index], ball_to_camera, ball_to_car, f"top_increments[{index}]"
            )
            top_axis, top_roll_rate, top_spin = split_omega(top_car, dt)
            horizontal_parts.append(top_axis * top_roll_rate)
            spins.append(top_spin)
        if bottom_valid:
            bottom_car = _camera_to_car_increment(
                bottom_increments[index],
                ball_to_camera,
                ball_to_car,
                f"bottom_increments[{index}]",
            )
            bottom_axis, bottom_roll_rate, bottom_spin = split_omega(bottom_car, dt)
            horizontal_parts.append(bottom_axis * bottom_roll_rate)
            spins.append(bottom_spin)

        roll_valid = bool(horizontal_parts)
        horizontal = (
            direction_sign * np.mean(horizontal_parts, axis=0)
            if horizontal_parts
            else np.zeros(3, dtype=float)
        )
        horizontal[2] = 0.0
        omega_roll = float(np.linalg.norm(horizontal))
        heading_valid = roll_valid and omega_roll > _HORIZONTAL_EPS
        if heading_valid:
            roll_axis = horizontal / omega_roll
            last_axis = roll_axis
        else:
            roll_axis = last_axis.copy()
        omega_spin = float(np.mean(spins)) if spins else 0.0

        top_confidence = _quality_confidence(result, index, "top", top_valid)
        bottom_confidence = _quality_confidence(result, index, "bottom", bottom_valid)
        confidence = 0.5 * (top_confidence + bottom_confidence)
        disagreement = (
            float(np.linalg.norm(horizontal_parts[0] - horizontal_parts[1]))
            if len(horizontal_parts) == 2
            else None
        )
        source = (
            "both"
            if top_valid and bottom_valid
            else "top"
            if top_valid
            else "bottom"
            if bottom_valid
            else "none"
        )
        raw = {
            "device": "ball",
            "interval_index": index,
            "interval_start_s": start,
            "interval_end_s": end,
            "dt_s": dt,
            "alpha": _midpoint_motion_value(result, "alpha", index),
            "beta1": _midpoint_motion_value(result, "beta_top", index),
            "beta2": _midpoint_motion_value(result, "beta_bottom", index),
            "beta1_dot": top_spin,
            "beta2_dot": bottom_spin,
            "top_omega_roll": top_roll_rate,
            "bottom_omega_roll": bottom_roll_rate,
            "top_valid": top_valid,
            "bottom_valid": bottom_valid,
            "roll_valid": roll_valid,
            "heading_valid": heading_valid,
            "confidence": confidence,
            "top_confidence": top_confidence,
            "bottom_confidence": bottom_confidence,
            "hemisphere_source": source,
            "horizontal_omega_disagreement_rad_s": disagreement,
            "omega_spin_policy": "mean_valid_hemisphere_vertical_components",
            "roll_direction_sign": direction_sign,
        }
        frames.append(
            CasterFrame(
                t=0.5 * (start + end),
                roll_axis_car=roll_axis,
                omega_roll=omega_roll,
                omega_spin=omega_spin,
                r_eff=radius,
                raw=raw,
            )
        )
    return frames


__all__ = ["adapt_ball_pipeline"]
