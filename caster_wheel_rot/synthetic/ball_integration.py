"""End-to-end synthetic ball-caster fixtures for the shared comparison seam.

Unlike a hand-built ``CasterFrame`` fixture, :func:`run_ideal_ball_case`
renders a speckled sphere, runs the production ball KLT/sphere/Kabsch
pipeline, and only then invokes :func:`ball.adapter.adapt_ball_pipeline`.
This gives S5/S6 an image-level oracle while keeping the renderer independent
from the estimator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ball.adapter import adapt_ball_pipeline
from ball.pipeline import PipelineResult, run_pipeline
from common.caster_frame import CasterFrame
from .generate_ball import (
    RenderConfig,
    RenderResult,
    Trajectory,
    default_camera,
    render_sequence,
)


SYNTHETIC_BALL_SEGMENT: dict[str, Any] = {
    "mode": "color",
    # The top population is red and deliberately wraps OpenCV hue zero.
    "top_hsv": {"lo": [170, 90, 60], "hi": [10, 255, 255]},
    "bottom_hsv": {"lo": [85, 90, 60], "hi": [105, 255, 255]},
    "yoke_hsv": {
        "lo": [0, 0, 0],
        "hi": [179, 255, 55],
        "enabled": True,
    },
    "morphology_px": 1,
    "ball_margin_px": 1.0,
}


@dataclass(frozen=True)
class SyntheticBallCase:
    """Rendered truth, production result, and adapted common frames."""

    rendered: RenderResult
    pipeline: PipelineResult
    frames: list[CasterFrame]
    contact_velocity_car: np.ndarray
    expected_roll_axis_car: np.ndarray
    expected_omega_roll: float


def _horizontal_unit(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 2-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def run_ideal_ball_case(
    contact_velocity_car: np.ndarray,
    *,
    num_frames: int,
    fps: float,
    r_eff: float,
    track_config: Mapping[str, Any] | None = None,
    estimate_config: Mapping[str, Any] | None = None,
    seed: int = 24_681_357,
    output_dir: str | Path | None = None,
) -> SyntheticBallCase:
    """Measure an ideal no-slip ball response to a constant car-frame demand.

    ``contact_velocity_car`` is the velocity at the caster mount/contact
    reference, not merely chassis-centre translation.  The ideal rolling
    direction follows that vector, so analytic alignment, scrub, and slip are
    all zero.  Every maneuver uses the same identity ball-to-car calibration;
    only the rendered material rotation axis changes, as a real ball's does.

    The production adapter's documented default ``roll_direction_sign=-1``
    is exercised.  Consequently the renderer rotates about the negative of
    the expected common-frame axis.
    """

    count = int(num_frames)
    rate_hz = float(fps)
    radius = float(r_eff)
    if count < 3:
        raise ValueError("num_frames must be at least 3")
    if not np.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("fps must be positive and finite")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("r_eff must be positive and finite")

    velocity = np.asarray(contact_velocity_car, dtype=float).reshape(-1)
    heading = _horizontal_unit(velocity, "contact_velocity_car")
    speed = float(np.linalg.norm(velocity))
    # z x roll_axis = rolling heading => roll_axis = -z x heading.
    expected_axis = np.array([heading[1], -heading[0], 0.0], dtype=float)
    rendered_axis = -expected_axis
    time_s = np.arange(count, dtype=float) / rate_hz
    angle = (speed / radius) * time_s
    zeros = np.zeros(count, dtype=float)
    trajectory = Trajectory(
        angle,
        zeros,
        zeros,
        fps=rate_hz,
        name="ideal_ball_constant_contact_velocity",
        roll_axis_ball=rendered_axis,
    )
    # The 600x800 oracle keeps the slow spin-in-place case below the spec's
    # 2% slip-error gate; 480x640 quantizes its roughly two-degree steps enough
    # to introduce a repeatable ~2.4% rate bias.
    camera = default_camera(image_size=(600, 800))
    rendered = render_sequence(
        output_dir,
        trajectory,
        RenderConfig(
            camera=camera,
            seed=int(seed),
            num_speckles=2_200,
            dot_radius_px=3.0,
        ),
        return_frames=True,
    )
    if rendered.frames is None:
        raise RuntimeError("ball renderer did not retain in-memory frames")

    result = run_pipeline(
        rendered.frames,
        K=camera.K,
        dist=camera.dist,
        circle=camera.circle,
        R_bc=camera.R_bc,
        segment_config=SYNTHETIC_BALL_SEGMENT,
        track_config=dict(track_config or {}),
        estimate_config=dict(estimate_config or {}),
        fps=rate_hz,
    )
    frames = adapt_ball_pipeline(
        result,
        R_bc=camera.R_bc,
        R_ball_to_car=np.eye(3),
        r_eff=radius,
        roll_direction_sign=-1.0,
    )
    return SyntheticBallCase(
        rendered=rendered,
        pipeline=result,
        frames=frames,
        contact_velocity_car=velocity,
        expected_roll_axis_car=expected_axis,
        expected_omega_roll=speed / radius,
    )


def ball_case_diagnostics(case: SyntheticBallCase) -> dict[str, float | int]:
    """Summarize image-to-common-frame error against the rendered oracle."""

    valid = [frame for frame in case.frames if frame.roll_valid]
    if valid:
        axes = np.asarray([frame.roll_axis_car for frame in valid], dtype=float)
        cosines = np.clip(axes @ case.expected_roll_axis_car, -1.0, 1.0)
        axis_error = np.rad2deg(np.arccos(cosines))
        rates = np.asarray([frame.omega_roll for frame in valid], dtype=float)
        spins = np.asarray([frame.omega_spin for frame in valid], dtype=float)
        axis_rmse = float(np.sqrt(np.mean(axis_error**2)))
        rate_relative_error = float(
            np.mean(np.abs(rates - case.expected_omega_roll))
            / case.expected_omega_roll
        )
        mean_abs_spin = float(np.mean(np.abs(spins)))
    else:
        axis_rmse = float("inf")
        rate_relative_error = float("inf")
        mean_abs_spin = float("inf")
    return {
        "source_frame_count": int(case.pipeline.frame_count),
        "common_frame_count": int(len(case.frames)),
        "valid_frame_count": int(len(valid)),
        "roll_coverage": float(len(valid) / len(case.frames)) if case.frames else 0.0,
        "roll_axis_rmse_deg": axis_rmse,
        "roll_rate_mean_relative_error": rate_relative_error,
        "mean_abs_spin_rad_s": mean_abs_spin,
        "expected_omega_roll_rad_s": float(case.expected_omega_roll),
    }


__all__ = [
    "SYNTHETIC_BALL_SEGMENT",
    "SyntheticBallCase",
    "ball_case_diagnostics",
    "run_ideal_ball_case",
]
