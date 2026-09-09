"""Deterministic synthetic data for the ball-caster rotation pipeline.

The package deliberately keeps rendering independent from the tracking code so
that generated images remain an external oracle for the rest of the project.
"""

from .generate import (
    CameraSetup,
    Degradations,
    RenderConfig,
    RenderResult,
    Trajectory,
    default_camera,
    fibonacci_sphere,
    generate_calibration_sequences,
    generate_robustness_case,
    generate_sequence,
    project,
    pure_roll_trajectory,
    pure_swivel_trajectory,
    render_sequence,
    scripted_trajectory,
    speed_sweep_trajectory,
)

__all__ = [
    "CameraSetup",
    "Degradations",
    "RenderConfig",
    "RenderResult",
    "Trajectory",
    "default_camera",
    "fibonacci_sphere",
    "generate_calibration_sequences",
    "generate_robustness_case",
    "generate_sequence",
    "project",
    "pure_roll_trajectory",
    "pure_swivel_trajectory",
    "render_sequence",
    "scripted_trajectory",
    "speed_sweep_trajectory",
]
