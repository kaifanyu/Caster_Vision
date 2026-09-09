"""Staged synthetic ground-truth validation for the complete CV pipeline.

The public :func:`run_validation` entry point executes the numeric Stage A
gate, a noise-free image baseline (Stages B/C), every requested robustness
sweep (Stage D), and the auto-circle plus isolated-axis calibration gate
(Stage E).  All reported angular measurements are degrees.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.config import load_config
from ball.integrate import calibrate_ball_frame
from ball.pipeline import PipelineResult, run_pipeline
from common.rotation import geodesic_angle
from synthetic.generate_ball import (
    Degradations,
    RenderConfig,
    RenderResult,
    Trajectory,
    default_camera,
    pure_roll_trajectory,
    pure_swivel_trajectory,
    render_sequence,
    scripted_trajectory,
    speed_sweep_trajectory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 24_681_357
NOISE_LEVELS_PX = (0.25, 0.5, 1.0, 2.0)
SPEED_LEVELS_DEG = (2.0, 5.0, 10.0, 15.0, 20.0)

THRESHOLDS = {
    "stage_b_median_deg": 0.5,
    "stage_b_p95_deg": 1.5,
    "stage_b_min_coverage": 0.90,
    "stage_c_q_rmse_deg": 1.0,
    "stage_c_gamma_median_deg": 0.5,
    "stage_c_alpha_agreement_rmse_deg": 1.0,
    "stage_c_min_coverage": 0.90,
    "stage_d_error_limit_deg": 2.0,
    "stage_d_min_speed_coverage": 0.70,
    "stage_e_q_rmse_deg": 2.0,
    "stage_e_min_coverage": 0.90,
}

_SYNTHETIC_SEGMENT_DEFAULTS: dict[str, Any] = {
    "mode": "color",
    "top_hsv": {"lo": [170, 90, 60], "hi": [10, 255, 255]},
    "bottom_hsv": {"lo": [85, 90, 60], "hi": [105, 255, 255]},
    "yoke_hsv": {"lo": [0, 0, 0], "hi": [179, 255, 55], "enabled": True},
    "morphology_px": 1,
    "ball_margin_px": 1.0,
}

# Renderer-matched settings carried over from the validated Approach-A
# harness.  The unified top-level ``track`` section is also used by the
# sidewall pipeline and is therefore a real-footage operating point, not a
# stable ball-oracle definition.  Callers may override these explicitly under
# ``ball.synthetic_validation`` without coupling one device's release gate to
# the other device's tuning.
_SYNTHETIC_TRACK_DEFAULTS: dict[str, Any] = {
    "max_corners": 400,
    "quality": 0.01,
    "min_distance_px": 6,
    "klt_win": 21,
    "pyramid_levels": 3,
    "fwd_bwd_err_px": 1.0,
    "limb_cull_deg": 65.0,
}
_SYNTHETIC_ESTIMATE_DEFAULTS: dict[str, Any] = {
    "ransac_iters": 200,
    "ransac_inlier_deg": 1.0,
    "min_inliers": 8,
    "random_seed": 7,
}


def _finite(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _stat(values: Sequence[float] | np.ndarray, operation: str) -> float:
    finite = _finite(values)
    if not len(finite):
        return float("nan")
    if operation == "median":
        return float(np.median(finite))
    if operation == "p95":
        return float(np.percentile(finite, 95))
    if operation == "mean":
        return float(np.mean(finite))
    raise ValueError(f"unsupported statistic: {operation}")


def _rmse(values: Sequence[float] | np.ndarray) -> float:
    finite = _finite(values)
    return float(np.sqrt(np.mean(finite * finite))) if len(finite) else float("nan")


def _is_below(value: float, threshold: float) -> bool:
    return bool(np.isfinite(value) and value < threshold)


def _circle_from_truth(truth: Mapping[str, Any]) -> tuple[float, float, float]:
    circle = truth["circle"]
    return float(circle["u0"]), float(circle["v0"]), float(circle["r_px"])


def _synthetic_segment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Use renderer-matched colours while retaining benign mask tuning."""

    resolved = dict(_SYNTHETIC_SEGMENT_DEFAULTS)
    ball = config.get("ball", {})
    supplied = ball.get("segment", {}) if isinstance(ball, Mapping) else {}
    if isinstance(supplied, Mapping):
        for key in (
            "morphology_px",
            "morph_kernel",
            "morph_iterations",
            "ball_margin_px",
        ):
            if key in supplied:
                resolved[key] = supplied[key]
    return resolved


def _pipeline_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    ball = config.get("ball", {})
    validation = (
        ball.get("synthetic_validation", {}) if isinstance(ball, Mapping) else {}
    )
    if not isinstance(validation, Mapping):
        raise ValueError("ball.synthetic_validation must be a mapping")
    track_override = validation.get("track", {})
    estimate_override = validation.get("estimate", {})
    if not isinstance(track_override, Mapping) or not isinstance(
        estimate_override, Mapping
    ):
        raise ValueError("ball.synthetic_validation track/estimate must be mappings")
    track = {**_SYNTHETIC_TRACK_DEFAULTS, **dict(track_override)}
    estimate = {**_SYNTHETIC_ESTIMATE_DEFAULTS, **dict(estimate_override)}
    return {
        "segment_config": _synthetic_segment_config(config),
        "track_config": track,
        "estimate_config": estimate,
    }


def _render_config(degradations: Degradations, seed: int) -> RenderConfig:
    return RenderConfig(
        camera=default_camera(),
        seed=seed,
        degradations=degradations,
    )


def _render(
    trajectory: Trajectory,
    degradations: Degradations,
    seed: int,
) -> RenderResult:
    return render_sequence(
        None,
        trajectory,
        _render_config(degradations, seed),
        return_frames=True,
    )


def _measure(
    rendered: RenderResult,
    config: Mapping[str, Any],
    *,
    known_circle: bool,
    R_bc: np.ndarray | None,
) -> PipelineResult:
    frames = rendered.frames
    if frames is None:
        raise RuntimeError("renderer did not return in-memory frames")
    truth = rendered.ground_truth
    return run_pipeline(
        frames,
        K=np.asarray(truth["K"], dtype=float),
        dist=np.asarray(truth["dist"], dtype=float),
        circle=_circle_from_truth(truth) if known_circle else None,
        R_bc=R_bc,
        fps=float(truth["fps"]),
        **_pipeline_kwargs(config),
    )


def _increment_metrics(
    result: PipelineResult, truth: Mapping[str, Any]
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    all_errors: list[np.ndarray] = []
    for name, estimates, truth_key in (
        ("top", result.top_increments, "R_top_increment_camera"),
        ("bottom", result.bottom_increments, "R_bottom_increment_camera"),
    ):
        expected = np.asarray(truth[truth_key], dtype=float)[1:]
        if len(expected) != len(estimates):
            raise ValueError(
                f"{name} increment length mismatch: {len(estimates)} vs {len(expected)}"
            )
        errors = np.full(len(estimates), np.nan, dtype=float)
        for index, (estimated, target) in enumerate(zip(estimates, expected)):
            if estimated is not None:
                errors[index] = np.rad2deg(geodesic_angle(estimated, target))
        finite = np.isfinite(errors)
        if np.any(finite):
            all_errors.append(errors[finite])
        metrics[name] = {
            "median_deg": _stat(errors, "median"),
            "p95_deg": _stat(errors, "p95"),
            "mean_deg": _stat(errors, "mean"),
            "coverage": float(finite.mean()) if len(finite) else 0.0,
            "valid_steps": int(finite.sum()),
            "total_steps": int(len(errors)),
        }
    pooled = np.concatenate(all_errors) if all_errors else np.empty(0)
    metrics["pooled"] = {
        "median_deg": _stat(pooled, "median"),
        "p95_deg": _stat(pooled, "p95"),
    }
    return metrics


def _angle_error_deg(estimated: np.ndarray, truth: np.ndarray) -> np.ndarray:
    estimated = np.asarray(estimated, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if estimated.shape != truth.shape:
        raise ValueError(f"angle shape mismatch: {estimated.shape} vs {truth.shape}")
    # Both sequences start at the rendered identity.  Unwrapping the truth
    # makes comparison compatible with decompose_hemispheres across +/-pi.
    return np.rad2deg(estimated - np.unwrap(truth))


def _q_metrics(result: PipelineResult, truth: Mapping[str, Any]) -> dict[str, Any]:
    if result.motion is None:
        raise ValueError("pipeline result has no decomposition; R_bc was not supplied")
    motion = result.motion
    fields = (
        ("alpha", motion.alpha, "alpha"),
        ("beta_top", motion.beta_top, "beta_top"),
        ("beta_bottom", motion.beta_bottom, "beta_bottom"),
    )
    metrics: dict[str, Any] = {}
    pooled: list[np.ndarray] = []
    for name, estimated, truth_key in fields:
        error = _angle_error_deg(estimated, np.asarray(truth[truth_key], dtype=float))
        finite = np.isfinite(error)
        if np.any(finite):
            pooled.append(np.abs(error[finite]))
        metrics[name] = {
            "rmse_deg": _rmse(error),
            "median_abs_deg": _stat(np.abs(error), "median"),
            "p95_abs_deg": _stat(np.abs(error), "p95"),
            "coverage": float(finite.mean()) if len(finite) else 0.0,
            "valid_frames": int(finite.sum()),
            "total_frames": int(len(error)),
        }

    gamma = np.abs(
        np.rad2deg(
            np.concatenate(
                [np.asarray(motion.gamma_top), np.asarray(motion.gamma_bottom)]
            )
        )
    )
    agreement = np.rad2deg(
        np.asarray(motion.alpha_top) - np.asarray(motion.alpha_bottom)
    )
    pooled_error = np.concatenate(pooled) if pooled else np.empty(0)
    rmse_values = [metrics[name]["rmse_deg"] for name, _, _ in fields]
    coverage_values = [metrics[name]["coverage"] for name, _, _ in fields]
    metrics["gamma_median_abs_deg"] = _stat(gamma, "median")
    metrics["alpha_agreement_rmse_deg"] = _rmse(agreement)
    metrics["pooled_median_abs_deg"] = _stat(pooled_error, "median")
    metrics["pooled_p95_abs_deg"] = _stat(pooled_error, "p95")
    metrics["max_q_rmse_deg"] = (
        float(max(rmse_values)) if all(np.isfinite(rmse_values)) else float("nan")
    )
    metrics["min_q_coverage"] = float(min(coverage_values))
    return metrics


def _quality_metrics(result: PipelineResult) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for hemisphere in ("top", "bottom"):
        qualities = [getattr(frame, hemisphere) for frame in result.qualities]
        output[hemisphere] = {
            "median_inlier_ratio": _stat(
                [item.inlier_ratio for item in qualities], "median"
            ),
            "median_residual_deg": _stat(
                [item.median_residual_deg for item in qualities], "median"
            ),
            "median_tracked_count": _stat(
                [item.matched_count for item in qualities], "median"
            ),
        }
    return output


def _stage_a() -> dict[str, Any]:
    started = time.perf_counter()
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(PROJECT_ROOT / "tests" / "test_ball_units.py"),
        "-q",
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    lines = [line for line in stdout.splitlines() if line.strip()]
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "passed": completed.returncode == 0,
        "exit_code": int(completed.returncode),
        "pytest_summary": lines[-1] if lines else "no pytest summary",
        "stdout": stdout,
        "stderr": stderr,
        "runtime_s": time.perf_counter() - started,
        "gate": "tests/test_ball_units.py exits 0",
    }


def _stage_b(result: PipelineResult, truth: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _increment_metrics(result, truth)
    passed = all(
        _is_below(metrics[name]["median_deg"], THRESHOLDS["stage_b_median_deg"])
        and _is_below(metrics[name]["p95_deg"], THRESHOLDS["stage_b_p95_deg"])
        and metrics[name]["coverage"] >= THRESHOLDS["stage_b_min_coverage"]
        for name in ("top", "bottom")
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "metrics": metrics,
        "quality": _quality_metrics(result),
        "gate": (
            "each hemisphere median < 0.5 deg, p95 < 1.5 deg, "
            "and step coverage >= 0.90"
        ),
    }


def _stage_c(result: PipelineResult, truth: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _q_metrics(result, truth)
    q_pass = all(
        _is_below(metrics[name]["rmse_deg"], THRESHOLDS["stage_c_q_rmse_deg"])
        and metrics[name]["coverage"] >= THRESHOLDS["stage_c_min_coverage"]
        for name in ("alpha", "beta_top", "beta_bottom")
    )
    passed = (
        q_pass
        and _is_below(
            metrics["gamma_median_abs_deg"],
            THRESHOLDS["stage_c_gamma_median_deg"],
        )
        and _is_below(
            metrics["alpha_agreement_rmse_deg"],
            THRESHOLDS["stage_c_alpha_agreement_rmse_deg"],
        )
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "metrics": metrics,
        "gate": (
            "all q RMSE < 1 deg, median |gamma| < 0.5 deg, "
            "top/bottom alpha agreement RMSE < 1 deg, coverage >= 0.90"
        ),
    }


def _prefix_trajectory(frame_count: int) -> Trajectory:
    full = scripted_trajectory(121, 60.0)
    return Trajectory(
        full.alpha[:frame_count],
        full.beta_top[:frame_count],
        full.beta_bottom[:frame_count],
        fps=full.fps,
        name=f"scripted_prefix_{frame_count}",
    )


def _robustness_case(
    name: str,
    trajectory: Trajectory,
    degradations: Degradations,
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        rendered = _render(trajectory, degradations, seed)
        truth = rendered.ground_truth
        camera = np.asarray(truth["R_bc"], dtype=float)
        result = _measure(rendered, config, known_circle=True, R_bc=camera)
        return {
            "name": name,
            "completed": True,
            "q": _q_metrics(result, truth),
            "increment": _increment_metrics(result, truth),
            "quality": _quality_metrics(result),
            "runtime_s": time.perf_counter() - started,
        }
    except Exception as exc:  # Keep all requested sweep points in the report.
        return {
            "name": name,
            "completed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_s": time.perf_counter() - started,
        }


def _speed_selection_error(case: Mapping[str, Any]) -> float:
    if not case.get("completed"):
        return float("inf")
    q = case["q"]
    if q["min_q_coverage"] < THRESHOLDS["stage_d_min_speed_coverage"]:
        return float("inf")
    value = float(q["pooled_median_abs_deg"])
    return value if np.isfinite(value) else float("inf")


def _stage_d(
    config: Mapping[str, Any],
    *,
    quick: bool,
    seed: int,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    mixed = _prefix_trajectory(31 if quick else 61)
    speed_frames = 21 if quick else 41
    baseline_degradation = Degradations()

    progress("Stage D: baseline degradation case")
    baseline = _robustness_case(
        "baseline", mixed, baseline_degradation, config, seed
    )

    noise_cases: list[dict[str, Any]] = []
    for sigma in NOISE_LEVELS_PX:
        progress(f"Stage D: pixel noise sigma={sigma:g} px")
        case = _robustness_case(
            f"noise_{sigma:g}px",
            mixed,
            replace(baseline_degradation, pixel_noise_std_px=sigma),
            config,
            seed,
        )
        case["sigma_px"] = sigma
        noise_cases.append(case)

    progress("Stage D: motion blur on")
    blur_on = _robustness_case(
        "motion_blur_on",
        mixed,
        replace(baseline_degradation, motion_blur_samples=5),
        config,
        seed,
    )
    progress("Stage D: glare on")
    glare_on = _robustness_case(
        "glare_on",
        mixed,
        replace(baseline_degradation, glare=True),
        config,
        seed,
    )
    progress("Stage D: half speckle density")
    density_half = _robustness_case(
        "density_half",
        mixed,
        replace(baseline_degradation, density_scale=0.5),
        config,
        seed,
    )

    speed_cases: list[dict[str, Any]] = []
    for delta in SPEED_LEVELS_DEG:
        progress(f"Stage D: inter-frame rotation={delta:g} deg")
        trajectory = speed_sweep_trajectory(delta, speed_frames, 60.0)
        case = _robustness_case(
            f"speed_{delta:g}deg_per_frame",
            trajectory,
            baseline_degradation,
            config,
            seed,
        )
        case["delta_deg_per_frame"] = delta
        case["selection_error_deg"] = _speed_selection_error(case)
        speed_cases.append(case)

    first_exceeds: float | None = None
    delta_max = 0.0
    for case in speed_cases:
        delta = float(case["delta_deg_per_frame"])
        if case["selection_error_deg"] > THRESHOLDS["stage_d_error_limit_deg"]:
            first_exceeds = delta
            break
        delta_max = delta
    characterization = "bounded" if first_exceeds is not None else "at_least_tested_max"

    def q_delta(case: Mapping[str, Any]) -> float:
        if not case.get("completed") or not baseline.get("completed"):
            return float("nan")
        return float(case["q"]["max_q_rmse_deg"] - baseline["q"]["max_q_rmse_deg"])

    effects = {
        "motion_blur": {
            "off": baseline,
            "on": blur_on,
            "max_q_rmse_delta_deg": q_delta(blur_on),
        },
        "glare": {
            "off": baseline,
            "on": glare_on,
            "max_q_rmse_delta_deg": q_delta(glare_on),
        },
        "density": {
            "full": baseline,
            "half": density_half,
            "max_q_rmse_delta_deg": q_delta(density_half),
        },
    }
    all_cases = [baseline, *noise_cases, blur_on, glare_on, density_half, *speed_cases]
    completed = all(bool(case.get("completed")) for case in all_cases)
    noise_expected = all(
        bool(case.get("completed"))
        and float(case["q"]["max_q_rmse_deg"])
        < THRESHOLDS["stage_d_error_limit_deg"]
        for case in noise_cases
        if float(case["sigma_px"]) <= 1.0
    )
    passed = completed and np.isfinite(delta_max)
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "baseline": baseline,
        "noise_sweep": noise_cases,
        "speed_sweep": speed_cases,
        "effects": effects,
        "delta_max_deg_per_frame": delta_max,
        "delta_max_characterization": characterization,
        "first_exceeds_2deg_at_delta": first_exceeds,
        "speed_error_definition": (
            "pooled median absolute q error; coverage below 0.70 counts as exceeded"
        ),
        "minimum_fps_formula": "minimum_fps = expected_peak_speed_deg_s / delta_max_deg_per_frame",
        "noise_through_1px_under_2deg": noise_expected,
        "gate": "all requested sweeps completed and delta_max is stated",
    }


def _combine_increments(result: PipelineResult) -> list[np.ndarray | None]:
    return [*result.top_increments, *result.bottom_increments]


def _stage_e(
    baseline_render: RenderResult,
    config: Mapping[str, Any],
    *,
    quick: bool,
    seed: int,
    progress: Callable[[str], None],
) -> tuple[dict[str, Any], PipelineResult | None]:
    started = time.perf_counter()
    try:
        camera = _render_config(Degradations(), seed).camera
        calibration_frames = 41 if quick else 61
        progress("Stage E: pure-roll clip with automatic circle")
        roll_render = _render(
            pure_roll_trajectory(calibration_frames, 60.0), Degradations(), seed
        )
        roll_result = _measure(
            roll_render, config, known_circle=False, R_bc=None
        )
        progress("Stage E: pure-swivel clip with automatic circle")
        swivel_render = _render(
            pure_swivel_trajectory(calibration_frames, 60.0), Degradations(), seed
        )
        swivel_result = _measure(
            swivel_render, config, known_circle=False, R_bc=None
        )

        estimated_frame, axis_diagnostics = calibrate_ball_frame(
            _combine_increments(roll_result),
            _combine_increments(swivel_result),
        )
        frame_error = float(
            np.rad2deg(geodesic_angle(estimated_frame, camera.R_bc))
        )

        progress("Stage E: scripted clip with estimated circle and R_bc")
        measured = _measure(
            baseline_render,
            config,
            known_circle=False,
            R_bc=estimated_frame,
        )
        truth = baseline_render.ground_truth
        q = _q_metrics(measured, truth)
        true_circle = np.asarray(_circle_from_truth(truth), dtype=float)
        estimated_circle = np.asarray(measured.circle, dtype=float)
        q_pass = all(
            _is_below(q[name]["rmse_deg"], THRESHOLDS["stage_e_q_rmse_deg"])
            and q[name]["coverage"] >= THRESHOLDS["stage_e_min_coverage"]
            for name in ("alpha", "beta_top", "beta_bottom")
        )
        stage = {
            "status": "PASS" if q_pass else "FAIL",
            "passed": q_pass,
            "metrics": q,
            "quality": _quality_metrics(measured),
            "estimated_R_bc": estimated_frame.tolist(),
            "R_bc_geodesic_error_deg": frame_error,
            "axis_calibration": axis_diagnostics,
            "circles": {
                "true": true_circle.tolist(),
                "scripted_estimated": estimated_circle.tolist(),
                "pure_roll_estimated": list(roll_result.circle),
                "pure_swivel_estimated": list(swivel_result.circle),
                "center_error_px": float(
                    np.linalg.norm(estimated_circle[:2] - true_circle[:2])
                ),
                "radius_error_px": float(estimated_circle[2] - true_circle[2]),
            },
            "runtime_s": time.perf_counter() - started,
            "gate": "all q RMSE < 2 deg using estimated circle and estimated R_bc",
        }
        return stage, measured
    except Exception as exc:
        return (
            {
                "status": "FAIL",
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "runtime_s": time.perf_counter() - started,
                "gate": "all q RMSE < 2 deg using estimated circle and estimated R_bc",
            },
            None,
        )


def _plot_motion(
    result: PipelineResult,
    truth: Mapping[str, Any],
    path: Path,
    title: str,
) -> Path:
    if result.motion is None:
        raise ValueError("cannot plot motion without decomposition")
    time_s = np.asarray(result.timestamps_s, dtype=float)
    series = (
        ("alpha", result.motion.alpha, np.asarray(truth["alpha"])),
        ("beta top", result.motion.beta_top, np.asarray(truth["beta_top"])),
        ("beta bottom", result.motion.beta_bottom, np.asarray(truth["beta_bottom"])),
    )
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for axis, (label, estimated, target) in zip(axes, series):
        axis.plot(time_s, np.rad2deg(target), "k--", linewidth=1.4, label="truth")
        axis.plot(time_s, np.rad2deg(estimated), linewidth=1.2, label="recovered")
        axis.set_ylabel(f"{label} (deg)")
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("time (s)")
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _case_q_value(case: Mapping[str, Any]) -> float:
    if not case.get("completed"):
        return float("nan")
    return float(case["q"]["max_q_rmse_deg"])


def _plot_stage_d(stage: Mapping[str, Any], path: Path) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    noise = stage["noise_sweep"]
    axes[0].plot(
        [case["sigma_px"] for case in noise],
        [_case_q_value(case) for case in noise],
        marker="o",
    )
    axes[0].axhline(2.0, color="r", linestyle="--", linewidth=1)
    axes[0].set(xlabel="pixel noise sigma (px)", ylabel="max q RMSE (deg)", title="Pixel noise")

    speed = stage["speed_sweep"]
    selection = [
        value if np.isfinite(value) else np.nan
        for value in (float(case["selection_error_deg"]) for case in speed)
    ]
    axes[1].plot(
        [case["delta_deg_per_frame"] for case in speed], selection, marker="o"
    )
    axes[1].axhline(2.0, color="r", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="rotation step (deg/frame)",
        ylabel="median q error (deg)",
        title=f"Speed (delta_max={stage['delta_max_deg_per_frame']:g})",
    )

    effects = stage["effects"]
    labels = ["motion blur", "glare", "half density"]
    values = [
        effects["motion_blur"]["max_q_rmse_delta_deg"],
        effects["glare"]["max_q_rmse_delta_deg"],
        effects["density"]["max_q_rmse_delta_deg"],
    ]
    axes[2].bar(labels, values, color=["#5277b8", "#e39c37", "#56a36c"])
    axes[2].axhline(0.0, color="k", linewidth=0.8)
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].set(ylabel="max q RMSE delta (deg)", title="Degradation deltas")
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.suptitle("Stage D robustness characterization")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_report(report: Mapping[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _skipped(reason: str) -> dict[str, Any]:
    return {"status": "SKIP", "passed": False, "reason": reason}


def run_validation(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    quick: bool = False,
    seed: int = DEFAULT_SEED,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Run Stages A-E and write plots plus ``validation_report.json``.

    ``quick`` still exercises every requested Stage-D sweep value, but uses
    shorter deterministic clips.  Full mode is the release/acceptance run.
    """

    emit = progress if progress is not None else (lambda _message: None)
    config, resolved_config = load_config(config_path)
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (resolved_config.parent / "out" / "synthetic" / "ball").resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema": "synthetic-ball-validation-v1",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "quick" if quick else "full",
        "quick": bool(quick),
        "config": str(resolved_config),
        "output_dir": str(destination),
        "seed": int(seed),
        "thresholds": dict(THRESHOLDS),
        "resolved_assumptions": {
            "camera_fixed_to_chassis": True,
            "synthetic_intrinsics_source": "known renderer K",
            "segmentation_mode": "color (renderer-matched HSV)",
            "tracking_profile": "pinned Approach-A synthetic oracle defaults",
            "stages_b_c_circle_source": "known renderer circle",
            "stages_b_c_R_bc_source": "known renderer R_bc",
            "stage_e_circle_source": "automatic image fit",
            "stage_e_R_bc_source": "pure-roll plus pure-swivel clips",
        },
        "stages": {},
        "plots": [],
    }

    emit("Stage A: numeric unit tests")
    stage_a = _stage_a()
    report["stages"]["A"] = stage_a
    if not stage_a["passed"]:
        reason = "Stage A failed; later hard gates were not run"
        for name in "BCDE":
            report["stages"][name] = _skipped(reason)
        report["overall_pass"] = False
        report["runtime_s"] = time.perf_counter() - started
        _write_report(report, destination / "validation_report.json")
        return _json_safe(report)

    baseline_frames = 61 if quick else 121
    baseline_render: RenderResult | None = None
    baseline_result: PipelineResult | None = None
    try:
        emit(f"Stages B/C: noise-free scripted clip ({baseline_frames} frames)")
        baseline_render = _render(
            scripted_trajectory(baseline_frames, 60.0), Degradations(), seed
        )
        known_R_bc = np.asarray(baseline_render.ground_truth["R_bc"], dtype=float)
        baseline_result = _measure(
            baseline_render, config, known_circle=True, R_bc=known_R_bc
        )
        report["stages"]["B"] = _stage_b(
            baseline_result, baseline_render.ground_truth
        )
    except Exception as exc:
        report["stages"]["B"] = {
            "status": "FAIL",
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "gate": "noise-free per-frame rotation accuracy",
        }

    if not report["stages"]["B"]["passed"]:
        reason = "Stage B failed; later hard gates were not run"
        for name in "CDE":
            report["stages"][name] = _skipped(reason)
        report["overall_pass"] = False
        report["runtime_s"] = time.perf_counter() - started
        _write_report(report, destination / "validation_report.json")
        return _json_safe(report)

    assert baseline_render is not None and baseline_result is not None
    report["stages"]["C"] = _stage_c(
        baseline_result, baseline_render.ground_truth
    )
    stage_c_plot = _plot_motion(
        baseline_result,
        baseline_render.ground_truth,
        destination / "stage_c_angles.png",
        "Stage C: known-frame decomposition",
    )
    report["plots"].append(str(stage_c_plot))
    if not report["stages"]["C"]["passed"]:
        reason = "Stage C failed; later hard gates were not run"
        report["stages"]["D"] = _skipped(reason)
        report["stages"]["E"] = _skipped(reason)
        report["overall_pass"] = False
        report["runtime_s"] = time.perf_counter() - started
        _write_report(report, destination / "validation_report.json")
        return _json_safe(report)

    stage_d = _stage_d(
        config, quick=quick, seed=seed, progress=emit
    )
    report["stages"]["D"] = stage_d
    stage_d_plot = _plot_stage_d(stage_d, destination / "stage_d_robustness.png")
    report["plots"].append(str(stage_d_plot))
    if not stage_d["passed"]:
        report["stages"]["E"] = _skipped(
            "Stage D sweep did not complete; Stage E was not run"
        )
        report["overall_pass"] = False
        report["runtime_s"] = time.perf_counter() - started
        _write_report(report, destination / "validation_report.json")
        return _json_safe(report)

    stage_e, stage_e_result = _stage_e(
        baseline_render,
        config,
        quick=quick,
        seed=seed,
        progress=emit,
    )
    report["stages"]["E"] = stage_e
    if stage_e_result is not None:
        stage_e_plot = _plot_motion(
            stage_e_result,
            baseline_render.ground_truth,
            destination / "stage_e_angles.png",
            "Stage E: estimated circle and ball frame",
        )
        report["plots"].append(str(stage_e_plot))

    report["overall_pass"] = all(
        bool(report["stages"][name].get("passed")) for name in "ABCDE"
    )
    report["runtime_s"] = time.perf_counter() - started
    report_path = _write_report(report, destination / "validation_report.json")
    report["report_path"] = str(report_path)
    # Re-write once with its own path recorded.
    _write_report(report, report_path)
    return _json_safe(report)


def _metric(stage: Mapping[str, Any], path: Sequence[str]) -> float | None:
    current: Any = stage
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    try:
        number = float(current)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def format_summary_table(report: Mapping[str, Any]) -> str:
    """Format the single human-readable PASS/FAIL table required by the spec."""

    stages = report.get("stages", {})
    rows: list[tuple[str, str, str]] = []
    a = stages.get("A", {})
    rows.append(
        (
            "A",
            str(a.get("status", "SKIP")),
            f"{a.get('pytest_summary', a.get('reason', ''))}; gate: exit 0",
        )
    )

    b = stages.get("B", {})
    if b.get("status") == "PASS" or b.get("status") == "FAIL":
        top_med = _metric(b, ("metrics", "top", "median_deg"))
        bottom_med = _metric(b, ("metrics", "bottom", "median_deg"))
        top_p95 = _metric(b, ("metrics", "top", "p95_deg"))
        bottom_p95 = _metric(b, ("metrics", "bottom", "p95_deg"))
        measured = (
            f"T/B median {top_med:.3f}/{bottom_med:.3f} <0.5; "
            f"p95 {top_p95:.3f}/{bottom_p95:.3f} <1.5 deg"
            if None not in (top_med, bottom_med, top_p95, bottom_p95)
            else str(b.get("error", "metrics unavailable"))
        )
    else:
        measured = str(b.get("reason", "not run"))
    rows.append(("B", str(b.get("status", "SKIP")), measured))

    c = stages.get("C", {})
    c_values = [
        _metric(c, ("metrics", name, "rmse_deg"))
        for name in ("alpha", "beta_top", "beta_bottom")
    ]
    if all(value is not None for value in c_values):
        measured = "q RMSE " + "/".join(f"{value:.3f}" for value in c_values) + " <1 deg"
    else:
        measured = str(c.get("error", c.get("reason", "not run")))
    rows.append(("C", str(c.get("status", "SKIP")), measured))

    d = stages.get("D", {})
    delta = _metric(d, ("delta_max_deg_per_frame",))
    if delta is not None:
        measured = (
            f"delta_max {delta:g} deg/frame; noise<=1px <2deg: "
            f"{'yes' if d.get('noise_through_1px_under_2deg') else 'no'}"
        )
    else:
        measured = str(d.get("error", d.get("reason", "not run")))
    rows.append(("D", str(d.get("status", "SKIP")), measured))

    e = stages.get("E", {})
    e_values = [
        _metric(e, ("metrics", name, "rmse_deg"))
        for name in ("alpha", "beta_top", "beta_bottom")
    ]
    if all(value is not None for value in e_values):
        measured = "q RMSE " + "/".join(f"{value:.3f}" for value in e_values) + " <2 deg"
    else:
        measured = str(e.get("error", e.get("reason", "not run")))
    rows.append(("E", str(e.get("status", "SKIP")), measured))

    headers = ("Stage", "Result", "Measured vs gate")
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def line(parts: Sequence[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(parts)) + " |"

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    output = [separator, line(headers), separator]
    output.extend(line(row) for row in rows)
    output.append(separator)
    output.append(f"OVERALL: {'PASS' if report.get('overall_pass') else 'FAIL'}")
    if report.get("report_path"):
        output.append(f"Report: {report['report_path']}")
    return "\n".join(output)


__all__ = ["format_summary_table", "run_validation"]
