"""Diagnostics, result serialization, and plots for ball rotation runs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .integrate import DecomposedMotion, angular_velocity
from .rotation import geodesic_angle
from .trajectory import angular_rates, rotation_rates


@dataclass
class HemisphereQuality:
    matched_count: int = 0
    usable_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = float("nan")
    mean_residual_deg: float = float("nan")
    median_residual_deg: float = float("nan")
    median_fb_error_px: float = float("nan")
    success: bool = False


@dataclass
class FrameQuality:
    frame_index: int
    top: HemisphereQuality
    bottom: HemisphereQuality


def _finite_stat(values: Sequence[float], operation: str) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    if operation == "median":
        return float(np.median(array))
    if operation == "mean":
        return float(np.mean(array))
    if operation == "p95":
        return float(np.percentile(array, 95))
    if operation == "max":
        return float(np.max(array))
    if operation == "min":
        return float(np.min(array))
    raise ValueError(operation)


def summarize_quality(
    qualities: Sequence[FrameQuality],
    motion: DecomposedMotion,
    top_absolute: np.ndarray,
    bottom_absolute: np.ndarray,
) -> dict[str, Any]:
    """Return self-consistency metrics without pretending they are ground truth."""

    summary: dict[str, Any] = {"frame_pairs": len(qualities)}
    for hemisphere in ("top", "bottom"):
        items = [getattr(q, hemisphere) for q in qualities]
        summary[hemisphere] = {
            "success_rate": float(np.mean([item.success for item in items])) if items else None,
            "tracked_count_median": _finite_stat(
                [item.matched_count for item in items], "median"
            ),
            "tracked_count_min": (
                int(min(item.matched_count for item in items)) if items else None
            ),
            "inlier_ratio_median": _finite_stat(
                [item.inlier_ratio for item in items], "median"
            ),
            "inlier_ratio_min": _finite_stat(
                [item.inlier_ratio for item in items], "min"
            ),
            "mean_inlier_residual_deg_median": _finite_stat(
                [item.mean_residual_deg for item in items], "median"
            ),
            "mean_inlier_residual_deg_max": _finite_stat(
                [item.mean_residual_deg for item in items], "max"
            ),
            "forward_backward_error_px_median": _finite_stat(
                [item.median_fb_error_px for item in items], "median"
            ),
            "forward_backward_error_px_max": _finite_stat(
                [item.median_fb_error_px for item in items], "max"
            ),
            "gamma_residual_deg_median": _finite_stat(
                np.abs(
                    np.rad2deg(
                        motion.gamma_top
                        if hemisphere == "top"
                        else motion.gamma_bottom
                    )[1:]
                ),
                "median",
            ),
        }
    alpha_disagreement = np.rad2deg(
        np.abs(motion.alpha_top[1:] - motion.alpha_bottom[1:])
    )
    summary["alpha_top_bottom_disagreement_deg_median"] = _finite_stat(
        alpha_disagreement, "median"
    )
    summary["end_to_start_rotation_deg"] = {
        "top": float(np.rad2deg(geodesic_angle(top_absolute[-1], top_absolute[0]))),
        "bottom": float(
            np.rad2deg(geodesic_angle(bottom_absolute[-1], bottom_absolute[0]))
        ),
        "note": "This is loop-closure drift only if the physical clip ends at its marked start pose.",
    }
    return summary


def unconstrained_result(
    motion: DecomposedMotion, top_absolute: np.ndarray, bottom_absolute: np.ndarray,
    qualities: Sequence[FrameQuality],
) -> dict[str, Any]:
    """Keep the independent estimates available when a joint model is fitted."""
    return {
        "note": "Independent estimates before mechanical constraints; missing poses remain invalid.",
        "angle_unit": "radians",
        "frames": asdict(motion),
        "top_absolute": np.asarray(top_absolute),
        "bottom_absolute": np.asarray(bottom_absolute),
        "summary": summarize_quality(qualities, motion, top_absolute, bottom_absolute),
    }


def write_run_outputs(
    output_dir: str | Path,
    timestamps_s: np.ndarray,
    motion: DecomposedMotion,
    qualities: Sequence[FrameQuality],
    top_absolute: np.ndarray,
    bottom_absolute: np.ndarray,
    *,
    metadata: Mapping[str, Any] | None = None,
    extra_summary: Mapping[str, Any] | None = None,
    temporal: Mapping[str, Any] | None = None,
    offline: Mapping[str, Any] | None = None,
    mechanical: Mapping[str, Any] | None = None,
    unconstrained: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the results CSV/JSON and diagnostic plots.

    ``extra_summary`` is merged into the JSON ``summary`` block.  Checks a
    caller computes for itself belong on disk beside the ones computed here,
    not only in the terminal, so a stored run can still be audited later.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    times = np.asarray(timestamps_s, dtype=float)
    if times.shape != motion.alpha.shape:
        raise ValueError("timestamp count must equal decomposed frame count")
    velocities = {
        "alpha": angular_velocity(motion.alpha, times),
        "beta_top": angular_velocity(motion.beta_top, times),
        "beta_bottom": angular_velocity(motion.beta_bottom, times),
    }
    camera_rates = {}
    rate_metadata = None
    mechanically_fitted = bool(mechanical is not None and mechanical.get("config", {}).get("enabled", False))
    if mechanically_fitted or (offline is not None and offline.get("config", {}).get("enabled", False)):
        options = (mechanical.get("offline_config", (offline or {}).get("config", {}))
                   if mechanically_fitted else offline["config"])
        rate_options = {"window_s": float(options.get("rate_window_s", 0.25)),
                        "polynomial_order": int(options.get("rate_polynomial_order", 2))}
        velocities = {
            # Independent poses need both shells for a consistent average;
            # a joint fit estimates the same roll from either visible shell.
            "alpha": angular_rates(motion.alpha, times,
                                   (motion.valid_top | motion.valid_bottom) if mechanically_fitted
                                   else (motion.valid_top & motion.valid_bottom), **rate_options),
            "beta_top": angular_rates(motion.beta_top, times, motion.valid_top, **rate_options),
            "beta_bottom": angular_rates(motion.beta_bottom, times, motion.valid_bottom, **rate_options),
        }
        camera_rates = {name: rotation_rates(poses, times, valid, **rate_options)
                        for name, poses, valid in (
                            ("top", top_absolute, motion.valid_top),
                            ("bottom", bottom_absolute, motion.valid_bottom))}
        rate_metadata = {
            "method": "local_polynomial_on_native_timestamps", **rate_options,
            "shared_roll_support": ("either shell valid under shared-roll model" if mechanically_fitted
                                    else "both shells valid"),
            "camera_omega_convention": "spatial angular velocity: dR/dt = skew(omega) @ R",
            "note": "Rates do not bridge invalid poses. Short segments yield missing rates. The window can attenuate fast responses; poses are not smoothed.",
        }

    quality_by_frame = {item.frame_index: item for item in qualities}
    csv_path = output / "results.csv"
    fields = [
        "frame",
        "time_s",
        "alpha_rad",
        "beta_top_rad",
        "beta_bottom_rad",
        "alpha_deg",
        "beta_top_deg",
        "beta_bottom_deg",
        "alpha_velocity_rad_s",
        "beta_top_velocity_rad_s",
        "beta_bottom_velocity_rad_s",
        "alpha_top_deg",
        "alpha_bottom_deg",
        "gamma_top_deg",
        "gamma_bottom_deg",
        "top_inlier_ratio",
        "bottom_inlier_ratio",
        "top_residual_deg",
        "bottom_residual_deg",
        "top_tracked",
        "bottom_tracked",
        "top_pose_valid",
        "bottom_pose_valid",
    ]
    fields += [f"omega_{name}_camera_{axis}_rad_s" for name in camera_rates for axis in "xyz"]
    offline_status = {}
    if offline is not None:
        offline_status = {
            name: {entry["frame_index"]: entry["status"]
                   for entry in offline.get("shells", {}).get(name, {}).get("frames", [])}
            for name in ("top", "bottom")}
        fields += [f"{name}_offline_status" for name in offline_status]
    mechanical_status = {}
    if mechanical is not None:
        mechanical_status = {
            name: {entry["frame_index"]: entry["status"]
                   for entry in mechanical.get("frames", {}).get(name, [])}
            for name in ("top", "bottom")}
        fields += [f"{name}_mechanical_status" for name in mechanical_status]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, time_s in enumerate(times):
            quality = quality_by_frame.get(index)
            top_q = quality.top if quality else HemisphereQuality()
            bottom_q = quality.bottom if quality else HemisphereQuality()
            writer.writerow(
                {
                    "frame": index,
                    "time_s": time_s,
                    "alpha_rad": motion.alpha[index],
                    "beta_top_rad": motion.beta_top[index],
                    "beta_bottom_rad": motion.beta_bottom[index],
                    "alpha_deg": np.rad2deg(motion.alpha[index]),
                    "beta_top_deg": np.rad2deg(motion.beta_top[index]),
                    "beta_bottom_deg": np.rad2deg(motion.beta_bottom[index]),
                    "alpha_velocity_rad_s": velocities["alpha"][index],
                    "beta_top_velocity_rad_s": velocities["beta_top"][index],
                    "beta_bottom_velocity_rad_s": velocities["beta_bottom"][index],
                    "alpha_top_deg": np.rad2deg(motion.alpha_top[index]),
                    "alpha_bottom_deg": np.rad2deg(motion.alpha_bottom[index]),
                    "gamma_top_deg": np.rad2deg(motion.gamma_top[index]),
                    "gamma_bottom_deg": np.rad2deg(motion.gamma_bottom[index]),
                    "top_inlier_ratio": top_q.inlier_ratio,
                    "bottom_inlier_ratio": bottom_q.inlier_ratio,
                    "top_residual_deg": top_q.mean_residual_deg,
                    "bottom_residual_deg": bottom_q.mean_residual_deg,
                    "top_tracked": top_q.matched_count,
                    "bottom_tracked": bottom_q.matched_count,
                    "top_pose_valid": bool(motion.valid_top[index]),
                    "bottom_pose_valid": bool(motion.valid_bottom[index]),
                    **{f"omega_{name}_camera_{axis}_rad_s": values[index, component]
                       for name, values in camera_rates.items()
                       for component, axis in enumerate("xyz")},
                    **{f"{name}_offline_status": values.get(index, "unavailable")
                       for name, values in offline_status.items()},
                    **{f"{name}_mechanical_status": values.get(index, "unavailable")
                       for name, values in mechanical_status.items()},
                }
            )

    summary = summarize_quality(
        qualities, motion, top_absolute=top_absolute, bottom_absolute=bottom_absolute
    )
    if extra_summary:
        summary = {**summary, **dict(extra_summary)}
    if mechanically_fitted:
        summary["mechanical_tracking"] = mechanical.get("summary", {})
        summary["constraint_metrics_note"] = (
            "Zero gamma and agreement of shell roll are enforced by the model, "
            "not independent accuracy checks. Use image residuals, unresolved frames, "
            "and the preserved unconstrained checks to audit the fit.")
        if unconstrained is not None:
            summary["unconstrained_checks"] = unconstrained.get("summary", {})
    results_json_path = output / "results.json"
    payload = {
        "metadata": dict(metadata or {}),
        "summary": summary,
        "frames": {
            "time_s": times.tolist(),
            "alpha_rad": motion.alpha.tolist(),
            "beta_top_rad": motion.beta_top.tolist(),
            "beta_bottom_rad": motion.beta_bottom.tolist(),
            "alpha_velocity_rad_s": velocities["alpha"].tolist(),
            "beta_top_velocity_rad_s": velocities["beta_top"].tolist(),
            "beta_bottom_velocity_rad_s": velocities["beta_bottom"].tolist(),
            "alpha_top_rad": motion.alpha_top.tolist(),
            "alpha_bottom_rad": motion.alpha_bottom.tolist(),
            "gamma_top_rad": motion.gamma_top.tolist(),
            "gamma_bottom_rad": motion.gamma_bottom.tolist(),
            "valid_top": motion.valid_top.tolist(),
            "valid_bottom": motion.valid_bottom.tolist(),
        },
        "quality": [asdict(item) for item in qualities],
    }
    if temporal is not None:
        payload["temporal"] = dict(temporal)
    if offline is not None:
        payload["offline"] = dict(offline)
    if mechanical is not None:
        payload["mechanical"] = dict(mechanical)
    if unconstrained is not None:
        payload["unconstrained"] = dict(unconstrained)
    if rate_metadata is not None:
        payload["metadata"]["rate_processing"] = rate_metadata
        payload["frames"].update({f"omega_{name}_camera_rad_s": values
                                  for name, values in camera_rates.items()})
    results_json_path.write_text(
        json.dumps(_json_clean(payload), indent=2, allow_nan=False), encoding="utf-8"
    )

    angle_plot = output / "angles.png"
    _plot_angles(angle_plot, times, motion)
    velocity_plot = output / "angular_velocities.png"
    _plot_velocities(velocity_plot, times, velocities)
    diagnostic_plot = output / "diagnostics.png"
    _plot_diagnostics(diagnostic_plot, qualities, motion)
    return {
        "csv": csv_path,
        "json": results_json_path,
        "angles_plot": angle_plot,
        "velocities_plot": velocity_plot,
        "diagnostics_plot": diagnostic_plot,
    }


def _json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _plot_angles(path: Path, times: np.ndarray, motion: DecomposedMotion) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(times, np.rad2deg(motion.alpha), label="alpha (shared)")
    axes[0].plot(times, np.rad2deg(motion.beta_top), label="beta top")
    axes[0].plot(times, np.rad2deg(motion.beta_bottom), label="beta bottom")
    axes[0].set_ylabel("angle (deg)")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)
    axes[1].plot(times, np.rad2deg(motion.alpha_top), label="alpha top")
    axes[1].plot(times, np.rad2deg(motion.alpha_bottom), label="alpha bottom")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("alpha (deg)")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_velocities(
    path: Path, times: np.ndarray, velocities: Mapping[str, np.ndarray]
) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for name, values in velocities.items():
        axis.plot(times, values, label=name.replace("_", " "))
    axis.set(xlabel="time (s)", ylabel="angular velocity (rad/s)")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_diagnostics(
    path: Path, qualities: Sequence[FrameQuality], motion: DecomposedMotion
) -> None:
    frames = np.array([item.frame_index for item in qualities], dtype=int)
    figure, axes = plt.subplots(3, 2, figsize=(11, 10), sharex="col")
    for hemisphere, color in (("top", "tab:cyan"), ("bottom", "tab:pink")):
        values = [getattr(item, hemisphere) for item in qualities]
        axes[0, 0].plot(
            frames,
            [v.mean_residual_deg for v in values],
            label=hemisphere,
            color=color,
        )
        axes[0, 1].plot(
            frames, [v.inlier_ratio for v in values], label=hemisphere, color=color
        )
        axes[1, 0].plot(
            frames, [v.matched_count for v in values], label=hemisphere, color=color
        )
        axes[1, 1].plot(
            frames,
            [v.median_fb_error_px for v in values],
            label=hemisphere,
            color=color,
        )
    axes[2, 0].plot(np.rad2deg(motion.gamma_top), label="top", color="tab:cyan")
    axes[2, 0].plot(
        np.rad2deg(motion.gamma_bottom), label="bottom", color="tab:pink"
    )
    axes[2, 1].plot(
        np.rad2deg(np.abs(motion.alpha_top - motion.alpha_bottom)),
        label="|top - bottom|",
        color="tab:orange",
    )
    axes[0, 0].set_ylabel("inlier residual (deg)")
    axes[0, 1].set_ylabel("inlier ratio")
    axes[1, 0].set_ylabel("tracked features")
    axes[1, 1].set_ylabel("forward/backward error (px)")
    axes[2, 0].set(xlabel="frame", ylabel="gamma residual (deg)")
    axes[2, 1].set(xlabel="frame", ylabel="alpha disagreement (deg)")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
