"""Run the S-A through S-E synthetic acceptance gates."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from common.caster_frame import save_caster_frames
from common.kinematics import CarTrack, contact_velocity_car
from compare.run_comparison import run_comparison
from metrics.metrics import MetricConfig, compute_metrics
from .ball_integration import ball_case_diagnostics, run_ideal_ball_case
from swivel.adapter import SwivelSeries, adapt_swivel_series
from swivel.geometry import SwivelGeometry
from swivel.pipeline import SwivelPipelineResult, run_pipeline
from .generate_swivel import (
    Degradations,
    RenderConfig,
    RenderResult,
    SyntheticGeometry,
    canonical_trajectory,
    render_sequence,
    tracking_trajectory,
)


@dataclass(frozen=True)
class Gate:
    stage: str
    check: str
    measured: float | str
    threshold: str
    passed: bool


def _geometry(rendered: RenderResult) -> SwivelGeometry:
    camera, geom = rendered.camera, rendered.geometry
    return SwivelGeometry(
        camera.T_cam_from_car[:3, :3],
        camera.T_cam_from_car[:3, 3],
        geom.swivel_axis_car,
        geom.hub_offset0_car,
        geom.axle0_car,
        geom.wheel_radius_m,
        geom.wheel_width_m,
    )


def _pipeline(
    rendered: RenderResult,
    fps: float,
    track: Mapping[str, Any],
    estimate: Mapping[str, Any],
) -> SwivelPipelineResult:
    return run_pipeline(
        rendered.frames,
        K=rendered.camera.K,
        geometry=_geometry(rendered),
        marker_size_m=rendered.geometry.tag_size_m,
        fps=fps,
        r_eff=rendered.geometry.wheel_radius_m,
        track_config=track,
        estimate_config=estimate,
        max_tag_reprojection_error_px=3.0,
        max_off_axis_deg=3.0,
        edge_on_min_cos=0.12,
    )


def _tracking_case(
    *,
    frames: int,
    fps: float,
    delta_phi_deg: float,
    delta_psi_deg: float,
    degradations: Degradations,
    track: Mapping[str, Any],
    estimate: Mapping[str, Any],
) -> dict[str, Any]:
    trajectory = tracking_trajectory(
        frames,
        fps,
        delta_phi_deg=delta_phi_deg,
        delta_psi_deg=delta_psi_deg,
    )
    rendered = render_sequence(
        trajectory,
        config=RenderConfig(degradations=degradations),
    )
    result = _pipeline(rendered, fps, track, estimate)
    tag_error = np.rad2deg(result.psi - trajectory.psi)
    tag_valid = result.psi_valid & np.isfinite(tag_error)
    increments = np.array(
        [item.delta_phi if item is not None and item.success else np.nan for item in result.roll_estimates]
    )
    roll_error = np.rad2deg(increments - np.diff(trajectory.phi))
    roll_success = np.array(
        [bool(item is not None and item.success) for item in result.roll_estimates]
    )
    quality_valid = np.array([bool(item.get("roll_valid", False)) for item in result.interval_quality])
    off_axis = np.array(
        [
            np.rad2deg(item.off_axis_residual_rad)
            if item is not None and np.isfinite(item.off_axis_residual_rad)
            else np.nan
            for item in result.roll_estimates
        ]
    )
    reference_error = np.rad2deg(result.reference_phase_error)
    reference_valid = np.isfinite(reference_error)
    return {
        "tag_rmse_deg": float(np.sqrt(np.nanmean(tag_error[tag_valid] ** 2))) if np.any(tag_valid) else float("inf"),
        "tag_p95_abs_deg": float(np.nanpercentile(np.abs(tag_error[tag_valid]), 95)) if np.any(tag_valid) else float("inf"),
        "tag_coverage": float(np.mean(tag_valid)),
        "roll_rmse_deg": float(np.sqrt(np.nanmean(roll_error[roll_success] ** 2))) if np.any(roll_success) else float("inf"),
        "roll_p95_abs_deg": float(np.nanpercentile(np.abs(roll_error[roll_success]), 95)) if np.any(roll_success) else float("inf"),
        "roll_coverage": float(np.mean(quality_valid)),
        "off_axis_median_deg": float(np.nanmedian(off_axis)) if np.any(np.isfinite(off_axis)) else float("inf"),
        "reference_phase_rmse_deg": float(np.sqrt(np.nanmean(reference_error[reference_valid] ** 2))) if np.any(reference_valid) else float("inf"),
        "reference_coverage": float(np.mean([item.valid for item in result.reference_observations])),
        "result": result,
        "trajectory": trajectory,
    }


def _stage_a() -> tuple[list[Gate], dict[str, Any]]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_conventions.py"]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    gate = Gate("S-A", "numeric conventions/plane/axis tests", completed.returncode, "exit code 0", completed.returncode == 0)
    return [gate], {"pytest_output": completed.stdout, "returncode": completed.returncode}


def _stage_metrics(
    fps: float = 60.0,
    *,
    comparison_output: Path | None = None,
    num_frames: int = 61,
    track: Mapping[str, Any] | None = None,
    estimate: Mapping[str, Any] | None = None,
) -> tuple[list[Gate], dict[str, Any]]:
    geom = SyntheticGeometry()
    reports: dict[str, Any] = {}
    gates: list[Gate] = []
    comparison_runs: list[dict[str, Any]] = []
    direct_summaries: dict[tuple[str, str], Mapping[str, Any]] = {}
    cases = {
        "straight": {"alignment_deg": 0.0, "scrub_fraction": 0.0, "slip": 0.0},
        "spin_in_place": {"alignment_deg": 90.0, "scrub_speed": 0.9 * geom.swivel_axis_car[0]},
        "circle": {"alignment_deg": 8.0, "slip": 0.0},
    }
    for name, expected in cases.items():
        trajectory = canonical_trajectory(name, num_frames=num_frames, fps=fps, geometry=geom)
        series = SwivelSeries(
            trajectory.t,
            trajectory.phi,
            trajectory.psi,
            np.ones(len(trajectory.t), dtype=bool),
            np.ones(len(trajectory.t), dtype=bool),
        )
        common = adapt_swivel_series(
            series,
            r_eff=geom.wheel_radius_m,
            axle0_car=geom.axle0_car,
        )
        theta = trajectory.car_theta
        if name == "straight":
            vx = np.full_like(theta, 0.25)
            vy = np.zeros_like(theta)
            omega = np.zeros_like(theta)
        elif name == "spin_in_place":
            vx = np.zeros_like(theta)
            vy = np.zeros_like(theta)
            omega = np.full_like(theta, 0.9)
        else:
            speed, yaw_rate = 0.25, 0.45
            vx = speed * np.cos(theta)
            vy = speed * np.sin(theta)
            omega = np.full_like(theta, yaw_rate)
        car = CarTrack(
            trajectory.t,
            trajectory.car_x,
            trajectory.car_y,
            theta,
            vx,
            vy,
            omega,
        )
        swivel_metric = compute_metrics(
            common,
            car,
            r_arm_car=geom.swivel_axis_car[:2],
            config=MetricConfig(min_contact_speed_mps=1e-6, min_longitudinal_speed_mps=1e-5),
        )
        contact = np.asarray(
            [
                contact_velocity_car(
                    np.array([vx[index], vy[index]]),
                    omega[index],
                    theta[index],
                    geom.swivel_axis_car[:2],
                )
                for index in range(len(theta))
            ]
        )
        if not np.allclose(contact, contact[0], atol=1e-10):
            raise RuntimeError(f"canonical {name} contact velocity is not constant in car axes")
        ball_case = run_ideal_ball_case(
            contact[0],
            num_frames=num_frames,
            fps=fps,
            r_eff=geom.wheel_radius_m,
            track_config=track,
            estimate_config=estimate,
            # Reuse one material pattern: paired maneuvers represent the same
            # physical ball and fixed camera rig, not three lucky textures.
            seed=24_681_357,
        )
        ball_metric = compute_metrics(
            ball_case.frames,
            car,
            r_arm_car=geom.swivel_axis_car[:2],
            config=MetricConfig(min_contact_speed_mps=1e-6, min_longitudinal_speed_mps=1e-5),
        )
        ball_diagnostics = ball_case_diagnostics(ball_case)
        rendered_swivel = render_sequence(
            trajectory,
            config=RenderConfig(degradations=Degradations()),
        )
        swivel_image_result = _pipeline(
            rendered_swivel,
            fps,
            dict(track or {}),
            dict(estimate or {}),
        )
        swivel_image_metric = compute_metrics(
            swivel_image_result.caster_frames,
            car,
            r_arm_car=geom.swivel_axis_car[:2],
            config=MetricConfig(min_contact_speed_mps=1e-6, min_longitudinal_speed_mps=1e-5),
        )
        swivel_image_oracle_frames = adapt_swivel_series(
            series,
            r_eff=geom.wheel_radius_m,
            axle0_car=geom.axle0_car,
            swivel_axis_car=geom.swivel_axis_car,
            hub_offset0_car=geom.hub_offset0_car,
        )
        swivel_image_oracle_metric = compute_metrics(
            swivel_image_oracle_frames,
            car,
            r_arm_car=geom.swivel_axis_car[:2],
            config=MetricConfig(min_contact_speed_mps=1e-6, min_longitudinal_speed_mps=1e-5),
        )
        swivel_image_diagnostics = {
            "source_frame_count": int(swivel_image_result.frame_count),
            "common_frame_count": int(len(swivel_image_result.caster_frames)),
            "roll_coverage": float(
                np.mean([frame.roll_valid for frame in swivel_image_result.caster_frames])
            ),
            "heading_coverage": float(
                np.mean([frame.heading_valid for frame in swivel_image_result.caster_frames])
            ),
        }
        reports[name] = {
            "analytic_truth": {
                "ball_alignment_deg": 0.0,
                "ball_scrub_fraction": 0.0,
                "ball_slip": 0.0,
            },
            "swivel_analytic_frames": swivel_metric.to_dict(),
            "swivel_rendered_pipeline": {
                "metrics": swivel_image_metric.to_dict(),
                "diagnostics": swivel_image_diagnostics,
            },
            "swivel_rendered_oracle": swivel_image_oracle_metric.to_dict(),
            "ball": ball_metric.to_dict(),
            "ball_image_pipeline": ball_diagnostics,
        }
        direct_summaries[(name, "swivel")] = swivel_image_metric.to_dict()["summary"]
        direct_summaries[(name, "ball")] = ball_metric.to_dict()["summary"]
        if comparison_output is not None:
            fixture = comparison_output / "fixtures"
            fixture.mkdir(parents=True, exist_ok=True)
            swivel_path = fixture / f"{name}_swivel.json"
            ball_path = fixture / f"{name}_ball.json"
            save_caster_frames(
                swivel_path,
                swivel_image_result.caster_frames,
                metadata={
                    "device": "swivel",
                    "synthetic": True,
                    "source": "rendered images -> swivel.pipeline -> swivel.adapter",
                    "renderer": rendered_swivel.truth["schema"],
                },
            )
            save_caster_frames(
                ball_path,
                ball_case.frames,
                metadata={
                    "device": "ball",
                    "synthetic": True,
                    "source": "rendered images -> ball.pipeline -> ball.adapter",
                    "renderer": ball_case.rendered.ground_truth["generator"],
                },
            )
            car_path = fixture / f"{name}_car.csv"
            with car_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["t", "x", "y", "theta", "vx_world", "vy_world", "omega"])
                writer.writerows(zip(car.t, car.x, car.y, car.theta, car.vx_world, car.vy_world, car.omega))
            for device, frames_path in (("ball", ball_path), ("swivel", swivel_path)):
                comparison_runs.append(
                    {
                        "id": f"{name}-{device}",
                        "maneuver": name,
                        "device": device,
                        "caster_frames": str(frames_path.relative_to(comparison_output)),
                        "car_track": str(car_path.relative_to(comparison_output)),
                        "r_arm_car_m": geom.swivel_axis_car[:2].tolist(),
                    }
                )
        summary = swivel_metric.summary
        if name == "straight":
            checks = [
                ("straight alignment", summary.mean_abs_alignment_deg, 0.05, "< 0.05 deg"),
                ("straight scrub fraction", summary.scrub_fraction, 1e-4, "< 1e-4"),
                ("straight slip", abs(summary.mean_slip), 0.002, "< 0.002"),
            ]
        elif name == "spin_in_place":
            mean_scrub = float(np.nanmean(swivel_metric.series.scrub_speed_mps))
            checks = [
                ("spin alignment", abs(summary.mean_abs_alignment_deg - expected["alignment_deg"]), 0.1, "error < 0.1 deg"),
                ("spin scrub speed", abs(mean_scrub - expected["scrub_speed"]), 1e-4, "error < 1e-4 m/s"),
            ]
        else:
            checks = [
                ("circle alignment", abs(summary.mean_abs_alignment_deg - expected["alignment_deg"]), 0.15, "error < 0.15 deg"),
                ("circle slip", abs(summary.mean_slip), 0.01, "< 0.01"),
            ]
        gates.extend(Gate("S-D", label, float(value), threshold, bool(value < limit)) for label, value, limit, threshold in checks)
        image_summary = swivel_image_metric.summary
        image_oracle_summary = swivel_image_oracle_metric.summary
        image_checks = [
            (
                f"{name} rendered swivel alignment vs oracle",
                abs(
                    image_summary.mean_abs_alignment_deg
                    - image_oracle_summary.mean_abs_alignment_deg
                ),
                2.0,
                "error < 2 deg",
            ),
            (
                f"{name} rendered swivel scrub fraction vs oracle",
                abs(image_summary.scrub_fraction - image_oracle_summary.scrub_fraction),
                0.04,
                "error < 0.04",
            ),
        ]
        if name != "spin_in_place":
            image_checks.append(
                (
                    f"{name} rendered swivel slip vs oracle",
                    abs(image_summary.mean_slip - image_oracle_summary.mean_slip),
                    0.02,
                    "error < 0.02",
                )
            )
        gates.extend(
            Gate("S6", label, float(value), threshold, bool(value < limit))
            for label, value, limit, threshold in image_checks
        )
        gates.extend(
            [
                Gate(
                    "S6",
                    f"{name} swivel image-pipeline roll coverage",
                    float(swivel_image_diagnostics["roll_coverage"]),
                    ">= 0.90",
                    float(swivel_image_diagnostics["roll_coverage"]) >= 0.90,
                ),
                Gate(
                    "S6",
                    f"{name} swivel image-pipeline heading coverage",
                    float(swivel_image_diagnostics["heading_coverage"]),
                    ">= 0.90",
                    float(swivel_image_diagnostics["heading_coverage"]) >= 0.90,
                ),
            ]
        )
        ball_summary = ball_metric.summary
        gates.extend(
            [
                Gate(
                    "S5",
                    f"{name} ball image-pipeline coverage",
                    float(ball_diagnostics["roll_coverage"]),
                    ">= 0.90",
                    float(ball_diagnostics["roll_coverage"]) >= 0.90,
                ),
                Gate(
                    "S5",
                    f"{name} ball alignment",
                    float(ball_summary.mean_abs_alignment_deg),
                    "< 2 deg",
                    float(ball_summary.mean_abs_alignment_deg) < 2.0,
                ),
                Gate(
                    "S5",
                    f"{name} ball scrub fraction",
                    float(ball_summary.scrub_fraction),
                    "< 0.04",
                    float(ball_summary.scrub_fraction) < 0.04,
                ),
                Gate(
                    "S5",
                    f"{name} ball slip",
                    float(abs(ball_summary.mean_slip)),
                    "< 0.02",
                    float(abs(ball_summary.mean_slip)) < 0.02,
                ),
            ]
        )
    if comparison_output is not None:
        manifest_path = comparison_output / "manifest.yaml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            yaml.safe_dump(
                {
                    "output_dir": "report",
                    "metrics": {
                        "min_contact_speed_mps": 1e-6,
                        "min_longitudinal_speed_mps": 1e-5,
                    },
                    "runs": comparison_runs,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        comparison = run_comparison(manifest_path)
        pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
        for row in comparison.summary_rows:
            pairs.setdefault(str(row["maneuver"]), {})[str(row["device"])] = row
        keys = ("mean_abs_alignment_deg", "total_scrub_m", "scrub_fraction", "rolling_efficiency", "mean_slip")
        differences = []
        for row in comparison.summary_rows:
            expected_summary = direct_summaries[
                (str(row["maneuver"]), str(row["device"]))
            ]
            for key in keys:
                first, second = row.get(key), expected_summary.get(key)
                if first is not None and second is not None and np.isfinite(first) and np.isfinite(second):
                    differences.append(abs(float(first) - float(second)))
        max_difference = max(differences, default=0.0)
        all_pairs_present = all(set(devices) == {"ball", "swivel"} for devices in pairs.values())
        gates.extend(
            [
                Gate(
                    "S6",
                    "both rendered devices in every comparison pair",
                    len(pairs),
                    f"{len(cases)} complete pairs",
                    len(pairs) == len(cases) and all_pairs_present,
                ),
                Gate(
                    "S6",
                    "comparison reproduces direct shared metrics",
                    max_difference,
                    "< 1e-12",
                    max_difference < 1e-12,
                ),
            ]
        )
        reports["comparison_report"] = str(comparison.output_dir)
    return gates, reports


def _characterized_limit(cases: list[dict[str, Any]]) -> str:
    passing = [case["delta_deg"] for case in cases if case["roll_rmse_deg"] < 2.0 and case["roll_coverage"] >= 0.80]
    if not passing:
        return "none"
    maximum = max(passing)
    tested = max(case["delta_deg"] for case in cases)
    return f">={maximum:g}" if maximum == tested else f"{maximum:g}"


def _stage_robustness(
    *,
    frames: int,
    fps: float,
    track: Mapping[str, Any],
    estimate: Mapping[str, Any],
) -> tuple[list[Gate], dict[str, Any]]:
    report: dict[str, Any] = {"noise": [], "delta_phi": [], "delta_psi": [], "degradations": []}
    for noise in (0.25, 0.5, 1.0, 2.0):
        case = _tracking_case(
            frames=frames, fps=fps, delta_phi_deg=2.0, delta_psi_deg=0.5,
            degradations=Degradations(pixel_noise_sigma=noise), track=track, estimate=estimate,
        )
        report["noise"].append({"sigma_px": noise, **{k: v for k, v in case.items() if k not in {"result", "trajectory"}}})
    for delta in (2.0, 5.0, 10.0, 15.0, 20.0):
        phi_case = _tracking_case(
            frames=frames, fps=fps, delta_phi_deg=delta, delta_psi_deg=0.35,
            degradations=Degradations(), track=track, estimate=estimate,
        )
        psi_case = _tracking_case(
            frames=frames, fps=fps, delta_phi_deg=2.0, delta_psi_deg=delta,
            degradations=Degradations(), track=track, estimate=estimate,
        )
        report["delta_phi"].append({"delta_deg": delta, **{k: v for k, v in phi_case.items() if k not in {"result", "trajectory"}}})
        report["delta_psi"].append({"delta_deg": delta, **{k: v for k, v in psi_case.items() if k not in {"result", "trajectory"}}})
    for label, degradation in (
        ("motion_blur", Degradations(motion_blur=True)),
        ("glare", Degradations(glare=True)),
        ("half_density", Degradations(speckle_fraction=0.5)),
    ):
        case = _tracking_case(
            frames=frames, fps=fps, delta_phi_deg=2.0, delta_psi_deg=0.5,
            degradations=degradation, track=track, estimate=estimate,
        )
        report["degradations"].append({"name": label, **{k: v for k, v in case.items() if k not in {"result", "trajectory"}}})

    report["delta_phi_max_deg_per_frame"] = _characterized_limit(report["delta_phi"])
    # The same error/coverage fields score roll recovery under fast swivel.
    report["delta_psi_max_deg_per_frame"] = _characterized_limit(report["delta_psi"])

    # Full swivel sweep verifies that failures are labeled where geometry says
    # a side face is edge-on, while the top marker remains readable.
    edge_frames = max(49, frames)
    trajectory = tracking_trajectory(edge_frames, fps, delta_phi_deg=1.5, delta_psi_deg=0.0)
    trajectory = type(trajectory)(
        trajectory.t,
        trajectory.phi,
        np.linspace(-np.pi, np.pi, edge_frames),
        trajectory.car_x,
        trajectory.car_y,
        trajectory.car_theta,
    )
    rendered = render_sequence(trajectory)
    result = _pipeline(rendered, fps, track, estimate)
    geom = _geometry(rendered)
    predicted = np.array([
        min(
            geom.sidewall_view_confidence(trajectory.psi[i]),
            geom.sidewall_view_confidence(trajectory.psi[i + 1]),
        ) >= 0.12
        for i in range(edge_frames - 1)
    ])
    emitted = np.array([bool(item.get("roll_valid")) for item in result.interval_quality])
    low = ~predicted
    low_flag_agreement = float(np.mean(~emitted[low])) if np.any(low) else 1.0
    report["edge_sweep"] = {
        "predicted_edge_interval_count": int(np.count_nonzero(low)),
        "flagged_edge_interval_count": int(np.count_nonzero(low & ~emitted)),
        "low_view_flag_agreement": low_flag_agreement,
        "tag_coverage": float(np.mean(result.psi_valid)),
    }
    noise_one = next(item for item in report["noise"] if item["sigma_px"] == 1.0)
    gates = [
        Gate("S-E", "1 px tag RMSE", noise_one["tag_rmse_deg"], "< 2 deg", noise_one["tag_rmse_deg"] < 2.0),
        Gate("S-E", "1 px roll RMSE", noise_one["roll_rmse_deg"], "< 2 deg", noise_one["roll_rmse_deg"] < 2.0),
        Gate("S-E", "1 px roll coverage", noise_one["roll_coverage"], ">= 0.80", noise_one["roll_coverage"] >= 0.80),
        Gate("S-E", "edge dropout flagged", low_flag_agreement, ">= 0.90", low_flag_agreement >= 0.90),
        Gate("S-E", "edge sweep tag coverage", report["edge_sweep"]["tag_coverage"], ">= 0.90", report["edge_sweep"]["tag_coverage"] >= 0.90),
        Gate("S-E", "delta-phi limit stated", report["delta_phi_max_deg_per_frame"], "not none", report["delta_phi_max_deg_per_frame"] != "none"),
        Gate("S-E", "delta-psi limit stated", report["delta_psi_max_deg_per_frame"], "not none", report["delta_psi_max_deg_per_frame"] != "none"),
    ]
    return gates, report


def _plot_robustness(report: Mapping[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    noise = report["noise"]
    axes[0].plot([item["sigma_px"] for item in noise], [item["roll_rmse_deg"] for item in noise], "o-", label="roll")
    axes[0].plot([item["sigma_px"] for item in noise], [item["tag_rmse_deg"] for item in noise], "s-", label="tag")
    axes[0].set(xlabel="spatial noise sigma (px)", ylabel="RMSE (deg)", title="Noise")
    axes[0].legend()
    for axis, key, title in ((axes[1], "delta_phi", "Roll speed"), (axes[2], "delta_psi", "Swivel speed")):
        values = report[key]
        axis.plot([item["delta_deg"] for item in values], [item["roll_rmse_deg"] for item in values], "o-")
        axis.axhline(2.0, color="r", linestyle="--", linewidth=1)
        axis.set(xlabel="degrees/frame", ylabel="roll RMSE (deg)", title=title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def format_table(gates: Sequence[Gate]) -> str:
    widths = [6, 34, 14, 16, 6]
    header = ["Stage", "Check", "Measured", "Threshold", "Result"]
    lines = ["  ".join(value.ljust(width) for value, width in zip(header, widths))]
    lines.append("  ".join("-" * width for width in widths))
    for gate in gates:
        measured = f"{gate.measured:.5g}" if isinstance(gate.measured, float) else str(gate.measured)
        row = [gate.stage, gate.check, measured, gate.threshold, "PASS" if gate.passed else "FAIL"]
        lines.append("  ".join(value[:width].ljust(width) for value, width in zip(row, widths)))
    return "\n".join(lines)


def run_validation(
    *,
    output_dir: str | Path,
    quick: bool = False,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    values = dict(config or {})
    track = {
        "max_corners": 420 if quick else 520,
        "quality": 0.005,
        "min_distance_px": 4,
        "klt_win": 25,
        "max_level": 3,
        "fwd_bwd_err_px": 1.5,
        **dict(values.get("track", {})),
    }
    estimate = {
        "ransac_iters": 150 if quick else 250,
        "ransac_inlier_deg": 2.0,
        "min_inliers": 8,
        **dict(values.get("estimate", {})),
    }
    fps = 60.0
    baseline_frames = 25 if quick else 51
    sweep_frames = 13 if quick else 25
    gates, stage_a = _stage_a()
    clean = _tracking_case(
        frames=baseline_frames, fps=fps, delta_phi_deg=2.0, delta_psi_deg=0.5,
        degradations=Degradations(), track=track, estimate=estimate,
    )
    noisy = _tracking_case(
        frames=baseline_frames, fps=fps, delta_phi_deg=2.0, delta_psi_deg=0.5,
        degradations=Degradations(pixel_noise_sigma=1.0), track=track, estimate=estimate,
    )
    gates.extend(
        [
            Gate("S-B", "tag psi clean RMSE", clean["tag_rmse_deg"], "< 1 deg", clean["tag_rmse_deg"] < 1.0),
            Gate("S-B", "tag psi 1px RMSE", noisy["tag_rmse_deg"], "< 2 deg", noisy["tag_rmse_deg"] < 2.0),
            Gate("S-B", "tag clean coverage", clean["tag_coverage"], ">= 0.98", clean["tag_coverage"] >= 0.98),
            Gate("S-C", "roll phi clean RMSE", clean["roll_rmse_deg"], "< 1 deg", clean["roll_rmse_deg"] < 1.0),
            Gate("S-C", "roll phi 1px RMSE", noisy["roll_rmse_deg"], "< 2 deg", noisy["roll_rmse_deg"] < 2.0),
            Gate("S-C", "roll clean coverage", clean["roll_coverage"], ">= 0.90", clean["roll_coverage"] >= 0.90),
            Gate("S-C", "off-axis median", clean["off_axis_median_deg"], "< 1 deg", clean["off_axis_median_deg"] < 1.0),
            Gate("S-C", "reference-dot phase RMSE", clean["reference_phase_rmse_deg"], "< 2 deg", clean["reference_phase_rmse_deg"] < 2.0),
            Gate("S-C", "reference-dot coverage", clean["reference_coverage"], ">= 0.90", clean["reference_coverage"] >= 0.90),
        ]
    )
    metric_gates, metric_report = _stage_metrics(
        fps,
        comparison_output=output / "synthetic_comparison",
        num_frames=31 if quick else 61,
        track=track,
        estimate=estimate,
    )
    gates.extend(metric_gates)
    robust_gates, robust_report = _stage_robustness(
        frames=sweep_frames, fps=fps, track=track, estimate=estimate
    )
    gates.extend(robust_gates)
    _plot_robustness(robust_report, output / "robustness.png")
    report = {
        "schema": "synthetic-swivel-validation-v1",
        "quick": quick,
        "overall_pass": all(gate.passed for gate in gates),
        "gates": [asdict(gate) for gate in gates],
        "stage_a": stage_a,
        "baseline": {key: value for key, value in clean.items() if key not in {"result", "trajectory"}},
        "noise_1px": {key: value for key, value in noisy.items() if key not in {"result", "trajectory"}},
        "metrics": metric_report,
        "robustness": robust_report,
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "PASS_FAIL.txt").write_text(format_table(gates) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.example.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "out" / "synthetic")
    parser.add_argument("--quick", action="store_true", help="Shorter clips; all stages and sweep values still run")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config else {}
    report = run_validation(output_dir=args.output, quick=args.quick, config=config)
    gates = [Gate(**item) for item in report["gates"]]
    print(format_table(gates))
    print()
    print(f"delta phi limit: {report['robustness']['delta_phi_max_deg_per_frame']} deg/frame")
    print(f"delta psi limit: {report['robustness']['delta_psi_max_deg_per_frame']} deg/frame")
    print(f"Report: {Path(args.output).resolve() / 'validation_report.json'}")
    print("OVERALL: " + ("PASS" if report["overall_pass"] else "FAIL"))
    return 0 if report["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
