#!/usr/bin/env python
"""Fuse one saved visual angle source with a constant-velocity Kalman prior.

Writes a separate experimental schema, never accepted mechanical poses.
Render it with visualize_fused_axes.py, not simulate_measured.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from ballrot.diagnostics import _json_clean
from ballrot.fusion import COORDINATES, visual_angle_observations
from ballrot.motion_filter import MotionFilterConfig, filter_angles


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True, help="Original mechanical results with preserved visual poses")
    p.add_argument("--output", type=Path, required=True, help="Separate experimental output directory")
    p.add_argument("--measurement-source", choices=("unconstrained", "mechanical"), default="unconstrained")
    p.add_argument("--measurement-std-deg", type=float, default=1.5)
    p.add_argument("--max-tilt-deg", type=float, default=5)
    p.add_argument("--max-roll-disagreement-deg", type=float, default=10)
    p.add_argument("--accel-noise-deg-s2", type=float, default=180)
    p.add_argument("--initial-velocity-std-deg-s", type=float, default=180)
    p.add_argument("--innovation-gate-sigma", type=float, default=5)
    p.add_argument("--max-prediction-s", type=float, default=.35)
    p.add_argument("--max-angle-std-deg", type=float, default=10)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    source, output = args.results.resolve(), args.output.resolve()
    if source.parent == output:
        raise ValueError("use a separate output directory to preserve the source run")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not payload.get("mechanical", {}).get("enabled", False):
        raise ValueError("source must identify a mechanical run for the original-validity comparison")
    observations = visual_angle_observations(
        payload, source=args.measurement_source, base_std_deg=args.measurement_std_deg,
        max_tilt_deg=args.max_tilt_deg, max_roll_disagreement_deg=args.max_roll_disagreement_deg,
    )
    baseline = {}
    for key in ("valid_top", "valid_bottom", "alpha_rad"):
        boolean = key.startswith("valid_")
        values = np.asarray(payload["frames"][key], dtype=None if boolean else float)
        if values.shape != observations.timestamps.shape or (boolean and values.dtype.kind != "b"):
            raise ValueError(f"source mechanical reference {key} has invalid shape or type")
        baseline[key] = values
    if np.any((baseline["valid_top"] | baseline["valid_bottom"]) & ~np.isfinite(baseline["alpha_rad"])):
        raise ValueError("source mechanical reference has nonfinite accepted roll")
    config = MotionFilterConfig(
        accel_noise_deg_s2=args.accel_noise_deg_s2,
        initial_velocity_std_deg_s=args.initial_velocity_std_deg_s,
        innovation_gate_sigma=args.innovation_gate_sigma,
        max_prediction_s=args.max_prediction_s, max_angle_std_deg=args.max_angle_std_deg,
    )
    result = filter_angles(observations.timestamps, observations.angles, observations.std, config)
    frames = {"time_s": observations.timestamps.tolist()}
    summary = {}
    for column, name in enumerate(COORDINATES):
        for suffix, values in (
            ("_rad", result.angles), ("_velocity_rad_s", result.velocities),
            ("_std_rad", result.std), ("_status", result.status),
            ("_vision_rad", observations.angles), ("_vision_std_rad", observations.std),
            ("_prediction_age_s", result.age_s), ("_prior_rad", result.predicted_angles),
            ("_measurement_updated", result.updated), ("_measurement_rejected", result.rejected),
            ("_reinitialized", result.reinitialized),
        ):
            frames[name + suffix] = values[:, column].tolist()
        frames[name + "_covariance"] = result.covariance[:, column].tolist()
        supported = result.status[:, column] != "unresolved"
        measured = np.isfinite(observations.angles[:, column])
        summary[name] = {
            **{state + "_frames": int(np.count_nonzero(result.status[:, column] == state))
               for state in ("vision", "predicted", "unresolved")},
            "source_measurement_frames": int(measured.sum()),
            "last_source_measurement_s": float(observations.timestamps[measured][-1]) if measured.any() else None,
            "last_displayed_s": float(observations.timestamps[supported][-1]) if supported.any() else None,
            "innovation_rejections": int(result.rejected[:, column].sum()),
            "reinitializations": int(result.reinitialized[:, column].sum()),
        }
    for shell in ("top", "bottom"):
        frames["mechanical_valid_" + shell] = baseline["valid_" + shell].tolist()
    frames["mechanical_alpha_rad"] = baseline["alpha_rad"].tolist()
    limitations = [
        "Experimental motion prediction plus ONE visual pose input, not a second independent sensor.",
        "Unconstrained source poses bypass mechanical acceptance; extra coverage is not recovered mechanical validity.",
        "Noise and covariance are tuned conditional estimates; they do not quantify systematic calibration error or correlated tracking drift.",
        "The source's unwrapped branches are inherited; whole turns during gaps are not observed.",
        "Missing shell spin is never inferred from the other shell. Long/uncertain predictions are hidden.",
        "No new pixel fit, shell calibration, reference connectivity validation, or backward smoothing is performed.",
    ]
    data = {
        "schema_version": 1, "method": "experimental_angle_kalman",
        "metadata": payload["metadata"], "source_results": str(source),
        "measurement_source": args.measurement_source,
        "parameters": {**asdict(config), "measurement_std_deg": args.measurement_std_deg,
                       "max_tilt_deg": args.max_tilt_deg, "max_roll_disagreement_deg": args.max_roll_disagreement_deg},
        "summary": summary, "limitations": limitations, "frames": frames,
        "measurement_diagnostics": observations.diagnostics,
        "filter_diagnostics": result.diagnostics,
        "covariance_state_order": ["angle_rad", "angular_velocity_rad_s"],
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    destination.write_text(json.dumps(_json_clean(data), indent=2, allow_nan=False), encoding="utf-8")
    report = {key: value for key, value in data.items() if key not in ("frames", "metadata")}
    (output / "fusion_report.json").write_text(json.dumps(_json_clean(report), indent=2, allow_nan=False), encoding="utf-8")
    csv_keys = [key for key in frames if not key.endswith("_covariance")]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(csv_keys)
        writer.writerows(zip(*[_json_clean(frames[key]) for key in csv_keys]))
    print(f"EXPERIMENTAL Kalman filter; visual source: {args.measurement_source}")
    for name, stats in summary.items():
        print(f"  {name}: {stats['vision_frames']} vision updates, {stats['predicted_frames']} predictions, "
              f"{stats['unresolved_frames']} unresolved; last displayed {stats['last_displayed_s']} s")
    print(f"Results: {destination}")
    print("Render with:")
    print(f'.\\.venv\\Scripts\\python.exe .\\scripts\\visualize_fused_axes.py --results "{destination}" --output "{output / "axis_inspection"}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
