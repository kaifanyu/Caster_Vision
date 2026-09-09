"""Headless plots and tabular exports for one metrics run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import MetricResult


def save_metric_diagnostics(
    result: MetricResult,
    output_dir: str | Path,
    *,
    title: str = "Caster contact metrics",
) -> dict[str, Path]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    series = result.series
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()
    axes[0].plot(series.t, np.rad2deg(series.alignment_error_rad))
    axes[0].set_ylabel("alignment error (deg)")
    axes[1].plot(series.t, series.scrub_speed_mps)
    axes[1].set_ylabel("scrub speed (m/s)")
    axes[2].plot(series.t, series.v_long_mps, label="longitudinal")
    axes[2].plot(series.t, series.v_roll_mps, label="rolling")
    axes[2].set_ylabel("speed (m/s)")
    axes[2].legend()
    axes[3].plot(series.t, series.slip_ratio)
    axes[3].set_ylabel("longitudinal slip")
    axes[4].plot(series.t, np.rad2deg(series.heading_angle_rad), label="caster")
    axes[4].plot(series.t, np.rad2deg(series.demand_heading_angle_rad), label="demand")
    axes[4].set_ylabel("heading (deg)")
    axes[4].legend()
    axes[5].plot(series.t, series.confidence)
    axes[5].fill_between(series.t, 0, 1, where=~series.roll_valid, alpha=0.2, color="red", label="roll dropout")
    axes[5].set_ylabel("confidence")
    axes[5].set_ylim(-0.05, 1.05)
    axes[5].legend()
    for axis in axes[-2:]:
        axis.set_xlabel("time (s)")
    figure.suptitle(title)
    figure.tight_layout()
    plot_path = destination / "metrics.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    json_path = destination / "metrics.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    csv_path = destination / "metrics_series.csv"
    names = [
        "t", "heading_angle_rad", "demand_heading_angle_rad", "contact_speed_mps",
        "v_long_mps", "scrub_speed_mps", "v_roll_mps", "alignment_error_rad",
        "slip_ratio", "confidence", "heading_valid", "roll_valid", "slip_valid",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for index in range(len(series.t)):
            writer.writerow([getattr(series, name)[index] for name in names])
    return {"plot": plot_path, "json": json_path, "csv": csv_path}


__all__ = ["save_metric_diagnostics"]
