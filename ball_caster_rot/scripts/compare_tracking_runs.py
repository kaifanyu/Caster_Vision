#!/usr/bin/env python
"""Summarize two experimental filtered runs without equating coverage to accuracy."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from scripts.visualize_fused_axes import _load_results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--additional-validation", type=Path, action="append", default=[],
                        help="Also summarize another shared-observation check, e.g. uniform time sampling")
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline, candidate = (_load_results(path) for path in (args.baseline, args.candidate))
    times = np.asarray(baseline["frames"]["time_s"], float)
    if (not np.array_equal(times, candidate["frames"]["time_s"])
            or Path(baseline["metadata"]["input"]).resolve() != Path(candidate["metadata"]["input"]).resolve()):
        raise ValueError("comparison requires the same video frames and timestamps")
    validations = []
    for report_path in (args.validation, *args.additional_validation):
        item = json.loads(report_path.read_text(encoding="utf-8"))
        if any(Path(item[label]["path"]).resolve() != path.resolve()
               for label, path in (("baseline", args.baseline), ("candidate", args.candidate))):
            raise ValueError("validation report does not describe these two filtered runs")
        validations.append(item)
    validation = validations[0]
    geometry = json.loads(args.geometry.read_text(encoding="utf-8")) if args.geometry else None
    args.output.mkdir(parents=True, exist_ok=True)
    colors = {"vision": "#46b78b", "predicted": "#efab4b", "unresolved": "#475569"}
    names = {"alpha": "Shared roll / moving axes", "beta_top": "Top shell spin", "beta_bottom": "Bottom shell spin"}
    coverage, rows = {}, []
    fig, axes = plt.subplots(3, 1, figsize=(13, 5.8), sharex=True, facecolor="#101820")
    for axis, (name, label) in zip(axes, names.items()):
        axis.set_facecolor("#101820")
        coverage[name] = {}
        for y, (variant, data) in enumerate((("Baseline", baseline), ("Recovery", candidate))):
            status = np.asarray(data["frames"][name + "_status"])
            edges = np.r_[0, np.flatnonzero(status[1:] != status[:-1]) + 1, len(times)]
            for first, stop in zip(edges[:-1], edges[1:]):
                end = times[stop] if stop < len(times) else times[-1]
                axis.broken_barh([(times[first], end-times[first])], (1-y-.28, .56),
                                facecolors=colors[status[first]], edgecolors="none")
            stats = data["summary"][name]
            coverage[name][variant.lower()] = stats
            rows.append(f"<tr><td>{html.escape(label)}</td><td>{variant}</td>"
                        f"<td>{stats['vision_frames']}</td><td>{stats['predicted_frames']}</td>"
                        f"<td>{stats['unresolved_frames']}</td><td>{stats['last_displayed_s']}</td></tr>")
        axis.set_yticks([1, 0], ["Baseline", "Recovery"], color="#e5edf4")
        axis.tick_params(axis="x", colors="#e5edf4")
        axis.set_title(label, loc="left", color="#e5edf4", fontsize=11)
        axis.set_ylim(-.5, 1.5)
        for spine in axis.spines.values():
            spine.set_visible(False)
    axes[-1].set_xlim(times[0], times[-1])
    axes[-1].set_xlabel("Source video time (seconds)", color="#e5edf4")
    fig.suptitle("Frame support before and after image-based recovery", color="white", fontsize=15)
    fig.legend(handles=[Patch(facecolor=color, label=label) for label, color in colors.items()],
               loc="lower center", ncol=3, facecolor="#101820", labelcolor="white", edgecolor="none")
    fig.tight_layout(rect=(0, .05, 1, .96))
    fig.savefig(args.output / "coverage_comparison.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    joint_coverage, joint_rows = {}, []
    for shell in ("top", "bottom"):
        joint_coverage[shell] = {}
        for variant, data in (("Baseline", baseline), ("Recovery", candidate)):
            roll = np.asarray(data["frames"]["alpha_status"])
            spin = np.asarray(data["frames"][f"beta_{shell}_status"])
            displayed = (roll != "unresolved") & (spin != "unresolved")
            vision = (roll == "vision") & (spin == "vision")
            stats = {"displayed_frames": int(displayed.sum()),
                     "both_vision_frames": int(vision.sum()),
                     "includes_prediction_frames": int((displayed & ~vision).sum()),
                     "unresolved_frames": int((~displayed).sum()),
                     "last_displayed_s": float(times[displayed][-1]) if displayed.any() else None}
            joint_coverage[shell][variant.lower()] = stats
            joint_rows.append(f"<tr><td>{shell.title()}</td><td>{variant}</td>"
                              f"<td>{stats['both_vision_frames']}</td>"
                              f"<td>{stats['includes_prediction_frames']}</td>"
                              f"<td>{stats['unresolved_frames']}</td></tr>")

    pixel_metrics = {key: {field: value[field] for field in (
        "total_observations", "aggregate_all_observations", "aggregate_common_support", "holdout_statement")}
        for key, value in validation["observation_sets"].items()}
    for index, extra in enumerate(validations[1:], start=1):
        for key, value in extra["observation_sets"].items():
            pixel_metrics[f"{args.additional_validation[index-1].parent.name} / {key}"] = {
                field: value[field] for field in ("total_observations", "aggregate_all_observations",
                                                "aggregate_common_support", "holdout_statement")}
    report = {"baseline": str(args.baseline.resolve()), "candidate": str(args.candidate.resolve()),
              "frames": len(times), "coverage": coverage, "pixel_validation": pixel_metrics,
              "joint_shell_coverage": joint_coverage,
              "validation_report": str(args.validation.resolve()),
              "additional_validation_reports": [str(path.resolve()) for path in args.additional_validation],
              "geometry_status": geometry.get("status") if geometry else None,
              "geometry_rejection_reasons": geometry.get("rejection_reasons") if geometry else None,
              "known_angle_ground_truth": validation.get("known_angle_ground_truth"),
              "interpretation": "More displayed or vision-updated frames do not prove better angular accuracy. "
                                "Pixel errors test conditional image consistency; fixed X calibration is not tracking."}
    (args.output / "comparison_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    pixel_rows = []
    support_rows = []
    pixel_notes = []
    for name, values in pixel_metrics.items():
        pixel_notes.append(f"<p><strong>{html.escape(name)}</strong>: "
                           f"{html.escape(values['holdout_statement'])}</p>")
        for variant, metrics in values["aggregate_all_observations"].items():
            fmt = lambda value: "n/a" if value is None else f"{value:.3f}"
            support_rows.append(f"<tr><td>{html.escape(name)}</td><td>{variant}</td>"
                                f"<td>{metrics['projectable_count']} / {values['total_observations']}</td>"
                                f"<td>{fmt(metrics['median_px'])}</td><td>{fmt(metrics['p95_px'])}</td></tr>")
        for variant, metrics in values["aggregate_common_support"].items():
            fmt = lambda value: "n/a" if value is None else f"{value:.3f}"
            pixel_rows.append(f"<tr><td>{html.escape(name)}</td><td>{variant}</td>"
                              f"<td>{metrics['projectable_count']}</td><td>{fmt(metrics['median_px'])}</td>"
                              f"<td>{fmt(metrics['p95_px'])}</td></tr>")
    # All files are local artifacts from these runs; relative links work when
    # the workspace directory is moved or opened directly in a browser.
    import os
    relative = lambda path: html.escape(Path(os.path.relpath(path, args.output)).as_posix(), quote=True)
    links = []
    for label, path in (("New interactive axes", args.candidate.parent / "axis_inspection/fused_axes_timeline.html"),
                        ("New overlay video", args.candidate.parent / "axis_inspection/fused_axes_overlay.mp4"),
                        ("Baseline interactive axes", args.baseline.parent / "axis_inspection/fused_axes_timeline.html"),
                        ("Pixel validation report", args.validation)):
        links.append(f"<a href='{relative(path)}'>{label}</a>")
    page = f"""<!doctype html><html lang='en'><meta charset='utf-8'><title>Tracking recovery comparison</title>
<style>body{{font:16px system-ui;background:#101820;color:#e5edf4;max-width:1200px;margin:30px auto;padding:0 18px}}a{{color:#8acaff}}img{{width:100%}}table{{border-collapse:collapse;margin:18px 0}}td,th{{border:1px solid #506070;padding:8px}}p{{line-height:1.5}}</style>
<h1>Tracking recovery comparison</h1><p>{' &nbsp; | &nbsp; '.join(links)}</p>
<p>Green: an accepted visual update to the experimental filter. Orange: prediction only. Gray: unresolved.
This is frame coverage, not a guarantee of correct orientation. Original mechanical acceptance remains separate.</p>
<img src='coverage_comparison.png' alt='Frame support before and after recovery'>
<table><tr><th>Coordinate</th><th>Run</th><th>Vision frames</th><th>Predictions</th><th>Unresolved</th><th>Last displayed, s</th></tr>{''.join(rows)}</table>
<h2>Complete shell orientation available to the overlay</h2>
<p>A shell grid needs both shared roll and that shell's spin. A spin estimate alone does not provide a complete shell orientation.</p>
<table><tr><th>Shell</th><th>Run</th><th>Both vision-updated</th><th>Includes prediction</th><th>Unresolved</th></tr>{''.join(joint_rows)}</table>
<h2>Image error on observations shared by both runs</h2><p>Both columns use the same observed pixels. Missing estimates are included in coverage metrics in the full report.</p>
<table><tr><th>Observation set</th><th>Run</th><th>Common points</th><th>Median px</th><th>p95 px</th></tr>{''.join(pixel_rows)}</table>
<h2>Projection coverage and error on each run's available support</h2>
<p>Missing and nonprojectable estimates stay in the denominator. These error columns can use different subsets; the common-point table above controls for that difference.</p>
<table><tr><th>Observation set</th><th>Run</th><th>Projected / observed points</th><th>Median px</th><th>p95 px</th></tr>{''.join(support_rows)}</table>
{''.join(pixel_notes)}
<p>Geometry check: {html.escape(str(report['geometry_status']))}. Known-angle reference data: {'provided; see full report' if report['known_angle_ground_truth'] else 'not provided; absolute angular accuracy is unverified'}.</p>
<p><a href='comparison_report.json'>Full comparison report</a></p></html>"""
    (args.output / "comparison.html").write_text(page, encoding="utf-8")
    print(args.output / "comparison.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
