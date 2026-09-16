#!/usr/bin/env python
"""Plot saved caster motion, separating camera rotations from calibrated angles."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.reorient_results import reconstruct_camera_rotations

COLORS = {"top": "#c44843", "bottom": "#008a73"}
LABELS = {"top": "Red shell", "bottom": "Green shell"}
INK = "#20334a"
AMBER = "#c78321"


def compute_history(payload: dict) -> dict:
    """Recover native-time rotation metrics without introducing motion at gaps.

    D[i] maps initial camera-coordinate surface vectors into the current view.
    For temporal results, both endpoint poses must be valid before interpreting
    D[i] @ D[i-1].T as one interval's rotation. A recovered absolute pose after
    a gap includes missing motion and must not become a one-frame speed spike.
    Legacy pair-only results retain their original held-accumulation semantics.
    """
    recovered = reconstruct_camera_rotations(payload)
    times = np.asarray(payload["frames"]["time_s"], dtype=float)
    result = {"time_s": times, "mid_time_s": (times[1:] + times[:-1]) / 2,
              "dt_s": np.diff(times), "shells": {},
              "temporal_tracking": recovered.temporal is not None}
    for name in COLORS:
        D = getattr(recovered, f"{name}_absolute")
        valid = getattr(recovered, f"{name}_step_valid")
        valid_step = valid.copy()
        if result["temporal_tracking"]:
            valid_step[1:] &= valid[:-1]
        increments = D[1:] @ D[:-1].transpose(0, 2, 1)
        vectors = np.rad2deg(Rotation.from_matrix(increments).as_rotvec())
        step = np.linalg.norm(vectors, axis=1)
        step[~valid_step[1:]] = np.nan
        omega = vectors / np.diff(times)[:, None]
        omega[~valid_step[1:]] = np.nan
        speed = np.linalg.norm(omega, axis=1)
        observed_arc = np.r_[0.0, np.cumsum(np.nan_to_num(step))]
        incomplete = np.logical_or.accumulate(~valid_step)
        net = np.rad2deg(Rotation.from_matrix(D).magnitude())
        net[~valid] = np.nan
        q = Rotation.from_matrix(D).as_quat()
        for i in range(1, len(q)):
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] *= -1
        quality = {key: np.asarray([item[name][key] for item in payload["quality"]], float)
                   for key in ("matched_count", "inlier_ratio", "mean_residual_deg",
                               "median_fb_error_px")}
        min_inliers = 8  # existing acceptance threshold, not a confidence bound
        caution = ((quality["matched_count"] < min_inliers)
                   | ~np.isfinite(quality["inlier_ratio"]) | (quality["inlier_ratio"] < .7)
                   | ~np.isfinite(quality["mean_residual_deg"]) | (quality["mean_residual_deg"] > .5)
                   | ~np.isfinite(quality["median_fb_error_px"]) | (quality["median_fb_error_px"] > 1))
        status = np.where(~valid_step[1:], 2, np.where(caution, 1, 0))
        result["shells"][name] = {
            "R_camera": D, "quaternion_xyzw": q, "valid_step": valid_step,
            "valid_pose": valid,
            "incomplete_history": incomplete, "net_deg": net,
            "step_deg": step, "omega_camera_deg_s": omega,
            "speed_deg_s": speed, "observed_arc_deg": observed_arc,
            "quality": quality, "status": status,
        }
    return result


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#bac4cf",
        "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK, "ytick.color": INK,
        "grid.color": "#dce2e9", "grid.linewidth": .65,
        "savefig.facecolor": "white", "svg.fonttype": "none",
    })


def _setup(axes, history):
    for ax in np.atleast_1d(axes).flat:
        ax.grid(axis="y", alpha=.8)
        ax.set_xlim(history["time_s"][0], history["time_s"][-1])
        ax.set_axisbelow(True)


def _footer(fig, text):
    fig.text(.09, .018, text, va="bottom", fontsize=8.5, color="#53647a")


def _trajectory(ax, times, values, shell, color, label):
    """Dash after the first lost step, keeping individual failed frames missing."""
    values = np.asarray(values, float).copy()
    values[~shell.get("valid_pose", shell["valid_step"])] = np.nan
    complete = ~shell["incomplete_history"]
    ax.plot(times, np.where(complete, values, np.nan), color=color, lw=1.6, label=label)
    dashed = np.where(~complete, values, np.nan)
    lost = np.flatnonzero(~complete)
    if len(lost) and lost[0] > 0:
        dashed[lost[0] - 1] = values[lost[0] - 1]
    ax.plot(times, dashed, color=color, lw=1.4, ls="--")


def _failures(ax, history, *, annotate=False):
    for offset, (name, shell) in enumerate(history["shells"].items()):
        bad = np.flatnonzero(~shell["valid_step"])
        if len(bad):
            first = history["time_s"][bad[0]]
            ax.axvline(first, color=COLORS[name], ls=":", lw=1, alpha=.8)
            if annotate:
                ax.text(first + .15, .97 - .12 * offset,
                        f"First {name.replace('top', 'red').replace('bottom', 'green')} loss: {first:.2f} s",
                        transform=ax.get_xaxis_transform(), color=COLORS[name],
                        fontsize=8, va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=.8))


def overview_figure(history):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.2), sharex=True)
    fig.suptitle("Ball caster motion through time", x=.09, ha="left", fontsize=21, fontweight="bold")
    fig.text(.09, .927, "Camera-relative estimates for each shell • saved frame timestamps • independent of roll/swivel-axis calibration", fontsize=10)
    titles = ["1  Orientation difference from the first frame",
              "2  Rotation speed during each captured interval",
              "3  Accumulated magnitude of successfully tracked rotation"]
    for ax, title in zip(axes, titles): ax.set_title(title, loc="left", pad=9)
    for name, shell in history["shells"].items():
        _trajectory(axes[0], history["time_s"], shell["net_deg"], shell, COLORS[name], LABELS[name])
        axes[1].plot(history["mid_time_s"], shell["speed_deg_s"], color=COLORS[name], lw=1.1, alpha=.85)
        _trajectory(axes[2], history["time_s"], shell["observed_arc_deg"], shell, COLORS[name], LABELS[name])
    axes[0].set_ylim(0, 190)
    axes[0].set_ylabel("Shortest angle (deg)")
    axes[0].legend(loc="lower right", frameon=True)
    axes[1].set_ylabel("Speed (deg/s)");axes[1].set_ylim(bottom=0)
    axes[2].set_ylabel("Observed arc (deg)");axes[2].set_xlabel("Time from first frame (s)")
    _setup(axes, history);_failures(axes[0], history, annotate=True)
    _footer(fig, "Solid = before the shell's first failed step; dashed = accumulated history contains missing motion. Gaps are not zero motion.\n"
            "Shortest angle is limited to 180° and does not count turns. Observed arc is unsigned, includes tracking noise, and excludes failed steps.")
    fig.subplots_adjust(left=.09, right=.97, bottom=.10, top=.875, hspace=.36)
    return fig


def angles_figure(payload, history):
    f = payload["frames"];t = history["time_s"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 10.3), sharex=True)
    fig.suptitle("Roll and shell spin in the calibrated frame", x=.09, ha="left", fontsize=20, fontweight="bold")
    fig.text(.09, .929, f"Current initial-roll offset: {payload['metadata'].get('initial_roll_deg', 0):g}° • angles start at zero • calibration-dependent", fontsize=10)
    titles = ["1  Roll estimated independently from each shell",
              "2  Spin of each shell about its own axis",
              "3  Difference between the two roll estimates",
              "4  Off-model rotation (gamma)"]
    for ax, title in zip(axes, titles): ax.set_title(title, loc="left", pad=9)
    for name, shell in history["shells"].items():
        for ax, component in [(axes[0], "alpha"), (axes[1], "beta"), (axes[3], "gamma")]:
            values = np.rad2deg(np.asarray(f[f"{component}_{name}_rad"], float))
            if component == "gamma": values = np.abs(values)
            _trajectory(ax, t, values, shell, COLORS[name], LABELS[name])
    difference = np.rad2deg(np.abs(np.asarray(f["alpha_top_rad"], float)-np.asarray(f["alpha_bottom_rad"], float)))
    joint = {"valid_step": history["shells"]["top"]["valid_step"] & history["shells"]["bottom"]["valid_step"],
             "valid_pose": history["shells"]["top"]["valid_pose"] & history["shells"]["bottom"]["valid_pose"],
             "incomplete_history": history["shells"]["top"]["incomplete_history"] | history["shells"]["bottom"]["incomplete_history"]}
    _trajectory(axes[2], t, difference, joint, INK, "Absolute disagreement")
    for ax in axes[2:]:
        ax.axhline(1, color=AMBER, lw=1, ls=":", label="1° reference")
        ax.set_ylim(bottom=0)
    for ax, label in zip(axes, ["Roll (deg)", "Shell spin (deg)", "Difference (deg)", "|Gamma| (deg)"]): ax.set_ylabel(label)
    axes[0].legend(loc="lower left", frameon=True)
    axes[2].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Time from first frame (s)")
    _setup(axes, history)
    _footer(fig, "Dashed segments contain earlier missed motion. These curves are diagnostic estimates, not validated mechanical angles.\n"
            "The existing Rx(alpha) Rz(beta) model expects shared roll and small gamma; the 1° line is a consistency reference, not an error bound.")
    fig.subplots_adjust(left=.09, right=.97, bottom=.09, top=.875, hspace=.41)
    return fig


def quality_figure(history):
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True,
                            gridspec_kw={"height_ratios": [.6, 1, 1, 1, 1]})
    fig.suptitle("Where the rotation estimates lose support", x=.10, ha="left", fontsize=20, fontweight="bold")
    fig.text(.10, .931, "Per-frame-pair diagnostics; passing a numerical solve does not guarantee accurate motion", fontsize=10)
    status = np.vstack([s["status"] for s in history["shells"].values()])
    axes[0].pcolormesh(history["time_s"], [0, 1, 2], status,
                       cmap=ListedColormap(["#bed7df", "#e6b657", "#ad3343"]), vmin=0, vmax=2, shading="flat")
    axes[0].set_yticks([.5, 1.5], ["Red", "Green"]);axes[0].invert_yaxis()
    axes[0].set_title("Blue = passes plotted checks   |   amber = solved with caution   |   dark red = interval unavailable", loc="left", fontsize=10)
    specs = [("matched_count", "Tracked features", None), ("inlier_ratio", "Inlier fraction", .7),
             ("mean_residual_deg", "Fit residual (deg)", .5)]
    for ax, (key, label, limit) in zip(axes[1:4], specs):
        for name, shell in history["shells"].items():
            ax.plot(history["mid_time_s"], shell["quality"][key], color=COLORS[name], lw=1.15, label=LABELS[name])
        if limit is not None: ax.axhline(limit, color=AMBER, ls=":", lw=1)
        ax.set_ylabel(label);ax.set_ylim(bottom=0)
    axes[1].legend(loc="upper left", fontsize=8)
    axes[2].set_ylim(0, 1.04)
    for name, shell in history["shells"].items():
        axes[4].plot(history["mid_time_s"], shell["step_deg"], color=COLORS[name], lw=1.1)
    axes[4].axhline(5, color=AMBER, ls="--", lw=1, label="5°/frame synthetic characterization")
    axes[4].legend(loc="upper left", fontsize=8)
    axes[4].set_ylabel("Step rotation (deg)");axes[4].set_ylim(bottom=0)
    axes[4].set_xlabel("Time from first frame (s)")
    _setup(axes[1:], history)
    _footer(fig, "Caution: tracks <8, inlier fraction <0.7, mean fit residual >0.5°, median forward/backward error >1 px, or a non-finite diagnostic.\n"
            "The synthetic 5°/frame characterization is not a hard physical limit. Blue segments are not ground-truth validation.")
    fig.subplots_adjust(left=.10, right=.97, bottom=.09, top=.87, hspace=.35)
    return fig


def camera_rate_figure(history):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Rotation resolved along the camera axes", x=.09, ha="left", fontsize=20, fontweight="bold")
    fig.text(.09, .926, "Computed directly from consecutive 3D rotation matrices; independent of roll/swivel-axis calibration", fontsize=10)
    for i, (ax, title) in enumerate(zip(axes, ["Camera X: image-right", "Camera Y: image-down", "Camera Z: forward into the scene"])):
        ax.set_title(title, loc="left")
        for name, shell in history["shells"].items():
            ax.plot(history["mid_time_s"], shell["omega_camera_deg_s"][:, i], color=COLORS[name], lw=1.1, label=LABELS[name])
        ax.axhline(0, color="#8a99aa", lw=.8)
        ax.set_ylabel("Angular rate (deg/s)")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[-1].set_xlabel("Time from first frame (s)");_setup(axes, history)
    _footer(fig, "Each value is the equivalent rotation vector for one captured interval divided by that interval's actual duration; right-hand signs.\n"
            + ("Missing poses and recovery spans are excluded; a recovered absolute pose does not measure the motion inside the gap."
               if history["temporal_tracking"] else
               "Failed intervals are missing. Successful local increments after a gap can remain useful even though the accumulated orientation has lost motion."))
    fig.subplots_adjust(left=.09, right=.97, bottom=.11, top=.865, hspace=.32)
    return fig


def export_data(history, output):
    arrays = {"time_s": history["time_s"], "interval_mid_time_s": history["mid_time_s"]}
    for name, shell in history["shells"].items():
        for key in ("R_camera", "quaternion_xyzw", "valid_step", "valid_pose", "incomplete_history", "omega_camera_deg_s", "step_deg"):
            arrays[f"{name}_{key}"] = shell[key]
    np.savez_compressed(output / "rotation_history.npz", **arrays)
    columns = ["frame", "time_s", "interval_mid_time_s", "dt_s"]
    metrics = ["valid_step", "valid_pose", "incomplete_history", "shortest_angle_deg", "observed_arc_deg",
               "step_deg", "speed_deg_s", "omega_camera_x_deg_s", "omega_camera_y_deg_s", "omega_camera_z_deg_s",
               "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w"]
    columns += [f"{name}_{m}" for name in COLORS for m in metrics]
    with (output / "motion_history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f);writer.writerow(columns)
        for i, t in enumerate(history["time_s"]):
            row = [i, t, history["mid_time_s"][i-1] if i else "", history["dt_s"][i-1] if i else ""]
            for shell in history["shells"].values():
                row.extend([bool(shell["valid_step"][i]), bool(shell["valid_pose"][i]), bool(shell["incomplete_history"][i]), shell["net_deg"][i], shell["observed_arc_deg"][i],
                            shell["step_deg"][i-1] if i else np.nan, shell["speed_deg_s"][i-1] if i else np.nan])
                row.extend(shell["omega_camera_deg_s"][i-1] if i else [np.nan]*3)
                row.extend(shell["quaternion_xyzw"][i])
            writer.writerow(["" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v for v in row])


def summarize(history, payload):
    def number(value):
        return float(value) if np.isfinite(value) else None

    def percentile(values, p):
        finite = values[np.isfinite(values)]
        return float(np.percentile(finite, p)) if len(finite) else None

    summary = {"frames": len(history["time_s"]), "last_timestamp_s": float(history["time_s"][-1]),
               "timing_source": payload["metadata"].get("timing_source", "unspecified in source"), "shells": {},
               "interpretation": "Camera-relative per-shell estimates. No chassis translation or world orientation inferred. Observed arc is not mechanical wheel turns."}
    for name, shell in history["shells"].items():
        failures = np.flatnonzero(~shell["valid_step"])
        summary["shells"][name] = {
            "failed_pairs": len(failures),
            "unresolved_poses": int(np.count_nonzero(~shell["valid_pose"])),
            "first_failure_time_s": float(history["time_s"][failures[0]]) if len(failures) else None,
            "final_shortest_angle_deg": number(shell["net_deg"][-1]),
            "observed_arc_deg": float(shell["observed_arc_deg"][-1]),
            "successful_speed_median_deg_s": percentile(shell["speed_deg_s"], 50),
            "successful_speed_p95_deg_s": percentile(shell["speed_deg_s"], 95),
            "successful_step_p95_deg": percentile(shell["step_deg"], 95),
            "status_counts": {label: int(np.count_nonzero(shell["status"] == i)) for i, label in enumerate(["passes_plotted_checks", "solved_with_caution", "failed"])},
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-clip", action="store_true", help="Include actual frame timeline from metadata.input")
    args = parser.parse_args(argv)
    output = args.output.resolve();output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    history = compute_history(payload);_style()
    figures = [("01_motion_overview", overview_figure(history)),
               ("02_calibrated_angles", angles_figure(payload, history)),
               ("03_tracking_quality", quality_figure(history)),
               ("04_camera_angular_velocity", camera_rate_figure(history))]
    with PdfPages(output / "motion_history.pdf") as pdf:
        for name, fig in figures:
            fig.savefig(output / f"{name}.png", dpi=180)
            fig.savefig(output / f"{name}.svg")
            pdf.savefig(fig);plt.close(fig)
        if args.with_clip:
            from scripts.plot_clip_timeline import make_clip_timeline
            paths = make_clip_timeline(payload, output)
            fig, ax = plt.subplots(figsize=(12, 9));ax.axis("off")
            ax.imshow(plt.imread(paths["png"]));fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            pdf.savefig(fig);plt.close(fig)
    export_data(history, output)
    summary = summarize(history, payload)
    summary["source_results"] = str(args.results.resolve())
    (output / "motion_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(history['time_s'])} frames of plots, PDF, CSV and matrices to {output}")
    for name, item in summary["shells"].items():
        speed = item['successful_speed_median_deg_s']
        speed_text = f"{speed:.1f} deg/s" if speed is not None else "unavailable"
        print(f"{LABELS[name]}: {item['failed_pairs']} failed steps; observed arc {item['observed_arc_deg']:.1f} deg; median successful speed {speed_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
