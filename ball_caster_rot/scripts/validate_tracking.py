#!/usr/bin/env python
"""Compare saved trajectories using one fresh set of image correspondences.

Example:
  python scripts/validate_tracking.py --baseline out/real_fused/results.json \
    --candidate out/new/results.json --candidate-source fused --output out/validation \
    --sanity-exclusion-px 4

Default exclusion is 30 full-resolution pixels at both endpoints of a match,
against the union of saved tracking pixels for both methods. A separate optional
small-exclusion sanity check is not called independent held-out validation.
Known-angle CSV: frame_index or time_s, alpha_deg, beta_top_deg, beta_bottom_deg;
angles must use the baseline's saved effective initial-frame convention, which
is held fixed for both methods even when candidate calibration differs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from ballrot.axis_validation import (archived_observations, check_comparable, compare_observations,
    default_pairs, distance_from_training, known_angle_errors, load_trajectory, observe_pair, paint_masks)
from ballrot.camera import undistort_image
from ballrot.io_frames import FrameSource


def _pairs(value, count):
    try:
        pairs = [tuple(map(int, token.split(":"))) for token in value.split(",")]
    except ValueError as exc:
        raise ValueError("Pairs must use source:target frame indices separated by commas") from exc
    if any(len(pair) != 2 or pair[0] == pair[1] or min(pair) < 0 or max(pair) >= count for pair in pairs):
        raise ValueError("Pair indices must identify two different available frames")
    return list(dict.fromkeys(pairs))


def _contact_sheet(report, images, times, circle, output, title):
    available_pairs = list(dict.fromkeys((row["source_frame"], row["target_frame"]) for row in report["pairs"]))
    target_times = (2.4, 4.68, 15.31)
    chosen = [min(available_pairs, key=lambda pair: abs(times[pair[0]]-time) + .001*abs(pair[1]-pair[0])) for time in target_times]
    chosen.append(max(available_pairs, key=lambda pair: times[pair[1]]-times[pair[0]]))
    fig, axes = plt.subplots(4, 2, figsize=(15, 17), facecolor="#101820")
    for row_index, (first, second) in enumerate(chosen):
        evidence = [row for row in report["pairs"] if row["source_frame"] == first and row["target_frame"] == second]
        image = images[second]
        u, v, radius = circle
        x0, y0, x1, y1 = max(0, int(u-1.25*radius)), max(0, int(v-1.2*radius)), min(image.shape[1], int(u+1.25*radius)), min(image.shape[0], int(v+1.2*radius))
        for column, method in enumerate(("baseline", "candidate")):
            panel = axes[row_index, column]
            panel.imshow(cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
            captions = []
            for shell in evidence:
                values = shell[method]
                observed = np.asarray(shell["observed_uv"], float).reshape(-1, 2)
                selected = np.linspace(0, len(observed)-1, min(40, len(observed)), dtype=int)
                for index in selected:
                    actual = observed[index] - [x0, y0]
                    panel.plot(*actual, marker="o", markersize=3.5, markerfacecolor="none", markeredgecolor="white", markeredgewidth=.8)
                    predicted = values["predicted_uv"][index]
                    if predicted is None:
                        continue
                    predicted = np.asarray(predicted) - [x0, y0]
                    color = "#75ed8e" if values["error_px"][index] <= report["threshold_px"] else "#ff746e"
                    panel.plot([actual[0], predicted[0]], [actual[1], predicted[1]], color=color, linewidth=.9)
                    panel.plot(*predicted, marker="x", color=color, markersize=4)
                median = "--" if values["median_px"] is None else f"{values['median_px']:.2f}px"
                captions.append(f"{shell['shell']}: {values['projectable_count']}/{len(observed)} projected; median {median}"
                                + ("; POSE UNAVAILABLE" if not values["pose_available"] else ""))
            panel.set_title(f"{method.upper()} | {times[first]:.3f}s -> {times[second]:.3f}s\n" + "\n".join(captions),
                            color="white", fontsize=9)
            panel.set_xlim(0, x1-x0); panel.set_ylim(y1-y0, 0); panel.set_axis_off()
    fig.suptitle(title, color="white", fontsize=17, y=.993)
    fig.text(.5, .016, "White circles: image-only target matches. Crosses: pose predictions. Green <=2px; red >2px. Same observations in both columns.\nNo pixels/poses are filled across gaps. Pairwise image consistency is not ground-truth axis accuracy.",
             color="#d5e2ed", ha="center", fontsize=9)
    fig.subplots_adjust(left=.02, right=.98, top=.95, bottom=.055, hspace=.27, wspace=.08)
    fig.savefig(output, dpi=145, facecolor=fig.get_facecolor()); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-source", choices=("auto", "unconstrained", "mechanical", "fused"), default="auto")
    parser.add_argument("--candidate-source", choices=("auto", "unconstrained", "mechanical", "fused"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Color thresholds only; poses/camera always come from saved results")
    parser.add_argument("--pairs", help="Fixed source:target frame indices, e.g. 68:69,132:134,437:439,0:556")
    parser.add_argument("--exclusion-px", type=float, default=30.)
    parser.add_argument("--sanity-exclusion-px", type=float, help="Additional explicitly non-independent small-exclusion check")
    parser.add_argument("--max-corners", type=int, default=300)
    parser.add_argument("--observations", type=Path, help="Reuse previously exported validation_observations.json without rematching")
    parser.add_argument("--known-angles", type=Path, help="External reference CSV; no ground truth is assumed if omitted")
    args = parser.parse_args(argv)
    if args.exclusion_px < 30 or args.max_corners < 1:
        parser.error("--exclusion-px must be >=30 for the held-out base-patch check; --max-corners must be positive")
    if args.sanity_exclusion_px is not None and not 0 <= args.sanity_exclusion_px < args.exclusion_px:
        parser.error("--sanity-exclusion-px must be nonnegative and below --exclusion-px")
    baseline, candidate = load_trajectory(args.baseline, args.baseline_source), load_trajectory(args.candidate, args.candidate_source)
    check_comparable(baseline, candidate)
    pairs = _pairs(args.pairs, len(baseline.times)) if args.pairs else default_pairs(baseline.times)
    needed = {index for pair in pairs for index in pair}
    config_path = args.config or Path(baseline.metadata["config"])
    segment_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["segment"]
    K, dist = np.asarray(baseline.metadata["K"], float), np.asarray(baseline.metadata["dist"], float)
    clip = Path(baseline.metadata["input"])
    images = {record.index: undistort_image(record.image, K, dist)
              for record in FrameSource(clip, max_frames=max(needed)+1) if record.index in needed}
    if set(images) != needed:
        raise ValueError("Source clip does not contain all validation frames")
    training, archives = archived_observations([args.baseline, args.candidate], needed)
    masks = {index: paint_masks(image, baseline.metadata["circle"], segment_config) for index, image in images.items()}
    provenance = {"source_clip": str(clip.resolve()), "K": K.tolist(), "dist": dist.tolist(),
                  "time_s": baseline.times.tolist(), "pairs": pairs, "training_archives": archives,
                  "segment_config": segment_config,
                  "image_mask": {"source_circle": baseline.metadata["circle"], "radius_scale": 1.1,
                                 "component_min_area_px": 35, "component_max_area_over_radius_squared": .025,
                                 "component_max_aspect_ratio": 6, "paper_ring_min_fraction": .4,
                                 "paper_max_saturation": 64, "paper_min_value": 135},
                  "matching_method": "fresh_ShiTomasi_LK_forward_backward_patch_NCC; no pose initialization or geometry RANSAC"}
    if args.observations:
        frozen = json.loads(args.observations.read_text(encoding="utf-8"))
        saved = frozen["provenance"]
        if (saved["source_clip"] != provenance["source_clip"]
                or not np.array_equal(saved["pairs"], pairs)
                or not np.allclose(saved["K"], K) or not np.allclose(saved["dist"], dist)
                or not np.allclose(saved["time_s"], baseline.times)):
            raise ValueError("Frozen observations do not match requested video/camera/frame pairs")
        sets = frozen["observation_sets"]
        provenance["reused_from"] = str(args.observations.resolve())
        provenance["frozen_observation_provenance"] = saved
        # Frozen observations remain unchanged. New training pixels can destroy
        # hold-out status, which must be disclosed rather than silently filtered.
        provenance["frozen_training_archives_match"] = saved["training_archives"] == archives
    else:
        sets = {}
        radii = {"heldout_base_patches": args.exclusion_px}
        if args.sanity_exclusion_px is not None:
            radii["overlapping_patch_sanity"] = args.sanity_exclusion_px
        for name, exclusion in radii.items():
            records = []
            for first, second in pairs:
                for shell in ("top", "bottom"):
                    observation = observe_pair(images[first], images[second], masks[first][shell], masks[second][shell],
                                               training[first], training[second], exclusion_px=exclusion, max_corners=args.max_corners)
                    records.append({"source_frame": first, "target_frame": second, "shell": shell, **observation})
            sets[name] = records
    args.output.mkdir(parents=True, exist_ok=True)
    frozen_path = args.output / "validation_observations.json"
    frozen_path.write_text(json.dumps({"schema_version": 1, "provenance": provenance, "observation_sets": sets}, indent=2)+"\n", encoding="utf-8")
    result = {"schema_version": 1, "baseline": {"path": str(baseline.path), "source": baseline.source, "geometry": baseline.geometry, "gap_fraction": baseline.gap, "pivot": baseline.center.tolist()},
              "candidate": {"path": str(candidate.path), "source": candidate.source, "geometry": candidate.geometry, "gap_fraction": candidate.gap, "pivot": candidate.center.tolist()},
              "provenance": provenance, "observation_sets": {}, "known_angle_ground_truth": None,
              "limitations": [
                  "Fresh image correspondences test relative projected pixels, not absolute axis calibration or true motion.",
                  "Source pixels are re-anchored on each method's assumed shell; short pairs may conceal accumulated drift.",
                  "30px exclusion separates 21px base patches; pyramid context, paint identity and the same video remain shared dependencies.",
                  "Only saved training observations can be excluded; discarded/unsaved tracker pixels are unknown.",
                  "Repeated markings, blur and large motion can still produce wrong forward/backward-consistent matches.",
                  "Long-baseline pairs are unconditional rematch attempts, not assumed known physical returns.",
                  "Median/p95 concern projectable observations; overall coverage and common-support scores must accompany them.",
                  "Ray misses, source view angles >70 degrees, cap violations, hidden surfaces and unavailable poses remain failed support in the all-observation denominator.",
                  "Experimental fused predictions retain their source/status labels and are not promoted to mechanical acceptance."]}
    if not archives:
        result["limitations"].append("No saved observation archive was found: no set can claim exclusion from training pixels.")
    for name, records in sets.items():
        comparison = compare_observations(baseline, candidate, records)
        overlap_counts = {method: 0 for method in ("source", "target")}
        for row in records:
            radius = row["diagnostics"]["exclusion_radius_px"]
            for endpoint in overlap_counts:
                pixels = np.asarray(row[f"{endpoint}_uv"], float).reshape(-1, 2)
                overlap_counts[endpoint] += int(np.count_nonzero(distance_from_training(pixels, training[row[f"{endpoint}_frame"]]) < radius))
        comparison["current_archive_exclusion_violations"] = overlap_counts
        excluded = name == "heldout_base_patches" and archives and not any(overlap_counts.values())
        comparison["holdout_statement"] = (
            "Base-patch centers excluded from available saved training observations at both endpoints; not ground truth."
            if excluded else
            "Fresh correspondence sanity check only; independent held-out pixels are not established.")
        result["observation_sets"][name] = comparison
        image_path = args.output / f"{name}_residuals.png"
        _contact_sheet(comparison, images, baseline.times, baseline.metadata["circle"], image_path,
                       "Excluded base-patch residuals (saved training pixels only)" if excluded else "Fresh-match sanity residuals (independence not established)")
        comparison["contact_sheet"] = str(image_path.resolve())
        print(f"{name}: {comparison['total_observations']} image-only matches")
        for label, metrics in comparison["aggregate_all_observations"].items():
            print(f"  {label}: coverage={metrics['coverage']}; median_px={metrics['median_px']}; p95_px={metrics['p95_px']}")
    if args.known_angles:
        reference_frame = baseline.metadata["R_bc"]
        result["known_angle_ground_truth"] = {
            "reference_convention": "External CSV angles must be relative to the baseline's saved R_bc; that single frame is used for both methods.",
            "baseline": known_angle_errors(baseline, args.known_angles, reference_frame=reference_frame),
            "candidate": known_angle_errors(candidate, args.known_angles, reference_frame=reference_frame)}
    report_path = args.output / "validation_report.json"
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Report: {report_path}\nFrozen observations: {frozen_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
