"""Bound joint image fits while retaining one measured global pose reference.

Every window uses the original calibrated frame and absolute relative-to-first-
image angles. Accepted overlap angles are fixed, so there is no Euler-offset or
rotation-composition approximation when windows start at nonzero roll/spin.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, replace

import numpy as np

from .mechanical import MechanicalResult, _initial_angles, _model_rotations, _refine_mechanical_batch
from .offline import OfflineConfig


def _global_frame_report(entry, start, window_index):
    result = dict(entry)
    result["frame_index"] += start
    if "anchor_frame" in result:
        result["anchor_frame"] += start
    result["window_index"] = window_index
    return result


def _gaps(valid, reports):
    gaps = []
    missing = np.flatnonzero(~valid)
    for indices in np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1):
        if len(indices):
            gaps.append({"start_frame": int(indices[0]), "end_frame_inclusive": int(indices[-1]),
                         "reason_counts": dict(Counter(reports[int(i)]["reason"] for i in indices))})
    return gaps


def refine_mechanical_windows(observations, initial_rotations, initial_valid, K, C, F,
                              *, radius, config, offline_config, top_shell_sign):
    """Fit bounded windows, joining only through accepted image-supported poses."""
    opt = offline_config if isinstance(offline_config, OfflineConfig) else OfflineConfig.from_mapping(offline_config)
    if config.window_frames < opt.min_track_length:
        raise ValueError("mechanical.window_frames must cover offline.min_track_length")
    batch_config = replace(config, window_frames=0, overlap_frames=0)
    names = ("top", "bottom")
    # Validation and repeated overlapping fits must see the same observations,
    # including when callers supply one-shot iterators instead of lists.
    data = ({name: list(observations[name]) for name in names}
            if isinstance(observations, Mapping) and set(observations) == set(names) else observations)
    # Reuse the batch input checks without fitting or inventing initial validity.
    checked = _refine_mechanical_batch(
        data, initial_rotations, initial_valid, K, C, F, radius=radius,
        config=replace(batch_config, enabled=False), offline_config=opt, top_shell_sign=top_shell_sign)
    poses, original_valid = checked.rotations, checked.valid
    n = len(poses["top"])
    frame = np.asarray(F, dtype=float)
    alpha, beta = (_initial_angles(poses, original_valid, frame) if n else
                   (np.empty(0), {name: np.empty(0) for name in names}))
    measured = {name: np.zeros(n, dtype=bool) for name in names}
    reports = {name: [{"frame_index": index, "status": "unresolved",
                       "reason": "not_reached_by_measured_window"} for index in range(n)] for name in names}
    by_frame = {name: defaultdict(list) for name in names}
    for name in names:
        for entry in data[name]:
            by_frame[name][entry.frame_index].append(entry)
    report = {
        **checked.diagnostics, "enabled": True, "config": asdict(config),
        "method": "overlapping_joint_mechanical_reprojection_fit", "frames": reports,
        "components": [], "windows": [], "roll_anchor_frames": [],
        "selected_observation_count": {name: 0 for name in names},
        "selected_observation_count_includes_repeated_overlap": True,
        "reference_policy": "Original first-image gauge, then fixed accepted overlap poses in the original global frame.",
        "limitations": [
            "Pixel residuals and mechanical consistency are not absolute accuracy estimates.",
            "Fixed camera, sphere, calibrated roll axis and starting roll are assumed.",
            "A later shell component needs its own accepted overlap spin reference.",
            "A missing shell spin is not inferred from the other shell's measurements.",
            "A gap with no observed connection to accepted poses remains unresolved.",
            "Accepted overlap angles are fixed; their measurement error is not removed by later windows.",
        ],
    }
    minimum_length = opt.min_track_length
    start = 0
    reached_end = False
    while n - start >= minimum_length:
        length = min(config.window_frames, n - start)
        prior_indices = np.flatnonzero(measured["top"] | measured["bottom"])
        frontier_before = int(prior_indices[-1]) if len(prior_indices) else -1
        if start and not any(np.any(measured[name][start:start+length]) for name in names):
            report["windows"].append({"start_frame": start, "end_frame_exclusive": start+length,
                                      "status": "no_measured_overlap_reference", "fit": None})
            break
        while True:
            end = start + length
            references = {"alpha": {}, "top": {}, "bottom": {}}
            for name in names:
                for index in np.flatnonzero(measured[name][start:end]) + start:
                    references[name][int(index-start)] = float(beta[name][index])
                    references["alpha"][int(index-start)] = float(alpha[index])
                # The first image defines the original relative orientation;
                # it still must pass the batch's actual image-quality gates.
                if start == 0 and n and original_valid[name][0]:
                    references[name][0] = 0.
                    references["alpha"][0] = 0.
            local_data = {
                name: [replace(entry, frame_index=index-start)
                       for index in range(start, end) for entry in by_frame[name][index]]
                for name in names
            }
            local_poses = {name: poses[name][start:end].copy() for name in names}
            local_valid = {name: original_valid[name][start:end] | measured[name][start:end] for name in names}
            for name in names:
                accepted = measured[name][start:end]
                local_poses[name][accepted] = _model_rotations(
                    alpha[start:end][accepted], beta[name][start:end][accepted], frame)
            fitted = _refine_mechanical_batch(
                local_data, local_poses, local_valid, K, C, F, radius=radius,
                config=batch_config, offline_config=opt, top_shell_sign=top_shell_sign,
                fixed_references=references,
                initial_angles=(alpha[start:end], {name: beta[name][start:end] for name in names}))
            window_index = len(report["windows"])
            new_counts = {name: 0 for name in names}
            # The accepted overlap is immutable. In particular, a bad later
            # solve cannot delete an earlier accepted pose or shift its gauge.
            for name in names:
                for local_index, entry in enumerate(fitted.diagnostics["frames"][name]):
                    index = start + local_index
                    if measured[name][index]:
                        continue
                    reports[name][index] = _global_frame_report(entry, start, window_index)
                    if fitted.valid[name][local_index]:
                        alpha[index] = fitted.alpha[local_index]
                        beta[name][index] = fitted.beta[name][local_index]
                        measured[name][index] = True
                        new_counts[name] += 1
                report["selected_observation_count"][name] += fitted.diagnostics["selected_observation_count"][name]
            report["windows"].append({
                "start_frame": start, "end_frame_exclusive": end,
                "status": fitted.diagnostics["status"], "new_valid_frames": new_counts,
                "reference_frames": {name: [start+i for i in refs] for name, refs in references.items()},
                "fit": fitted.diagnostics,
            })
            for component in fitted.diagnostics["components"]:
                mapped = {**component, "window_index": window_index,
                          "anchor_frame": start+component["anchor_frame"],
                          "frame_indices": [start+i for i in component["frame_indices"]]}
                for key in ("accepted_frames",):
                    if key in mapped:
                        mapped[key] = [start+i for i in mapped[key]]
                if "inlier_graph_components" in mapped:
                    mapped["inlier_graph_components"] = [[start+i for i in group]
                                                         for group in mapped["inlier_graph_components"]]
                report["components"].append(mapped)
            report["roll_anchor_frames"].extend(start+i for i in references["alpha"])
            accepted_indices = np.flatnonzero(measured["top"] | measured["bottom"])
            frontier = int(accepted_indices[-1]) if len(accepted_indices) else -1
            if frontier > max(frontier_before, start) or length <= minimum_length:
                break
            # An unsuccessful broad fit must not erase useful shorter nearby
            # intervals. Halving is bounded and retains the same measured gauge.
            smaller = max(minimum_length, length // 2)
            if smaller == length:
                break
            length = smaller
        if frontier <= max(frontier_before, start):
            report["stopped_reason"] = "no_forward_extension_after_bounded_retries"
            break
        if start + length == n and frontier > frontier_before:
            reached_end = True
            break
        if frontier <= start:
            break
        # Back up far enough to retain measured overlap if only the beginning
        # of a window was usable. A failed family of bounded retries stops
        # above, rather than repeatedly shifting the same failed fit by a frame.
        overlap = min(config.overlap_frames, max(1, (frontier-start+1)//2), length-1)
        next_start = min(start+length-overlap, frontier-overlap+1)
        if next_start <= start:
            break
        start = next_start

    for name in names:
        report["summary"][name] = {
            "valid_frames": int(measured[name].sum()),
            "refined_frames": sum(entry["status"] == "refined" for entry in reports[name]),
            "recovered_frames": sum(entry["status"] == "recovered" for entry in reports[name]),
            "unresolved_frames": np.flatnonzero(~measured[name]).tolist(),
        }
    report["roll_anchor_frames"] = sorted(set(report["roll_anchor_frames"]))
    report["gaps"] = {name: _gaps(measured[name], reports[name]) for name in names}
    report["status"] = ("completed" if reached_end else "stopped_at_unresolved_interval")
    if n < minimum_length:
        report["status"] = "insufficient_observations"
    rotations = {name: _model_rotations(alpha, beta[name], frame) for name in names}
    return MechanicalResult(
        rotations, measured, np.where(measured["top"] | measured["bottom"], alpha, np.nan),
        {name: np.where(measured[name], beta[name], np.nan) for name in names}, report)
