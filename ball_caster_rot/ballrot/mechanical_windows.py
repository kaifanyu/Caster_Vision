"""Bound joint image fits while retaining one measured global pose reference.

Every window uses the original calibrated frame and absolute relative-to-first-
image angles. Accepted overlap angles are fixed, so there is no Euler-offset or
rotation-composition approximation when windows start at nonzero roll/spin.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, replace
import logging

import numpy as np

from .mechanical import MechanicalResult, _initial_angles, _model_rotations, _refine_mechanical_batch
from .offline import OfflineConfig

logger = logging.getLogger(__name__)


def _global_frame_report(entry, start, window_index):
    result = dict(entry)
    result["frame_index"] += start
    if result.get("anchor_frame") is not None:
        result["anchor_frame"] += start
    if "bridge" in result:
        result["bridge"] = {**result["bridge"], "direct_reference_links": [
            {**link, "reference_frame": start+link["reference_frame"]}
            for link in result["bridge"].get("direct_reference_links", [])]}
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


def _failure_evidence(entry):
    """Prefer a measured failed candidate over a later unattempted fit."""
    median = entry.get("after", {}).get("median_px")
    median = float(median) if median is not None and np.isfinite(median) else float("inf")
    return ("inlier_count" in entry,
            entry.get("inlier_fraction", -1.), entry.get("inlier_count", -1),
            -median,
            entry.get("observation_count", -1))


def refine_mechanical_windows(observations, initial_rotations, initial_valid, K, C, F,
                              *, radius, config, offline_config, top_shell_sign,
                              observation_provider=None):
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
    landmarks = {name: {} for name in names}
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
        "reference_policy": "Original first-image gauge, then fixed accepted overlap poses or trusted same-track surface landmarks in the original global frame.",
        "limitations": [
            "Pixel residuals and mechanical consistency are not absolute accuracy estimates.",
            "Fixed camera, shell curvature radius, assembly pivot, calibrated roll axis and starting roll are assumed.",
            "A later shell component needs its own accepted overlap spin reference or trusted same-track landmarks.",
            "A missing shell spin is not inferred from the other shell's measurements.",
            "A gap with no observed connection to accepted poses or trusted landmarks remains unresolved.",
            "Accepted overlap angles are fixed; their measurement error is not removed by later windows.",
        ],
    }
    minimum_length = opt.min_track_length

    def frontiers():
        return {name: int(indices[-1]) if len(indices := np.flatnonzero(measured[name])) else -1
                for name in names}

    def fit_family(start, end, targets, kind):
        """Try a bounded interval without shrinking away its unresolved target."""
        initial_end = end
        rematched = {name: 0 for name in names}
        if observation_provider is not None and any(landmarks.values()):
            proposed = observation_provider.propose(
                start, end, landmarks, alpha, beta, measured, by_frame, reports)
            for name in names:
                existing = {(entry.frame_index, entry.track_id)
                            for index in range(start, end) for entry in by_frame[name][index]}
                for entry in proposed[name]:
                    if not start <= entry.frame_index < end or entry.track_id not in landmarks[name]:
                        raise ValueError("image rematching must retain known map IDs inside its requested window")
                    key = (entry.frame_index, entry.track_id)
                    if key in existing or measured[name][entry.frame_index]:
                        continue
                    by_frame[name][entry.frame_index].append(entry)
                    existing.add(key)
                    rematched[name] += 1
        while True:
            frontier_before = frontiers()
            map_counts_before = {name: len(landmarks[name]) for name in names}
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
                fixed_landmarks=landmarks,
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
                    candidate = _global_frame_report(entry, start, window_index)
                    if (fitted.valid[name][local_index]
                            or reports[name][index]["reason"] == "not_reached_by_measured_window"
                            or _failure_evidence(candidate) > _failure_evidence(reports[name][index])):
                        reports[name][index] = candidate
                    if fitted.valid[name][local_index]:
                        if not any(measured[shell][index] for shell in names):
                            alpha[index] = fitted.alpha[local_index]
                        beta[name][index] = fitted.beta[name][local_index]
                        measured[name][index] = True
                        new_counts[name] += 1
                report["selected_observation_count"][name] += fitted.diagnostics["selected_observation_count"][name]
                # Only the batch's image-validated landmarks can become future
                # references. Once published, their global coordinates are fixed.
                for identifier, point in fitted.landmarks.get(name, {}).items():
                    landmarks[name].setdefault(int(identifier), np.array(point, dtype=float, copy=True))
            # Per-point residual vectors belong to the chosen global frame
            # diagnostics. Do not multiply them by every overlap/retry window.
            window_fit = {**fitted.diagnostics, "frames": {
                name: [{key: value for key, value in entry.items() if key != "reprojection"}
                       for entry in fitted.diagnostics["frames"][name]] for name in names}}
            report["windows"].append({
                "start_frame": start, "end_frame_exclusive": end,
                "kind": kind, "retry": end != initial_end,
                "target_frames": targets.copy(),
                "status": fitted.diagnostics["status"], "new_valid_frames": new_counts,
                "reference_frames": {name: [start+i for i in refs] for name, refs in references.items()},
                "frontiers_before": frontier_before, "frontiers_after": frontiers(),
                "trusted_landmark_counts_before": map_counts_before,
                "trusted_landmark_counts_after": {name: len(landmarks[name]) for name in names},
                "rematched_observations": rematched if end == initial_end else {name: 0 for name in names},
                "fit": window_fit,
            })
            for component in fitted.diagnostics["components"]:
                mapped = {**component, "window_index": window_index,
                          "frame_indices": [start+i for i in component["frame_indices"]]}
                if component.get("anchor_frame") is not None:
                    mapped["anchor_frame"] = start+component["anchor_frame"]
                for key in ("accepted_frames", "map_reference_frames"):
                    if key in mapped:
                        mapped[key] = [start+i for i in mapped[key]]
                if "inlier_graph_components" in mapped:
                    mapped["inlier_graph_components"] = [[start+i for i in group]
                                                         for group in mapped["inlier_graph_components"]]
                report["components"].append(mapped)
            report["roll_anchor_frames"].extend(start+i for i in references["alpha"])
            logger.info("Mechanical window %d:%d (%s%s): %s, solver=%s, evaluations=%s; "
                        "new top=%d bottom=%d; trusted landmarks top=%d bottom=%d",
                        start, end, kind, ", retry" if end != initial_end else "",
                        fitted.diagnostics["status"], fitted.diagnostics.get("solver_status", "not run"),
                        fitted.diagnostics.get("solver_nfev", 0), new_counts["top"], new_counts["bottom"],
                        len(landmarks["top"]), len(landmarks["bottom"]))
            pending = {name: target for name, target in targets.items()
                       if not np.any(measured[name][target:end])}
            if not pending:
                break
            # Halve the right-hand context, retaining both the original overlap
            # and the forward target. A retry wholly behind the target cannot
            # recover it and must never consume a fit attempt.
            smallest_end = max(start+minimum_length, max(pending.values())+1)
            smaller_end = max(smallest_end, start+(end-start)//2)
            if smaller_end >= end:
                break
            targets = pending
            end = smaller_end

    # Scan the whole recording, even after a gap. A later image can recover the
    # original reference by observing previously measured landmark IDs. Raw
    # initial poses never create a new reference in these later windows.
    start = 0
    scan_complete = False
    catchup_attempts = set()
    while n-start >= minimum_length:
        if start:
            # A leading shell must not evict the independent spin reference of
            # a lagging shell. Give each new lagging frontier a bounded catch-up
            # family before the regular scan advances beyond it.
            catchup_families = {}
            for name in names:
                frontier = frontiers()[name]
                if not 0 <= frontier < start or (name, frontier) in catchup_attempts:
                    continue
                catchup_attempts.add((name, frontier))
                catchup_start = max(0, frontier-config.overlap_frames+1)
                catchup_end = min(n, catchup_start+config.window_frames)
                if catchup_end-catchup_start >= minimum_length and frontier+1 < catchup_end:
                    catchup_families.setdefault((catchup_start, catchup_end), {})[name] = frontier+1
            for (catchup_start, catchup_end), targets in catchup_families.items():
                fit_family(catchup_start, catchup_end, targets, "shell_catch_up")
        end = min(n, start+config.window_frames)
        current = frontiers()
        targets = {name: max(start, current[name]+1) for name in names
                   if max(start, current[name]+1) < end}
        if targets:
            fit_family(start, end, targets, "scan")
        if end == n:
            scan_complete = True
            break
        start = min(start+config.window_frames-config.overlap_frames, n-minimum_length)

    for name in names:
        report["summary"][name] = {
            "valid_frames": int(measured[name].sum()),
            "refined_frames": sum(entry["status"] == "refined" for entry in reports[name]),
            "recovered_frames": sum(entry["status"] == "recovered" for entry in reports[name]),
            "unresolved_frames": np.flatnonzero(~measured[name]).tolist(),
        }
    report["roll_anchor_frames"] = sorted(set(report["roll_anchor_frames"]))
    report["gaps"] = {name: _gaps(measured[name], reports[name]) for name in names}
    report["scan_complete"] = scan_complete
    report["trusted_landmark_counts"] = {name: len(landmarks[name]) for name in names}
    if observation_provider is not None:
        report["image_rematching"] = observation_provider.diagnostics
    report["status"] = ("completed" if all(np.all(measured[name]) for name in names)
                        else "completed_with_unresolved_intervals")
    if n < minimum_length:
        report["status"] = "insufficient_observations"
    rotations = {name: _model_rotations(alpha, beta[name], frame) for name in names}
    return MechanicalResult(
        rotations, measured, np.where(measured["top"] | measured["bottom"], alpha, np.nan),
        {name: np.where(measured[name], beta[name], np.nan) for name in names}, report,
        landmarks=landmarks)
