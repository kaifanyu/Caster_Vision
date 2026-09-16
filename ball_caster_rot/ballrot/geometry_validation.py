"""Conditional multi-view checks of separated-hemisphere geometry.

Each correspondence transfers an observed source pixel to a later image using
the preserved camera rotation estimates. Geometry is fitted on training views
and evaluated on other target frames AND other persistent landmark IDs. The
poses themselves were estimated from this footage: this is a conditional model
check, not independent angular ground truth or an absolute calibration test.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .camera import pixel_to_ray, validate_camera_matrix
from .rotation import is_rotation_matrix
from .shell_geometry import shell_centers, surface_camera, unproject_shell

SHELLS = ("top", "bottom")
PARAMETER_SCALE = np.array([.08, .08, .25, *np.deg2rad([2., 2., 2.]), .04])


@dataclass
class TransferDataset:
    samples: list[dict]
    rotations: dict[str, np.ndarray]
    valid: dict[str, np.ndarray]
    selection: dict
    raw_valid: dict[str, np.ndarray] | None = None


def _motion_arrays(rotations, valid):
    matrices, flags = {}, {}
    count = None
    for shell in SHELLS:
        r = np.asarray(rotations[shell], dtype=float)
        v = np.asarray(valid[shell])
        if (r.ndim != 3 or r.shape[1:] != (3, 3) or not len(r)
                or v.shape != (len(r),) or v.dtype.kind != "b"
                or not np.all(np.isfinite(r))
                or not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=1e-6)
                or not np.allclose(np.linalg.det(r), 1., atol=1e-6)):
            raise ValueError("camera rotations and boolean validity must have matching proper (N,3,3)/(N,) shapes")
        if count is not None and len(r) != count:
            raise ValueError("shell trajectories must have the same frame count")
        count = len(r)
        matrices[shell], flags[shell] = r.copy(), v.copy()
    return matrices, flags


def mechanical_orientations(rotations, valid, frame):
    """Re-express fixed camera poses for this candidate frame on every call.

    Removing gamma and averaging the two roll directions enforces the same
    shared-roll/independent-spin model used by the mechanical estimator. The
    average is circular because only orientation, not gap turn count, is used.
    This never changes the camera-pose evidence or grants missing poses validity.
    """
    components = {}
    for shell in SHELLS:
        local = frame.T @ rotations[shell] @ frame
        components[shell] = Rotation.from_matrix(local).as_euler("XYZ")
    at, ab = (components[shell][:, 0] for shell in SHELLS)
    vt, vb = (valid[shell] for shell in SHELLS)
    alpha = np.where(vt, at, ab)
    both = vt & vb
    alpha[both] = np.arctan2(np.sin(at[both]) + np.sin(ab[both]),
                            np.cos(at[both]) + np.cos(ab[both]))
    orientations = {}
    for shell in SHELLS:
        angles = np.column_stack((alpha, np.zeros(len(alpha)), components[shell][:, 2]))
        orientations[shell] = frame @ Rotation.from_euler("XYZ", angles).as_matrix()
    return orientations, components


def quality_pose_flags(rotations, valid, frame):
    """Use one fixed quality rule for fresh samples and frozen evidence replay."""
    _, components = mechanical_orientations(rotations, valid, frame)
    usable = {shell: valid[shell] & (np.abs(components[shell][:, 1]) <= np.deg2rad(5.))
              for shell in SHELLS}
    difference = components["top"][:, 0] - components["bottom"][:, 0]
    disagreement = np.abs(np.arctan2(np.sin(difference), np.cos(difference)))
    conflict = usable["top"] & usable["bottom"] & (disagreement > np.deg2rad(10.))
    for shell in SHELLS:
        usable[shell] &= ~conflict
    return usable


def _heldout_id(identifier):
    # Stable across Python processes; neither coordinates nor fit errors choose IDs.
    return (((int(identifier) * 2654435761) ^ (int(identifier) >> 16)) & 3) == 0


def _spread_select(rows, maximum, minimum_distance_px=8.):
    """Keep spatially separated targets; duplicate track families add no support."""
    if not rows:
        return []
    points = np.array([row["observed_uv"] for row in rows])
    chosen, remaining = [], np.ones(len(rows), bool)
    score = np.sum((points - np.mean(points, axis=0))**2, axis=1)
    while remaining.any() and len(chosen) < maximum:
        index = int(np.argmax(np.where(remaining, score, -np.inf)))
        chosen.append(rows[index])
        distances = np.sum((points - points[index])**2, axis=1)
        remaining &= distances >= minimum_distance_px**2
        remaining[index] = False
        score = distances if len(chosen) == 1 else np.minimum(score, distances)
    return chosen


def build_transfer_dataset(observations, rotations, valid, K, C, F, gap_fraction,
                           *, top_shell_sign=1, view_count=18, max_per_view=40,
                           min_rotation_deg=3., max_rotation_deg=35., max_source_age=45):
    """Deterministically select observed, source-valid landmark transfers.

    Targets use approximately ``view_count`` frames across both shells. Every
    third selected target per shell is held out. Held-out IDs are never fitted,
    and a held-out target frame is never used as a training source. Selection
    uses source geometry, motion span and spatial spread, never target residual.
    """
    camera = validate_camera_matrix(K)
    if not is_rotation_matrix(F) or np.shape(C) != (3,) or not np.all(np.isfinite(C)):
        raise ValueError("initial F and normalized pivot must be finite proper geometry")
    if (isinstance(view_count, bool) or not isinstance(view_count, int) or view_count < 12
            or isinstance(max_per_view, bool) or not isinstance(max_per_view, int) or max_per_view < 8
            or not 0 < min_rotation_deg < max_rotation_deg < 180
            or not isinstance(max_source_age, int) or max_source_age < 2):
        raise ValueError("need >=12 views, >=8 samples/view and valid motion/source-age limits")
    if isinstance(top_shell_sign, bool) or top_shell_sign not in (-1, 1) or not 0 <= gap_fraction < 1:
        raise ValueError("invalid shell sign or gap")
    matrices, flags = _motion_arrays(rotations, valid)
    usable = quality_pose_flags(matrices, flags, F)
    orientations, _ = mechanical_orientations(matrices, usable, F)
    targets = {}
    for shell in SHELLS:
        frames = np.flatnonzero(usable[shell])
        frames = frames[frames > 0]
        indices = np.linspace(0, max(0, len(frames)-1), min(view_count//2, len(frames))).round().astype(int)
        targets[shell] = frames[indices].tolist()
    heldout_frames = set(index for shell in SHELLS for index in targets[shell][1::3])
    samples, used_ids = [], set()
    duplicate_rows = 0
    for shell in SHELLS:
        by_id, at_frame = defaultdict(dict), defaultdict(set)
        for entry in observations[shell]:
            index, identifier = int(entry.frame_index), int(entry.track_id)
            uv = np.asarray(entry.uv, dtype=float)
            if index < 0 or index >= len(usable[shell]) or uv.shape != (2,) or not np.all(np.isfinite(uv)):
                raise ValueError("observation has invalid frame index or pixel coordinates")
            if not usable[shell][index]:
                continue
            previous = by_id[identifier].get(index)
            if previous is not None:
                if not np.allclose(previous, uv, atol=1e-6, rtol=0):
                    raise ValueError("one persistent track has conflicting observations in the same frame")
                duplicate_rows += 1
                continue
            by_id[identifier][index] = uv
            at_frame[index].add(identifier)
        sign = top_shell_sign if shell == "top" else -top_shell_sign
        for target in targets[shell]:
            split = "heldout" if target in heldout_frames else "training"
            rows = []
            for identifier in sorted(at_frame[target]):
                if (shell, identifier) in used_ids or _heldout_id(identifier) != (split == "heldout"):
                    continue
                choices = []
                for source, uv in by_id[identifier].items():
                    if source >= target or target-source > max_source_age or source in heldout_frames:
                        continue
                    relative = orientations[shell][target] @ orientations[shell][source].T
                    angle = float(np.rad2deg(Rotation.from_matrix(relative).magnitude()))
                    if min_rotation_deg <= angle <= max_rotation_deg:
                        choices.append((angle, source, uv))
                if not choices:
                    continue
                angle, source, uv = max(choices, key=lambda item: (item[0], -item[1]))
                normal, hit, view = unproject_shell(uv[None], camera, C, 1., orientations[shell][source],
                                                   sign, gap_fraction, "separated_hemispheres")
                local = orientations[shell][source].T @ normal[0]
                if not hit[0] or view[0] > np.deg2rad(70.) or sign*local[2] < .02:
                    continue
                rows.append({"shell": shell, "track_id": identifier, "source_frame": source,
                             "target_frame": target, "source_uv": uv.tolist(),
                             "observed_uv": by_id[identifier][target].tolist(), "split": split,
                             "baseline_motion_deg": angle})
            chosen = _spread_select(rows, max_per_view)
            samples.extend(chosen)
            used_ids.update((shell, row["track_id"]) for row in chosen)
    selection = {
        "requested_view_count": view_count, "max_samples_per_view": max_per_view,
        "selected_targets": targets, "reserved_heldout_frames": sorted(heldout_frames),
        "min_transfer_rotation_deg": min_rotation_deg, "max_transfer_rotation_deg": max_rotation_deg,
        "max_source_age_frames": max_source_age, "duplicate_observation_rows_removed": duplicate_rows,
        "initial_tilt_limit_deg": 5., "initial_shared_roll_disagreement_limit_deg": 10.,
        "split_note": "Held-out target frames and landmark IDs are excluded from geometry fitting. Source poses were estimated from the same footage.",
    }
    return TransferDataset(samples, matrices, usable, selection, raw_valid=flags)


def _geometry(parameters, initial_pivot, initial_frame, initial_gap):
    physical = np.asarray(parameters) * PARAMETER_SCALE
    return (initial_pivot + physical[:3],
            Rotation.from_rotvec(physical[3:6]).as_matrix() @ initial_frame,
            initial_gap + physical[6])


def _other_cap_occluded(rays, depth, orientations, pivot, signs, gap):
    """Both shell spins share z, so the other cap needs no inferred spin angle."""
    other_centers = shell_centers(orientations, pivot, 1., -signs, gap, "separated_hemispheres")
    along = np.einsum("ij,ij->i", rays, other_centers)
    discriminant = along**2 - np.sum(other_centers**2, axis=1) + 1.
    root = np.sqrt(np.maximum(discriminant, 0.))
    occluded = np.zeros(len(rays), bool)
    for intersection in (along-root, along+root):
        normal = intersection[:, None]*rays - other_centers
        on_cap = -signs*np.einsum("ij,ij->i", normal, orientations[:, :, 2]) >= 0
        occluded |= ((discriminant > 0) & (intersection > 0)
                     & (intersection < depth-1e-7) & on_cap)
    return occluded


def transfer_predictions(dataset, K, pivot, frame, gap_fraction, *, top_shell_sign=1):
    """Transfer source rays; invalid intersections remain explicit failures."""
    count = len(dataset.samples)
    if not count:
        return np.empty((0, 2)), np.empty(0, bool), np.empty((0, 3))
    orientations, _ = mechanical_orientations(dataset.rotations, dataset.valid, frame)
    source = np.array([orientations[row["shell"]][row["source_frame"]] for row in dataset.samples])
    target = np.array([orientations[row["shell"]][row["target_frame"]] for row in dataset.samples])
    signs = np.array([top_shell_sign if row["shell"] == "top" else -top_shell_sign for row in dataset.samples])
    rays = pixel_to_ray([row["source_uv"] for row in dataset.samples], K)
    centers = shell_centers(source, pivot, 1., signs, gap_fraction, "separated_hemispheres")
    along = np.einsum("ij,ij->i", rays, centers)
    discriminant = along**2 - np.sum(centers**2, axis=1) + 1.
    distance = along - np.sqrt(np.maximum(discriminant, 1e-10))
    normals = distance[:, None]*rays - centers
    local = np.einsum("nji,nj->ni", source, normals)
    xyz = surface_camera(local, target, pivot, 1., signs, gap_fraction, "separated_hemispheres")
    rotated_normal = np.einsum("nij,nj->ni", target, local)
    cosine = -np.einsum("ij,ij->i", rotated_normal, xyz) / np.maximum(np.linalg.norm(xyz, axis=1), 1e-10)
    half = signs*local[:, 2]
    homogeneous = xyz @ K.T
    prediction = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:3], 1e-6)
    valid = (discriminant > 0) & (distance > 0) & (xyz[:, 2] > 0) & (cosine > 0) & (half >= 0)
    target_depth = np.linalg.norm(xyz, axis=1)
    target_rays = xyz / np.maximum(target_depth[:, None], 1e-10)
    valid &= ~_other_cap_occluded(rays, distance, source, pivot, signs, gap_fraction)
    valid &= ~_other_cap_occluded(target_rays, target_depth, target, pivot, signs, gap_fraction)
    # These residuals prevent the optimizer from hiding behind the clipped
    # square root. They are not counted as pixel measurements or observability.
    penalties = np.column_stack((100*np.maximum(-discriminant, 0),
                                 100*np.maximum(-cosine, 0), 100*np.maximum(-half, 0)))
    return prediction, valid, penalties


def _metrics(errors, physical, rows, mask, source_quality):
    selected = np.flatnonzero(mask)
    if not len(selected):
        return {"count": 0, "median_px": None, "p90_px": None, "inlier_fraction_3px": 0.,
                "physical_fraction": 0., "source_quality_fraction": 0., "source_quality_unavailable_count": 0,
                "target_frames": [], "landmark_count": 0}
    # An invalid ray/cap/visibility is a failure, not a sample removed from scoring.
    supported = source_quality[selected]
    score = np.where(physical[selected] & supported, errors[selected], np.maximum(errors[selected], 1000.))
    return {"count": len(selected), "median_px": float(np.median(score)),
            "p90_px": float(np.percentile(score, 90)),
            "inlier_fraction_3px": float(np.mean((errors[selected] <= 3.) & physical[selected] & supported)),
            "physical_fraction": float(np.mean(physical[selected][supported])) if supported.any() else 0.,
            "source_quality_fraction": float(np.mean(supported)),
            "source_quality_unavailable_count": int(np.count_nonzero(~supported)),
            "target_frames": sorted({rows[i]["target_frame"] for i in selected}),
            "landmark_count": len({(rows[i]["shell"], rows[i]["track_id"]) for i in selected})}


def validate_geometry(dataset, K, initial_pivot, initial_frame, initial_gap,
                      *, top_shell_sign=1, max_nfev=100, fit_candidate=True):
    """Fit a bounded candidate and gate conditional held-out improvements.

    Numerical thresholds are explicit software criteria, not calibrated
    confidence levels. No configuration is changed by this function.
    """
    camera = validate_camera_matrix(K)
    center, frame = np.asarray(initial_pivot, float), np.asarray(initial_frame, float)
    if (center.shape != (3,) or not np.all(np.isfinite(center)) or not is_rotation_matrix(frame)
            or not 0 <= initial_gap < .3 or center[2] <= 1 + initial_gap
            or top_shell_sign not in (-1, 1) or isinstance(top_shell_sign, bool)):
        raise ValueError("initial separated geometry must have a valid frame, pivot in front of the camera, and gap in [0,.3)")
    rows = dataset.samples
    n = len(rows)
    training = np.array([row["split"] == "training" for row in rows], bool)
    heldout = ~training
    observed = np.asarray([row["observed_uv"] for row in rows], float).reshape(-1, 2)
    masks = {split: mask for split, mask in (("training", training), ("heldout", heldout))}
    for split, mask in list(masks.items()):
        for shell in SHELLS:
            masks[split+"_"+shell] = mask & np.array([row["shell"] == shell for row in rows], bool)
    train_ids = {(row["shell"], row["track_id"]) for row in rows if row["split"] == "training"}
    hold_ids = {(row["shell"], row["track_id"]) for row in rows if row["split"] == "heldout"}
    hold_frames = {row["target_frame"] for row in rows if row["split"] == "heldout"}
    if train_ids & hold_ids or any(row["source_frame"] in hold_frames or row["target_frame"] in hold_frames
                                  for row in rows if row["split"] == "training"):
        raise ValueError("held-out landmark/frame leakage into training correspondences")
    if any(row["split"] not in ("training", "heldout") or row["shell"] not in SHELLS for row in rows):
        raise ValueError("samples must specify shell and training/heldout split")
    raw_valid = dataset.valid if dataset.raw_valid is None else dataset.raw_valid
    for row in rows:
        if (row["source_frame"] >= row["target_frame"]
                or not raw_valid[row["shell"]][row["source_frame"]]
                or not raw_valid[row["shell"]][row["target_frame"]]):
            raise ValueError("each transfer needs two ordered source-valid observations")
    source_quality = np.array([dataset.valid[row["shell"]][row["source_frame"]]
                               and dataset.valid[row["shell"]][row["target_frame"]] for row in rows], bool)
    fitted_training = training & source_quality
    lower_physical = np.array([-.2, -.2, -.6, *np.deg2rad([-5., -5., -5.]), -min(.08, initial_gap)])
    upper_physical = np.array([.2, .2, .6, *np.deg2rad([5., 5., 5.]), min(.08, .3-initial_gap)])
    lower, upper = lower_physical/PARAMETER_SCALE, upper_physical/PARAMETER_SCALE
    # At gap zero the optimizer can still move into the positive interval.
    x0 = np.minimum(np.maximum(np.zeros(7), lower+1e-9), upper-1e-9)

    def evaluate(parameters):
        c, f, gap = _geometry(parameters, center, frame, initial_gap)
        return transfer_predictions(dataset, camera, c, f, gap, top_shell_sign=top_shell_sign)

    def objective(parameters):
        prediction, _, penalties = evaluate(parameters)
        return np.r_[(prediction-observed)[fitted_training].ravel(), penalties[fitted_training].ravel(), parameters]

    baseline_prediction, baseline_valid, _ = evaluate(np.zeros(7))
    attempted = fit_candidate and int(fitted_training.sum()) >= 14
    fit = least_squares(objective, x0, bounds=(lower, upper), loss="soft_l1", f_scale=2.,
                        max_nfev=max_nfev, xtol=1e-8, ftol=1e-8, gtol=1e-8) if attempted else None
    parameters = fit.x if fit is not None else np.zeros(7)
    candidate_center, candidate_frame, candidate_gap = _geometry(parameters, center, frame, initial_gap)
    prediction, physical, _ = evaluate(parameters)
    baseline_errors = np.linalg.norm(baseline_prediction-observed, axis=1)
    errors = np.linalg.norm(prediction-observed, axis=1)
    baseline = {name: _metrics(baseline_errors, baseline_valid, rows, mask, source_quality) for name, mask in masks.items()}
    candidate = {name: _metrics(errors, physical, rows, mask, source_quality) for name, mask in masks.items()}
    reasons = []
    if fit_candidate and (fit is None or not fit.success):
        reasons.append("solver_not_completed")
    for split, minimum, minimum_views in (("training", 40, 3), ("heldout", 20, 2)):
        for shell in SHELLS:
            stats = candidate[split+"_"+shell]
            if stats["landmark_count"] < minimum or len(stats["target_frames"]) < minimum_views:
                reasons.append(split+"_"+shell+"_insufficient_independent_support")
            if stats["source_quality_fraction"] < 1.:
                reasons.append(split+"_"+shell+"_source_pose_quality_unavailable")
            if stats["physical_fraction"] < .95:
                reasons.append(split+"_"+shell+"_invalid_projection_support")
            if stats["inlier_fraction_3px"] < .7:
                reasons.append(split+"_"+shell+"_low_pixel_agreement")
    if (candidate_center[2] <= 1 + candidate_gap
            or np.linalg.norm(parameters[3:6]*PARAMETER_SCALE[3:6]) > np.deg2rad(5.)):
        reasons.append("physical_or_axis_adjustment_limit")
    normalized_distance = np.minimum(parameters-lower, upper-parameters)/(upper-lower)
    active_bounds = [int(i) for i in np.flatnonzero(normalized_distance < .01)]
    if fit_candidate and active_bounds:
        reasons.append("candidate_at_parameter_bound")

    inliers = fitted_training & physical & (errors <= 3.)
    singular_values = np.array([])
    if np.count_nonzero(inliers) >= 14:
        columns = []
        for axis in range(7):
            delta = np.zeros(7)
            delta[axis] = 1e-4
            plus, _, _ = evaluate(parameters+delta)
            minus, _, _ = evaluate(parameters-delta)
            columns.append(((plus-minus)[inliers]/2e-4).ravel())
        jacobian = np.column_stack(columns)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.count_nonzero(singular_values > singular_values[0]*1e-6)) if len(singular_values) else 0
    condition = float(singular_values[0]/singular_values[-1]) if len(singular_values) and singular_values[-1] > 0 else None
    if rank != 7 or condition is None or condition > 500 or singular_values[-1] < .5:
        reasons.append("geometry_not_observable_without_priors")
    for split in (("training", "heldout") if fit_candidate else ()):
        old, new = baseline[split], candidate[split]
        if (old["median_px"] is None or new["median_px"] is None
                or new["median_px"] > .9*old["median_px"]
                or new["p90_px"] > 1.05*old["p90_px"]):
            reasons.append(split+"_improvement_not_demonstrated")
    for shell in (SHELLS if fit_candidate else ()):
        old, new = baseline["heldout_"+shell], candidate["heldout_"+shell]
        if old["median_px"] is None or new["median_px"] is None or new["median_px"] > old["median_px"]:
            reasons.append("heldout_"+shell+"_degraded")

    scored_rows = []
    for index, row in enumerate(rows):
        row = dict(row)
        row.update(baseline_predicted_uv=baseline_prediction[index].tolist(),
                   candidate_predicted_uv=prediction[index].tolist(),
                   baseline_error_px=float(baseline_errors[index]), candidate_error_px=float(errors[index]),
                   baseline_physical=bool(baseline_valid[index]), candidate_physical=bool(physical[index]),
                   source_pose_quality_supported=bool(source_quality[index]))
        scored_rows.append(row)
    status = ("candidate_passed_conditional_checks" if not reasons else "candidate_rejected") if fit_candidate else (
        "validation_passed_conditional_checks" if not reasons else "validation_failed")
    return {
        "status": status, "evaluation_only": not fit_candidate,
        "candidate_config_permitted": bool(fit_candidate and not reasons), "rejection_reasons": reasons,
        "method": "observed_source_ray_to_heldout_target_pixel_transfer",
        "geometry": "separated_hemispheres", "top_shell_sign": top_shell_sign,
        "initial_geometry": {"pivot_camera": center.tolist(), "measurement_frame": frame.tolist(), "gap_fraction": initial_gap},
        "candidate_geometry": {"pivot_camera": candidate_center.tolist(), "measurement_frame": candidate_frame.tolist(),
                               "gap_fraction": float(candidate_gap),
                               "camera_axis_adjustment_deg": (parameters[3:6]*PARAMETER_SCALE[3:6]*180/np.pi).tolist()},
        "baseline": baseline, "candidate": candidate,
        "solver": {"attempted": attempted, "success": bool(fit.success) if fit is not None else False,
                   "message": str(fit.message) if fit is not None else ("evaluation only" if not fit_candidate else "insufficient training pairs"),
                   "nfev": int(fit.nfev) if fit is not None else 0, "active_parameter_bounds": active_bounds,
                   "fitted_training_transfer_count": int(fitted_training.sum())},
        "observability": {"parameter_order": ["pivot_x", "pivot_y", "pivot_z", "axis_rx", "axis_ry", "axis_rz", "gap"],
                          "parameter_scales": PARAMETER_SCALE.tolist(), "training_inlier_jacobian_rank": rank,
                          "singular_values": singular_values.tolist(), "condition_number": condition,
                          "note": "Data-only Jacobian on training inliers, excluding priors. Parameter scales define conditioning units."},
        "gates": {"max_residual_px": 3., "minimum_inlier_fraction": .7, "minimum_physical_fraction": .95,
                  "minimum_training_landmarks_per_shell": 40, "minimum_heldout_landmarks_per_shell": 20,
                  "minimum_training_views_per_shell": 3, "minimum_heldout_views_per_shell": 2,
                  "minimum_median_improvement_fraction": .1, "maximum_p90_regression_fraction": .05,
                  "maximum_data_jacobian_condition": 500, "minimum_scaled_singular_value": .5},
        "fit_priors_and_bounds": {"parameter_scales": PARAMETER_SCALE.tolist(),
                                  "lower_parameter_offsets": lower_physical.tolist(),
                                  "upper_parameter_offsets": upper_physical.tolist(),
                                  "note": "Offsets use shell-radius units for pivot/gap and radians for axis rotation. Priors penalize scaled parameter offsets but are excluded from observability."},
        "selection": dataset.selection, "samples": scored_rows,
        "limitations": [
            "Conditional consistency test: source camera poses were estimated using the same footage, including held-out pixels. They are not angular ground truth.",
            "For every candidate frame, preserved camera rotations are decomposed anew; shared roll and zero sideways tilt are imposed. Camera poses themselves are not re-estimated here.",
            "Held-out target frames and persistent landmark IDs do not enter the geometry fit. Source-ray pixels and noisy source motion still affect target predictions.",
            "Frozen evidence keeps every pair. A source pose failing the same 5-degree tilt/10-degree shared-roll screen is an unavailable transfer, scored as a failure (at least 1000 px) instead of being silently removed. Geometric physical fraction is conditional on source-quality support.",
            "Numerical gates and parameter priors are chosen software criteria, not statistical confidence or physical accuracy estimates.",
            "Tracked identities may still be wrong or correlated across tracking families. This test creates no new image observations and cannot recover absent orientations.",
            "Equal shell curvature radii, fixed camera intrinsics, stationary pivot and rigid complete hemispheres remain assumptions. Mutual shell-cap occlusion is checked; yoke and inner-disc occlusions are not modeled. No metric radius is inferred.",
            "Even a passing candidate requires a fresh pixel refit and another held-out check before use as a calibration; the main configuration is never changed.",
        ],
    }
