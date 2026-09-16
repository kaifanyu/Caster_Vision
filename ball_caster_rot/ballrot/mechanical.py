"""Joint image fitting for a fixed-center, shared-roll two-shell caster.

The estimator fits actual undistorted pixels, not independently fitted Euler
angles. Both caps have one roll angle and their own spin angle. Landmark polar
angles are bounded to their respective cap, maintaining a constant physical
gap. No constraint on slip, equal spin, or temporal smoothness is imposed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from .camera import validate_camera_matrix
from .offline import (OfflineConfig, SurfaceObservation, _components,
                      _exclude_unsupported_invalid_frames, _select_observations,
                      _spread, _summary)
from .rotation import is_rotation_matrix
from .sphere import unproject_to_sphere, viewing_angle


@dataclass(frozen=True)
class MechanicalConfig:
    enabled: bool = False
    # Separation of the complementary cap boundary planes / sphere diameter.
    # Zero is an ideal shared equator, not a measured nonzero hardware gap.
    gap_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("mechanical.enabled must be boolean")
        if (isinstance(self.gap_fraction, bool)
                or not isinstance(self.gap_fraction, (int, float, np.number))
                or not np.isfinite(self.gap_fraction)
                or not 0 <= self.gap_fraction < 1):
            raise ValueError("mechanical.gap_fraction must be finite in [0, 1)")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> MechanicalConfig:
        if values is not None and not isinstance(values, Mapping):
            raise ValueError("mechanical settings must be a mapping")
        mapping = {} if values is None else dict(values)
        unknown = set(mapping) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown mechanical settings: {', '.join(sorted(unknown))}")
        return cls(**mapping)


@dataclass
class MechanicalResult:
    rotations: dict[str, np.ndarray]
    valid: dict[str, np.ndarray]
    alpha: np.ndarray
    beta: dict[str, np.ndarray]
    diagnostics: dict[str, Any]


def _model_rotations(alpha: np.ndarray, beta: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Camera-relative F Rx(alpha) Rz(beta) F.T, vectorized."""
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    model = np.empty((len(alpha), 3, 3))
    model[:, 0, :] = np.column_stack((cb, -sb, np.zeros(len(alpha))))
    model[:, 1, :] = np.column_stack((ca * sb, ca * cb, -sa))
    model[:, 2, :] = np.column_stack((sa * sb, sa * cb, ca))
    return F @ model @ F.T


def _initial_angles(rotations, valid, F):
    angles = {name: Rotation.from_matrix(F.T @ poses @ F).as_euler("XYZ")
              for name, poses in rotations.items()}
    alpha = np.zeros(len(rotations["top"]))
    beta = {name: np.unwrap(value[:, 2]) for name, value in angles.items()}
    for frame in range(len(alpha)):
        values = [angles[name][frame, 0] for name in ("top", "bottom") if valid[name][frame]]
        if not values:
            values = [angles[name][frame, 0] for name in ("top", "bottom")]
        alpha[frame] = np.arctan2(np.sin(values).sum(), np.cos(values).sum())
    alpha = np.unwrap(alpha)
    if len(alpha):
        # Absolute input poses are relative to the first camera image. The
        # effective calibrated frame F already includes measured initial roll.
        alpha[0] = beta["top"][0] = beta["bottom"][0] = 0.0
    return alpha, beta


def refine_mechanical_trajectory(
    observations: dict[str, list[SurfaceObservation]],
    initial_rotations: dict[str, np.ndarray],
    initial_valid: dict[str, np.ndarray],
    K: np.ndarray,
    C: np.ndarray,
    F: np.ndarray,
    radius: float = 1.0,
    config: MechanicalConfig | Mapping[str, Any] | None = None,
    offline_config: OfflineConfig | Mapping[str, Any] | None = None,
    top_shell_sign: int = 1,
) -> MechanicalResult:
    """Fit shared roll and independently observed shell spins to image tracks.

    The output always lies on the mechanical manifold when enabled. Failed or
    unobserved poses contain constrained numeric placeholders with false
    validity; they are not silently replaced by independent forward estimates.
    Each strong shell-track component inherits its earliest trusted projected
    input pose. Recovery requires real observations supported by trusted images.
    A visible shell can measure shared roll while the other spin stays unknown.
    """
    cfg = config if isinstance(config, MechanicalConfig) else MechanicalConfig.from_mapping(config)
    opt = offline_config if isinstance(offline_config, OfflineConfig) else OfflineConfig.from_mapping(offline_config)
    names = ("top", "bottom")
    if any(not isinstance(value, Mapping) or set(value) != set(names)
           for value in (observations, initial_rotations, initial_valid)):
        raise ValueError("observations, initial_rotations and initial_valid require top and bottom")
    poses = {name: np.asarray(initial_rotations[name], dtype=float).copy() for name in names}
    validity = {name: np.asarray(initial_valid[name], dtype=bool).copy() for name in names}
    n = len(poses["top"])
    for name in names:
        candidate = poses[name]
        if candidate.shape != (n, 3, 3) or validity[name].shape != (n,):
            raise ValueError("initial_rotations must be (N,3,3), initial_valid must be (N,)")
        if (not np.all(np.isfinite(candidate))
                or not np.allclose(candidate @ candidate.transpose(0, 2, 1), np.eye(3), atol=1e-6)
                or np.any(np.linalg.det(candidate) < 0)):
            raise ValueError("initial_rotations must contain proper finite rotations")
    frame = np.asarray(F, dtype=float)
    if not is_rotation_matrix(frame, atol=1e-6):
        raise ValueError("F must be a proper ball-to-camera rotation")
    camera = validate_camera_matrix(K)
    center = np.asarray(C, dtype=float)
    if (center.shape != (3,) or not np.all(np.isfinite(center))
            or not np.isfinite(radius) or radius <= 0 or center[2] <= radius):
        raise ValueError("sphere must have positive radius and lie in front of the camera")
    if isinstance(top_shell_sign, bool) or top_shell_sign not in (-1, 1):
        raise ValueError("top_shell_sign must be +1 or -1")
    data = {name: list(observations[name]) for name in names}
    for entries in data.values():
        if any(not isinstance(entry, SurfaceObservation) for entry in entries):
            raise ValueError("observations must contain SurfaceObservation instances")
        if any(entry.frame_index >= n for entry in entries):
            raise ValueError("observation frame_index exceeds trajectory length")
    alpha, beta = _initial_angles(poses, validity, frame) if n else (np.empty(0), {name: np.empty(0) for name in names})
    projected = {name: _model_rotations(alpha, beta[name], frame) for name in names}
    output_valid = {name: np.zeros(n, dtype=bool) for name in names}
    frame_reports = {name: [{"frame_index": index, "status": "unresolved",
                            "reason": "insufficient_observations"} for index in range(n)] for name in names}
    report = {
        "enabled": cfg.enabled, "model": "shared_roll_independent_spin",
        "method": "joint_mechanical_reprojection_fit", "config": asdict(cfg),
        "offline_config": asdict(opt), "top_shell_sign": int(top_shell_sign),
        "gap_distance_sphere_units": float(2 * radius * cfg.gap_fraction),
        "frames": frame_reports, "components": [], "summary": {},
        "input_observation_count": {name: len(data[name]) for name in names},
        "selected_observation_count": {name: 0 for name in names},
        "limitations": [
            "Pixel residuals and mechanical consistency are not absolute accuracy estimates.",
            "Fixed camera, sphere, calibrated roll axis and starting roll are assumed.",
            "A disconnected component inherits its earliest trusted projected input pose.",
            "Unobserved shell spin remains invalid even when the other shell measures shared roll.",
            "No equality of shell spins, no-slip condition or temporal smoothness is imposed.",
        ],
    }

    def finish(status):
        report["status"] = status
        for name in names:
            accepted = output_valid[name]
            report["summary"][name] = {
                "valid_frames": int(np.count_nonzero(accepted)),
                "refined_frames": sum(entry["status"] == "refined" for entry in frame_reports[name]),
                "recovered_frames": sum(entry["status"] == "recovered" for entry in frame_reports[name]),
                "unresolved_frames": np.flatnonzero(~accepted).tolist(),
            }
        measured_alpha = np.where(output_valid["top"] | output_valid["bottom"], alpha, np.nan)
        measured_beta = {name: np.where(output_valid[name], beta[name], np.nan) for name in names}
        return MechanicalResult(projected, output_valid, measured_alpha, measured_beta, report)

    if not cfg.enabled:
        for name in names:
            output_valid[name] = validity[name]
            projected[name] = poses[name]
            for index, entry in enumerate(frame_reports[name]):
                entry.update(status="unrefined" if validity[name][index] else "unresolved",
                             reason="disabled")
        return finish("disabled")
    if n < opt.min_track_length:
        return finish("insufficient_observations")
    radius_px = float(np.mean((camera[0, 0], camera[1, 1])) * radius /
                      np.sqrt(np.dot(center, center) - radius**2))
    components = []
    for name in names:
        selected = data[name]
        if not selected:
            continue
        pixels = np.asarray([entry.uv for entry in selected])
        directions, keep = unproject_to_sphere(pixels, camera, center, radius)
        keep &= viewing_angle(directions, center, radius) <= np.deg2rad(opt.limb_cull_deg)
        selected = [entry for entry, good in zip(selected, keep) if good]
        while selected:
            count = len(selected)
            selected, excluded = _exclude_unsupported_invalid_frames(selected, validity[name], radius_px, opt)
            for index, details in excluded.items():
                frame_reports[name][index].update(details)
            selected = _select_observations(selected, opt)
            if len(selected) == count:
                break
        for group in _components(selected, radius_px, opt):
            group = _select_observations(group, opt)
            if not group:
                continue
            frame_ids = sorted({entry.frame_index for entry in group})
            trusted = [index for index in frame_ids if validity[name][index]]
            if not trusted:
                for index in frame_ids:
                    frame_reports[name][index]["reason"] = "no_trusted_pose_anchor"
                continue
            details = {"shell": name, "anchor_frame": trusted[0], "frame_indices": frame_ids,
                       "observation_count": len(group), "inherits_input_reference": True}
            report["components"].append(details)
            components.append((name, group, details))
            report["selected_observation_count"][name] += len(group)
    if not components:
        return finish("insufficient_observations")

    # Each shell component needs its own spin reference. Roll is shared, so a
    # newly observed shell must not pin a roll that the other shell already
    # measures continuously. Anchor roll once per joint connected frame group.
    parent = {}

    def root(index):
        parent.setdefault(index, index)
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for _, _, details in components:
        first = details["frame_indices"][0]
        for index in details["frame_indices"]:
            parent[root(index)] = root(first)
    roll_anchors = {}
    for _, _, details in components:
        group = root(details["anchor_frame"])
        roll_anchors[group] = min(roll_anchors.get(group, details["anchor_frame"]), details["anchor_frame"])
    alpha_anchors = set(roll_anchors.values()) | {0}
    report["roll_anchor_frames"] = sorted(set(roll_anchors.values()))
    alpha_frames = sorted({entry.frame_index for _, group, _ in components for entry in group} - alpha_anchors)
    alpha_index = {index: variable for variable, index in enumerate(alpha_frames)}
    beta_index = {}
    variable_count = len(alpha_frames)
    for name, group, details in components:
        for index in details["frame_indices"]:
            if index != details["anchor_frame"] and index != 0:
                beta_index[(name, index)] = variable_count
                variable_count += 1
    pose_dimension = variable_count
    obs_shell, obs_frame, obs_track, obs_component = [], [], [], []
    pixels, weights, tracks, track_sign = [], [], [], []
    for component_index, (name, group, details) in enumerate(components):
        lookup = {}
        for entry in group:
            if entry.track_id not in lookup:
                lookup[entry.track_id] = len(tracks)
                tracks.append((name, entry.track_id))
                track_sign.append(top_shell_sign if name == "top" else -top_shell_sign)
            obs_shell.append(name)
            obs_frame.append(entry.frame_index)
            obs_track.append(lookup[entry.track_id])
            obs_component.append(component_index)
            pixels.append(entry.uv)
            weights.append(np.sqrt(entry.weight))
    oi = np.asarray(obs_frame, dtype=int)
    ti = np.asarray(obs_track, dtype=int)
    ci = np.asarray(obs_component, dtype=int)
    si = np.asarray([0 if name == "top" else 1 for name in obs_shell], dtype=int)
    pixels = np.asarray(pixels)
    weights = np.asarray(weights)
    signs = np.asarray(track_sign)
    directions, _ = unproject_to_sphere(pixels, camera, center, radius)
    inverse = np.empty_like(directions)
    for shell_id, name in enumerate(names):
        mask = si == shell_id
        inverse[mask] = np.einsum("nji,nj->ni", projected[name][oi[mask]] @ frame, directions[mask])
    landmarks = np.array([np.median(inverse[ti == index], axis=0) for index in range(len(tracks))])
    norms = np.linalg.norm(landmarks, axis=1)
    if np.any(norms < 1e-8):
        return finish("degenerate_landmark_initialization")
    landmarks /= norms[:, None]
    signed_z = signs * landmarks[:, 2]
    report["initial_landmark_cap_support"] = {}
    for name in names:
        shell_tracks = np.asarray([track_name == name for track_name, _ in tracks])
        values = signed_z[shell_tracks]
        report["initial_landmark_cap_support"][name] = {
            "landmark_count": int(np.count_nonzero(shell_tracks)),
            "outside_cap_count": int(np.count_nonzero(values < cfg.gap_fraction)),
            "median_signed_z": float(np.median(values)) if len(values) else None,
        }
    max_polar = float(np.arccos(cfg.gap_fraction))
    polar_margin = min(1e-7, max_polar * .1)
    polar = np.clip(np.arccos(np.clip(signs * landmarks[:, 2], -1, 1)), polar_margin, max_polar - polar_margin)
    azimuth = np.arctan2(landmarks[:, 1], landmarks[:, 0])
    initial = np.r_[np.zeros(pose_dimension), np.column_stack((polar, azimuth)).ravel()]
    lower = np.full(len(initial), -np.inf)
    upper = np.full(len(initial), np.inf)
    lower[pose_dimension::2] = 0.0
    upper[pose_dimension::2] = max_polar
    base_alpha = alpha.copy()
    base_beta = {name: value.copy() for name, value in beta.items()}
    base_projected = {name: value.copy() for name, value in projected.items()}

    def decode(parameters):
        a = base_alpha.copy()
        b = {name: value.copy() for name, value in base_beta.items()}
        if alpha_frames:
            a[alpha_frames] += parameters[:len(alpha_frames)]
        for (name, index), variable in beta_index.items():
            b[name][index] += parameters[variable]
        theta, phi = parameters[pose_dimension:].reshape(-1, 2).T
        points = np.column_stack((np.sin(theta) * np.cos(phi),
                                  np.sin(theta) * np.sin(phi), signs * np.cos(theta)))
        return a, b, points

    def project(parameters):
        a, b, points = decode(parameters)
        normals = np.empty_like(directions)
        for shell_id, name in enumerate(names):
            select = si == shell_id
            orientations = _model_rotations(a[oi[select]], b[name][oi[select]], frame) @ frame
            normals[select] = np.einsum("nij,nj->ni", orientations, points[ti[select]])
        xyz = center + radius * normals
        homogeneous = xyz @ camera.T
        return homogeneous[:, :2] / np.maximum(homogeneous[:, 2:], 1e-8), normals, xyz

    def residual(parameters):
        return ((project(parameters)[0] - pixels) * weights[:, None]).ravel()

    sparsity = lil_matrix((2 * len(pixels), len(initial)), dtype=np.int8)
    for index, (name, frame_id, track) in enumerate(zip(obs_shell, oi, ti)):
        if frame_id in alpha_index:
            sparsity[2*index:2*index+2, alpha_index[frame_id]] = 1
        if (name, frame_id) in beta_index:
            sparsity[2*index:2*index+2, beta_index[(name, frame_id)]] = 1
        offset = pose_dimension + 2 * track
        sparsity[2*index:2*index+2, offset:offset+2] = 1
    before = np.linalg.norm(project(initial)[0] - pixels, axis=1)
    report["before"] = _summary(before)
    try:
        fit = least_squares(residual, initial, jac_sparsity=sparsity.tocsr(),
                            bounds=(lower, upper), method="trf", tr_solver="lsmr",
                            loss="soft_l1", f_scale=opt.robust_loss_scale_px,
                            max_nfev=opt.max_nfev, x_scale="jac",
                            ftol=1e-5, xtol=1e-5, gtol=1e-5)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        report["solver_message"] = str(exc)
        return finish("solver_exception")
    report.update(solver_status=int(fit.status), solver_nfev=int(fit.nfev), solver_message=str(fit.message))
    if not fit.success or not np.all(np.isfinite(fit.x)):
        for name, _, details in components:
            for index in details["frame_indices"]:
                frame_reports[name][index]["reason"] = "solver_did_not_converge"
        return finish("solver_did_not_converge")
    predicted, normals, xyz = project(fit.x)
    after = np.linalg.norm(predicted - pixels, axis=1)
    report["after"] = _summary(after)
    robust_cost = lambda values: float(np.sum(np.sqrt(1 + (values / opt.robust_loss_scale_px)**2) - 1))
    report["robust_cost_before"] = robust_cost(residual(initial))
    report["robust_cost_after"] = robust_cost(residual(fit.x))
    if not np.all(np.isfinite(after)) or report["robust_cost_after"] > report["robust_cost_before"] + 1e-7:
        return finish("reprojection_objective_increased")
    fitted_alpha, fitted_beta, fitted_points = decode(fit.x)
    report["minimum_signed_landmark_z"] = float(np.min(signs * fitted_points[:, 2]))
    fitted_poses = {name: _model_rotations(fitted_alpha, fitted_beta[name], frame) for name in names}
    visible = np.einsum("ni,ni->n", normals, xyz) < 0
    inlier = (after <= opt.max_reprojection_px) & visible
    for component_index, (name, group, details) in enumerate(components):
        selected = ci == component_index
        local_inliers = [entry for entry, good in zip(group, inlier[selected]) if good]
        connected = _components(local_inliers, radius_px, opt)
        anchor = details["anchor_frame"]
        connected_frames = set()
        details["inlier_graph_components"] = []
        for connected_group in connected:
            group_frames = {entry.frame_index for entry in connected_group}
            details["inlier_graph_components"].append(sorted(group_frames))
            if anchor in group_frames:
                connected_frames.update(group_frames)
        correction = np.rad2deg(Rotation.from_matrix(
            fitted_poses[name] @ base_projected[name].transpose(0, 2, 1)).magnitude())
        projection_change = np.rad2deg(Rotation.from_matrix(
            base_projected[name] @ poses[name].transpose(0, 2, 1)).magnitude())
        for index in details["frame_indices"]:
            select = selected & (oi == index)
            good = select & inlier
            count = int(np.count_nonzero(good))
            fraction = float(count / np.count_nonzero(select))
            spread = _spread(pixels[good], radius_px)
            reason = None
            if count < opt.min_observations_per_frame or fraction < opt.min_inlier_fraction or spread < opt.min_spatial_spread_fraction:
                reason = "final_frame_quality_failed"
            elif index not in connected_frames:
                reason = "disconnected_inlier_graph"
            elif validity[name][index] and correction[index] > opt.max_pose_change_deg:
                reason = "pose_correction_exceeds_limit"
            entry = {"frame_index": index, "status": "unresolved" if reason else
                     ("refined" if validity[name][index] else "recovered"),
                     "reason": reason or "accepted", "anchor_frame": anchor,
                     "observation_count": int(np.count_nonzero(select)), "inlier_count": count,
                     "inlier_fraction": fraction, "spatial_spread_fraction": spread,
                     "correction_deg": float(correction[index]),
                     "mechanical_projection_change_deg": float(projection_change[index]),
                     "before": _summary(before[select]), "after": _summary(after[select])}
            frame_reports[name][index] = entry
            output_valid[name][index] = reason is None
        # An anchor failing pixel quality cannot establish this component's
        # absolute reference, even when downstream frames fit their own points.
        if not output_valid[name][anchor]:
            for index in details["frame_indices"]:
                output_valid[name][index] = False
                frame_reports[name][index].update(status="unresolved", reason="anchor_quality_failed")
        details["accepted_frames"] = [index for index in details["frame_indices"] if output_valid[name][index]]
        details["accepted"] = bool(details["accepted_frames"])
    # Use one common numeric roll even at unsupported times, but never expose
    # an unobserved independent spin as a valid result. Placeholders are solely
    # for stable serialization/rendering APIs; validity is authoritative.
    alpha = fitted_alpha
    beta = fitted_beta
    projected = fitted_poses
    return finish("completed")


__all__ = ["MechanicalConfig", "MechanicalResult", "refine_mechanical_trajectory"]
