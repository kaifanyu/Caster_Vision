"""Offline spherical bundle adjustment from persistent image observations.

This fits camera-space poses and unknown material points on a fixed sphere.
It does not smooth motion, assume no slip, or infer an absolute mechanical
home pose. Each connected observation graph inherits one trusted input pose.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
import heapq
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from .camera import validate_camera_matrix
from .estimate import ransac_kabsch
from .sphere import unproject_to_sphere, viewing_angle


@dataclass(frozen=True)
class OfflineConfig:
    enabled: bool = False
    min_track_length: int = 3
    max_tracks_per_frame: int = 100
    robust_loss_scale_px: float = 1.0
    max_nfev: int = 40
    min_observations_per_frame: int = 12
    max_pose_change_deg: float = 5.0
    max_reprojection_px: float = 2.0
    min_inlier_fraction: float = 0.7
    min_spatial_spread_fraction: float = 0.04
    limb_cull_deg: float = 65.0
    recover_invalid_frames: bool = True
    backward_window_frames: int = 12
    backward_stride: int = 3
    rate_window_s: float = 0.25
    rate_polynomial_order: int = 2
    window_frames: int = 0
    window_overlap_frames: int = 12
    graph_recovery_enabled: bool = False
    graph_recovery_max_step_deg: float = 30.0

    def __post_init__(self) -> None:
        for name in ("enabled", "recover_invalid_frames", "graph_recovery_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"offline.{name} must be boolean")
        for name, minimum in (("min_track_length", 3), ("max_tracks_per_frame", 3),
                              ("max_nfev", 1), ("min_observations_per_frame", 3),
                              ("backward_window_frames", 1), ("backward_stride", 1),
                              ("rate_polynomial_order", 1), ("window_frames", 0),
                              ("window_overlap_frames", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"offline.{name} must be an integer >= {minimum}")
        if self.max_tracks_per_frame < self.min_observations_per_frame:
            raise ValueError("offline.max_tracks_per_frame must cover min_observations_per_frame")
        for name in ("robust_loss_scale_px", "max_pose_change_deg", "max_reprojection_px",
                     "min_spatial_spread_fraction", "limb_cull_deg", "rate_window_s",
                     "graph_recovery_max_step_deg"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"offline.{name} must be finite and positive")
        if isinstance(self.min_inlier_fraction, bool) or not np.isfinite(self.min_inlier_fraction) or not 0 < self.min_inlier_fraction <= 1:
            raise ValueError("offline.min_inlier_fraction must lie in (0, 1]")
        if self.max_pose_change_deg >= 90 or self.limb_cull_deg >= 90:
            raise ValueError("offline angular limits must be below 90 degrees")
        if self.rate_polynomial_order > 3:
            raise ValueError("offline.rate_polynomial_order must be at most three")
        if self.backward_stride > self.backward_window_frames:
            raise ValueError("offline.backward_stride must not exceed backward_window_frames")
        if self.window_frames:
            if self.window_frames < self.min_track_length:
                raise ValueError("offline.window_frames must cover min_track_length")
            if self.window_overlap_frames < 2 or self.window_overlap_frames >= self.window_frames:
                raise ValueError("offline.window_overlap_frames must be >= 2 and below window_frames")
        if self.graph_recovery_max_step_deg >= 90:
            raise ValueError("offline.graph_recovery_max_step_deg must be below 90 degrees")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> OfflineConfig:
        mapping = {} if values is None else dict(values)
        unknown = set(mapping) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown offline settings: {', '.join(sorted(unknown))}")
        return cls(**mapping)


@dataclass(frozen=True)
class SurfaceObservation:
    """Undistorted pixel observation of one persistent, per-shell material point."""

    frame_index: int
    track_id: int
    uv: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("frame_index", "track_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        pixel = np.asarray(self.uv, dtype=float)
        if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
            raise ValueError("uv must be a finite pair of undistorted pixel coordinates")
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("observation weight must be finite and positive")
        object.__setattr__(self, "uv", pixel.copy())


@dataclass
class OfflineResult:
    rotations: np.ndarray
    valid: np.ndarray
    diagnostics: dict[str, Any]


def _spread(uv: np.ndarray, radius_px: float) -> float:
    if len(uv) < 3:
        return 0.0
    covariance = np.cov(uv, rowvar=False)
    return float(np.sqrt(max(0.0, np.linalg.eigvalsh(covariance)[0])) / radius_px)


def _summary(errors: np.ndarray) -> dict[str, float | None]:
    if not len(errors):
        return {"rms_px": None, "median_px": None, "p95_px": None}
    return {"rms_px": float(np.sqrt(np.mean(errors ** 2))),
            "median_px": float(np.median(errors)),
            "p95_px": float(np.percentile(errors, 95))}


def _select_observations(observations: list[SurfaceObservation], config: OfflineConfig,
                         *, known_track_ids=(), priority_track_ids=()):
    """Bound image budgets without severing the tracks supplying their evidence.

    Selecting every image independently can spend a keyframe's entire budget on
    long tracks, then discard the shorter templates selected in intervening
    images because their reference observations were omitted. Admit a landmark's
    observations together, with at least ``min_track_length`` distinct images.
    Frame and spatial-bin occupancy balance the admitted bundles; all pixel and
    pose quality checks still happen in the estimator after this selection.
    """
    known_track_ids = set(known_track_ids)
    priority_track_ids = set(priority_track_ids) | known_track_ids
    unique: dict[tuple[int, int], SurfaceObservation] = {}
    for observation in observations:
        key = (int(observation.frame_index), int(observation.track_id))
        if key not in unique or observation.weight > unique[key].weight:
            unique[key] = observation

    def supported(entries):
        # Remove intrinsically unsupported image/landmark pairs before spending
        # the budget, and check again after selection if a frame was starved.
        entries = list(entries)
        while entries:
            tracks, frames_count = defaultdict(int), defaultdict(int)
            for entry in entries:
                tracks[entry.track_id] += 1
                frames_count[entry.frame_index] += 1
            retained = [entry for entry in entries
                        if (entry.track_id in known_track_ids or tracks[entry.track_id] >= config.min_track_length)
                        and frames_count[entry.frame_index] >= config.min_observations_per_frame]
            if len(retained) == len(entries):
                break
            entries = retained
        return entries

    candidates = supported(unique.values())
    if not candidates:
        return []
    by_frame = defaultdict(list)
    by_track = defaultdict(list)
    for entry in sorted(candidates, key=lambda item: (item.frame_index, item.track_id)):
        by_frame[entry.frame_index].append(entry)
        by_track[entry.track_id].append(entry)
    cap = config.max_tracks_per_frame
    if max(map(len, by_frame.values())) <= cap:
        return sorted(candidates, key=lambda item: (item.frame_index, item.track_id))

    cell = {}
    for frame, entries in by_frame.items():
        uv = np.array([entry.uv for entry in entries])
        extent = np.maximum(np.ptp(uv, axis=0), 1.)
        bins = np.minimum(3, ((uv - uv.min(axis=0)) / extent * 4).astype(int))
        for entry, (x, y) in zip(entries, bins):
            cell[(frame, entry.track_id)] = int(4*y+x)

    chosen = {}
    counts = defaultdict(int)
    bins_used = defaultdict(int)
    selected_track_count = defaultdict(int)
    blocked_frames = set()

    def admit(track_ids, limits):
        """Greedy spatial coverage, with lazy scores that only decrease."""
        def bundle(identifier):
            entries = by_track[identifier]
            available = [entry for entry in entries
                         if (entry.frame_index, identifier) not in chosen
                         and entry.frame_index not in blocked_frames
                         and counts[entry.frame_index] < limits[entry.frame_index]]
            if (not available or (identifier not in known_track_ids and
                    selected_track_count[identifier] + len(available) < config.min_track_length)):
                return 0., []
            # Divide by the original bundle size: removing an unavailable
            # observation cannot increase a stale heap priority. Prefer views
            # and bins with little evidence, rather than the longest template.
            gain = sum(entry.weight / ((1 + counts[entry.frame_index]) *
                       (1 + bins_used[(entry.frame_index, entry.weight > 1.,
                                       cell[(entry.frame_index, identifier)])]))
                       for entry in available) / len(entries)
            return gain, available

        queue = []
        for identifier in sorted(track_ids):
            score, available = bundle(identifier)
            if score:
                heapq.heappush(queue, (-score, -len(available), identifier))
        while queue:
            _, _, identifier = heapq.heappop(queue)
            score, available = bundle(identifier)
            if not score:
                continue
            priority = (-score, -len(available), identifier)
            if queue and priority > queue[0]:
                heapq.heappush(queue, priority)
                continue
            for entry in available:
                key = (entry.frame_index, identifier)
                chosen[key] = entry
                counts[entry.frame_index] += 1
                bins_used[(entry.frame_index, entry.weight > 1., cell[key])] += 1
                selected_track_count[identifier] += 1

    adjacent_ids = [identifier for identifier, entries in by_track.items()
                    if all(entry.weight <= 1. for entry in entries)]
    adjacent_set = set(adjacent_ids)
    direct_ids = [identifier for identifier in by_track if identifier not in adjacent_set]
    reserve = min(cap // 2, 2 * config.min_observations_per_frame)
    has_direct = {frame: any(entry.weight > 1. for entry in entries)
                  for frame, entries in by_frame.items()}
    reserved_limits = {frame: reserve if has_direct[frame] else cap for frame in by_frame}
    full_limits = {frame: cap for frame in by_frame}
    while True:
        # Keep the adjacent-track reservation independent of mapped/direct
        # priority: an absolute frame limit already spent on direct tracks
        # cannot reserve any adjacent observations afterward.
        admit(adjacent_ids, reserved_limits)
        # Preserve measured map/overlap connections before spending the rest
        # of the image budget. Known 3D landmarks need no new three-image track.
        priority_limits = {frame: max(reserve, cap // 2) for frame in by_frame}
        admit(priority_track_ids & set(by_track), priority_limits)
        # Reserve connected adjacent tracks, then prefer independently matched
        # templates. Refill unused capacity with either surviving source.
        admit(direct_ids, full_limits)
        admit(adjacent_ids, full_limits)
        retained = supported(chosen.values())
        if len(retained) == len(chosen):
            break
        lost_frames = set(counts) - {entry.frame_index for entry in retained}
        blocked_frames.update(lost_frames)
        chosen.clear()
        counts.clear()
        bins_used.clear()
        selected_track_count.clear()
        for entry in retained:
            key = (entry.frame_index, entry.track_id)
            chosen[key] = entry
            counts[entry.frame_index] += 1
            bins_used[(entry.frame_index, entry.weight > 1., cell[key])] += 1
            selected_track_count[entry.track_id] += 1
        # Lost frames cannot provide a complete track. Their removal may free
        # slots in supported frames; the next pass fills those slots coherently.
    return sorted(chosen.values(), key=lambda entry: (entry.frame_index, entry.track_id))


def _exclude_unsupported_invalid_frames(observations, valid, radius_px, config, *, known_track_ids=()):
    """Do not let a missing pose prevent supported poses from being refined.

    Recovery needs several well-spread landmarks each already observed in two
    initially trusted images. An unsupported invalid pose contributes no image
    constraints to this fit, and keeps its original false validity. Duplicated
    anchor observations do not count as independent images.
    """
    unique = {}
    for entry in observations:
        key = (entry.frame_index, entry.track_id)
        if key not in unique or entry.weight > unique[key].weight:
            unique[key] = entry
    trusted_support = defaultdict(int)
    by_invalid_frame = defaultdict(list)
    for entry in unique.values():
        if valid[entry.frame_index]:
            trusted_support[entry.track_id] += 1
        else:
            by_invalid_frame[entry.frame_index].append(entry)
    excluded = {}
    for frame, entries in sorted(by_invalid_frame.items()):
        supported = [entry for entry in entries if entry.track_id in known_track_ids
                     or trusted_support[entry.track_id] >= 2]
        spread = _spread(np.asarray([entry.uv for entry in supported]), radius_px)
        reason = None
        if not config.recover_invalid_frames:
            reason = "invalid_frame_recovery_disabled"
        elif len(supported) < config.min_observations_per_frame:
            reason = "invalid_frame_without_independent_landmark_support"
        elif spread < config.min_spatial_spread_fraction:
            reason = "invalid_frame_without_spread_landmark_support"
        if reason is not None:
            excluded[frame] = {"frame_index": frame, "reason": reason,
                               "supported_observation_count": len(supported),
                               "support_spatial_spread_fraction": spread}
    return ([entry for entry in observations if entry.frame_index not in excluded], excluded)


def _components(observations: list[SurfaceObservation], radius_px: float, config: OfflineConfig):
    """Connect frames only through a sufficiently spread group of tracks.

    One common point connects the abstract graph but cannot determine all
    rotational degrees of freedom. Do not let such a bridge establish an
    orientation reference for an otherwise disconnected segment.
    """
    parent: dict[int, int] = {}
    by_track: dict[int, list[SurfaceObservation]] = defaultdict(list)

    def root(index):
        parent.setdefault(index, index)
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for entry in observations:
        root(entry.frame_index)
        by_track[entry.track_id].append(entry)
    edges: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for entries in by_track.values():
        ordered = sorted(entries, key=lambda entry: entry.frame_index)
        for first, second in zip(ordered, ordered[1:]):
            edges[(first.frame_index, second.frame_index)].append((first.uv, second.uv))
    for (first, second), samples in edges.items():
        if len(samples) < config.min_observations_per_frame:
            continue
        pixels = np.asarray(samples)
        if min(_spread(pixels[:, 0], radius_px), _spread(pixels[:, 1], radius_px)) < config.min_spatial_spread_fraction:
            continue
        parent[root(first)] = root(second)
    groups: dict[int, list[SurfaceObservation]] = defaultdict(list)
    for entry in observations:
        groups[root(entry.frame_index)].append(entry)
    return sorted(groups.values(), key=lambda group: group[0].frame_index)


def _tangent_basis(directions: np.ndarray) -> np.ndarray:
    axes = np.eye(3)[np.argmin(np.abs(directions), axis=1)]
    first = np.cross(directions, axes)
    first /= np.linalg.norm(first, axis=1)[:, None]
    return np.stack((first, np.cross(directions, first)), axis=2)


def _fit_component(observations, rotations, valid, K, C, radius, radius_px, config,
                   fixed_frames=()):
    frame_ids = sorted({entry.frame_index for entry in observations})
    trusted = [frame for frame in frame_ids if valid[frame]]
    report: dict[str, Any] = {"frame_indices": frame_ids, "observation_count": len(observations),
                              "accepted": False}
    if not trusted:
        report["reason"] = "no_trusted_pose_anchor"
        return None, report
    anchor = trusted[0]
    report["anchor_frame"] = anchor
    report["inherits_input_reference"] = True
    track_ids = sorted({entry.track_id for entry in observations})
    frame_lookup = {frame: index for index, frame in enumerate(frame_ids)}
    track_lookup = {track: index for index, track in enumerate(track_ids)}
    fi = np.array([frame_lookup[entry.frame_index] for entry in observations])
    ti = np.array([track_lookup[entry.track_id] for entry in observations])
    pixels = np.array([entry.uv for entry in observations])
    weights = np.sqrt(np.array([entry.weight for entry in observations]))
    directions, _ = unproject_to_sphere(pixels, K, C, radius)
    base_poses = rotations[frame_ids].copy()
    trusted_observation = valid[np.asarray(frame_ids)[fi]]
    inverse_directions = np.einsum("nji,nj->ni", base_poses[fi], directions)
    landmarks = np.empty((len(track_ids), 3))
    supported_by_trusted = np.zeros(len(track_ids), dtype=int)
    for index in range(len(track_ids)):
        select = (ti == index) & trusted_observation
        supported_by_trusted[index] = np.count_nonzero(select)
        if not np.any(select):
            select = ti == index
        landmarks[index] = np.median(inverse_directions[select], axis=0)
    norms = np.linalg.norm(landmarks, axis=1)
    if np.any(norms < 1e-8):
        report["reason"] = "degenerate_landmark_initialization"
        return None, report
    landmarks /= norms[:, None]

    # Recover a missing pose only when landmarks are independently observed in
    # at least two trusted images. No interpolation through an unobserved gap.
    for local_frame, frame in enumerate(frame_ids):
        if valid[frame]:
            continue
        select = (fi == local_frame) & (supported_by_trusted[ti] >= 2)
        if not config.recover_invalid_frames or np.count_nonzero(select) < config.min_observations_per_frame:
            report["reason"] = "invalid_frame_without_independent_landmark_support"
            return None, report
        fit = ransac_kabsch(landmarks[ti[select]], directions[select], iters=100,
                            inlier_rad=np.deg2rad(1.0),
                            min_inliers=config.min_observations_per_frame, rng=0)
        if fit is None:
            report["reason"] = "invalid_frame_initialization_failed"
            return None, report
        base_poses[local_frame] = fit[0]

    for index in range(len(frame_ids)):
        if _spread(pixels[fi == index], radius_px) < config.min_spatial_spread_fraction:
            report["reason"] = "insufficient_spatial_spread"
            return None, report

    fixed = ({int(index) for index in fixed_frames if index in frame_ids and valid[index]}
             | {anchor})
    report["fixed_anchor_frames"] = sorted(fixed)
    active_frames = [frame for frame in frame_ids if frame not in fixed]
    pose_lookup = {frame: index for index, frame in enumerate(active_frames)}
    pose_index = np.array([pose_lookup.get(frame, -1) for frame in frame_ids])
    pose_dimension = 3 * len(active_frames)
    dimension = pose_dimension + 2 * len(track_ids)
    basis = _tangent_basis(landmarks)

    def decode(parameters):
        poses = base_poses.copy()
        if active_frames:
            delta = Rotation.from_rotvec(parameters[:pose_dimension].reshape(-1, 3)).as_matrix()
            active = pose_index >= 0
            poses[active] = delta[pose_index[active]] @ base_poses[active]
        tangent = parameters[pose_dimension:].reshape(-1, 2)
        points = landmarks + np.einsum("nij,nj->ni", basis, tangent)
        points /= np.linalg.norm(points, axis=1)[:, None]
        return poses, points

    def project(parameters):
        poses, points = decode(parameters)
        normals = np.einsum("nij,nj->ni", poses[fi], points[ti])
        xyz = C + radius * normals
        homogeneous = xyz @ K.T
        return homogeneous[:, :2] / np.maximum(homogeneous[:, 2:], 1e-8), normals, xyz

    def residual(parameters):
        prediction, _, _ = project(parameters)
        return ((prediction - pixels) * weights[:, None]).ravel()

    sparsity = lil_matrix((2 * len(observations), dimension), dtype=np.int8)
    for index, (frame, track) in enumerate(zip(fi, ti)):
        local_pose = pose_index[frame]
        if local_pose >= 0:
            sparsity[2*index:2*index+2, 3*local_pose:3*local_pose+3] = 1
        offset = pose_dimension + 2 * track
        sparsity[2*index:2*index+2, offset:offset+2] = 1
    zeros = np.zeros(dimension)
    before = np.linalg.norm(project(zeros)[0] - pixels, axis=1)
    report["before"] = _summary(before)
    try:
        solution = least_squares(residual, zeros, jac_sparsity=sparsity.tocsr(),
                                 method="trf", tr_solver="lsmr", loss="soft_l1",
                                 f_scale=config.robust_loss_scale_px, max_nfev=config.max_nfev,
                                 ftol=1e-5, xtol=1e-5, gtol=1e-5,
                                 x_scale="jac")
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        report["reason"] = "solver_exception"
        report["message"] = str(exc)
        return None, report
    report["solver_status"] = int(solution.status)
    report["solver_nfev"] = int(solution.nfev)
    report["solver_message"] = str(solution.message)
    report["landmark_count"] = len(track_ids)
    if not solution.success or not np.all(np.isfinite(solution.x)):
        report["reason"] = "solver_did_not_converge"
        return None, report
    fitted, _ = decode(solution.x)
    prediction, normals, xyz = project(solution.x)
    after = np.linalg.norm(prediction - pixels, axis=1)
    report["after"] = _summary(after)
    scale = config.robust_loss_scale_px
    robust_before = float(np.sum(np.sqrt(1 + (residual(zeros) / scale)**2) - 1))
    robust_after = float(np.sum(np.sqrt(1 + (residual(solution.x) / scale)**2) - 1))
    report["robust_cost_before"] = robust_before
    report["robust_cost_after"] = robust_after
    if not np.all(np.isfinite(after)) or robust_after > robust_before + 1e-7:
        report["reason"] = "reprojection_objective_increased"
        return None, report
    corrections = np.rad2deg(Rotation.from_matrix(fitted @ rotations[frame_ids].transpose(0, 2, 1)).magnitude())
    report["max_correction_deg"] = float(corrections.max())
    # A recovered pose may legitimately move far from a held invalid input.
    # Limit changes to trusted input poses; observations gate recovered poses.
    if np.any(corrections[valid[frame_ids]] > config.max_pose_change_deg):
        report["reason"] = "pose_correction_exceeds_limit"
        return None, report
    visible = np.einsum("ni,ni->n", normals, xyz) < 0
    per_frame = []
    for local_frame, frame in enumerate(frame_ids):
        select = fi == local_frame
        inliers = select & (after <= config.max_reprojection_px) & visible
        count = int(np.count_nonzero(inliers))
        fraction = count / np.count_nonzero(select)
        spread = _spread(pixels[inliers], radius_px)
        entry = {"frame_index": frame, "status": "refined" if valid[frame] else "recovered",
                 "observation_count": int(np.count_nonzero(select)), "inlier_count": count,
                 "inlier_fraction": float(fraction), "spatial_spread_fraction": spread,
                 "correction_deg": float(corrections[local_frame]),
                 "before": _summary(before[select]), "after": _summary(after[select])}
        if frame in fixed:
            entry["status"] = "anchor"
        per_frame.append(entry)
        if count < config.min_observations_per_frame or fraction < config.min_inlier_fraction or spread < config.min_spatial_spread_fraction:
            report["reason"] = "final_frame_quality_failed"
            report["failed_frame"] = frame
            report["frames"] = per_frame
            return None, report
    # Per-frame support alone is insufficient: separate groups can each fit
    # their own landmarks while their only shared tracks are outliers. Such a
    # solution has no reliable image connection back to this component's gauge
    # anchor. Require the geometrically supported inlier graph to remain whole.
    retained = [entry for entry, keep in zip(observations,
                (after <= config.max_reprojection_px) & visible) if keep]
    inlier_groups = _components(retained, radius_px, config)
    report["inlier_graph_components"] = [sorted({entry.frame_index for entry in group})
                                          for group in inlier_groups]
    if len(inlier_groups) != 1 or set(report["inlier_graph_components"][0]) != set(frame_ids):
        report["reason"] = "disconnected_inlier_graph"
        report["frames"] = per_frame
        return None, report
    report["accepted"] = True
    report["reason"] = "accepted"
    report["frames"] = per_frame
    return fitted, report


def _refine_trajectory_batch(
    observations: Iterable[SurfaceObservation],
    initial_rotations: np.ndarray,
    initial_valid: np.ndarray,
    K: np.ndarray,
    C: np.ndarray,
    radius: float = 1.0,
    config: OfflineConfig | Mapping[str, Any] | None = None,
    *, fixed_frames=(),
) -> OfflineResult:
    """Jointly fit supported poses and material landmarks to actual pixels.

    Rotations map first-image surface directions to camera-space directions.
    ``initial_valid`` describes absolute pose validity, not just pair success.
    Unsupported or rejected components retain input poses and validity. No
    temporal interpolation, kinematic constraint, or guessed home pose is used.
    ``diagnostics`` is JSON serializable and includes every frame's status.
    """
    cfg = config if isinstance(config, OfflineConfig) else OfflineConfig.from_mapping(config)
    rotations = np.asarray(initial_rotations, dtype=float).copy()
    valid = np.asarray(initial_valid, dtype=bool).copy()
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3) or valid.shape != (len(rotations),):
        raise ValueError("initial_rotations must be (N,3,3), initial_valid must be (N,)")
    if not np.all(np.isfinite(rotations)) or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-6) or np.any(np.linalg.det(rotations) < 0):
        raise ValueError("initial_rotations must contain finite proper rotations")
    camera = validate_camera_matrix(K)
    center = np.asarray(C, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)) or not np.isfinite(radius) or radius <= 0 or np.linalg.norm(center) <= radius or center[2] <= radius:
        raise ValueError("sphere must have positive radius and lie in front of the camera")
    data = list(observations)
    if any(not isinstance(entry, SurfaceObservation) for entry in data):
        raise ValueError("observations must contain SurfaceObservation instances")
    if any(entry.frame_index >= len(rotations) for entry in data):
        raise ValueError("observation frame_index exceeds trajectory length")
    frame_reports = [{"frame_index": index, "status": "unrefined" if valid[index] else "unresolved",
                      "correction_deg": 0.0} for index in range(len(rotations))]
    report: dict[str, Any] = {
        "enabled": cfg.enabled, "method": "spherical_bundle_adjustment",
        "input_observation_count": len(data), "selected_observation_count": 0,
        "optimized_components": 0, "rejected_components": 0, "refined_frames": 0,
        "recovered_frames": 0, "components": [], "frames": frame_reports,
        "excluded_invalid_frames": [],
        "limitations": ["Reprojection residuals are not absolute accuracy estimates.",
                        "Fixed camera/sphere geometry and persistent feature identities are assumed.",
                        "Each component inherits its earliest trusted input pose; no absolute home pose is estimated.",
                        "Unsupported intervals remain unresolved; no motion is interpolated."]}
    if not cfg.enabled or not data or len(rotations) < cfg.min_track_length:
        report["status"] = "disabled" if not cfg.enabled else "insufficient_observations"
        return OfflineResult(rotations, valid, report)
    pixels = np.array([entry.uv for entry in data])
    directions, geometry_valid = unproject_to_sphere(pixels, camera, center, radius)
    geometry_valid &= viewing_angle(directions, center, radius) <= np.deg2rad(cfg.limb_cull_deg)
    report["geometry_rejected_observations"] = int(np.count_nonzero(~geometry_valid))
    radius_px = float(np.mean((camera[0, 0], camera[1, 1])) * radius / np.sqrt(np.dot(center, center) - radius**2))
    selected = [entry for entry, keep in zip(data, geometry_valid) if keep]
    excluded_frames = {}
    # Spatial selection or minimum-length pruning can remove supporting
    # trusted observations. Recheck until the retained graph is stable.
    while selected:
        previous_count = len(selected)
        selected, excluded = _exclude_unsupported_invalid_frames(selected, valid, radius_px, cfg)
        excluded_frames.update(excluded)
        selected = _select_observations(selected, cfg)
        if len(selected) == previous_count:
            break
    report["excluded_invalid_frames"] = list(excluded_frames.values())
    for index, entry in excluded_frames.items():
        frame_reports[index].update(entry)
    report["selected_observation_count"] = len(selected)
    for component in _components(selected, radius_px, cfg):
        component_selected = _select_observations(component, cfg)
        if not component_selected:
            # No multi-frame evidence survived within this geometrically
            # connected segment. Leave the original trajectory untouched.
            continue
        fitted, details = _fit_component(component_selected, rotations, valid, camera, center, radius, radius_px, cfg,
                                         fixed_frames=fixed_frames)
        report["components"].append(details)
        frame_ids = details["frame_indices"]
        if fitted is None:
            report["rejected_components"] += 1
            for index in frame_ids:
                frame_reports[index]["status"] = "rejected" if valid[index] else "unresolved"
                frame_reports[index]["reason"] = details["reason"]
            continue
        report["optimized_components"] += 1
        for entry in details["frames"]:
            index = entry["frame_index"]
            if entry["status"] == "recovered":
                report["recovered_frames"] += 1
            elif entry["status"] == "refined":
                report["refined_frames"] += 1
            frame_reports[index] = entry
        rotations[frame_ids] = fitted
        valid[frame_ids] = True
    report["status"] = "completed" if selected else "insufficient_observations"
    return OfflineResult(rotations, valid, report)


def _refine_trajectory_windows(
    observations: Iterable[SurfaceObservation],
    initial_rotations: np.ndarray,
    initial_valid: np.ndarray,
    K: np.ndarray,
    C: np.ndarray,
    radius: float = 1.0,
    config: OfflineConfig | Mapping[str, Any] | None = None,
) -> OfflineResult:
    """Refine image-supported rotations in one batch or overlapping windows.

    A window keeps accepted overlap poses fixed in the original camera gauge.
    Failed windows retry without explicitly failed observations and then use
    smaller neighboring windows. No pose is interpolated across missing data;
    rejected components retain their input poses and validity. Every component
    requires a trusted input pose or an earlier image-supported accepted pose.
    """

    cfg = config if isinstance(config, OfflineConfig) else OfflineConfig.from_mapping(config)
    if not cfg.enabled or not cfg.window_frames:
        return _refine_trajectory_batch(observations, initial_rotations, initial_valid, K, C, radius, cfg)
    data = list(observations)
    # Reuse batch validation and initial diagnostics without performing a fit.
    base = _refine_trajectory_batch(data, initial_rotations, initial_valid, K, C, radius,
                                    replace(cfg, enabled=False, window_frames=0))
    rotations, valid, report = base.rotations, base.valid, base.diagnostics
    original_rotations, original_valid = rotations.copy(), valid.copy()
    n = len(rotations)
    report.update(enabled=True, mode="overlapping_windows", window_frames=cfg.window_frames,
                  window_overlap_frames=cfg.window_overlap_frames, windows=[], status="completed")
    report["selected_observation_count_note"] = (
        "Counts selections across overlapping fit attempts; the same observation may occur more than once."
    )
    report["limitations"].extend([
        "Accepted overlap poses are fixed; windows inherit their input camera gauge.",
        "Rejected image fits retain the original pose validity rather than inventing motion.",
    ])
    if not data or n < cfg.min_track_length:
        report["status"] = "insufficient_observations"
        return OfflineResult(rotations, valid, report)
    batch_cfg = replace(cfg, window_frames=0)
    accepted = np.zeros(n, dtype=bool)
    visited = set()
    selected_counts = np.zeros(n, dtype=int)
    observed_frames = {entry.frame_index for entry in data}
    min_window = max(cfg.min_track_length, min(12, cfg.window_frames))

    def fit_window(start, end, depth=0):
        if (start, end) in visited or end - start < cfg.min_track_length:
            return
        visited.add((start, end))
        if all(accepted[index] for index in observed_frames if start <= index < end):
            return
        entries = [entry for entry in data if start <= entry.frame_index < end]
        fixed = np.flatnonzero(accepted[start:end]) + start
        window_report = {"start_frame": start, "end_frame_exclusive": end,
                         "fallback_depth": depth, "fixed_overlap_frames": fixed.tolist(),
                         "pruned_observation_frames": [], "accepted_frames": [], "attempts": []}
        report["windows"].append(window_report)
        accepted_here = set()
        # A single bad image must not reject an otherwise coherent local fit.
        # Prune only frames explicitly identified by the fit's pixel gate;
        # remaining poses are refitted to their own observations afterward.
        for attempt in range(3):
            fitted = _refine_trajectory_batch(entries, rotations, valid, K, C, radius,
                                               batch_cfg, fixed_frames=fixed)
            details = fitted.diagnostics
            report["selected_observation_count"] += details["selected_observation_count"]
            window_report["attempts"].append({
                "status": details["status"],
                "optimized_components": details["optimized_components"],
                "rejected_components": details["rejected_components"],
            })
            failed_frames = set()
            for component in details["components"]:
                component = dict(component, window_start=start, window_end_exclusive=end,
                                 fallback_depth=depth, attempt=attempt)
                indices = np.asarray(component["frame_indices"], dtype=int)
                if component["accepted"]:
                    trusted = indices[original_valid[indices]]
                    change = np.rad2deg(Rotation.from_matrix(
                        fitted.rotations[trusted] @ original_rotations[trusted].transpose(0, 2, 1)
                    ).magnitude()) if len(trusted) else np.empty(0)
                    component["max_original_trusted_correction_deg"] = float(change.max()) if len(change) else 0.0
                    if np.any(change > cfg.max_pose_change_deg + 1e-9):
                        component.update(accepted=False, reason="original_pose_correction_exceeds_limit")
                report["components"].append(component)
                if not component["accepted"]:
                    report["rejected_components"] += 1
                    if component.get("reason") == "final_frame_quality_failed":
                        failed_frames.add(component["failed_frame"])
                    for index in indices:
                        if not accepted[index]:
                            report["frames"][index].update(
                                status="rejected" if original_valid[index] else "unresolved",
                                reason=component["reason"],
                            )
                    continue
                report["optimized_components"] += 1
                for entry in component["frames"]:
                    index = entry["frame_index"]
                    selected_counts[index] = max(selected_counts[index], entry["observation_count"])
                    if accepted[index]:
                        # It was a fixed overlap anchor. Preserve its original
                        # accepted provenance and exact pose, including gauge.
                        continue
                    rotations[index] = fitted.rotations[index]
                    valid[index] = True
                    accepted[index] = True
                    accepted_here.add(index)
                    entry = dict(entry, window_start=start, window_end_exclusive=end)
                    if not original_valid[index]:
                        entry["status"] = "recovered"
                    entry["correction_deg"] = float(np.rad2deg(Rotation.from_matrix(
                        rotations[index] @ original_rotations[index].T).magnitude()))
                    report["frames"][index] = entry
            report["excluded_invalid_frames"].extend(details.get("excluded_invalid_frames", []))
            failed_frames -= set(window_report["pruned_observation_frames"])
            failed_frames -= set(np.flatnonzero(accepted))
            if not failed_frames or attempt == 2:
                break
            window_report["pruned_observation_frames"].extend(sorted(failed_frames))
            entries = [entry for entry in entries if entry.frame_index not in failed_frames]
            fixed = np.flatnonzero(accepted[start:end]) + start
        window_report["accepted_frames"] = sorted(accepted_here)
        unresolved_observed = [index for index in observed_frames
                               if start <= index < end and not accepted[index]]
        if not unresolved_observed or end - start <= min_window:
            return
        middle = (start + end) // 2
        overlap = min(cfg.window_overlap_frames, (end - start) // 4)
        left_end = middle + (overlap + 1) // 2
        right_start = middle - overlap // 2
        fit_window(start, left_end, depth + 1)
        fit_window(right_start, end, depth + 1)

    step = cfg.window_frames - cfg.window_overlap_frames
    for start in range(0, n, step):
        end = min(n, start + cfg.window_frames)
        if end - start < cfg.min_track_length:
            start = max(0, n - cfg.window_frames)
        fit_window(start, end)
        if end == n:
            break
    report["accepted_observation_count"] = int(selected_counts.sum())
    report["refined_frames"] = sum(entry["status"] == "refined" for entry in report["frames"])
    report["recovered_frames"] = int(np.count_nonzero(accepted & ~original_valid))
    report["accepted_frames"] = np.flatnonzero(accepted).tolist()
    if not report["components"]:
        report["status"] = "insufficient_observations"
    return OfflineResult(rotations, valid, report)


def refine_trajectory(
    observations: Iterable[SurfaceObservation],
    initial_rotations: np.ndarray,
    initial_valid: np.ndarray,
    K: np.ndarray,
    C: np.ndarray,
    radius: float = 1.0,
    config: OfflineConfig | Mapping[str, Any] | None = None,
) -> OfflineResult:
    """Fit actual image tracks, optionally recovering references first.

    Image-graph recovery requires agreeing observed connections to existing
    trusted poses. Its measured estimates may survive a rejected bundle fit,
    with their separate provenance retained in the diagnostics.
    """

    cfg = config if isinstance(config, OfflineConfig) else OfflineConfig.from_mapping(config)
    if not cfg.enabled or not cfg.graph_recovery_enabled:
        return _refine_trajectory_windows(observations, initial_rotations, initial_valid, K, C, radius, cfg)
    data = list(observations)
    validated = _refine_trajectory_batch(data, initial_rotations, initial_valid, K, C, radius,
                                         replace(cfg, enabled=False, window_frames=0))
    original_valid = validated.valid.copy()
    from .recovery import recover_observation_graph
    recovered = recover_observation_graph(data, validated.rotations, validated.valid,
                                           np.asarray(K, dtype=float), np.asarray(C, dtype=float),
                                           radius, cfg)
    result = _refine_trajectory_windows(data, recovered.rotations, recovered.valid, K, C, radius, cfg)
    result.diagnostics["graph_recovery"] = recovered.diagnostics
    newly_anchored = recovered.valid & ~original_valid
    for index in np.flatnonzero(newly_anchored):
        entry = result.diagnostics["frames"][index]
        refined = entry["status"] in ("anchor", "refined", "recovered")
        entry["refinement_status"] = entry["status"]
        entry["status"] = "recovered" if refined else "graph_recovered"
        entry["pose_source"] = "image_graph_then_bundle_adjustment" if refined else "image_graph"
    result.diagnostics["refined_frames"] = sum(
        entry["status"] == "refined" for entry in result.diagnostics["frames"])
    result.diagnostics["recovered_frames"] = int(np.count_nonzero(result.valid & ~original_valid))
    return result


__all__ = ["OfflineConfig", "SurfaceObservation", "OfflineResult", "refine_trajectory"]
