"""Recover pose references from overlapping, measured image tracks.

An accepted frame needs two agreeing image connections to already anchored
frames. A successful adjacent increment alone cannot cross an unknown gap.
This initializes subsequent pixel fitting; it is not an accuracy certificate.
"""

from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation

from .estimate import ransac_kabsch
from .offline import OfflineConfig, OfflineResult, _select_observations, _spread
from .sphere import unproject_to_sphere, viewing_angle


def recover_observation_graph(observations, initial_rotations, initial_valid,
                              K, center, radius=1.0, config=None):
    """Use nearby forward/reverse observations, retaining the original gauge.

    Both endpoints' pixel support, spread, visibility and rotation are checked.
    Newly anchored images may extend recovery, but never across an interval
    lacking a pair of image-supported connections. Held poses are not anchors.
    """
    cfg = config if isinstance(config, OfflineConfig) else OfflineConfig.from_mapping(config)
    poses = np.asarray(initial_rotations, dtype=float).copy()
    valid = np.asarray(initial_valid, dtype=bool).copy()
    if poses.ndim != 3 or poses.shape[1:] != (3, 3) or valid.shape != (len(poses),):
        raise ValueError("recovery requires (N,3,3) rotations and (N,) validity")
    if (not np.all(np.isfinite(poses)) or not np.allclose(poses @ poses.transpose(0, 2, 1), np.eye(3), atol=1e-6)
            or np.any(np.linalg.det(poses) < 0)):
        raise ValueError("recovery requires proper finite rotations")
    original_valid = valid.copy()
    report = {"enabled": cfg.graph_recovery_enabled, "method": "two_anchor_image_graph",
              "frames": [], "accepted_edges": 0, "rejected_edges": 0,
              "recovered_frames": 0, "note": "Measured initialization for pixel refinement; no interpolation or absolute home estimation."}
    if not cfg.enabled or not cfg.graph_recovery_enabled or not cfg.recover_invalid_frames:
        report["status"] = "disabled"
        return OfflineResult(poses, valid, report)
    data = list(observations)
    if any(entry.frame_index >= len(poses) for entry in data):
        raise ValueError("observation frame_index exceeds trajectory length")
    if not data or np.all(valid) or not np.any(valid):
        report["status"] = "no_missing_frames" if np.all(valid) else "no_anchored_observations"
        return OfflineResult(poses, valid, report)
    pixels = np.asarray([entry.uv for entry in data])
    directions, keep = unproject_to_sphere(pixels, K, center, radius)
    keep &= viewing_angle(directions, center, radius) <= np.deg2rad(cfg.limb_cull_deg)
    data = _select_observations([entry for entry, good in zip(data, keep) if good], cfg)
    if not data:
        report["status"] = "insufficient_observations"
        return OfflineResult(poses, valid, report)
    pixels = np.asarray([entry.uv for entry in data])
    directions, _ = unproject_to_sphere(pixels, K, center, radius)
    radius_px = float(np.mean((K[0, 0], K[1, 1])) * radius / np.sqrt(np.dot(center, center) - radius**2))
    tracks = defaultdict(list)
    for index, entry in enumerate(data):
        tracks[entry.track_id].append(index)
    edges = defaultdict(list)
    neighbors = defaultdict(set)
    for indices in tracks.values():
        ordered = sorted(indices, key=lambda index: data[index].frame_index)
        for offset, first in enumerate(ordered):
            a = data[first].frame_index
            for second in ordered[offset + 1:]:
                b = data[second].frame_index
                if b - a > cfg.backward_window_frames:
                    break
                edges[a, b].append((first, second))
                neighbors[a].add(b)
                neighbors[b].add(a)
    cache = {}

    def edge(a, b):
        key = tuple(sorted((a, b)))
        if key not in cache:
            pairs = np.asarray(edges[key], dtype=int)
            # Templates can observe the same corner; don't count duplicates as
            # independent evidence just because their namespaces differ.
            _, unique = np.unique(np.round(pixels[pairs].reshape(-1, 4), 1), axis=0, return_index=True)
            pairs = pairs[np.sort(unique)]
            result = None
            if len(pairs) >= cfg.min_observations_per_frame:
                first, second = pairs.T
                fit = ransac_kabsch(directions[first], directions[second], iters=100,
                                    inlier_rad=np.deg2rad(1.0),
                                    min_inliers=cfg.min_observations_per_frame, rng=key[0] * 1009 + key[1])
                if fit is not None:
                    rotation, inliers = fit
                    angle = np.rad2deg(Rotation.from_matrix(rotation).magnitude())
                    support = min(_spread(pixels[first[inliers]], radius_px),
                                  _spread(pixels[second[inliers]], radius_px))
                    if (np.mean(inliers) >= cfg.min_inlier_fraction
                            and support >= cfg.min_spatial_spread_fraction
                            and angle <= cfg.graph_recovery_max_step_deg):
                        result = (rotation, first[inliers], second[inliers])
            cache[key] = result
            report["accepted_edges" if result is not None else "rejected_edges"] += 1
        value = cache[key]
        if value is None:
            return None
        rotation, first, second = value
        return (rotation, first, second) if a < b else (rotation.T, second, first)

    # Alternating directions also permits recovery before a trustworthy later
    # image; it does not preferentially assume that motion runs forwards.
    changed = True
    while changed:
        changed = False
        for order in (range(len(poses)), range(len(poses) - 1, -1, -1)):
            for target in order:
                if valid[target]:
                    continue
                anchors = sorted((j for j in neighbors[target] if valid[j]),
                                 key=lambda j: (abs(j - target), j))
                candidates = []
                for anchor in anchors[:8]:
                    value = edge(anchor, target)
                    if value is not None:
                        relative, source_ids, target_ids = value
                        candidates.append((anchor, relative @ poses[anchor], source_ids, target_ids))
                if len(candidates) < 2:
                    continue
                proposals = np.array([item[1] for item in candidates])
                disagreement = max(float(np.rad2deg(Rotation.from_matrix(a @ b.T).magnitude()))
                                   for i, a in enumerate(proposals) for b in proposals[i + 1:])
                if disagreement > cfg.max_pose_change_deg:
                    continue
                reference = np.concatenate([directions[src] @ poses[anchor]
                                            for anchor, _, src, _ in candidates])
                observed = np.concatenate([directions[dst] for _, _, _, dst in candidates])
                fit = ransac_kabsch(reference, observed, iters=150, inlier_rad=np.deg2rad(1.0),
                                    min_inliers=cfg.min_observations_per_frame, rng=target + 1701)
                if fit is None:
                    continue
                candidate = fit[0]
                supported = []
                for anchor, _, source_ids, target_ids in candidates:
                    normals = (directions[source_ids] @ poses[anchor]) @ candidate.T
                    xyz = np.asarray(center) + radius * normals
                    projected = xyz @ K.T
                    predicted = projected[:, :2] / projected[:, 2:]
                    error = np.linalg.norm(predicted - pixels[target_ids], axis=1)
                    inliers = ((error <= cfg.max_reprojection_px)
                               & (np.einsum("ni,ni->n", normals, xyz) < 0))
                    spread = min(_spread(pixels[target_ids[inliers]], radius_px),
                                 _spread(pixels[source_ids[inliers]], radius_px))
                    if (np.count_nonzero(inliers) >= cfg.min_observations_per_frame
                            and np.mean(inliers) >= cfg.min_inlier_fraction
                            and spread >= cfg.min_spatial_spread_fraction):
                        supported.append({"frame_index": anchor, "inliers": int(inliers.sum()),
                                          "median_error_px": float(np.median(error[inliers]))})
                if len(supported) < 2:
                    continue
                # Do not average over inconsistent edges and quietly keep only
                # a convenient subset of the tested, trustworthy references.
                if len(supported) != len(candidates):
                    continue
                poses[target] = candidate
                valid[target] = True
                changed = True
                report["frames"].append({"frame_index": target, "status": "image_graph_recovered",
                                          "anchors": supported, "anchor_disagreement_deg": disagreement})
    report["recovered_frames"] = int(np.count_nonzero(valid & ~original_valid))
    report["unresolved_frames"] = np.flatnonzero(~valid).tolist()
    report["status"] = "completed"
    return OfflineResult(poses, valid, report)
