"""Recover a shell pose from an existing, immutable material landmark map.

PnP supplies an optional starting orientation only. Acceptance always uses the
fixed camera/pivot and the two mechanical angles, with the caller's original
pixel, visibility and spatial-support gates. No image can redefine the map.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares

from .camera import validate_camera_matrix
from .offline import _spread, _summary
from .rotation import Rx, Rz, is_rotation_matrix


def estimate_landmark_pose(
    observations, fixed_landmarks, K, C, F, radius=1.0, *,
    gap_fraction=0.0, geometry="common_sphere_caps", sign=1,
    initial_alpha=0.0, initial_beta=0.0, max_reprojection_px=2.0,
    min_inliers=12, min_inlier_fraction=0.7, min_spread_fraction=0.04,
):
    """Return ``(alpha, beta, diagnostics)`` or ``None`` without moving landmarks.

    Observations must describe one shell in one image. Map values are unit
    material directions in the original calibrated shell frame. Separated
    hemispheres have object coordinates ``radius * (p + sign * gap * z)``;
    common-sphere caps retain ``radius * p``. Unknown IDs supply no pose support.
    Angles are placed on the branch nearest the initial angles; re-observation
    alone cannot establish how many complete revolutions occurred in a gap.
    """
    camera = validate_camera_matrix(K)
    center = np.asarray(C, dtype=float)
    frame = np.asarray(F, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)) or center[2] <= radius:
        raise ValueError("C must be a finite sphere pivot in front of the camera")
    if not is_rotation_matrix(frame):
        raise ValueError("F must be a proper ball-to-camera rotation")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if geometry not in ("common_sphere_caps", "separated_hemispheres"):
        raise ValueError("unknown shell geometry")
    if not np.isfinite(gap_fraction) or not 0 <= gap_fraction < 1:
        raise ValueError("gap_fraction must be finite and in [0, 1)")
    if isinstance(sign, bool) or sign not in (-1, 1):
        raise ValueError("sign must be +1 or -1")
    if isinstance(min_inliers, bool) or not isinstance(min_inliers, (int, np.integer)) or min_inliers < 1:
        raise ValueError("min_inliers must be a positive integer")
    if not np.isfinite(min_inlier_fraction) or not 0 < min_inlier_fraction <= 1:
        raise ValueError("min_inlier_fraction must lie in (0, 1]")
    if any(not np.isfinite(value) or value <= 0 for value in (max_reprojection_px, min_spread_fraction)):
        raise ValueError("reprojection and spread limits must be positive")
    initial = np.asarray([initial_alpha, initial_beta], dtype=float)
    if not np.all(np.isfinite(initial)):
        raise ValueError("initial angles must be finite")

    entries = list(observations)
    if len({entry.frame_index for entry in entries}) > 1:
        raise ValueError("relocalization observations must belong to one image")
    # Multiple templates for the same map ID do not create independent support.
    by_id = {}
    for entry in entries:
        if entry.track_id in fixed_landmarks:
            previous = by_id.get(entry.track_id)
            if previous is None or entry.weight > previous.weight:
                by_id[entry.track_id] = entry
    required = max(12, int(min_inliers))
    if len(by_id) < required:
        return None
    identifiers = sorted(by_id)
    points = np.asarray([fixed_landmarks[identifier] for identifier in identifiers], dtype=float)
    pixels = np.asarray([by_id[identifier].uv for identifier in identifiers], dtype=float)
    if (points.shape != (len(identifiers), 3) or not np.all(np.isfinite(points))
            or not np.allclose(np.linalg.norm(points, axis=1), 1., atol=1e-6)):
        raise ValueError("fixed landmarks must be finite unit material directions")
    cap_cut = gap_fraction if geometry == "common_sphere_caps" else 0.
    if np.any(sign * points[:, 2] < cap_cut - 1e-7):
        return None
    object_points = points.copy()
    if geometry == "separated_hemispheres":
        object_points[:, 2] += sign * gap_fraction
    object_points *= radius
    radius_px = float(np.mean([camera[0, 0], camera[1, 1]]) * radius /
                      np.sqrt(center @ center - radius**2))

    def project(angles):
        orientation = frame @ Rx(angles[0]) @ Rz(angles[1])
        xyz = center + object_points @ orientation.T
        normals = points @ orientation.T
        homogeneous = xyz @ camera.T
        uv = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:], 1e-8)
        visible = (xyz[:, 2] > 0) & (np.einsum("ij,ij->i", normals, xyz) < 0)
        return uv, visible

    starts = [("initial_angles", initial)]
    pnp_inliers = 0
    try:
        rotation_guess = cv2.Rodrigues(frame @ Rx(initial[0]) @ Rz(initial[1]))[0]
        success, rvec, _, indices = cv2.solvePnPRansac(
            np.ascontiguousarray(object_points), np.ascontiguousarray(pixels), camera, None,
            rvec=rotation_guess, tvec=center.reshape(3, 1).copy(), useExtrinsicGuess=True,
            iterationsCount=200, reprojectionError=float(max_reprojection_px), confidence=.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if success and indices is not None:
            pnp_inliers = len(indices)
            local = frame.T @ cv2.Rodrigues(rvec)[0]
            pnp_angles = np.array([np.arctan2(-local[1, 2], local[2, 2]),
                                   np.arctan2(-local[0, 1], local[0, 0])])
            # Find the nearest mechanical orientation before the pixel solve;
            # PnP's translation is deliberately discarded.
            projected = least_squares(
                lambda angles: (Rx(angles[0]) @ Rz(angles[1]) - local).ravel(),
                pnp_angles, max_nfev=50,
            )
            if projected.success and np.all(np.isfinite(projected.x)):
                starts.append(("pnp_ransac", projected.x))
    except cv2.error:
        # Degenerate PnP samples must not prevent a supported local angle fit.
        pass

    accepted = []
    for method, start in starts:
        try:
            fit = least_squares(lambda angles: (project(angles)[0] - pixels).ravel(), start,
                                loss="soft_l1", f_scale=1., max_nfev=150,
                                ftol=1e-10, xtol=1e-10, gtol=1e-10)
            if not fit.success or not np.all(np.isfinite(fit.x)):
                continue
            angles = fit.x
            prediction, visible = project(angles)
            errors = np.linalg.norm(prediction - pixels, axis=1)
            inliers = (errors <= max_reprojection_px) & visible
            # Polish only a supported consensus, then reassess every supplied
            # known correspondence. Outliers never disappear from the denominator.
            if np.count_nonzero(inliers) >= required:
                polished = least_squares(
                    lambda value: (project(value)[0][inliers] - pixels[inliers]).ravel(),
                    angles, max_nfev=60, ftol=1e-10, xtol=1e-10, gtol=1e-10,
                )
                if polished.success and np.all(np.isfinite(polished.x)):
                    angles = polished.x
                    prediction, visible = project(angles)
                    errors = np.linalg.norm(prediction - pixels, axis=1)
                    inliers = (errors <= max_reprojection_px) & visible
        except (ValueError, np.linalg.LinAlgError):
            continue
        count = int(np.count_nonzero(inliers))
        fraction = count / len(pixels)
        spread = _spread(pixels[inliers], radius_px)
        if count < required or fraction < min_inlier_fraction or spread < min_spread_fraction:
            continue
        angles = initial + (angles - initial + np.pi) % (2 * np.pi) - np.pi
        diagnostics = {
            "method": "fixed_landmark_relocalization", "initialization": method,
            "known_observation_count": len(pixels), "inlier_count": count,
            "inlier_fraction": float(fraction), "spatial_spread_fraction": float(spread),
            "pnp_inlier_count": int(pnp_inliers), "reprojection": _summary(errors),
            "inlier_reprojection": _summary(errors[inliers]),
            "inlier_track_ids": [int(identifier) for identifier, keep in zip(identifiers, inliers) if keep],
            "angle_branch_note": "Nearest initial branch; complete turns inside a gap are unobserved.",
        }
        accepted.append((count, -float(np.median(errors[inliers])), angles, diagnostics))
    if not accepted:
        return None
    _, _, angles, diagnostics = max(accepted, key=lambda entry: entry[:2])
    return float(angles[0]), float(angles[1]), diagnostics


__all__ = ["estimate_landmark_pose"]
