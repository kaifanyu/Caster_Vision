"""Calibrate a separated-hemisphere pivot from reviewed outer silhouette arcs.

The equal shell curvature radius is the length unit. With known camera and
initial axes, the only unknown is the three-dimensional pivot divided by that
radius. A silhouette ray is tangent to its shell's translated parent sphere;
rim edges and yoke edges are not measurements of that tangent cone.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .camera import validate_camera_matrix
from .rotation import Rx, is_rotation_matrix
from .sphere import fit_circle_points, sphere_pose_from_circle


def _samples(frames, K, F, top_shell_sign):
    rows, directions, signs = [], [], []
    seen = set()
    for frame in frames:
        index = frame.get("frame_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in seen:
            raise ValueError("each frame_index must be a unique nonnegative integer")
        seen.add(index)
        if index != 0 and "alpha_deg" not in frame:
            raise ValueError("nonzero frame_index requires an explicit known alpha_deg")
        alpha = float(frame.get("alpha_deg", 0.0))
        if not np.isfinite(alpha):
            raise ValueError("alpha_deg must be a finite known roll relative to the first frame")
        normal = (F @ Rx(np.deg2rad(alpha)))[:, 2]
        for shell in ("top", "bottom"):
            points = np.asarray(frame.get(shell, []), dtype=float)
            if not points.size:
                continue
            if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
                raise ValueError(f"{shell} points must be finite [u, v] pairs")
            if len(np.unique(points, axis=0)) != len(points):
                raise ValueError("duplicate silhouette points do not supply independent support")
            sign = top_shell_sign if shell == "top" else -top_shell_sign
            for point in points:
                rows.append({"frame_index": index, "alpha_deg": alpha, "shell": shell,
                             "observed_uv": point.tolist()})
                directions.append(normal)
                signs.append(sign)
    if not rows:
        raise ValueError("no reviewed silhouette points were supplied")
    pixels = np.array([row["observed_uv"] for row in rows])
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(K).T
    return rows, pixels, rays, np.array(directions), np.array(signs)


def _tangent_residual(pivot, rays, offsets, K):
    """Signed Sampson distance of the exact projected sphere cone, in pixels."""
    centers = pivot + offsets
    along = np.einsum("ij,ij->i", rays, centers)
    constant = np.sum(centers**2, axis=1) - 1.0
    implicit = along**2 - constant * np.sum(rays**2, axis=1)
    gradient_ray = 2.0 * (along[:, None] * centers - constant[:, None] * rays)
    gradient_pixel = gradient_ray @ np.linalg.inv(K)[:, :2]
    return implicit / np.maximum(np.linalg.norm(gradient_pixel, axis=1), 1e-12)


def _tangent_prediction(pivot, rays, offsets, K):
    centers = pivot + offsets
    distances = np.linalg.norm(centers, axis=1)
    axis = centers / distances[:, None]
    ray = rays / np.linalg.norm(rays, axis=1)[:, None]
    radial = ray - np.einsum("ij,ij->i", ray, axis)[:, None] * axis
    radial /= np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-12)
    sine = 1.0 / distances
    cosine = np.sqrt(np.maximum(1.0 - sine**2, 0.0))
    tangent_ray = cosine[:, None] * axis + sine[:, None] * radial
    tangent_distance = np.sqrt(np.maximum(distances**2 - 1.0, 0.0))
    normals = tangent_distance[:, None] * tangent_ray - centers
    homogeneous = tangent_ray @ K.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    basis = np.tile([1.0, 0.0, 0.0], (len(axis), 1))
    near_x = np.abs(axis[:, 0]) > 0.9
    basis[near_x] = [0.0, 1.0, 0.0]
    first = basis - np.einsum("ij,ij->i", basis, axis)[:, None] * axis
    first /= np.linalg.norm(first, axis=1)[:, None]
    second = np.cross(axis, first)
    angle = np.arctan2(np.einsum("ij,ij->i", radial, second),
                       np.einsum("ij,ij->i", radial, first))
    return pixels, normals, angle


def _coverage(angles):
    if len(angles) < 2:
        return 0.0
    angles = np.sort(np.mod(angles, 2 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    return float(np.rad2deg(2 * np.pi - np.max(gaps)))


def calibrate_shell_pivot(
    frames, K, F, *, gap_fraction=0.1, top_shell_sign=1, initial_pivot=None,
    min_points_per_shell=6, max_residual_px=3.0, min_inlier_fraction=0.8,
    min_arc_coverage_deg=60.0, max_condition_number=100.0,
):
    """Return a gated candidate calibration; this function never changes config.

    ``frames`` contains ``frame_index``, known ``alpha_deg`` (only frame zero
    defaults to zero), and ``top``/``bottom`` lists of reviewed undistorted
    [u,v] silhouette points. Shell names follow color labels and top_shell_sign.
    """
    camera = validate_camera_matrix(K)
    frame = np.asarray(F, dtype=float)
    if not is_rotation_matrix(frame):
        raise ValueError("F must be a proper ball-to-camera orientation")
    if not np.isfinite(gap_fraction) or not 0 <= gap_fraction < 1:
        raise ValueError("gap_fraction must be finite in [0, 1)")
    if isinstance(top_shell_sign, bool) or top_shell_sign not in (-1, 1):
        raise ValueError("top_shell_sign must be +1 or -1")
    if isinstance(min_points_per_shell, bool) or not isinstance(min_points_per_shell, int) or min_points_per_shell < 3:
        raise ValueError("min_points_per_shell must be an integer >= 3")
    if (not np.isfinite(max_residual_px) or max_residual_px <= 0
            or not 0 < min_inlier_fraction <= 1 or not 0 < min_arc_coverage_deg <= 360
            or not np.isfinite(max_condition_number) or max_condition_number <= 1):
        raise ValueError("invalid residual, inlier fraction, arc coverage or conditioning limits")
    rows, pixels, rays, directions, signs = _samples(list(frames), camera, frame, top_shell_sign)
    shell_masks = {name: np.array([row["shell"] == name for row in rows]) for name in ("top", "bottom")}
    for name, mask in shell_masks.items():
        if mask.sum() < min_points_per_shell:
            raise ValueError(f"need at least {min_points_per_shell} reviewed outer-silhouette points for {name}")
    offsets = signs[:, None] * gap_fraction * directions
    if initial_pivot is None:
        seeds = []
        for name, mask in shell_masks.items():
            for index in {row["frame_index"] for row in rows}:
                selected = mask & np.array([row["frame_index"] == index for row in rows])
                if selected.sum() >= 3:
                    try:
                        center, _ = sphere_pose_from_circle(*fit_circle_points(pixels[selected]), camera)
                        seeds.append(center - offsets[selected][0])
                    except ValueError:
                        pass
        if not seeds:
            raise ValueError("silhouette arcs cannot initialize a sphere; supply initial_pivot")
        initial_pivot = np.median(seeds, axis=0)
    initial = np.asarray(initial_pivot, dtype=float)
    if initial.shape != (3,) or not np.all(np.isfinite(initial)) or initial[2] <= 1 + gap_fraction:
        raise ValueError("initial_pivot must be a finite 3-vector with z > 1 + gap_fraction")
    residual = lambda value: _tangent_residual(value, rays, offsets, camera)
    fit = least_squares(residual, initial, bounds=([-np.inf, -np.inf, 1 + gap_fraction + 1e-6], np.inf),
                        loss="soft_l1", f_scale=1.0, max_nfev=300, x_scale="jac",
                        ftol=1e-11, xtol=1e-11, gtol=1e-11)
    prediction, normals, angles = _tangent_prediction(fit.x, rays, offsets, camera)
    errors = np.linalg.norm(prediction - pixels, axis=1)
    correct_half = signs * np.einsum("ij,ij->i", normals, directions) >= -1e-6
    inliers = (errors <= max_residual_px) & correct_half
    reasons = []
    if not fit.success or not np.all(np.isfinite(fit.x)):
        reasons.append("solver_did_not_converge")
    per_shell = {}
    for name, mask in shell_masks.items():
        good = mask & inliers
        per_frame_coverage = {
            str(index): _coverage(angles[good & np.array([row["frame_index"] == index for row in rows])])
            for index in sorted({row["frame_index"] for row in rows})
        }
        # Tangent-cone bases differ between images. Pooling their angles would
        # fabricate broad support from individually narrow contour fragments.
        coverage = max(per_frame_coverage.values(), default=0.0)
        per_shell[name] = {
            "point_count": int(mask.sum()), "inlier_count": int(good.sum()),
            "inlier_fraction": float(good.sum() / mask.sum()),
            "arc_coverage_deg": coverage, "arc_coverage_by_frame_deg": per_frame_coverage,
            "median_error_px": float(np.median(errors[mask])),
            "p95_error_px": float(np.percentile(errors[mask], 95)),
            "wrong_hemisphere_count": int(np.count_nonzero(mask & ~correct_half)),
        }
        if good.sum() < min_points_per_shell or good.sum() / mask.sum() < min_inlier_fraction:
            reasons.append(f"{name}_insufficient_inlier_support")
        if coverage < min_arc_coverage_deg:
            reasons.append(f"{name}_arc_too_narrow")
    jacobian = np.empty((int(inliers.sum()), 3))
    for column in range(3):
        step = 1e-6 * max(1.0, abs(fit.x[column]))
        delta = np.zeros(3)
        delta[column] = step
        jacobian[:, column] = ((residual(fit.x + delta) - residual(fit.x - delta)) / (2 * step))[inliers]
    singular = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if len(singular) == 3 and singular[-1] > 1e-8 else float("inf")
    if condition > max_condition_number:
        reasons.append("poorly_conditioned_pivot")
    for row, predicted, error, good, hemisphere in zip(rows, prediction, errors, inliers, correct_half):
        row.update(predicted_uv=predicted.tolist(), error_px=float(error), inlier=bool(good),
                   correct_hemisphere=bool(hemisphere))
    return {
        "schema_version": 1, "status": "candidate_passed" if not reasons else "candidate_rejected",
        "accepted": not reasons, "reasons": reasons, "geometry": "separated_hemispheres",
        "coordinate_system": "undistorted_pixels", "pivot_camera": fit.x.tolist(),
        "radius_units": "one shell curvature radius", "normalized_shell_radius": 1.0,
        "gap_fraction": float(gap_fraction), "top_shell_sign": int(top_shell_sign),
        "K": camera.tolist(), "initial_frame": frame.tolist(), "initial_pivot": initial.tolist(),
        "solver_message": str(fit.message), "solver_evaluations": int(fit.nfev),
        "per_shell": per_shell, "condition_number": condition if np.isfinite(condition) else None,
        "jacobian_singular_values": singular.tolist(), "observations": rows,
        "limits": {"min_points_per_shell": min_points_per_shell, "max_residual_px": max_residual_px,
                   "min_inlier_fraction": min_inlier_fraction, "min_arc_coverage_deg": min_arc_coverage_deg,
                   "max_condition_number": max_condition_number},
        "limitations": [
            "This is a candidate calibration from supplied contour points; pixel residuals are not an absolute accuracy estimate.",
            "Camera, initial axes, known frame rolls and gap ratio are fixed inputs and are not calibrated here.",
            "Only the pivot divided by shell curvature radius is measured; metric radius cannot be inferred from these images.",
            "Outer silhouette arcs must exclude rims, yoke, shadows and paint boundaries. Incorrect edge labels can bias the fit.",
        ],
    }
