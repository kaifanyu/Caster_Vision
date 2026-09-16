"""Image-only correspondence checks for saved pose trajectories.

The observations are chosen without either trajectory, then reused for every
method. Source pixels are re-anchored to the assumed shell surface: this tests
relative pixel consistency, not absolute axis accuracy or unobserved turns.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .rotation import Rx, Rz
from .shell_geometry import cap_boundary, shell_centers, surface_camera, unproject_shell
from .sphere import sphere_pose_from_circle


@dataclass
class ValidationTrajectory:
    path: Path
    source: str
    metadata: dict
    times: np.ndarray
    orientations: dict
    valid: dict
    status: dict
    geometry: str
    gap: float
    center: np.ndarray
    radius: float


def load_trajectory(path, source="auto"):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    fused = data.get("method") == "experimental_angle_kalman"
    if source == "auto":
        source = "fused" if fused else "mechanical"
    if source not in ("fused", "mechanical", "unconstrained"):
        raise ValueError("Unknown trajectory source")
    if fused and source != "fused":
        reference = Path(data["source_results"])
        return load_trajectory(reference if reference.is_absolute() else path.parent / reference, source)
    if source == "fused" and not fused:
        raise ValueError("fused source requires experimental_angle_kalman results")
    metadata = data["metadata"]
    times = np.asarray(data["frames"]["time_s"], dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("Trajectory timestamps must be finite and strictly increasing")
    F = np.asarray(metadata["R_bc"], dtype=float)
    K = np.asarray(metadata["K"], dtype=float)
    center, radius = sphere_pose_from_circle(*metadata["circle"], K)
    model = metadata.get("mechanical_model") or {}
    geometry = model.get("geometry", "common_sphere_caps")
    gap = float(model.get("gap_fraction", 0.))
    cap_boundary(gap, geometry)
    if model.get("pivot_camera") is not None:
        center = radius * np.asarray(model["pivot_camera"], dtype=float)
    orientations, valid, status = {}, {}, {}
    frames = data["frames"]
    for shell in ("top", "bottom"):
        if source == "unconstrained":
            raw = data.get("unconstrained")
            if raw is None:
                raise ValueError("Results do not preserve unconstrained poses")
            orientations[shell] = np.asarray(raw[f"{shell}_absolute"], dtype=float) @ F
            valid[shell] = np.asarray(raw["frames"][f"valid_{shell}"], dtype=bool)
            status[shell] = np.where(valid[shell], "unconstrained_visual", "unresolved")
        else:
            alpha = np.asarray(frames["alpha_rad"] if fused else frames[f"alpha_{shell}_rad"], dtype=float)
            beta = np.asarray(frames[f"beta_{shell}_rad"], dtype=float)
            gamma = np.zeros(len(times)) if fused else np.asarray(frames[f"gamma_{shell}_rad"], dtype=float)
            finite = np.isfinite(alpha) & np.isfinite(beta) & np.isfinite(gamma)
            matrices = np.full((len(times), 3, 3), np.nan)
            matrices[finite] = F @ Rotation.from_euler("XYZ", np.column_stack((alpha[finite], gamma[finite], beta[finite]))).as_matrix()
            orientations[shell] = matrices
            if fused:
                astate = np.asarray(frames["alpha_status"])
                bstate = np.asarray(frames[f"beta_{shell}_status"])
                valid[shell] = finite & (astate != "unresolved") & (bstate != "unresolved")
                status[shell] = np.where(~valid[shell], "unresolved", np.where(
                    (astate == "predicted") | (bstate == "predicted"), "experimental_predicted", "experimental_vision_fused"))
            else:
                valid[shell] = finite & np.asarray(frames[f"valid_{shell}"], dtype=bool)
                status[shell] = np.where(valid[shell], "mechanical_accepted", "unresolved")
        if orientations[shell].shape != (len(times), 3, 3) or valid[shell].shape != times.shape:
            raise ValueError("Pose and timestamp lengths differ")
    return ValidationTrajectory(path, source, metadata, times, orientations, valid, status,
                                geometry, gap, center, radius)


def check_comparable(first, second):
    """The exact same undistorted image measurements must apply to both runs."""
    if Path(first.metadata["input"]).resolve() != Path(second.metadata["input"]).resolve():
        raise ValueError("Comparison requires the same source video")
    if first.times.shape != second.times.shape or not np.allclose(first.times, second.times, atol=1e-9):
        raise ValueError("Comparison requires the same frame timestamps")
    for name in ("K", "dist"):
        a, b = np.asarray(first.metadata[name]), np.asarray(second.metadata[name])
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-10):
            raise ValueError("Comparison requires the same camera/undistorted pixel system")


def default_pairs(times):
    """Fixed time targets, independent of tracking success or observed errors."""
    times = np.asarray(times)
    near = lambda time: int(np.argmin(np.abs(times - time)))
    short_times = (0., 1.2, 2., 2.4, 2.55, 3.15, 4.6, 4.68, 4.95, 8., 12., 15.2, 15.31, 15.55, 18., 19.4)
    pairs = [(min(near(time), len(times)-2), min(near(time), len(times)-2)+1) for time in short_times]
    pairs += [(near(1.952), near(2.032))]
    pairs += [(0, near(time)) for time in (4.68, 8., 15.31, times[-1])]
    return list(dict.fromkeys((a, b) for a, b in pairs if a != b))


def archived_observations(paths, frames):
    """Union of saved fitting/tracking pixels from all result provenance chains."""
    needed = set(frames)
    visited, archives = set(), set()

    def visit(path):
        path = Path(path).resolve()
        if path in visited or not path.exists():
            return
        visited.add(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        archive = path.parent / "offline_observations.npz"
        if archive.exists():
            archives.add(archive.resolve())
        meta = data.get("metadata", {})
        saved = meta.get("source_observations")
        if saved:
            saved = Path(saved)
            saved = saved if saved.is_absolute() else path.parent / saved
            if saved.exists():
                archives.add(saved.resolve())
        for reference in (data.get("source_results"), meta.get("source_results")):
            if reference:
                reference = Path(reference)
                visit(reference if reference.is_absolute() else path.parent / reference)

    for path in paths:
        visit(path)
    values = {frame: [] for frame in needed}
    for path in sorted(archives):
        with np.load(path, allow_pickle=False) as data:
            for shell in ("top", "bottom"):
                indices = data[f"{shell}_frame_index"]
                uv = data[f"{shell}_uv"]
                for frame in needed:
                    pixels = uv[indices == frame]
                    if len(pixels):
                        values[frame].append(pixels)
    values = {frame: np.unique(np.concatenate(parts), axis=0) if parts else np.empty((0, 2))
              for frame, parts in values.items()}
    return values, [str(path) for path in sorted(archives)]


def paint_masks(image, circle, segment_config):
    """Color-only validation regions; no trajectory or projected axis is used."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
    u, v, radius = circle
    region = (xx-u)**2 + (yy-v)**2 <= (1.1*radius)**2
    paper = (hsv[:, :, 1] < 65) & (hsv[:, :, 2] >= 135)
    masks = {}
    for shell in ("top", "bottom"):
        if f"{shell}_hsv" not in segment_config:
            raise ValueError("Validation requires explicit top_hsv/bottom_hsv color ranges")
        limits = segment_config[f"{shell}_hsv"]
        lower, upper = np.asarray(limits["lo"], np.uint8), np.asarray(limits["hi"], np.uint8)
        if lower[0] <= upper[0]:
            color = cv2.inRange(hsv, lower, upper)
        else:
            color = cv2.inRange(hsv, lower, np.array([179, upper[1], upper[2]], np.uint8))
            color |= cv2.inRange(hsv, np.array([0, lower[1], lower[2]], np.uint8), upper)
        color[(hsv[:, :, 1] < 40) | (hsv[:, :, 2] < 45)] = 0
        color[~region] = 0
        # Reject furniture/cables that share paint hues. A physical paint
        # marker is a compact color component adjoining light, low-saturation
        # shell paper. This uses the images only and is frozen for both runs.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(color, connectivity=8)
        markers = np.zeros_like(color)
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if area < 35 or area > .025*radius**2 or max(width, height) > 6*min(width, height):
                continue
            left, right = max(0, x-8), min(image.shape[1], x+width+8)
            top, bottom = max(0, y-8), min(image.shape[0], y+height+8)
            component = labels[top:bottom, left:right] == index
            ring = cv2.dilate(component.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~component
            if not ring.any() or np.mean(paper[top:bottom, left:right][ring]) < .4:
                continue
            markers[top:bottom, left:right][component] = 255
        color = markers
        mask = cv2.dilate(color, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        masks[shell] = mask & region
    overlap = masks["top"] & masks["bottom"]
    masks["top"][overlap] = masks["bottom"][overlap] = False
    return masks


def distance_from_training(points, training):
    if not len(points):
        return np.empty(0)
    return np.full(len(points), np.inf) if not len(training) else cKDTree(training).query(points)[0]


def observe_pair(source, target, source_mask, target_mask, source_training, target_training,
                 *, exclusion_px=30., max_corners=300, fb_limit_px=.75, ncc_min=.75, window=21):
    """Fresh corner/LK/patch checks; no pose prediction or geometry RANSAC."""
    if exclusion_px < 0 or window < 3 or window % 2 != 1:
        raise ValueError("Exclusion must be nonnegative and LK window odd >=3")
    first = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.ndim == 3 else source
    second = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target
    mask = source_mask.astype(np.uint8) * 255
    for uv in source_training:
        cv2.circle(mask, tuple(np.rint(uv).astype(int)), int(np.ceil(exclusion_px)), 0, -1)
    margin = window // 2 + 1
    mask[:margin] = mask[-margin:] = 0
    mask[:, :margin] = mask[:, -margin:] = 0
    detected = cv2.goodFeaturesToTrack(first, maxCorners=max_corners, qualityLevel=.01,
                                       minDistance=8., blockSize=5, mask=mask)
    initial = np.empty((0, 2)) if detected is None else detected.reshape(-1, 2).astype(float)
    initial = initial[distance_from_training(initial, source_training) >= exclusion_px]
    report = {"detected_heldout_corners": len(initial), "forward_backward_pass": 0,
              "target_exclusion_pass": 0, "patch_check_pass": 0, "exclusion_radius_px": float(exclusion_px),
              "lk_window_px": window, "fb_limit_px": fb_limit_px, "ncc_min": ncc_min,
              "observation_coverage": 0.}
    empty = {"source_uv": [], "target_uv": [], "fb_error_px": [], "patch_ncc": [], "diagnostics": report}
    if not len(initial):
        return empty
    options = dict(winSize=(window, window), maxLevel=4,
                   criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, .01))
    destination, forward_ok, _ = cv2.calcOpticalFlowPyrLK(first, second, initial.astype(np.float32), None, **options)
    if destination is None:
        return empty
    finite = np.isfinite(destination).all(axis=1)
    candidates = np.flatnonzero(forward_ok.ravel().astype(bool) & finite)
    if not len(candidates):
        return empty
    backward, back_ok, _ = cv2.calcOpticalFlowPyrLK(second, first, destination[candidates], None, **options)
    if backward is None:
        return empty
    error = np.linalg.norm(backward - initial[candidates], axis=1)
    keep = back_ok.ravel().astype(bool) & np.isfinite(error) & (error <= fb_limit_px)
    candidates, error = candidates[keep], error[keep]
    report["forward_backward_pass"] = len(candidates)
    targets = destination[candidates]
    height, width = second.shape
    xy = np.rint(targets).astype(int)
    interior = ((xy[:, 0] >= margin) & (xy[:, 0] < width-margin)
                & (xy[:, 1] >= margin) & (xy[:, 1] < height-margin))
    allowed = np.zeros(len(xy), bool)
    allowed[interior] = target_mask[xy[interior, 1], xy[interior, 0]]
    allowed &= distance_from_training(targets, target_training) >= exclusion_px
    candidates, targets, error = candidates[allowed], targets[allowed], error[allowed]
    report["target_exclusion_pass"] = len(candidates)
    scores = []
    for index, target_uv in zip(candidates, targets):
        a = cv2.getRectSubPix(first, (window, window), tuple(initial[index].astype(float))).astype(float).ravel()
        b = cv2.getRectSubPix(second, (window, window), tuple(target_uv.astype(float))).astype(float).ravel()
        a -= a.mean(); b -= b.mean()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        scores.append(float(a @ b / denominator) if denominator > 1e-9 else -1.)
    scores = np.asarray(scores)
    keep = scores >= ncc_min
    report["patch_check_pass"] = int(keep.sum())
    report["observation_coverage"] = float(keep.sum() / len(initial))
    return {"source_uv": initial[candidates[keep]].tolist(), "target_uv": targets[keep].tolist(),
            "fb_error_px": error[keep].tolist(), "patch_ncc": scores[keep].tolist(), "diagnostics": report}


def predict_pair(trajectory, first, second, shell, pixels, *, limb_cull_deg=70.):
    points = np.asarray(pixels, dtype=float).reshape(-1, 2)
    prediction = np.full(points.shape, np.nan)
    valid = np.zeros(len(points), bool)
    available = bool(trajectory.valid[shell][first] and trajectory.valid[shell][second])
    if not available or not len(points):
        return prediction, valid, available
    K = np.asarray(trajectory.metadata["K"], dtype=float)
    sign = int(trajectory.metadata.get("top_shell_sign", 1)) * (1 if shell == "top" else -1)
    source_R, target_R = trajectory.orientations[shell][[first, second]]
    normals, geometric, angles = unproject_shell(points, K, trajectory.center, trajectory.radius,
                                                source_R, sign, trajectory.gap, trajectory.geometry)
    material = normals @ source_R
    geometric &= angles <= np.deg2rad(limb_cull_deg)
    geometric &= sign * material[:, 2] >= cap_boundary(trajectory.gap, trajectory.geometry) - 1e-7
    xyz = surface_camera(material, target_R, trajectory.center, trajectory.radius,
                         sign, trajectory.gap, trajectory.geometry)
    target_normals = material @ target_R.T
    geometric &= (xyz[:, 2] > 0) & (np.einsum("ij,ij->i", xyz, target_normals) < 0)
    # The other complete hemisphere can be in front of this surface near the
    # gap. Its spin does not affect the cap, so shared-roll trajectories can
    # check it even when that shell's spin is unresolved.
    other = "bottom" if shell == "top" else "top"
    if trajectory.valid[other][second] or trajectory.source != "unconstrained":
        other_R = trajectory.orientations[other][second] if trajectory.valid[other][second] else target_R
        other_center = shell_centers(other_R, trajectory.center, trajectory.radius,
                                     -sign, trajectory.gap, trajectory.geometry)
        distances = np.linalg.norm(xyz, axis=1)
        rays = xyz / distances[:, None]
        along = rays @ other_center
        discriminant = along**2 - (other_center @ other_center - trajectory.radius**2)
        depth = along - np.sqrt(np.maximum(discriminant, 0.))
        normals_other = (rays * depth[:, None] - other_center) / trajectory.radius
        covered = ((discriminant > 0) & (depth > 0) & (depth < distances - 1e-6)
                   & (-sign * (normals_other @ other_R[:, 2]) >= cap_boundary(trajectory.gap, trajectory.geometry)))
        geometric &= ~covered
    h = xyz @ K.T
    prediction[geometric] = h[geometric, :2] / h[geometric, 2:]
    return prediction, geometric, available


def error_summary(errors, denominator, threshold=2.):
    values = np.asarray(errors, float)
    finite = values[np.isfinite(values)]
    count = len(finite)
    return {"observation_count": int(denominator), "projectable_count": count,
            "coverage": count / denominator if denominator else None,
            "median_px": float(np.median(finite)) if count else None,
            "p95_px": float(np.percentile(finite, 95)) if count else None,
            "rmse_px": float(np.sqrt(np.mean(finite**2))) if count else None,
            "inlier_count": int(np.count_nonzero(finite <= threshold)),
            "inlier_fraction_all_observations": float(np.count_nonzero(finite <= threshold) / denominator) if denominator else None,
            "inlier_fraction_projectable": float(np.mean(finite <= threshold)) if count else None}


def compare_observations(first, second, records, threshold=2.):
    """Score a fixed observation set, including unavailable poses in coverage."""
    check_comparable(first, second)
    reports, all_errors = [], {"baseline": [], "candidate": []}
    common_errors = {"baseline": [], "candidate": []}
    for record in records:
        source, target, shell = record["source_frame"], record["target_frame"], record["shell"]
        observed = np.asarray(record["target_uv"], float).reshape(-1, 2)
        results, errors = {}, {}
        for label, trajectory in (("baseline", first), ("candidate", second)):
            predicted, projectable, available = predict_pair(trajectory, source, target, shell, record["source_uv"])
            values = np.full(len(observed), np.nan)
            values[projectable] = np.linalg.norm(predicted[projectable] - observed[projectable], axis=1)
            errors[label] = values
            all_errors[label].extend(values.tolist())
            results[label] = {"pose_available": available,
                              "source_status": str(trajectory.status[shell][source]),
                              "target_status": str(trajectory.status[shell][target]),
                              **error_summary(values, len(observed), threshold),
                              "predicted_uv": [[float(v) for v in uv] if np.all(np.isfinite(uv)) else None for uv in predicted],
                              "error_px": [float(v) if np.isfinite(v) else None for v in values]}
        common = np.isfinite(errors["baseline"]) & np.isfinite(errors["candidate"])
        paired = {}
        for label in results:
            common_errors[label].extend(errors[label][common].tolist())
            paired[label] = error_summary(errors[label][common], int(common.sum()), threshold)
        reports.append({"source_frame": source, "target_frame": target, "shell": shell,
                        "source_time_s": float(first.times[source]), "target_time_s": float(first.times[target]),
                        "matching": record["diagnostics"], "source_uv": record["source_uv"], "observed_uv": record["target_uv"],
                        **results, "paired_common_support": paired})
    total = sum(len(record["target_uv"]) for record in records)
    return {"threshold_px": threshold, "total_observations": total,
            "image_pair_shell_count": len(reports),
            "pose_available_pair_shell_count": {label: sum(row[label]["pose_available"] for row in reports)
                                                for label in all_errors},
            "aggregate_all_observations": {label: error_summary(values, total, threshold) for label, values in all_errors.items()},
            "aggregate_common_support": {label: error_summary(values, len(values), threshold) for label, values in common_errors.items()},
            "pairs": reports}


def known_angle_errors(trajectory, csv_path, tolerance_s=.03, *, reference_frame=None):
    """External angles in degrees relative to the saved effective initial frame.

    Rows use frame_index or time_s and alpha_deg/beta_top_deg/beta_bottom_deg.
    Full shell comparisons require alpha and that shell's beta. Orientation
    errors are modulo complete revolutions; no turn count is inferred.
    """
    F = np.asarray(trajectory.metadata["R_bc"] if reference_frame is None else reference_frame, float)
    if F.shape != (3, 3) or not np.all(np.isfinite(F)) or not np.allclose(F.T @ F, np.eye(3), atol=1e-7) or np.linalg.det(F) <= 0:
        raise ValueError("Known-angle reference frame must be a proper finite rotation matrix")
    rows, errors = [], {"top": [], "bottom": []}
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        for reference in csv.DictReader(handle):
            if reference.get("frame_index", "").strip():
                index = int(reference["frame_index"])
                matched = 0 <= index < len(trajectory.times)
            elif reference.get("time_s", "").strip():
                time = float(reference["time_s"])
                index = int(np.argmin(np.abs(trajectory.times-time)))
                matched = abs(trajectory.times[index]-time) <= tolerance_s
            else:
                raise ValueError("Known-angle CSV needs frame_index or time_s")
            result = {"reference": reference, "frame_index": index if matched else None, "shells": {}}
            for shell in errors:
                if not matched or not trajectory.valid[shell][index] or not reference.get("alpha_deg") or not reference.get(f"beta_{shell}_deg"):
                    result["shells"][shell] = None
                    continue
                a, b = np.deg2rad([float(reference["alpha_deg"]), float(reference[f"beta_{shell}_deg"])])
                truth = F @ Rx(a) @ Rz(b)
                error = float(np.rad2deg(Rotation.from_matrix(trajectory.orientations[shell][index] @ truth.T).magnitude()))
                errors[shell].append(error)
                result["shells"][shell] = {"orientation_error_deg": error}
            rows.append(result)
    return {"reference_file": str(Path(csv_path).resolve()), "reference_frame_camera": F.tolist(),
            "reference_convention": "Degrees relative to the fixed reference_frame_camera; intrinsic Rx(alpha) Rz(beta).",
            "turn_count_ambiguity": "Orientation comparison is modulo complete turns; hidden revolution counts are not validated.",
            "shells": {shell: {"compared": len(values), "rmse_deg": float(np.sqrt(np.mean(np.square(values)))) if values else None,
                                "max_deg": float(np.max(values)) if values else None} for shell, values in errors.items()}, "rows": rows}
