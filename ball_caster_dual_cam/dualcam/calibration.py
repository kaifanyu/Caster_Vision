"""Checkerboard calibration independent of camera hardware and tracking code.

Camera 1 is C920; camera 2 is Brio 101. All stereo translations are metres.
A plain board has a 180-degree origin ambiguity: stereo calibration resolves it
across tilted poses, and refuses datasets that cannot resolve it reliably.
"""
from __future__ import annotations

import glob
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


class CalibrationError(ValueError):
    """The observations cannot support a calibration that passes the gates."""


def board_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    if cols < 3 or rows < 3 or cols == rows:
        raise CalibrationError("Use a rectangular checkerboard with at least 3 inner corners per axis, e.g. --cols 9 --rows 6.")
    if not np.isfinite(square_m) or square_m <= 0:
        raise CalibrationError("--square-m must be a positive measured square edge length in metres.")
    points = np.zeros((rows * cols, 3), np.float32)
    points[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * square_m
    return points


def image_files(spec: str | Path) -> list[Path]:
    path = Path(spec).expanduser()
    if path.is_dir():
        paths = sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    else:
        paths = [Path(p) for p in sorted(glob.glob(str(path)))]
    if not paths:
        raise CalibrationError(f"No calibration images found: {spec}")
    return paths


def detect_board(path: Path, cols: int, rows: int) -> tuple[tuple[int, int], np.ndarray | None]:
    frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise CalibrationError(f"Cannot read image: {path}")
    size = (int(frame.shape[1]), int(frame.shape[0]))
    # SB is more robust on low-contrast webcam images and already returns subpixel corners.
    found, corners = cv2.findChessboardCornersSB(frame, (cols, rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return size, None
    return size, np.asarray(corners, dtype=np.float32).reshape(-1, 2)


def _validate_size(size) -> tuple[int, int]:
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise CalibrationError("image_size must be [width, height].")
    if any(not isinstance(x, (int, np.integer)) or x <= 0 for x in size):
        raise CalibrationError("image_size must contain positive integer dimensions.")
    return int(size[0]), int(size[1])


def load_intrinsics(path: str | Path) -> dict:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise CalibrationError(f"Cannot load intrinsics {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CalibrationError(f"Intrinsics file must be a mapping: {path}")
    try:
        K = np.asarray(data["K"], dtype=float)
        dist = np.asarray(data["dist"], dtype=float).reshape(-1)
        size = _validate_size(data["image_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(f"Fill K, dist and image_size in {path}: {exc}") from exc
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise CalibrationError(f"Invalid camera matrix K in {path}.")
    if len(dist) not in (4, 5, 8, 12, 14) or not np.isfinite(dist).all():
        raise CalibrationError(f"dist in {path} must contain 4, 5, 8, 12, or 14 finite OpenCV distortion coefficients; use five zeros only for an already-undistorted image model.")
    return {**data, "K": K, "dist": dist, "image_size": size}


def _observations(observations, count: int, min_views: int) -> list[np.ndarray]:
    if min_views < 3:
        raise CalibrationError("min_views must be at least 3; 10 or more are recommended.")
    if len(observations) < min_views:
        raise CalibrationError(f"Only {len(observations)} usable checkerboard views; need at least {min_views}.")
    result = []
    for idx, points in enumerate(observations):
        points = np.asarray(points, np.float32).reshape(-1, 2)
        if points.shape != (count, 2) or not np.isfinite(points).all():
            raise CalibrationError(f"View {idx} must contain {count} finite image corners.")
        result.append(points)
    return result


def _max_tilt_degrees(rotations) -> float:
    normals = np.stack([R[:, 2] for R in rotations])
    return float(np.degrees(np.arccos(np.clip(normals @ normals.T, -1, 1))).max())


def _rms_gate(rms: float, max_rms_px: float, per_view=None):
    if not np.isfinite(max_rms_px) or max_rms_px <= 0:
        raise CalibrationError("max_rms_px must be positive.")
    if not np.isfinite(rms) or rms > max_rms_px:
        raise CalibrationError(f"Reprojection RMS {rms:.3f} px exceeds limit {max_rms_px:.3f} px. Improve board sharpness, corner coverage, synchronization, and calibration before retrying.")
    if per_view is not None and max(per_view) > max_rms_px * 2:
        raise CalibrationError(f"A view has RMS {max(per_view):.3f} px, exceeding the per-view limit {max_rms_px * 2:.3f} px. Inspect the per-view report.")


def fit_intrinsics(observations, image_size, cols, rows, square_m, *, min_views=10, max_rms_px=1.0, min_tilt_degrees=8.0):
    obj = board_points(cols, rows, square_m)
    size = _validate_size(image_size)
    points = _observations(observations, len(obj), min_views)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj] * len(points), points, size, None, None)
    errors = []
    for actual, rvec, tvec in zip(points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        errors.append(float(np.sqrt(np.mean(np.sum((actual - projected.reshape(-1, 2)) ** 2, axis=1)))))
    tilt = _max_tilt_degrees([cv2.Rodrigues(r)[0] for r in rvecs])
    report = {"rms_px": float(rms), "per_view_rms_px": errors, "max_board_tilt_separation_deg": tilt, "views_used": len(points)}
    # Return diagnostics even if a downstream acceptance gate fails.
    result = {"calibration_kind": "intrinsics", "image_size": list(size), "K": K.tolist(), "dist": dist.reshape(-1).tolist(), "rms_px": float(rms), "board": {"cols": cols, "rows": rows, "square_m": square_m}, "views_used": len(points)}
    try:
        _rms_gate(rms, max_rms_px, errors)
        if tilt < min_tilt_degrees:
            raise CalibrationError(f"Board tilt variation is only {tilt:.1f} degrees; use diverse tilts of at least {min_tilt_degrees:.1f} degrees and cover the image.")
        if not np.isfinite(K).all() or not np.isfinite(dist).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise CalibrationError("Intrinsic fit produced invalid camera parameters.")
    except CalibrationError as exc:
        exc.report = report
        raise
    return result, report


def _pose(obj, points, intrinsics):
    okay, r, t = cv2.solvePnP(obj, points, intrinsics["K"], intrinsics["dist"], flags=cv2.SOLVEPNP_ITERATIVE)
    if not okay or not np.isfinite(r).all() or not np.isfinite(t).all():
        raise CalibrationError("Checkerboard pose estimation failed.")
    R = cv2.Rodrigues(r)[0]
    if np.min((R @ obj.T + t)[2]) <= 0:
        raise CalibrationError("Checkerboard pose lies behind a camera.")
    return R, t.reshape(3)


def resolve_corner_orders(left, right, obj, intrinsics1, intrinsics2, *, min_tilt_degrees=8.0):
    """Resolve the 180-degree plain-checkerboard ambiguity using rigid-rig consensus.

    Camera 1 defines a board coordinate frame separately in every pair. Camera 2
    may see that frame's inner corners in the opposite detector order. True
    camera-to-camera transforms agree across poses; false reversals do not when
    board normals vary. No temporal ordering or "first corner is top-left" claim
    is used.
    """
    hypotheses, left_poses = [], []
    for p1, p2 in zip(left, right):
        R1, t1 = _pose(obj, p1, intrinsics1)
        left_poses.append((R1, t1))
        candidates = []
        for ordered in (p2, p2[::-1].copy()):
            R2, t2 = _pose(obj, ordered, intrinsics2)
            R21 = R2 @ R1.T
            candidates.append((R21, t2 - R21 @ t1))
        hypotheses.append(candidates)
    tilt = _max_tilt_degrees([x[0] for x in left_poses])
    if tilt < min_tilt_degrees:
        raise CalibrationError(f"Board tilt variation is only {tilt:.1f} degrees. At least {min_tilt_degrees:.1f} degrees is required to resolve the checkerboard's 180-degree ambiguity; collect varied board tilts.")
    board_span = float(np.linalg.norm(np.ptp(obj[:, :2], axis=0)))
    scale = max(board_span, 1e-3)

    def distance(R, t, candidate):
        Rc, tc = candidate
        angle = Rotation.from_matrix(Rc @ R.T).magnitude()
        return float(angle * angle + np.sum(((tc - t) / scale) ** 2))

    solutions = {}
    for pair in hypotheses:
        for R, t in pair:
            choices = None
            for _ in range(10):
                new_choices = tuple(int(np.argmin([distance(R, t, c) for c in hs])) for hs in hypotheses)
                selected = [hs[c] for hs, c in zip(hypotheses, new_choices)]
                R = Rotation.from_matrix(np.stack([h[0] for h in selected])).mean().as_matrix()
                t = np.mean([h[1] for h in selected], axis=0)
                if choices == new_choices:
                    break
                choices = new_choices
            distances = [distance(R, t, hs[c]) for hs, c in zip(hypotheses, new_choices)]
            solutions[new_choices] = (float(np.mean(distances)), R, t, distances)
    ranked = sorted(solutions.items(), key=lambda x: x[1][0])
    choices, (score, R, t, distances) = ranked[0]
    # The incorrect all-reversed family is close to a consistent rig if the board
    # has insufficient pose diversity. Require substantial evidence over it.
    alternative_score = ranked[1][1][0] if len(ranked) > 1 else None
    if alternative_score is not None and alternative_score < max(0.01, score * 4):
        raise CalibrationError("Checkerboard origin ordering remains ambiguous. Capture sharper images at substantially different board tilts and positions; use a marked board/ChArUco for difficult geometry.")
    angle_errors = [float(np.degrees(Rotation.from_matrix(hs[c][0] @ R.T).magnitude())) for hs, c in zip(hypotheses, choices)]
    translation_errors = [float(np.linalg.norm(hs[c][1] - t)) for hs, c in zip(hypotheses, choices)]
    if max(angle_errors) > 10 or max(translation_errors) > scale * 0.3:
        raise CalibrationError("Per-pair camera transforms are inconsistent. Check paired filenames, stationary board capture, visibility, and intrinsics.")
    return [p[::-1].copy() if reverse else p for p, reverse in zip(right, choices)], {
        "right_corner_order_reversed": [bool(c) for c in choices],
        "ordering_consensus_score": score,
        "ordering_alternative_score": alternative_score,
        "per_pair_pose_rotation_error_deg": angle_errors,
        "per_pair_pose_translation_error_m": translation_errors,
        "max_board_tilt_separation_deg": tilt,
        "ordering_method": "180-degree ordering hypotheses, resolved by rigid-rig consensus across varied board tilts",
    }


def fit_stereo(left_observations, right_observations, intrinsics1, intrinsics2, cols, rows, square_m, *, min_views=10, max_rms_px=1.0, min_tilt_degrees=8.0):
    obj = board_points(cols, rows, square_m)
    left = _observations(left_observations, len(obj), min_views)
    right = _observations(right_observations, len(obj), min_views)
    if len(left) != len(right):
        raise CalibrationError("Left/right observations must contain the same paired views.")
    size1, size2 = _validate_size(intrinsics1["image_size"]), _validate_size(intrinsics2["image_size"])
    right, report = resolve_corner_orders(left, right, obj, intrinsics1, intrinsics2, min_tilt_degrees=min_tilt_degrees)
    # FIX_INTRINSIC means imageSize is not used to initialize intrinsics; different
    # camera dimensions are therefore allowed, each checked against its own file.
    rms, _, _, _, _, R21, t21, _, _ = cv2.stereoCalibrate(
        [obj] * len(left), left, right,
        intrinsics1["K"].copy(), intrinsics1["dist"].copy(),
        intrinsics2["K"].copy(), intrinsics2["dist"].copy(), size1,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9),
    )
    t21 = t21.reshape(3)
    # Diagnostic uses each board pose estimated from camera 1, then projects into
    # camera 2. Unlike the optimizer RMS, this also exposes one-view PnP noise.
    right_errors = []
    for p1, p2 in zip(left, right):
        R1, t1 = _pose(obj, p1, intrinsics1)
        R2, t2 = R21 @ R1, R21 @ t1 + t21
        prediction, _ = cv2.projectPoints(obj, cv2.Rodrigues(R2)[0], t2, intrinsics2["K"], intrinsics2["dist"])
        right_errors.append(float(np.sqrt(np.mean(np.sum((p2 - prediction.reshape(-1, 2)) ** 2, axis=1)))))
    baseline = float(np.linalg.norm(t21))
    report.update({"rms_px": float(rms), "views_used": len(left), "baseline_m": baseline, "per_pair_right_projection_rms_px": right_errors})
    try:
        _rms_gate(rms, max_rms_px)
        # Grossly bad individual pairs must not be hidden by the total RMS.
        if max(right_errors) > max_rms_px * 4:
            raise CalibrationError(f"One paired view has right reprojection RMS {max(right_errors):.3f} px; inspect frame pairing and intrinsics.")
        if not np.isfinite(R21).all() or not np.isfinite(t21).all() or baseline < 1e-4:
            raise CalibrationError("Stereo fit has invalid or effectively zero baseline. Verify that distinct cameras supplied each image folder.")
    except CalibrationError as exc:
        exc.report = report
        raise
    return {
        "calibration_kind": "stereo", "camera1": "c920", "camera2": "brio101",
        "R_21": R21.tolist(), "t_21_m": t21.tolist(),
        "convention": "X_brio101=R_21@X_c920+t_21_m", "rms_px": float(rms), "baseline_m": baseline,
        "board": {"cols": cols, "rows": rows, "square_m": square_m}, "views_used": len(left),
        "image_size_c920": list(size1), "image_size_brio101": list(size2),
    }, report


def read_capture_profile(images_spec, explicit_path=None):
    """Load recorded settings, never silently infer them from the current rig."""
    path = Path(explicit_path) if explicit_path else Path(images_spec) / "capture_profile.yaml"
    if not path.is_file():
        if explicit_path:
            raise CalibrationError(f"Capture profile manifest does not exist: {path}")
        return None
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("capture_profile"), dict):
        raise CalibrationError(f"Expected {{capture_profile: ...}} in {path}.")
    return data


def check_capture_profile(manifest, intrinsics, camera):
    if manifest and intrinsics.get("capture_profile") is not None and manifest["capture_profile"] != intrinsics["capture_profile"]:
        raise CalibrationError(f"{camera}: image recording profile differs from the intrinsic calibration profile. Restore calibrated capture settings or recalibrate intrinsics.")


def write_yaml_atomic(path, data):
    """Only called after all acceptance gates pass; preserve previous good files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_report(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
