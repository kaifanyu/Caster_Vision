#!/usr/bin/env python
"""Non-interactive calibration tools for the swivel-caster camera and clocks.

Run ``python scripts/calibrate_swivel.py --help`` for the available, atomic
configuration updates.  Checkerboard/point files replace fragile GUI clicks,
so every calibration can be repeated and reviewed from its numeric report.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from common.camera import undistort_image
from common.caster_frame import CasterFrame, load_caster_frames
from common.config import (
    camera_matrix,
    distortion_coefficients,
    load_config,
    resolve_from_config,
    update_yaml,
)
from common.io_frames import open_frame_source
from common.kinematics import estimate_clock_offset
from common.rotation import is_rotation_matrix
from swivel.geometry import SwivelGeometry
from swivel.tag import ArucoTagTracker


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_report(path: Path | None, report: Mapping[str, Any]) -> Path | None:
    if path is None:
        return None
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _print_result(title: str, report: Mapping[str, Any], report_path: Path | None) -> None:
    print(f"PASS: {title}")
    for key, value in report.items():
        if key in {"per_view_error_px", "per_point_error_px", "accepted", "rejected"}:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            print(f"  {key}: {value}")
    if report_path is not None:
        print(f"  report: {report_path}")


def _expand_images(patterns: Sequence[str]) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    paths: set[Path] = set()
    for text in patterns:
        candidate = Path(text).expanduser()
        if candidate.is_dir():
            paths.update(
                child.resolve()
                for child in candidate.iterdir()
                if child.is_file() and child.suffix.casefold() in extensions
            )
        elif candidate.is_file():
            paths.add(candidate.resolve())
        else:
            paths.update(
                Path(match).resolve()
                for match in glob.glob(str(candidate), recursive=True)
                if Path(match).suffix.casefold() in extensions
            )
    return sorted(paths)


def calibrate_intrinsics(
    image_paths: Sequence[str | Path],
    board_cols: int,
    board_rows: int,
    *,
    square_size: float = 1.0,
    min_detections: int = 8,
    preview_dir: Path | None = None,
) -> dict[str, Any]:
    """Calibrate a pinhole camera from checkerboard inner corners."""

    columns, rows = int(board_cols), int(board_rows)
    scale = float(square_size)
    if columns < 2 or rows < 2:
        raise ValueError("checkerboard inner-corner counts must both be at least two")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("square_size must be positive and finite")
    if int(min_detections) < 3:
        raise ValueError("min_detections must be at least three")
    paths = [Path(path).expanduser().resolve() for path in image_paths]
    if not paths:
        raise ValueError("no calibration images matched --images")

    template = np.zeros((rows * columns, 3), dtype=np.float32)
    template[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * scale
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    accepted: list[str] = []
    rejected: list[str] = []
    image_size: tuple[int, int] | None = None
    if preview_dir is not None:
        preview_dir = preview_dir.expanduser().resolve()
        preview_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rejected.append(f"{path} (unreadable)")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = size
        if size != image_size:
            rejected.append(f"{path} (resolution {size}, expected {image_size})")
            continue
        if hasattr(cv2, "findChessboardCornersSB"):
            found, corners = cv2.findChessboardCornersSB(
                gray,
                (columns, rows),
                flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
            )
        else:  # pragma: no cover - retained for older OpenCV wheels.
            found, corners = cv2.findChessboardCorners(gray, (columns, rows))
            if found:
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3),
                )
        if not found or corners is None:
            rejected.append(f"{path} (checkerboard not found)")
            continue
        object_points.append(template.copy())
        image_points.append(np.asarray(corners, dtype=np.float32))
        accepted.append(str(path))
        if preview_dir is not None:
            preview = image.copy()
            cv2.drawChessboardCorners(preview, (columns, rows), corners, True)
            cv2.imwrite(str(preview_dir / path.name), preview)

    if len(accepted) < int(min_detections):
        hint = f" Inspect previews in {preview_dir}." if preview_dir is not None else ""
        raise ValueError(
            f"only {len(accepted)} usable checkerboard views; need {int(min_detections)}. "
            f"Rejected {len(rejected)}.{hint} Retake sharp views spanning the image."
        )
    assert image_size is not None
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    per_view: list[float] = []
    for object_view, image_view, rvec, tvec in zip(
        object_points, image_points, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(object_view, rvec, tvec, K, dist)
        residual = projected.reshape(-1, 2) - image_view.reshape(-1, 2)
        per_view.append(float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))))
    return {
        "K": K,
        "dist": np.asarray(dist, dtype=float).reshape(-1),
        "rms_reprojection_error_px": float(rms),
        "per_view_error_px": per_view,
        "image_size": list(image_size),
        "accepted": accepted,
        "rejected": rejected,
    }


def validate_T_car_from_cam(value: Any) -> np.ndarray:
    """Validate a rigid 4x4 camera-to-car point transform."""

    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_car_from_cam must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("T_car_from_cam must end in [0, 0, 0, 1]")
    if not is_rotation_matrix(transform[:3, :3], atol=1e-6):
        raise ValueError("T_car_from_cam contains a reflected/scaled, not proper, rotation")
    return transform.copy()


def _json_argument(value: str, label: str) -> Any:
    text = str(value).strip()
    candidate_text = text[1:] if text.startswith("@") else text
    candidate = Path(candidate_text).expanduser()
    try:
        is_file = candidate.is_file()
    except OSError:
        is_file = False
    if is_file:
        text = candidate.read_text(encoding="utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{label} must be JSON or a path to a JSON file: {exc.msg}"
        ) from exc


def _point_array(value: Any, columns: int, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != columns or not np.all(np.isfinite(points)):
        raise ValueError(f"{label} must be a finite JSON array with shape (N, {columns})")
    return points


def _load_correspondences(path: Path) -> tuple[np.ndarray, np.ndarray]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"correspondence file does not exist: {source}")
    if source.suffix.casefold() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("correspondence CSV contains no data rows")
        lowered = [{str(key).strip().casefold(): value for key, value in row.items()} for row in rows]
        try:
            car = np.array([[float(row[name]) for name in ("x", "y", "z")] for row in lowered])
            pixels = np.array([[float(row[name]) for name in ("u", "v")] for row in lowered])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("correspondence CSV needs numeric x,y,z,u,v columns") from exc
        return car, pixels
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if isinstance(payload, Mapping):
        if "car_points" not in payload or "image_points" not in payload:
            raise ValueError("correspondence JSON needs car_points and image_points")
        return (
            _point_array(payload["car_points"], 3, "car_points"),
            _point_array(payload["image_points"], 2, "image_points"),
        )
    if isinstance(payload, list) and payload and isinstance(payload[0], Mapping):
        try:
            return (
                _point_array([item["car"] for item in payload], 3, "car points"),
                _point_array([item["pixel"] for item in payload], 2, "image points"),
            )
        except KeyError as exc:
            raise ValueError("correspondence records need car and pixel fields") from exc
    raise ValueError("unsupported correspondence JSON; use a mapping or record list")


def solve_extrinsics(
    car_points: Any,
    image_points: Any,
    K: Any,
    dist: Any = None,
) -> dict[str, Any]:
    """Solve car-to-camera pose and return its camera-to-car inverse."""

    object_points = _point_array(car_points, 3, "car_points")
    pixels = _point_array(image_points, 2, "image_points")
    if len(object_points) != len(pixels):
        raise ValueError("car_points and image_points must have the same length")
    if len(object_points) < 4:
        raise ValueError("at least four 3D-to-2D correspondences are required")
    matrix = np.asarray(K, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("K must be a finite 3x3 matrix")
    coefficients = None if dist is None else np.asarray(dist, dtype=float).reshape(-1)
    flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_EPNP)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            pixels,
            matrix,
            coefficients,
            flags=flag,
        )
    except cv2.error as exc:
        raise ValueError(
            "solvePnP failed; ensure points are accurate, span the image, and are not collinear"
        ) from exc
    if not ok:
        raise ValueError("solvePnP returned no pose; check point order and units")
    if hasattr(cv2, "solvePnPRefineLM"):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points, pixels, matrix, coefficients, rvec, tvec
            )
        except cv2.error:
            pass
    R_camera_from_car, _ = cv2.Rodrigues(rvec)
    t_camera_from_car = np.asarray(tvec, dtype=float).reshape(3)
    R_car_from_camera = R_camera_from_car.T
    t_car_from_camera = -R_car_from_camera @ t_camera_from_car
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = R_car_from_camera
    transform[:3, 3] = t_car_from_camera
    transform = validate_T_car_from_cam(transform)
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, matrix, coefficients
    )
    residual = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
    positive_depth = (
        object_points @ R_camera_from_car.T + t_camera_from_car
    )[:, 2] > 0.0
    if not np.all(positive_depth):
        raise ValueError("solved pose puts one or more calibration points behind the camera")
    return {
        "T_car_from_cam": transform,
        "T_cam_from_car": np.linalg.inv(transform),
        "rms_reprojection_error_px": float(np.sqrt(np.mean(residual * residual))),
        "max_reprojection_error_px": float(np.max(residual)),
        "per_point_error_px": residual,
        "point_count": len(object_points),
    }


def roll_sign_from_delta(
    delta_phi: float,
    *,
    axle0_car: Any = (0.0, -1.0, 0.0),
    heading_direction_sign: float = 1.0,
    minimum_motion_rad: float = 1e-3,
) -> float:
    """Choose the configured sign that makes a forward run point car +x."""

    delta = float(delta_phi)
    if not np.isfinite(delta) or abs(delta) < float(minimum_motion_rad):
        raise ValueError(
            "forward calibration contains too little wheel rotation; use a longer straight run"
        )
    axle = np.asarray(axle0_car, dtype=float).reshape(3)
    axle[2] = 0.0
    norm = float(np.linalg.norm(axle))
    if not np.isfinite(norm) or norm <= 1e-9:
        raise ValueError("axle0_car must have a nonzero horizontal component")
    axle /= norm
    direction_sign = float(heading_direction_sign)
    if direction_sign not in (-1.0, 1.0):
        raise ValueError("heading_direction_sign must be +1 or -1")
    base_forward = float(np.cross([0.0, 0.0, 1.0], direction_sign * axle)[0])
    if abs(base_forward) < 0.5:
        raise ValueError(
            "psi-zero/axle calibration is not approximately car-forward; calibrate psi-zero first"
        )
    return float(np.sign(delta * base_forward))


def roll_sign_from_frames(
    frames: Sequence[CasterFrame],
    *,
    axle0_car: Any,
    heading_direction_sign: float,
    current_roll_sign: float,
    minimum_motion_rad_s: float = 1e-3,
) -> tuple[float, dict[str, Any]]:
    """Infer sign from a known-forward common-record clip."""

    raw_rates = np.array(
        [
            float(frame.raw.get("phi_dot", np.nan))
            for frame in frames
            if frame.roll_valid
        ],
        dtype=float,
    )
    raw_rates = raw_rates[np.isfinite(raw_rates) & (np.abs(raw_rates) >= minimum_motion_rad_s)]
    if len(raw_rates):
        median = float(np.median(raw_rates))
        sign = roll_sign_from_delta(
            median,
            axle0_car=axle0_car,
            heading_direction_sign=heading_direction_sign,
            minimum_motion_rad=minimum_motion_rad_s,
        )
        consistency = float(np.mean(np.sign(raw_rates) == np.sign(median)))
        source = "raw.phi_dot"
        sample_count = len(raw_rates)
    else:
        headings = []
        for frame in frames:
            if frame.roll_valid and frame.omega_roll >= minimum_motion_rad_s:
                headings.append(float(np.cross([0.0, 0.0, 1.0], frame.roll_axis_car)[0]))
        headings_array = np.asarray(headings, dtype=float)
        headings_array = headings_array[np.isfinite(headings_array)]
        if not len(headings_array):
            raise ValueError("CasterFrame JSON contains no usable phi_dot or rolling intervals")
        median_heading = float(np.median(headings_array))
        if abs(median_heading) < 0.5:
            raise ValueError("known-forward CasterFrames do not point approximately along car x")
        correction = float(np.sign(median_heading))
        sign = float(current_roll_sign) * correction
        consistency = float(np.mean(np.sign(headings_array) == np.sign(median_heading)))
        median = median_heading
        source = "CasterFrame rolling heading"
        sample_count = len(headings_array)
    if consistency < 0.6:
        raise ValueError(
            f"forward-run direction is inconsistent ({consistency:.1%}); trim turns/stops from the clip"
        )
    return sign, {
        "source": source,
        "sample_count": int(sample_count),
        "median_signal": median,
        "direction_consistency": consistency,
    }


def effective_radius(distance_m: float, delta_phi_rad: float) -> float:
    """Return loaded rolling radius from known distance / wheel angle."""

    distance = abs(float(distance_m))
    angle = abs(float(delta_phi_rad))
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("known distance must be positive and finite")
    if not np.isfinite(angle) or angle <= 1e-6:
        raise ValueError("wheel angle/revolution count must be nonzero")
    return distance / angle


def event_clock_offset(caster_event_s: float, car_event_s: float) -> float:
    """Offset added to raw caster timestamps: car event minus caster event."""

    caster = float(caster_event_s)
    car = float(car_event_s)
    if not np.isfinite(caster) or not np.isfinite(car):
        raise ValueError("event timestamps must be finite")
    return car - caster


def load_signal_csv(
    path: str | Path,
    *,
    time_column: str = "t",
    signal_column: str = "signal",
) -> tuple[np.ndarray, np.ndarray]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"signal CSV does not exist: {source}")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 3:
        raise ValueError(f"signal CSV needs at least three data rows: {source}")
    requested_time = time_column.strip().casefold()
    requested_signal = signal_column.strip().casefold()
    values: list[tuple[float, float]] = []
    for number, row in enumerate(rows, start=2):
        lowered = {str(key).strip().casefold(): value for key, value in row.items()}
        if requested_time not in lowered or requested_signal not in lowered:
            raise ValueError(
                f"{source} needs columns {time_column!r} and {signal_column!r}"
            )
        try:
            values.append(
                (float(lowered[requested_time]), float(lowered[requested_signal]))
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric signal value at {source}:{number}") from exc
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(np.diff(array[:, 0]) <= 0.0):
        raise ValueError(f"{source} timestamps must be finite and strictly increasing")
    return array[:, 0], array[:, 1]


def _handle_intrinsics(args: argparse.Namespace, config: Mapping[str, Any], config_path: Path) -> int:
    paths = _expand_images(args.images)
    result = calibrate_intrinsics(
        paths,
        args.board_cols,
        args.board_rows,
        square_size=args.square_size,
        min_detections=args.min_detections,
        preview_dir=args.preview_dir,
    )
    update_yaml(
        config_path,
        {"camera": {"K": result["K"].tolist(), "dist": result["dist"].tolist()}},
    )
    report = {
        **result,
        "config_written": str(config_path),
        "warning": (
            "RMS exceeds 1 px; retake sharper, more varied checkerboard views."
            if result["rms_reprojection_error_px"] > 1.0
            else None
        ),
    }
    report_path = _write_report(args.report, report)
    _print_result("camera intrinsics calibrated", report, report_path)
    return 0


def _handle_extrinsics(args: argparse.Namespace, config: Mapping[str, Any], config_path: Path) -> int:
    if args.T_car_from_cam is not None:
        if args.correspondences is not None or args.car_points is not None or args.image_points is not None:
            raise ValueError("--T-car-from-cam cannot be combined with point correspondences")
        transform = validate_T_car_from_cam(
            _json_argument(args.T_car_from_cam, "--T-car-from-cam")
        )
        report: dict[str, Any] = {
            "method": "manual",
            "T_car_from_cam": transform,
            "T_cam_from_car": np.linalg.inv(transform),
            "reprojection_available": False,
        }
    else:
        camera = config.get("camera", {})
        K_value = camera.get("K") if isinstance(camera, Mapping) else None
        if K_value is None:
            raise ValueError("camera.K is null; run the intrinsics subcommand first")
        K = np.asarray(K_value, dtype=float)
        dist = distortion_coefficients(camera)
        if args.correspondences is not None:
            if args.car_points is not None or args.image_points is not None:
                raise ValueError("use --correspondences or the two point-array options, not both")
            car, pixels = _load_correspondences(args.correspondences)
        else:
            if args.car_points is None or args.image_points is None:
                raise ValueError(
                    "provide --correspondences, both --car-points/--image-points, or --T-car-from-cam"
                )
            car = _point_array(_json_argument(args.car_points, "--car-points"), 3, "car_points")
            pixels = _point_array(
                _json_argument(args.image_points, "--image-points"), 2, "image_points"
            )
        report = {"method": "solvePnP", **solve_extrinsics(car, pixels, K, dist)}
        transform = report["T_car_from_cam"]
        if report["rms_reprojection_error_px"] > args.max_rms_px and not args.force:
            raise ValueError(
                f"extrinsic RMS is {report['rms_reprojection_error_px']:.3f}px, above "
                f"--max-rms-px {args.max_rms_px:.3f}; fix correspondences or pass --force"
            )
    update_yaml(config_path, {"camera": {"T_car_from_cam": np.asarray(transform).tolist()}})
    report["config_written"] = str(config_path)
    report_path = _write_report(args.report, report)
    _print_result("camera extrinsics calibrated", report, report_path)
    return 0


def _handle_geometry(args: argparse.Namespace, config: Mapping[str, Any], config_path: Path) -> int:
    """Store a measured/CAD caster geometry after strict unit validation."""

    swivel_point = np.asarray(args.swivel_axis_car, dtype=float)
    hub_offset = np.asarray(args.hub_offset0_car, dtype=float)
    axle = np.asarray(args.axle0_car, dtype=float)
    radius = float(args.wheel_radius_m)
    width = float(args.wheel_width_m)
    if not all(
        vector.shape == (3,) and np.all(np.isfinite(vector))
        for vector in (swivel_point, hub_offset, axle)
    ):
        raise ValueError("all geometry vectors must contain three finite values")
    axle_norm = float(np.linalg.norm(axle))
    if axle_norm <= 1e-9:
        raise ValueError("--axle0-car must be nonzero")
    axle /= axle_norm
    if np.linalg.norm(np.cross([0.0, 0.0, 1.0], axle)) < 1e-6:
        raise ValueError("--axle0-car cannot be parallel to car +z")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("--wheel-radius-m must be positive and finite")
    if not np.isfinite(width) or width < 0.0:
        raise ValueError("--wheel-width-m must be non-negative and finite")

    geometry = {
        "calibrated": True,
        "swivel_axis_car": swivel_point.tolist(),
        "hub_offset0_car": hub_offset.tolist(),
        "axle0_car": axle.tolist(),
        "wheel_radius_m": radius,
        "wheel_width_m": width,
    }
    update_yaml(config_path, {"swivel": {"geometry": geometry}})
    heading = np.cross([0.0, 0.0, 1.0], axle)
    heading[2] = 0.0
    heading /= np.linalg.norm(heading)
    report = {
        "method": "manual measurement or CAD",
        **geometry,
        "heading0_car": heading,
        "signed_trail_m": float(-hub_offset[:2] @ heading[:2]),
        "config_written": str(config_path),
    }
    report_path = _write_report(args.report, report)
    _print_result("caster geometry recorded", report, report_path)
    return 0


def _handle_psi_zero(args: argparse.Namespace, config: Mapping[str, Any], config_path: Path) -> int:
    input_config = config.get("input", {})
    clip = (
        args.clip.expanduser().resolve()
        if args.clip is not None
        else resolve_from_config(config_path, input_config.get("swivel_clip", ""))
    )
    source = open_frame_source(
        clip,
        input_type=args.input_type,
        max_frames=args.max_frames,
        image_fps=args.fps,
    )
    fps = float(args.fps or source.fps or input_config.get("fps_override") or 0.0)
    if fps <= 0.0:
        raise ValueError("image sequences need --fps so frames have timestamps")
    camera_config = config.get("camera", {})
    K, approximate = camera_matrix(camera_config, (source.size[1], source.size[0], 3))
    if approximate:
        raise ValueError("camera.K is null; calibrate intrinsics before psi-zero")
    dist = distortion_coefficients(camera_config)
    geometry = SwivelGeometry.from_mapping(config)
    swivel = config.get("swivel", {})
    marker = swivel.get("marker", {})
    tracker = ArucoTagTracker(
        K,
        float(args.marker_size or marker.get("size_m", 0.04)),
        marker_id=int(marker.get("id", 0) if args.marker_id is None else args.marker_id),
        dictionary=str(args.dictionary or marker.get("dictionary", "DICT_4X4_50")),
        dist=np.zeros_like(dist),
        R_car_from_camera=geometry.R_car_from_camera,
        R_tag_to_car_zero=np.eye(3),
        expected_normal_camera=geometry.car_vectors_to_camera([0.0, 0.0, 1.0]),
        max_reprojection_error_px=float(
            marker.get("max_reprojection_error_px", 2.0)
        ),
    )
    selected = 0
    observations = []
    for record in source:
        timestamp = record.index / fps
        if args.start_s is not None and timestamp < args.start_s:
            continue
        if args.end_s is not None and timestamp > args.end_s:
            continue
        selected += 1
        image = undistort_image(record.image, K, dist)
        observation = tracker.track(image)
        if observation.valid:
            observations.append(observation)
    if selected == 0:
        raise ValueError("the requested --start-s/--end-s interval contains no frames")
    coverage = len(observations) / selected
    if len(observations) < args.min_detections or coverage < args.min_coverage:
        raise ValueError(
            f"tag detected in {len(observations)}/{selected} frames ({coverage:.1%}); need "
            f"at least {args.min_detections} detections and {args.min_coverage:.1%} coverage. "
            "Improve lighting/marker size or correct intrinsics/extrinsics."
        )
    yaws = np.asarray([item.psi for item in observations], dtype=float)
    center = float(np.median(yaws))
    deviations = np.abs((yaws - center + np.pi) % (2.0 * np.pi) - np.pi)
    spread_deg = float(np.rad2deg(np.median(deviations)))
    if spread_deg > args.max_spread_deg and not args.force:
        raise ValueError(
            f"tag yaw moved during the zero clip (median deviation {spread_deg:.3f}deg); "
            "park the wheel straight and still, trim the interval, or pass --force"
        )
    psi0 = float((center + np.pi) % (2.0 * np.pi) - np.pi)
    errors = np.asarray([item.reprojection_error_px for item in observations])
    update_yaml(config_path, {"swivel": {"marker": {"psi0_rad": psi0}}})
    report = {
        "psi0_rad": psi0,
        "psi0_deg": float(np.rad2deg(psi0)),
        "selected_frames": selected,
        "valid_detections": len(observations),
        "coverage": coverage,
        "median_yaw_deviation_deg": spread_deg,
        "median_reprojection_error_px": float(np.median(errors)),
        "max_reprojection_error_px": float(np.max(errors)),
        "config_written": str(config_path),
    }
    report_path = _write_report(args.report, report)
    _print_result("swivel zero calibrated", report, report_path)
    return 0


def _handle_roll_sign(args: argparse.Namespace, config: Mapping[str, Any], config_path: Path) -> int:
    swivel = config.get("swivel", {})
    geometry = swivel.get("geometry", {})
    axle = geometry.get("axle0_car", [0.0, -1.0, 0.0])
    heading_sign = float(swivel.get("heading_direction_sign", 1.0))
    current_sign = float(swivel.get("roll_direction_sign", 1.0))
    if args.caster_frames is not None:
        if args.phi_start is not None or args.phi_end is not None:
            raise ValueError("use --caster-frames or phi endpoints, not both")
        frames, metadata = load_caster_frames(args.caster_frames)
        sign, detail = roll_sign_from_frames(
            frames,
            axle0_car=axle,
            heading_direction_sign=heading_sign,
            current_roll_sign=current_sign,
            minimum_motion_rad_s=args.minimum_motion,
        )
        report: dict[str, Any] = {
            "method": "known-forward CasterFrame JSON",
            "input": str(args.caster_frames.resolve()),
            "input_metadata": metadata,
            **detail,
        }
    else:
        if args.phi_start is None or args.phi_end is None:
            raise ValueError("provide --caster-frames or both --phi-start/--phi-end")
        delta = float(args.phi_end - args.phi_start)
        sign = roll_sign_from_delta(
            delta,
            axle0_car=axle,
            heading_direction_sign=heading_sign,
            minimum_motion_rad=args.minimum_motion,
        )
        report = {"method": "known-forward phi endpoints", "delta_phi_rad": delta}
    update_yaml(
        config_path,
        {
            "swivel": {
                "roll_direction_sign": sign,
                "direction_sign_calibrated": True,
            }
        },
    )
    report.update(
        {
            "roll_direction_sign": sign,
            "direction_sign_calibrated": True,
            "config_written": str(config_path),
        }
    )
    report_path = _write_report(args.report, report)
    _print_result("roll direction calibrated", report, report_path)
    return 0


def _handle_radius(args: argparse.Namespace, config: Mapping[str, Any], config_path: Path) -> int:
    if args.revolutions is not None:
        if args.phi_start is not None or args.phi_end is not None:
            raise ValueError("use --revolutions or phi endpoints, not both")
        revolutions = abs(float(args.revolutions))
        if not np.isfinite(revolutions) or revolutions <= 0.0:
            raise ValueError("--revolutions must be positive and finite")
        delta_phi = 2.0 * np.pi * revolutions
        method = "known distance and revolution count"
    else:
        if args.phi_start is None or args.phi_end is None:
            raise ValueError("provide --revolutions or both --phi-start/--phi-end")
        delta_phi = abs(float(args.phi_end - args.phi_start))
        revolutions = delta_phi / (2.0 * np.pi)
        method = "known distance and phi endpoints"
    radius = effective_radius(args.distance_m, delta_phi)
    update_yaml(config_path, {"swivel": {"r_eff_m": radius}})
    report = {
        "method": method,
        "known_distance_m": abs(float(args.distance_m)),
        "delta_phi_rad": delta_phi,
        "revolutions": revolutions,
        "r_eff_m": radius,
        "config_written": str(config_path),
    }
    report_path = _write_report(args.report, report)
    _print_result("loaded effective radius calibrated", report, report_path)
    return 0


def _sync_update(offset: float, method: str, extra: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "assumptions": {"car_and_caster_clocks_synchronized": True},
        "car_track": {"time_offset_s": float(offset)},
        "sync": {"method": method, **dict(extra)},
    }


def _handle_sync_event(args: argparse.Namespace, config: Mapping[str, Any], config_path: Path) -> int:
    offset = event_clock_offset(args.caster_event_s, args.car_event_s)
    update_yaml(
        config_path,
        _sync_update(
            offset,
            "common_event",
            {
                "caster_event_time_s": float(args.caster_event_s),
                "car_event_time_s": float(args.car_event_s),
            },
        ),
    )
    report = {
        "method": "common event",
        "caster_event_time_s": float(args.caster_event_s),
        "car_event_time_s": float(args.car_event_s),
        "offset_s_added_to_caster_timestamps": offset,
        "config_written": str(config_path),
    }
    report_path = _write_report(args.report, report)
    _print_result("clock event synchronized", report, report_path)
    return 0


def _handle_sync_signals(args: argparse.Namespace, config: Mapping[str, Any], config_path: Path) -> int:
    caster_t, caster_signal = load_signal_csv(
        args.caster_csv,
        time_column=args.caster_time_column,
        signal_column=args.caster_signal_column,
    )
    car_t, car_signal = load_signal_csv(
        args.car_csv,
        time_column=args.car_time_column,
        signal_column=args.car_signal_column,
    )
    estimate = estimate_clock_offset(
        caster_t,
        caster_signal,
        car_t,
        car_signal,
        max_offset_s=args.max_offset_s,
    )
    if (
        estimate.peak_correlation < args.min_correlation
        or estimate.ambiguity_ratio < args.min_ambiguity
    ) and not args.force:
        raise ValueError(
            f"weak/ambiguous synchronization (correlation={estimate.peak_correlation:.3f}, "
            f"ambiguity={estimate.ambiguity_ratio:.3f}); use a sharper common event, "
            "increase recording overlap, or inspect and pass --force"
        )
    update_yaml(
        config_path,
        _sync_update(
            estimate.offset_s,
            "cross_correlation",
            {
                "caster_event_time_s": None,
                "car_event_time_s": None,
                "last_peak_correlation": estimate.peak_correlation,
                "last_ambiguity_ratio": estimate.ambiguity_ratio,
                "last_overlap_s": estimate.overlap_s,
            },
        ),
    )
    report = {
        "method": "cross correlation of a common signal",
        "offset_s_added_to_caster_timestamps": estimate.offset_s,
        "peak_correlation": estimate.peak_correlation,
        "ambiguity_ratio": estimate.ambiguity_ratio,
        "overlap_s": estimate.overlap_s,
        "caster_csv": str(args.caster_csv.resolve()),
        "car_csv": str(args.car_csv.resolve()),
        "config_written": str(config_path),
    }
    report_path = _write_report(args.report, report)
    _print_result("clock signals synchronized", report, report_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="YAML file updated atomically (place before the subcommand)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    intrinsics = commands.add_parser("intrinsics", help="checkerboard camera intrinsics")
    intrinsics.add_argument("--images", nargs="+", required=True, help="files, directories, or glob patterns")
    intrinsics.add_argument("--board-cols", type=int, required=True, help="inner corners across")
    intrinsics.add_argument("--board-rows", type=int, required=True, help="inner corners down")
    intrinsics.add_argument("--square-size", type=float, default=1.0, help="square edge, in any consistent unit")
    intrinsics.add_argument("--min-detections", type=int, default=8)
    intrinsics.add_argument("--preview-dir", type=Path)
    intrinsics.add_argument("--report", type=Path)
    intrinsics.set_defaults(handler=_handle_intrinsics)

    extrinsics = commands.add_parser("extrinsics", help="camera pose in the car frame")
    extrinsics.add_argument("--correspondences", type=Path, help="JSON or CSV with car 3D and image 2D points")
    extrinsics.add_argument("--car-points", help="JSON array or JSON file path")
    extrinsics.add_argument("--image-points", help="JSON array or JSON file path")
    extrinsics.add_argument(
        "--T-car-from-cam",
        "--t-car-from-cam",
        dest="T_car_from_cam",
        help="manually measured 4x4 JSON matrix or JSON file path",
    )
    extrinsics.add_argument("--max-rms-px", type=float, default=2.0)
    extrinsics.add_argument("--force", action="store_true")
    extrinsics.add_argument("--report", type=Path)
    extrinsics.set_defaults(handler=_handle_extrinsics)

    geometry = commands.add_parser(
        "geometry", help="measured/CAD swivel point, hub offset, axle, and wheel size"
    )
    geometry.add_argument(
        "--swivel-axis-car",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="swivel-axis reference point S in car coordinates, metres",
    )
    geometry.add_argument(
        "--hub-offset0-car",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="vector S-to-hub at psi=0 in car coordinates, metres",
    )
    geometry.add_argument(
        "--axle0-car",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="signed axle direction at psi=0; normalized before writing",
    )
    geometry.add_argument("--wheel-radius-m", type=float, required=True)
    geometry.add_argument("--wheel-width-m", type=float, required=True)
    geometry.add_argument("--report", type=Path)
    geometry.set_defaults(handler=_handle_geometry)

    psi_zero = commands.add_parser("psi-zero", help="parked-forward fork-tag yaw offset")
    psi_zero.add_argument("--clip", type=Path, help="defaults to input.swivel_clip")
    psi_zero.add_argument("--input-type", choices=("auto", "video", "images"), default="auto")
    psi_zero.add_argument("--fps", type=float, help="required for timestamped image sequences")
    psi_zero.add_argument("--max-frames", type=int)
    psi_zero.add_argument("--start-s", type=float)
    psi_zero.add_argument("--end-s", type=float)
    psi_zero.add_argument("--marker-id", type=int)
    psi_zero.add_argument("--marker-size", type=float)
    psi_zero.add_argument("--dictionary")
    psi_zero.add_argument("--min-detections", type=int, default=5)
    psi_zero.add_argument("--min-coverage", type=float, default=0.70)
    psi_zero.add_argument("--max-spread-deg", type=float, default=2.0)
    psi_zero.add_argument("--force", action="store_true")
    psi_zero.add_argument("--report", type=Path)
    psi_zero.set_defaults(handler=_handle_psi_zero)

    roll_sign = commands.add_parser("roll-sign", help="known-forward roll direction")
    roll_sign.add_argument("--caster-frames", type=Path)
    roll_sign.add_argument("--phi-start", type=float)
    roll_sign.add_argument("--phi-end", type=float)
    roll_sign.add_argument("--minimum-motion", type=float, default=1e-3)
    roll_sign.add_argument("--report", type=Path)
    roll_sign.set_defaults(handler=_handle_roll_sign)

    radius = commands.add_parser("radius", help="loaded effective wheel radius")
    radius.add_argument("--distance-m", "--known-distance-m", dest="distance_m", type=float, required=True)
    radius.add_argument("--revolutions", type=float)
    radius.add_argument("--phi-start", type=float)
    radius.add_argument("--phi-end", type=float)
    radius.add_argument("--report", type=Path)
    radius.set_defaults(handler=_handle_radius)

    sync_event = commands.add_parser("sync-event", help="align one common timestamped event")
    sync_event.add_argument("--caster-event-s", type=float, required=True)
    sync_event.add_argument("--car-event-s", type=float, required=True)
    sync_event.add_argument("--report", type=Path)
    sync_event.set_defaults(handler=_handle_sync_event)

    sync_signals = commands.add_parser("sync-signals", help="cross-correlate common signals from CSV")
    sync_signals.add_argument("--caster-csv", type=Path, required=True)
    sync_signals.add_argument("--car-csv", type=Path, required=True)
    sync_signals.add_argument("--caster-time-column", default="t")
    sync_signals.add_argument("--caster-signal-column", default="signal")
    sync_signals.add_argument("--car-time-column", default="t")
    sync_signals.add_argument("--car-signal-column", default="signal")
    sync_signals.add_argument("--max-offset-s", type=float, default=1.0)
    sync_signals.add_argument("--min-correlation", type=float, default=0.30)
    sync_signals.add_argument("--min-ambiguity", type=float, default=1.05)
    sync_signals.add_argument("--force", action="store_true")
    sync_signals.add_argument("--report", type=Path)
    sync_signals.set_defaults(handler=_handle_sync_signals)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config, config_path = load_config(args.config)
        report = getattr(args, "report", None)
        if report is not None and report.expanduser().resolve() == config_path:
            raise ValueError("--report must not overwrite the calibration config")
        return int(args.handler(args, config, config_path))
    except (FileNotFoundError, KeyError, TypeError, ValueError, OSError, cv2.error) as exc:
        parser.error(f"{args.command}: {exc}")
    return 2  # argparse.error raises; this is only for static type checkers.


if __name__ == "__main__":
    raise SystemExit(main())
