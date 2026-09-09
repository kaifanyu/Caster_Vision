#!/usr/bin/env python
"""Calibrate OpenCV intrinsics from checkerboard photographs."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from ballrot.config import update_yaml


def _expand_images(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        candidate = Path(pattern).expanduser()
        if candidate.is_dir():
            for suffix in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
                paths.update(path.resolve() for path in candidate.glob(suffix))
        else:
            paths.update(Path(path).resolve() for path in glob.glob(str(candidate)))
    return sorted(paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate camera intrinsics from checkerboard images. Rows/columns "
        "are the INNER-corner counts, not square counts."
    )
    parser.add_argument(
        "--images", nargs="+", required=True, help="Image directories or glob patterns"
    )
    parser.add_argument("--board-cols", type=int, required=True, help="Inner corners across")
    parser.add_argument("--board-rows", type=int, required=True, help="Inner corners down")
    parser.add_argument(
        "--square-size", type=float, default=1.0, help="Square edge length (any unit)"
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument(
        "--preview-dir", type=Path, default=PROJECT_ROOT / "out" / "camera_calibration"
    )
    parser.add_argument("--min-detections", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    images = _expand_images(args.images)
    if not images:
        raise SystemExit("No calibration images matched --images.")
    if args.board_cols < 2 or args.board_rows < 2 or args.square_size <= 0:
        raise SystemExit("Board dimensions must be >=2 and square-size must be positive.")

    object_template = np.zeros((args.board_rows * args.board_cols, 3), np.float32)
    object_template[:, :2] = (
        np.mgrid[0 : args.board_cols, 0 : args.board_rows].T.reshape(-1, 2)
        * args.square_size
    )
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    accepted: list[str] = []
    rejected: list[str] = []
    image_size: tuple[int, int] | None = None
    args.preview_dir.mkdir(parents=True, exist_ok=True)

    for path in images:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            rejected.append(f"{path} (unreadable)")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = size
        if size != image_size:
            rejected.append(f"{path} (resolution {size}, expected {image_size})")
            continue
        found, corners = cv2.findChessboardCornersSB(
            gray,
            (args.board_cols, args.board_rows),
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
        if not found:
            rejected.append(f"{path} (checkerboard not found)")
            continue
        object_points.append(object_template.copy())
        image_points.append(corners.astype(np.float32))
        accepted.append(str(path))
        preview = frame.copy()
        cv2.drawChessboardCorners(
            preview, (args.board_cols, args.board_rows), corners, found
        )
        cv2.imwrite(str(args.preview_dir / path.name), preview)

    if len(accepted) < args.min_detections:
        raise SystemExit(
            f"Only {len(accepted)} usable views; need at least {args.min_detections}. "
            f"Rejected {len(rejected)}. See {args.preview_dir}."
        )
    assert image_size is not None
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    per_view_errors = []
    for object_view, image_view, rvec, tvec in zip(
        object_points, image_points, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(object_view, rvec, tvec, K, dist)
        error = cv2.norm(image_view, projected, cv2.NORM_L2) / np.sqrt(len(projected))
        per_view_errors.append(float(error))

    update_yaml(
        args.config,
        {"camera": {"K": K.tolist(), "dist": dist.reshape(-1).tolist()}},
    )
    report = {
        "rms_reprojection_error_px": float(rms),
        "per_view_error_px": per_view_errors,
        "image_size": list(image_size),
        "K": K.tolist(),
        "dist": dist.reshape(-1).tolist(),
        "accepted": accepted,
        "rejected": rejected,
        "config_written": str(args.config.resolve()),
    }
    report_path = args.preview_dir / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: used {len(accepted)}/{len(images)} checkerboard views")
    print(f"RMS reprojection error: {rms:.4f} px")
    print(f"Updated: {args.config.resolve()}")
    print(f"Report:  {report_path.resolve()}")
    if rms > 1.0:
        print("WARNING: RMS exceeds 1 px; retake sharper, more varied checkerboard views.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

