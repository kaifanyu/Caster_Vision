#!/usr/bin/env python3
"""Fit an individual camera from sharp, varied checkerboard photos."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import yaml
from dualcam.calibration import (
    CalibrationError, board_points, detect_board, fit_intrinsics, image_files,
    read_capture_profile, write_report, write_yaml_atomic,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Directory or quoted image glob; capture_profile.yaml in a directory is copied automatically")
    parser.add_argument("--cols", type=int, required=True, help="Number of INNER checkerboard corners along the longer dimension (e.g. 9)")
    parser.add_argument("--rows", type=int, required=True, help="Number of INNER checkerboard corners along the other dimension (e.g. 6)")
    parser.add_argument("--square-m", type=float, required=True, help="Measured printed square edge in metres (e.g. 0.025)")
    parser.add_argument("--output", required=True, help="Accepted intrinsics YAML path")
    parser.add_argument("--report", help="Diagnostics JSON path; default: output with .report.json suffix")
    parser.add_argument("--min-views", type=int, default=10)
    parser.add_argument("--max-rms-px", type=float, default=1.0)
    parser.add_argument("--capture-profile", help="Recorded YAML manifest containing capture_profile; do not substitute settings from a different recording")
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser()
    report_path = Path(args.report).expanduser() if args.report else output.with_suffix(".report.json")
    report = {"status": "failed", "kind": "intrinsics", "created_utc": datetime.now(timezone.utc).isoformat(), "opencv_version": cv2.__version__, "images_used": [], "images_without_board": [], "warnings": []}
    try:
        board_points(args.cols, args.rows, args.square_m)
        paths = image_files(args.images)
        manifest = read_capture_profile(args.images, args.capture_profile)
        points, image_size = [], None
        for path in paths:
            size, corners = detect_board(path, args.cols, args.rows)
            if image_size is not None and size != image_size:
                raise CalibrationError(f"Mixed image dimensions: {path} is {size}, expected {image_size}.")
            image_size = size
            if corners is None:
                report["images_without_board"].append(str(path))
            else:
                points.append(corners)
                report["images_used"].append(str(path))
        result, diagnostics = fit_intrinsics(points, image_size, args.cols, args.rows, args.square_m, min_views=args.min_views, max_rms_px=args.max_rms_px)
        report.update(diagnostics)
        result["capture_profile"] = manifest["capture_profile"] if manifest else None
        if manifest and manifest.get("device") is not None:
            result["device"] = manifest["device"]
        if manifest:
            profile = manifest["capture_profile"]
            if profile.get("width", image_size[0]) != image_size[0] or profile.get("height", image_size[1]) != image_size[1]:
                raise CalibrationError("Recorded capture profile dimensions differ from calibration images.")
        else:
            report["warnings"].append("No recorded capture_profile manifest. Resolution is verified, but matching focus, zoom, and other camera settings must be verified by the operator.")
        result["created_utc"] = report["created_utc"]
        report["status"] = "passed"
        write_report(report_path, report)
        write_yaml_atomic(output, result)
    except (CalibrationError, OSError, ValueError, cv2.error, yaml.YAMLError) as exc:
        report.update(getattr(exc, "report", {}))
        report.update(status="failed", error=str(exc))
        write_report(report_path, report)
        print(f"Calibration rejected: {exc}\nReport: {report_path}\nExisting calibration was not replaced.", file=sys.stderr)
        return 2
    print(f"Accepted {len(points)} views: RMS {result['rms_px']:.3f} px.\nIntrinsics: {output}\nReport: {report_path}")
    for warning in report["warnings"]:
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
