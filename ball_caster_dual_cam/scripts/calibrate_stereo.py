#!/usr/bin/env python3
"""Calibrate a fixed C920/Brio rig from paired stationary checkerboard photos."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import yaml
from dualcam.calibration import (
    CalibrationError, board_points, check_capture_profile, detect_board, fit_stereo,
    image_files, load_intrinsics, read_capture_profile, write_report, write_yaml_atomic,
)


def indexed_images(spec):
    result = {}
    for path in image_files(spec):
        if path.stem in result:
            raise CalibrationError(f"Duplicate image stem {path.stem!r}; use unique paired filenames.")
        result[path.stem] = path
    return result


def resolve_path(config_path, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else config_path.parent / path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/rig.yaml")
    parser.add_argument("--left-images", required=True, help="C920 directory or quoted glob; corresponding images must have identical filename stems")
    parser.add_argument("--right-images", required=True, help="Brio101 directory or quoted glob, paired by identical filename stems")
    parser.add_argument("--cols", type=int, required=True, help="Checkerboard INNER corner count (e.g. 9)")
    parser.add_argument("--rows", type=int, required=True, help="Checkerboard INNER corner count (e.g. 6)")
    parser.add_argument("--square-m", type=float, required=True, help="Measured printed square edge in metres")
    parser.add_argument("--output", help="Accepted stereo YAML, defaults to stereo.path in rig config; explicit relative paths are relative to the working directory")
    parser.add_argument("--report", help="Diagnostics JSON; default: output with .report.json suffix")
    parser.add_argument("--min-views", type=int, default=10)
    parser.add_argument("--max-rms-px", type=float, default=1.0)
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    # Provide a report path even if the rig file is malformed or missing.
    output = Path(args.output).expanduser() if args.output else config_path.parent / "stereo.yaml"
    report_path = Path(args.report).expanduser() if args.report else output.with_suffix(".report.json")
    report = {"status": "failed", "kind": "stereo", "created_utc": datetime.now(timezone.utc).isoformat(), "opencv_version": cv2.__version__, "pairs_used": [], "pairs_without_board": [], "warnings": [], "pairing": "Identical filename stems. Operator must keep the board stationary throughout each pair; hardware synchronization is not inferred."}
    try:
        board_points(args.cols, args.rows, args.square_m)
        config = yaml.safe_load(config_path.read_text())
        if not args.output:
            output = resolve_path(config_path, config["stereo"]["path"])
            if not args.report:
                report_path = output.with_suffix(".report.json")
        paths = {camera: resolve_path(config_path, config["cameras"][camera]["intrinsics"]) for camera in ("c920", "brio101")}
        intrinsics = {camera: load_intrinsics(path) for camera, path in paths.items()}
        manifests = {"c920": read_capture_profile(args.left_images), "brio101": read_capture_profile(args.right_images)}
        for camera in intrinsics:
            check_capture_profile(manifests[camera], intrinsics[camera], camera)
            if not manifests[camera] or intrinsics[camera].get("capture_profile") is None:
                report["warnings"].append(f"{camera}: complete capture-profile provenance unavailable; operator must ensure focus, zoom and other capture settings match intrinsic calibration.")
        left, right = indexed_images(args.left_images), indexed_images(args.right_images)
        if left.keys() != right.keys():
            raise CalibrationError(f"Unpaired image filenames: C920-only {sorted(left.keys() - right.keys())}; Brio-only {sorted(right.keys() - left.keys())}. Pair by identical stems, not chronological filename sorting.")
        left_points, right_points = [], []
        for stem in sorted(left):
            size1, p1 = detect_board(left[stem], args.cols, args.rows)
            size2, p2 = detect_board(right[stem], args.cols, args.rows)
            for camera, size in (("c920", size1), ("brio101", size2)):
                if size != intrinsics[camera]["image_size"]:
                    raise CalibrationError(f"{camera} image {stem} dimensions {size} do not match intrinsics {intrinsics[camera]['image_size']}. Do not resize calibration images.")
            if p1 is None or p2 is None:
                report["pairs_without_board"].append({"stem": stem, "c920_detected": p1 is not None, "brio101_detected": p2 is not None})
                continue
            left_points.append(p1)
            right_points.append(p2)
            report["pairs_used"].append(stem)
        result, diagnostics = fit_stereo(left_points, right_points, intrinsics["c920"], intrinsics["brio101"], args.cols, args.rows, args.square_m, min_views=args.min_views, max_rms_px=args.max_rms_px)
        report.update(diagnostics)
        result["intrinsics_sha256"] = {camera: hashlib.sha256(path.read_bytes()).hexdigest() for camera, path in paths.items()}
        result["created_utc"] = report["created_utc"]
        report["status"] = "passed"
        write_report(report_path, report)
        write_yaml_atomic(output, result)
    except (CalibrationError, OSError, KeyError, TypeError, ValueError, cv2.error, yaml.YAMLError) as exc:
        report.update(getattr(exc, "report", {}))
        report.update(status="failed", error=str(exc))
        write_report(report_path, report)
        print(f"Calibration rejected: {exc}\nReport: {report_path}\nExisting calibration was not replaced.", file=sys.stderr)
        return 2
    print(f"Accepted {len(left_points)} pairs: RMS {result['rms_px']:.3f} px, baseline {result['baseline_m']:.4f} m.\nStereo: {output}\nReport: {report_path}")
    for warning in report["warnings"]:
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
