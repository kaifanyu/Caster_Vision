#!/usr/bin/env python
"""Calibrate the pivot of two complete, separated hemispheres from OUTER arcs.

Use --annotate to click reviewed outer silhouette points on the undistorted
first frame, or --points with JSON in this form:

  {"coordinate_system": "undistorted_pixels", "frames": [
    {"frame_index": 0, "alpha_deg": 0,
     "top": [[u, v], ...], "bottom": [[u, v], ...]}
  ]}

Top/bottom mean the configured color labels. Supply at least six well-spread
points on EACH curved OUTER silhouette, preferably many more. Exclude the rim,
inner discs, yoke, shadows and paint edges. Additional frames require KNOWN
alpha_deg relative to the first frame; never use an unresolved fitted roll.

The camera, starting axes and measured gap are held fixed. The output is a
candidate pivot in units of one shell curvature radius, not a metric-radius
measurement. JSON and preview images are saved by default. Only --write writes
mechanical.pivot_camera after the residual, support and conditioning gates pass;
camera/circle calibration is preserved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from ballrot.camera import undistort_image
from ballrot.config import (camera_matrix, configured_circle, distortion_coefficients,
                            load_config, measurement_frame, resolve_from_config, update_yaml)
from ballrot.io_frames import FrameSource
from ballrot.shell_calibration import calibrate_shell_pivot
from ballrot.sphere import sphere_pose_from_circle


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--points", type=Path, help="Reviewed undistorted outer-silhouette JSON; see schema above")
    source.add_argument("--annotate", action="store_true", help="Open first-frame click interface; T/B switches shell")
    result.add_argument("--clip", type=Path, help="Video/image sequence used for previews (default: config input.path)")
    result.add_argument("--output", type=Path, default=PROJECT_ROOT / "out" / "shell_calibration")
    result.add_argument("--write", action="store_true", help="Apply a passed candidate to mechanical.pivot_camera")
    return result


def _annotate(image):
    points = {"top": [], "bottom": []}
    shell = "top"
    window = "OUTER shell silhouette: T/B shell, click add, U undo, R reset shell, Enter finish, Esc cancel"
    colors = {"top": (255, 220, 0), "bottom": (180, 0, 255)}

    def callback(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            point = [float(x), float(y)]
            if point not in points[shell]:
                points[shell].append(point)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, callback)
    try:
        while True:
            display = image.copy()
            for name, values in points.items():
                for index, point in enumerate(values):
                    position = tuple(np.rint(point).astype(int))
                    cv2.circle(display, position, 5, colors[name], 2, cv2.LINE_AA)
                    cv2.putText(display, f"{name[0].upper()}{index}", (position[0] + 6, position[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[name], 1, cv2.LINE_AA)
            lines = [f"Selecting {shell}: top={len(points['top'])}, bottom={len(points['bottom'])}",
                     "Click CURVED OUTER silhouette only: exclude rim, discs and yoke; >=6 broad arc points per shell.",
                     "T/B switch | U undo | R clear selected shell | Enter fit candidate | Esc cancel"]
            for index, line in enumerate(lines):
                y = 26 + index * 25
                cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("t"), ord("T")):
                shell = "top"
            elif key in (ord("b"), ord("B")):
                shell = "bottom"
            elif key in (ord("u"), ord("U"), 8) and points[shell]:
                points[shell].pop()
            elif key in (ord("r"), ord("R")):
                points[shell].clear()
            elif key in (13, 10) and min(map(len, points.values())) >= 6:
                return {"coordinate_system": "undistorted_pixels", "frames": [{"frame_index": 0, "alpha_deg": 0, **points}]}
            elif key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise KeyboardInterrupt("shell annotation cancelled")
    finally:
        cv2.destroyWindow(window)


def _read_frames(clip, indices, K, dist):
    wanted = set(indices)
    frames = {}
    for record in FrameSource(clip, input_type="auto"):
        if int(record.index) in wanted:
            frames[int(record.index)] = undistort_image(record.image, K, dist)
        if len(frames) == len(wanted):
            break
    missing = wanted - set(frames)
    if missing:
        raise ValueError(f"source clip does not contain requested frame indices: {sorted(missing)}")
    return frames


def _preview(image, report, index, destination):
    image = image.copy()
    for row in report["observations"]:
        if row["frame_index"] != index:
            continue
        observed = tuple(np.rint(row["observed_uv"]).astype(int))
        predicted = tuple(np.rint(row["predicted_uv"]).astype(int))
        color = (60, 220, 60) if row["inlier"] else (50, 60, 245)
        cv2.circle(image, observed, 5, color, 2, cv2.LINE_AA)
        cv2.drawMarker(image, predicted, color, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
        cv2.arrowedLine(image, observed, predicted, color, 1, cv2.LINE_AA, tipLength=0.2)
        cv2.putText(image, row["shell"][0].upper(), (observed[0] + 7, observed[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    lines = [f"CANDIDATE: {report['status']}  frame {index}",
             "Circle: supplied OUTER silhouette | cross: fitted tangent | green: inlier, red: rejected",
             f"Pivot / curvature radius: {np.round(report['pivot_camera'], 5).tolist()}",
             "Fixed camera, axes and gap; pixel agreement is not an absolute accuracy measurement."]
    for offset, line in enumerate(lines):
        y = 26 + 25 * offset
        scale = min(0.55, max(0.25, (image.shape[1] - 20) / (len(line) * 10.0)))
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"could not save calibration preview: {destination}")


def main(argv=None):
    args = parser().parse_args(argv)
    config, config_path = load_config(args.config)
    if config.get("camera", {}).get("K") is None:
        raise ValueError("camera.K must be calibrated before shell geometry")
    K, _ = camera_matrix(config["camera"], (1, 1, 3))
    dist = distortion_coefficients(config["camera"])
    F = measurement_frame(config.get("frame_calib", {}))
    if F is None:
        raise ValueError("frame_calib.R_bc is required")
    mechanics = config.get("mechanical", {})
    if mechanics.get("geometry", "common_sphere_caps") != "separated_hemispheres":
        raise ValueError("this tool requires mechanical.geometry: separated_hemispheres")
    if "gap_fraction" not in mechanics:
        raise ValueError("set the measured mechanical.gap_fraction before calibrating shell geometry")
    clip = args.clip or resolve_from_config(config_path, config["input"]["path"])
    if args.annotate:
        images = _read_frames(clip, [0], K, dist)
        evidence = _annotate(images[0])
    else:
        evidence = json.loads(args.points.read_text(encoding="utf-8"))
        images = None
    if evidence.get("coordinate_system") != "undistorted_pixels":
        raise ValueError("points JSON must declare coordinate_system: undistorted_pixels")
    frames = evidence.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("points JSON requires a nonempty frames list")
    initial = mechanics.get("pivot_camera")
    if initial is None:
        circle = configured_circle(config.get("circle", {}))
        if circle is not None:
            initial, _ = sphere_pose_from_circle(*circle, K)
    report = calibrate_shell_pivot(frames, K, F, gap_fraction=mechanics["gap_fraction"],
                                  top_shell_sign=config.get("frame_calib", {}).get("top_shell_sign", 1),
                                  initial_pivot=initial)
    report.update(config=str(config_path), source_clip=str(Path(clip).resolve()),
                  source_points=str(args.points.resolve()) if args.points else "interactive reviewed clicks",
                  distortion_coefficients=dist.tolist(), config_written=False)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "boundary_points.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    if images is None:
        images = _read_frames(clip, [frame["frame_index"] for frame in frames], K, dist)
    for index, image in images.items():
        _preview(image, report, index, output / f"frame_{index:06d}.png")
    if args.write and report["accepted"]:
        update_yaml(config_path, {"mechanical": {"pivot_camera": report["pivot_camera"]}})
        report["config_written"] = True
    report_path = output / "shell_calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Shell geometry: {report['status']}")
    print(f"Candidate pivot / shell curvature radius: {report['pivot_camera']}")
    for name, details in report["per_shell"].items():
        print(f"  {name}: {details['inlier_count']}/{details['point_count']} inliers; "
              f"median={details['median_error_px']:.3f}px, p95={details['p95_error_px']:.3f}px; "
              f"arc={details['arc_coverage_deg']:.1f}deg")
    print(f"Report and reviewed evidence: {report_path}")
    if report["reasons"]:
        print("Rejected: " + ", ".join(report["reasons"]))
    if report["config_written"]:
        print("Updated mechanical.pivot_camera; refit saved observations using this config.")
    else:
        print("Candidate only; config unchanged. --write applies only a candidate that passes every gate.")
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
