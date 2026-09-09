#!/usr/bin/env python
"""Map a kinematic wheel replay and persistent material probes to real video.

Uses the run's saved calibration, not a subsequently edited config. This is
a geometric replay of measurements, not a physics prediction or ground truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from swivel.angular import angular_intervals, replay_phases, write_angular_csv
from swivel.geometry import SwivelGeometry, project_camera_points
from swivel.masking import wheel_tracking_mask
from swivel.roll import RollEstimator


def phase_spokes(geometry, psi, phase, face_sign):
    first, second = geometry.radial_basis_zero()
    theta = phase + np.arange(6)*np.pi/3
    radial = .85*geometry.wheel_radius*(np.cos(theta)[:, None]*first +
                                       np.sin(theta)[:, None]*second)
    return geometry.face_center_car(psi, face_sign) + radial @ geometry.R_car_from_fork(psi).T


def draw_model(image, geometry, K, psi, phase, *, solid=False):
    def uv(points):
        pixels, valid = project_camera_points(geometry.car_points_to_camera(points), K)
        if not valid.all() or not np.isfinite(pixels).all() or np.abs(pixels).max() > 1e6:
            return None
        return np.rint(pixels).astype(np.int32)

    sign = geometry.visible_face_sign(psi)
    rims = [uv(geometry.sidewall_boundary_car(psi, side)) for side in (-sign, sign)]
    if any(r is None for r in rims):
        return
    if solid:
        cv2.fillConvexPoly(image, cv2.convexHull(np.concatenate(rims)), (85, 85, 85), cv2.LINE_AA)
        cv2.fillPoly(image, [rims[1]], (220, 225, 225), cv2.LINE_AA)
    for rim, color in zip(rims, ((160, 120, 60), (40, 220, 255))):
        cv2.polylines(image, [rim], True, color, 2, cv2.LINE_AA)
    # The fork is a schematic centerline; its silhouette was not measured.
    axis = uv(np.array([geometry.swivel_axis_car, geometry.hub_center_car(psi)]))
    axle = uv(np.array([geometry.face_center_car(psi, -1), geometry.face_center_car(psi, 1)]))
    for line in (axis, axle):
        if line is not None:
            cv2.line(image, tuple(line[0]), tuple(line[1]), (255, 180, 80), 3, cv2.LINE_AA)
    if np.isfinite(phase):
        center = uv(geometry.face_center_car(psi, sign)[None])
        spokes = uv(phase_spokes(geometry, psi, phase, sign))
        if center is not None and spokes is not None:
            for i, endpoint in enumerate(spokes):
                cv2.line(image, tuple(center[0]), tuple(endpoint),
                         (80, 60, 230) if i == 0 else (150, 150, 80), 2, cv2.LINE_AA)


def motion_plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [r["t_mid_s"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    for column, name, label in ((0, "psi", "Swivel"), (1, "phi", "Roll")):
        for index, key, unit in ((0, f"delta_{name}_deg", "change (deg/frame pair)"),
                                 (1, f"{name}_dot_deg_s", "rate (deg/s)")):
            axes[index, column].plot(t, [np.nan if r[key] is None else r[key] for r in rows], lw=.8)
            axes[index, column].set_ylabel(f"{label} {unit}")
            axes[index, column].grid(alpha=.25)
        axes[1, column].set_xlabel("Video time (s)")
    fig.suptitle("Accepted interval measurements | gaps unavailable | raw rates, no smoothing")
    fig.tight_layout()
    fig.savefig(output / "angular_motion.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--clip", type=Path, help="Override saved input path; must be the same frame sequence")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-only", action="store_true", help="Only angular CSV, plot and report")
    parser.add_argument("--width", type=int, default=1920, help="Combined video width (even)")
    args = parser.parse_args(argv)
    if args.width < 640 or args.width % 2:
        parser.error("--width must be even and at least 640")
    cv2.setNumThreads(2)
    data = json.loads(args.results.read_text(encoding="utf-8"))
    rows = angular_intervals(data)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_angular_csv(output / "angular_motion.csv", rows)
    motion_plot(rows, output)
    meta = data["metadata"]
    report = {
        "source_results": str(args.results.resolve()),
        "type": "Measured kinematic replay; not a dynamics simulation or independent accuracy test",
        "angular_rate_convention": "Signed raw phi/psi interval averages, radians/s and degrees/s; unavailable is null/blank",
        "timestamp_source": meta.get("timestamp_source", "frame_index / reported_fps"),
        "time_offset_s": meta.get("time_offset_s", 0),
        "physical_direction_sign_calibrated": meta.get("direction_sign_calibrated", False),
        "interval_count": len(rows),
        "roll_valid_intervals": sum(r["roll_valid"] for r in rows),
        "swivel_valid_intervals": sum(r["swivel_valid"] for r in rows),
        "accumulated_roll_complete": bool(data["phi_valid"][-1]),
        "calibration_source": "results.metadata.camera and results.metadata.swivel_calibration",
        "limitations": ["Constant-FPS timestamps; variable frame timing requires recorded presentation timestamps",
                        "Finite differences amplify angle noise; rates are interval averages",
                        "Geometry and camera must stay fixed relative to the caster mount",
                        "Roll spokes have arbitrary local zero; probes re-anchor after every loss or face switch",
                        "Probe alignment is a visual consistency check, not encoder ground truth"],
    }
    if not args.export_only:
        report.update(render_video(data, rows, args, output))
    (output / "replay_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "roll_valid": report["roll_valid_intervals"],
                      "swivel_valid": report["swivel_valid_intervals"], "intervals": len(rows)}))
    return 0


def render_video(data, rows, args, output):
    meta = data["metadata"]
    camera = meta["camera"]
    calibration = meta["swivel_calibration"]
    geometry = SwivelGeometry.from_config(camera, calibration["geometry"])
    K = np.asarray(camera["K"], float)
    dist = None if camera.get("dist") is None else np.asarray(camera["dist"], float)
    psi = np.asarray(data["psi_rad"], float)
    psi_valid = np.asarray(data["psi_valid"], bool) & np.isfinite(psi)
    faces = [geometry.visible_face_sign(a) if v else 0 for a, v in zip(psi, psi_valid)]
    threshold = float(calibration["geometry"].get("edge_on_min_cos", .18))
    phase_valid = [v and geometry.sidewall_view_confidence(a) >= threshold for a, v in zip(psi, psi_valid)]
    phase, segments = replay_phases(rows, phase_valid, faces)
    mask_cfg = meta.get("processing_config", {}).get("mask", {})
    marker = calibration["marker"]
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, marker.get("dictionary", "DICT_4X4_50")))
    detector = cv2.aruco.ArucoDetector(dictionary)
    predictor = RollEstimator(geometry=geometry, K=K)
    clip = args.clip or Path(meta["input"])
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {clip}")
    writer = None
    map1 = map2 = None
    probes = None
    anchor_psi = anchor_phase = 0.
    anchors = []
    snapshots = []
    snapshot_indices = set(np.linspace(0, len(psi)-1, 5, dtype=int).tolist())
    try:
        if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(psi):
            raise ValueError("Video frame count differs from results; supply the original clip")
        for i in range(len(psi)):
            ok, raw = cap.read()
            if not ok:
                raise ValueError(f"Video ended before results at frame {i}")
            h, w = raw.shape[:2]
            if writer is None:
                panel_width = args.width//2
                panel_height = 2*round((h/w*panel_width)/2)
                writer = cv2.VideoWriter(str(output / "measured_replay.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), float(meta["fps"]),
                                         (args.width, panel_height))
                if not writer.isOpened():
                    raise RuntimeError("Could not open MP4 writer")
                if dist is not None and np.any(dist):
                    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
            frame = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR) if map1 is not None else raw
            overlay = frame.copy()
            model = np.full_like(frame, 35)
            status = "SWIVEL UNAVAILABLE - model hidden"
            if psi_valid[i]:
                draw_model(overlay, geometry, K, psi[i], phase[i])
                draw_model(model, geometry, K, psi[i], phase[i], solid=True)
                status = f"psi={np.rad2deg(psi[i]):.1f} deg | roll unavailable / edge-on"
                if np.isfinite(phase[i]):
                    corners, ids, _ = detector.detectMarkers(frame)
                    observation = SimpleNamespace(detected_corners={} if ids is None else
                        {int(key): c.reshape(4, 2) for key, c in zip(ids.flatten(), corners)}, corners_uv=None)
                    mask, _ = wheel_tracking_mask(frame, geometry, K, psi[i], face_sign=faces[i],
                        inner_fraction=float(calibration["geometry"].get("sidewall_inner_fraction", .2)),
                        config=mask_cfg, observation=observation)
                    if i == 0 or segments[i] != segments[i-1]:
                        found = cv2.goodFeaturesToTrack(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                            35, .03, 12, mask=mask.astype(np.uint8)*255)
                        probes = np.empty((0, 2)) if found is None else found.reshape(-1, 2)
                        anchor_psi, anchor_phase = psi[i], phase[i]
                        anchors.append({"frame": i, "segment": int(segments[i]), "face_sign": faces[i],
                                        "probe_count": len(probes)})
                    if probes is not None and len(probes):
                        pixels, good = predictor.predict_pixels(probes, anchor_psi, psi[i], faces[i], phase[i]-anchor_phase)
                        for pixel in pixels[good]:
                            x, y = np.rint(pixel).astype(int)
                            if 0 <= x < w and 0 <= y < h and mask[y, x]:
                                cv2.circle(overlay, (x, y), 5, (255, 40, 240), 1, cv2.LINE_AA)
                                cv2.circle(model, (x, y), 4, (255, 40, 240), -1, cv2.LINE_AA)
                    status = f"psi={np.rad2deg(psi[i]):.1f} deg | local roll={np.rad2deg(phase[i]):.1f} deg | segment {segments[i]}"
                    if i == 0 or segments[i] != segments[i-1]:
                        status += " RE-ANCHOR"
            panels = [cv2.resize(x, (panel_width, panel_height)) for x in (overlay, model)]
            top = ["VIDEO + GEOMETRY | magenta: predicted material probes", "MEASURED KINEMATIC REPLAY | schematic fork"]
            for panel, title in zip(panels, top):
                cv2.rectangle(panel, (0, 0), (panel_width, 80), (20, 20, 20), -1)
                for j, line in enumerate([title, f"frame {i} | {status}",
                    "Arbitrary local roll zero; gaps reset probes; yellow = visible wheel rim"]):
                    cv2.putText(panel, line, (10, 21+24*j), cv2.FONT_HERSHEY_SIMPLEX, .43, (240, 240, 240), 1, cv2.LINE_AA)
                if i:
                    r = rows[i-1]
                    fmt = lambda value: "unavailable" if value is None else f"{value:.1f} deg/s"
                    line = f"swivel rate: {fmt(r['psi_dot_deg_s'])} | roll rate: {fmt(r['phi_dot_deg_s'])}"
                    cv2.rectangle(panel, (0, panel_height-35), (panel_width, panel_height), (20, 20, 20), -1)
                    cv2.putText(panel, line, (10, panel_height-12), cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1, cv2.LINE_AA)
            combined = np.hstack(panels)
            writer.write(combined)
            if i in snapshot_indices:
                snapshots.append(cv2.resize(combined, (960, round(panel_height*960/args.width))))
        if snapshots:
            cv2.imwrite(str(output / "replay_preview.jpg"), np.vstack(snapshots))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return {"video": str(clip.resolve()), "rendered_frames": len(psi), "probe_anchors": anchors,
            "replay_segments": len(anchors), "images_undistorted": map1 is not None}


if __name__ == "__main__":
    raise SystemExit(main())
