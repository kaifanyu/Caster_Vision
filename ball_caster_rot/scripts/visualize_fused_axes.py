#!/usr/bin/env python
"""Inspect experimental angle-Kalman outputs without relabelling pose validity.

Produces an all-frame video, an eight-panel overview and a standalone HTML
slider. The saved camera, calibrated frame and surface geometry remain fixed.
The displayed 2-sigma band describes tuned filter uncertainty conditional on
that calibration; it is not an estimate of absolute measurement accuracy.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ballrot.camera import undistort_image
from ballrot.io_frames import FrameSource
from ballrot.rotation import Rx, Rz
from ballrot.sphere import sphere_pose_from_circle
from scripts.simulate_measured import _effective_pivot, _graticule, _mechanical_model, _project_shell

COLORS = {"x": "#67ed77", "y": "#68b8ff", "z": "#ffab4b", "top": "#49e5ee",
          "bottom": "#ed7ee2", "vision": "#c9d1d9", "mechanical": "#f7efb0"}
CHANNELS = ("alpha", "beta_top", "beta_bottom")
STATES = ("vision", "predicted", "unresolved")


def _load_results(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("method") != "experimental_angle_kalman":
        raise ValueError("Expected schema 1 experimental_angle_kalman results")
    if data.get("measurement_source") not in ("unconstrained", "mechanical"):
        raise ValueError("measurement_source must identify unconstrained or mechanical visual poses")
    frames = data["frames"]
    times = np.asarray(frames["time_s"], dtype=float)
    if len(times) < 2 or times.ndim != 1 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("time_s must contain strictly increasing finite timestamps")
    for name in CHANNELS:
        required = [f"{name}_rad", f"{name}_std_rad", f"{name}_status", f"{name}_vision_rad"]
        if any(key not in frames or len(frames[key]) != len(times) for key in required):
            raise ValueError(f"Missing or inconsistent arrays for {name}")
        for angle, deviation, state in zip(*(frames[key] for key in required[:3])):
            if state not in STATES:
                raise ValueError(f"Unknown filter status: {state}")
            if state != "unresolved" and (angle is None or deviation is None
                    or not np.isfinite(angle) or not np.isfinite(deviation) or deviation < 0):
                raise ValueError("Supported filter values require finite angles and nonnegative uncertainty")
        for suffix in ("prediction_age_s", "reinitialized"):
            key = f"{name}_{suffix}"
            if key in frames and len(frames[key]) != len(times):
                raise ValueError(f"Inconsistent {key}")
    for key in ("mechanical_valid_top", "mechanical_valid_bottom", "mechanical_alpha_rad"):
        if key not in frames or len(frames[key]) != len(times):
            raise ValueError(f"Missing or inconsistent {key}")
    for key in ("mechanical_valid_top", "mechanical_valid_bottom"):
        if any(not isinstance(value, bool) for value in frames[key]):
            raise ValueError("Original mechanical validity must be boolean")
    return data


def _segments(pixels, visible):
    result, segment = [], []
    for point, shown in zip(pixels, visible):
        if shown and np.all(np.isfinite(point)):
            segment.append(point.tolist())
        else:
            if len(segment) > 1:
                result.append(segment)
            segment = []
    if len(segment) > 1:
        result.append(segment)
    return result


def _sample_geometry(frames, index, K, F, C, radius, model, crop, scale, size, top_sign=1):
    """Project one explicitly labelled filter state in the saved source frame."""
    x0, y0 = crop[:2]

    def screen(pixels):
        return (np.asarray(pixels) - [x0, y0]) * scale

    def point(xyz):
        projected = K @ xyz
        if projected[2] <= 1e-9 or not np.all(np.isfinite(projected)):
            return None
        return screen(projected[:2] / projected[2]).tolist()

    def arrow(direction, name, status):
        endpoint = point(C + 1.08 * radius * direction)
        return None if endpoint is None else {"name": name, "origin": origin, "end": endpoint, "status": status}

    origin = point(C)
    status = {name: frames[f"{name}_status"][index] for name in CHANNELS}
    values = {name: (None if status[name] == "unresolved" else float(frames[f"{name}_rad"][index]))
              for name in CHANNELS}
    sigma = {name: (None if status[name] == "unresolved" else float(frames[f"{name}_std_rad"][index]))
             for name in CHANNELS}
    age = {name: frames.get(f"{name}_prediction_age_s", [None] * len(frames["time_s"]))[index]
           for name in CHANNELS}
    reinitialized = {name: bool(frames.get(f"{name}_reinitialized", [False] * len(frames["time_s"]))[index])
                     for name in CHANNELS}
    axes = [arrow(F[:, 0], "x", "calibrated")]
    alpha = values["alpha"]
    current = None if alpha is None else F @ Rx(alpha)
    envelope = []
    if current is not None:
        axes += [arrow(current[:, j], name, status["alpha"]) for j, name in ((1, "y"), (2, "z"))]
        half_width = min(np.pi, 2 * sigma["alpha"])
        envelope = [point(C + 1.08 * radius * (F @ Rx(value))[:, 2])
                    for value in np.linspace(alpha - half_width, alpha + half_width, 49)]
        if any(value is None for value in envelope):
            envelope = []
    comparison = []
    raw = frames["alpha_vision_rad"][index]
    if raw is not None and np.isfinite(raw):
        comparison.append(arrow((F @ Rx(float(raw)))[:, 2], "vision", "reference"))
    mechanical_ok = bool(frames["mechanical_valid_top"][index] or frames["mechanical_valid_bottom"][index])
    mechanical_alpha = frames["mechanical_alpha_rad"][index]
    if mechanical_ok and mechanical_alpha is not None and np.isfinite(mechanical_alpha):
        comparison.append(arrow((F @ Rx(float(mechanical_alpha)))[:, 2], "mechanical", "reference"))

    grids = []
    geometry, gap = model["geometry"], model["gap_fraction"]
    for shell, sign in (("top", top_sign), ("bottom", -top_sign)):
        if current is None or values[f"beta_{shell}"] is None:
            continue
        orientation = current @ Rz(values[f"beta_{shell}"])
        mode = "predicted" if "predicted" in (status["alpha"], status[f"beta_{shell}"]) else "vision"
        for curve in _graticule(45., sign, 40, gap if geometry == "common_sphere_caps" else 0.):
            pixels, visible = _project_shell(curve, orientation, K, C, radius, sign, gap, geometry)
            grids += [{"shell": shell, "status": mode, "points": segment}
                      for segment in _segments(screen(pixels), visible)]
    return {"frame": index, "time": float(frames["time_s"][index]), "size": size,
            "status": status, "angles_deg": {key: None if value is None else float(np.rad2deg(value))
                                             for key, value in values.items()},
            "prediction_age_s": age, "reinitialized": reinitialized,
            "sigma_deg": {key: None if value is None else float(np.rad2deg(value)) for key, value in sigma.items()},
            "mechanical_top": bool(frames["mechanical_valid_top"][index]),
            "mechanical_bottom": bool(frames["mechanical_valid_bottom"][index]),
            "axes": [value for value in axes if value is not None], "origin": origin,
            "envelope": envelope, "comparison": [value for value in comparison if value is not None],
            "grids": grids}


def _bgr(name):
    value = COLORS[name].lstrip("#")
    return tuple(int(value[i:i+2], 16) for i in (4, 2, 0))


def _line(image, points, color, thickness=1, dashed=False):
    points = np.asarray(points, dtype=float)
    if not dashed:
        cv2.polylines(image, [np.rint(points).astype(np.int32)], False, color, thickness, cv2.LINE_AA)
        return
    phase = 0.
    for first, second in zip(points[:-1], points[1:]):
        distance = float(np.linalg.norm(second - first))
        if distance < 1e-9:
            continue
        offset = 0.
        while offset < distance:
            length = min(distance - offset, 14. - phase)
            if phase < 8.:
                length = min(length, 8. - phase)
                start = first + (second - first) * offset / distance
                end = first + (second - first) * (offset + length) / distance
                cv2.line(image, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)),
                         color, thickness, cv2.LINE_AA)
            offset += length
            phase = (phase + length) % 14.


def _overlay(background, row, source_label):
    image = background.copy()
    if row["envelope"]:
        translucent = image.copy()
        polygon = np.rint([row["origin"], *row["envelope"]]).astype(np.int32)
        cv2.fillPoly(translucent, [polygon], _bgr("z"), cv2.LINE_AA)
        image = cv2.addWeighted(translucent, .18, image, .82, 0)
        _line(image, row["envelope"], _bgr("z"), 1)
    for grid in row["grids"]:
        _line(image, grid["points"], _bgr(grid["shell"]), 1, grid["status"] == "predicted")
    for axis in row["axes"]:
        start, end = np.asarray(axis["origin"]), np.asarray(axis["end"])
        predicted = axis["status"] == "predicted"
        color = _bgr(axis["name"])
        if predicted:
            _line(image, [start, end], color, 3, True)
        else:
            cv2.arrowedLine(image, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)),
                            color, 3, cv2.LINE_AA, tipLength=.06)
        label = np.clip(end + [8, -8], [8, 115], np.array(row["size"]) - [25, 60])
        cv2.putText(image, axis["name"].upper(), tuple(np.rint(label).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (15, 20, 25), 5, cv2.LINE_AA)
        cv2.putText(image, axis["name"].upper(), tuple(np.rint(label).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2, cv2.LINE_AA)
    state = {"vision": "VISION-FUSED", "predicted": "PREDICTION ONLY", "unresolved": "UNRESOLVED"}[row["status"]["alpha"]]
    angle, sigma = row["angles_deg"]["alpha"], row["sigma_deg"]["alpha"]
    angle_text = "" if angle is None else f" | relative roll {angle:.1f} deg | sigma {sigma:.2f} deg"
    header = [f"EXPERIMENTAL - {source_label} + motion prior",
              f"{row['time']:.3f} s | frame {row['frame']} | {state}{angle_text}",
              f"Shell spin: top {row['status']['beta_top']} / bottom {row['status']['beta_bottom']} | original mechanical: "
              f"top {'accepted' if row['mechanical_top'] else 'unresolved'}, bottom {'accepted' if row['mechanical_bottom'] else 'unresolved'}"]
    footer = ["Solid: vision-fused. Dashed: predicted. X: fixed calibration. Shaded Z: +/-2 sigma.",
              "Uncertainty uses tuned noise and fixed calibration. No new pixel validation or accuracy guarantee."]
    cv2.rectangle(image, (0, 0), (image.shape[1], 88), (14, 22, 29), -1)
    cv2.rectangle(image, (0, image.shape[0]-51), (image.shape[1], image.shape[0]), (14, 22, 29), -1)
    for y, label in [*zip((24, 50, 76), header), *zip((image.shape[0]-31, image.shape[0]-10), footer)]:
        width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .53, 1)[0][0]
        scale = .53 * min(1., (image.shape[1]-20) / max(width, 1))
        cv2.putText(image, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 239, 242), 1, cv2.LINE_AA)
    return image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    data = _load_results(args.results)
    frames, metadata = data["frames"], data["metadata"]
    times = np.asarray(frames["time_s"], dtype=float)
    out = args.output or args.results.parent / "axis_inspection"
    out.mkdir(parents=True, exist_ok=True)
    model = _mechanical_model({"metadata": metadata}) or {
        "geometry": "common_sphere_caps", "gap_fraction": 0., "pivot_camera": None}
    K, dist, F = (np.asarray(metadata[key], dtype=float) for key in ("K", "dist", "R_bc"))
    circle = np.asarray(metadata["circle"], dtype=float)
    C, radius = sphere_pose_from_circle(*circle, K)
    C = _effective_pivot(model, C, radius)
    top_sign = int(metadata.get("top_shell_sign", 1))
    if top_sign not in (-1, 1):
        raise ValueError("Saved top_shell_sign must be +1 or -1")
    source_label = "unconstrained visual pose" if data["measurement_source"] == "unconstrained" else "accepted mechanical pose"
    nearest = lambda value: int(np.argmin(np.abs(times - value)))
    montage = [nearest(value) for value in (0., 1.2, 2.4, 3.2, 4.7, 8., 14., times[-1])]
    predicted = np.flatnonzero(np.asarray(frames["alpha_status"]) == "predicted")
    if len(predicted):
        montage[-2] = int(predicted[-1])
    selected = set(np.flatnonzero(times <= 3.).tolist())
    selected.update(nearest(value) for value in np.arange(3., times[-1], .25))
    for name in CHANNELS:
        states = np.asarray(frames[f"{name}_status"])
        transitions = np.flatnonzero(states[1:] != states[:-1]) + 1
        selected.update(transitions.tolist())
        selected.update((transitions - 1).tolist())
        selected.update(np.flatnonzero(frames.get(f"{name}_reinitialized", [])).tolist())
    selected.update(montage)
    html_rows, overview = {}, {}
    video = out / "fused_axes_overlay.mp4"
    writer, crop, count = None, None, 0
    try:
        for record in FrameSource(Path(metadata["input"]), max_frames=len(times)):
            index = int(record.index)
            background = undistort_image(record.image, K, dist)
            if crop is None:
                h, w = background.shape[:2]
                u, v, r = circle
                crop = (max(0, int(u-1.30*r)), max(0, int(v-1.17*r)),
                        min(w, int(u+1.30*r)), min(h, int(v+1.17*r)))
                scale = 860. / (crop[2]-crop[0])
                size = [860, 2 * int(round((crop[3]-crop[1])*scale / 2))]
                scale = np.asarray(size) / [crop[2]-crop[0], crop[3]-crop[1]]
                writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                                         float(metadata["fps"]), tuple(size))
                if not writer.isOpened():
                    raise OSError("Could not create fused_axes_overlay.mp4")
            x0, y0, x1, y1 = crop
            background = cv2.resize(background[y0:y1, x0:x1], tuple(size), interpolation=cv2.INTER_AREA)
            row = _sample_geometry(frames, index, K, F, C, radius, model, crop, scale, size, top_sign)
            overlay = _overlay(background, row, source_label)
            writer.write(overlay)
            if index in montage:
                overview[index] = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            if index in selected:
                ok, encoded = cv2.imencode(".jpg", background, [cv2.IMWRITE_JPEG_QUALITY, 86])
                if not ok:
                    raise OSError("Could not encode source image for timeline")
                row["image"] = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
                html_rows[index] = row
            count += 1
    finally:
        if writer is not None:
            writer.release()
    if count != len(times) or set(html_rows) != selected:
        raise ValueError("Source video does not contain all requested filter frames")

    fig, panels = plt.subplots(2, 4, figsize=(20, 10), facecolor="#101820")
    for panel, index in zip(panels.flat, montage):
        panel.imshow(overview[index])
        panel.set_axis_off()
    fig.suptitle("Experimental fused axes: visual pose + motion prior", color="white", fontsize=20, y=.98)
    fig.text(.5, .942, "GREEN X: fixed roll axis    ORANGE Z: fused spin axis    BLUE Y: transverse axis",
             color="#d9e5ec", ha="center", fontsize=11)
    fig.text(.5, .025, "Solid = vision-fused; dashed = prediction. Uncertainty is conditional on fixed calibration. Original mechanical validity is preserved.",
             color="#d9e5ec", ha="center", fontsize=10)
    fig.subplots_adjust(left=.012, right=.988, bottom=.055, top=.91, hspace=.045, wspace=.04)
    png = out / "fused_axes_at_times.png"
    fig.savefig(png, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    info = {"measurement_source": data["measurement_source"], "source_label": source_label,
            "source_results": data["source_results"], "model": model,
            "initial_roll_deg": metadata.get("initial_roll_deg"), "parameters": data["parameters"],
            "uncertainty_note": "Tuned filter covariance conditional on fixed calibration and chosen noise; not absolute accuracy."}
    template = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    html = out / "fused_axes_timeline.html"
    substitutions = {"__FUSED_DATA__": [html_rows[index] for index in sorted(html_rows)],
                     "__FUSED_COLORS__": COLORS, "__FUSED_INFO__": info}
    for token, value in substitutions.items():
        template = template.replace(token, json.dumps(value, allow_nan=False).replace("</", "<\\/"))
    html.write_text(template, encoding="utf-8")
    manifest = {"results": str(args.results.resolve()), "source_results": data["source_results"],
                "method": data["method"], "measurement_source": data["measurement_source"],
                "frames_in_video": count, "sample_frames": sorted(selected), "overview_frames": montage,
                "model": model, "pivot": C.tolist(), "initial_frame": F.tolist(),
                "uncertainty_note": info["uncertainty_note"],
                "timing_note": "Constant-rate video playback; labels and timeline use saved source timestamps.",
                "validation_note": "Experimental visualization adds no pixel validation and does not change original mechanical validity.",
                "outputs": {"video": str(video.resolve()), "overview": str(png.resolve()), "timeline": str(html.resolve())}}
    (out / "fused_axis_inspection_report.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"Video: {video}\nOverview: {png}\nTimeline: {html}\nVideo frames: {count}; timeline samples: {len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
