#!/usr/bin/env python
"""Create a contact sheet and standalone time slider of saved mechanical axes.

The roll axis is the fixed calibrated F[:, 0]. The current spin axis is
(F @ Rx(alpha))[:, 2], and is shown only when a shell supports shared roll.
Missing motion is never interpolated or held. Saved geometry is authoritative.
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
from scripts.simulate_measured import (
    _effective_pivot, _graticule, _measured, _mechanical_model,
    _pose_validity, _project_shell,
)

COLORS = {"x": "#67ed77", "y": "#68b8ff", "z": "#ffab4b",
          "top": "#49e5ee", "bottom": "#ed7ee2"}


def _segments(pixels, visible):
    result, current = [], []
    for point, shown in zip(pixels, visible):
        if shown and np.all(np.isfinite(point)):
            current.append(point.tolist())
        else:
            if len(current) > 1:
                result.append(current)
            current = []
    if len(current) > 1:
        result.append(current)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "out/real_separated_recovered/results.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    measured = _measured(args.results, None, 1)
    model = _mechanical_model(measured)
    if model is None:
        raise ValueError("This visualization requires saved shared-roll mechanical results")
    out = args.output or args.results.parent / "axis_inspection"
    out.mkdir(parents=True, exist_ok=True)
    metadata = measured["metadata"]
    K = np.asarray(metadata["K"], dtype=float)
    dist = np.asarray(metadata["dist"], dtype=float)
    F = np.asarray(metadata["R_bc"], dtype=float)
    circle = np.asarray(metadata["circle"], dtype=float)
    C, radius = sphere_pose_from_circle(*circle, K)
    C = _effective_pivot(model, C, radius)
    times = measured["time_s"]
    validity = {shell: _pose_validity(measured, shell) for shell in ("top", "bottom")}
    supported = validity["top"] | validity["bottom"]
    both = validity["top"] & validity["bottom"]
    accepted = np.flatnonzero(supported)
    joint = np.flatnonzero(both)
    last = int(accepted[-1]) if len(accepted) else 0
    last_joint = int(joint[-1]) if len(joint) else 0

    def nearest(t):
        return int(np.argmin(np.abs(times - t)))

    # Include all early source frames, where validity changes quickly, and
    # samples across the rest of the clip. Times are saved source timestamps.
    indices = set(np.flatnonzero(times <= 3.0).tolist())
    indices.update(nearest(t) for t in np.arange(3.0, times[-1], 0.5))
    indices.update([0, last, last_joint, len(times)-1])
    early_targets = [0.0, 0.65, 1.3]
    montage = [int(accepted[np.argmin(np.abs(times[accepted] - t))])
               if len(accepted) else nearest(t) for t in early_targets]
    montage += [last_joint, nearest(2.13), last, nearest(5), len(times)-1]
    indices.update(montage)
    samples, images = {}, {}
    crop = None
    gap, geometry = model["gap_fraction"], model["geometry"]
    top_sign = int(metadata.get("top_shell_sign", 1))
    curves = {s: _graticule(45, s, 80, gap if geometry == "common_sphere_caps" else 0)
              for s in (-1, 1)}

    for record in FrameSource(Path(metadata["input"]), max_frames=len(times)):
        i = int(record.index)
        if i not in indices:
            continue
        frame = undistort_image(record.image, K, dist)
        if crop is None:
            h, w = frame.shape[:2]
            u, v, r = circle
            crop = (max(0, int(u-1.30*r)), max(0, int(v-1.17*r)),
                    min(w, int(u+1.30*r)), min(h, int(v+1.17*r)))
        x0, y0, x1, y1 = crop
        scale = 860.0 / (x1-x0)
        size = [860, int(round((y1-y0)*scale))]
        background = cv2.resize(frame[y0:y1, x0:x1], tuple(size), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", background, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise OSError("Could not encode source frame")

        def screen(uv):
            return (np.asarray(uv)-[x0, y0])*scale

        def project(point):
            value = K @ point
            return screen(value[:2]/value[2]).tolist()

        origin = project(C)
        alpha = float(measured["alpha_rad"][i]) if supported[i] else None
        current = F @ Rx(alpha) if alpha is not None else None
        axes, home, grids = [], [], []
        for j, name in enumerate(("x", "y", "z")):
            home.append({"name": name, "origin": origin,
                         "end": project(C+1.08*radius*F[:, j])})
            if name == "x" or current is not None:
                axis = F[:, j] if name == "x" else current[:, j]
                axes.append({"name": name, "origin": origin,
                             "end": project(C+1.08*radius*axis)})
        for shell, sign in (("top", top_sign), ("bottom", -top_sign)):
            if not validity[shell][i]:
                continue
            orientation = current @ Rz(float(measured[f"beta_{shell}_rad"][i]))
            for curve in curves[sign]:
                pixels, shown = _project_shell(curve, orientation, K, C, radius, sign, gap, geometry)
                grids.extend({"shell": shell, "points": segment}
                             for segment in _segments(screen(pixels), shown))
        row = {"frame": i, "time": float(times[i]), "size": size,
               "top": bool(validity["top"][i]), "bottom": bool(validity["bottom"][i]),
               "alpha_deg": None if alpha is None else float(np.rad2deg(alpha)),
               "axes": axes, "home": home, "grids": grids,
               "image": "data:image/jpeg;base64,"+base64.b64encode(encoded).decode("ascii")}
        samples[i] = row
        if i in montage:
            images[i] = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    if set(samples) != indices:
        raise ValueError("The source video did not provide all requested frames")

    fig, panels = plt.subplots(2, 4, figsize=(18, 9.7), facecolor="#101820")
    for panel, i in zip(panels.flat, montage):
        row = samples[i]
        panel.imshow(images[i])
        for grid in row["grids"]:
            points = np.asarray(grid["points"])
            panel.plot(points[:, 0], points[:, 1], color=COLORS[grid["shell"]], lw=.65, alpha=.7)
        for axis in row["axes"]:
            origin, end = np.asarray(axis["origin"]), np.asarray(axis["end"])
            color = COLORS[axis["name"]]
            panel.annotate("", end, origin, arrowprops={"arrowstyle": "->", "color": color, "lw": 2})
            position = np.clip(end+[8, -8], [15, 20], np.array(row["size"])-[30, 15])
            panel.text(*position, axis["name"].upper(), color=color, fontsize=10, weight="bold",
                       bbox={"facecolor": "#101820", "alpha": .75, "edgecolor": "none", "pad": 1})
        state = ("both shells" if row["top"] and row["bottom"] else
                 "top shell only" if row["top"] else "bottom shell only")
        if row["alpha_deg"] is None:
            state = "NO CURRENT SPIN AXIS"
        else:
            state += f" | relative roll {row['alpha_deg']:.1f} deg"
        panel.set_title(f"{row['time']:.3f} s | frame {i}\n{state}",
                        color="white" if row["alpha_deg"] is not None else "#ffc670", fontsize=10)
        panel.set_xlim(0, row["size"][0])
        panel.set_ylim(row["size"][1], 0)
        panel.set_axis_off()
    fig.suptitle("Current mechanical axes across the video", fontsize=20, color="white", y=.98)
    fig.text(.5, .936, "GREEN X: fixed calibrated roll axis    ORANGE Z: current spin axis    BLUE Y: current transverse axis",
             ha="center", color="#d9e5ec", fontsize=10)
    fig.text(.5, .035, "Cyan / magenta grids: accepted top / bottom poses. Missing motion is not filled. Times use saved source timestamps.",
             ha="center", color="#d9e5ec", fontsize=10)
    offset = float(metadata.get("initial_roll_deg", 0.0))
    geometry_note = (f"Saved model: {geometry.replace('_', ' ')}, gap / diameter {gap:g}, "
                     f"initial roll offset {offset:g} deg. "
                     + ("Pivot uses the original circle seed." if model.get("pivot_camera") is None
                        else "Pivot uses the saved calibration."))
    fig.text(.5, .014, geometry_note,
             ha="center", color="#aabac7", fontsize=9)
    fig.subplots_adjust(left=.015, right=.985, top=.885, bottom=.075, hspace=.24, wspace=.06)
    png = out / "axes_at_times.png"
    fig.savefig(png, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    data = [samples[i] for i in sorted(samples)]
    template = Path(__file__).with_name("visualize_axes.html").read_text(encoding="utf-8")
    html = out / "axes_timeline.html"
    html.write_text(template.replace("__AXIS_DATA__", json.dumps(data, allow_nan=False))
                   .replace("__AXIS_COLORS__", json.dumps(COLORS))
                   .replace("__GEOMETRY_NOTE__", geometry_note)
                   .replace("__INITIAL_ROLL__", f"{offset:g}"), encoding="utf-8")
    manifest = {"results": str(args.results.resolve()), "source_clip": metadata["input"],
                "model": model, "initial_frame": F.tolist(), "pivot": C.tolist(),
                "initial_roll_deg": metadata.get("initial_roll_deg"),
                "sample_frames": [r["frame"] for r in data], "contact_sheet_frames": montage,
                "last_supported_frame": last if len(accepted) else None,
                "last_supported_time_s": float(times[last]) if len(accepted) else None,
                "definition": "Current frame F @ Rx(alpha); X is fixed; Y/Z hidden unless either shell supports roll.",
                "outputs": [str(png.resolve()), str(html.resolve())]}
    (out / "axis_inspection_report.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"Contact sheet: {png}\nInteractive timeline: {html}\nSamples: {len(data)}")


if __name__ == "__main__":
    main()
