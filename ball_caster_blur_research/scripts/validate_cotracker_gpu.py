"""Quantify learned tracking on rendered, separated textured hemispheres.

This is a controlled synthetic 3D observation test, not validation of the real
cameras, calibration, or complete reconstruction pipeline. Run --render-only
without a GPU. The normal run loads only the local CoTracker checkout/weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def summary(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "median": None, "p90": None, "p95": None, "rms": None, "max": None}
    return {"count": int(len(values)), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
            "rms": float(np.sqrt(np.mean(values ** 2))), "max": float(values.max())}


class HemisphereScene:
    """Pinhole camera and radius-one hemispheres; gap is in radius units."""

    def __init__(self, size=512, seed=218, gap=.2):
        self.size = size
        self.radius = 1.
        self.gap = float(gap)
        self.focal = size * 1.31
        self.principal = np.array([(size - 1) / 2] * 2)
        self.pivot = np.array([0., 0., 4.5])
        tilt = np.deg2rad(8.)
        rx = np.array([[1., 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
        # Home shell axis is horizontal in the camera; tilt is known here.
        self.carrier = np.array([[0., 0, 1], [0, 1, 0], [-1, 0, 0]]) @ rx
        self.signs = np.array([1., -1.])
        self.centers = self.pivot + (self.signs[:, None] * np.array([0., 0., self.gap / 2])) @ self.carrier.T
        yy, xx = np.indices((size, size))
        self.pixel_grid = np.stack((xx, yy), axis=-1).reshape(-1, 2)
        self.texture = [self._texture(seed + i, i) for i in range(2)]
        self.owner, self.normal, self.depth, self.incidence = self.intersections(self.pixel_grid)

    @staticmethod
    def _texture(seed, shell):
        rng = np.random.default_rng(seed)
        tex = np.full((512, 1024, 3), [232, 230, 218], np.uint8)
        ink = (172, 32, 45) if shell == 0 else (30, 128, 58)
        for _ in range(700):
            x, y = rng.integers([0, 0], [1024, 512])
            width, height = rng.integers(3, 13, 2)
            color = ink if rng.random() < .8 else (28, 34, 31)
            cv2.rectangle(tex, (int(x), int(y)), (int(x + width), int(y + height)), color, -1)
        return tex

    def _yoke(self, pixels):
        x, y = np.asarray(pixels).T
        return (np.abs(x - self.principal[0]) < self.size * .022) & (y < self.principal[1] + self.size * .018)

    def rays(self, pixels):
        xy = (np.asarray(pixels) - self.principal) / self.focal
        ray = np.column_stack((xy, np.ones(len(xy))))
        return ray / np.linalg.norm(ray, axis=1, keepdims=True)

    def intersections(self, pixels):
        """Nearest front-facing outer surface; gap/interior is not a surface."""
        rays = self.rays(pixels)
        count = len(rays)
        depth = np.full(count, np.inf)
        owner = np.full(count, -1, np.int8)
        normal = np.zeros((count, 3))
        incidence = np.zeros(count)
        for shell in range(2):
            center = self.centers[shell]
            projection = rays @ center
            discriminant = projection ** 2 - (center @ center - self.radius ** 2)
            distance = projection - np.sqrt(np.maximum(discriminant, 0))
            normal_camera = (rays * distance[:, None] - center) / self.radius
            normal_carrier = normal_camera @ self.carrier
            valid = (discriminant > 0) & (distance > 0) & (normal_carrier[:, 2] * self.signs[shell] > 0) & (distance < depth)
            depth[valid] = distance[valid]
            owner[valid] = shell
            normal[valid] = normal_carrier[valid]
            incidence[valid] = -np.sum(normal_camera[valid] * rays[valid], axis=1)
        owner[self._yoke(pixels)] = -1
        return owner, normal, depth, incidence

    @staticmethod
    def phases(time_s, speed_deg_s):
        return np.deg2rad(speed_deg_s) * time_s * np.array([1., -.73])

    def render(self, time_s, speed_deg_s):
        image = np.full((self.size * self.size, 3), [62, 68, 77], np.uint8)
        phases = self.phases(time_s, speed_deg_s)
        for shell in range(2):
            choose = self.owner == shell
            material = self.normal[choose] @ rz(phases[shell])
            u = np.mod(np.arctan2(material[:, 1], material[:, 0]), 2 * np.pi) / (2 * np.pi) * 1024
            v = (1 - material[:, 2]) * .5 * 511
            # OpenCV remap limits each output dimension to 32767; pack samples
            # into narrow rows instead of a potentially 40000-pixel column.
            pad = (-len(u)) % 64
            map_u = np.pad(u.astype(np.float32), (0, pad), mode="edge").reshape(-1, 64)
            map_v = np.pad(v.astype(np.float32), (0, pad), mode="edge").reshape(-1, 64)
            texels = cv2.remap(self.texture[shell], map_u, map_v, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_WRAP).reshape(-1, 3)[:len(u)]
            shade = .72 + .28 * self.incidence[choose]
            image[choose] = np.clip(texels * shade[:, None], 0, 255).astype(np.uint8)
        image[self._yoke(self.pixel_grid)] = [38, 42, 45]
        return image.reshape(self.size, self.size, 3)

    def exposure(self, time_s, speed_deg_s, exposure_s, samples=9):
        if exposure_s == 0:
            return self.render(time_s, speed_deg_s)
        nodes, weights = np.polynomial.legendre.leggauss(samples)
        result = np.zeros((self.size, self.size, 3), np.float64)
        for node, weight in zip(nodes, weights):
            result += self.render(time_s + node * exposure_s / 2, speed_deg_s) * (weight / 2)
        return np.rint(np.clip(result, 0, 255)).astype(np.uint8)

    def queries(self, per_shell):
        image = self.render(0., 0.)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        coordinates, labels, materials = [], [], []
        for shell in range(2):
            good = ((self.owner == shell) & (self.incidence > .4) &
                    (np.linalg.norm(self.normal[:, :2], axis=1) > .3) & (np.abs(self.normal[:, 2]) > .1))
            mask = good.reshape(self.size, self.size).astype(np.uint8) * 255
            mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
            points = cv2.goodFeaturesToTrack(gray, maxCorners=per_shell, qualityLevel=.015, minDistance=7, mask=mask)
            if points is None or len(points) < 12:
                raise RuntimeError("Synthetic texture did not yield enough material queries")
            xy = points[:, 0].astype(np.float64)
            owner, normals, _, _ = self.intersections(xy)
            assert np.all(owner == shell)
            coordinates.extend(xy)
            labels.extend([shell] * len(xy))
            materials.extend(normals)
        return np.asarray(coordinates), np.asarray(labels, np.int8), np.asarray(materials)

    def truth(self, time_s, speed_deg_s, labels, materials):
        phases = self.phases(time_s, speed_deg_s)
        camera = np.empty_like(materials)
        for shell in range(2):
            choose = labels == shell
            camera[choose] = (materials[choose] @ rz(phases[shell]).T) @ self.carrier.T + self.centers[shell]
        pixels = camera[:, :2] / camera[:, 2:] * self.focal + self.principal
        owner, _, distance, _ = self.intersections(pixels)
        visible = (owner == labels) & (np.abs(distance - np.linalg.norm(camera, axis=1)) < 1e-6)
        visible &= (pixels[:, 0] >= 1) & (pixels[:, 0] < self.size - 1) & (pixels[:, 1] >= 1) & (pixels[:, 1] < self.size - 1)
        return pixels, visible


def conditional_rate_metrics(estimated_phase_rad, true_phase_rad, times_s):
    """Compare interval-average spin rates without crossing unsupported gaps.

    Estimates may be wrapped. Signed shortest-angle increments are unambiguous
    only when the true phase change in one sampled interval is strictly below
    180 degrees. The known synthetic truth verifies that assumption; aliased
    intervals are excluded and counted. Frame zero is the supplied query, so
    the interval from frame zero to frame one is always excluded as well.
    """
    angle = np.asarray(estimated_phase_rad, dtype=float)
    truth = np.asarray(true_phase_rad, dtype=float)
    times = np.asarray(times_s, dtype=float)
    if angle.ndim != 1 or angle.shape != truth.shape or angle.shape != times.shape:
        raise ValueError("phase and time arrays must have the same one-dimensional shape")
    if not np.isfinite(truth).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("truth must be finite and times strictly increasing")
    dt = np.diff(times)
    delta_truth = np.diff(truth)
    candidate = np.arange(len(dt)) >= 1
    supported = candidate & np.isfinite(angle[:-1]) & np.isfinite(angle[1:])
    unaliased = np.abs(delta_truth) < np.pi
    scored = supported & unaliased
    estimated_rate = np.rad2deg(wrap(np.diff(angle))[scored]) / dt[scored]
    true_rate = np.rad2deg(delta_truth[scored]) / dt[scored]
    error = estimated_rate - true_rate
    absolute = np.abs(error)
    return {
        "definition": "Interval-average signed shortest-angle increment / actual time difference; same supported intervals in truth and prediction",
        "candidate_intervals_excluding_query": int(candidate.sum()),
        "consecutive_supported_intervals": int(supported.sum()),
        "scored_intervals": int(scored.sum()),
        "fraction_candidate_intervals_scored": float(scored.sum() / max(1, candidate.sum())),
        "strictly_less_than_180deg_true_increment_assumption": bool(np.all(unaliased[candidate])),
        "max_abs_true_increment_deg": float(np.rad2deg(np.abs(delta_truth[candidate])).max()) if candidate.any() else None,
        "supported_intervals_rejected_for_aliasing": int((supported & ~unaliased).sum()),
        "signed_error_median": float(np.median(error)) if len(error) else None,
        "absolute_error_median": float(np.median(absolute)) if len(error) else None,
        "mae": float(np.mean(absolute)) if len(error) else None,
        "rmse": float(np.sqrt(np.mean(error ** 2))) if len(error) else None,
        "absolute_error_p95": float(np.percentile(absolute, 95)) if len(error) else None,
    }


def evaluate(scene, tracks, visibility, confidence, truth, true_visible, labels, materials, phases, times_s):
    error = np.linalg.norm(tracks - truth, axis=-1)
    evaluated = np.ones(true_visible.shape, bool)
    evaluated[0] = False  # All queries are supplied in frame zero.
    visible = true_visible & evaluated
    predicted_visible = (visibility >= .7) & (visibility * confidence >= .6)
    accepted = visible & predicted_visible
    angles = np.full((len(tracks), 2), np.nan)
    supports = np.zeros_like(angles, dtype=int)
    for frame in range(1, len(tracks)):
        owners, normals, _, incidence = scene.intersections(tracks[frame])
        for shell in range(2):
            use = (accepted[frame] & (labels == shell) & (owners == shell) & (incidence > .25) &
                   (np.linalg.norm(normals[:, :2], axis=1) > .2) & (np.abs(normals[:, 2] - materials[:, 2]) < .12))
            if use.sum() < 6:
                continue
            values = wrap(np.arctan2(normals[use, 1], normals[use, 0]) - np.arctan2(materials[use, 1], materials[use, 0]))
            weight = visibility[frame, use] * confidence[frame, use]
            center = np.angle(np.sum(weight * np.exp(1j * values)))
            inlier = np.abs(wrap(values - center)) < np.deg2rad(8.)
            if inlier.sum() < 6:
                continue
            angles[frame, shell] = np.angle(np.sum(weight[inlier] * np.exp(1j * values[inlier])))
            supports[frame, shell] = int(inlier.sum())
    hidden = ~true_visible & evaluated
    # A visible observation after any earlier occluded exposure tests whether
    # the original query identity survived/reappeared, rather than only flow.
    reappeared = visible & (np.cumsum(hidden, axis=0) > 0)
    result = {
        "all_true_visible_pixel_error": summary(error[visible]),
        "score_accepted_true_visible_pixel_error": summary(error[accepted]),
        "true_visible_score_coverage": float(accepted.sum() / max(1, visible.sum())),
        "true_visible_within_3px_fraction": float(((error <= 3) & visible).sum() / max(1, visible.sum())),
        "score_accepted_within_3px_fraction": float(((error <= 3) & accepted).sum() / max(1, accepted.sum())),
        "predicted_visible_fraction_of_not_fully_visible": float((predicted_visible & hidden).sum() / max(1, hidden.sum())),
        "not_fully_visible_sample_count": int(hidden.sum()),
        "reappeared_true_visible_pixel_error": summary(error[reappeared]),
        "reappeared_score_accepted_pixel_error": summary(error[reappeared & accepted]),
        "reappeared_score_accepted_within_3px_recall": float(((error <= 3) & accepted & reappeared).sum() / max(1, reappeared.sum())),
        "query_samples_excluded": int(len(labels)),
        "score_gate": {"visibility_min": .7, "visibility_times_confidence_min": .6},
        "conditional_spin_error_deg": {},
        "conditional_interval_angular_rate_error_deg_s": {},
    }
    for shell, name in enumerate(("red", "green")):
        valid = np.isfinite(angles[:, shell])
        result["conditional_spin_error_deg"][name] = summary(np.abs(np.rad2deg(wrap(angles[valid, shell] - phases[valid, shell]))))
        result["conditional_spin_error_deg"][name]["supported_fraction_excluding_query"] = float(valid[1:].mean())
        result["conditional_interval_angular_rate_error_deg_s"][name] = conditional_rate_metrics(angles[:, shell], phases[:, shell], times_s)
    return result, angles, supports


def write_video(path, frames, fps, tracks=None, true_visible=None, truth=None):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames.shape[2], frames.shape[1]))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video {path}")
    try:
        for frame, rgb in enumerate(frames):
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if tracks is not None:
                for point in np.flatnonzero(true_visible[frame]):
                    true_xy = tuple(np.rint(truth[frame, point]).astype(int))
                    pred_xy = tuple(np.rint(tracks[frame, point]).astype(int))
                    cv2.circle(bgr, true_xy, 2, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.line(bgr, true_xy, pred_xy, (0, 160, 255), 1, cv2.LINE_AA)
                    cv2.circle(bgr, pred_xy, 1, (0, 160, 255), -1)
            writer.write(bgr)
    finally:
        writer.release()


def write_report(output, report):
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if report["render_only"]:
        return
    geometry = report["geometry"]
    lines = ["# Synthetic CoTracker3 ground-truth validation", "", report["purpose"], "",
             f"Geometry uses normalized radius {geometry['radius']:g} and gap {geometry['gap']:g}. At a 100 mm physical radius this corresponds to a {100 * geometry['gap']:g} mm gap. The camera and texture are synthetic; the known carrier tilt is 8 degrees.", "",
             f"All {report['queries']} material queries originate in frame zero. This tests preservation and reacquisition of those identities; the real collector's later query anchors and second camera are not simulated.", "",
             "White circles in tracking videos mark true visible material locations; orange marks are predictions. Query-frame samples are excluded. Positions refer to mid-exposure time. A point enters pixel-error statistics only when visible at all nine exposure samples.", "",
             "| Motion | Exposure ms | All-visible median / p95 px | Score-accepted p95 px | Score coverage | Red / green spin median deg |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for case in report["cases"]:
        e = case["all_true_visible_pixel_error"]
        accepted = case["score_accepted_true_visible_pixel_error"]
        a = case["conditional_spin_error_deg"]
        red = a["red"]["median"]
        green = a["green"]["median"]
        red_s = "unavailable" if red is None else f"{red:.3f}"
        green_s = "unavailable" if green is None else f"{green:.3f}"
        lines.append(f"| {case['name']} | {case['exposure_ms']:g} | {e['median']:.3f} / {e['p95']:.3f} | {accepted['p95']:.3f} | {case['true_visible_score_coverage']:.1%} | {red_s} / {green_s} |")
    lines += ["", "Reappearance is evaluated only when an original query becomes visible again after a previous exposure was not fully visible:", "",
              "| Case | Reappeared samples | Reappeared error p95 px | Reappeared accepted + within 3px recall |", "| --- | ---: | ---: | ---: |"]
    for case in report["cases"]:
        e = case["reappeared_true_visible_pixel_error"]
        p95 = "unavailable" if e["p95"] is None else f"{e['p95']:.2f}"
        recall = "unavailable" if not e["count"] else f"{case['reappeared_score_accepted_within_3px_recall']:.1%}"
        lines.append(f"| {case['name']} | {e['count']} | {p95} | {recall} |")
    lines += ["", "Conditional angle support (known carrier/home points, oracle visibility, at least six geometrically consistent model observations):", "",
              "| Case | Red supported frames | Green supported frames |", "| --- | ---: | ---: |"]
    for case in report["cases"]:
        a = case["conditional_spin_error_deg"]
        lines.append(f"| {case['name']} | {a['red']['supported_fraction_excluding_query']:.1%} | {a['green']['supported_fraction_excluding_query']:.1%} |")
    lines += ["", "Conditional angular-rate error in degrees/second. These are averages over individual frame intervals, not instantaneous angular velocities. Only adjacent frames with supported phase estimates are scored; no interval spans a missing estimate or touches query frame zero. Prediction and truth use the same intervals. Signed shortest-angle increments require a true change strictly below 180 degrees per interval, verified for these synthetic motions. Turn count remains untested.", "",
              "| Case | Shell | Median absolute error | MAE | RMSE | p95 absolute error | Candidate intervals scored |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for case in report["cases"]:
        for shell in ("red", "green"):
            rates = case["conditional_interval_angular_rate_error_deg_s"][shell]
            values = ["unavailable" if rates[key] is None else f"{rates[key]:.3f}" for key in ("absolute_error_median", "mae", "rmse", "absolute_error_p95")]
            lines.append(f"| {case['name']} | {shell} | " + " | ".join(values) + f" | {rates['scored_intervals']}/{rates['candidate_intervals_excluding_query']} ({rates['fraction_candidate_intervals_scored']:.1%}) |")
    if any((case["all_true_visible_pixel_error"]["p95"] or 0) > 10 for case in report["cases"]):
        lines += ["", "Some cases contain major identity/localization failures despite small median errors. Small median error therefore does not establish reliable correspondence through the complete sequence."]
    if any((case["score_accepted_true_visible_pixel_error"]["max"] or 0) > 15 for case in report["cases"]):
        lines += ["", "Some high-score predictions are still badly wrong. A geometrically checked fit and explicit unsupported intervals remain necessary."]
    lines += ["", "This single texture and motion profile do not establish a general exposure/error relationship. Blur can also reduce sampling aliasing; compare both error and correspondence coverage, and test real controls separately.", "",
              "Limitations:", ""] + [f"- {text}" for text in report["limitations"]]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/gpu_validation_rig_gap")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=float, default=30.)
    parser.add_argument("--points-per-shell", type=int, default=60)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--gap-over-radius", type=float, default=.2, help="Normalized shell-center gap; actual rig 20mm/100mm = 0.2")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--metrics-only", action="store_true", help="Recompute metrics from saved predictions; no GPU or model loading")
    parser.add_argument("--repo", type=Path, default=ROOT / "third_party/co-tracker")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/scaled_offline.pth")
    args = parser.parse_args()
    if args.frames < 3 or args.fps <= 0 or args.points_per_shell < 12 or args.size < 256 or not 0 < args.gap_over_radius < 1:
        parser.error("Need frames>=3, fps>0, points-per-shell>=12, size>=256, 0<gap-over-radius<1")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.metrics_only:
        report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
        if report["render_only"]:
            parser.error("Saved run has no learned predictions")
        scene = HemisphereScene(report["geometry"]["image_size"], gap=report["geometry"]["gap"])
        for case in report["cases"]:
            with np.load(args.output / f"{case['name']}.npz") as cached:
                metrics, _, _ = evaluate(scene, cached["tracks"], cached["visibility"], cached["confidence"],
                                         cached["true_tracks"], cached["true_visible"], cached["labels"],
                                         cached["material_normals"], cached["true_phases_rad"], cached["times_s"])
            case.pop("false_visible_fraction_of_hidden", None)
            case.pop("hidden_sample_count", None)
            case.update(metrics)
        write_report(args.output, report)
        print("Recomputed saved metrics without loading a GPU model.")
        return
    scene = HemisphereScene(args.size, gap=args.gap_over_radius)
    query_xy, labels, materials = scene.queries(args.points_per_shell)
    queries = np.column_stack((np.zeros(len(query_xy)), query_xy)).astype(np.float32)
    times = np.arange(args.frames) / args.fps
    backend = None
    if not args.render_only:
        from blurtrack.cotracker_backend import OfflineCoTracker
        backend = OfflineCoTracker(args.repo, args.checkpoint, feature_chunk_size=16)
    report = {
        "purpose": "Controlled learned 2D material tracking on known separated 3D hemispheres; not real-camera or end-to-end pose accuracy",
        "limitations": ["Known ideal pinhole geometry and home material coordinates; no calibration estimation",
                        "Known constant 8-degree carrier tilt; spin fits condition on this true carrier",
                        "Spin fit uses oracle true visibility plus model score gates; it tests conditional angle precision",
                        "Angular errors are modulo 360 degrees; this does not validate recovered turn count",
                        "Synthetic ink texture, noise-free exposure, and static foreground yoke differ from real camera images",
                        "Rendered RGB values are treated as linear intensity; no camera response, sensor noise or rolling shutter",
                        "Inference uses uncompressed rendered RGB; exported MP4s are visual previews"],
        "geometry": {"radius": scene.radius, "gap": scene.gap, "camera_center_depth": 4.5,
                     "focal_px": scene.focal, "carrier_tilt_deg": 8, "image_size": args.size,
                     "units": "Normalized by shell radius; gap/radius .2 matches 20mm gap /100mm radius. Camera and textures are synthetic, not fitted real-camera geometry."},
        "fps": args.fps, "frames": args.frames, "queries": len(queries),
        "exposure_integration": "Nine-point Gauss-Legendre quadrature; material visibility must hold at every sampled exposure time",
        "cases": [], "render_only": args.render_only,
    }
    previews = []
    for mode, speed in (("slow", 45.), ("fast", 360.)):
        truth, center_visibility = zip(*(scene.truth(t, speed, labels, materials) for t in times))
        truth = np.asarray(truth)
        phases = np.array([scene.phases(t, speed) for t in times])
        for exposure_ms in (0., 8., 15.6):
            name = f"{mode}_{str(exposure_ms).replace('.', 'p')}ms"
            started = time.perf_counter()
            frames = np.array([scene.exposure(t, speed, exposure_ms / 1000) for t in times])
            true_visible = np.array(center_visibility)
            if exposure_ms:
                nodes, _ = np.polynomial.legendre.leggauss(9)
                for frame, t in enumerate(times):
                    for node in nodes:
                        _, visible = scene.truth(t + node * exposure_ms / 2000, speed, labels, materials)
                        true_visible[frame] &= visible
            # Verify the independent ray/phase computation using ideal pixels.
            ideal, _, _ = evaluate(scene, truth, np.ones(truth.shape[:2]), np.ones(truth.shape[:2]),
                                    truth, true_visible, labels, materials, phases, times)
            for shell in ("red", "green"):
                if ideal["conditional_spin_error_deg"][shell]["count"]:
                    assert ideal["conditional_spin_error_deg"][shell]["max"] < 1e-4
            write_video(args.output / f"{name}_source.mp4", frames, args.fps)
            cv2.imwrite(str(args.output / f"{name}_preview.jpg"), cv2.cvtColor(frames[args.frames // 2], cv2.COLOR_RGB2BGR))
            tile = cv2.cvtColor(frames[args.frames // 2], cv2.COLOR_RGB2BGR)
            cv2.putText(tile, f"{mode}: {speed:g} deg/s, {exposure_ms:g} ms", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
            previews.append(tile)
            case = {"name": name, "red_speed_deg_s": speed, "green_speed_deg_s": -.73 * speed,
                    "exposure_ms": exposure_ms, "render_seconds": time.perf_counter() - started,
                    "geometry_selfcheck_max_error_deg": max(ideal["conditional_spin_error_deg"][s]["max"] or 0 for s in ("red", "green"))}
            arrays = dict(times_s=times, queries=queries, labels=labels, material_normals=materials,
                          true_tracks=truth, true_visible=true_visible, true_phases_rad=phases)
            if backend is not None:
                predicted = backend.predict(frames, queries, backward=False)
                metrics, estimated_angles, supports = evaluate(scene, predicted["tracks"], predicted["visibility"],
                                                               predicted["confidence"], truth, true_visible, labels, materials, phases, times)
                case.update(metrics)
                case["inference"] = predicted["metrics"]
                arrays.update(tracks=predicted["tracks"], visibility=predicted["visibility"], confidence=predicted["confidence"],
                              estimated_phases_rad=estimated_angles, phase_support=supports)
                write_video(args.output / f"{name}_tracking.mp4", frames, args.fps, predicted["tracks"], true_visible, truth)
            np.savez_compressed(args.output / f"{name}.npz", **arrays)
            report["cases"].append(case)
            write_report(args.output, report)
            print(json.dumps(case, allow_nan=False), flush=True)
    cv2.imwrite(str(args.output / "exposure_comparison.jpg"), np.concatenate((np.concatenate(previews[:3], axis=1), np.concatenate(previews[3:], axis=1)), axis=0))
    write_report(args.output, report)


if __name__ == "__main__":
    main()
