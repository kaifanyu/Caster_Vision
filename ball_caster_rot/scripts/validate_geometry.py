#!/usr/bin/env python
"""Test separated-shell geometry across saved image correspondences and poses.

The fit uses other target frames and persistent landmark IDs than the held-out
score. Saved visual camera poses are re-decomposed for every candidate axis
frame; they are conditional inputs, NOT known true angles. The original config
is never changed. A separate candidate_config.yaml is written only when all
support, physical, observability and held-out improvement gates pass. Even then
it requires a fresh pixel refit and repeated held-out evaluation before use.
"""
from __future__ import annotations

import argparse
import copy
import html
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import yaml

from ballrot.camera import undistort_image
from ballrot.config import distortion_coefficients, load_config, measurement_frame, resolve_from_config
from ballrot.diagnostics import _json_clean
from ballrot.geometry_validation import TransferDataset, build_transfer_dataset, quality_pose_flags, validate_geometry
from ballrot.io_frames import FrameSource
from ballrot.rotation import Rx
from scripts.refine_mechanical import _same_geometry, load_archive


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--results", type=Path, required=True, help="Source results with preserved raw camera poses")
    result.add_argument("--observations", type=Path, help="Defaults to offline_observations.npz beside results")
    result.add_argument("--output", type=Path, required=True, help="Separate candidate/diagnostic directory")
    result.add_argument("--views", type=int, default=18, help="Approximate total target views across both shells (>=12)")
    result.add_argument("--samples-per-view", type=int, default=40)
    result.add_argument("--max-nfev", type=int, default=100)
    result.add_argument("--pose-source", choices=("unconstrained", "mechanical"), default="unconstrained",
                        help="Preserved camera poses, or accepted mechanical poses for verification after a fresh pixel refit")
    result.add_argument("--evidence", type=Path, help="Reuse the EXACT transfers/holdout partition from an earlier report; camera, clip and timestamps must match")
    result.add_argument("--evaluate-only", action="store_true", help="Evaluate the supplied geometry/poses without fitting or writing a candidate config")
    result.add_argument("--no-previews", action="store_true", help="Numerical report only; omit footage inspection images")
    return result


def _candidate_config(config, config_path, output, report):
    if not report["candidate_config_permitted"]:
        return None
    candidate = copy.deepcopy(config)
    geometry = report["candidate_geometry"]
    candidate.setdefault("mechanical", {}).update(
        pivot_camera=geometry["pivot_camera"], gap_fraction=geometry["gap_fraction"])
    offset = float(candidate["frame_calib"].get("initial_roll_deg", 0.))
    candidate["frame_calib"]["R_bc"] = (np.asarray(geometry["measurement_frame"]) @ Rx(-np.deg2rad(offset))).tolist()
    candidate["input"]["path"] = str(resolve_from_config(config_path, config["input"]["path"]))
    candidate.setdefault("output", {})["dir"] = str(output / "candidate_run")
    destination = output / "candidate_config.yaml"
    text = ("# EXPERIMENTAL conditional geometry candidate. Original config was not changed.\n"
            "# Requires a fresh pixel refit and held-out verification; not independent angle calibration.\n"
            + yaml.safe_dump(candidate, sort_keys=False))
    destination.write_text(text, encoding="utf-8")
    return str(destination)


def _reuse_evidence(path, archive, metadata, dist, rotations, validity, frame):
    evidence = json.loads(Path(path).read_text(encoding="utf-8"))
    if (evidence.get("coordinate_system") != "undistorted_pixels"
            or evidence.get("method") != "observed_source_ray_to_heldout_target_pixel_transfer"
            or Path(evidence["source_clip"]).resolve() != Path(metadata["input"]).resolve()
            or not np.allclose(evidence["K"], archive.K, atol=1e-8, rtol=0)
            or np.shape(evidence["dist"]) != np.shape(dist)
            or not np.allclose(evidence["dist"], dist, atol=1e-8, rtol=0)):
        raise ValueError("saved transfer evidence must use the same clip, undistorted camera and coordinate system")
    samples = []
    keys = ("shell", "track_id", "source_frame", "target_frame", "source_uv", "observed_uv", "split", "baseline_motion_deg")
    for supplied in evidence["samples"]:
        row = {key: supplied[key] for key in keys}
        for key in ("source_frame", "target_frame"):
            index = row[key]
            if (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < archive.frame_count
                    or not np.isclose(evidence["endpoint_times_s"][str(index)], archive.timestamps[index], atol=1e-8, rtol=0)):
                raise ValueError("fixed validation evidence requires matching endpoint frame indices and timestamps")
            if not validity[row["shell"]][index]:
                raise ValueError("fixed evidence endpoint has no valid new source pose; cannot silently discard it from validation")
        samples.append(row)
    selection = {**evidence["selection"], "reused_evidence": str(Path(path).resolve()),
                 "identity_note": "Fixed observed pixel pairs retain IDs from their original archive, not identities remapped to a new tracker run."}
    usable = quality_pose_flags(rotations, validity, frame)
    return TransferDataset(samples, rotations, usable, selection, raw_valid=validity), evidence


def _panel(image, rows, variant, index, timestamp, crop, label=None):
    x0, y0, x1, y1 = crop
    panel = image[y0:y1, x0:x1].copy()
    scale = 720 / panel.shape[1]
    panel = cv2.resize(panel, (720, int(round(panel.shape[0]*scale))))
    for row in rows:
        def point(values):
            xy = (np.asarray(values) - [x0, y0])*scale
            return tuple(np.clip(np.rint(xy), -100000, 100000).astype(int))
        observed = point(row["observed_uv"])
        if not row.get("source_pose_quality_supported", True):
            cv2.circle(panel, observed, 5, (20, 170, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, "Q", (observed[0]+5, observed[1]-4), cv2.FONT_HERSHEY_SIMPLEX,
                        .35, (20, 170, 255), 1, cv2.LINE_AA)
            continue
        predicted = point(row[variant+"_predicted_uv"])
        color = ((80, 235, 80) if row[variant+"_physical"] and row[variant+"_error_px"] <= 3.
                 else (60, 80, 245))
        cv2.circle(panel, observed, 4, (240, 230, 80), 1, cv2.LINE_AA)
        cv2.drawMarker(panel, predicted, color, cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
        cv2.arrowedLine(panel, observed, predicted, color, 1, cv2.LINE_AA, tipLength=.2)
        label = f"{row['shell'][0].upper()}{row['track_id']}"
        cv2.putText(panel, label, (observed[0]+5, observed[1]-4), cv2.FONT_HERSHEY_SIMPLEX, .28,
                    color, 1, cv2.LINE_AA)
    split = ", ".join(sorted({row["split"] for row in rows}))
    banner = np.zeros((118, 720, 3), np.uint8)
    lines = [f"{label or variant.upper()} | frame {index} | {timestamp:.3f} s | {split}",
             "Circle: observed target; cross/arrow: transferred source pixel",
             "Green <=3 px and valid shell geometry; red failed. IDs are saved tracks.",
             "Orange Q: source pose failed quality; transfer unavailable, counted as failure.",
             "Fixed visual-pose evidence; this is NOT known-angle validation."]
    for offset, line in enumerate(lines):
        cv2.putText(banner, line, (8, 19+22*offset), cv2.FONT_HERSHEY_SIMPLEX, .42,
                    (245, 245, 245), 1, cv2.LINE_AA)
    return np.vstack((banner, panel))


def write_previews(clip, K, dist, circle, times, report, output, earlier=None):
    wanted = sorted({row["target_frame"] for row in report["samples"]})
    by_frame = {index: [row for row in report["samples"] if row["target_frame"] == index]
                for index in wanted}
    previews, thumbs = [], []
    iterator = iter(FrameSource(clip, input_type="auto"))
    try:
        for record in iterator:
            if record.index not in by_frame:
                continue
            image = undistort_image(record.image, K, dist)
            u, v, radius = circle
            height, width = image.shape[:2]
            crop = (max(0, int(u-radius-65)), max(0, int(v-radius-65)),
                    min(width, int(u+radius+65)), min(height, int(v+radius+65)))
            if earlier is not None and report.get("evaluation_only"):
                previous = [row for row in earlier["samples"] if row["target_frame"] == record.index]
                panels = [_panel(image, previous, "baseline", record.index, times[record.index], crop,
                                 "EARLIER SOURCE"),
                          _panel(image, by_frame[record.index], "baseline", record.index, times[record.index], crop,
                                 "CURRENT SOURCE")]
            else:
                panels = [_panel(image, by_frame[record.index], variant, record.index,
                                 times[record.index], crop) for variant in ("baseline", "candidate")]
            composite = np.hstack(panels)
            destination = output / f"transfer_{record.index:06d}.png"
            if not cv2.imwrite(str(destination), composite):
                raise OSError(f"could not write {destination}")
            previews.append({"frame_index": record.index, "time_s": float(times[record.index]),
                             "path": destination.name})
            thumbs.append(cv2.resize(composite, (960, int(round(composite.shape[0]*960/composite.shape[1])))))
            if len(previews) == len(wanted):
                break
    finally:
        close = getattr(iterator, "close", None)
        if close:
            close()
    if len(previews) != len(wanted):
        raise ValueError("source footage ended before all sampled target frames")
    if thumbs:
        selected = np.linspace(0, len(thumbs)-1, min(6, len(thumbs))).round().astype(int)
        montage = np.vstack([thumbs[index] for index in selected])
        if not cv2.imwrite(str(output / "geometry_alignment_montage.png"), montage):
            raise OSError("could not write geometry montage")
    return previews


def _html(report, previews, output):
    def number(value):
        return "n/a" if value is None else f"{value:.3f}"
    rows = []
    comparison = report.get("earlier_evidence") if report.get("evaluation_only") else None
    old_metrics = comparison["baseline"] if comparison else report["baseline"]
    new_metrics = report["baseline"] if comparison else report["candidate"]
    old_label, new_label = ("Earlier source", "Current source") if comparison else ("Baseline", "Candidate")
    for name in ("training_top", "training_bottom", "heldout_top", "heldout_bottom"):
        old, new = old_metrics[name], new_metrics[name]
        rows.append(f"<tr><td>{name}</td><td>{old['count']}</td><td>{number(old['median_px'])}</td>"
                    f"<td>{number(new['median_px'])}</td><td>{number(old['p90_px'])}</td>"
                    f"<td>{number(new['p90_px'])}</td><td>{new['inlier_fraction_3px']:.1%}</td></tr>")
    images = "".join(f"<figure><a href='{html.escape(item['path'])}'><img loading='lazy' src='{html.escape(item['path'])}'></a>"
                     f"<figcaption>frame {item['frame_index']}, {item['time_s']:.3f} s</figcaption></figure>" for item in previews)
    content = f"""<!doctype html><html lang='en'><meta charset='utf-8'><title>Geometry transfer check</title>
<style>body{{font:16px system-ui;background:#10151d;color:#e5ebf3;margin:25px;max-width:1450px}}a{{color:#80c8ff}}
table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #546}}img{{width:100%}}figure{{margin:24px 0}}</style>
<h1>Conditional geometry transfer check: {html.escape(report['status'])}</h1>
<p>Observed source rays are projected into other image views. Training and held-out targets use different frames and landmark IDs.
The saved pose estimates came from this same footage, so these errors do not independently establish angular accuracy.</p>
<p>Original configuration unchanged. Rejection reasons: {html.escape(', '.join(report['rejection_reasons']) or 'none; a fresh refit remains required')}.</p>
<p>Unavailable source-quality support remains a failure in the full-set scores (at least 1000 px); it is not an observed pixel error. Orange Q marks these cases.</p>
<table><tr><th>Subset</th><th>Transfers</th><th>{old_label} median px</th><th>{new_label} median px</th><th>{old_label} p90 px</th><th>{new_label} p90 px</th><th>{new_label} inliers</th></tr>{''.join(rows)}</table>
<p><a href='geometry_validation_report.json'>Full report and pixel evidence</a></p>{images}</html>"""
    (output / "geometry_alignment.html").write_text(content, encoding="utf-8")


def main(argv=None):
    args = parser().parse_args(argv)
    if args.max_nfev < 1:
        raise ValueError("max-nfev must be positive")
    config, config_path = load_config(args.config)
    source = args.results.expanduser().resolve()
    archive_path = args.observations.expanduser().resolve() if args.observations else source.with_name("offline_observations.npz")
    output = args.output.expanduser().resolve()
    if output in (source.parent, archive_path.parent, config_path.parent):
        raise ValueError("use a separate output directory to preserve source artifacts and config")
    if args.evidence and output == args.evidence.expanduser().resolve().parent:
        raise ValueError("use a separate output directory to preserve the original fixed validation evidence")
    payload = json.loads(source.read_text(encoding="utf-8"))
    _same_geometry(config, config_path, payload["metadata"])
    archive = load_archive(archive_path, payload)
    rotations, validity = archive.unconstrained_rotations, archive.unconstrained_valid
    if args.pose_source == "mechanical":
        if payload.get("mechanical") is None:
            raise ValueError("mechanical pose source requires accepted mechanical results")
        # load_archive already verified these arrays against result angles/flags.
        with np.load(archive_path, allow_pickle=False) as arrays:
            rotations = {shell: arrays[shell+"_refined_rotations"].copy() for shell in ("top", "bottom")}
            validity = {shell: arrays[shell+"_refined_valid"].copy() for shell in ("top", "bottom")}
    model = config.get("mechanical", {})
    if model.get("geometry") != "separated_hemispheres":
        raise ValueError("this check requires mechanical.geometry: separated_hemispheres")
    frame = measurement_frame(config.get("frame_calib", {}))
    if frame is None:
        raise ValueError("frame_calib.R_bc is required")
    center = np.asarray(model.get("pivot_camera") if model.get("pivot_camera") is not None
                        else archive.center/archive.radius, float)
    gap = float(model.get("gap_fraction", .1))
    sign = config["frame_calib"].get("top_shell_sign", 1)
    dist = distortion_coefficients(config["camera"])
    earlier = None
    if args.evidence:
        print("Reusing fixed source/target pixels and held-out partition ...", flush=True)
        dataset, earlier = _reuse_evidence(args.evidence, archive, payload["metadata"], dist, rotations, validity, frame)
    else:
        print("Selecting source-valid persistent landmark transfers across the clip ...", flush=True)
        dataset = build_transfer_dataset(archive.observations, rotations, validity, archive.K, center, frame, gap,
                                         top_shell_sign=sign, view_count=args.views, max_per_view=args.samples_per_view)
    action = "Evaluating fixed geometry" if args.evaluate_only else "Fitting seven bounded geometry parameters"
    print(f"{action} from {len(dataset.samples)} sampled transfers ...", flush=True)
    report = validate_geometry(dataset, archive.K, center, frame, gap, top_shell_sign=sign,
                               max_nfev=args.max_nfev, fit_candidate=not args.evaluate_only)
    endpoints = {row[key] for row in dataset.samples for key in ("source_frame", "target_frame")}
    report.update(config=str(config_path), source_results=str(source), source_observations=str(archive_path),
                  source_clip=payload["metadata"]["input"], main_config_written=False,
                  source_pose_kind=("preserved_unconstrained_camera_rotations" if args.pose_source == "unconstrained"
                                    else "accepted_mechanical_camera_rotations"), requires_fresh_pixel_refit=True,
                  K=archive.K.tolist(), dist=dist.tolist(), coordinate_system="undistorted_pixels",
                  endpoint_times_s={str(index): float(archive.timestamps[index]) for index in sorted(endpoints)})
    if earlier is not None:
        report["earlier_evidence"] = {"report": str(args.evidence.resolve()),
                                      "source_results": earlier["source_results"],
                                      "baseline": earlier["baseline"], "candidate": earlier["candidate"]}
    report["transfer_observation_source"] = (earlier.get("transfer_observation_source", earlier["source_observations"])
                                               if earlier is not None else str(archive_path))
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = _candidate_config(config, config_path, output, report)
    report["candidate_config"] = candidate_path
    # A rejected rerun must never leave a previously passing config looking current.
    if candidate_path is None and (output / "candidate_config.yaml").exists():
        (output / "candidate_config.yaml").unlink()
    previews = []
    if not args.no_previews and report["samples"]:
        print("Rendering observed/predicted alignment on sampled real frames ...", flush=True)
        previews = write_previews(payload["metadata"]["input"], archive.K,
                                  dist, payload["metadata"]["circle"],
                                  archive.timestamps, report, output, earlier=earlier)
    report["previews"] = previews
    destination = output / "geometry_validation_report.json"
    destination.write_text(json.dumps(_json_clean(report), indent=2, allow_nan=False), encoding="utf-8")
    _html(report, previews, output)
    print(f"Geometry: {report['status']}")
    for name in ("training", "heldout"):
        old, new = report["baseline"][name], report["candidate"][name]
        print(f"  {name}: {old['count']} transfers; median pixels {old['median_px']} -> {new['median_px']}")
    print("Reasons: " + (", ".join(report["rejection_reasons"]) or "all conditional checks passed; fresh pixel refit required"))
    print(f"Report: {destination}")
    print("Original config unchanged. Pixel transfer consistency is not independent angular accuracy.")
    return 0 if report["status"] in ("candidate_passed_conditional_checks", "validation_passed_conditional_checks") else 2


if __name__ == "__main__":
    raise SystemExit(main())
