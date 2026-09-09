#!/usr/bin/env python
"""Replay measured angles as motion and check them against the real clip.

Three independent verifications are produced, weakest assumption first:

``axes``
    Draws the calibrated ball axes and a ball-fixed graticule on the real
    footage, rotated by the measured per-frame orientation.  If the measured
    rotation is right the graticule stays glued to the surface: a speckle that
    starts inside one cell stays inside it for the whole clip.  Sliding means
    the rotation, the circle, or ``K`` is wrong.  This test uses no synthetic
    model at all.

``replay``
    Renders the measured trajectory with the synthetic ball renderer using the
    real camera, circle, and ``R_bc``, beside the real clip.  Rendering the
    modeled ``Rx(alpha) @ Rz(beta)`` manifold and the full measured orientation
    separately shows how much of the motion the two-degree-of-freedom caster
    model actually captures.

``roundtrip``
    Re-measures the rendered sequence with the same pipeline and compares the
    recovered angles to the ones that were fed in.  This isolates estimator and
    convention error from real-world calibration error: a large round-trip
    error is a code or convention bug, while a small round-trip error alongside
    a bad ``axes`` overlay points at calibration or the motion model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ballrot.camera import undistort_image
from ballrot.config import (
    camera_matrix,
    configured_circle,
    configured_frame,
    distortion_coefficients,
    load_config,
    resolve_from_config,
)
from ballrot.io_frames import FrameSource
from ballrot.pipeline import run_pipeline
from ballrot.rotation import Rx, Rz
from ballrot.sphere import sphere_pose_from_circle
from synthetic.generate import CameraSetup, RenderConfig, Trajectory, render_sequence

TESTS = ("axes", "replay", "roundtrip")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Measured results.json (default: <output.dir>/results.json)",
    )
    parser.add_argument(
        "--clip",
        type=Path,
        default=None,
        help="Real footage to compare against (default: config input.path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <output.dir>/simulation)",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=TESTS,
        default=list(TESTS),
        help="Which verifications to run (default: all)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Use only the first N measured frames (default: all)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth frame; speeds up rendering on long clips",
    )
    parser.add_argument(
        "--graticule-step-deg",
        type=float,
        default=45.0,
        help="Spacing of the ball-fixed graticule drawn by the axes test",
    )
    parser.add_argument(
        "--num-speckles",
        type=int,
        default=1800,
        help="Speckle count for the synthetic replay (default: 1800)",
    )
    parser.add_argument(
        "--swap-shells",
        action="store_true",
        help=(
            "Render beta_top on the ball -z shell. The renderer paints the "
            "ball +z shell with the top colour, but nothing ties the top HSV "
            "class to ball +z: the sign of +z was fixed by --swivel-sign "
            "during axis calibration, not by which shell the class sits on. "
            "Use this when the replay's colours come out mirrored against the "
            "real clip."
        ),
    )
    parser.add_argument("--codec", default="mp4v", help="FourCC for written videos")
    return parser


# --------------------------------------------------------------------------
# geometry helpers


def _project(
    directions: np.ndarray, K: np.ndarray, C: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Project unit ball-surface normals to pixels; also return visibility."""

    normals = np.asarray(directions, dtype=float).reshape(-1, 3)
    points = C[None, :] + radius * normals
    toward_camera = -points
    toward_camera /= np.linalg.norm(toward_camera, axis=1)[:, None]
    visible = np.einsum("ij,ij->i", normals, toward_camera) > 0.05
    depth = points[:, 2]
    visible &= depth > 1e-9
    uv = np.full((len(normals), 2), np.nan)
    safe = visible & (np.abs(depth) > 1e-9)
    projected = points[safe] @ K.T
    uv[safe] = projected[:, :2] / projected[:, 2:3]
    return uv, visible


def _graticule(
    step_deg: float, hemisphere: int, samples: int = 60
) -> list[np.ndarray]:
    """Meridians and parallels about ball ``+z``, restricted to one hemisphere.

    ``hemisphere`` is ``+1`` for ball ``z >= 0`` (the "top" shell) and ``-1``
    for ``z <= 0``.  Each shell is an independently rotating rigid body, so
    drawing its graticule only over its own surface keeps the two overlays
    from covering each other.
    """

    curves: list[np.ndarray] = []
    step = np.deg2rad(float(step_deg))
    lo, hi = (0.0, np.pi / 2) if hemisphere >= 0 else (-np.pi / 2, 0.0)
    for longitude in np.arange(0.0, 2 * np.pi, step):
        t = np.linspace(lo, hi, samples)
        curves.append(
            np.column_stack(
                [
                    np.cos(t) * np.cos(longitude),
                    np.cos(t) * np.sin(longitude),
                    np.sin(t),
                ]
            )
        )
    latitudes = np.arange(lo, hi + 1e-9, step) if hemisphere >= 0 else np.arange(hi, lo - 1e-9, -step)
    for latitude in latitudes:
        if abs(abs(latitude) - np.pi / 2) < 1e-6:
            continue
        t = np.linspace(0.0, 2 * np.pi, samples * 2)
        curves.append(
            np.column_stack(
                [
                    np.cos(latitude) * np.cos(t),
                    np.cos(latitude) * np.sin(t),
                    np.full_like(t, np.sin(latitude)),
                ]
            )
        )
    return curves


def _probe_points(
    R_bc: np.ndarray, hemisphere: int, count: int = 14, seed: int = 12345
) -> np.ndarray:
    """Ball-fixed markers spread over the cap that faces the camera at t=0.

    These are the decisive check: each marker is pinned to one point of the
    physical shell, so if the measured rotation is right the marker stays on
    the same speckle for the whole clip.  A marker that slides off its speckle
    shows the error directly, in pixels, with no model in the loop.
    """

    rng = np.random.default_rng(seed)
    toward_camera_ball = R_bc.T @ np.array([0.0, 0.0, -1.0])
    points: list[np.ndarray] = []
    while len(points) < count:
        candidate = rng.normal(size=3)
        candidate /= np.linalg.norm(candidate)
        if np.sign(candidate[2]) != np.sign(hemisphere):
            continue
        if candidate @ toward_camera_ball < 0.45:
            continue
        points.append(candidate)
    return np.asarray(points)


def _draw_curve(
    image: np.ndarray,
    ball_points: np.ndarray,
    R_total: np.ndarray,
    K: np.ndarray,
    C: np.ndarray,
    radius: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    uv, visible = _project(ball_points @ R_total.T, K, C, radius)
    for start in range(len(uv) - 1):
        if not (visible[start] and visible[start + 1]):
            continue
        p0 = uv[start]
        p1 = uv[start + 1]
        if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
            continue
        cv2.line(
            image,
            tuple(np.rint(p0).astype(int)),
            tuple(np.rint(p1).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )


def _draw_axes(
    image: np.ndarray,
    R_bc: np.ndarray,
    K: np.ndarray,
    C: np.ndarray,
    radius: float,
    circle: tuple[float, float, float],
) -> None:
    """Draw the fixed calibrated ball axes as rays from the projected centre."""

    labels = (("+x roll", (60, 220, 60)), ("+y", (220, 220, 60)), ("+z swivel", (60, 120, 255)))
    origin = np.array([circle[0], circle[1]])
    for index, (label, color) in enumerate(labels):
        axis_camera = R_bc[:, index]
        tip = C + radius * 1.35 * axis_camera
        if tip[2] <= 1e-9:
            continue
        projected = K @ tip
        end = projected[:2] / projected[2]
        toward = axis_camera @ (-C / np.linalg.norm(C))
        thickness = 2 if toward >= 0 else 1
        cv2.arrowedLine(
            image,
            tuple(np.rint(origin).astype(int)),
            tuple(np.rint(end).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
            tipLength=0.06,
        )
        cv2.putText(
            image,
            label + ("" if toward >= 0 else " (behind)"),
            tuple(np.rint(end + np.array([6, -6])).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


# --------------------------------------------------------------------------
# data loading


def _measured(results_path: Path, frames: int | None, stride: int) -> dict[str, np.ndarray]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    series = payload["frames"]
    keys = (
        "time_s",
        "alpha_rad",
        "beta_top_rad",
        "beta_bottom_rad",
        "alpha_top_rad",
        "alpha_bottom_rad",
        "gamma_top_rad",
        "gamma_bottom_rad",
    )
    data = {key: np.asarray(series[key], dtype=float) for key in keys}
    count = len(data["time_s"])
    selected = np.arange(0, count if frames is None else min(count, frames), max(1, stride))
    output = {key: value[selected] for key, value in data.items()}
    output["frame_index"] = selected
    output["metadata"] = payload["metadata"]
    return output


def _filled(values: np.ndarray) -> np.ndarray:
    """Hold the last finite value across gaps so a render never breaks."""

    result = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(result)
    if not finite.any():
        return np.zeros_like(result)
    first = int(np.flatnonzero(finite)[0])
    result[:first] = result[first]
    for index in range(first + 1, len(result)):
        if not np.isfinite(result[index]):
            result[index] = result[index - 1]
    return result


def _writer(path: Path, fps: float, size: tuple[int, int], codec: str) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
    if not writer.isOpened():
        raise OSError(f"could not open video writer: {path}")
    return writer


def _panel(image: np.ndarray, title: str, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    resized = cv2.resize(image, (width, int(round(image.shape[0] * scale))))
    cv2.rectangle(resized, (0, 0), (width, 30), (0, 0, 0), -1)
    cv2.putText(
        resized, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
    )
    return resized


# --------------------------------------------------------------------------
# tests


def _axes_overlay(
    clip_path: Path,
    measured: dict[str, np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    circle: tuple[float, float, float],
    R_bc: np.ndarray,
    C: np.ndarray,
    radius: float,
    fps: float,
    destination: Path,
    step_deg: float,
    codec: str,
) -> Path:
    wanted = {int(value): position for position, value in enumerate(measured["frame_index"])}
    curves = {1: _graticule(step_deg, 1), -1: _graticule(step_deg, -1)}
    probes = {1: _probe_points(R_bc, 1), -1: _probe_points(R_bc, -1)}
    alpha_top = _filled(measured["alpha_top_rad"])
    alpha_bottom = _filled(measured["alpha_bottom_rad"])
    beta_top = _filled(measured["beta_top_rad"])
    beta_bottom = _filled(measured["beta_bottom_rad"])
    gamma_top = _filled(measured["gamma_top_rad"])
    gamma_bottom = _filled(measured["gamma_bottom_rad"])

    def ball_rotation(alpha: float, gamma: float, beta: float) -> np.ndarray:
        cos_g, sin_g = np.cos(gamma), np.sin(gamma)
        Ry = np.array([[cos_g, 0.0, sin_g], [0.0, 1.0, 0.0], [-sin_g, 0.0, cos_g]])
        return Rx(alpha) @ Ry @ Rz(beta)

    writer = None
    source = FrameSource(clip_path, input_type="auto")
    try:
        for record in source:
            position = wanted.get(int(record.index))
            if position is None:
                continue
            frame = undistort_image(record.image, K, dist)
            if writer is None:
                writer = _writer(
                    destination, fps, (frame.shape[1], frame.shape[0]), codec
                )
            cv2.circle(
                frame,
                (int(round(circle[0])), int(round(circle[1]))),
                int(round(circle[2])),
                (200, 200, 200),
                1,
            )
            for hemisphere, angles, color in (
                (
                    1,
                    (alpha_top[position], gamma_top[position], beta_top[position]),
                    (255, 220, 0),
                ),
                (
                    -1,
                    (
                        alpha_bottom[position],
                        gamma_bottom[position],
                        beta_bottom[position],
                    ),
                    (180, 0, 255),
                ),
            ):
                R_total = R_bc @ ball_rotation(*angles)
                for curve in curves[hemisphere]:
                    _draw_curve(frame, curve, R_total, K, C, radius, color)
                uv, visible = _project(probes[hemisphere] @ R_total.T, K, C, radius)
                for marker, (point, shown) in enumerate(zip(uv, visible)):
                    if not (shown and np.all(np.isfinite(point))):
                        continue
                    center = tuple(np.rint(point).astype(int))
                    cv2.circle(frame, center, 9, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.circle(frame, center, 9, color, 2, cv2.LINE_AA)
                    cv2.putText(
                        frame,
                        str(marker),
                        (center[0] + 11, center[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
            _draw_axes(frame, R_bc, K, C, radius, circle)
            cv2.putText(
                frame,
                f"frame {record.index}   cyan = top shell   magenta = bottom shell"
                "   -- each numbered ring should stay on its own speckle",
                (10, frame.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
    return destination


def _camera_setup(
    K: np.ndarray,
    R_bc: np.ndarray,
    circle: tuple[float, float, float],
    image_size: tuple[int, int],
) -> CameraSetup:
    C, radius = sphere_pose_from_circle(*circle, K)
    return CameraSetup(
        image_size=image_size,
        K=np.asarray(K, dtype=float),
        R_bc=np.asarray(R_bc, dtype=float),
        C=C,
        radius=radius,
        circle=circle,
        dist=np.zeros(5),
    )


def _trajectories(
    measured: dict[str, np.ndarray], fps: float, swap_shells: bool = False
) -> tuple[Trajectory, Trajectory]:
    alpha = _filled(measured["alpha_rad"])
    beta_top = _filled(measured["beta_top_rad"])
    beta_bottom = _filled(measured["beta_bottom_rad"])
    gamma_top = _filled(measured["gamma_top_rad"])
    gamma_bottom = _filled(measured["gamma_bottom_rad"])
    if swap_shells:
        beta_top, beta_bottom = beta_bottom, beta_top
        gamma_top, gamma_bottom = gamma_bottom, gamma_top
    model = Trajectory(alpha, beta_top, beta_bottom, fps=fps, name="measured_model")
    full = Trajectory(
        alpha,
        beta_top,
        beta_bottom,
        fps=fps,
        name="measured_full",
        gamma_top=gamma_top,
        gamma_bottom=gamma_bottom,
    )
    return model, full


def _replay_video(
    clip_path: Path,
    measured: dict[str, np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    rendered: dict[str, Sequence[np.ndarray]],
    fps: float,
    destination: Path,
    codec: str,
) -> Path:
    wanted = {int(value): position for position, value in enumerate(measured["frame_index"])}
    panel_width = 640
    writer = None
    source = FrameSource(clip_path, input_type="auto")
    try:
        for record in source:
            position = wanted.get(int(record.index))
            if position is None:
                continue
            real = undistort_image(record.image, K, dist)
            panels = [_panel(real, f"real  frame {record.index}", panel_width)]
            for name, frames in rendered.items():
                panels.append(_panel(frames[position], name, panel_width))
            composite = np.hstack(panels)
            if writer is None:
                writer = _writer(
                    destination, fps, (composite.shape[1], composite.shape[0]), codec
                )
            writer.write(composite)
    finally:
        if writer is not None:
            writer.release()
    return destination


# The Stage D characterization in out/synthetic/validation_report.json; a
# round trip run above this loses tracking and measures the sampling choice
# rather than the estimator.
DELTA_MAX_DEG_PER_FRAME = 5.0


def _step_size(trajectory: Trajectory) -> dict[str, Any]:
    """Per-frame rotation of the sequence that is about to be re-measured."""

    steps = []
    for name in ("top", "bottom"):
        beta = getattr(trajectory, f"beta_{name}")
        gamma = getattr(trajectory, f"gamma_{name}")
        angles = np.column_stack([trajectory.alpha, gamma, beta])
        rotations = Rotation.from_euler("XYZ", angles)
        steps.append(np.rad2deg((rotations[:-1].inv() * rotations[1:]).magnitude()))
    combined = np.concatenate(steps)
    return {
        "step_deg_per_frame_median": float(np.median(combined)),
        "step_deg_per_frame_p95": float(np.percentile(combined, 95)),
        "characterized_limit_deg_per_frame": DELTA_MAX_DEG_PER_FRAME,
        "within_characterized_limit": bool(
            np.percentile(combined, 95) <= DELTA_MAX_DEG_PER_FRAME
        ),
    }


def _roundtrip(
    frames: Sequence[np.ndarray],
    trajectory: Trajectory,
    K: np.ndarray,
    circle: tuple[float, float, float],
    R_bc: np.ndarray,
    config: dict[str, Any],
    fps: float,
) -> dict[str, Any]:
    """Re-measure a rendered sequence and compare against its own input."""

    segment_config = dict(config.get("segment", {}))
    # The renderer paints solid, well-separated speckle colours; reuse the
    # synthetic HSV ranges rather than the ranges tuned for real paint.
    segment_config.update(
        {
            "mode": "color",
            "top_hsv": {"lo": [0, 120, 90], "hi": [12, 255, 255]},
            "bottom_hsv": {"lo": [95, 120, 90], "hi": [125, 255, 255]},
            "yoke_hsv": {"lo": [0, 0, 0], "hi": [179, 120, 60], "enabled": True},
            "grow_px": 0.0,
            "separation_px": 0.0,
            "yoke_dilate_px": 0.0,
            "color_wins_over_yoke": False,
            "morphology_px": 3,
        }
    )
    result = run_pipeline(
        list(frames),
        K=K,
        dist=None,
        circle=circle,
        R_bc=R_bc,
        segment_config=segment_config,
        track_config=config.get("track", {}),
        estimate_config=config.get("estimate", {}),
        fps=fps,
    )
    motion = result.motion
    report: dict[str, Any] = {
        "frames": int(result.frame_count),
        "top_solve_success_rate": float(
            np.mean([estimate.top.success for estimate in result.estimates])
        ),
        "bottom_solve_success_rate": float(
            np.mean([estimate.bottom.success for estimate in result.estimates])
        ),
        **_step_size(trajectory),
    }
    for name, recovered, truth in (
        ("alpha", motion.alpha, trajectory.alpha),
        ("beta_top", motion.beta_top, trajectory.beta_top),
        ("beta_bottom", motion.beta_bottom, trajectory.beta_bottom),
    ):
        count = min(len(recovered), len(truth))
        error = np.rad2deg(recovered[:count] - truth[:count])
        finite = error[np.isfinite(error)]
        report[name] = {
            "rmse_deg": float(np.sqrt(np.mean(finite**2))) if len(finite) else None,
            "max_abs_deg": float(np.max(np.abs(finite))) if len(finite) else None,
            "compared_frames": int(len(finite)),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stride < 1:
        raise SystemExit("--stride must be at least 1")
    config, config_path = load_config(args.config)
    output_root = resolve_from_config(config_path, config.get("output", {}).get("dir", "out/real"))
    results_path = args.results or output_root / "results.json"
    if not results_path.exists():
        raise SystemExit(f"measured results not found: {results_path}. Run scripts/run.py first.")
    clip_path = args.clip or resolve_from_config(config_path, config["input"]["path"])
    output_dir = args.output or output_root / "simulation"
    output_dir.mkdir(parents=True, exist_ok=True)

    measured = _measured(results_path, args.frames, args.stride)
    metadata = measured["metadata"]
    K = np.asarray(metadata["K"], dtype=float)
    dist = distortion_coefficients(config.get("camera", {}))
    circle = configured_circle(config.get("circle", {})) or tuple(metadata["circle"])
    R_bc = configured_frame(config.get("frame_calib", {}))
    if R_bc is None:
        R_bc = np.asarray(metadata["R_bc"], dtype=float)
    fps = float(metadata["fps"]) / max(1, args.stride)
    C, radius = sphere_pose_from_circle(*circle, K)

    probe = FrameSource(clip_path, input_type="auto", max_frames=1)
    first = next(iter(probe))
    image_size = (first.image.shape[0], first.image.shape[1])
    del probe

    written: dict[str, Path] = {}
    report: dict[str, Any] = {
        "results": str(results_path),
        "clip": str(clip_path),
        "frames_used": int(len(measured["frame_index"])),
        "stride": args.stride,
    }

    if "axes" in args.tests:
        print("axes: drawing the measured ball frame on the real footage ...")
        written["axes_overlay"] = _axes_overlay(
            clip_path,
            measured,
            K,
            dist,
            circle,
            R_bc,
            C,
            radius,
            fps,
            output_dir / "axes_overlay.mp4",
            args.graticule_step_deg,
            args.codec,
        )

    rendered: dict[str, list[np.ndarray]] = {}
    model_trajectory, full_trajectory = _trajectories(measured, fps, args.swap_shells)
    if {"replay", "roundtrip"} & set(args.tests):
        setup = _camera_setup(K, R_bc, circle, image_size)
        render_config = RenderConfig(
            camera=setup,
            num_speckles=args.num_speckles,
            dot_radius_px=max(2.0, circle[2] * 0.008),
            draw_yoke=False,
        )
        for label, trajectory in (
            ("simulated: Rx(alpha) Rz(beta) model only", model_trajectory),
            ("simulated: full measured orientation", full_trajectory),
        ):
            print(f"render: {trajectory.name} ({trajectory.num_frames} frames) ...")
            rendered[label] = render_sequence(None, trajectory, render_config).frames

    if "replay" in args.tests:
        print("replay: writing the side-by-side comparison ...")
        written["replay"] = _replay_video(
            clip_path,
            measured,
            K,
            dist,
            rendered,
            fps,
            output_dir / "replay.mp4",
            args.codec,
        )

    if "roundtrip" in args.tests:
        print("roundtrip: re-measuring the rendered sequence ...")
        label = "simulated: full measured orientation"
        report["roundtrip"] = _roundtrip(
            rendered[label], full_trajectory, K, circle, R_bc, config, fps
        )
        summary = report["roundtrip"]
        print(
            f"  rendered motion: {summary['step_deg_per_frame_median']:.2f} deg/frame "
            f"median, {summary['step_deg_per_frame_p95']:.2f} deg/frame p95 "
            f"(characterized limit {summary['characterized_limit_deg_per_frame']:.1f})"
        )
        print("  recovered-minus-input angle error:")
        for name in ("alpha", "beta_top", "beta_bottom"):
            values = summary[name]
            print(
                f"    {name:12s} RMSE {values['rmse_deg']:.3f} deg   "
                f"max {values['max_abs_deg']:.3f} deg"
            )
        if not summary["within_characterized_limit"]:
            print(
                "  IGNORE these numbers: --stride multiplies the per-frame "
                "rotation, and this sequence is past the tracking limit the "
                "synthetic suite characterized. Re-run the round trip with "
                "--stride 1 and --frames to bound the cost instead."
            )
        else:
            print(
                "  A small round-trip error means the estimator and angle "
                "conventions are self-consistent; it does not validate R_bc, "
                "the circle, or the physical motion model."
            )

    report["outputs"] = {name: str(path) for name, path in written.items()}
    report_path = output_dir / "simulation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Outputs:")
    for name, path in written.items():
        print(f"  {name}: {path}")
    print(f"  report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
