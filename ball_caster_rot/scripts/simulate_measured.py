#!/usr/bin/env python
"""Replay measured angles as motion and check them against the real clip.

Visual checks and a synthetic round-trip test are available:

``axes``
    Draws the calibrated ball axes and a ball-fixed graticule on the real
    footage, rotated by the measured per-frame orientation. The graticule and
    numbered probes are synthetic surface locations, not tracked paint marks.
    Their motion can expose sliding, but agreement at the first frame does not
    validate the calibration or fitted motion.

``residuals``
    Draws the actual saved feature pixels and their fitted predictions, with
    residual arrows and persistent track IDs. Unaccepted candidate fits remain
    explicitly labelled, including when their pixels fit but lack an anchor.

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
    measurement_frame,
    resolve_from_config,
)
from ballrot.io_frames import FrameSource
from ballrot.pipeline import run_pipeline
from ballrot.rotation import Rx, Rz
from ballrot.shell_geometry import surface_camera
from ballrot.sphere import sphere_pose_from_circle
from synthetic.generate import CameraSetup, RenderConfig, Trajectory, render_sequence

TESTS = ("axes", "residuals", "replay", "roundtrip")


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
        "--max-residuals", type=int, default=100,
        help="Maximum observed/predicted feature pairs per shell in residuals (default: 100)",
    )
    parser.add_argument(
        "--swap-shells",
        action="store_true",
        help=(
            "Toggle the saved top_shell_sign for this replay: swap which "
            "geometric cap carries each tracked colour and its full motion. "
            "This changes cap assignment, not the calibration or angle signs."
        ),
    )
    parser.add_argument("--codec", default="mp4v", help="FourCC for written videos")
    parser.add_argument(
        "--render-width", type=int, default=None,
        help="Synthetic panel width for a smaller replay render; omit for roundtrip tests",
    )
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
    step_deg: float, hemisphere: int, samples: int = 60, gap_fraction: float = 0.0
) -> list[np.ndarray]:
    """Meridians and parallels about ball ``+z``, restricted to one hemisphere.

    ``hemisphere`` is ``+1`` for ball ``z >= 0`` (the "top" shell) and ``-1``
    for ``z <= 0``.  Each shell is an independently rotating rigid body, so
    drawing its graticule only over its own surface keeps the two overlays
    from covering each other.
    """

    if not np.isfinite(gap_fraction) or not 0.0 <= gap_fraction < 1.0:
        raise ValueError("gap_fraction must be finite and in [0, 1)")
    curves: list[np.ndarray] = []
    step = np.deg2rad(float(step_deg))
    edge = np.arcsin(gap_fraction)
    lo, hi = (edge, np.pi / 2) if hemisphere >= 0 else (-np.pi / 2, -edge)
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
    R_bc: np.ndarray, hemisphere: int, count: int = 14, seed: int = 12345,
    gap_fraction: float = 0.0,
) -> np.ndarray:
    """Synthetic surface probes, not observed paint features or track IDs.

    Their relative motion can help inspect sliding, but their initial locations
    are random and neither paint correspondence nor yoke visibility is tested.
    """

    if not np.isfinite(gap_fraction) or not 0.0 <= gap_fraction < 1.0:
        raise ValueError("gap_fraction must be finite and in [0, 1)")
    rng = np.random.default_rng(seed)
    toward_camera_ball = R_bc.T @ np.array([0.0, 0.0, -1.0])
    candidates = rng.normal(size=(max(2000, count * 1000), 3))
    candidates /= np.linalg.norm(candidates, axis=1)[:, None]
    usable = (np.sign(hemisphere) * candidates[:, 2] >= gap_fraction) & (
        candidates @ toward_camera_ball >= 0.45
    )
    # A completely hidden shell has no visible probe locations. Never wait
    # forever trying to sample points from an empty visible cap.
    return candidates[usable][:count]


def _draw_curve(
    image: np.ndarray,
    ball_points: np.ndarray,
    R_total: np.ndarray,
    K: np.ndarray,
    C: np.ndarray,
    radius: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    *,
    sign: int = 1,
    gap_fraction: float = 0.0,
    geometry: str = "common_sphere_caps",
) -> None:
    uv, visible = _project_shell(ball_points, R_total, K, C, radius, sign, gap_fraction, geometry)
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


def _project_shell(ball_points, orientation, K, C, radius, sign, gap_fraction, geometry):
    """Project the configured shell surface, including its moving center."""
    points = surface_camera(ball_points, orientation, C, radius, sign, gap_fraction, geometry)
    normals = np.asarray(ball_points) @ np.asarray(orientation).T
    toward_camera = -points / np.maximum(np.linalg.norm(points, axis=-1, keepdims=True), 1e-12)
    visible = (np.einsum("ij,ij->i", normals, toward_camera) > 0.05) & (points[:, 2] > 1e-9)
    homogeneous = points @ np.asarray(K).T
    uv = np.full((len(points), 2), np.nan)
    uv[visible] = homogeneous[visible, :2] / homogeneous[visible, 2:3]
    return uv, visible


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
    for shell in ("top", "bottom"):
        key = f"valid_{shell}"
        if key in series:
            values = np.asarray(series[key])
            if values.shape != (count,) or values.dtype.kind != "b":
                raise ValueError(f"{key} must contain one boolean per measured frame")
            data[key] = values
    selected = np.arange(0, count if frames is None else min(count, frames), max(1, stride))
    output = {key: value[selected] for key, value in data.items()}
    output["frame_index"] = selected
    output["metadata"] = payload["metadata"]
    output["mechanical"] = payload.get("mechanical")
    _validate_mechanical_poses(output)
    return output


def _mechanical_model(measured: dict) -> dict | None:
    """Return the saved fit geometry; current config cannot relabel old poses."""

    model = measured.get("metadata", {}).get("mechanical_model")
    diagnostic = measured.get("mechanical") or {}
    if not model or not model.get("enabled", False):
        if diagnostic.get("config", {}).get("enabled", False):
            raise ValueError("Mechanical results are missing saved mechanical_model geometry")
        return None
    if model.get("model") != "shared_roll_independent_spin":
        raise ValueError("Unsupported saved mechanical model")
    gap = float(model.get("gap_fraction", 0.0))
    if not np.isfinite(gap) or not 0.0 <= gap < 1.0:
        raise ValueError("Saved mechanical gap_fraction must be finite and in [0, 1)")
    geometry = model.get("geometry", "common_sphere_caps")
    if geometry not in ("common_sphere_caps", "separated_hemispheres"):
        raise ValueError("Unsupported saved mechanical geometry")
    pivot = _validated_pivot(model.get("pivot_camera"))
    return {**model, "gap_fraction": gap, "geometry": geometry,
            "pivot_camera": None if pivot is None else pivot.tolist()}


def _validated_pivot(value) -> np.ndarray | None:
    """A supplied pivot is expressed in shell-radius units, independently of K."""
    if value is None:
        return None
    pivot = np.asarray(value, dtype=float)
    if pivot.shape != (3,) or not np.all(np.isfinite(pivot)) or pivot[2] <= 1.0:
        raise ValueError("pivot_camera must be a finite C/R vector with z > 1")
    return pivot


def _effective_pivot(model: dict | None, source_center: np.ndarray, radius: float) -> np.ndarray:
    pivot = _validated_pivot((model or {}).get("pivot_camera"))
    return np.asarray(source_center, dtype=float).copy() if pivot is None else radius * pivot


def _validate_mechanical_poses(measured: dict) -> None:
    """Never silently draw independent rotations as a constrained result."""

    if _mechanical_model(measured) is None:
        return
    shared = np.asarray(measured["alpha_rad"], dtype=float)
    for shell in ("top", "bottom"):
        valid = _pose_validity(measured, shell)
        alpha = np.asarray(measured[f"alpha_{shell}_rad"], dtype=float)
        gamma = np.asarray(measured[f"gamma_{shell}_rad"], dtype=float)
        difference = np.arctan2(np.sin(alpha - shared), np.cos(alpha - shared))
        if (not np.all(np.isfinite(shared[valid]))
            or np.any(np.abs(difference[valid]) > 1e-7)
            or np.any(np.abs(gamma[valid]) > 1e-7)):
            raise ValueError(
                "Saved mechanical poses violate shared roll or zero sideways tilt; "
                "rerun mechanical fitting before replaying."
            )


def _result_mechanical_model(config: dict, measured: dict) -> dict | None:
    model = _mechanical_model(measured)
    if model is not None and "pivot_camera" in config.get("mechanical", {}):
        requested = _validated_pivot(config["mechanical"]["pivot_camera"])
        saved = _validated_pivot(model.get("pivot_camera"))
        mismatch = ((requested is None) != (saved is None)
                    or (requested is not None and saved is not None
                        and not np.allclose(requested, saved, atol=1e-10, rtol=0.0)))
        if mismatch:
            raise ValueError(
                "Config mechanical pivot_camera differs from the saved results; rerun "
                "mechanical fitting before changing the rendered geometry."
            )
    if model is not None and "geometry" in config.get("mechanical", {}):
        if config["mechanical"]["geometry"] != model["geometry"]:
            raise ValueError(
                "Config mechanical geometry differs from the saved results; rerun "
                "mechanical fitting before changing the rendered geometry."
            )
    if model is not None and "gap_fraction" in config.get("mechanical", {}):
        requested = float(config["mechanical"]["gap_fraction"])
        if not np.isclose(requested, model["gap_fraction"], atol=1e-10, rtol=0.0):
            raise ValueError(
                "Config mechanical gap differs from the saved results; rerun "
                "mechanical fitting before changing the rendered geometry."
            )
    return model


def _pose_validity(measured: dict[str, np.ndarray], shell: str) -> np.ndarray:
    """A drawable shell pose needs finite angles and any saved validity flag."""

    finite = np.all(
        np.isfinite(np.column_stack([
            measured[f"{component}_{shell}_rad"]
            for component in ("alpha", "gamma", "beta")
        ])),
        axis=1,
    )
    key = f"valid_{shell}"
    if key in measured:
        declared = np.asarray(measured[key], dtype=bool)
        if declared.shape != finite.shape:
            raise ValueError(f"{key} must contain one boolean per measured frame")
        finite &= declared
    return finite


def _render_component(measured: dict[str, np.ndarray], component: str, shell: str) -> np.ndarray:
    """Supply a display fallback without turning invalid poses into measurements."""

    values = np.asarray(measured[f"{component}_{shell}_rad"], dtype=float).copy()
    if f"valid_{shell}" in measured:
        values[~_pose_validity(measured, shell)] = np.nan
    return _filled(values)


def _draw_missing_pose_notice(
    image: np.ndarray, missing: list[str], *, display_fallback: bool = False
) -> None:
    if not missing:
        return
    lines = [f"UNRESOLVED: {' / '.join(missing)} shell pose"]
    lines.append(
        "Display fallback only; missing motion is unmeasured"
        if display_fallback else "Missing shell grid and probes are hidden"
    )
    # Place below replay-panel titles, on an opaque backing for bright footage.
    for offset, message in enumerate(lines):
        y = 53 + 23 * offset
        scale = min(0.58, max(0.3, (image.shape[1] - 20) / (len(message) * 11.0)))
        (width, height), baseline = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(image, (6, y - height - 4), (min(image.shape[1] - 1, width + 14), y + baseline + 3), (0, 0, 0), -1)
        cv2.putText(image, message, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 200, 255), 1, cv2.LINE_AA)


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


def _reprojection_entry(measured: dict, shell: str, frame_index: int) -> dict:
    """Diagnostics retain original frame indices even when playback is strided."""
    entries = (measured.get("mechanical") or {}).get("frames", {}).get(shell, [])
    if 0 <= frame_index < len(entries) and entries[frame_index].get("frame_index") == frame_index:
        return entries[frame_index]
    return next((entry for entry in entries if entry.get("frame_index") == frame_index), {})


def _diagnostic_text(image: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    for row, (message, color) in enumerate(lines):
        y = 23 + 24 * row
        scale = min(0.55, max(0.22, (image.shape[1] - 20) / max(1, len(message) * 10.0)))
        (width, height), baseline = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(image, (4, y - height - 4),
                      (min(image.shape[1] - 1, width + 16), y + baseline + 3), (0, 0, 0), -1)
        cv2.putText(image, message, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _draw_reprojection_frame(image: np.ndarray, measured: dict, position: int, max_points: int = 100) -> dict:
    """Draw saved fit evidence without promoting candidate predictions to poses."""
    frame_index = int(measured["frame_index"][position])
    lines = [
        (f"frame {frame_index}  t={measured['time_s'][position]:.3f}s  FEATURE REPROJECTION", (255, 255, 255)),
        ("circle: observed -> cross: predicted | green: pixel inlier, red: outlier | ?: unanchored", (255, 255, 255)),
    ]
    summary = {}
    for shell in ("top", "bottom"):
        entry = _reprojection_entry(measured, shell, frame_index)
        accepted = bool(_pose_validity(measured, shell)[position])
        pairs = entry.get("reprojection") or []
        usable = []
        for point in pairs:
            observed = np.asarray(point.get("observed_uv"), dtype=float)
            predicted = np.asarray(point.get("predicted_uv"), dtype=float)
            if observed.shape == predicted.shape == (2,) and np.all(np.isfinite([observed, predicted])):
                usable.append((point, observed, predicted))
        usable.sort(key=lambda item: int(item[0]["track_id"]))
        selected = usable
        if len(usable) > max_points:
            selected = [usable[index] for index in np.linspace(0, len(usable) - 1, max_points, dtype=int)]
        for point, observed, predicted in selected:
            inlier = bool(point.get("inlier", False))
            color = (60, 220, 60) if inlier else (50, 60, 245)
            # Clip extreme candidate projections to OpenCV's safe integer range.
            observed_px = tuple(np.rint(np.clip(observed, -1000000, 1000000)).astype(int))
            predicted_px = tuple(np.rint(np.clip(predicted, -1000000, 1000000)).astype(int))
            cv2.arrowedLine(image, observed_px, predicted_px, color, 1, cv2.LINE_AA, tipLength=0.2)
            cv2.circle(image, observed_px, 4, color, 1, cv2.LINE_AA)
            cv2.drawMarker(image, predicted_px, color, cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
            label = f"{shell[0].upper()}{point['track_id']}" + ("" if point.get("anchored", False) else "?")
            label_at = (observed_px[0] + 5, observed_px[1] - 5)
            cv2.putText(image, label, label_at, cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, label, label_at, cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
        reason = entry.get("reason", "no_saved_fit_diagnostics")
        if not usable:
            status = f"{shell}: NO SAVED PREDICTIONS | {reason}"
        else:
            state = "ACCEPTED" if accepted else "UNACCEPTED CANDIDATE"
            inliers = sum(bool(point.get("inlier", False)) for point, _, _ in usable)
            anchored = sum(bool(point.get("anchored", False)) for point, _, _ in usable)
            status = (f"{shell}: {state} | {reason} | inliers {inliers}/{len(usable)}, "
                      f"anchored {anchored}/{len(usable)}, shown {len(selected)}")
        lines.append((status, (60, 220, 60) if accepted else (0, 200, 255)))
        bridge = entry.get("bridge")
        if isinstance(bridge, dict):
            details = ", ".join(f"{key}={value}" for key, value in bridge.items()
                                if isinstance(value, (str, int, float, bool)))
            if details:
                lines.append((f"{shell} bridge: {details}", (210, 210, 210)))
        summary[shell] = {"accepted": accepted, "available": len(usable), "shown": len(selected), "reason": reason}
    _diagnostic_text(image, lines)
    return summary


def _reprojection_overlay(
    clip_path: Path, measured: dict, K: np.ndarray, dist: np.ndarray,
    fps: float, destination: Path, codec: str, max_points: int = 100,
) -> Path:
    if max_points < 1:
        raise ValueError("max_points must be positive")
    wanted = {int(value): position for position, value in enumerate(measured["frame_index"])}
    writer = None
    source = FrameSource(clip_path, input_type="auto")
    try:
        for record in source:
            position = wanted.get(int(record.index))
            if position is None:
                continue
            frame = undistort_image(record.image, K, dist)
            _draw_reprojection_frame(frame, measured, position, max_points)
            if writer is None:
                writer = _writer(destination, fps, (frame.shape[1], frame.shape[0]), codec)
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
    if writer is None:
        raise ValueError("No requested measured frames were found in the clip")
    return destination


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
    swap_shells: bool = False,
) -> Path:
    _validate_mechanical_poses(measured)
    mechanical = _mechanical_model(measured)
    C = _effective_pivot(mechanical, C, radius)
    gap = mechanical["gap_fraction"] if mechanical is not None else 0.0
    geometry = mechanical["geometry"] if mechanical is not None else "common_sphere_caps"
    cap_gap = gap if geometry == "common_sphere_caps" else 0.0
    wanted = {int(value): position for position, value in enumerate(measured["frame_index"])}
    curves = {sign: _graticule(step_deg, sign, 60, cap_gap) for sign in (1, -1)}
    probes = {sign: _probe_points(R_bc, sign, 14, 12345, cap_gap) for sign in (1, -1)}
    alpha_top = measured["alpha_top_rad"]
    alpha_bottom = measured["alpha_bottom_rad"]
    beta_top = measured["beta_top_rad"]
    beta_bottom = measured["beta_bottom_rad"]
    gamma_top = measured["gamma_top_rad"]
    gamma_bottom = measured["gamma_bottom_rad"]
    validity = {shell: _pose_validity(measured, shell) for shell in ("top", "bottom")}

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
            if geometry == "common_sphere_caps":
                cv2.circle(
                    frame,
                    (int(round(circle[0])), int(round(circle[1]))),
                    int(round(circle[2])),
                    (200, 200, 200),
                    1,
                )
            for shell, hemisphere, angles, color in (
                (
                    "top",
                    -1 if swap_shells else 1,
                    (alpha_top[position], gamma_top[position], beta_top[position]),
                    (255, 220, 0),
                ),
                (
                    "bottom",
                    1 if swap_shells else -1,
                    (
                        alpha_bottom[position],
                        gamma_bottom[position],
                        beta_bottom[position],
                    ),
                    (180, 0, 255),
                ),
            ):
                if not validity[shell][position]:
                    continue
                R_total = R_bc @ ball_rotation(*angles)
                for curve in curves[hemisphere]:
                    if geometry == "common_sphere_caps":
                        _draw_curve(frame, curve, R_total, K, C, radius, color)
                    else:
                        _draw_curve(frame, curve, R_total, K, C, radius, color,
                                    sign=hemisphere, gap_fraction=gap, geometry=geometry)
                if geometry == "common_sphere_caps":
                    uv, visible = _project(probes[hemisphere] @ R_total.T, K, C, radius)
                else:
                    uv, visible = _project_shell(probes[hemisphere], R_total, K, C, radius,
                                                 hemisphere, gap, geometry)
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
            _draw_missing_pose_notice(
                frame, [shell for shell in ("top", "bottom") if not validity[shell][position]]
            )
            footer = (
                f"frame {record.index}   cyan = top shell   magenta = bottom shell",
                "Numbered rings: synthetic probes. See residuals for tracked features.",
            )
            for row, label in enumerate(footer):
                width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .6, 1)[0][0]
                scale = .6 * min(1., max(1., frame.shape[1] - 20) / max(width, 1))
                cv2.putText(frame, label, (10, frame.shape[0] - 34 + 20 * row),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
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
    pivot_camera: Sequence[float] | None = None,
) -> CameraSetup:
    C, radius = sphere_pose_from_circle(*circle, K)
    C = _effective_pivot({"pivot_camera": pivot_camera}, C, radius)
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
    _validate_mechanical_poses(measured)
    constrained = _mechanical_model(measured) is not None
    alpha = _filled(measured["alpha_rad"])
    beta_top = _render_component(measured, "beta", "top")
    beta_bottom = _render_component(measured, "beta", "bottom")
    gamma_top = _render_component(measured, "gamma", "top")
    gamma_bottom = _render_component(measured, "gamma", "bottom")
    alpha_top = _render_component(measured, "alpha", "top")
    alpha_bottom = _render_component(measured, "alpha", "bottom")
    if constrained:
        # A missing spin is a labelled display hold. Both rigid caps still
        # follow the available shared roll; holding shell roll separately
        # would recreate the impossible relative tilt during missing data.
        alpha_top = alpha.copy()
        alpha_bottom = alpha.copy()
        gamma_top = np.zeros_like(alpha)
        gamma_bottom = np.zeros_like(alpha)
    if swap_shells:
        beta_top, beta_bottom = beta_bottom, beta_top
        gamma_top, gamma_bottom = gamma_bottom, gamma_top
        alpha_top, alpha_bottom = alpha_bottom, alpha_top
    model = Trajectory(alpha, beta_top, beta_bottom, fps=fps, name="measured_model")
    full = Trajectory(
        alpha,
        beta_top,
        beta_bottom,
        fps=fps,
        name="measured_full",
        gamma_top=gamma_top,
        gamma_bottom=gamma_bottom,
        alpha_top=alpha_top,
        alpha_bottom=alpha_bottom,
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
    validity = {shell: _pose_validity(measured, shell) for shell in ("top", "bottom")}
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
                panel = _panel(frames[position], name, panel_width)
                _draw_missing_pose_notice(
                    panel,
                    [shell for shell in ("top", "bottom") if not validity[shell][position]],
                    display_fallback=True,
                )
                panels.append(panel)
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


def _roundtrip_skip_reason(model: dict | None) -> str | None:
    """The legacy raw-increment test has no translated-shell geometry model."""
    if (model or {}).get("geometry", "common_sphere_caps") == "separated_hemispheres":
        return ("Raw-increment roundtrip assumes a common sphere and cannot validate "
                "separated hemispheres; use geometry-aware mechanical fit tests.")
    if (model or {}).get("pivot_camera") is not None:
        return ("Raw-increment roundtrip derives its center from the source circle and "
                "cannot validate a mechanical pivot_camera override.")
    return None


def _step_size(trajectory: Trajectory) -> dict[str, Any]:
    """Per-frame rotation of the sequence that is about to be re-measured."""

    steps = []
    for name in ("top", "bottom"):
        beta = getattr(trajectory, f"beta_{name}")
        gamma = getattr(trajectory, f"gamma_{name}")
        angles = np.column_stack([getattr(trajectory, f"alpha_{name}"), gamma, beta])
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
    swap_shells: bool = False,
    mechanical_model: dict | None = None,
) -> dict[str, Any]:
    """Re-measure a rendered sequence and compare against its own input."""

    model = mechanical_model if mechanical_model is not None else config.get("mechanical")
    reason = _roundtrip_skip_reason(model)
    if reason is not None:
        return {"status": "skipped", "reason": reason}
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
        ("alpha", motion.alpha, 0.5 * (trajectory.alpha_top + trajectory.alpha_bottom)),
        ("beta_top", motion.beta_top, trajectory.beta_bottom if swap_shells else trajectory.beta_top),
        ("beta_bottom", motion.beta_bottom, trajectory.beta_top if swap_shells else trajectory.beta_bottom),
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


def _result_geometry(config: dict, metadata: dict):
    """Use the saved coordinate frame and reject incompatible config changes."""
    K = np.asarray(metadata["K"], dtype=float)
    dist = distortion_coefficients({"dist": metadata["dist"]})
    circle = tuple(metadata["circle"])
    frame = configured_frame({"R_bc": metadata["R_bc"]})
    requested = measurement_frame(config.get("frame_calib", {}))
    if requested is not None and not np.allclose(requested, frame, atol=1e-7):
        raise ValueError(
            "Config initial orientation differs from these results. Run "
            "scripts/reorient_results.py or scripts/run.py before replaying; "
            "changing the rendered frame alone would misinterpret the angles."
        )
    configured = configured_circle(config.get("circle", {}))
    cfg_K = config.get("camera", {}).get("K")
    cfg_dist = distortion_coefficients(config.get("camera", {}))
    if ((configured is not None and not np.allclose(configured, circle))
        or (cfg_K is not None and not np.allclose(cfg_K, K))
        or cfg_dist.shape != dist.shape or not np.allclose(cfg_dist, dist)):
        raise ValueError("Config camera/circle differs from the saved results; rerun tracking.")
    return K, dist, circle, frame


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stride < 1:
        raise SystemExit("--stride must be at least 1")
    if args.max_residuals < 1:
        raise SystemExit("--max-residuals must be at least 1")
    if args.render_width is not None and (args.render_width < 64 or "roundtrip" in args.tests):
        raise SystemExit("--render-width must be >=64 and is for visual playback only; omit it for roundtrip")
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
    K, dist, circle, R_bc = _result_geometry(config, metadata)
    mechanical = _result_mechanical_model(config, measured)
    if mechanical is None and config.get("mechanical", {}).get("enabled", False):
        print("These saved results are unconstrained. Run scripts/run.py to apply the mechanical fit.")
    top_shell_sign = int(metadata.get("top_shell_sign", 1))
    if top_shell_sign not in (-1, 1):
        raise ValueError("top_shell_sign must be +1 or -1")
    swap_shells = (top_shell_sign == -1) != args.swap_shells
    fps = float(metadata["fps"]) / max(1, args.stride)
    C, radius = sphere_pose_from_circle(*circle, K)
    C = _effective_pivot(mechanical, C, radius)
    roundtrip_skip = _roundtrip_skip_reason(mechanical)

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
        "initial_roll_deg": metadata.get("initial_roll_deg", 0.0),
        "top_shell_sign": -1 if swap_shells else 1,
        "mechanical_model": mechanical,
        "pivot_camera": (C / radius).tolist(),
        "pivot_source": "saved_mechanical_override" if mechanical is not None and mechanical.get("pivot_camera") is not None else "source_circle",
        "surface_geometry": mechanical["geometry"] if mechanical is not None else "common_sphere_caps",
        "pose_source": "mechanical_fit" if mechanical is not None else "independent_shell_estimates",
        "mechanical_note": (
            "Both shells share roll and have independent swivel/spin. Their rim planes "
            "remain separated by gap_fraction times the sphere diameter in 3D; "
            "the apparent image gap changes with view. This geometric consistency "
            "does not establish measurement accuracy."
            if mechanical is not None else None
        ),
        "geometry_note": (
            "Complete hemispheres have centers offset by +/-radius*gap_fraction along the rotated spin axis."
            if mechanical is not None and mechanical["geometry"] == "separated_hemispheres" else
            "Both shell caps lie on a common sphere; a nonzero gap excludes a band around its equator."
        ),
        "calibration_note": (
            "Camera, reference center/radius and initial calibrated axes are fixed inputs. "
            "Axes lines show that initial frame. Grids and numbered probes are synthetic surface "
            "locations, not observed feature IDs or independent accuracy measurements."
        ),
        "fit_method": (measured.get("mechanical") or {}).get("method"),
        "timing_note": "Videos use constant-rate diagnostic playback; result time_s contains measurement timestamps.",
        "tracking_note": (
            "Unresolved shell poses have no moving grid or probes in the axes overlay. "
            "Replay explicitly labels display fallbacks across missing poses; these are not measurements."
        ),
        "reprojection_note": (
            "Residuals use saved observed and predicted feature pixels in the undistorted image. "
            "Persistent track IDs identify source observations. Pixel inliers may still belong to an "
            "unaccepted or unanchored candidate; frame labels and question marks distinguish these. "
            "Missing predictions are not extrapolated."
        ),
        "unresolved_frames": {
            shell: int(np.count_nonzero(~_pose_validity(measured, shell)))
            for shell in ("top", "bottom")
        },
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
            swap_shells=swap_shells,
        )

    if "residuals" in args.tests:
        print("residuals: drawing saved observed and predicted feature pixels ...")
        written["reprojection_overlay"] = _reprojection_overlay(
            clip_path, measured, K, dist, fps, output_dir / "reprojection_overlay.mp4",
            args.codec, args.max_residuals,
        )

    rendered: dict[str, list[np.ndarray]] = {}
    model_trajectory, full_trajectory = _trajectories(measured, fps, swap_shells)
    full_label = (
        "simulated: mechanically constrained fit" if mechanical is not None
        else "simulated: full measured orientation"
    )
    if "replay" in args.tests or ("roundtrip" in args.tests and roundtrip_skip is None):
        render_K, render_circle, render_size = K, circle, image_size
        if args.render_width is not None:
            scale = args.render_width / image_size[1]
            render_K = np.diag([scale, scale, 1.0]) @ K
            render_circle = tuple(value * scale for value in circle)
            render_size = (int(round(image_size[0] * scale)), args.render_width)
        setup = _camera_setup(render_K, R_bc, render_circle, render_size,
                              pivot_camera=(mechanical or {}).get("pivot_camera"))
        defaults = RenderConfig()
        render_config = RenderConfig(
            camera=setup,
            num_speckles=args.num_speckles,
            dot_radius_px=max(1.0, render_circle[2] * 0.008),
            draw_yoke=False,
            top_color_bgr=defaults.bottom_color_bgr if swap_shells else defaults.top_color_bgr,
            bottom_color_bgr=defaults.top_color_bgr if swap_shells else defaults.bottom_color_bgr,
            gap_fraction=mechanical["gap_fraction"] if mechanical is not None else 0.0,
            geometry=mechanical["geometry"] if mechanical is not None else "common_sphere_caps",
        )
        trajectories = [(full_label, full_trajectory)]
        if mechanical is None:
            trajectories.insert(0, ("simulated: Rx(alpha) Rz(beta) model only", model_trajectory))
        for label, trajectory in trajectories:
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
        if roundtrip_skip is not None:
            report["roundtrip"] = {"status": "skipped", "reason": roundtrip_skip}
            print(f"roundtrip: skipped. {roundtrip_skip}")
    if "roundtrip" in args.tests and roundtrip_skip is None:
        print("roundtrip: re-measuring the rendered sequence ...")
        report["roundtrip"] = _roundtrip(
            rendered[full_label], full_trajectory, K, circle, R_bc, config, fps,
            swap_shells=swap_shells,
            mechanical_model=mechanical or {"geometry": "common_sphere_caps"},
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
