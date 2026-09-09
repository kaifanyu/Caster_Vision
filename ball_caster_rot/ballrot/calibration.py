"""Front-end calibration helpers for the projected circle and ball axes."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .sphere import fit_circle_points


def detect_circle_auto(frame_bgr: np.ndarray) -> tuple[float, float, float, dict[str, float]]:
    """Detect and refine the ball silhouette on a clear frame.

    Hough detection supplies a coarse circle.  Only Canny edges in a narrow
    annulus are then used for a least-squares refinement, which rejects yoke
    edges and most internal speckle edges.  A contour fallback handles flat,
    high-contrast synthetic or laboratory backgrounds.
    """

    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("frame_bgr must be a uint8 BGR image")
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    min_radius = max(10, int(min(height, width) * 0.06))
    max_radius = max(min_radius + 1, int(min(height, width) * 0.49))
    threshold_candidate = _closed_threshold_circle(blurred)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=min(height, width) * 0.25,
        param1=80,
        param2=24,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    source = "closed-threshold silhouette"
    if threshold_candidate is not None:
        u0, v0, radius = threshold_candidate
    elif circles is not None and len(circles[0]):
        source = "hough"
        candidates = circles[0]
        # Favor a substantial circle fully contained in the frame.
        scores = []
        for u0, v0, radius in candidates:
            containment = min(u0, v0, width - 1 - u0, height - 1 - v0) / radius
            scores.append(radius * min(max(containment, 0.0), 1.0))
        u0, v0, radius = map(float, candidates[int(np.argmax(scores))])
    else:
        u0, v0, radius = _threshold_circle_fallback(blurred)
        source = "threshold-contour"

    edges = cv2.Canny(blurred, 40, 120)
    yy, xx = np.nonzero(edges)
    refined_count = 0
    for annulus_width in (max(2.5, radius * 0.02), max(1.75, radius * 0.012)):
        radial = np.sqrt((xx - u0) ** 2 + (yy - v0) ** 2)
        selected = np.abs(radial - radius) <= annulus_width
        if selected.sum() < 30:
            break
        points = np.column_stack([xx[selected], yy[selected]])
        try:
            next_u, next_v, next_radius = fit_circle_points(points)
        except ValueError:
            break
        if (
            abs(next_u - u0) > radius * 0.1
            or abs(next_v - v0) > radius * 0.1
            or abs(next_radius - radius) > radius * 0.1
        ):
            break
        u0, v0, radius = next_u, next_v, next_radius
        refined_count = int(len(points))

    if not (0 <= u0 < width and 0 <= v0 < height and radius > 0):
        raise ValueError("automatic circle detector returned an implausible circle")
    diagnostics = {
        "edge_points_used": refined_count,
        "detector": source,
        "u0": float(u0),
        "v0": float(v0),
        "r_px": float(radius),
    }
    return float(u0), float(v0), float(radius), diagnostics


def _closed_threshold_circle(gray: np.ndarray) -> tuple[float, float, float] | None:
    """Recover a silhouette even when a dark yoke splits it into two caps."""

    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    height, width = gray.shape
    image_area = height * width
    bridge = max(9, int(round(min(height, width) * 0.075)))
    if bridge % 2 == 0:
        bridge += 1
    kernels = (
        cv2.getStructuringElement(cv2.MORPH_RECT, (bridge, 5)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, bridge)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge, bridge)),
    )
    candidates: list[tuple[float, float, float, float]] = []
    for polarity in (threshold, cv2.bitwise_not(threshold)):
        for kernel in kernels:
            closed = cv2.morphologyEx(polarity, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if not image_area * 0.02 <= area <= image_area * 0.72 or len(contour) < 30:
                    continue
                perimeter = float(cv2.arcLength(contour, True))
                (u0, v0), radius = cv2.minEnclosingCircle(contour)
                if radius <= 0:
                    continue
                circle_area = np.pi * radius * radius
                fill = area / circle_area
                circularity = 4 * np.pi * area / max(perimeter * perimeter, 1.0)
                # The sphere is expected to be close to circular and mostly
                # filled after bridging. Radius favors the primary object.
                score = radius * min(fill, 1.0) ** 2 * min(circularity, 1.0) ** 2
                if fill >= 0.72 and circularity >= 0.55:
                    candidates.append((score, float(u0), float(v0), float(radius)))
    if not candidates:
        return None
    _, u0, v0, radius = max(candidates, key=lambda item: item[0])
    return u0, v0, radius


def _threshold_circle_fallback(gray: np.ndarray) -> tuple[float, float, float]:
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates = []
    image_area = gray.shape[0] * gray.shape[1]
    for binary in (threshold, cv2.bitwise_not(threshold)):
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not image_area * 0.01 <= area <= image_area * 0.75 or len(contour) < 12:
                continue
            (u0, v0), radius = cv2.minEnclosingCircle(contour)
            if radius <= 0:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / max(perimeter * perimeter, 1.0)
            fill = area / (np.pi * radius * radius)
            score = area * max(circularity, 0.01) * max(fill, 0.01)
            candidates.append((score, float(u0), float(v0), float(radius)))
    if not candidates:
        raise ValueError(
            "could not auto-detect a ball silhouette; use three-point manual calibration"
        )
    _, u0, v0, radius = max(candidates, key=lambda item: item[0])
    return u0, v0, radius


def circle_from_three_points(points_uv: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    points = np.asarray(points_uv, dtype=float)
    if points.shape != (3, 2):
        raise ValueError("exactly three (u, v) silhouette points are required")
    return fit_circle_points(points)


def draw_circle_preview(
    frame_bgr: np.ndarray,
    circle: Sequence[float],
    destination: str | Path,
    *,
    label: str = "fitted ball circle",
) -> Path:
    u0, v0, radius = map(float, circle)
    preview = np.asarray(frame_bgr).copy()
    cv2.circle(
        preview, (int(round(u0)), int(round(v0))), int(round(radius)), (0, 255, 0), 2
    )
    cv2.drawMarker(
        preview, (int(round(u0)), int(round(v0))), (0, 255, 255), cv2.MARKER_CROSS, 18, 2
    )
    cv2.putText(
        preview,
        f"{label}: ({u0:.1f}, {v0:.1f}), r={radius:.1f}px",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), preview):
        raise OSError(f"could not write preview: {output}")
    return output
