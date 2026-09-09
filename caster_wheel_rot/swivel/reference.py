"""Bold sidewall reference-dot detection and absolute roll phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from common.camera import validate_camera_matrix
from .geometry import SwivelGeometry, project_sidewall, unproject_to_plane


@dataclass(frozen=True)
class ReferenceObservation:
    valid: bool
    phase_wrapped: float = float("nan")
    uv: np.ndarray | None = None
    radial_fraction: float = float("nan")
    face_sign: int = 1
    failure_reason: str | None = None


def _ranges(config: Mapping[str, Any] | None) -> list[tuple[np.ndarray, np.ndarray]]:
    values = dict(config or {})
    supplied = values.get("hsv_ranges")
    if supplied is None:
        supplied = [
            {"lo": [0, 80, 70], "hi": [12, 255, 255]},
            {"lo": [168, 80, 70], "hi": [179, 255, 255]},
        ]
    result = []
    for item in supplied:
        if not isinstance(item, Mapping) or "lo" not in item or "hi" not in item:
            raise ValueError("reference_dot.hsv_ranges entries need lo and hi")
        lo = np.asarray(item["lo"], dtype=np.uint8).reshape(3)
        hi = np.asarray(item["hi"], dtype=np.uint8).reshape(3)
        result.append((lo, hi))
    return result


def detect_reference_phase(
    frame_bgr: np.ndarray,
    psi: float,
    geometry: SwivelGeometry,
    K: np.ndarray,
    *,
    face_sign: int | None = None,
    config: Mapping[str, Any] | None = None,
    allowed_mask: np.ndarray | None = None,
) -> ReferenceObservation:
    """Detect a distinct colored dot and return its fork-frame disk angle."""

    values = dict(config or {})
    sign = geometry.visible_face_sign(psi) if face_sign is None else int(face_sign)
    if sign not in {-1, 1}:
        raise ValueError("face_sign must be +1 or -1")
    image = np.asarray(frame_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("frame_bgr must be a uint8 BGR image")
    matrix = validate_camera_matrix(K)
    projection = project_sidewall(
        geometry,
        psi,
        matrix,
        image.shape,
        face_sign=sign,
        inner_radius_fraction=float(values.get("inner_radius_fraction", 0.20)),
        margin_px=int(values.get("margin_px", 0)),
    )
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    binary = np.zeros(image.shape[:2], dtype=np.uint8)
    for lo, hi in _ranges(values):
        binary = cv2.bitwise_or(binary, cv2.inRange(hsv, lo, hi))
    binary[~projection.mask] = 0
    if allowed_mask is not None:
        binary[~np.asarray(allowed_mask, dtype=bool)] = 0
    kernel_size = int(values.get("morphology_px", 3))
    if kernel_size > 1:
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = float(values.get("min_area_px2", 8.0))
    max_area = float(values.get("max_area_px2", 1000.0))
    candidates = [contour for contour in contours if min_area <= cv2.contourArea(contour) <= max_area]
    if not candidates:
        return ReferenceObservation(False, face_sign=sign, failure_reason="reference dot not detected")
    contour = max(candidates, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if abs(moments["m00"]) <= 1e-12:
        return ReferenceObservation(False, face_sign=sign, failure_reason="reference dot has zero image moment")
    uv = np.array([[moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]])
    center, normal = geometry.face_plane_camera(psi, sign)
    point, valid = unproject_to_plane(uv, matrix, center, normal)
    if not valid[0]:
        return ReferenceObservation(False, uv=uv[0], face_sign=sign, failure_reason="reference ray misses sidewall plane")
    vector_camera = point[0] - center
    vector_fork = geometry.camera_vectors_to_fork(vector_camera, psi)
    vector_fork -= float(vector_fork @ geometry.axle_zero_car) * geometry.axle_zero_car
    radius = float(np.linalg.norm(vector_fork))
    fraction = radius / geometry.wheel_radius
    if not float(values.get("min_radial_fraction", 0.20)) <= fraction <= float(values.get("max_radial_fraction", 1.05)):
        return ReferenceObservation(False, uv=uv[0], radial_fraction=fraction, face_sign=sign, failure_reason="reference dot lies outside configured annulus")
    first, second = geometry.radial_basis_zero()
    phase = float(np.arctan2(vector_fork @ second, vector_fork @ first))
    return ReferenceObservation(True, phase, uv[0], fraction, sign)


def unwrap_reference_phase(
    observations: Sequence[ReferenceObservation],
    *,
    max_gap_frames: int = 5,
) -> np.ndarray:
    """Unwrap detected phase across short gaps; long gaps start a new segment."""

    output = np.full(len(observations), np.nan)
    last_index: int | None = None
    last_wrapped = 0.0
    last_unwrapped = 0.0
    for index, observation in enumerate(observations):
        if not observation.valid:
            continue
        wrapped = float(observation.phase_wrapped)
        if last_index is None or index - last_index - 1 > int(max_gap_frames):
            unwrapped = wrapped
        else:
            delta = (wrapped - last_wrapped + np.pi) % (2.0 * np.pi) - np.pi
            unwrapped = last_unwrapped + delta
        output[index] = unwrapped
        last_index, last_wrapped, last_unwrapped = index, wrapped, unwrapped
    return output


__all__ = ["ReferenceObservation", "detect_reference_phase", "unwrap_reference_phase"]
