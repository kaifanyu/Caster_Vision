"""Visible wheel-face masks, with explicit configurable occluder exclusions."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import project_camera_points, project_sidewall


def wheel_tracking_mask(image, geometry, K, psi, *, face_sign=None,
                        inner_fraction=.2, margin_px=2, config=None, observation=None):
    values = dict(config or {})
    projection = project_sidewall(geometry, psi, K, image.shape, face_sign=face_sign,
                                 inner_radius_fraction=inner_fraction, margin_px=margin_px)
    mask = projection.mask.copy()
    if not values.get("enabled", False):
        return mask, projection
    # Optional material mask for a white printed wheel face. Closing fills
    # small dark speckles without admitting the much larger dark fork.
    material = values.get("white_face", {})
    if material.get("enabled", False):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white = ((hsv[:, :, 1] <= int(material.get("max_saturation", 100))) &
                 (hsv[:, :, 2] >= int(material.get("min_value", 95)))).astype(np.uint8)
        radius = int(material.get("close_radius_px", 5))
        if radius < 0:
            raise ValueError("close_radius_px must be nonnegative")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
        mask &= white.astype(bool)
    excluded = np.zeros(mask.shape, np.uint8)
    if values.get("exclude_markers", True) and observation is not None:
        corners_by_id = dict(observation.detected_corners)
        if observation.corners_uv is not None:
            corners_by_id.setdefault(observation.marker_id, observation.corners_uv)
        # Coordinates in marker-side units: x right, y towards its top edge.
        source = np.array([[-.5,.5],[.5,.5],[.5,-.5],[-.5,-.5]], np.float32)
        # A generic tag has no known carrier direction. Calibrated per-ID
        # polygons can extend this square along the actual carrier mounting.
        default = [[-.8,-.8],[.8,-.8],[.8,.8],[-.8,.8]]
        polygons = values.get("marker_polygons", {})
        for key, corners in corners_by_id.items():
            polygon = polygons.get(str(key), polygons.get(key, default))
            polygon = np.asarray(polygon, np.float32).reshape(-1, 2)
            if len(polygon) < 3 or not np.isfinite(polygon).all():
                raise ValueError("marker exclusion polygons require at least three finite points")
            H = cv2.getPerspectiveTransform(source, np.asarray(corners, np.float32).reshape(4,2))
            uv = cv2.perspectiveTransform(polygon[None], H)[0]
            if np.isfinite(uv).all():
                cv2.fillPoly(excluded, [np.rint(uv).astype(np.int32)], 255)
    # Optional measured polygons, expressed relative to S at fork psi=0.
    for polygon in values.get("fork_polygons_m", []):
        points = np.asarray(polygon, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
            raise ValueError("fork_polygons_m entries must have shape (N>=3,3)")
        car = points @ geometry.R_car_from_fork(psi).T + geometry.swivel_axis_car
        uv, valid = project_camera_points(geometry.car_points_to_camera(car), K)
        if valid.all():
            cv2.fillPoly(excluded, [np.rint(uv).astype(np.int32)], 255)
    mask &= excluded == 0
    # Keep feature centers away from mixed-surface tracking patches.
    support = int(values.get("support_margin_px", 3))
    if support < 0:
        raise ValueError("support_margin_px must be nonnegative")
    if support:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*support+1, 2*support+1))
        mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    return mask, projection
