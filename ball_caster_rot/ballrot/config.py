"""Small YAML configuration helpers shared by command-line entry points."""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np
import yaml


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    return loaded, config_path


def resolve_from_config(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def camera_matrix(camera: Mapping[str, Any], image_shape: tuple[int, ...]) -> tuple[np.ndarray, bool]:
    """Return K and whether it came from the approximate-FOV fallback."""

    configured = camera.get("K")
    if configured is not None:
        K = np.asarray(configured, dtype=float)
        if K.shape != (3, 3) or not np.all(np.isfinite(K)):
            raise ValueError("camera.K must be a finite 3x3 matrix")
        if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.isclose(K[2, 2], 1.0):
            raise ValueError("camera.K has invalid focal lengths or homogeneous scale")
        return K, False
    height, width = image_shape[:2]
    fov = float(camera.get("fov_deg", 60.0))
    if not 1.0 < fov < 179.0:
        raise ValueError("camera.fov_deg must be between 1 and 179 degrees")
    focal = 0.5 * width / np.tan(np.deg2rad(fov) / 2.0)
    warnings.warn(
        "camera.K is null: using an approximate pinhole camera from image width "
        f"and fov_deg={fov:g}. Calibrate the camera before trusting metric accuracy.",
        RuntimeWarning,
        stacklevel=2,
    )
    return np.array(
        [[focal, 0.0, (width - 1) / 2], [0.0, focal, (height - 1) / 2], [0, 0, 1]],
        dtype=float,
    ), True


def distortion_coefficients(camera: Mapping[str, Any]) -> np.ndarray:
    values = camera.get("dist", [0, 0, 0, 0, 0])
    result = np.asarray(values, dtype=float).reshape(-1)
    if len(result) not in (4, 5, 8, 12, 14) or not np.all(np.isfinite(result)):
        raise ValueError("camera.dist must contain 4, 5, 8, 12, or 14 finite values")
    return result


def configured_circle(circle: Mapping[str, Any]) -> tuple[float, float, float] | None:
    values = [circle.get("u0"), circle.get("v0"), circle.get("r_px")]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("circle.u0, circle.v0, and circle.r_px must be all set or all null")
    u0, v0, radius = map(float, values)
    if not np.all(np.isfinite([u0, v0, radius])) or radius <= 0:
        raise ValueError("configured circle must be finite and have positive r_px")
    return u0, v0, radius


def configured_frame(camera_frame: Mapping[str, Any]) -> np.ndarray | None:
    value = camera_frame.get("R_bc")
    if value is None:
        return None
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("frame_calib.R_bc must be a 3x3 matrix or null")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(matrix), 1.0, atol=1e-4
    ):
        raise ValueError("frame_calib.R_bc is not a proper rotation matrix")
    return matrix


def measurement_frame(camera_frame: Mapping[str, Any]) -> np.ndarray | None:
    """Map the clip's initial ball axes into camera coordinates.

    ``R_bc`` remains the home-pose calibration. A clip starting at a different
    roll uses ``R_bc @ Rx(initial_roll_deg)`` for relative-motion decomposition.
    The offset rotates the reference frame; adding it to already decomposed
    angles would not correct spin/roll mixing.
    """
    from .rotation import Rx

    angle = float(camera_frame.get("initial_roll_deg", 0.0))
    if not np.isfinite(angle):
        raise ValueError("frame_calib.initial_roll_deg must be finite")
    sign = camera_frame.get("top_shell_sign", 1)
    if isinstance(sign, bool) or sign not in (-1, 1):
        raise ValueError("frame_calib.top_shell_sign must be +1 or -1")
    home = configured_frame(camera_frame)
    return None if home is None else home @ Rx(np.deg2rad(angle))


def update_yaml(path: str | Path, updates: Mapping[str, Any]) -> None:
    """Recursively update a YAML mapping and write it back atomically."""

    config_path = Path(path).expanduser().resolve()
    config, _ = load_config(config_path)
    merged = copy.deepcopy(config)
    _deep_update(merged, updates)
    temporary = config_path.with_suffix(config_path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(merged, sort_keys=False, default_flow_style=False), encoding="utf-8"
    )
    temporary.replace(config_path)


def _deep_update(target: MutableMapping[str, Any], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), MutableMapping):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
