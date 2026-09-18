"""Validated configuration, calibration provenance, and portable serialization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from ballrot.camera import CameraModel

CAMERA_NAMES = ("c920", "brio101")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "rig.yaml"


def read_yaml(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration_hashes_match(paths, expected):
    """Check text-file provenance, allowing only LF/CRLF conversion.

    Keep saved hashes byte-exact for compatibility with existing calibrations.
    Git and cross-platform copies can change newlines without changing any
    calibration content; try both newline conventions against the saved hash.
    """
    if not isinstance(expected, dict) or expected.keys() != paths.keys():
        return False
    for name, path in paths.items():
        data = Path(path).read_bytes()
        lf = data.replace(b"\r\n", b"\n")
        if not any(hashlib.sha256(candidate).hexdigest() == expected[name]
                   for candidate in (data, lf, lf.replace(b"\n", b"\r\n"))):
            return False
    return True


def rotation(value, label):
    a = np.asarray(value, dtype=float)
    if (a.shape != (3, 3) or not np.isfinite(a).all()
            or not np.allclose(a.T @ a, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(a), 1, atol=1e-6)):
        raise ValueError(f"{label} must be a measured, proper 3x3 rotation matrix")
    return a


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).expanduser().resolve()
    cfg = read_yaml(path)
    if set(cfg.get("cameras", {})) != set(CAMERA_NAMES):
        raise ValueError("cameras must contain exactly c920 and brio101")
    for name in CAMERA_NAMES:
        camera = cfg["cameras"][name]
        camera["intrinsics"] = str((path.parent / camera["intrinsics"]).resolve())
        capture = camera["capture"]
        for key in ("width", "height", "fps"):
            value = capture[key]
            if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} capture dimensions/fps must be positive and finite")
            if key != "fps" and int(value) != value:
                raise ValueError(f"{name} capture pixel dimensions must be integers")
        if name == "brio101" and capture.get("focus") is not None:
            raise ValueError("Brio101 is fixed-focus; remove its focus setting")
    for name in ("stereo", "axes"):
        cfg[name]["path"] = str((path.parent / cfg[name]["path"]).resolve())
    geo = cfg["geometry"]
    if not (np.isfinite(geo["radius_m"]) and float(geo["radius_m"]) > 0):
        raise ValueError("geometry.radius_m must be a positive measured radius in meters")
    if not (np.isfinite(geo["gap_m"]) and 0 <= float(geo["gap_m"]) < 2*geo["radius_m"]):
        raise ValueError("geometry.gap_m must be nonnegative and less than shell diameter")
    if geo.get("red_shell_sign", 1) not in (-1, 1):
        raise ValueError("geometry.red_shell_sign must be +1 or -1")
    timing = cfg["timing"]
    if not np.isfinite(timing.get("brio_offset_s", 0)):
        raise ValueError("timing.brio_offset_s must be finite")
    skew = timing.get("max_pair_skew_ms", 12)
    if not np.isfinite(skew) or not 0 < skew < 1000/min(c["capture"]["fps"] for c in cfg["cameras"].values())/2:
        raise ValueError("max_pair_skew_ms must be positive and below half a frame interval")
    cfg["_config_path"] = str(path)
    return cfg


def load_intrinsics(camera_cfg):
    path = camera_cfg["intrinsics"]
    data = read_yaml(path)
    if data.get("K") is None or data.get("dist") is None:
        raise ValueError(f"Fill measured K and dist in {path}, or run calibrate_intrinsics.py")
    camera = CameraModel(data["K"], data["dist"])
    capture = camera_cfg["capture"]
    expected = [int(capture["width"]), int(capture["height"])]
    if data.get("image_size") != expected:
        raise ValueError(f"{path}: image_size must equal capture resolution {expected}; do not silently rescale K")
    profile = data.get("capture_profile")
    if profile is not None:
        if not isinstance(profile, dict):
            raise ValueError(f"{path}: capture_profile must be a mapping")
        differences = [k for k in sorted(set(profile) | set(capture))
                       if k not in profile or k not in capture or profile[k] != capture[k]]
        if differences:
            raise ValueError(f"{path}: calibrated capture_profile differs from rig: {differences}")
    return {**data, "K": camera.K, "dist": camera.dist,
            "sha256": sha256(path), "path": path,
            "profile_verified": profile is not None}


def load_rig(cfg):
    intrinsics = [load_intrinsics(cfg["cameras"][name]) for name in CAMERA_NAMES]
    stereo = read_yaml(cfg["stereo"]["path"])
    intrinsic_paths = {name: info["path"] for name, info in zip(CAMERA_NAMES, intrinsics)}
    if stereo.get("intrinsics_sha256") is not None and not calibration_hashes_match(intrinsic_paths, stereo["intrinsics_sha256"]):
        raise ValueError("Stereo calibration is stale: its intrinsic-file hashes differ. Recalibrate stereo before fitting axes or motion.")
    if stereo.get("R_21") is None or stereo.get("t_21_m") is None:
        raise ValueError("Calibrate stereo R_21 and t_21_m with calibrate_stereo.py before fitting motion")
    R = rotation(stereo["R_21"], "stereo.R_21")
    t = np.asarray(stereo["t_21_m"], dtype=float)
    if t.shape != (3,) or not np.isfinite(t).all() or np.linalg.norm(t) < 1e-5:
        raise ValueError("stereo.t_21_m must be a nonzero, finite camera translation in meters")
    cameras = [{"K": intrinsics[0]["K"], "R": np.eye(3), "t": np.zeros(3)},
               {"K": intrinsics[1]["K"], "R": R, "t": t}]
    return cameras, intrinsics, stereo


def jsonable(value):
    """JSON-safe data: missing estimates are null, never nonstandard NaN tokens."""
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_yaml(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(yaml.safe_dump(jsonable(value), sort_keys=False), encoding="utf-8")
    temporary.replace(path)
