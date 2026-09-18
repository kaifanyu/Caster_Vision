"""Strict V4L2 camera profiles and independent, receive-timestamped recording.

Nothing touches a device on import. A Brio 101 is fixed-focus; this module never
writes focus controls on that model. Host receive times are not exposure times.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any

import cv2
import numpy as np
import yaml

from .mjpeg_avi import MJPEGAVIWriter, SegmentedMJPEGAVIWriter, jpeg_dimensions


CAMERAS = ("c920", "brio101")
WARMUP_FRAMES = 5
WARMUP_TIMEOUT_S = 15.0
WORKER_STOP_TIMEOUT_S = 5.0
FINAL_VERIFY_TIMEOUT_S = 2.0


class CameraError(RuntimeError):
    """An unsupported profile, readback mismatch, or capture failure."""


def load_capture_config(path):
    path = Path(path).expanduser().resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or set(config.get("cameras", {})) != set(CAMERAS):
        raise CameraError("Rig YAML must contain cameras.c920 and cameras.brio101.")
    return config, path


def resolve_rig_path(config_path, value):
    """Rig paths are relative to the rig YAML, independent of working directory."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else Path(config_path).parent / p


def sha256_file(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def parse_controls(output):
    controls = {}
    current = None
    for line in output.splitlines():
        match = re.match(r"\s*(\w+)\s+0x[0-9a-f]+\s+\(([^)]+)\)\s*:\s*(.*)", line)
        if match:
            name, kind, rest = match.groups()
            current = {"type": kind, "menu": {}, "raw": rest}
            for key, val in re.findall(r"\b(min|max|step|default|value)=(-?\d+)", rest):
                current[key] = int(val)
            controls[name] = current
        elif current is not None:
            item = re.match(r"\s*(-?\d+):\s*(.*)", line)
            if item:
                current["menu"][int(item.group(1))] = item.group(2)
    return controls


def parse_formats(output):
    formats = []
    fourcc = size = None
    for line in output.splitlines():
        match = re.search(r"\[\d+\]:\s*'([^']+)'", line)
        if match:
            fourcc = match.group(1)
            size = None
        match = re.search(r"Size:\s+Discrete\s+(\d+)x(\d+)", line)
        if match:
            size = tuple(map(int, match.groups()))
        match = re.search(r"Interval:\s+Discrete.*\(([\d.]+) fps\)", line)
        if match and fourcc and size:
            formats.append({"fourcc": fourcc, "width": size[0], "height": size[1],
                            "fps": float(match.group(1))})
    return formats


def parse_active_format(output):
    size = re.search(r"Width/Height\s*:\s*(\d+)\s*/\s*(\d+)", output)
    fmt = re.search(r"Pixel Format\s*:\s*'([^']+)'", output)
    fps = re.search(r"Frames per second\s*:\s*([\d.]+)", output)
    if not size or not fmt or not fps:
        raise CameraError("Cannot read back V4L2 width, height, pixel format and frame rate.")
    return {"width": int(size.group(1)), "height": int(size.group(2)),
            "fourcc": fmt.group(1), "fps": float(fps.group(1))}


def validate_format(actual, requested):
    for name in ("width", "height", "fourcc"):
        if actual[name] != requested[name]:
            raise CameraError(f"Camera {name} mismatch: requested {requested[name]}, got {actual[name]}.")
    if not math.isclose(actual["fps"], float(requested["fps"]), abs_tol=0.1):
        raise CameraError(f"Camera FPS mismatch: requested {requested['fps']}, got {actual['fps']}.")


def validate_requested(profile):
    required = ("width", "height", "fps", "fourcc", "exposure_us", "gain", "white_balance_kelvin",
                "power_line_frequency")
    allowed = set(required) | {"focus", "zoom", "brightness", "contrast", "saturation",
                               "sharpness", "backlight_compensation"}
    unknown = set(profile) - allowed
    if unknown:
        raise CameraError(f"Unknown capture settings: {sorted(unknown)}; refusing to ignore them.")
    missing = [key for key in required if profile.get(key) is None]
    if missing:
        raise CameraError(f"Missing capture settings: {', '.join(missing)}.")
    for name in ("width", "height", "fps", "exposure_us"):
        if not isinstance(profile[name], (float, int)) or not math.isfinite(profile[name]) or profile[name] <= 0:
            raise CameraError(f"capture.{name} must be a positive finite number.")
    for name in ("width", "height"):
        if int(profile[name]) != profile[name]:
            raise CameraError(f"capture.{name} must be an integer.")
    if len(str(profile["fourcc"])) != 4:
        raise CameraError("capture.fourcc must have four characters, for example MJPG.")
    if profile["exposure_us"] % 100:
        raise CameraError("capture.exposure_us must be a multiple of 100 microseconds for V4L2.")
    if profile["exposure_us"] >= 1e6 / profile["fps"]:
        raise CameraError("Exposure must be shorter than the requested frame interval.")


class V4L2Camera:
    def __init__(self, name, camera_config, runner=subprocess.run):
        if name not in CAMERAS:
            raise CameraError(f"Unknown camera {name!r}.")
        self.name = name
        self.device = str(camera_config["device"])
        self.profile = dict(camera_config["capture"])
        self.runner = runner

    def command(self, *args, timeout=12):
        try:
            result = self.runner(["v4l2-ctl", "-d", self.device, *args], capture_output=True,
                                 text=True, check=False, timeout=timeout)
        except FileNotFoundError as error:
            raise CameraError("v4l2-ctl is required; install the v4l-utils package.") from error
        except subprocess.TimeoutExpired as error:
            raise CameraError(f"{self.name}: v4l2-ctl timed out.") from error
        if result.returncode:
            raise CameraError(f"{self.name} ({self.device}): {result.stderr.strip() or result.stdout.strip()} "
                              "Check USB connection, device path, permissions, and competing applications.")
        return result.stdout

    def inspect(self):
        controls_raw = self.command("--list-ctrls-menus")
        formats_raw = self.command("--list-formats-ext")
        return {"device": self.device, "controls": parse_controls(controls_raw),
                "formats": parse_formats(formats_raw), "controls_raw": controls_raw,
                "formats_raw": formats_raw}

    def plan_controls(self, controls):
        validate_requested(self.profile)
        plan = []
        optional_absent = []

        def add(aliases, value, optional=False):
            found = next((name for name in aliases if name in controls), None)
            if found is None:
                if optional:
                    optional_absent.append(aliases[0])
                    return
                raise CameraError(f"{self.name}: required control {' / '.join(aliases)} is unavailable; "
                                  "this requested profile cannot be applied.")
            info = controls[found]
            if "read-only" in info.get("raw", "") or "disabled" in info.get("raw", ""):
                raise CameraError(f"{self.name}: control {found} is not writable.")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
                raise CameraError(f"{self.name}: {found} must be an integer.")
            value = int(value)
            lower = info.get("min", 0 if info["type"] == "bool" else value)
            upper = info.get("max", 1 if info["type"] == "bool" else value)
            if not lower <= value <= upper or (value - lower) % info.get("step", 1):
                raise CameraError(f"{self.name}: {found}={value} outside range/step {info}.")
            if info["menu"] and value not in info["menu"]:
                raise CameraError(f"{self.name}: {found}={value} not in supported menu {info['menu']}.")
            plan.append((found, value))

        exposure_name = next((n for n in ("auto_exposure", "exposure_auto") if n in controls), None)
        if exposure_name is None:
            raise CameraError(f"{self.name}: no manual exposure selector exposed by this driver.")
        manual = next((v for v, label in controls[exposure_name]["menu"].items()
                       if "manual" in label.lower()), None)
        if manual is None:
            raise CameraError(f"{self.name}: manual exposure mode not advertised in the control menu.")
        add((exposure_name,), manual)
        # Manual exposure is fixed. Auto-exposure-priority is optional if the driver has no such control.
        add(("exposure_dynamic_framerate", "exposure_auto_priority"), 0, optional=True)
        add(("exposure_time_absolute", "exposure_absolute"), self.profile["exposure_us"] // 100)
        add(("gain",), self.profile["gain"])
        add(("white_balance_automatic", "white_balance_temperature_auto", "auto_white_balance"), 0)
        add(("white_balance_temperature",), self.profile["white_balance_kelvin"])
        add(("power_line_frequency",), self.profile["power_line_frequency"])
        if self.name == "c920":
            if self.profile.get("focus") is None:
                raise CameraError("c920: set cameras.c920.capture.focus to the manually chosen lens "
                                  "position before applying/recording; inspect controls and preview first.")
            add(("focus_automatic_continuous", "focus_auto"), 0)
            add(("focus_absolute",), self.profile["focus"])
            add(("zoom_absolute",), self.profile.get("zoom", 100))
        else:
            if self.profile.get("focus") is not None:
                raise CameraError("brio101 is fixed-focus: remove capture.focus; no focus controls will be written.")
            if "zoom" in self.profile:
                add(("zoom_absolute",), self.profile["zoom"])
        for name in ("brightness", "contrast", "saturation", "sharpness", "backlight_compensation"):
            if name in self.profile:
                add((name,), self.profile[name])
        return plan, optional_absent

    def prepare(self):
        capabilities = self.inspect()
        plan, absent = self.plan_controls(capabilities["controls"])
        supported = any(all(mode[key] == self.profile[key] for key in ("width", "height", "fourcc"))
                        and math.isclose(mode["fps"], self.profile["fps"], abs_tol=0.1)
                        for mode in capabilities["formats"])
        if not supported:
            raise CameraError(f"{self.name}: requested {self.profile['fourcc']} "
                              f"{self.profile['width']}x{self.profile['height']} at {self.profile['fps']} fps "
                              "is not advertised. Run camera_setup.py --inspect.")
        return plan, absent

    def verify_controls(self, plan, *, timeout=12):
        actual = parse_controls(self.command("--list-ctrls-menus", timeout=timeout))
        for name, expected in plan:
            value = actual.get(name, {}).get("value")
            if value != expected:
                raise CameraError(f"{self.name}: control {name} readback mismatch: requested {expected}, got {value}.")
        return {name: item.get("value") for name, item in actual.items()}

    def apply(self, *, configure_format=True, prepared=None):
        plan, absent = prepared or self.prepare()
        p = self.profile
        if configure_format:
            self.command(f"--set-fmt-video=width={p['width']},height={p['height']},pixelformat={p['fourcc']}")
            self.command(f"--set-parm={p['fps']}")
        # Apply individually: manual switches must activate dependent controls before their write.
        for name, value in plan:
            self.command(f"--set-ctrl={name}={value}")
        actual = parse_active_format(self.command("--get-fmt-video", "--get-parm"))
        validate_format(actual, p)
        return {"device": self.device, "requested": dict(p), "actual_format": actual,
                "controls": self.verify_controls(plan), "expected_controls": dict(plan),
                "optional_controls_not_exposed": absent,
                "focus_type": "fixed" if self.name == "brio101" else "manual"}


def capture_format(capture):
    code = int(capture.get(cv2.CAP_PROP_FOURCC))
    return {"width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "fourcc": "".join(chr((code >> (8 * i)) & 255) for i in range(4))}


def configure_open_capture(capture, camera, prepared=None, *, encoded=False):
    if not capture.isOpened():
        raise CameraError(f"Could not open {camera.device}. Close other applications using the device.")
    p = camera.profile
    for prop, value in ((cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*p["fourcc"])),
                        (cv2.CAP_PROP_FRAME_WIDTH, p["width"]),
                        (cv2.CAP_PROP_FRAME_HEIGHT, p["height"]),
                        (cv2.CAP_PROP_FPS, p["fps"])):
        capture.set(prop, value)
    # Four buffers provide queue headroom on the measured rig. Receive timestamps
    # still include unknown USB/driver buffering; this does not guarantee integrity.
    buffer_requested = bool(capture.set(cv2.CAP_PROP_BUFFERSIZE, 4))
    reported_buffers = float(capture.get(cv2.CAP_PROP_BUFFERSIZE))
    actual_buffers = int(reported_buffers) if math.isfinite(reported_buffers) and reported_buffers > 0 else None
    if encoded:
        if actual_buffers is not None and actual_buffers < 2:
            raise CameraError(f"{camera.name}: raw MJPEG needs at least two V4L2 buffers to avoid capture starvation.")
        if p["fourcc"] != "MJPG":
            raise CameraError("Compressed passthrough requires the camera's MJPG capture format.")
        if not capture.set(cv2.CAP_PROP_CONVERT_RGB, 0):
            raise CameraError(f"{camera.name}: backend cannot disable RGB conversion for MJPEG passthrough.")
        if capture.get(cv2.CAP_PROP_CONVERT_RGB) != 0:
            raise CameraError(f"{camera.name}: RGB conversion remains enabled; raw MJPEG recording is unavailable.")
    prepared = prepared if prepared is not None else camera.prepare()
    before_stream = camera.apply(configure_format=False, prepared=prepared)
    negotiated = capture_format(capture)
    validate_format(negotiated, p)
    # C920 firmware can reset exposure on the first STREAMON, after all initial
    # setters and readbacks succeeded. Start streaming before the final apply.
    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        raise CameraError(f"{camera.name}: first streaming frame could not be read.")
    validate_frame(first_frame, camera, encoded=encoded)
    profile = camera.apply(configure_format=False, prepared=prepared)
    profile["opencv_format"] = capture_format(capture)
    validate_format(profile["opencv_format"], p)
    profile["controls_before_stream"] = before_stream["controls"]
    profile["controls_applied_while_streaming"] = True
    profile["buffer_size_requested"] = 4
    profile["buffer_size_set_accepted"] = buffer_requested
    profile["buffer_size_actual"] = actual_buffers
    profile["frame_representation"] = "original_camera_jpeg" if encoded else "decoded_bgr"
    return profile


def validate_frame(frame, camera, *, encoded=False):
    """Check resolution without decoding when the caller explicitly requested JPEGs."""
    if encoded:
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim not in (1, 2) or (frame.ndim == 2 and 1 not in frame.shape)):
            raise CameraError(f"{camera.name}: raw MJPEG capture returned a decoded/non-byte frame.")
        try:
            dimensions = jpeg_dimensions(frame)
        except (TypeError, ValueError) as error:
            raise CameraError(f"{camera.name}: invalid raw MJPEG frame: {error}") from error
    else:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise CameraError(f"{camera.name}: expected a decoded BGR image.")
        dimensions = (frame.shape[1], frame.shape[0])
    expected = (camera.profile["width"], camera.profile["height"])
    if dimensions != expected:
        raise CameraError(f"{camera.name}: frame dimensions changed to {dimensions}; expected {expected}, no resizing allowed.")


def warmup_capture(capture, camera, expected_controls, stop_event=None, *, encoded=False):
    """Drain frames acquired after streaming controls were applied, then verify."""
    for _ in range(WARMUP_FRAMES):
        if stop_event is not None and stop_event.is_set():
            return None
        ok, frame = capture.read()
        if not ok or frame is None:
            raise CameraError(f"{camera.name}: read failed while warming up.")
        validate_frame(frame, camera, encoded=encoded)
    return camera.verify_controls(expected_controls)


def check_intrinsics_profile(config_path, camera_config):
    path = resolve_rig_path(config_path, camera_config["intrinsics"])
    warnings = []
    profile = None
    if path.is_file():
        data = yaml.safe_load(path.read_text()) or {}
        size = data.get("image_size")
        if size is not None and list(size) != [camera_config["capture"]["width"], camera_config["capture"]["height"]]:
            raise CameraError(f"{path}: intrinsics image_size does not match the requested capture size.")
        profile = data.get("capture_profile")
        if profile is not None:
            requested = camera_config["capture"]
            if not isinstance(profile, dict) or not profile:
                raise CameraError(f"{path}: capture_profile must be the complete nonempty capture mapping "
                                  "from calibration, or null when that metadata is unknown.")
            missing = set(requested) - set(profile)
            extra = set(profile) - set(requested)
            if missing or extra:
                raise CameraError(f"{path}: capture_profile keys differ from the requested capture profile; "
                                  f"missing calibrated settings={sorted(missing)}, "
                                  f"settings removed from rig={sorted(extra)}. Restore the measured complete "
                                  "profile or explicitly verify and document the changed setup.")
            for name, expected in profile.items():
                if requested[name] != expected:
                    raise CameraError(f"{path}: calibrated capture_profile.{name}={expected!r} differs "
                                      f"from rig setting {requested[name]!r}.")
        if data.get("K") is None:
            warnings.append("Intrinsics have not been filled in; recording is allowed for calibration.")
    else:
        warnings.append("Intrinsics file is absent; recording is allowed for calibration.")
    if profile is None:
        warnings.append("Intrinsics capture_profile is absent; settings consistency is not verified.")
    return {"intrinsics_path": str(path), "intrinsics_sha256": sha256_file(path),
            "intrinsics_capture_profile_verified": profile is not None, "warnings": warnings}


def timing_statistics(times, requested_fps):
    values = np.asarray(times, dtype=float)
    intervals = np.diff(values)
    return {"frames": len(values), "first_timestamp_s": float(values[0]) if len(values) else None,
            "last_timestamp_s": float(values[-1]) if len(values) else None,
            "observed_fps": float((len(values) - 1) / (values[-1] - values[0])) if len(values) > 1 else None,
            "median_interval_ms": float(np.median(intervals) * 1000) if len(intervals) else None,
            "max_interval_ms": float(np.max(intervals) * 1000) if len(intervals) else None,
            "long_intervals": int(np.sum(intervals > 1.5 / requested_fps)),
            "estimated_missing_frames": int(np.maximum(np.rint(intervals * requested_fps) - 1, 0).sum()),
            "drop_estimate_note": "Host interval estimate; camera frame counters are unavailable."}


def record_session(config, config_path, output, duration, mode, stop_event=None, *,
                   max_video_file_bytes=None):
    """Record both cameras for seconds, or until ``stop_event`` when duration is None.

    File rotation is opt-in and requires native MJPG capture. Each manifest
    segment uses global camera-frame indices, matching the single timestamp CSV.
    """
    if (duration is not None and
            (isinstance(duration, bool) or not isinstance(duration, (int, float))
             or not math.isfinite(duration) or duration <= 0)):
        raise CameraError("--duration must be a positive finite number of seconds.")
    if max_video_file_bytes is not None:
        if (isinstance(max_video_file_bytes, bool) or not isinstance(max_video_file_bytes, int)
                or max_video_file_bytes <= 0):
            raise CameraError("max_video_file_bytes must be a positive integer number of bytes.")
        if any(config["cameras"][name]["capture"]["fourcc"] != "MJPG" for name in CAMERAS):
            raise CameraError("Segmented recording requires MJPG capture on both cameras.")
    if mode not in ("roll", "swivel", "motion", "checkerboard", "timing"):
        raise CameraError("Unknown recording mode.")
    output = Path(output)
    if output.exists():
        raise CameraError(f"Refusing to overwrite existing session directory: {output}")
    cameras = {name: V4L2Camera(name, config["cameras"][name]) for name in CAMERAS}
    # Preflight both profiles before opening or writing either camera.
    prepared = {name: cam.prepare() for name, cam in cameras.items()}
    calibration = {name: check_intrinsics_profile(config_path, config["cameras"][name]) for name in CAMERAS}
    output.mkdir(parents=True, exist_ok=False)
    stop = stop_event if stop_event is not None else threading.Event()
    start = threading.Event()
    ready = {name: threading.Event() for name in CAMERAS}
    captures = {}
    writers = {}
    threads = []
    joined_threads = set()
    times = {name: [] for name in CAMERAS}
    errors = []
    session = {"schema_version": 1, "mode": mode, "status": "starting",
               "requested_duration_s": duration, "config_snapshot": config,
               "continuous": duration is None, "max_video_file_bytes": max_video_file_bytes,
               "config_sha256": sha256_file(config_path),
               "calibration_sha256": {name: sha256_file(resolve_rig_path(config_path, config[name]["path"]))
                                      for name in ("stereo", "axes") if config.get(name, {}).get("path")},
               "cameras": {},
               "timing": config.get("timing", {}), "timestamp_source": "host_monotonic_read_return",
               "timestamp_note": "Receive timestamps share one clock; unknown camera/USB buffering and "
                                 "exposure latency remain. These webcams are not hardware synchronized.",
               "errors": errors}
    origin = None

    def save_manifest():
        # Readers see either complete manifest version, including during a long run.
        temporary = output / "session.json.tmp"
        temporary.write_text(json.dumps(session, indent=2) + "\n")
        temporary.replace(output / "session.json")

    def stop_workers():
        stop.set()
        start.set()
        deadline = time.monotonic() + WORKER_STOP_TIMEOUT_S
        for thread in threads:
            if thread not in joined_threads:
                thread.join(timeout=max(0., deadline-time.monotonic()))
                joined_threads.add(thread)
                if thread.is_alive():
                    errors.append(f"{thread.name}: camera read did not stop within {WORKER_STOP_TIMEOUT_S:g} seconds.")

    def worker(name):
        cap, writer = captures[name], writers[name]
        p = cameras[name].profile
        try:
            # Both streams must finish warmup and readback before the shared clock starts.
            encoded = p["fourcc"] == "MJPG"
            actual = warmup_capture(cap, cameras[name], prepared[name][0], stop, encoded=encoded)
            if actual is None:
                return
            session["cameras"][name]["warmup_controls"] = actual
            session["cameras"][name]["warmup_frames"] = WARMUP_FRAMES
            ready[name].set()
            start.wait()
            with (output / f"{name}_timestamps.csv").open("w", newline="") as stream:
                table = csv.writer(stream)
                table.writerow(["frame_index", "timestamp_s", "read_start_s", "read_end_s"])
                while not stop.is_set():
                    before = time.monotonic()
                    ok, frame = cap.read()
                    after = time.monotonic()
                    if not ok or frame is None:
                        raise CameraError(f"{name}: camera read failed; session ended.")
                    if stop.is_set() or (duration is not None and after - origin > duration):
                        break
                    validate_frame(frame, cameras[name], encoded=encoded)
                    writer.write(frame)
                    stamp = after - origin
                    table.writerow([len(times[name]), f"{stamp:.9f}", f"{before - origin:.9f}", f"{stamp:.9f}"])
                    times[name].append(stamp)
                    if max_video_file_bytes is None:
                        session["cameras"][name]["video_segments"][0]["frame_count"] = len(times[name])
        except Exception as error:
            errors.append(str(error))
            stop.set()
        finally:
            for label, resource in (("video file", writer), ("camera", cap)):
                try:
                    resource.release()
                except Exception as error:
                    errors.append(f"{name}: failed to finalize {label}: {error}")
                    stop.set()

    try:
        cv2.setNumThreads(1)
        # Open both before applying either profile: backend opening can reset camera controls.
        for name, camera in cameras.items():
            captures[name] = cv2.VideoCapture(camera.device, cv2.CAP_V4L2)
        for name, cap in captures.items():
            if not cap.isOpened():
                raise CameraError(f"Could not open {cameras[name].device}. Close competing camera applications.")
        for name, camera in cameras.items():
            result = configure_open_capture(captures[name], camera, prepared[name],
                                            encoded=camera.profile["fourcc"] == "MJPG")
            result.update(calibration[name])
            session["cameras"][name] = result
            p = camera.profile
            if p["fourcc"] == "MJPG":
                if max_video_file_bytes is None:
                    writer = MJPEGAVIWriter(output / f"{name}.avi", p["fps"], (p["width"], p["height"]))
                else:
                    writer = SegmentedMJPEGAVIWriter(output / f"{name}.avi", p["fps"],
                                                     (p["width"], p["height"]),
                                                     max_file_size=max_video_file_bytes)
                result["recording_storage"] = "camera_jpeg_passthrough_avi"
            else:
                writer = cv2.VideoWriter(str(output / f"{name}.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                         p["fps"], (p["width"], p["height"]))
                result["recording_storage"] = "decoded_then_mjpeg_reencoded"
            writers[name] = writer
            result["video_segments"] = (writer.segments if max_video_file_bytes is not None else
                                        [{"path": f"{name}.avi", "start_frame": 0, "frame_count": 0}])
            if not writer.isOpened():
                raise CameraError(f"Could not open MJPG video writer for {name}.")
        session["status"] = "warming_up"
        save_manifest()
        warmup_started = time.monotonic()
        for name in CAMERAS:
            thread = threading.Thread(target=worker, args=(name,), name=f"capture-{name}", daemon=True)
            thread.start()
            threads.append(thread)
        while not all(event.is_set() for event in ready.values()):
            if stop.is_set():
                raise CameraError("Recording stopped before both camera warmups completed.")
            if time.monotonic() - warmup_started > WARMUP_TIMEOUT_S:
                pending = [name for name, event in ready.items() if not event.is_set()]
                raise CameraError(f"Camera warmup timed out after {WARMUP_TIMEOUT_S:g}s: {pending}")
            time.sleep(0.01)
        if stop.is_set():
            raise CameraError("Recording stopped before its timed interval began.")
        anchor_before = time.monotonic()
        unix_anchor = time.time()
        anchor_after = time.monotonic()
        origin = (anchor_before + anchor_after) / 2
        session["host_monotonic_zero_s"] = origin
        session["started_unix_s"] = unix_anchor
        session["clock_anchor"] = {"host_monotonic_before_s": anchor_before,
                                   "unix_s": unix_anchor,
                                   "host_monotonic_after_s": anchor_after,
                                   "host_monotonic_midpoint_s": origin,
                                   "uncertainty_s": (anchor_after-anchor_before)/2,
                                   "note": "Host wall-clock/monotonic anchor; not camera exposure time or hardware synchronization."}
        session["warmup_elapsed_s"] = origin - warmup_started
        session["status"] = "recording"
        save_manifest()
        interval = "until stopped" if duration is None else f"for {duration:g} seconds"
        print(f"Recording NOW {interval} ({mode}); camera warmup is complete.", flush=True)
        start.set()
        next_manifest = origin + 5
        while (duration is None or time.monotonic() - origin < duration) and not stop.is_set():
            if time.monotonic() >= next_manifest:
                save_manifest()
                next_manifest = time.monotonic() + 5
            time.sleep(0.02)
        session["recording_elapsed_s"] = time.monotonic() - origin
        interrupted = duration is not None and stop.is_set() and not errors
        stop_workers()
        for name, camera in cameras.items():
            if not any(t.is_alive() for t in threads):
                session["cameras"][name]["final_controls"] = camera.verify_controls(
                    prepared[name][0], timeout=FINAL_VERIFY_TIMEOUT_S)
        if not all(times.values()):
            errors.append("At least one camera recorded no frames.")
        session["status"] = "failed" if errors else "interrupted" if interrupted else "complete"
    except BaseException as error:
        stop.set()
        start.set()
        errors.append(str(error) or type(error).__name__)
        session["status"] = "failed"
    finally:
        # Also join workers if opening another worker or an intervening operation failed.
        stop_workers()
        if errors:
            session["status"] = "failed"
        # A live blocked worker owns its capture/writer; avoid release racing with read().
        for name in captures:
            if not any(t.name == f"capture-{name}" and t.is_alive() for t in threads):
                for label, resource in (("camera", captures[name]), ("video file", writers.get(name))):
                    if resource is not None:
                        try:
                            resource.release()
                        except Exception as error:
                            errors.append(f"{name}: failed to finalize {label}: {error}")
        if errors:
            session["status"] = "failed"
        session["elapsed_s"] = time.monotonic() - origin if origin is not None else 0
        session["finished_unix_s"] = time.time()
        session["stats"] = {name: timing_statistics(times[name], cameras[name].profile["fps"]) for name in CAMERAS}
        save_manifest()
    if errors:
        raise CameraError(f"Session failed (details saved in {output / 'session.json'}): {'; '.join(errors)}")
    return session
