import copy
import csv
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from dualcam.capture import (CameraError, V4L2Camera, check_intrinsics_profile,
                             configure_open_capture, parse_controls, record_session, validate_frame)
from dualcam.mjpeg_avi import SegmentedMJPEGAVIWriter
from dualcam.session import SelectedVideo


CONTROLS = """
 white_balance_automatic 0x0098090c (bool) : default=1 value=1
 gain 0x00980913 (int) : min=0 max=255 step=1 default=0 value=180
 power_line_frequency 0x00980918 (menu) : min=0 max=2 default=2 value=2
   0: Disabled
   1: 50 Hz
   2: 60 Hz
 white_balance_temperature 0x0098091a (int) : min=2000 max=6500 step=1 value=3272 flags=inactive
 auto_exposure 0x009a0901 (menu) : min=0 max=3 default=3 value=3
   1: Manual Mode
   3: Aperture Priority Mode
 exposure_time_absolute 0x009a0902 (int) : min=3 max=2047 step=1 value=77
 exposure_dynamic_framerate 0x009a0903 (bool) : default=0 value=1
 focus_absolute 0x009a090a (int) : min=0 max=250 step=5 value=50
 focus_automatic_continuous 0x009a090c (bool) : default=1 value=1
 zoom_absolute 0x009a090d (int) : min=100 max=500 step=1 value=100
"""
FORMATS = """
 [0]: 'YUYV' (YUYV 4:2:2)
   Size: Discrete 1920x1080
     Interval: Discrete 0.200s (5.000 fps)
 [1]: 'MJPG' (Motion-JPEG)
   Size: Discrete 1920x1080
     Interval: Discrete 0.033s (30.000 fps)
"""
PROFILE = {"width": 1920, "height": 1080, "fps": 30, "fourcc": "MJPG", "exposure_us": 4000,
           "gain": 0, "white_balance_kelvin": 4000, "power_line_frequency": 2}


class FakeV4L2:
    def __init__(self, text=CONTROLS, ignore=None):
        self.controls = parse_controls(text)
        self.calls = []
        self.ignore = ignore

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        arg = command[3]
        if arg.startswith("--set-ctrl="):
            name, value = arg.split("=", 1)[1].split("=")
            if name != self.ignore:
                self.controls[name]["value"] = int(value)
            output = ""
        elif arg == "--list-formats-ext":
            output = FORMATS
        elif arg == "--get-fmt-video":
            output = "Width/Height : 1920/1080\nPixel Format : 'MJPG'\nFrames per second : 30.000 (30/1)\n"
        elif arg == "--list-ctrls-menus":
            output = ""
            for name, control in self.controls.items():
                fields = " ".join(f"{key}={control[key]}" for key in ("min", "max", "step", "value") if key in control)
                output += f"{name} 0x00000000 ({control['type']}) : {fields}\n"
                output += "".join(f"  {key}: {value}\n" for key, value in control["menu"].items())
        else:
            output = ""
        return SimpleNamespace(stdout=output, stderr="", returncode=0)


def camera_config(name):
    profile = copy.deepcopy(PROFILE)
    if name == "c920":
        profile.update(focus=55, zoom=100)
    return {"device": f"/dev/fake-{name}", "capture": profile, "intrinsics": f"../calibration/{name}.yaml"}


class FakeCapture:
    def __init__(self, *args, mismatch=False):
        self.props = {}
        self.mismatch = mismatch
        self.released = False
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ok, data = cv2.imencode(".jpg", self.frame)
        assert ok
        self.jpeg = data.reshape(1, -1)

    def isOpened(self):
        return True

    def set(self, key, val):
        self.props[key] = val
        return True

    def get(self, key):
        if self.mismatch and key == cv2.CAP_PROP_FRAME_WIDTH:
            return 640
        return self.props[key]

    def read(self):
        time.sleep(.003)
        return True, self.jpeg if self.props.get(cv2.CAP_PROP_CONVERT_RGB, 1) == 0 else self.frame

    def release(self):
        self.released = True


class StreamingResetCapture(FakeCapture):
    def __init__(self, runner, reset_on=(1,), slow_warmup=False):
        super().__init__()
        self.runner = runner
        self.reset_on = set(reset_on)
        self.read_times = []
        self.slow_warmup = slow_warmup

    def read(self):
        index = len(self.read_times) + 1
        time.sleep(.015 if self.slow_warmup and index <= 6 else .002)
        if index in self.reset_on:
            self.runner.controls["exposure_time_absolute"]["value"] = 312
        self.read_times.append(time.monotonic())
        return True, self.jpeg if self.props.get(cv2.CAP_PROP_CONVERT_RGB, 1) == 0 else self.frame


class FakeWriter:
    def __init__(self, *args):
        self.frames = 0
        self.released = False

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames += 1

    def release(self):
        self.released = True


class CaptureTests(unittest.TestCase):
    def test_invalid_duration_and_segment_limit_fail_before_touching_devices(self):
        config = {"cameras": {name: camera_config(name) for name in ("c920", "brio101")}}
        with patch("dualcam.capture.V4L2Camera") as camera:
            for duration in (0, -1, float("inf"), float("nan"), True, "10"):
                with self.subTest(duration=duration), self.assertRaises(CameraError):
                    record_session(config, "unused.yaml", "unused-session", duration, "motion")
            for limit in (0, -1, 1.5, True):
                with self.subTest(limit=limit), self.assertRaises(CameraError):
                    record_session(config, "unused.yaml", "unused-session", None, "motion",
                                   max_video_file_bytes=limit)
            camera.assert_not_called()

    def test_continuous_stop_rotates_files_and_finalizes_one_global_timestamp_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"cameras": {name: camera_config(name) for name in ("c920", "brio101")}}
            config_path = root/"rig.yaml"
            config_path.write_text(yaml.safe_dump(config))
            runners = {name: FakeV4L2() for name in config["cameras"]}
            cameras = {name: V4L2Camera(name, cfg, runners[name]) for name, cfg in config["cameras"].items()}
            captures = [FakeCapture(), FakeCapture()]
            stop, writers = threading.Event(), []
            limit = 256+3*(len(captures[0].jpeg.reshape(-1))+24)

            def make_writer(*args, **kwargs):
                writer = SegmentedMJPEGAVIWriter(*args, **kwargs)
                original_write = writer.write
                def write_and_stop(frame):
                    original_write(frame)
                    if len(writers) == 2 and all(item.frames >= 9 for item in writers):
                        stop.set()
                writer.write = write_and_stop
                writers.append(writer)
                return writer

            with patch("dualcam.capture.V4L2Camera", side_effect=lambda name, cfg: cameras[name]), \
                 patch("dualcam.capture.cv2.VideoCapture", side_effect=captures), \
                 patch("dualcam.capture.SegmentedMJPEGAVIWriter", side_effect=make_writer):
                session = record_session(config, config_path, root/"session", None, "motion", stop,
                                         max_video_file_bytes=limit)
            self.assertEqual(session["status"], "complete")
            self.assertTrue(session["continuous"])
            self.assertIsNone(session["requested_duration_s"])
            self.assertFalse(session["errors"])
            self.assertTrue(all(cap.released for cap in captures))
            self.assertTrue(all(not writer.isOpened() for writer in writers))
            anchor = session["clock_anchor"]
            self.assertLessEqual(anchor["host_monotonic_before_s"], session["host_monotonic_zero_s"])
            self.assertGreaterEqual(anchor["host_monotonic_after_s"], session["host_monotonic_zero_s"])
            self.assertEqual(session["started_unix_s"], anchor["unix_s"])
            saved = json.loads((root/"session/session.json").read_text())
            self.assertEqual(saved["status"], "complete")
            for name, source in zip(cameras, captures):
                rows = list(csv.DictReader((root/f"session/{name}_timestamps.csv").open()))
                segments = saved["cameras"][name]["video_segments"]
                self.assertGreaterEqual(len(segments), 3)
                self.assertEqual([int(row["frame_index"]) for row in rows], list(range(len(rows))))
                self.assertEqual(sum(part["frame_count"] for part in segments), len(rows))
                self.assertEqual(saved["stats"][name]["frames"], len(rows))
                for part in segments:
                    self.assertLessEqual((root/"session"/part["path"]).stat().st_size, limit)
                video = SelectedVideo(root/f"session/{name}.avi", [1920, 1080])
                try:
                    for index in range(len(rows)):
                        self.assertEqual(video.read(index).shape, source.frame.shape)
                finally:
                    video.close()

    def test_continuous_camera_failure_finalizes_prior_segments_and_failed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"cameras": {name: camera_config(name) for name in ("c920", "brio101")}}
            config_path = root/"rig.yaml"
            config_path.write_text(yaml.safe_dump(config))
            cameras = {name: V4L2Camera(name, cfg, FakeV4L2()) for name, cfg in config["cameras"].items()}
            captures = [FakeCapture(), FakeCapture()]
            original_read, reads = captures[0].read, []
            def eventually_fail():
                result = original_read()
                reads.append(True)
                return (False, None) if len(reads) > 13 else result
            captures[0].read = eventually_fail
            limit = 256+2*(len(captures[0].jpeg.reshape(-1))+24)
            with patch("dualcam.capture.V4L2Camera", side_effect=lambda name, cfg: cameras[name]), \
                 patch("dualcam.capture.cv2.VideoCapture", side_effect=captures):
                with self.assertRaisesRegex(CameraError, "camera read failed"):
                    record_session(config, config_path, root/"session", None, "motion",
                                   max_video_file_bytes=limit)
            saved = json.loads((root/"session/session.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertTrue(all(cap.released for cap in captures))
            self.assertGreater(len(saved["cameras"]["c920"]["video_segments"]), 1)
            for name in cameras:
                rows = list(csv.DictReader((root/f"session/{name}_timestamps.csv").open()))
                parts = saved["cameras"][name]["video_segments"]
                self.assertEqual(sum(part["frame_count"] for part in parts), len(rows))
                for part in parts:
                    video = cv2.VideoCapture(str(root/"session"/part["path"]))
                    try:
                        self.assertTrue(video.isOpened())
                        self.assertEqual(round(video.get(cv2.CAP_PROP_FRAME_COUNT)), part["frame_count"])
                    finally:
                        video.release()

    def test_raw_capture_requires_jpeg_bytes_and_checks_header_dimensions(self):
        runner = FakeV4L2()
        camera = V4L2Camera("c920", camera_config("c920"), runner)
        cap = FakeCapture()
        profile = configure_open_capture(cap, camera, encoded=True)
        self.assertEqual(cap.get(cv2.CAP_PROP_CONVERT_RGB), 0)
        self.assertEqual(profile["buffer_size_requested"], 4)
        self.assertEqual(profile["buffer_size_actual"], 4)
        self.assertEqual(profile["frame_representation"], "original_camera_jpeg")
        with self.assertRaisesRegex(CameraError, "decoded/non-byte"):
            validate_frame(cap.frame, camera, encoded=True)
        with self.assertRaisesRegex(CameraError, "invalid raw MJPEG"):
            validate_frame(np.array([[1, 2, 3]], np.uint8), camera, encoded=True)
        ok, wrong = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))
        self.assertTrue(ok)
        with self.assertRaisesRegex(CameraError, "dimensions changed"):
            validate_frame(wrong.reshape(1, -1), camera, encoded=True)

    def test_raw_recording_uses_real_muxer_without_jpeg_decode_or_reencode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"cameras": {name: camera_config(name) for name in ("c920", "brio101")}}
            config_path = root / "rig.yaml"
            config_path.write_text(yaml.safe_dump(config))
            cameras = {name: V4L2Camera(name, cfg, FakeV4L2()) for name, cfg in config["cameras"].items()}
            captures = [FakeCapture(), FakeCapture()]
            with patch("dualcam.capture.V4L2Camera", side_effect=lambda name, cfg: cameras[name]), \
                 patch("dualcam.capture.cv2.VideoCapture", side_effect=captures), \
                 patch("dualcam.capture.cv2.VideoWriter", side_effect=AssertionError("No re-encoding allowed")), \
                 patch("dualcam.capture.cv2.imencode", side_effect=AssertionError("No JPEG encoding allowed")), \
                 patch("dualcam.capture.cv2.imdecode", side_effect=AssertionError("No JPEG decoding allowed")):
                session = record_session(config, config_path, root / "session", .05, "motion")
            self.assertEqual(session["status"], "complete")
            for name, source in zip(cameras, captures):
                self.assertEqual(session["cameras"][name]["recording_storage"], "camera_jpeg_passthrough_avi")
                content = (root / f"session/{name}.avi").read_bytes()
                self.assertEqual(content.count(source.jpeg.tobytes()), session["stats"][name]["frames"])
                video = cv2.VideoCapture(str(root / f"session/{name}.avi"))
                try:
                    self.assertEqual(round(video.get(cv2.CAP_PROP_FRAME_COUNT)), session["stats"][name]["frames"])
                    ok, decoded = video.read()
                    self.assertTrue(ok)
                    self.assertEqual(decoded.shape, (1080, 1920, 3))
                finally:
                    video.release()

    def test_video_finalization_failure_marks_session_failed_and_releases_cameras(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"cameras": {name: camera_config(name) for name in ("c920", "brio101")}}
            config_path = root / "rig.yaml"
            config_path.write_text(yaml.safe_dump(config))
            cameras = {name: V4L2Camera(name, cfg, FakeV4L2()) for name, cfg in config["cameras"].items()}
            captures, writers = [FakeCapture(), FakeCapture()], [FakeWriter(), FakeWriter()]

            def fail_release():
                raise OSError("simulated full disk on index finalization")

            writers[0].release = fail_release
            with patch("dualcam.capture.V4L2Camera", side_effect=lambda name, cfg: cameras[name]), \
                 patch("dualcam.capture.cv2.VideoCapture", side_effect=captures), \
                 patch("dualcam.capture.MJPEGAVIWriter", side_effect=writers):
                with self.assertRaisesRegex(CameraError, "full disk"):
                    record_session(config, config_path, root / "session", .03, "motion")
            self.assertTrue(all(cap.released for cap in captures))
            session = json.loads((root / "session/session.json").read_text())
            self.assertEqual(session["status"], "failed")

    def test_streamon_exposure_reset_is_reapplied_while_camera_is_streaming(self):
        runner = FakeV4L2()
        camera = V4L2Camera("c920", camera_config("c920"), runner)
        capture = StreamingResetCapture(runner)
        result = configure_open_capture(capture, camera)
        self.assertEqual(len(capture.read_times), 1)
        self.assertEqual(runner.controls["exposure_time_absolute"]["value"], 40)
        self.assertEqual(result["controls"]["exposure_time_absolute"], 40)
        self.assertTrue(result["controls_applied_while_streaming"])
        writes = [call[3] for call in runner.calls if call[3] == "--set-ctrl=exposure_time_absolute=40"]
        self.assertEqual(len(writes), 2)

    def test_recording_clock_starts_after_both_verified_warmups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"cameras": {name: camera_config(name) for name in ("c920", "brio101")}}
            config_path = root / "rig.yaml"
            config_path.write_text(yaml.safe_dump(config))
            runners = {name: FakeV4L2() for name in config["cameras"]}
            cameras = {name: V4L2Camera(name, cfg, runners[name]) for name, cfg in config["cameras"].items()}
            captures = [StreamingResetCapture(runners[name], slow_warmup=True) for name in cameras]
            writers = [FakeWriter(), FakeWriter()]
            with patch("dualcam.capture.V4L2Camera", side_effect=lambda name, cfg: cameras[name]), \
                 patch("dualcam.capture.cv2.VideoCapture", side_effect=captures), \
                 patch("dualcam.capture.MJPEGAVIWriter", side_effect=writers):
                session = record_session(config, config_path, root / "session", .060, "motion")
            self.assertEqual(session["status"], "complete")
            self.assertGreater(session["warmup_elapsed_s"], .05)
            for name, capture in zip(cameras, captures):
                self.assertGreaterEqual(session["host_monotonic_zero_s"], capture.read_times[5])
                self.assertGreater(session["stats"][name]["frames"], 5)
                self.assertEqual(session["cameras"][name]["warmup_controls"]["exposure_time_absolute"], 40)
                self.assertEqual(session["cameras"][name]["final_controls"]["exposure_time_absolute"], 40)
                self.assertEqual(session["cameras"][name]["warmup_frames"], 5)

    def test_delayed_warmup_reset_fails_before_recording_clock_or_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"cameras": {name: camera_config(name) for name in ("c920", "brio101")}}
            config_path = root / "rig.yaml"
            config_path.write_text(yaml.safe_dump(config))
            runners = {name: FakeV4L2() for name in config["cameras"]}
            cameras = {name: V4L2Camera(name, cfg, runners[name]) for name, cfg in config["cameras"].items()}
            captures = [StreamingResetCapture(runners["c920"], reset_on=(1, 4)),
                        StreamingResetCapture(runners["brio101"], reset_on=())]
            writers = [FakeWriter(), FakeWriter()]
            with patch("dualcam.capture.V4L2Camera", side_effect=lambda name, cfg: cameras[name]), \
                 patch("dualcam.capture.cv2.VideoCapture", side_effect=captures), \
                 patch("dualcam.capture.MJPEGAVIWriter", side_effect=writers):
                with self.assertRaisesRegex(CameraError, "readback mismatch"):
                    record_session(config, config_path, root / "session", .06, "motion")
            session = json.loads((root / "session/session.json").read_text())
            self.assertEqual(session["status"], "failed")
            self.assertNotIn("host_monotonic_zero_s", session)
            self.assertTrue(all(writer.frames == 0 and writer.released for writer in writers))
            self.assertTrue(all(capture.released for capture in captures))

    def test_warmup_timeout_stops_worker_and_preserves_failed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"cameras": {name: camera_config(name) for name in ("c920", "brio101")}}
            config_path = root / "rig.yaml"
            config_path.write_text(yaml.safe_dump(config))
            cameras = {name: V4L2Camera(name, cfg, FakeV4L2()) for name, cfg in config["cameras"].items()}
            captures, writers = [FakeCapture(), FakeCapture()], [FakeWriter(), FakeWriter()]
            original_read = captures[0].read
            read_count = [0]

            def briefly_block_warmup():
                read_count[0] += 1
                if read_count[0] == 2:
                    time.sleep(.10)
                return original_read()

            captures[0].read = briefly_block_warmup
            with patch("dualcam.capture.V4L2Camera", side_effect=lambda name, cfg: cameras[name]), \
                 patch("dualcam.capture.cv2.VideoCapture", side_effect=captures), \
                 patch("dualcam.capture.MJPEGAVIWriter", side_effect=writers), \
                 patch("dualcam.capture.WARMUP_TIMEOUT_S", .02):
                with self.assertRaisesRegex(CameraError, "warmup timed out"):
                    record_session(config, config_path, root / "session", .06, "motion")
            session = json.loads((root / "session/session.json").read_text())
            self.assertEqual(session["status"], "failed")
            self.assertTrue(all(writer.frames == 0 and writer.released for writer in writers))
            self.assertTrue(all(capture.released for capture in captures))

    def test_c920_current_aliases_lock_focus_exposure_and_balance(self):
        device = FakeV4L2()
        result = V4L2Camera("c920", camera_config("c920"), device).apply()
        self.assertEqual(result["controls"]["focus_absolute"], 55)
        self.assertEqual(result["controls"]["focus_automatic_continuous"], 0)
        self.assertEqual(result["controls"]["auto_exposure"], 1)
        self.assertEqual(result["controls"]["exposure_time_absolute"], 40)
        self.assertEqual(result["controls"]["exposure_dynamic_framerate"], 0)
        self.assertEqual(result["controls"]["white_balance_automatic"], 0)

    def test_old_v4l2_aliases(self):
        text = CONTROLS.replace("auto_exposure", "exposure_auto").replace("exposure_time_absolute", "exposure_absolute")
        text = text.replace("exposure_dynamic_framerate", "exposure_auto_priority")
        text = text.replace("white_balance_automatic", "white_balance_temperature_auto")
        text = text.replace("focus_automatic_continuous", "focus_auto")
        result = V4L2Camera("c920", camera_config("c920"), FakeV4L2(text)).apply()
        self.assertEqual(result["controls"]["focus_auto"], 0)
        self.assertEqual(result["controls"]["exposure_absolute"], 40)

    def test_brio_never_writes_focus_even_if_driver_advertises_it(self):
        device = FakeV4L2()
        V4L2Camera("brio101", camera_config("brio101"), device).apply()
        writes = " ".join(call[3] for call in device.calls if call[3].startswith("--set-ctrl"))
        self.assertNotIn("focus", writes)

    def test_brio_focus_request_is_rejected(self):
        cfg = camera_config("brio101")
        cfg["capture"]["focus"] = 0
        with self.assertRaisesRegex(CameraError, "fixed-focus"):
            V4L2Camera("brio101", cfg, FakeV4L2()).prepare()

    def test_missing_or_invalid_focus_fails_before_any_write(self):
        for value in (None, 52):
            device = FakeV4L2()
            cfg = camera_config("c920")
            cfg["capture"]["focus"] = value
            with self.assertRaises(CameraError):
                V4L2Camera("c920", cfg, device).apply()
            self.assertFalse(any("--set-" in " ".join(call) for call in device.calls))

    def test_silent_control_failure_is_detected(self):
        with self.assertRaisesRegex(CameraError, "readback mismatch"):
            V4L2Camera("c920", camera_config("c920"), FakeV4L2(ignore="gain")).apply()

    def test_unsupported_capture_mode_is_not_silently_substituted(self):
        cfg = camera_config("c920")
        cfg["capture"]["fourcc"] = "YUYV"
        with self.assertRaisesRegex(CameraError, "not advertised"):
            V4L2Camera("c920", cfg, FakeV4L2()).apply()

    def test_unsupported_required_manual_control_is_not_ignored(self):
        device = FakeV4L2()
        del device.controls["white_balance_temperature"]
        with self.assertRaisesRegex(CameraError, "required control"):
            V4L2Camera("brio101", camera_config("brio101"), device).apply()

    def test_unknown_capture_setting_is_not_ignored(self):
        cfg = camera_config("c920")
        cfg["capture"]["autofocus"] = False
        with self.assertRaisesRegex(CameraError, "Unknown capture settings"):
            V4L2Camera("c920", cfg, FakeV4L2()).prepare()

    def test_opencv_format_readback_disagreement_stops_recording(self):
        cam = V4L2Camera("c920", camera_config("c920"), FakeV4L2())
        with self.assertRaisesRegex(CameraError, "width mismatch"):
            configure_open_capture(FakeCapture(mismatch=True), cam)

    def test_explicit_intrinsics_profile_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "calibration").mkdir()
            cfg = camera_config("c920")
            path = root / "calibration/c920.yaml"
            profile = dict(cfg["capture"], focus=60)
            path.write_text(yaml.safe_dump({"image_size": [1920, 1080], "K": None,
                                            "capture_profile": profile}))
            with self.assertRaisesRegex(CameraError, "capture_profile.focus"):
                check_intrinsics_profile(root / "config/rig.yaml", cfg)

    def test_intrinsics_profile_requires_exact_keys_and_accepts_unknown_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = camera_config("c920")
            cfg["intrinsics"] = "c920.yaml"
            full = dict(cfg["capture"])
            partial = dict(full)
            del partial["focus"]
            extra = dict(full, sharpness=128)
            for profile in ({}, partial, extra, []):
                with self.subTest(profile=profile):
                    (root / "c920.yaml").write_text(yaml.safe_dump({"capture_profile": profile}))
                    with self.assertRaisesRegex(CameraError, "capture_profile"):
                        check_intrinsics_profile(root / "rig.yaml", cfg)
            for profile, verified in ((None, False), (full, True)):
                (root / "c920.yaml").write_text(yaml.safe_dump({"capture_profile": profile}))
                result = check_intrinsics_profile(root / "rig.yaml", cfg)
                self.assertEqual(result["intrinsics_capture_profile_verified"], verified)

    def test_added_rig_control_is_not_implicitly_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = camera_config("c920")
            cfg["intrinsics"] = "c920.yaml"
            (root / "c920.yaml").write_text(yaml.safe_dump({"capture_profile": cfg["capture"]}))
            cfg["capture"]["sharpness"] = 128
            with self.assertRaisesRegex(CameraError, "keys differ"):
                check_intrinsics_profile(root / "rig.yaml", cfg)

    def test_failed_second_open_releases_both_before_any_control_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"cameras": {name: camera_config(name) for name in ("c920", "brio101")}}
            config_path = root / "rig.yaml"
            config_path.write_text(yaml.safe_dump(config))
            runners = {name: FakeV4L2() for name in config["cameras"]}
            cameras = {name: V4L2Camera(name, cfg, runners[name]) for name, cfg in config["cameras"].items()}
            captures = [FakeCapture(), FakeCapture()]
            captures[1].isOpened = lambda: False
            with patch("dualcam.capture.V4L2Camera", side_effect=lambda name, cfg: cameras[name]), \
                 patch("dualcam.capture.cv2.VideoCapture", side_effect=captures), \
                 patch("dualcam.capture.MJPEGAVIWriter") as writer:
                with self.assertRaisesRegex(CameraError, "Could not open"):
                    record_session(config, config_path, root / "session", .12, "checkerboard")
                writer.assert_not_called()
            self.assertTrue(all(cap.released for cap in captures))
            self.assertFalse(any("--set-" in " ".join(call) for runner in runners.values() for call in runner.calls))
            saved = json.loads((root / "session/session.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertTrue(saved["errors"])

    def test_worker_start_failure_joins_running_worker_before_final_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"cameras": {name: camera_config(name) for name in ("c920", "brio101")}}
            config_path = root / "rig.yaml"
            config_path.write_text(yaml.safe_dump(config))
            cameras = {name: V4L2Camera(name, cfg, FakeV4L2()) for name, cfg in config["cameras"].items()}
            captures, writers, threads = [FakeCapture(), FakeCapture()], [FakeWriter(), FakeWriter()], []
            real_thread = threading.Thread

            def new_thread(**kwargs):
                thread = real_thread(**kwargs)
                if threads:
                    def fail_start():
                        raise RuntimeError("simulated worker startup failure")
                    thread.start = fail_start
                threads.append(thread)
                return thread

            with patch("dualcam.capture.V4L2Camera", side_effect=lambda name, cfg: cameras[name]), \
                 patch("dualcam.capture.cv2.VideoCapture", side_effect=captures), \
                 patch("dualcam.capture.MJPEGAVIWriter", side_effect=writers), \
                 patch("dualcam.capture.threading.Thread", side_effect=new_thread):
                with self.assertRaisesRegex(CameraError, "simulated worker startup failure"):
                    record_session(config, config_path, root / "session", .12, "checkerboard")
            self.assertFalse(threads[0].is_alive())
            self.assertTrue(all(cap.released for cap in captures))
            self.assertTrue(all(writer.released for writer in writers))
            saved = json.loads((root / "session/session.json").read_text())
            self.assertEqual(saved["status"], "failed")

    def test_invalid_frame_returned_after_stop_is_not_saved_or_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"cameras": {name: camera_config(name) for name in ("c920", "brio101")}}
            config_path = root / "rig.yaml"
            config_path.write_text(yaml.safe_dump(config))
            cameras = {name: V4L2Camera(name, cfg, FakeV4L2()) for name, cfg in config["cameras"].items()}
            captures, writers = [FakeCapture(), FakeCapture()], [FakeWriter(), FakeWriter()]
            stop = threading.Event()
            original_read = captures[0].read
            returned_invalid = []

            def stop_during_final_read():
                result = original_read()
                if all(writer.frames >= 3 for writer in writers):
                    stop.set()
                    returned_invalid.append(True)
                    return True, np.array([[1, 2, 3]], dtype=np.uint8)
                return result

            captures[0].read = stop_during_final_read
            with patch("dualcam.capture.V4L2Camera", side_effect=lambda name, cfg: cameras[name]), \
                 patch("dualcam.capture.cv2.VideoCapture", side_effect=captures), \
                 patch("dualcam.capture.MJPEGAVIWriter", side_effect=writers):
                session = record_session(config, config_path, root / "session", 2, "motion", stop)
            self.assertTrue(returned_invalid)
            self.assertEqual(session["status"], "interrupted")
            self.assertFalse(session["errors"])
            self.assertTrue(all(writer.frames >= 3 and writer.released for writer in writers))

    def test_requested_stop_finalizes_interrupted_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"cameras": {name: camera_config(name) for name in ("c920", "brio101")}}
            config_path = root / "rig.yaml"
            config_path.write_text(yaml.safe_dump(config))
            cameras = {name: V4L2Camera(name, cfg, FakeV4L2()) for name, cfg in config["cameras"].items()}
            captures, writers = [FakeCapture(), FakeCapture()], [FakeWriter(), FakeWriter()]
            stop = threading.Event()
            original_write = writers[0].write

            def stop_when_both_recorded(frame):
                original_write(frame)
                if all(writer.frames >= 3 for writer in writers):
                    stop.set()

            writers[0].write = stop_when_both_recorded
            with patch("dualcam.capture.V4L2Camera", side_effect=lambda name, cfg: cameras[name]), \
                 patch("dualcam.capture.cv2.VideoCapture", side_effect=captures), \
                 patch("dualcam.capture.MJPEGAVIWriter", side_effect=writers):
                session = record_session(config, config_path, root / "session", 2, "motion", stop)
            self.assertEqual(session["status"], "interrupted")
            self.assertFalse(session["errors"])
            self.assertTrue(all(writer.released for writer in writers))
            saved = json.loads((root / "session/session.json").read_text())
            self.assertEqual(saved["status"], "interrupted")

    def test_fake_recording_writes_both_timestamp_streams_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"cameras": {name: camera_config(name) for name in ("c920", "brio101")},
                      "timing": {"verified": False, "brio_offset_s": 0}}
            config_path = root / "rig.yaml"
            config_path.write_text(yaml.safe_dump(config))
            cameras = {name: V4L2Camera(name, cfg, FakeV4L2()) for name, cfg in config["cameras"].items()}
            captures, writers = [], []

            def new_capture(*args):
                result = FakeCapture()
                captures.append(result)
                return result

            def new_writer(*args):
                result = FakeWriter()
                writers.append(result)
                return result

            with patch("dualcam.capture.V4L2Camera", side_effect=lambda name, cfg: cameras[name]), \
                 patch("dualcam.capture.cv2.VideoCapture", side_effect=new_capture), \
                 patch("dualcam.capture.MJPEGAVIWriter", side_effect=new_writer):
                session = record_session(config, config_path, root / "session", .12, "checkerboard")
            self.assertEqual(session["status"], "complete")
            self.assertTrue(all(cap.released for cap in captures))
            self.assertTrue(all(writer.released and writer.frames > 0 for writer in writers))
            saved = json.loads((root / "session/session.json").read_text())
            self.assertEqual(saved["timestamp_source"], "host_monotonic_read_return")
            for name in cameras:
                with (root / f"session/{name}_timestamps.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), session["stats"][name]["frames"])
                timestamps = [float(row["timestamp_s"]) for row in rows]
                self.assertGreater(len(timestamps), 2)
                self.assertTrue(all(0 <= t <= .12 for t in timestamps))
                self.assertTrue(all(b > a for a, b in zip(timestamps, timestamps[1:])))


if __name__ == "__main__":
    unittest.main()
