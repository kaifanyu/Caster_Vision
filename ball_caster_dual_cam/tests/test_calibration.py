"""Numerical regression tests for calibration geometry and acceptance gates."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from dualcam.calibration import CalibrationError, board_points, fit_intrinsics, fit_stereo, load_intrinsics
from scripts.calibrate_intrinsics import main as intrinsics_main
from scripts.calibrate_stereo import main as stereo_main


def synthetic_views(noise=0.05, reverse=True, identical_pose=False):
    rng = np.random.default_rng(16)
    obj = board_points(9, 6, 0.025)
    i1 = {"K": np.array([[1050., 0, 960], [0, 1040., 540], [0, 0, 1]]), "dist": np.zeros(5), "image_size": (1920, 1080)}
    i2 = {"K": np.array([[1180., 0, 955], [0, 1170., 544], [0, 0, 1]]), "dist": np.zeros(5), "image_size": (1920, 1080)}
    # The second camera is almost upside down, as in the supplied photographs.
    R21 = Rotation.from_euler("xyz", [4, 12, 178], degrees=True).as_matrix()
    t21 = np.array([.12, .015, .01])
    left, right = [], []
    for idx in range(12):
        euler = [0, 0, 0] if identical_pose else [-22 + 4 * idx, -18 + 6 * (idx % 7), -10 + 2 * idx]
        R1 = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
        t1 = np.array([-.1 + .015 * (idx % 4), -.075 + .012 * (idx % 3), .55 + .025 * (idx % 5)])
        if identical_pose:
            t1 = np.array([-.1, -.075, .65])
        p1, _ = cv2.projectPoints(obj, cv2.Rodrigues(R1)[0], t1, i1["K"], i1["dist"])
        p2, _ = cv2.projectPoints(obj, cv2.Rodrigues(R21 @ R1)[0], R21 @ t1 + t21, i2["K"], i2["dist"])
        p1 = p1.reshape(-1, 2) + rng.normal(0, noise, (len(obj), 2))
        p2 = p2.reshape(-1, 2) + rng.normal(0, noise, (len(obj), 2))
        left.append(p1.astype(np.float32))
        right.append((p2[::-1] if reverse and idx % 2 else p2).astype(np.float32))
    return left, right, i1, i2, R21, t21


class CalibrationTests(unittest.TestCase):
    def test_stereo_recovers_upside_down_camera_and_reversed_corner_order(self):
        left, right, i1, i2, R, t = synthetic_views()
        result, report = fit_stereo(left, right, i1, i2, 9, 6, .025)
        error = Rotation.from_matrix(np.asarray(result["R_21"]) @ R.T).magnitude()
        self.assertLess(np.degrees(error), .03)
        np.testing.assert_allclose(result["t_21_m"], t, atol=3e-4)
        self.assertEqual(report["right_corner_order_reversed"], [bool(i % 2) for i in range(12)])
        self.assertLess(result["rms_px"], .1)

    def test_stereo_translation_uses_measured_square_size(self):
        left, right, i1, i2, _, t = synthetic_views(noise=0)
        result, _ = fit_stereo(left, right, i1, i2, 9, 6, .05)
        np.testing.assert_allclose(result["t_21_m"], 2 * t, atol=1e-5)

    def test_intrinsics_recovers_camera(self):
        left, _, intrinsics, _, _, _ = synthetic_views(noise=.05)
        result, report = fit_intrinsics(left, (1920, 1080), 9, 6, .025)
        np.testing.assert_allclose(np.diag(result["K"])[:2], np.diag(intrinsics["K"])[:2], rtol=.01)
        self.assertLess(report["rms_px"], .1)

    def test_ambiguous_unvaried_board_is_rejected(self):
        left, right, i1, i2, _, _ = synthetic_views(noise=0, identical_pose=True)
        with self.assertRaisesRegex(CalibrationError, "tilt variation"):
            fit_stereo(left, right, i1, i2, 9, 6, .025)

    def test_bad_reprojection_is_rejected(self):
        left, _, _, _, _, _ = synthetic_views(noise=3)
        with self.assertRaisesRegex(CalibrationError, "RMS"):
            fit_intrinsics(left, (1920, 1080), 9, 6, .025, max_rms_px=.5)

    def test_zero_stereo_baseline_is_rejected(self):
        left, _, i1, _, _, _ = synthetic_views(noise=0)
        with self.assertRaisesRegex(CalibrationError, "baseline"):
            fit_stereo(left, left, i1, i1, 9, 6, .025)

    def test_invalid_board_scale_is_rejected(self):
        for scale in (0, -1, float("nan")):
            with self.assertRaises(CalibrationError):
                board_points(9, 6, scale)

    def test_intrinsics_cli_mixed_size_preserves_measured_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "one.jpg").touch()
            (root / "two.jpg").touch()
            output = root / "intrinsics.yaml"
            output.write_text("previous valid calibration\n")
            with patch("scripts.calibrate_intrinsics.detect_board", side_effect=[((1920, 1080), None), ((640, 480), None)]):
                code = intrinsics_main(["--images", str(root), "--cols", "9", "--rows", "6", "--square-m", ".025", "--output", str(output)])
            self.assertEqual(code, 2)
            self.assertEqual(output.read_text(), "previous valid calibration\n")
            report = json.loads(output.with_suffix(".report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("Mixed image dimensions", report["error"])

    def test_stereo_cli_requires_same_image_dimensions_as_intrinsics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for camera in ("c920", "brio101"):
                (root / camera).mkdir()
                (root / camera / "0001.png").touch()
                (root / (camera + ".yaml")).write_text(yaml.safe_dump({"K": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], "dist": [0] * 5, "image_size": [1920, 1080]}))
            config = {"cameras": {camera: {"intrinsics": camera + ".yaml"} for camera in ("c920", "brio101")}, "stereo": {"path": "accepted_stereo.yaml"}}
            (root / "rig.yaml").write_text(yaml.safe_dump(config))
            with patch("scripts.calibrate_stereo.detect_board", return_value=((640, 480), np.zeros((54, 2), np.float32))):
                code = stereo_main(["--config", str(root / "rig.yaml"), "--left-images", str(root / "c920"), "--right-images", str(root / "brio101"), "--cols", "9", "--rows", "6", "--square-m", ".025"])
            self.assertEqual(code, 2)
            self.assertFalse((root / "accepted_stereo.yaml").exists())
            self.assertIn("do not match intrinsics", json.loads((root / "accepted_stereo.report.json").read_text())["error"])

    def test_successful_stereo_cli_writes_config_relative_output_and_hashes(self):
        left, right, i1, i2, _, _ = synthetic_views()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            camera_intrinsics = {"c920": i1, "brio101": i2}
            capture_profile = {"width": 1920, "height": 1080, "fps": 30, "fourcc": "MJPG"}
            for camera, intrinsics in camera_intrinsics.items():
                (root / camera).mkdir()
                for idx in range(len(left)):
                    (root / camera / f"{idx:04d}.png").touch()
                data = {"K": intrinsics["K"].tolist(), "dist": intrinsics["dist"].tolist(), "image_size": list(intrinsics["image_size"]), "capture_profile": capture_profile}
                (root / (camera + ".yaml")).write_text(yaml.safe_dump(data))
                (root / camera / "capture_profile.yaml").write_text(yaml.safe_dump({"capture_profile": capture_profile}))
            config = {"cameras": {camera: {"intrinsics": camera + ".yaml"} for camera in camera_intrinsics}, "stereo": {"path": "results/stereo.yaml"}}
            (root / "rig.yaml").write_text(yaml.safe_dump(config))
            def detect(path, cols, rows):
                idx = int(path.stem)
                return (1920, 1080), left[idx] if path.parent.name == "c920" else right[idx]
            with patch("scripts.calibrate_stereo.detect_board", side_effect=detect):
                code = stereo_main(["--config", str(root / "rig.yaml"), "--left-images", str(root / "c920"), "--right-images", str(root / "brio101"), "--cols", "9", "--rows", "6", "--square-m", ".025"])
            self.assertEqual(code, 0)
            output = yaml.safe_load((root / "results/stereo.yaml").read_text())
            self.assertEqual(set(output["intrinsics_sha256"]), {"c920", "brio101"})
            self.assertTrue(all(len(value) == 64 for value in output["intrinsics_sha256"].values()))
            self.assertEqual(json.loads((root / "results/stereo.report.json").read_text())["status"], "passed")

    def test_placeholder_intrinsics_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera.yaml"
            path.write_text("K: null\ndist: null\nimage_size: [1920,1080]\n")
            with self.assertRaises(CalibrationError):
                load_intrinsics(path)


if __name__ == "__main__":
    unittest.main()
