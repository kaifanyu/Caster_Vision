"""Preview CLI persistence without a display or changes to the real rig."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from scripts.preview import main, save_circle


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = self.root / "custom-rig.yaml"
        self.image = self.root / "frame.png"
        self.output = self.root / "preview.png"
        self.assertTrue(cv2.imwrite(str(self.image), np.zeros((120, 160, 3), np.uint8)))
        camera_blocks = []
        for name in ("c920", "brio101"):
            camera_blocks.append(f"""  {name}:
    intrinsics: {name}.yaml # keep this relative path
    capture: {{width: 160, height: 120, fps: 30}}
    circle: [80, 60, 40] # reviewed circle
    segment:
      red_hsv: {{lo: [170, 100, 30], hi: [12, 255, 255]}}
      green_hsv: {{lo: [40, 70, 30], hi: [100, 255, 255]}}
""")
            (self.root / f"{name}.yaml").write_text(yaml.safe_dump({
                "image_size": [160, 120], "K": [[100, 0, 80], [0, 100, 60], [0, 0, 1]],
                "dist": [0.] * 5}), encoding="utf-8")
        source = ("# Rig settings; preserve comments and CRLF\ncameras:\n"
                  + "".join(camera_blocks)
                  + "stereo: {path: stereo.yaml}\naxes: {path: axes.yaml}\n"
                    "geometry: {radius_m: 0.1, gap_m: 0.02}\n"
                    "timing: {brio_offset_s: 0, max_pair_skew_ms: 12}\n"
                    "tracking: {max_corners: 20}\n")
        self.original = source.replace("\n", "\r\n").encode("utf-8")
        self.config.write_bytes(self.original)

    def args(self, camera="c920"):
        return ["--config", str(self.config), "--camera", camera,
                "--image", str(self.image), "--output", str(self.output)]

    def test_picks_save_each_camera_and_preserve_all_other_text(self):
        with patch("scripts.preview.pick_circle", return_value=(75.5, 60.5, 45.25)):
            self.assertEqual(main(self.args() + ["--pick-circle"]), 0)
        expected = self.original.replace(b"[80, 60, 40]", b"[75.5, 60.5, 45.25]", 1)
        self.assertEqual(self.config.read_bytes(), expected)
        with patch("scripts.preview.pick_circle", return_value=(85., 65., 42.)):
            self.assertEqual(main(self.args("brio101") + ["--pick-circle"]), 0)
        expected = expected.replace(b"[80, 60, 40]", b"[85.0, 65.0, 42.0]", 1)
        self.assertEqual(self.config.read_bytes(), expected)
        diagnostic = yaml.safe_load(self.output.with_suffix(".yaml").read_text())
        self.assertEqual(diagnostic["circle"], [85., 65., 42.])
        self.assertEqual(diagnostic["camera"], "brio101")
        self.assertEqual(cv2.imread(str(self.output)).shape, (120, 160, 3))

    def test_explicit_circle_is_saved(self):
        self.assertEqual(main(self.args() + ["--circle", "75", "65", "42"]), 0)
        expected = self.original.replace(b"[80, 60, 40]", b"[75.0, 65.0, 42.0]", 1)
        self.assertEqual(self.config.read_bytes(), expected)

    def test_plain_preview_and_no_save_leave_config_untouched(self):
        for options in ([], ["--circle", "75", "65", "42", "--no-save"],
                        ["--pick-circle", "--no-save"]):
            with self.subTest(options=options):
                with patch("scripts.preview.pick_circle", return_value=(75., 65., 42.)):
                    self.assertEqual(main(self.args() + options), 0)
                self.assertEqual(self.config.read_bytes(), self.original)

    def test_cancel_does_not_save(self):
        with patch("scripts.preview.pick_circle", side_effect=ValueError("Circle selection canceled")):
            with self.assertRaisesRegex(ValueError, "canceled"):
                main(self.args() + ["--pick-circle"])
        self.assertEqual(self.config.read_bytes(), self.original)
        self.assertFalse(self.output.exists())

    def test_failed_preview_write_does_not_save(self):
        with patch("scripts.preview.cv2.imwrite", return_value=False):
            with self.assertRaisesRegex(OSError, "Could not save"):
                main(self.args() + ["--circle", "75", "65", "42"])
        self.assertEqual(self.config.read_bytes(), self.original)

    def test_alternative_yaml_circle_layouts_keep_other_values(self):
        variants = (
            self.original.replace(b"    circle: [80, 60, 40] # reviewed circle\r\n", b"", 1),
            self.original.replace(b"[80, 60, 40]", b"null", 1),
            self.original.replace(b"[80, 60, 40] # reviewed circle",
                                  b"\r\n      - 80\r\n      - 60\r\n      - 40", 1),
        )
        for source in variants:
            with self.subTest(source=source):
                self.config.write_bytes(source)
                expected = yaml.safe_load(source)
                expected["cameras"]["c920"]["circle"] = [75., 65., 42.]
                save_circle(self.config, "c920", (75., 65., 42.))
                self.assertEqual(yaml.safe_load(self.config.read_bytes()), expected)
                self.assertIn(b"# keep this relative path", self.config.read_bytes())

    def test_output_yaml_cannot_overwrite_config(self):
        args = self.args()
        args[-1] = str(self.config.with_suffix(".png"))
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            main(args + ["--circle", "75", "65", "42"])
        self.assertEqual(self.config.read_bytes(), self.original)


if __name__ == "__main__":
    unittest.main()
