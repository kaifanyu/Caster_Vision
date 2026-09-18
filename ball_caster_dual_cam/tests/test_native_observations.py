"""Native extraction preserves timing, identities and cache provenance."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from dualcam.native_observations import collect_native_tracks, load_native_tracks


class NativeObservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        profile = {"width": 128, "height": 128, "fps": 30, "fourcc": "MJPG"}
        segment = {"red_hsv": {"lo": [170, 110, 50], "hi": [12, 255, 255]},
                   "green_hsv": {"lo": [40, 70, 30], "hi": [100, 255, 255]},
                   "grow_px": 5, "separation_px": 3, "morphology_px": 1,
                   "boundary_margin_px": 0}
        self.cfg = {"cameras": {}, "timing": {"brio_offset_s": .004},
                    "tracking": {"max_corners": 80, "min_distance_px": 3, "quality": .005}}
        meta = {"status": "complete", "mode": "motion", "cameras": {}}
        for ci, name in enumerate(("c920", "brio101")):
            intrinsics = self.root / (name + ".yaml")
            intrinsics.write_text(yaml.safe_dump({"K": [[100., 0, 64.], [0, 100., 64.], [0, 0, 1.]],
                                                 "dist": [0., 0., 0., 0.], "image_size": [128, 128]}))
            self.cfg["cameras"][name] = {"capture": profile, "intrinsics": str(intrinsics),
                                          "circle": [64, 64, 62], "segment": segment}
            meta["cameras"][name] = {"requested": profile}
            times = np.arange(5 + ci) / 30. + ci * .016
            (self.root / (name + "_timestamps.csv")).write_text(
                "frame_index,timestamp_s\n" + "".join(f"{i},{t:.9f}\n" for i, t in enumerate(times)))
            writer = cv2.VideoWriter(str(self.root / (name + ".avi")), cv2.VideoWriter_fourcc(*"MJPG"),
                                     30., (128, 128))
            self.assertTrue(writer.isOpened())
            for i in range(len(times)):
                image = np.full((128, 128, 3), 230, np.uint8)
                for y, color in ((34, (0, 0, 255)), (78, (0, 220, 0))):
                    for x in (33, 53, 73):
                        cv2.rectangle(image, (x + i, y), (x + i + 11, y + 12), color, -1)
                writer.write(image)
            writer.release()
        (self.root / "session.json").write_text(json.dumps(meta))

    def test_all_native_frames_and_both_endpoints_keep_original_clock(self):
        tracked = collect_native_tracks(self.cfg, self.root, progress=None)
        self.assertEqual([len(t) for t in tracked["times"]], [5, 6])
        self.assertAlmostEqual(tracked["times"][0][0], 0.)
        self.assertAlmostEqual(tracked["times"][1][0], .020)
        self.assertEqual([len(f) for f in tracked["initial_frames"]], [3, 3])
        obs = tracked["observations"]
        for ci, n in enumerate((5, 6)):
            for shell in (0, 1):
                selected = (obs["camera"] == ci) & (obs["shell"] == shell)
                np.testing.assert_array_equal(np.unique(obs["frame"][selected]), np.arange(n))
        keys = np.column_stack([obs[k] for k in ("camera", "shell", "frame", "track")])
        self.assertEqual(len(np.unique(keys, axis=0)), len(keys))
        self.assertTrue(np.all((obs["weight"] > 0) & (obs["weight"] <= 1)))

    def test_cache_round_trip_and_timing_change_invalidation(self):
        output = self.root / "cache"
        first = collect_native_tracks(self.cfg, self.root, output, progress=None)
        with patch("dualcam.native_observations.SelectedVideo", side_effect=AssertionError("cache missed")):
            cached = collect_native_tracks(self.cfg, self.root, output, progress=None)
        np.testing.assert_array_equal(cached["observations"]["uv"], first["observations"]["uv"])
        np.testing.assert_array_equal(cached["initial_frames"][1], first["initial_frames"][1])
        self.assertIsInstance(cached["videos"][0], Path)
        self.cfg["timing"]["brio_offset_s"] = .010
        changed = collect_native_tracks(self.cfg, self.root, output, progress=None)
        self.assertAlmostEqual(changed["times"][1][0], .026)
        with self.assertRaisesRegex(ValueError, "provenance"):
            load_native_tracks(output, expected_provenance=first["provenance"])

    def test_max_frames_applies_independently_to_both_cameras(self):
        tracked = collect_native_tracks(self.cfg, self.root, max_frames=3, progress=None)
        self.assertEqual([len(t) for t in tracked["times"]], [3, 3])
        self.assertEqual(int(tracked["observations"]["frame"].max()), 2)


if __name__ == "__main__":
    unittest.main()
