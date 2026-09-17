"""Decode explicit AVI segments using the timestamp CSV's global frame indices."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from dualcam.session import SelectedVideo, load_session


class SegmentedVideoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def write_video(self, name, values, size=(64, 48)):
        path = self.directory / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, size)
        self.assertTrue(writer.isOpened())
        try:
            for value in values:
                writer.write(np.full((size[1], size[0], 3), value, np.uint8))
        finally:
            writer.release()
        return path

    def write_manifest(self, segments, camera="c920"):
        (self.directory / "session.json").write_text(json.dumps({
            "status": "complete", "cameras": {camera: {"video_segments": segments}}
        }))

    def open_video(self, name="c920.avi", image_size=(64, 48)):
        video = SelectedVideo(self.directory / name, image_size)
        self.addCleanup(video.close)
        return video

    def two_segments(self):
        self.write_video("c920.avi", [20, 40])
        self.write_video("c920_0001.avi", [60, 80, 100])
        return [{"path": "c920.avi", "start_frame": 0, "frame_count": 2},
                {"path": "c920_0001.avi", "start_frame": 2, "frame_count": 3}]

    def test_global_indices_cross_segments_without_seeks(self):
        self.write_manifest(self.two_segments())
        video = self.open_video()
        for index, value in ((0, 20), (2, 60), (4, 100)):
            self.assertAlmostEqual(float(video.read(index).mean()), value, delta=3)
            self.assertEqual(video.index, index)
        with self.assertRaisesRegex(ValueError, "increase"):
            video.read(4)
        with self.assertRaisesRegex(ValueError, "outside declared"):
            video.read(5)
        video.close()  # Cleanup is deliberately idempotent.

    def test_paired_timestamps_use_global_indices_with_different_segment_boundaries(self):
        metadata = {"status": "complete", "cameras": {}}
        cfg = {"cameras": {}, "timing": {"max_pair_skew_ms": 4.}}
        for camera, counts in (("c920", (2, 3, 1)), ("brio101", (1, 3, 2))):
            segments, start = [], 0
            for part, count in enumerate(counts):
                name = f"{camera}.avi" if part == 0 else f"{camera}_{part:04d}.avi"
                self.write_video(name, [30 + 30 * i for i in range(start, start + count)])
                segments.append({"path": name, "start_frame": start, "frame_count": count})
                start += count
            metadata["cameras"][camera] = {"requested": {}, "video_segments": segments}
            cfg["cameras"][camera] = {"capture": {}}
            offset = 0.003 if camera == "brio101" else 0.
            (self.directory / f"{camera}_timestamps.csv").write_text(
                "frame_index,timestamp_s\n" + "".join(f"{i},{i/30 + offset}\n" for i in range(6)))
        (self.directory / "session.json").write_text(json.dumps(metadata))
        session = load_session(self.directory, cfg)
        np.testing.assert_array_equal(session["pairs"], np.repeat(np.arange(6)[:, None], 2, axis=1))
        for ci, camera in enumerate(("c920", "brio101")):
            video = self.open_video(f"{camera}.avi")
            for pair in session["pairs"][[0, 2, 3, 5]]:
                index = pair[ci]
                self.assertAlmostEqual(float(video.read(index).mean()), 30 + 30*index, delta=3)

    def test_missing_and_prematurely_ended_segments_fail_instead_of_shifting_indices(self):
        segments = self.two_segments()
        missing = copy.deepcopy(segments)
        missing[1]["path"] = "missing.avi"
        self.write_manifest(missing)
        with self.assertRaisesRegex(ValueError, "Missing video segment"):
            self.open_video()
        segments[0]["frame_count"] = 3
        segments[1]["start_frame"] = 3
        self.write_manifest(segments)
        video = self.open_video()
        # Even selecting a later segment must verify the intervening frames.
        with self.assertRaisesRegex(ValueError, "recorded frame 2.*premature EOF"):
            video.read(3)

    def test_zero_count_segments_are_accepted_but_never_decoded(self):
        (self.directory / "c920.avi").touch()
        self.write_video("c920_0001.avi", [70])
        (self.directory / "empty.avi").touch()
        self.write_video("c920_0002.avi", [120])
        self.write_manifest([
            {"path": "c920.avi", "start_frame": 0, "frame_count": 0},
            {"path": "c920_0001.avi", "start_frame": 0, "frame_count": 1},
            {"path": "empty.avi", "start_frame": 1, "frame_count": 0},
            {"path": "c920_0002.avi", "start_frame": 1, "frame_count": 1},
        ])
        video = self.open_video()
        self.assertAlmostEqual(float(video.read(0).mean()), 70, delta=3)
        self.assertAlmostEqual(float(video.read(1).mean()), 120, delta=3)
        self.write_manifest([{"path": "c920.avi", "start_frame": 0, "frame_count": 0}])
        with self.assertRaisesRegex(ValueError, "outside declared"):
            self.open_video().read(0)

    def test_legacy_files_do_not_discover_undeclared_neighbor_segments(self):
        self.two_segments()
        for with_manifest in (False, True):
            if with_manifest:
                (self.directory / "session.json").write_text(json.dumps({"cameras": {"c920": {}}}))
            video = self.open_video()
            self.assertAlmostEqual(float(video.read(1).mean()), 40, delta=3)
            with self.assertRaisesRegex(ValueError, "recorded frame 2"):
                video.read(2)
        # Unrelated videos do not inherit a camera-specific manifest.
        self.write_video("sample.avi", [110])
        self.write_manifest(None)
        self.assertAlmostEqual(float(self.open_video("sample.avi").read(0).mean()), 110, delta=3)

    def test_resolution_is_checked_after_switching_segments(self):
        segments = self.two_segments()
        self.write_video("c920_0001.avi", [60, 80, 100], size=(96, 64))
        self.write_manifest(segments)
        video = self.open_video()
        video.read(0)
        with self.assertRaisesRegex(ValueError, "shape"):
            video.read(2)

    def test_segment_structure_paths_and_ranges_are_validated(self):
        good = self.two_segments()
        invalid = [None, [], {}, [None]]
        for field, value in (("path", ""), ("path", "../c920.avi"),
                             ("path", "/tmp/c920.avi"), ("path", "sub/c920.avi"),
                             ("path", "sub\\c920.avi"), ("path", "."), ("path", ".."),
                             ("path", "c920_0001.avi"), ("path", None),
                             ("start_frame", 1), ("start_frame", -1),
                             ("start_frame", True), ("start_frame", 0.0),
                             ("frame_count", -1), ("frame_count", False),
                             ("frame_count", 2.0), ("frame_count", None)):
            changed = copy.deepcopy(good)
            changed[0][field] = value
            invalid.append(changed)
        for start in (1, 3):  # Overlap and gap.
            changed = copy.deepcopy(good)
            changed[1]["start_frame"] = start
            invalid.append(changed)
        changed = copy.deepcopy(good)
        changed[1]["path"] = "c920.avi"
        invalid.append(changed)
        for entries in invalid:
            with self.subTest(entries=entries):
                self.write_manifest(entries)
                with self.assertRaises(ValueError):
                    self.open_video()
        (self.directory / "session.json").write_text("{broken json")
        with self.assertRaisesRegex(ValueError, "Invalid video segment metadata"):
            self.open_video()

    def test_symlink_escapes_and_alias_duplicates_are_rejected(self):
        segments = self.two_segments()
        alias = self.directory / "alias.avi"
        alias.symlink_to(self.directory / "c920.avi")
        segments[1]["path"] = alias.name
        self.write_manifest(segments)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.open_video()
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "external.avi"
            target.touch()
            alias.unlink()
            alias.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "escapes"):
                self.open_video()


if __name__ == "__main__":
    unittest.main()
