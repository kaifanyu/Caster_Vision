"""Round-trip original JPEGs through the lightweight AVI container writer."""
from pathlib import Path
import struct
import tempfile
import unittest

import cv2
import numpy as np

from dualcam.mjpeg_avi import (AVISizeLimitError, MJPEGAVIWriter, SegmentedMJPEGAVIWriter,
                              jpeg_dimensions, jpeg_payload)


def encoded_frames():
    random = np.random.default_rng(19)
    result = []
    for index in range(6):
        frame = np.full((64, 96, 3), 30*index, np.uint8)
        frame[8:40, 7+index:40+index] = random.integers(0, 256, (32, 33, 1), dtype=np.uint8)
        cv2.circle(frame, (65, 35), 5+index, (20, 200, 70), -1)
        ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78+index])
        if not ok:
            raise RuntimeError("JPEG encoder unavailable for muxer regression.")
        result.append(data)
    return result


class MJPEGAviTests(unittest.TestCase):
    def test_segment_rotation_preserves_every_original_frame_with_bounded_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"c920.avi"
            frames = encoded_frames()*3
            limit = 256+3*(max(map(len, frames))+24)
            writer = SegmentedMJPEGAVIWriter(path, 30, (96, 64), max_file_size=limit)
            for frame in frames:
                writer.write(frame)
            writer.release()
            writer.release()
            self.assertFalse(writer.isOpened())
            self.assertEqual(writer.frames, len(frames))
            self.assertGreater(len(writer.segments), 2)
            consumed = 0
            for index, segment in enumerate(writer.segments):
                self.assertEqual(segment["path"], "c920.avi" if index == 0 else f"c920_{index:04d}.avi")
                self.assertEqual(segment["start_frame"], consumed)
                content = (path.parent/segment["path"]).read_bytes()
                self.assertLessEqual(len(content), limit)
                cursor = content.index(b"movi")+4
                for frame in frames[consumed:consumed+segment["frame_count"]]:
                    self.assertEqual(content[cursor:cursor+4], b"00dc")
                    size = struct.unpack_from("<I", content, cursor+4)[0]
                    self.assertEqual(content[cursor+8:cursor+8+size], frame.tobytes())
                    cursor += 8+size+(size & 1)
                self.assertEqual(content[cursor:cursor+4], b"idx1")
                consumed += segment["frame_count"]
            self.assertEqual(consumed, len(frames))

    def test_segment_limit_cannot_silently_accept_an_oversized_single_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = SegmentedMJPEGAVIWriter(Path(temporary)/"c920.avi", 30, (96, 64),
                                             max_file_size=300)
            with self.assertRaisesRegex(AVISizeLimitError, "single JPEG frame"):
                writer.write(encoded_frames()[0])
            writer.release()
            self.assertEqual(writer.frames, 0)
            self.assertEqual(len(writer.segments), 1)
            self.assertEqual(writer.segments[0]["frame_count"], 0)

    def test_variable_jpegs_round_trip_with_original_payloads_index_and_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"raw.avi"
            frames = encoded_frames()
            self.assertGreater(len(set(map(len, frames))), 1)
            writer = MJPEGAVIWriter(path, 29.97, (96, 64))
            for frame in frames:
                self.assertEqual(jpeg_dimensions(frame), (96, 64))
                writer.write(frame)
            writer.release()
            writer.release()
            self.assertFalse(writer.isOpened())
            content = path.read_bytes()
            self.assertEqual(content[:4], b"RIFF")
            self.assertEqual(struct.unpack_from("<I", content, 4)[0], len(content)-8)
            movi = content.index(b"movi")
            position = movi+4
            offsets = []
            for frame in frames:
                self.assertEqual(content[position:position+4], b"00dc")
                size = struct.unpack_from("<I", content, position+4)[0]
                self.assertEqual(content[position+8:position+8+size], frame.tobytes())
                offsets.append(position-movi)
                position += 8+size+(size & 1)
            self.assertEqual(content[position:position+4], b"idx1")
            self.assertEqual(struct.unpack_from("<I", content, position+4)[0], 16*len(frames))
            for index, (offset, frame) in enumerate(zip(offsets, frames)):
                self.assertEqual(struct.unpack_from("<4sIII", content, position+8+16*index),
                                 (b"00dc", 0x10, offset, len(frame)))
            cap = cv2.VideoCapture(str(path))
            try:
                self.assertTrue(cap.isOpened())
                self.assertEqual(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(frames))
                self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 29.97, places=3)
                for frame in frames:
                    ok, decoded = cap.read()
                    self.assertTrue(ok)
                    self.assertEqual(decoded.shape, (64, 96, 3))
                    expected = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    self.assertLess(np.abs(decoded.astype(float)-expected).mean(), 1.)
                self.assertFalse(cap.read()[0])
                self.assertTrue(cap.set(cv2.CAP_PROP_POS_FRAMES, 3))
                ok, decoded = cap.read()
                self.assertTrue(ok)
                self.assertLess(np.abs(decoded.astype(float)-cv2.imdecode(frames[3], 1)).mean(), 1.)
            finally:
                cap.release()

    def test_arbitrary_uvc_tail_padding_is_trimmed_without_altering_jpeg(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"padded.avi"
            frames = encoded_frames()[:3]
            writer = MJPEGAVIWriter(path, 30, (96, 64))
            for frame, count in zip(frames, (27, 28, 31)):
                # Include another EOI in the arbitrary trailer: the JPEG's first
                # terminal marker must be selected, not the trailer's last one.
                padding = (b"\xa7\xff\xd9\x83"*8)[:count]
                padded = frame.tobytes()+padding
                self.assertEqual(bytes(jpeg_payload(padded)), frame.tobytes())
                self.assertEqual(jpeg_dimensions(padded), (96, 64))
                writer.write(padded)
            writer.release()
            content = path.read_bytes()
            for frame in frames:
                self.assertIn(frame.tobytes(), content)
            cap = cv2.VideoCapture(str(path))
            try:
                self.assertEqual(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
                for frame in frames:
                    ok, decoded = cap.read()
                    self.assertTrue(ok)
                    expected = cv2.imdecode(frame, 1)
                    self.assertLess(np.abs(decoded.astype(float)-expected).mean(), 1.)
                self.assertFalse(cap.read()[0])
            finally:
                cap.release()
            valid = frames[0].tobytes()
            with self.assertRaisesRegex(ValueError, "no EOI"):
                jpeg_dimensions(valid[:-2]+b"x"*31)
            with self.assertRaisesRegex(ValueError, "no EOI"):
                jpeg_dimensions(valid+b"x"*32)

    def test_size_limit_rejects_new_frame_but_finalizes_previous_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"limit.avi"
            frame = encoded_frames()[0]
            writer = MJPEGAVIWriter(path, 30, (96, 64))
            writer.limit = writer.stream.tell()+8+len(frame)+(len(frame) & 1)+8+16
            writer.write(frame)
            with self.assertRaisesRegex(ValueError, "RIFF size limit"):
                writer.write(frame)
            writer.release()
            self.assertEqual(path.stat().st_size, writer.limit)
            cap = cv2.VideoCapture(str(path))
            try:
                self.assertEqual(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
                self.assertTrue(cap.read()[0])
                self.assertFalse(cap.read()[0])
            finally:
                cap.release()

    def test_invalid_jpeg_and_changed_dimensions_are_rejected(self):
        for data in (b"", b"notjpeg", b"\xff\xd8\xff\xd9", b"\xff\xd8\xff\xe0\xff\xff\xff\xd9"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                jpeg_dimensions(data)
        with tempfile.TemporaryDirectory() as temporary:
            writer = MJPEGAVIWriter(Path(temporary)/"shape.avi", 30, (640, 480))
            try:
                with self.assertRaisesRegex(ValueError, "dimensions"):
                    writer.write(encoded_frames()[0])
                self.assertEqual(writer.frames, 0)
            finally:
                writer.release()


if __name__ == "__main__":
    unittest.main()
