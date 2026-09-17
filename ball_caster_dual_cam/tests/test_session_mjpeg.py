"""Decode camera metadata quirks without changing frames or hiding failures."""
import cv2
import numpy as np
import pytest

from dualcam.mjpeg_avi import MJPEGAVIWriter
from dualcam.session import SelectedVideo


def test_short_app0_metadata_decodes_without_warnings_or_frame_loss(tmp_path, capfd):
    if not cv2.videoio_registry.hasBackend(cv2.CAP_OPENCV_MJPEG):
        pytest.skip("OpenCV native MJPEG reader is unavailable")
    path = tmp_path / "brio101.avi"
    expected = []
    writer = MJPEGAVIWriter(path, 30, (96, 64))
    try:
        for index in range(4):
            frame = np.full((64, 96, 3), (20 + 30 * index, 80, 170), np.uint8)
            cv2.rectangle(frame, (10 + index, 12), (45 + index, 40), (200, 30, 90), -1)
            ok, jpeg = cv2.imencode(".jpg", frame)
            assert ok
            original = jpeg.tobytes()
            # Same short APP0 stub found in the Brio's recorded JPEGs: the
            # length includes its two length bytes and two zero payload bytes.
            with_stub = original[:2] + b"\xff\xe0\x00\x04\x00\x00" + original[2:]
            writer.write(with_stub)
            expected.append(cv2.imdecode(jpeg, cv2.IMREAD_COLOR))
    finally:
        writer.release()
    video = SelectedVideo(path, (96, 64))
    try:
        for index, frame in enumerate(expected):
            np.testing.assert_array_equal(video.read(index), frame)
        with pytest.raises(ValueError, match="cannot read recorded frame 4"):
            video.read(4)
    finally:
        video.close()
    captured = capfd.readouterr()
    assert "unable to decode APP fields" not in captured.err


def test_non_mjpeg_avi_falls_back_and_keeps_selected_frame_indices(tmp_path):
    path = tmp_path / "other_codec.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 30, (64, 48))
    if not writer.isOpened():
        writer.release()
        pytest.skip("FFV1 encoder is unavailable")
    try:
        for value in (20, 80, 140):
            writer.write(np.full((48, 64, 3), value, np.uint8))
    finally:
        writer.release()
    video = SelectedVideo(path, (64, 48))
    try:
        assert np.all(video.read(0) == 20)
        assert np.all(video.read(2) == 140)
        with pytest.raises(ValueError, match="cannot read recorded frame 3"):
            video.read(3)
    finally:
        video.close()
