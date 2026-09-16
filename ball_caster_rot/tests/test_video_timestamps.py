"""Native video timing must survive decoding, including variable frame intervals."""

from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest

from ballrot.io_frames import FrameSource


def _video_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    times_msec: list[float],
    *,
    fps: float = 30.0,
    max_frames: int | None = None,
    image_fps: float | None = None,
) -> tuple[FrameSource, list]:
    path = tmp_path / "video.mkv"
    path.touch()
    captures = []

    class FakeCapture:
        def __init__(self, _path: str) -> None:
            self.index = -1
            self.released = False
            captures.append(self)

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_POS_MSEC:
                assert self.index >= 0, "timestamps must be queried after decoding"
                return times_msec[self.index]
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 4,
                cv2.CAP_PROP_FRAME_HEIGHT: 3,
                cv2.CAP_PROP_FPS: fps,
                cv2.CAP_PROP_FRAME_COUNT: len(times_msec),
            }[prop]

        def read(self) -> tuple[bool, np.ndarray | None]:
            self.index += 1
            if self.index >= len(times_msec):
                return False, None
            return True, np.full((3, 4, 3), self.index, dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    return FrameSource(
        path, "video", max_frames=max_frames, image_fps=image_fps
    ), captures


def test_native_vfr_timing_normalizes_origin_and_ignores_nominal_rate(tmp_path, monkeypatch):
    source, captures = _video_source(
        tmp_path, monkeypatch, [1200.0, 1233.0, 1299.0, 1555.0], image_fps=60.0
    )
    with warnings.catch_warnings(record=True) as caught:
        records = list(source.records())
    assert not caught
    assert source.timing_source == "native_video_pts"
    assert source.fps == 30.0
    assert [record.index for record in records] == [0, 1, 2, 3]
    np.testing.assert_allclose([record.timestamp_s for record in records], [0, .033, .099, .355])
    assert [int(record.image[0, 0, 0]) for record in records] == [0, 1, 2, 3]
    assert all(capture.released for capture in captures)


def test_zero_pts_fallback_is_explicit_and_reiteration_is_stable(tmp_path, monkeypatch):
    source, captures = _video_source(tmp_path, monkeypatch, [0.0] * 4, fps=25.0)
    for _ in range(2):
        with pytest.warns(RuntimeWarning, match="falling back to nominal FPS") as caught:
            records = list(source.records())
        assert len(caught) == 1
        assert source.timing_source == "fps_fallback"
        np.testing.assert_allclose([record.timestamp_s for record in records], [0, .04, .08, .12])
    assert all(capture.released for capture in captures)


@pytest.mark.parametrize("times", [[0., 33., 33.], [0., 33., 0.], [1200., 1200.], [0., 33., 20.]])
def test_nonincreasing_native_pts_fails_instead_of_retiming(tmp_path, monkeypatch, times):
    source, captures = _video_source(tmp_path, monkeypatch, times)
    with pytest.raises(ValueError, match="not strictly increasing"):
        list(source.records())
    assert all(capture.released for capture in captures)


@pytest.mark.parametrize("bad_time", [float("nan"), float("inf"), -1.0])
@pytest.mark.parametrize("prefix", [[], [0., 33.]])
def test_malformed_pts_fails_and_closes_capture(tmp_path, monkeypatch, bad_time, prefix):
    source, captures = _video_source(tmp_path, monkeypatch, prefix + [bad_time])
    with pytest.raises(ValueError, match="invalid video presentation timestamp"):
        list(source.records())
    assert all(capture.released for capture in captures)


def test_native_pts_cannot_resume_after_unavailable_pts(tmp_path, monkeypatch):
    source, _ = _video_source(tmp_path, monkeypatch, [0., 0., 66.])
    with pytest.warns(RuntimeWarning, match="falling back"):
        with pytest.raises(ValueError, match="mixed unavailable and native"):
            list(source.records())


def test_unavailable_pts_without_fps_requires_explicit_timing(tmp_path, monkeypatch):
    source, _ = _video_source(tmp_path, monkeypatch, [0., 0., 0.], fps=0.0)
    with pytest.raises(ValueError, match="timestamps and nominal FPS are unavailable"):
        list(source.records())


def test_native_timing_does_not_require_nominal_fps(tmp_path, monkeypatch):
    source, _ = _video_source(tmp_path, monkeypatch, [0., 33., 100.], fps=0.0)
    assert source.fps is None
    np.testing.assert_allclose([record.timestamp_s for record in source], [0, .033, .1])


def test_arrays_can_be_decoded_for_explicit_retiming_despite_broken_pts(tmp_path, monkeypatch):
    source, captures = _video_source(tmp_path, monkeypatch, [float("nan"), -1., 0.])
    assert [int(frame[0, 0, 0]) for frame in source.frames()] == [0, 1, 2]
    assert source.timing_source == "unavailable"
    assert all(capture.released for capture in captures)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_frame_limit_keeps_selected_native_timing(tmp_path, monkeypatch, count):
    source, captures = _video_source(tmp_path, monkeypatch, [500., 550., 900.], max_frames=count)
    assert [record.timestamp_s for record in source.records()] == [0., .05][:count]
    assert all(capture.released for capture in captures)


def test_image_sequences_keep_explicit_uniform_timing(tmp_path):
    for index in range(3):
        assert cv2.imwrite(str(tmp_path / f"frame{index}.png"), np.zeros((3, 4, 3), np.uint8))
    timed = FrameSource(tmp_path, "images", image_fps=20.)
    untimed = FrameSource(tmp_path, "images")
    assert [record.timestamp_s for record in timed] == [0., .05, .1]
    assert [record.timestamp_s for record in untimed] == [None, None, None]
    assert timed.timing_source == "image_fps"
    assert untimed.timing_source == "unavailable"
