"""Uniform, repeatable loading of videos and image sequences."""

from __future__ import annotations

import glob
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


InputType = Literal["auto", "video", "images"]

IMAGE_EXTENSIONS = {
    ".bmp",
    ".dib",
    ".jpeg",
    ".jpg",
    ".jp2",
    ".png",
    ".pbm",
    ".pgm",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}

VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}


@dataclass(frozen=True)
class FrameRecord:
    """One decoded frame and stable source metadata."""

    index: int
    image: np.ndarray
    timestamp_s: float | None
    source: str

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("frame index cannot be negative")
        image = np.asarray(self.image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("decoded frame must be a uint8 BGR image")
        if self.timestamp_s is not None and (
            not np.isfinite(self.timestamp_s) or self.timestamp_s < 0
        ):
            raise ValueError("timestamp_s must be non-negative and finite")
        object.__setattr__(self, "image", image)


def _natural_key(path: str | Path) -> list[tuple[int, int | str]]:
    """Natural sort key: frame2.png precedes frame10.png."""

    parts = re.split(r"(\d+)", str(path).casefold())
    return [(0, int(part)) if part.isdigit() else (1, part) for part in parts]


def _contains_glob(path: str) -> bool:
    return any(character in path for character in "*?[")


def _resolve_image_paths(path: str | Path) -> list[Path]:
    text_path = str(path)
    candidate = Path(path).expanduser()
    if _contains_glob(text_path):
        matches = [Path(item) for item in glob.glob(text_path)]
    elif candidate.is_dir():
        matches = [item for item in candidate.iterdir() if item.is_file()]
    elif candidate.is_file():
        matches = [candidate]
    else:
        raise FileNotFoundError(f"input path does not exist: {candidate}")

    images = [item.resolve() for item in matches if item.suffix.casefold() in IMAGE_EXTENSIONS]
    images.sort(key=_natural_key)
    if not images:
        raise FileNotFoundError(f"no supported images found at: {path}")
    return images


def detect_input_type(path: str | Path) -> Literal["video", "images"]:
    """Classify a file, directory, or glob without decoding all frames."""

    text_path = str(path)
    candidate = Path(path).expanduser()
    if _contains_glob(text_path) or candidate.is_dir():
        _resolve_image_paths(path)  # Raises a useful error for an empty sequence.
        return "images"
    if not candidate.is_file():
        raise FileNotFoundError(f"input path does not exist: {candidate}")
    extension = candidate.suffix.casefold()
    if extension in IMAGE_EXTENSIONS:
        return "images"
    if extension in VIDEO_EXTENSIONS:
        return "video"

    # For uncommon extensions, test a single image decode first, then defer to
    # the video backend.  This keeps auto-detection useful for camera formats
    # without maintaining an exhaustive extension table.
    if cv2.imread(str(candidate), cv2.IMREAD_COLOR) is not None:
        return "images"
    capture = cv2.VideoCapture(str(candidate))
    try:
        if capture.isOpened():
            return "video"
    finally:
        capture.release()
    raise ValueError(f"cannot identify input as an image or video: {candidate}")


class FrameSource:
    """A re-iterable video or naturally sorted image-sequence source."""

    def __init__(
        self,
        path: str | Path,
        input_type: InputType = "auto",
        max_frames: int | None = None,
        *,
        image_fps: float | None = None,
    ) -> None:
        self.path = Path(path).expanduser() if not _contains_glob(str(path)) else Path(str(path))
        normalized_type = str(input_type).strip().lower()
        if normalized_type == "image":
            normalized_type = "images"
        if normalized_type not in {"auto", "video", "images"}:
            raise ValueError("input_type must be auto, video, or images")
        self.kind: Literal["video", "images"] = (
            detect_input_type(path)
            if normalized_type == "auto"
            else normalized_type  # type: ignore[assignment]
        )

        if max_frames is not None:
            if isinstance(max_frames, bool) or int(max_frames) != max_frames:
                raise ValueError("max_frames must be an integer or None")
            if int(max_frames) < 0:
                raise ValueError("max_frames cannot be negative")
            max_frames = int(max_frames)
        self.max_frames = max_frames

        if image_fps is not None and (
            not np.isfinite(image_fps) or float(image_fps) <= 0
        ):
            raise ValueError("image_fps must be positive and finite")
        self._image_fps = None if image_fps is None else float(image_fps)
        self._image_paths: list[Path] | None = None
        self._fps: float | None = None
        self._frame_count: int | None = None
        self._size: tuple[int, int] | None = None
        self._probe(path)

    def _probe(self, original_path: str | Path) -> None:
        if self.kind == "images":
            self._image_paths = _resolve_image_paths(original_path)
            first = cv2.imread(str(self._image_paths[0]), cv2.IMREAD_COLOR)
            if first is None:
                raise OSError(f"failed to decode image: {self._image_paths[0]}")
            self._size = (int(first.shape[1]), int(first.shape[0]))
            count = len(self._image_paths)
            self._frame_count = (
                count if self.max_frames is None else min(count, self.max_frames)
            )
            self._fps = self._image_fps
            return

        video_path = Path(original_path).expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"video does not exist: {video_path}")
        self.path = video_path
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise OSError(f"OpenCV could not open video: {video_path}")
            width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if width <= 0 or height <= 0:
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise OSError(f"video contains no decodable frames: {video_path}")
                height, width = frame.shape[:2]
            self._size = (width, height)

            fps = float(capture.get(cv2.CAP_PROP_FPS))
            self._fps = fps if np.isfinite(fps) and fps > 0 else None
            raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            count = int(round(raw_count)) if np.isfinite(raw_count) and raw_count > 0 else None
            self._frame_count = count
            if self.max_frames is not None:
                self._frame_count = (
                    self.max_frames
                    if count is None
                    else min(count, self.max_frames)
                )
        finally:
            capture.release()

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def frame_count(self) -> int | None:
        """Reported/known count after applying ``max_frames``."""

        return self._frame_count

    @property
    def size(self) -> tuple[int, int]:
        """Frame size as ``(width, height)``."""

        assert self._size is not None
        return self._size

    @property
    def image_paths(self) -> tuple[Path, ...]:
        return tuple(self._image_paths or ())

    def __len__(self) -> int:
        if self._frame_count is None:
            raise TypeError("this video's frame count is unknown until decoded")
        return self._frame_count

    def __iter__(self) -> Iterator[FrameRecord]:
        return self.records()

    def records(self) -> Iterator[FrameRecord]:
        if self.kind == "images":
            yield from self._iter_images()
        else:
            yield from self._iter_video()

    def frames(self) -> Iterator[np.ndarray]:
        """Iterate just BGR image arrays, omitting metadata."""

        for record in self.records():
            yield record.image

    def _iter_images(self) -> Iterator[FrameRecord]:
        assert self._image_paths is not None
        expected_width, expected_height = self.size
        for index, path in enumerate(self._image_paths):
            if self.max_frames is not None and index >= self.max_frames:
                break
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"failed to decode image: {path}")
            if image.shape[:2] != (expected_height, expected_width):
                raise ValueError(
                    f"image sequence changed size at {path}: "
                    f"got {(image.shape[1], image.shape[0])}, expected {self.size}"
                )
            timestamp = index / self._fps if self._fps is not None else None
            yield FrameRecord(index, image, timestamp, str(path))

    def _iter_video(self) -> Iterator[FrameRecord]:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise OSError(f"OpenCV could not open video: {self.path}")
        index = 0
        try:
            while self.max_frames is None or index < self.max_frames:
                ok, image = capture.read()
                if not ok:
                    break
                if image is None:
                    raise OSError(f"video decoder returned an empty frame at {index}")
                timestamp = index / self._fps if self._fps is not None else None
                yield FrameRecord(index, image, timestamp, str(self.path))
                index += 1
        finally:
            capture.release()


def open_frame_source(
    path: str | Path,
    input_type: InputType = "auto",
    max_frames: int | None = None,
    *,
    image_fps: float | None = None,
) -> FrameSource:
    return FrameSource(
        path,
        input_type=input_type,
        max_frames=max_frames,
        image_fps=image_fps,
    )


def iter_frame_records(
    path: str | Path,
    input_type: InputType = "auto",
    max_frames: int | None = None,
    *,
    image_fps: float | None = None,
) -> Iterator[FrameRecord]:
    """One-shot convenience iterator with metadata."""

    return open_frame_source(
        path, input_type, max_frames, image_fps=image_fps
    ).records()


def iter_frames(
    path: str | Path,
    input_type: InputType = "auto",
    max_frames: int | None = None,
    *,
    image_fps: float | None = None,
) -> Iterator[np.ndarray]:
    """One-shot convenience iterator yielding only BGR arrays."""

    return open_frame_source(
        path, input_type, max_frames, image_fps=image_fps
    ).frames()


def load_frames(
    path: str | Path,
    input_type: InputType = "auto",
    max_frames: int | None = None,
    *,
    image_fps: float | None = None,
) -> list[np.ndarray]:
    """Decode all selected frames into memory."""

    return list(
        iter_frames(path, input_type, max_frames, image_fps=image_fps)
    )

