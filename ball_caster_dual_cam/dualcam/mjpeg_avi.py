"""Small classic AVI muxer that retains original webcam JPEG bytes exactly.

This is a single video stream, all keyframes, with a conventional idx1 index.
No JPEG decoding/re-encoding or OpenDML continuation is performed. CSV receive
timestamps remain authoritative; AVI's nominal frame rate controls playback only.
"""
from fractions import Fraction
import math
from pathlib import Path
import struct


UINT32_MAX = (1 << 32) - 1
SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
               0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


class AVISizeLimitError(ValueError):
    """A frame would exceed the finalized classic AVI file-size limit."""


def jpeg_payload(data):
    """Return the original JPEG, trimming up to 31 bytes of UVC buffer padding.

    Some cameras report a 32-byte-aligned buffer length beyond the JPEG's EOI.
    Padding can contain arbitrary bytes. Search only this bounded tail; choosing
    its first EOI also handles padding that happens to contain another EOI pair.
    The large entropy-coded payload is neither scanned in Python nor copied.
    """
    data = memoryview(data).cast("B")
    if len(data) < 4 or bytes(data[:2]) != b"\xff\xd8":
        raise ValueError("Raw MJPEG frame must start with JPEG SOI.")
    tail_start = max(2, len(data)-33)
    marker = bytes(data[tail_start:]).find(b"\xff\xd9")
    if marker < 0:
        raise ValueError("Raw MJPEG frame has no EOI within the allowed 31-byte UVC padding bound.")
    return data[:tail_start+marker+2]


def jpeg_dimensions(data):
    """Read width/height from a JPEG SOF header without decompressing pixels."""
    data = jpeg_payload(data)
    offset = 2
    while offset < len(data)-2:
        if data[offset] != 0xFF:
            raise ValueError("Invalid JPEG marker sequence before image dimensions.")
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in (0xD9, 0xDA):
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if offset+2 > len(data):
            break
        length = (int(data[offset]) << 8) | int(data[offset+1])
        if length < 2 or offset+length > len(data):
            raise ValueError("Truncated JPEG header segment.")
        if marker in SOF_MARKERS:
            if length < 8:
                raise ValueError("Invalid JPEG start-of-frame header.")
            height = (int(data[offset+3]) << 8) | int(data[offset+4])
            width = (int(data[offset+5]) << 8) | int(data[offset+6])
            if not width or not height:
                raise ValueError("JPEG dimensions must be positive.")
            return width, height
        offset += length
    raise ValueError("JPEG image dimensions were not found before scan data.")


def _chunk(tag, payload):
    return tag + struct.pack("<I", len(payload)) + payload + b"\0"*(len(payload) & 1)


def _list(kind, payload):
    return _chunk(b"LIST", kind+payload)


class MJPEGAVIWriter:
    """OpenCV-like write/release interface for already compressed JPEG buffers."""
    def __init__(self, path, fps, image_size, *, max_file_size=UINT32_MAX+8):
        width, height = image_size
        if (not all(isinstance(v, int) and 0 < v <= 32767 for v in (width, height))
                or not math.isfinite(fps) or fps <= 0):
            raise ValueError("AVI needs positive dimensions <=32767 and a positive finite FPS.")
        rate = Fraction(str(fps)).limit_denominator(1000000)
        if max(rate.numerator, rate.denominator) > UINT32_MAX:
            raise ValueError("Frame rate cannot be represented by AVI's 32-bit fields.")
        micros = round(1e6/fps)
        if not 1 <= micros <= UINT32_MAX:
            raise ValueError("Frame interval cannot be represented by AVI.")
        avih = struct.pack("<14I", micros, 0, 0, 0x10, 0, 0, 1, 0, width, height, 0, 0, 0, 0)
        strh = struct.pack("<4s4sIHH8I4h", b"vids", b"MJPG", 0, 0, 0,
                           0, rate.denominator, rate.numerator, 0, 0, 0, UINT32_MAX, 0,
                           0, 0, width, height)
        strf = struct.pack("<IiiHH4sIiiII", 40, width, height, 1, 24, b"MJPG",
                           width*height*3, 0, 0, 0, 0)
        header = b"RIFF" + struct.pack("<I", 0) + b"AVI " + _list(
            b"hdrl", _chunk(b"avih", avih) + _list(b"strl", _chunk(b"strh", strh)+_chunk(b"strf", strf)))
        self.limit = min(int(max_file_size), UINT32_MAX+8)
        if self.limit < len(header)+20:
            raise ValueError("AVI size limit is too small even for its headers.")
        self.image_size = (width, height)
        self.frames = 0
        self.max_frame_bytes = 0
        self.index = bytearray()
        self._avih_offset = header.index(b"avih")+8
        self._strh_offset = header.index(b"strh")+8
        self.stream = Path(path).open("xb")
        try:
            self.stream.write(header)
            self._movi_list_offset = self.stream.tell()
            self.stream.write(b"LIST\0\0\0\0movi")
        except BaseException:
            self.stream.close()
            raise

    def isOpened(self):
        return not self.stream.closed

    def write(self, data):
        if self.stream.closed:
            raise ValueError("Cannot write a finalized AVI.")
        data = jpeg_payload(data)
        if jpeg_dimensions(data) != self.image_size:
            raise ValueError(f"JPEG frame dimensions differ from AVI {self.image_size}.")
        length = len(data)
        position = self.stream.tell()
        # Reserve the complete index now, so release() never exceeds the limit.
        projected_size = position + 8 + length + (length & 1) + 8 + len(self.index) + 16
        if projected_size > self.limit or self.frames >= UINT32_MAX:
            raise AVISizeLimitError("Classic AVI RIFF size limit reached; start a new recording session.")
        self.stream.write(b"00dc" + struct.pack("<I", length))
        self.stream.write(data)
        if length & 1:
            self.stream.write(b"\0")
        # idx1 offsets are relative to the 'movi' list type, so the first is 4.
        offset = position - (self._movi_list_offset+8)
        self.index.extend(struct.pack("<4sIII", b"00dc", 0x10, offset, length))
        self.frames += 1
        self.max_frame_bytes = max(self.max_frame_bytes, length)

    def release(self):
        if self.stream.closed:
            return
        try:
            movi_end = self.stream.tell()
            self.stream.write(_chunk(b"idx1", self.index))
            end = self.stream.tell()
            patches = {4: end-8,
                       self._movi_list_offset+4: movi_end-self._movi_list_offset-8,
                       self._avih_offset+16: self.frames,
                       self._avih_offset+28: self.max_frame_bytes,
                       self._strh_offset+32: self.frames,
                       self._strh_offset+36: self.max_frame_bytes}
            for offset, value in patches.items():
                self.stream.seek(offset)
                self.stream.write(struct.pack("<I", value))
            self.stream.flush()
        finally:
            self.stream.close()


class SegmentedMJPEGAVIWriter:
    """Rotate bounded classic AVI files without changing JPEGs or frame order.

    ``segments`` is the live manifest: paths are relative basenames and indices
    refer to the complete camera stream, including every previous segment.
    """
    def __init__(self, path, fps, image_size, *, max_file_size=1024**3):
        if (isinstance(max_file_size, bool) or not isinstance(max_file_size, int)
                or max_file_size <= 0):
            raise ValueError("AVI segment size must be a positive integer number of bytes.")
        self.path = Path(path)
        self.fps, self.image_size = fps, image_size
        self.limit = min(max_file_size, UINT32_MAX+8)
        self.frames = 0
        self.segments = []
        self._closed = False
        self.writer = self._open_segment()

    def _open_segment(self):
        index = len(self.segments)
        path = (self.path if index == 0 else
                self.path.with_name(f"{self.path.stem}_{index:04d}{self.path.suffix}"))
        writer = MJPEGAVIWriter(path, self.fps, self.image_size, max_file_size=self.limit)
        self.segments.append({"path": path.name, "start_frame": self.frames, "frame_count": 0})
        return writer

    def isOpened(self):
        return not self._closed and self.writer.isOpened()

    def write(self, data):
        if self._closed:
            raise ValueError("Cannot write a finalized segmented AVI.")
        try:
            self.writer.write(data)
        except AVISizeLimitError:
            if self.writer.frames == 0:
                raise AVISizeLimitError("A single JPEG frame cannot fit within the AVI segment size limit.")
            self.writer.release()
            self.writer = self._open_segment()
            try:
                self.writer.write(data)
            except AVISizeLimitError as error:
                raise AVISizeLimitError("A single JPEG frame cannot fit within the AVI segment size limit.") from error
        self.frames += 1
        self.segments[-1]["frame_count"] += 1

    def release(self):
        if not self._closed:
            try:
                self.writer.release()
            finally:
                self._closed = True
