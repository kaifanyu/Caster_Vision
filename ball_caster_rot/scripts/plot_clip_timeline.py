"""Export timestamp-checked real-video snapshots beside the motion-history plots.

Run from the project directory, for example::

    .venv\\Scripts\\python.exe scripts\\plot_clip_timeline.py \
        --results out/real_20260910_aligned/results.json \
        --output out/motion_history_20260910
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ballrot.camera import undistort_image
from ballrot.io_frames import FrameSource


def make_clip_timeline(payload: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Save a six-frame filmstrip using the geometry and timing in saved results.

    All source records are decoded sequentially and checked against results,
    including records not pictured.  A timestamp or count mismatch raises an
    error instead of silently presenting frames from a different run.
    """

    metadata = payload["metadata"]
    times = np.asarray(payload["frames"]["time_s"], dtype=float)
    if times.ndim != 1 or times.size < 2 or not np.isfinite(times).all():
        raise ValueError("Results require at least two finite frame timestamps")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Results timestamps must strictly increase")
    if int(metadata["frame_count"]) != times.size:
        raise ValueError("Results frame_count does not match the saved time array")

    source_path = Path(metadata["input"])
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    source = FrameSource(source_path, image_fps=float(metadata["fps"]))
    matrix = np.asarray(metadata["K"], dtype=float)
    distortion = np.asarray(metadata["dist"], dtype=float)
    circle = np.asarray(metadata["circle"], dtype=float)
    if circle.shape != (3,) or not np.isfinite(circle).all() or circle[2] <= 0:
        raise ValueError("Results require a finite, positive-radius circle")
    width, height = source.size
    radius = circle[2] + 80.0
    left = max(0, int(np.floor(circle[0] - radius)))
    right = min(width, int(np.ceil(circle[0] + radius)))
    upper = max(0, int(np.floor(circle[1] - radius)))
    lower = min(height, int(np.ceil(circle[1] + radius)))
    if right <= left or lower <= upper:
        raise ValueError("The saved circle does not intersect the source video")

    # Retain the requested five-second landmarks for this recording; for other
    # durations use equally spaced times to avoid duplicate final snapshots.
    targets = (
        np.asarray([0.0, 5.0, 10.0, 15.0, 20.0, times[-1]])
        if 20.0 < times[-1] < 25.0
        else np.linspace(times[0], times[-1], 6)
    )
    indices = [int(np.argmin(np.abs(times - target))) for target in targets]
    selected: dict[int, np.ndarray] = {}
    decoded_count = 0
    for record in source.records():
        if record.index >= times.size:
            raise ValueError("Source video has more frames than these saved results")
        if record.timestamp_s is None or not np.isclose(
            record.timestamp_s, times[record.index], rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                f"Source timestamp mismatch at frame {record.index}: "
                f"{record.timestamp_s!r} versus {times[record.index]:.9f} s. "
                "Use results with native source timing."
            )
        if record.index in indices:
            corrected = undistort_image(record.image, matrix, distortion)
            selected[record.index] = cv2.cvtColor(
                corrected[upper:lower, left:right], cv2.COLOR_BGR2RGB
            )
        decoded_count += 1
    if decoded_count != times.size:
        raise ValueError(
            f"Source decoded {decoded_count} frames; saved results have {times.size}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    style = {
        "font.family": "DejaVu Sans",
        "text.color": "#17243a",
        "axes.titlecolor": "#17243a",
        "figure.facecolor": "#f7f9fc",
        "savefig.facecolor": "#f7f9fc",
        "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(2, 3, figsize=(15.0, 11.0))
        figure.subplots_adjust(left=0.026, right=0.974, bottom=0.082, top=0.867,
                               wspace=0.022, hspace=0.15)
        figure.text(0.027, 0.955, "What the camera saw", fontsize=27, weight="bold")
        figure.text(
            0.027, 0.918,
            "Real frames at measured timestamps  |  Same view throughout the recording",
            fontsize=12.5, color="#506078",
        )
        for axis, index in zip(axes.flat, indices):
            axis.imshow(selected[index], interpolation="antialiased")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.set_xlabel(
                f"{times[index]:.3f} s   /   frame {index}",
                fontsize=13, labelpad=8, color="#17243a",
            )
        figure.text(
            0.027, 0.040,
            "Red and green marks identify the two tracked shells. "
            "Frames are lens-corrected and cropped around the annotated circle.",
            fontsize=10.7, color="#506078",
        )
        figure.text(
            0.027, 0.019,
            f"Source: {source_path.name}  |  {decoded_count} decoded frames  |  "
            f"{times[-1]:.3f} s  |  Frame numbers start at zero.",
            fontsize=10.2, color="#506078",
        )
        paths = {}
        for extension in ("png", "svg"):
            destination = output_dir / f"05_clip_timeline.{extension}"
            figure.savefig(destination, dpi=180)
            paths[extension] = str(destination.resolve())
        plt.close(figure)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.results.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    print(json.dumps(make_clip_timeline(payload, args.output), indent=2))


if __name__ == "__main__":
    main()
