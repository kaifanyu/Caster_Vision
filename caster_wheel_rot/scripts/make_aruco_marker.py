#!/usr/bin/env python
"""Create a print-to-100%-scale fork marker with a measured quiet border."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swivel.tag import aruco_dictionary, generate_marker_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--marker-size-mm", type=float, default=40.0, help="Black marker square, excluding white border")
    parser.add_argument("--quiet-border-mm", type=float, default=10.0)
    parser.add_argument("--paper", choices=("letter", "a4"), default="letter")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "out" / "calibration" / "aruco_marker.pdf")
    parser.add_argument("--dpi", type=int, default=600)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.marker_size_mm <= 0 or args.quiet_border_mm < 0 or args.dpi < 150:
        raise SystemExit("marker size must be positive, border nonnegative, and DPI >= 150")
    dictionary = aruco_dictionary(args.dictionary)
    marker = generate_marker_image(dictionary, args.id, 800)
    quiet_px = int(round(marker.shape[0] * args.quiet_border_mm / args.marker_size_mm))
    plate = np.pad(marker, quiet_px, constant_values=255)
    plate_mm = args.marker_size_mm + 2.0 * args.quiet_border_mm
    page_mm = (215.9, 279.4) if args.paper == "letter" else (210.0, 297.0)
    figure = plt.figure(figsize=(page_mm[0] / 25.4, page_mm[1] / 25.4), dpi=args.dpi)
    left = (page_mm[0] - plate_mm) / (2.0 * page_mm[0])
    bottom = (page_mm[1] - plate_mm) / (2.0 * page_mm[1])
    axis = figure.add_axes([left, bottom, plate_mm / page_mm[0], plate_mm / page_mm[1]])
    axis.imshow(plate, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axis.axis("off")
    figure.text(
        0.5,
        0.88,
        f"{args.dictionary} ID {args.id} — black square = {args.marker_size_mm:g} mm",
        ha="center",
        va="center",
        fontsize=10,
    )
    figure.text(
        0.5,
        0.10,
        "Print at Actual size / 100%. Measure the 20 mm bar before mounting.",
        ha="center",
        fontsize=9,
    )
    # Exact 20 mm verification bar in page coordinates.
    bar_width = 20.0 / page_mm[0]
    bar_axis = figure.add_axes([0.5 - bar_width / 2, 0.13, bar_width, 0.012])
    bar_axis.imshow(np.zeros((4, 100)), cmap="gray", vmin=0, vmax=255, aspect="auto")
    bar_axis.axis("off")
    figure.text(0.5, 0.147, "20 mm", ha="center", fontsize=8)
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=args.dpi, facecolor="white")
    preview = destination.with_suffix(".png")
    figure.savefig(preview, dpi=180, facecolor="white")
    plt.close(figure)
    print(f"Marker PDF: {destination}")
    print(f"Preview:    {preview}")
    print("Print at 100%/Actual size and verify the 20 mm bar with calipers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
