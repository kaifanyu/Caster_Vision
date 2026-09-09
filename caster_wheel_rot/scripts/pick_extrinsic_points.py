#!/usr/bin/env python
"""Click the extrinsics correspondence points on a still from the fixed camera.

You supply the 3D side - a small CSV of ``label,x,y,z`` measured in your chosen
car frame - and this clicks out the 2D side, writing the ``x,y,z,u,v`` file the
``extrinsics`` subcommand reads.

Pixels are taken on the raw, still-distorted frame: the solver applies
``camera.dist`` itself, so undistorting first would double-correct.  Picking is
two stage, a rough click on the fitted overview then a precise click in a
magnified crop, because a 1920-wide frame shrunk to fit a window costs several
pixels of accuracy and extrinsics are only as good as these clicks.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

ZOOM = 6
CROP = 60  # half-width of the magnified crop, in source pixels


def load_targets(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} has no data rows")
    out = []
    for i, row in enumerate(rows):
        low = {str(k).strip().casefold(): v for k, v in row.items()}
        try:
            out.append({
                "label": low.get("label") or f"point{i}",
                "x": float(low["x"]), "y": float(low["y"]), "z": float(low["z"]),
            })
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"{path} needs label,x,y,z columns: {exc}") from exc
    return out


def grab_frame(clip: Path, index: int) -> np.ndarray:
    if clip.suffix.casefold() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        image = cv2.imread(str(clip))
        if image is None:
            raise SystemExit(f"could not read image {clip}")
        return image
    cap = cv2.VideoCapture(str(clip))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {index} from {clip}")
    return image


def pick_one(frame: np.ndarray, label: str, done: list[tuple[float, float]]):
    """Rough click on the overview, then a precise click in a zoomed crop."""
    h, w = frame.shape[:2]
    scale = min(1.0, 1400.0 / w, 800.0 / h)
    view = cv2.resize(frame, (int(w * scale), int(h * scale)))
    for (u, v) in done:
        cv2.drawMarker(view, (int(u * scale), int(v * scale)), (0, 200, 0),
                       cv2.MARKER_CROSS, 12, 1)
    cv2.putText(view, f"click near: {label}   (s=skip, q=quit)", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    rough: list[tuple[int, int]] = []
    cv2.namedWindow("overview", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("overview", lambda e, x, y, f, p: (
        rough.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None))
    while not rough:
        cv2.imshow("overview", view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("aborted")
        if key == ord("s"):
            cv2.destroyWindow("overview")
            return None
    cv2.destroyWindow("overview")

    cu, cv_ = rough[0][0] / scale, rough[0][1] / scale
    x0 = int(np.clip(cu - CROP, 0, max(0, w - 2 * CROP)))
    y0 = int(np.clip(cv_ - CROP, 0, max(0, h - 2 * CROP)))
    crop = frame[y0:y0 + 2 * CROP, x0:x0 + 2 * CROP]
    big = cv2.resize(crop, (crop.shape[1] * ZOOM, crop.shape[0] * ZOOM),
                     interpolation=cv2.INTER_NEAREST)
    cv2.line(big, (big.shape[1] // 2, 0), (big.shape[1] // 2, big.shape[0]), (80, 80, 80), 1)
    cv2.line(big, (0, big.shape[0] // 2), (big.shape[1], big.shape[0] // 2), (80, 80, 80), 1)
    cv2.putText(big, f"precise: {label}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)

    fine: list[tuple[int, int]] = []
    cv2.namedWindow("zoom", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("zoom", lambda e, x, y, f, p: (
        fine.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None))
    while not fine:
        cv2.imshow("zoom", big)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("aborted")
        if key == ord("s"):
            cv2.destroyWindow("zoom")
            return None
    cv2.destroyWindow("zoom")
    return (x0 + fine[0][0] / ZOOM, y0 + fine[0][1] / ZOOM)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", type=Path, required=True, help="video or still image")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--points", type=Path, required=True, help="CSV of label,x,y,z")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--save-still", type=Path, help="also write the frame used")
    args = p.parse_args(argv)

    targets = load_targets(args.points)
    frame = grab_frame(args.clip, args.frame)
    if args.save_still:
        args.save_still.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_still), frame)
        print(f"still written to {args.save_still}")

    picked, done = [], []
    for t in targets:
        uv = pick_one(frame, t["label"], done)
        if uv is None:
            print(f"  skipped {t['label']}")
            continue
        done.append(uv)
        picked.append({**t, "u": uv[0], "v": uv[1]})
        print(f"  {t['label']:<20} car=({t['x']:.3f},{t['y']:.3f},{t['z']:.3f})  "
              f"pixel=({uv[0]:.1f},{uv[1]:.1f})")
    cv2.destroyAllWindows()

    if len(picked) < 4:
        raise SystemExit(f"only {len(picked)} points picked; the solver needs at least 4")
    heights = {round(r["z"], 4) for r in picked}
    if len(heights) < 2:
        print("\nWARNING every point shares one z. A planar set solves poorly - "
              "add points at a different height and re-run.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "z", "u", "v"])
        for r in picked:
            writer.writerow([r["x"], r["y"], r["z"], f"{r['u']:.2f}", f"{r['v']:.2f}"])
    print(f"\nwrote {len(picked)} correspondences to {args.output}")
    print("next: calibrate_swivel.py extrinsics --correspondences "
          f"{args.output} --max-rms-px 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
