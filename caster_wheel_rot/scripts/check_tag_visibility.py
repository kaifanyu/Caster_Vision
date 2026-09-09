#!/usr/bin/env python
"""Check that the fork tag stays visible through a full swivel sweep.

This deliberately needs no calibration at all - not intrinsics, extrinsics,
psi0, or geometry.  It answers the mounting questions: as the fork turns
through 360 degrees, does anything hide the marker, and is the marker actually
level?  Because psi gates roll, every frame that loses the tag also loses roll,
so a mount is only finished when coverage is continuous.

Tag tilt is recovered without knowing the camera pose.  A tag rigidly fixed to
the fork has its normal trace a cone about the swivel axis as the fork turns:
a level tag holds one fixed direction, a tilted one sweeps a circle.  Fitting a
plane to the observed normals therefore yields both the tilt (the cone's half
angle) and the swivel axis in camera coordinates, which in turn gives the
camera's elevation.  An approximate focal length is enough for these directions,
so this stays a diagnostic and is never a substitute for calibration.

Record a slow hand sweep from the camera's real position and run this before
spending time on the calibration chain.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from common.io_frames import open_frame_source
from common.rotation import geodesic_angle
from swivel.tag import (
    _DetectorBackend,
    aruco_dictionary,
    detector_parameters,
    solve_tag_pose,
)


def approx_camera_matrix(width: int, height: int, hfov_deg: float) -> np.ndarray:
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def reject_pose_flips(normals: np.ndarray, window: int = 3, max_jump_deg: float = 25.0):
    """Drop mirror-branch flips using temporal continuity.

    A planar marker has two IPPE solutions, and near-degenerate reprojection
    error lets the solver pick the mirror on the odd frame.  Such a flip throws
    the normal ~140 deg in a single frame, which the fork cannot physically do.
    A reprojection gate does not separate them - observed flips sat at 1.9-2.1 px
    while honest poses reached 3.5 px - so reject on motion instead.
    """
    n = len(normals)
    if n < 5:
        return np.ones(n, dtype=bool)
    keep = np.ones(n, dtype=bool)
    limit = math.cos(math.radians(max_jump_deg))
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        neighbours = np.delete(np.arange(lo, hi), np.where(np.arange(lo, hi) == i))
        if not len(neighbours):
            continue
        local = np.median(normals[neighbours], axis=0)
        norm = float(np.linalg.norm(local))
        if norm < 1e-9:
            continue
        if float(normals[i] @ (local / norm)) < limit:
            keep[i] = False
    return keep


def fit_normal_cone(normals: np.ndarray):
    """Fit the cone the tag normal sweeps; return (axis, tilt_deg, resid_deg).

    The normals lie on a circle on the unit sphere, so a plane fit recovers the
    cone: the plane's normal is the swivel axis and its offset is cos(tilt).
    Flipped poses are dropped first - a tight cluster plus two far outliers is
    still fitted well by a plane through both, so the residual cannot reveal
    them and a plain least-squares fit is skewed by tens of degrees.
    """
    if len(normals) < 8:
        return None, float("nan"), float("nan"), 0
    keep = reject_pose_flips(normals)
    kept = normals[keep]
    dropped = int((~keep).sum())
    if len(kept) < 8:
        return None, float("nan"), float("nan"), dropped
    centre = kept.mean(axis=0)
    _, _, vh = np.linalg.svd(kept - centre)
    axis = vh[-1]
    if float(axis @ centre) < 0.0:
        axis = -axis
    offset = float(np.mean(kept @ axis))
    tilt = math.degrees(math.acos(float(np.clip(offset, -1.0, 1.0))))
    resid = math.degrees(float(np.std(kept @ axis)))
    return axis, tilt, resid, dropped


def visible_arc_deg(rotations, axis):
    """True swivel arc a tag is visible over, in degrees.

    Counting detected frames only gives the arc if the sweep covered 360 deg at
    uniform speed, which a hand sweep never does.  Instead take each pose's
    azimuth about the swivel axis: the visible frames occupy an arc on that
    circle, so the arc width is 360 minus the largest empty gap between them.
    """
    if len(rotations) < 4:
        return float("nan")
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(seed @ axis)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = seed - float(seed @ axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    ang = []
    for R in rotations:
        v = R[:, 0] - float(R[:, 0] @ axis) * axis
        if np.linalg.norm(v) < 1e-9:
            continue
        ang.append(math.atan2(float(v @ e2), float(v @ e1)))
    if len(ang) < 4:
        return float("nan")
    a = np.sort(np.mod(np.asarray(ang), 2.0 * np.pi))
    gaps = np.diff(np.concatenate([a, a[:1] + 2.0 * np.pi]))
    return float(math.degrees(2.0 * np.pi - gaps.max()))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", type=Path, required=True)
    p.add_argument("--marker-id", type=int, nargs="+", default=[0],
                   help="one or more ids; each is reported separately and combined")
    p.add_argument("--dictionary", default="DICT_4X4_50")
    p.add_argument("--marker-size-m", type=float, default=0.040)
    p.add_argument("--hfov-deg", type=float, default=60.0,
                   help="approximate horizontal FOV, for the tilt diagnostic only")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--input-type", choices=("auto", "video", "images"), default="auto")
    p.add_argument("--min-coverage", type=float, default=0.98)
    p.add_argument("--min-area-px2", type=float, default=400.0)
    p.add_argument("--max-tilt-deg", type=float, default=10.0)
    p.add_argument("--dump-dir", type=Path,
                   help="write the last-good and mid-dropout frames here")
    args = p.parse_args(argv)

    backend = _DetectorBackend(aruco_dictionary(args.dictionary), detector_parameters())
    source = open_frame_source(args.clip, input_type=args.input_type, image_fps=args.fps)
    fps = float(source.fps or args.fps)

    ids_wanted = list(dict.fromkeys(int(v) for v in args.marker_id))
    per_id: dict[int, list[bool]] = {i: [] for i in ids_wanted}
    covis: list[int] = []
    seen: list[bool] = []
    areas: list[float] = []
    normals: dict[int, list[np.ndarray]] = {i: [] for i in ids_wanted}
    rots: dict[int, list[np.ndarray]] = {i: [] for i in ids_wanted}
    poses: list[dict[int, np.ndarray]] = []
    keep: dict[int, np.ndarray] = {}
    prev_rot: dict[int, np.ndarray] = {}
    K = None

    for record in source:
        image = np.asarray(record.image)
        if K is None:
            K = approx_camera_matrix(image.shape[1], image.shape[0], args.hfov_deg)
        corners, ids, _ = backend.detect(image)
        flat = (np.asarray(ids, dtype=int).reshape(-1)
                if ids is not None and len(ids) else np.empty(0, dtype=int))
        n_here = 0
        frame_poses: dict[int, np.ndarray] = {}
        for want in ids_wanted:
            match = np.flatnonzero(flat == want)
            per_id[want].append(bool(len(match)))
            if not len(match):
                continue
            n_here += 1
            quad = np.asarray(corners[match[0]], dtype=np.float32).reshape(4, 2)
            if want == ids_wanted[0]:
                areas.append(abs(float(cv2.contourArea(quad))))
            try:
                # Without a continuity hint the two planar IPPE branches are
                # near-degenerate in reprojection error, so the recovered normal
                # flips frame to frame and corrupts the cone fit.
                pose = solve_tag_pose(quad, args.marker_size_m, K,
                                      previous_R_tag_to_camera=prev_rot.get(want))
                prev_rot[want] = pose.R_tag_to_camera
                normals[want].append(pose.R_tag_to_camera[:, 2])
                rots[want].append(pose.R_tag_to_camera)
                frame_poses[want] = pose.R_tag_to_camera
            except Exception:
                pass
            if len(match) > 1:
                print(f"  frame {record.index}: {len(match)} markers share id {want} "
                      f"- the tracker keeps only the largest")
        covis.append(n_here)
        poses.append(frame_poses)
        seen.append(n_here > 0)
        if args.dump_dir is not None:
            keep[record.index] = image.copy()

    total = len(seen)
    if not total:
        raise SystemExit("no frames were read")
    flags = np.asarray(seen, dtype=bool)
    coverage = float(flags.mean())

    longest, run, start_at, best_at = 0, 0, 0, 0
    for i, ok in enumerate(flags):
        if ok:
            run = 0
        else:
            if run == 0:
                start_at = i
            run += 1
            if run > longest:
                longest, best_at = run, start_at

    print(f"\nframes                {total}")
    print(f"any tag detected      {int(flags.sum())}  ({coverage:.1%})")
    if len(ids_wanted) > 1:
        for want in ids_wanted:
            f = np.asarray(per_id[want], dtype=bool)
            print(f"  id {want:<3}               {int(f.sum()):>4}  ({f.mean():.1%} of frames)")
        c = np.asarray(covis)
        both = int((c >= 2).sum())
        print(f"  >=2 tags together    {both:>4}  ({float((c >= 2).mean()):.1%})"
              f"   ties the per-tag zeros")
        if both < 10:
            print("  WARNING too few co-visible frames: the per-tag zeros cannot be tied, "
                  "so psi will step at every handoff")
    print(f"longest dropout       {longest} frames ({longest / fps:.2f} s)"
          + (f", frames {best_at}-{best_at + longest - 1}" if longest else ""))
    if areas:
        a = np.asarray(areas)
        print(f"marker area px^2      median {np.median(a):.0f}  min {a.min():.0f}  max {a.max():.0f}")
        print(f"apparent side px      median {np.sqrt(np.median(a)):.1f}  min {np.sqrt(a.min()):.1f}")

    print("")
    tilts: dict[int, float] = {}
    axes: dict[int, np.ndarray] = {}
    for want in ids_wanted:
        pts = np.asarray(normals[want])
        if len(pts) < 8:
            print(f"tag {want} tilt: too few detections to fit")
            continue
        axis, t, resid, dropped = fit_normal_cone(pts)
        if axis is None:
            print(f"tag {want} tilt: fit failed after dropping flips")
            continue
        tilts[want] = t
        axes[want] = axis
        elev = math.degrees(math.atan2(-float(axis[2]), -float(axis[1])))
        flips = f", {dropped} flipped pose{'s' if dropped != 1 else ''} dropped" if dropped else ""
        print(f"tag {want} tilt from level  {t:.1f} deg  (cone residual {resid:.1f} deg, "
              f"camera elevation ~{elev:.0f} deg{flips})")
    worst_tilt = max(tilts.values()) if tilts else float("nan")

    arcs: dict[int, float] = {}
    if axes:
        shared = np.mean(np.asarray(list(axes.values())), axis=0)
        shared /= np.linalg.norm(shared)
        print("")
        for want in ids_wanted:
            arc = visible_arc_deg(rots[want], shared)
            if np.isfinite(arc):
                arcs[want] = arc
                print(f"tag {want} visible arc     {arc:.0f} deg of swivel "
                      f"(measured, not inferred from frame count)")

    # The relative rotation between two tags on one fork is constant and frame
    # independent, so its rotation angle is exactly the mounting separation.
    for i in range(len(ids_wanted)):
        for j in range(i + 1, len(ids_wanted)):
            a, b = ids_wanted[i], ids_wanted[j]
            seps = [math.degrees(float(geodesic_angle(fp[a].T @ fp[b], np.eye(3))))
                    for fp in poses if a in fp and b in fp]
            if len(seps) >= 3:
                print(f"tags {a}-{b} separation    {np.mean(seps):.1f} deg "
                      f"(sd {np.std(seps):.1f} deg over {len(seps)} co-visible frames)")
            else:
                print(f"tags {a}-{b} separation    not measurable - no co-visible frames")

    ok = True
    if coverage < args.min_coverage:
        print(f"\nFAIL coverage {coverage:.1%} < {args.min_coverage:.1%}")
        ok = False
    if longest > 2:
        print(f"FAIL {longest}-frame continuous dropout is occlusion, not blur")
        ok = False
    if np.isfinite(worst_tilt) and worst_tilt > args.max_tilt_deg:
        print(f"FAIL a tag is {worst_tilt:.0f} deg off level (want under "
              f"{args.max_tilt_deg:.0f}); its normal sweeps a cone and turns away")
        ok = False
    if len(ids_wanted) > 1 and arcs:
        need = 360.0 / len(ids_wanted) + 10.0
        thin = {i: a for i, a in arcs.items() if a < need}
        if thin:
            worst = min(thin, key=thin.get)
            print(f"FAIL with {len(ids_wanted)} tags each arc must exceed ~{need:.0f} deg "
                  f"to close the circle with overlap; id {worst} is only {thin[worst]:.0f} deg")
            ok = False
        if int((np.asarray(covis) >= 2).sum()) < 10:
            print("FAIL fewer than 10 co-visible frames; the per-tag zeros cannot be tied "
                  "and psi will step at every handoff")
            ok = False
    if areas and float(np.min(areas)) < args.min_area_px2:
        print(f"FAIL smallest marker is {np.sqrt(min(areas)):.0f} px across; pose gets unreliable")
        ok = False

    if args.dump_dir is not None and longest:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        for idx, name in ((max(best_at - 1, 0), "last_good"),
                          (best_at + longest // 2, "dropout_mid")):
            if idx in keep:
                cv2.imwrite(str(args.dump_dir / f"frame{idx:04d}_{name}.jpg"), keep[idx])
        print(f"\nwrote boundary frames to {args.dump_dir}")

    if ok:
        print("\nPASS - tag is level and continuously visible through the sweep")
    else:
        print("\nlevel the tag first, then move it clear of whatever occludes it")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
