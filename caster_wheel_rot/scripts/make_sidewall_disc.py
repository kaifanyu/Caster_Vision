#!/usr/bin/env python
"""Generate print-ready random-speckle sidewall discs for the swivel caster.

The roll estimator tracks grayscale KLT corners inside a geometrically derived
annulus, so the artwork only has to be flat, high contrast, and *rotationally
unambiguous*.  An evenly spaced ring of identical marks is the classic failure:
after one spacing the image repeats and the tracker locks onto the wrong mark.
This generator uses blue-noise placement with mixed shapes and sizes, then
measures the residual rotational self-similarity and refuses to emit artwork
that is still ambiguous.

Two faces are produced.  Face B mirrors the index notch and the reference dot so
that aligning both notches to the same physical datum puts both reference dots
at the same fork-frame angle, which is what keeps the phase continuous when the
visible sidewall flips during swivel.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

PAGE_W_MM = 215.9
PAGE_H_MM = 279.4
GREEN = (0.0, 0.65, 0.31)


def mm_per_pixel(standoff_m: float, hfov_deg: float, width_px: int) -> float:
    span_mm = 2.0 * standoff_m * math.tan(math.radians(hfov_deg) / 2.0) * 1000.0
    return span_mm / float(width_px)


def poisson_annulus(r_in, r_out, count, min_dist, rng, max_tries=400000):
    points: list[np.ndarray] = []
    tries = 0
    while len(points) < count and tries < max_tries:
        tries += 1
        r = math.sqrt(rng.uniform(r_in ** 2, r_out ** 2))
        a = rng.uniform(0.0, 2.0 * math.pi)
        p = np.array([r * math.cos(a), r * math.sin(a)])
        if points:
            d = np.linalg.norm(np.asarray(points) - p, axis=1)
            if d.min() < min_dist:
                continue
        points.append(p)
    return np.asarray(points)


def chance_match_fraction(points, tol_mm, r_lo, r_hi):
    """Fraction of dots a *random* rotation coincides with by luck alone.

    The gate below is only meaningful relative to this floor: with a loose
    tolerance a perfectly good blue-noise field already self-matches heavily,
    which says nothing about whether the pattern repeats under rotation.
    """
    annulus = math.pi * (r_hi ** 2 - r_lo ** 2)
    return float(len(points)) * math.pi * tol_mm ** 2 / annulus


def rotational_ambiguity(points, tol_mm, exclude_deg=3.0, step_deg=0.25):
    """Worst non-identity rotational self-match fraction of the dot field."""
    best_angle, best_frac = 0.0, 0.0
    for deg in np.arange(step_deg, 360.0, step_deg):
        if deg < exclude_deg or deg > 360.0 - exclude_deg:
            continue
        t = math.radians(float(deg))
        rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        moved = points @ rot.T
        d = np.linalg.norm(moved[:, None, :] - points[None, :, :], axis=2)
        frac = float(np.mean(d.min(axis=1) < tol_mm))
        if frac > best_frac:
            best_frac, best_angle = frac, float(deg)
    return best_frac, best_angle


def draw_face(path, points, sizes, shapes, angles, r_in, r_out, ref_xy, ref_d,
              notch_angle_deg, label, meta):
    fig = plt.figure(figsize=(PAGE_W_MM / 25.4, PAGE_H_MM / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W_MM)
    ax.set_ylim(0, PAGE_H_MM)
    ax.set_aspect("equal")
    ax.axis("off")

    cx, cy = PAGE_W_MM / 2.0, PAGE_H_MM / 2.0 + 12.0

    # White disc body so the red rim never shows through a thin print.
    ax.add_patch(Circle((cx, cy), r_out, facecolor="white", edgecolor="none", zorder=1))

    for (x, y), s, shape, ang in zip(points, sizes, shapes, angles):
        px, py = cx + x, cy + y
        if shape == 0:
            ax.add_patch(Circle((px, py), s / 2.0, facecolor="black",
                                edgecolor="none", zorder=3))
        elif shape == 1:
            ax.add_patch(Rectangle((px - s / 2.0, py - s / 2.0), s, s, angle=ang,
                                   rotation_point="center", facecolor="black",
                                   edgecolor="none", zorder=3))
        else:
            t = math.radians(ang)
            verts = [(px + s * 0.62 * math.cos(t + k * 2.0944),
                      py + s * 0.62 * math.sin(t + k * 2.0944)) for k in range(3)]
            ax.add_patch(Polygon(verts, closed=True, facecolor="black",
                                 edgecolor="none", zorder=3))

    ax.add_patch(Circle((cx + ref_xy[0], cy + ref_xy[1]), ref_d / 2.0,
                        facecolor=GREEN, edgecolor="none", zorder=4))

    # Cut guides: dashed so they contribute no continuous edge feature.
    for r in (r_in, r_out):
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor="0.75",
                            lw=0.3, ls=(0, (2, 2)), zorder=5))

    # Index notch for through-alignment of the two faces.
    t = math.radians(notch_angle_deg)
    tip = (cx + (r_out + 5.0) * math.cos(t), cy + (r_out + 5.0) * math.sin(t))
    base = 3.0
    n1 = (cx + r_out * math.cos(t) - base * math.sin(t),
          cy + r_out * math.sin(t) + base * math.cos(t))
    n2 = (cx + r_out * math.cos(t) + base * math.sin(t),
          cy + r_out * math.sin(t) - base * math.cos(t))
    ax.add_patch(Polygon([tip, n1, n2], closed=True, facecolor="black", zorder=6))

    y = 30.0
    ax.plot([20.0, 70.0], [y, y], color="black", lw=1.0)
    for xt in (20.0, 70.0):
        ax.plot([xt, xt], [y - 2.0, y + 2.0], color="black", lw=1.0)
    ax.text(72.0, y, "50.0 mm - caliper this after printing", va="center", fontsize=7)

    ax.text(20.0, PAGE_H_MM - 18.0, f"Swivel caster sidewall - FACE {label}",
            fontsize=11, weight="bold")
    ax.text(20.0, PAGE_H_MM - 25.0,
            f"OD {2 * r_out:.2f} mm   ID {2 * r_in:.2f} mm   {len(points)} marks   "
            f"dot {meta['dot_min_mm']:.2f}-{meta['dot_max_mm']:.2f} mm "
            f"({meta['dot_min_px']:.1f}-{meta['dot_max_px']:.1f} px @ "
            f"{meta['standoff_m']:.2f} m)",
            fontsize=7.5)
    ax.text(20.0, PAGE_H_MM - 31.0,
            f"Print at 100% / Actual Size. Rotational ambiguity "
            f"{meta['ambiguity']:.3f} (worst at {meta['ambiguity_angle']:.1f} deg). "
            f"Align notch to the shared datum.",
            fontsize=7.5)
    ax.text(20.0, 18.0,
            "Cut on the dashed circles. Burnish flat - creases break the planar "
            "sidewall model.", fontsize=7)

    fig.savefig(path, format="pdf")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outer-diameter-mm", type=float, required=True)
    p.add_argument("--inner-diameter-mm", type=float, required=True)
    p.add_argument("--standoff-m", type=float, default=0.35)
    p.add_argument("--hfov-deg", type=float, default=60.0)
    p.add_argument("--image-width-px", type=int, default=1920)
    p.add_argument("--marks", type=int, default=130)
    p.add_argument("--max-ambiguity", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).resolve().parents[1] / "out" / "calibration")
    args = p.parse_args()

    r_out = args.outer_diameter_mm / 2.0
    r_in = args.inner_diameter_mm / 2.0
    if not 0 < r_in < r_out:
        raise SystemExit("inner diameter must be positive and smaller than outer")

    mmpx = mm_per_pixel(args.standoff_m, args.hfov_deg, args.image_width_px)
    dot_min = max(3.0 * mmpx, 0.45)
    dot_max = max(8.0 * mmpx, dot_min * 1.9)

    margin = 1.5 + dot_max / 2.0
    r_lo, r_hi = r_in + margin, r_out - margin
    area = math.pi * (r_hi ** 2 - r_lo ** 2)
    min_dist = 0.75 * math.sqrt(area / args.marks)

    # Match tolerance is KLT localization scale (3 px), not dot size: a loose
    # tolerance measures density, not rotational symmetry.
    tol_mm = max(0.5, 3.0 * mmpx)

    rng = np.random.default_rng(args.seed)
    pts = poisson_annulus(r_lo, r_hi, args.marks, min_dist, rng)

    chance = chance_match_fraction(pts, tol_mm, r_lo, r_hi)
    amb, amb_angle = rotational_ambiguity(pts, tol_mm=tol_mm)
    if amb > args.max_ambiguity:
        raise SystemExit(
            f"pattern is rotationally ambiguous ({amb:.3f} at {amb_angle:.1f} deg, "
            f"chance floor {chance:.3f})"
        )

    sizes = rng.uniform(dot_min, dot_max, len(pts))
    shapes = rng.integers(0, 3, len(pts))
    angles = rng.uniform(0, 360, len(pts))

    ref_r = 0.5 * (r_lo + r_hi)
    ref_d = 3.5 * float(np.mean(sizes))
    notch_deg = 0.0
    ref_deg = 90.0

    meta = {
        "outer_diameter_mm": args.outer_diameter_mm,
        "inner_diameter_mm": args.inner_diameter_mm,
        "marks": int(len(pts)),
        "dot_min_mm": dot_min,
        "dot_max_mm": dot_max,
        "dot_min_px": dot_min / mmpx,
        "dot_max_px": dot_max / mmpx,
        "reference_dot_mm": ref_d,
        "reference_radius_mm": ref_r,
        "mm_per_px": mmpx,
        "standoff_m": args.standoff_m,
        "min_separation_mm": min_dist,
        "match_tolerance_mm": tol_mm,
        "ambiguity": amb,
        "ambiguity_angle": amb_angle,
        "ambiguity_chance_floor": chance,
        "wheel_radius_m_for_config": r_out / 1000.0,
        "sidewall_inner_fraction_for_config": r_in / r_out,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    a = args.output_dir / "sidewall_face_A.pdf"
    b = args.output_dir / "sidewall_face_B.pdf"

    ref_a = (ref_r * math.cos(math.radians(ref_deg)),
             ref_r * math.sin(math.radians(ref_deg)))
    draw_face(a, pts, sizes, shapes, angles, r_in, r_out, ref_a, ref_d, notch_deg, "A", meta)

    rng_b = np.random.default_rng(args.seed + 1)
    pts_b = poisson_annulus(r_lo, r_hi, args.marks, min_dist, rng_b)
    amb_b, _ = rotational_ambiguity(pts_b, tol_mm=tol_mm)
    if amb_b > args.max_ambiguity:
        raise SystemExit(f"face B pattern is rotationally ambiguous ({amb_b:.3f})")
    sizes_b = rng_b.uniform(dot_min, dot_max, len(pts_b))
    shapes_b = rng_b.integers(0, 3, len(pts_b))
    angles_b = rng_b.uniform(0, 360, len(pts_b))
    # Mirrored reference angle so through-alignment of the notches matches.
    ref_b = (ref_r * math.cos(math.radians(-ref_deg)),
             ref_r * math.sin(math.radians(-ref_deg)))
    draw_face(b, pts_b, sizes_b, shapes_b, angles_b, r_in, r_out, ref_b, ref_d,
              notch_deg, "B", meta)

    meta["face_b_ambiguity"] = amb_b
    report = args.output_dir / "sidewall_disc.json"
    report.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"\nwrote {a}\nwrote {b}\nwrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
