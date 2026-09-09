"""Angular measurements independent of effective radius and vehicle motion.

Rates are signed interval averages in the geometry's phi/psi conventions.
Missing measurements remain null; corrected accumulated phase is never
differentiated into a fictitious angular velocity.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def angular_intervals(payload):
    t = np.asarray(payload["timestamps_s"], float)
    phi = np.asarray(payload["phi_rad"], float)
    psi = np.asarray(payload["psi_rad"], float)
    pv = np.asarray(payload["phi_valid"], bool)
    sv = np.asarray(payload["psi_valid"], bool)
    quality = payload["interval_quality"]
    corrected = np.asarray(payload.get("reference_corrected", np.zeros(len(t))), bool)
    if t.ndim != 1 or any(a.shape != t.shape for a in (phi, psi, pv, sv, corrected)):
        raise ValueError("angular series must have equal one-dimensional shapes")
    if len(quality) != max(0, len(t)-1):
        raise ValueError("one quality record is required per frame interval")
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    rows = []
    for i, q in enumerate(quality):
        dt = float(t[i+1]-t[i])
        # Current runs store the actual fitted increment separately from phi.
        # Legacy fallback is safe only with complete, uncorrected endpoints.
        dphi = q.get("delta_phi_rad")
        if "delta_phi_rad" not in q and pv[i] and pv[i+1] and not corrected[i+1]:
            dphi = phi[i+1]-phi[i]
        roll_ok = bool(q.get("roll_valid", pv[i] and pv[i+1]))
        roll_ok &= dphi is not None and bool(np.isfinite(dphi))
        swivel_ok = bool(q.get("heading_valid", sv[i] and sv[i+1]))
        swivel_ok &= bool(sv[i] and sv[i+1] and np.isfinite(psi[i:i+2]).all())
        dphi = float(dphi) if roll_ok else None
        dpsi = float(psi[i+1]-psi[i]) if swivel_ok else None
        row = dict(frame_start=i, frame_end=i+1, t_start_s=float(t[i]),
                   t_end_s=float(t[i+1]), t_mid_s=float((t[i]+t[i+1])/2), dt_s=dt,
                   roll_valid=roll_ok, swivel_valid=swivel_ok,
                   accumulated_roll_complete=bool(pv[i+1]),
                   reference_corrected=bool(corrected[i+1]),
                   failure_reason=q.get("failure_reason"))
        for name, delta in (("phi", dphi), ("psi", dpsi)):
            row[f"delta_{name}_rad"] = delta
            row[f"delta_{name}_deg"] = None if delta is None else float(np.rad2deg(delta))
            row[f"{name}_dot_rad_s"] = None if delta is None else delta/dt
            row[f"{name}_dot_deg_s"] = None if delta is None else float(np.rad2deg(delta)/dt)
        rows.append(row)
    return rows


def write_angular_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def replay_phases(rows, psi_valid, face_signs):
    """Local continuous roll phase, re-anchored after gaps or face changes.

    Re-anchor frames have phase zero; unavailable headings have NaN phase.
    The output never asserts an absolute wheel mark or hidden turn count.
    """
    n = len(psi_valid)
    if len(face_signs) != n or len(rows) != max(0, n-1):
        raise ValueError("replay inputs have inconsistent lengths")
    phase = np.full(n, np.nan)
    segments = np.full(n, -1, dtype=int)
    segment = -1
    for i in range(n):
        if not psi_valid[i]:
            continue
        connected = (i > 0 and np.isfinite(phase[i-1]) and
                     rows[i-1]["roll_valid"] and rows[i-1]["swivel_valid"] and
                     face_signs[i] == face_signs[i-1])
        if connected:
            phase[i] = phase[i-1] + rows[i-1]["delta_phi_rad"]
        else:
            segment += 1
            phase[i] = 0.
        segments[i] = segment
    return phase, segments
