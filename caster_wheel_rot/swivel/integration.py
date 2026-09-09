"""Integrate measured increments without claiming unseen accumulated turns."""
from __future__ import annotations

import numpy as np


def integrate_roll(increments, valid, references, *, max_gap_frames=5,
                   max_resync_error_deg=45., max_step_deg=90.):
    """Return accumulated angle, completeness, segments and reference diagnostics.

    Numeric phi retains the sum of observed increments across gaps for plotting;
    phi_valid is true only when the angle is still connected to frame zero.
    A short reference bridge can restore that connection, provided the configured
    speed bound rules out half a turn over the entire reference separation.
    Future increments are integrated after any correction, never before it.
    """
    increments = np.asarray(increments, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if increments.ndim != 1 or valid.shape != increments.shape:
        raise ValueError("increments and valid must share shape (N,)")
    if np.any(valid & ~np.isfinite(increments)):
        raise ValueError("valid increments must be finite")
    n = len(increments) + 1
    if len(references) != n:
        raise ValueError("references must have one observation per frame")
    if max_gap_frames < 0 or not 0 < max_step_deg < 180 or not 0 < max_resync_error_deg < 180:
        raise ValueError("invalid reference gap/speed/resync limits")
    phi = np.zeros(n)
    complete = np.zeros(n, dtype=bool)
    complete[0] = True
    segments = np.zeros(n, dtype=int)
    errors = np.full(n, np.nan)
    corrected = np.zeros(n, dtype=bool)
    anchor = 0 if references[0].valid else None
    anchor_phi = 0.
    limit = np.deg2rad(max_resync_error_deg)
    for i in range(1, n):
        phi[i] = phi[i-1] + (increments[i-1] if valid[i-1] else 0.)
        complete[i] = complete[i-1] and valid[i-1]
        segments[i] = segments[i-1] + int(not valid[i-1] and (i == 1 or valid[i-2]))
        reference = references[i]
        if not reference.valid:
            continue
        if anchor is not None:
            step = i - anchor
            delta = (reference.phase_wrapped - references[anchor].phase_wrapped + np.pi) % (2*np.pi) - np.pi
            target = anchor_phi + delta
            errors[i] = (target - phi[i] + np.pi) % (2*np.pi) - np.pi
            short = step - 1 <= max_gap_frames and step * max_step_deg < 180.
            if not complete[i] and complete[anchor] and short and abs(target - phi[i]) <= limit:
                phi[i] = target
                complete[i] = True
                corrected[i] = True
        # Never establish an absolute reference origin from an incomplete angle.
        if complete[i]:
            anchor, anchor_phi = i, float(phi[i])
    return phi, complete, segments, errors, corrected
