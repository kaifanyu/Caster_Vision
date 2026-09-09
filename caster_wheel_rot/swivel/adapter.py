"""Convert tracked swivel angles into the common :class:`CasterFrame` seam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from common.caster_frame import CasterFrame
from common.rotation import Rz


@dataclass(frozen=True)
class SwivelSeries:
    """Absolute swivel/roll tracks at image-frame timestamps."""

    t: np.ndarray
    phi: np.ndarray
    psi: np.ndarray
    phi_valid: np.ndarray
    psi_valid: np.ndarray
    quality: Sequence[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        arrays = [np.asarray(getattr(self, name)) for name in (
            "t", "phi", "psi", "phi_valid", "psi_valid"
        )]
        length = len(arrays[0])
        if length < 2 or any(array.shape != (length,) for array in arrays):
            raise ValueError("SwivelSeries arrays must share shape (N,), N >= 2")
        t, phi, psi = (np.asarray(array, dtype=float) for array in arrays[:3])
        if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
            raise ValueError("SwivelSeries timestamps must be finite and increasing")
        phi_valid = np.asarray(arrays[3], dtype=bool)
        psi_valid = np.asarray(arrays[4], dtype=bool)
        if np.any(phi_valid & ~np.isfinite(phi)) or np.any(psi_valid & ~np.isfinite(psi)):
            raise ValueError("valid angle samples must be finite")
        if self.quality is not None and len(self.quality) not in {length, length - 1}:
            raise ValueError("quality must have N or N-1 entries")
        for name, value in zip(("t", "phi", "psi", "phi_valid", "psi_valid"), (t, phi, psi, phi_valid, psi_valid)):
            object.__setattr__(self, name, value)


def _horizontal_unit(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(3).copy()
    vector[2] = 0.0
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{name} must have a nonzero horizontal component")
    return vector / norm


def adapt_swivel_series(
    series: SwivelSeries,
    *,
    r_eff: float,
    axle0_car: np.ndarray = np.array([0.0, -1.0, 0.0]),
    roll_direction_sign: float = 1.0,
    heading_direction_sign: float = 1.0,
    swivel_axis_car: np.ndarray | None = None,
    hub_offset0_car: np.ndarray | None = None,
) -> list[CasterFrame]:
    """Emit one common record per frame interval.

    ``roll_direction_sign`` is fixed by the known-distance forward calibration.
    ``heading_direction_sign`` aligns the canonical wheel direction so a
    straight forward run produces ``+x`` under the specification's
    ``z cross roll_axis`` heading convention.
    """

    if not np.isfinite(r_eff) or r_eff <= 0.0:
        raise ValueError("r_eff must be positive and finite")
    for value, name in ((roll_direction_sign, "roll_direction_sign"), (heading_direction_sign, "heading_direction_sign")):
        if float(value) not in {-1.0, 1.0}:
            raise ValueError(f"{name} must be +1 or -1")
    axle0 = _horizontal_unit(axle0_car, "axle0_car")
    swivel_axis = None if swivel_axis_car is None else np.asarray(swivel_axis_car, dtype=float).reshape(3)
    hub_offset = None if hub_offset0_car is None else np.asarray(hub_offset0_car, dtype=float).reshape(3)
    if (swivel_axis is None) != (hub_offset is None):
        raise ValueError("swivel_axis_car and hub_offset0_car must be supplied together")

    frames: list[CasterFrame] = []
    z = np.array([0.0, 0.0, 1.0])
    for index, dt in enumerate(np.diff(series.t)):
        quality: dict[str, Any] = {}
        if series.quality is not None:
            quality = dict(series.quality[min(index, len(series.quality) - 1)])
        heading_valid = bool(
            quality.get(
                "heading_valid",
                series.psi_valid[index] and series.psi_valid[index + 1],
            )
        )
        roll_valid = bool(
            quality.get(
                "roll_valid",
                series.phi_valid[index] and series.phi_valid[index + 1],
            )
        )
        psi_mid = (
            0.5 * (series.psi[index] + series.psi[index + 1])
            if heading_valid
            else (series.psi[index] if series.psi_valid[index] else series.psi[index + 1])
        )
        if not np.isfinite(psi_mid):
            psi_mid = 0.0
            heading_valid = False
        psi_dot = (
            float((series.psi[index + 1] - series.psi[index]) / dt)
            if heading_valid
            else 0.0
        )
        phi_dot_raw = (
            float(quality.get("delta_phi_rad", series.phi[index + 1] - series.phi[index]) / dt)
            if roll_valid
            else 0.0
        )
        signed_rate = float(roll_direction_sign) * phi_dot_raw
        canonical_axis = _horizontal_unit(Rz(float(psi_mid)) @ axle0, "rotated axle")
        canonical_axis *= float(heading_direction_sign)
        direction = 1.0 if abs(signed_rate) <= 1e-12 else float(np.sign(signed_rate))
        roll_axis = direction * canonical_axis
        rolling_heading = np.cross(z, roll_axis)[:2]
        rolling_heading /= np.linalg.norm(rolling_heading)

        confidence = float(quality.get("confidence", 1.0 if roll_valid else 0.0))
        raw: dict[str, Any] = {
            "phi": float(0.5 * (series.phi[index] + series.phi[index + 1])) if series.phi_valid[index] and series.phi_valid[index+1] else None,
            "psi": float(psi_mid),
            "phi_dot": phi_dot_raw,
            "psi_dot": psi_dot,
            "roll_valid": roll_valid,
            "heading_valid": heading_valid,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "rolling_heading_car": rolling_heading.tolist(),
            "steering_heading_car": np.cross(z, canonical_axis)[:2].tolist(),
            **quality,
        }
        if swivel_axis is not None and hub_offset is not None:
            offset = swivel_axis + Rz(float(psi_mid)) @ hub_offset
            relative = psi_dot * np.cross(z, Rz(float(psi_mid)) @ hub_offset)
            raw["contact_offset_car"] = offset[:2].tolist()
            raw["contact_velocity_rel_car"] = relative[:2].tolist()
        frames.append(
            CasterFrame(
                t=float(0.5 * (series.t[index] + series.t[index + 1])),
                roll_axis_car=roll_axis,
                omega_roll=abs(signed_rate),
                omega_spin=psi_dot,
                r_eff=float(r_eff),
                raw=raw,
            )
        )
    return frames


__all__ = ["SwivelSeries", "adapt_swivel_series"]
