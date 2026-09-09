"""The common per-frame seam used by both caster implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class CasterFrame:
    """One device-independent sample, expressed in the chassis/car frame.

    ``raw['roll_valid']`` is the shared dropout flag.  Keeping it in ``raw``
    preserves the exact interface from the specification while exposing the
    :attr:`roll_valid` property to metrics code.
    """

    t: float
    roll_axis_car: np.ndarray
    omega_roll: float
    omega_spin: float
    r_eff: float
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        timestamp = float(self.t)
        axis = np.asarray(self.roll_axis_car, dtype=np.float64).reshape(-1)
        omega_roll = float(self.omega_roll)
        omega_spin = float(self.omega_spin)
        radius = float(self.r_eff)
        if not np.isfinite(timestamp):
            raise ValueError("t must be finite")
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("roll_axis_car must be a finite (3,) vector")
        norm = float(np.linalg.norm(axis))
        if not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError("roll_axis_car must be unit length")
        if abs(float(axis[2])) > 1e-6:
            raise ValueError("roll_axis_car must be horizontal in the car frame")
        if not np.isfinite(omega_roll) or omega_roll < 0.0:
            raise ValueError("omega_roll must be finite and non-negative")
        if not np.isfinite(omega_spin):
            raise ValueError("omega_spin must be finite")
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("r_eff must be finite and positive")
        if not isinstance(self.raw, dict):
            raise TypeError("raw must be a dict")
        object.__setattr__(self, "t", timestamp)
        object.__setattr__(self, "roll_axis_car", axis / norm)
        object.__setattr__(self, "omega_roll", omega_roll)
        object.__setattr__(self, "omega_spin", omega_spin)
        object.__setattr__(self, "r_eff", radius)

    @property
    def roll_valid(self) -> bool:
        return bool(self.raw.get("roll_valid", True))

    @property
    def heading_valid(self) -> bool:
        return bool(self.raw.get("heading_valid", True))

    @property
    def confidence(self) -> float:
        value = float(self.raw.get("confidence", 1.0))
        return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "roll_axis_car": self.roll_axis_car.tolist(),
            "omega_roll": self.omega_roll,
            "omega_spin": self.omega_spin,
            "r_eff": self.r_eff,
            "raw": _json_value(self.raw),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CasterFrame":
        return cls(
            t=float(value["t"]),
            roll_axis_car=np.asarray(value["roll_axis_car"], dtype=float),
            omega_roll=float(value["omega_roll"]),
            omega_spin=float(value["omega_spin"]),
            r_eff=float(value["r_eff"]),
            raw=dict(value.get("raw", {})),
        )


def save_caster_frames(
    path: str | Path,
    frames: Iterable[CasterFrame],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "caster-frame-v1",
        "metadata": _json_value(dict(metadata or {})),
        "frames": [frame.to_dict() for frame in frames],
    }
    destination.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    return destination


def load_caster_frames(path: str | Path) -> tuple[list[CasterFrame], dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [CasterFrame.from_dict(item) for item in payload], {}
    if payload.get("schema") != "caster-frame-v1":
        raise ValueError(f"unsupported CasterFrame schema in {source}")
    return [CasterFrame.from_dict(item) for item in payload["frames"]], dict(
        payload.get("metadata", {})
    )
