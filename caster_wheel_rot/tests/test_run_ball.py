"""Tests for the unified real-ball command-line entry point."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from common.caster_frame import load_caster_frames
from scripts import run_ball


def _pipeline_result() -> SimpleNamespace:
    identity = np.eye(3)
    motion = SimpleNamespace(
        alpha=np.zeros(3),
        beta_top=np.zeros(3),
        beta_bottom=np.zeros(3),
        alpha_top=np.zeros(3),
        alpha_bottom=np.zeros(3),
        gamma_top=np.zeros(3),
        gamma_bottom=np.zeros(3),
        valid_top=np.ones(3, dtype=bool),
        valid_bottom=np.ones(3, dtype=bool),
    )
    return SimpleNamespace(
        timestamps_s=np.array([0.0, 0.05, 0.10]),
        top_increments=[identity.copy(), identity.copy()],
        bottom_increments=[identity.copy(), identity.copy()],
        top_step_valid=np.array([True, True, True]),
        bottom_step_valid=np.array([True, True, True]),
        top_absolute=np.repeat(identity[None], 3, axis=0),
        bottom_absolute=np.repeat(identity[None], 3, axis=0),
        motion=motion,
        qualities=[],
        circle=(320.0, 240.0, 120.0),
        frame_count=3,
        overlay_path=None,
    )


def test_fixed_camera_and_stationary_center_assumptions_are_enforced() -> None:
    run_ball._check_assumptions(
        {"assumptions": {"camera_fixed_to_chassis": True}}
    )
    with pytest.raises(SystemExit, match="camera_fixed_to_chassis"):
        run_ball._check_assumptions(
            {"assumptions": {"camera_fixed_to_chassis": False}}
        )
    with pytest.raises(SystemExit, match="ball_center_stationary_in_image"):
        run_ball._check_assumptions(
            {
                "assumptions": {
                    "camera_fixed_to_chassis": True,
                    "ball_center_stationary_in_image": False,
                }
            }
        )


def test_calibration_is_strict_unless_tracking_only() -> None:
    with pytest.raises(SystemExit, match="ball.R_bc"):
        run_ball._resolve_ball_calibration({}, tracking_only=False)
    with pytest.raises(SystemExit, match="ball.r_eff_m"):
        run_ball._resolve_ball_calibration(
            {"R_bc": np.eye(3).tolist()}, tracking_only=False
        )
    with pytest.raises(SystemExit, match="ball.R_ball_to_car"):
        run_ball._resolve_ball_calibration(
            {"R_bc": np.eye(3).tolist(), "r_eff_m": 0.08},
            tracking_only=False,
        )
    with pytest.raises(SystemExit, match="direction_sign_calibrated"):
        run_ball._resolve_ball_calibration(
            {
                "R_bc": np.eye(3).tolist(),
                "R_ball_to_car": np.eye(3).tolist(),
                "r_eff_m": 0.08,
            },
            tracking_only=False,
        )

    R_bc, R_ball_to_car, radius, flags = run_ball._resolve_ball_calibration(
        {}, tracking_only=True
    )
    np.testing.assert_allclose(R_bc, np.eye(3))
    np.testing.assert_allclose(R_ball_to_car, np.eye(3))
    assert radius == 1.0
    assert flags == {
        "R_bc_calibrated": False,
        "R_ball_to_car_calibrated": False,
        "direction_sign_calibrated": False,
        "r_eff_calibrated": False,
    }


def test_increment_payload_and_csv_preserve_matrices_and_validity(
    tmp_path: Path,
) -> None:
    result = _pipeline_result()
    result.top_increments[1] = None
    result.top_step_valid[2] = False
    # Preserve a raw matrix even when the explicit geometry/solve mask rejects it.
    result.bottom_step_valid[1] = False

    payload = run_ball._increment_payload(result)

    assert len(payload["top_camera"]) == 2
    np.testing.assert_allclose(payload["top_camera"][0], np.eye(3))
    assert payload["top_camera"][1] is None
    assert payload["top_valid"] == [True, False]
    assert payload["bottom_valid"] == [False, True]
    np.testing.assert_allclose(payload["time_mid_s"], [0.025, 0.075])

    path = run_ball._write_increment_csv(result, tmp_path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["bottom_valid"] == "False"
    assert float(rows[0]["bottom_r00"]) == 1.0
    assert rows[1]["top_valid"] == "False"
    assert rows[1]["top_r00"] == ""


def test_tracking_only_main_writes_common_and_lossless_raw_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
assumptions:
  camera_fixed_to_chassis: true
input:
  ball_clip: data/ball.mp4
  type: auto
  max_frames: null
  fps_override: null
camera:
  K: null
  dist: [0, 0, 0, 0, 0]
  fov_deg: 60
ball:
  R_bc: null
  r_eff_m: null
  R_ball_to_car: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  roll_direction_sign: -1
  segment: {}
track: {}
estimate: {}
car_track:
  time_offset_s: 0
output:
  dir: out
  save_overlay_video: true
""".strip(),
        encoding="utf-8",
    )
    fake_source = SimpleNamespace(
        frame_count=3,
        fps=20.0,
        size=(640, 480),
    )
    result = _pipeline_result()
    monkeypatch.setattr(run_ball, "open_frame_source", lambda *args, **kwargs: fake_source)
    monkeypatch.setattr(
        run_ball,
        "camera_matrix",
        lambda camera, shape: (np.eye(3), False),
    )
    monkeypatch.setattr(run_ball, "distortion_coefficients", lambda camera: np.zeros(5))
    monkeypatch.setattr(run_ball, "run_pipeline", lambda *args, **kwargs: result)
    monkeypatch.setattr(run_ball, "summarize_quality", lambda *args, **kwargs: {"ok": True})

    def fake_diagnostics(output, *args, **kwargs):
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        csv_path = destination / "results.csv"
        csv_path.write_text("frame,time_s\n0,0\n", encoding="utf-8")
        return {"csv": csv_path}

    monkeypatch.setattr(run_ball, "write_run_outputs", fake_diagnostics)
    output = tmp_path / "ball-output"

    assert (
        run_ball.main(
            [
                "--config",
                str(config),
                "--output",
                str(output),
                "--tracking-only",
                "--no-overlay",
            ]
        )
        == 0
    )

    frames, metadata = load_caster_frames(output / "caster_frames.json")
    assert len(frames) == 2
    assert metadata["trusted_measurement"] is False
    assert metadata["R_bc_calibrated"] is False
    assert (output / "results.csv").is_file()
    assert (output / "increments.csv").is_file()
    payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "ball-pipeline-results-v1"
    assert len(payload["increments"]["top_camera"]) == 2
    assert payload["increments"]["top_camera"][0] == np.eye(3).tolist()
    assert payload["increments"]["top_valid"] == [True, True]
    assert payload["metadata"]["tracking_only"] is True
