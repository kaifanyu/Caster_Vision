"""Regression tests for isolated-motion axis calibration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ballrot.integrate import (
    axis_disagreement_deg,
    calibrate_ball_frame,
    estimate_common_axis,
    inter_shell_swivel_axis,
)
from ballrot.rotation import Rx, Rz
from scripts import calibrate_axes


def _increment(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * np.deg2rad(angle_deg)).as_matrix()


def test_common_axis_rejects_low_motion_random_axis_jitter() -> None:
    rng = np.random.default_rng(20260830)
    jitter_axes = rng.normal(size=(12, 3))
    jitter_axes /= np.linalg.norm(jitter_axes, axis=1, keepdims=True)
    increments = [_increment(axis, 0.10) for axis in jitter_axes]
    increments.extend(_increment([1.0, 0.0, 0.0], 1.0) for _ in range(9))

    axis, diagnostics = estimate_common_axis(increments, min_step_deg=0.20)

    assert abs(float(axis @ np.array([1.0, 0.0, 0.0]))) == pytest.approx(
        1.0, abs=1e-12
    )
    assert diagnostics == {
        "raw_step_count": 21,
        "rejected_low_motion_count": 12,
        "retained_step_count": 9,
        "sample_count": 9,
        "min_axis_step_deg": 0.20,
        "axis_spread_median_deg": pytest.approx(0.0, abs=1e-12),
        "axis_spread_p95_deg": pytest.approx(0.0, abs=1e-12),
        "pca_explained_ratio": pytest.approx(1.0, abs=1e-12),
    }


def test_common_axis_keeps_real_off_axis_motion_above_gate() -> None:
    tilted = np.array([np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0)), 0.0])
    increments = [_increment([1.0, 0.0, 0.0], 1.0) for _ in range(20)]
    increments.extend(_increment(tilted, 1.0) for _ in range(5))

    _, diagnostics = estimate_common_axis(increments, min_step_deg=0.20)

    assert diagnostics["raw_step_count"] == 25
    assert diagnostics["rejected_low_motion_count"] == 0
    assert diagnostics["retained_step_count"] == 25
    assert diagnostics["axis_spread_p95_deg"] > 15.0


def test_ball_frame_applies_same_motion_gate_to_both_clips() -> None:
    roll = [_increment([0.0, 1.0, 0.0], 0.10) for _ in range(4)]
    roll.extend(_increment([1.0, 0.0, 0.0], 1.0) for _ in range(10))
    swivel = [_increment([1.0, 0.0, 0.0], 0.10) for _ in range(7)]
    swivel.extend(_increment([0.0, 0.0, 1.0], 1.0) for _ in range(11))

    R_bc, diagnostics = calibrate_ball_frame(
        roll, swivel, min_step_deg=0.20
    )

    np.testing.assert_allclose(R_bc, np.eye(3), atol=1e-12)
    assert diagnostics["roll"]["raw_step_count"] == 14
    assert diagnostics["roll"]["rejected_low_motion_count"] == 4
    assert diagnostics["roll"]["retained_step_count"] == 10
    assert diagnostics["swivel"]["raw_step_count"] == 18
    assert diagnostics["swivel"]["rejected_low_motion_count"] == 7
    assert diagnostics["swivel"]["retained_step_count"] == 11


@pytest.mark.parametrize("value", [0.0, -0.1, np.nan, np.inf, -np.inf])
def test_common_axis_rejects_invalid_motion_gate(value: float) -> None:
    with pytest.raises(ValueError, match="min_step_deg"):
        estimate_common_axis([_increment([1.0, 0.0, 0.0], 1.0)], min_step_deg=value)


def test_axis_cli_defaults_to_point_two_degree_motion_gate() -> None:
    args = calibrate_axes.build_parser().parse_args(
        ["--roll", "roll.mp4", "--swivel", "swivel.mp4"]
    )
    assert args.min_axis_step_deg == pytest.approx(0.20)


@pytest.mark.parametrize("value", ["0", "-0.1", "nan", "inf", "-inf"])
def test_axis_cli_rejects_invalid_motion_gate(value: str) -> None:
    args = calibrate_axes.build_parser().parse_args(
        [
            "--roll",
            "roll.mp4",
            "--swivel",
            "swivel.mp4",
            f"--min-axis-step-deg={value}",
        ]
    )
    with pytest.raises(ValueError, match="--min-axis-step-deg"):
        calibrate_axes._validate_cli_thresholds(args)


def test_cli_reports_filtered_counts_and_does_not_write_if_too_few_remain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    report_path = tmp_path / "axis-report.json"
    config = {
        "camera": {
            "K": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "circle": {"u0": 960.0, "v0": 540.0, "r_px": 380.0},
        "output": {"dir": str(tmp_path / "out")},
    }
    monkeypatch.setattr(calibrate_axes, "load_config", lambda _: (config, config_path))

    sources = {
        "roll.mp4": SimpleNamespace(
            path=Path("roll.mp4"), size=(1920, 1080), fps=30.0, kind="video"
        ),
        "swivel.mp4": SimpleNamespace(
            path=Path("swivel.mp4"), size=(1920, 1080), fps=30.0, kind="video"
        ),
    }
    monkeypatch.setattr(
        calibrate_axes,
        "_open_source",
        lambda path, input_type, max_frames, image_fps: sources[path],
    )

    low_motion = [_increment([0.0, 1.0, 0.0], 0.10) for _ in range(6)]
    results = {
        "roll.mp4": SimpleNamespace(
            top_increments=low_motion
            + [_increment([1.0, 0.0, 0.0], 1.0) for _ in range(5)]
        ),
        "swivel.mp4": SimpleNamespace(
            top_increments=low_motion
            + [_increment([0.0, 0.0, 1.0], 1.0) for _ in range(5)]
        ),
    }
    monkeypatch.setattr(
        calibrate_axes,
        "_run_clip",
        lambda source, **kwargs: results[source.path.name],
    )
    monkeypatch.setattr(calibrate_axes, "_pipeline_summary", lambda result: {})
    writes: list[tuple[Path, dict[str, object]]] = []
    monkeypatch.setattr(
        calibrate_axes, "update_yaml", lambda path, update: writes.append((path, update))
    )

    return_code = calibrate_axes.main(
        [
            "--config",
            str(config_path),
            "--roll",
            "roll.mp4",
            "--swivel",
            "swivel.mp4",
            "--min-axis-step-deg",
            "0.20",
            "--min-valid-steps",
            "8",
            "--report",
            str(report_path),
        ]
    )

    assert return_code == 2
    assert writes == []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["config_updated"] is False
    assert report["thresholds"]["min_axis_step_deg"] == pytest.approx(0.20)
    for name in ("roll", "swivel"):
        diagnostics = report["diagnostics"][name]
        assert diagnostics["raw_step_count"] == 11
        assert diagnostics["rejected_low_motion_count"] == 6
        assert diagnostics["retained_step_count"] == 5
        assert diagnostics["sample_count"] == 5
        assert diagnostics["min_axis_step_deg"] == pytest.approx(0.20)
        assert any(
            failure.startswith(f"{name}: only 5") for failure in report["failures"]
        )


def _ball_to_camera_series(
    alpha: np.ndarray, beta: np.ndarray, R_bc: np.ndarray
) -> np.ndarray:
    return np.array(
        [R_bc @ (Rx(a) @ Rz(b)) @ R_bc.T for a, b in zip(alpha, beta)]
    )


def test_inter_shell_axis_recovers_swivel_without_using_R_bc() -> None:
    """A clip observes its own swivel axis through differential shell motion."""

    frames = 120
    R_bc = Rotation.from_euler("xyz", [17.0, -23.0, 41.0], degrees=True).as_matrix()
    alpha = np.deg2rad(np.linspace(0.0, 55.0, frames))
    beta_top = np.deg2rad(np.linspace(0.0, 70.0, frames))
    beta_bottom = np.deg2rad(np.linspace(0.0, -35.0, frames))
    top = _ball_to_camera_series(alpha, beta_top, R_bc)
    bottom = _ball_to_camera_series(alpha, beta_bottom, R_bc)

    axis, diagnostics = inter_shell_swivel_axis(top, bottom)
    assert axis is not None
    assert diagnostics["sample_count"] > 100
    assert diagnostics["pca_explained_ratio"] > 0.999
    # The recovered axis is the swivel axis in camera coordinates.
    assert axis_disagreement_deg(axis, R_bc[:, 2]) < 1e-6
    # It is genuinely independent of the roll axis, which also moved.
    assert axis_disagreement_deg(axis, R_bc[:, 0]) > 80.0


def test_inter_shell_axis_reports_a_reoriented_ball() -> None:
    """A ball re-seated after calibration shows up as an axis mismatch."""

    frames = 120
    calibrated = Rotation.from_euler("xyz", [17.0, -23.0, 41.0], degrees=True).as_matrix()
    # The same motion, recorded after the assembly was rotated 30 deg about the
    # camera x-axis. Per-frame tracking is unaffected; the frame is not.
    reseated = Rotation.from_euler("x", 30.0, degrees=True).as_matrix() @ calibrated
    alpha = np.deg2rad(np.linspace(0.0, 55.0, frames))
    top = _ball_to_camera_series(alpha, np.deg2rad(np.linspace(0.0, 70.0, frames)), reseated)
    bottom = _ball_to_camera_series(alpha, np.deg2rad(np.linspace(0.0, -35.0, frames)), reseated)

    axis, _ = inter_shell_swivel_axis(top, bottom)
    assert axis is not None
    # The clip reports its own true axis, not the stale calibrated one, so the
    # mismatch equals the real frame error the runner needs to warn about.
    assert axis_disagreement_deg(axis, reseated[:, 2]) < 1e-6
    assert axis_disagreement_deg(axis, calibrated[:, 2]) == pytest.approx(
        axis_disagreement_deg(reseated[:, 2], calibrated[:, 2]), abs=1e-6
    )
    assert axis_disagreement_deg(axis, calibrated[:, 2]) > 25.0


def test_inter_shell_axis_is_none_without_differential_motion() -> None:
    """Locked shells cannot observe a swivel axis; that must not be faked."""

    frames = 60
    R_bc = np.eye(3)
    alpha = np.deg2rad(np.linspace(0.0, 40.0, frames))
    beta = np.deg2rad(np.linspace(0.0, 25.0, frames))
    rigid = _ball_to_camera_series(alpha, beta, R_bc)

    axis, diagnostics = inter_shell_swivel_axis(rigid, rigid)
    assert axis is None
    assert diagnostics["sample_count"] == 0
