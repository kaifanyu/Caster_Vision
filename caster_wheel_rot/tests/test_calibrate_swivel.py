"""Headless formula and CLI gates for swivel calibration."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from common.caster_frame import CasterFrame, save_caster_frames
from scripts import calibrate_swivel as calibration
from synthetic.generate_swivel import (
    RenderConfig,
    SwivelTrajectory,
    render_sequence,
)


def _base_config() -> dict:
    return {
        "assumptions": {
            "camera_fixed_to_chassis": True,
            "car_and_caster_clocks_synchronized": False,
        },
        "input": {"swivel_clip": "unused.mp4", "fps_override": None},
        "camera": {
            "K": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
            "T_car_from_cam": np.eye(4).tolist(),
        },
        "swivel": {
            "marker": {
                "dictionary": "DICT_4X4_50",
                "id": 0,
                "size_m": 0.04,
                "psi0_rad": None,
                "max_reprojection_error_px": 2.0,
            },
            "geometry": {
                "swivel_axis_car": [0.18, 0.0, 0.16],
                "hub_offset0_car": [-0.055, 0.0, -0.08],
                "axle0_car": [0.0, -1.0, 0.0],
                "wheel_radius_m": 0.075,
                "wheel_width_m": 0.034,
            },
            "r_eff_m": None,
            "roll_direction_sign": 1.0,
            "heading_direction_sign": 1.0,
        },
        "car_track": {"time_offset_s": 0.0},
        "sync": {
            "method": "common_event",
            "caster_event_time_s": None,
            "car_event_time_s": None,
            "max_offset_s": 1.0,
        },
        "unrelated_value": {"must_survive": 17},
    }


def _write_config(path: Path, value: dict | None = None) -> Path:
    path.write_text(
        yaml.safe_dump(value or _base_config(), sort_keys=False), encoding="utf-8"
    )
    return path


def _read_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_calibration_formulas_and_transform_validation() -> None:
    assert calibration.effective_radius(2.0, 8.0 * np.pi) == pytest.approx(
        1.0 / (4.0 * np.pi)
    )
    assert calibration.event_clock_offset(4.25, 4.60) == pytest.approx(0.35)
    assert calibration.roll_sign_from_delta(2.0) == 1.0
    assert calibration.roll_sign_from_delta(-2.0) == -1.0

    valid = np.eye(4)
    valid[:3, 3] = [0.1, -0.2, 0.3]
    np.testing.assert_allclose(calibration.validate_T_car_from_cam(valid), valid)
    reflected = valid.copy()
    reflected[0, 0] = -1.0
    with pytest.raises(ValueError, match="proper"):
        calibration.validate_T_car_from_cam(reflected)
    with pytest.raises(ValueError, match="too little"):
        calibration.roll_sign_from_delta(1e-9)


def test_solve_extrinsics_recovers_known_car_camera_transform() -> None:
    K = np.array([[910.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]])
    car_points = np.array(
        [
            [-0.3, -0.2, 0.0],
            [0.3, -0.2, 0.0],
            [0.3, 0.2, 0.0],
            [-0.3, 0.2, 0.0],
            [-0.2, -0.15, 0.18],
            [0.25, -0.12, 0.22],
            [0.22, 0.16, 0.15],
            [-0.24, 0.14, 0.20],
        ],
        dtype=float,
    )
    R_camera_from_car = Rotation.from_euler("xyz", [0.12, -0.18, 0.09]).as_matrix()
    t_camera_from_car = np.array([0.04, -0.03, 1.35])
    rvec = Rotation.from_matrix(R_camera_from_car).as_rotvec()
    pixels, _ = cv2.projectPoints(
        car_points, rvec, t_camera_from_car, K, np.zeros(5)
    )
    result = calibration.solve_extrinsics(
        car_points, pixels.reshape(-1, 2), K, np.zeros(5)
    )
    expected = np.eye(4)
    expected[:3, :3] = R_camera_from_car.T
    expected[:3, 3] = -R_camera_from_car.T @ t_camera_from_car
    np.testing.assert_allclose(result["T_car_from_cam"], expected, atol=1e-7)
    assert result["rms_reprojection_error_px"] < 1e-7
    assert result["max_reprojection_error_px"] < 1e-7


def test_cli_updates_manual_extrinsics_radius_sign_and_event_sync_atomically(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path / "config.yaml")
    transform = np.eye(4)
    transform[:3, 3] = [0.4, -0.2, 0.8]
    extrinsic_report = tmp_path / "extrinsics.json"
    assert calibration.main(
        [
            "--config",
            str(config_path),
            "extrinsics",
            "--T-car-from-cam",
            json.dumps(transform.tolist()),
            "--report",
            str(extrinsic_report),
        ]
    ) == 0
    assert calibration.main(
        [
            "--config",
            str(config_path),
            "roll-sign",
            "--phi-start",
            "0.4",
            "--phi-end",
            "-9.6",
        ]
    ) == 0
    assert calibration.main(
        [
            "--config",
            str(config_path),
            "radius",
            "--distance-m",
            "2.0",
            "--revolutions",
            "4",
        ]
    ) == 0
    assert calibration.main(
        [
            "--config",
            str(config_path),
            "sync-event",
            "--caster-event-s",
            "3.10",
            "--car-event-s",
            "3.50",
        ]
    ) == 0

    updated = _read_config(config_path)
    np.testing.assert_allclose(updated["camera"]["T_car_from_cam"], transform)
    assert updated["swivel"]["roll_direction_sign"] == -1.0
    assert updated["swivel"]["r_eff_m"] == pytest.approx(2.0 / (8.0 * np.pi))
    assert updated["car_track"]["time_offset_s"] == pytest.approx(0.4)
    assert updated["assumptions"]["car_and_caster_clocks_synchronized"] is True
    assert updated["unrelated_value"]["must_survive"] == 17
    assert extrinsic_report.is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_geometry_cli_normalizes_axle_and_preserves_other_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml")
    report_path = tmp_path / "geometry.json"
    assert calibration.main(
        [
            "--config",
            str(config_path),
            "geometry",
            "--swivel-axis-car",
            "0.31",
            "0.02",
            "0.17",
            "--hub-offset0-car",
            "-0.061",
            "0",
            "-0.083",
            "--axle0-car",
            "0",
            "-2",
            "0",
            "--wheel-radius-m",
            "0.081",
            "--wheel-width-m",
            "0.038",
            "--report",
            str(report_path),
        ]
    ) == 0
    updated = _read_config(config_path)
    geometry = updated["swivel"]["geometry"]
    assert geometry["swivel_axis_car"] == [0.31, 0.02, 0.17]
    assert geometry["hub_offset0_car"] == [-0.061, 0.0, -0.083]
    assert geometry["axle0_car"] == [0.0, -1.0, 0.0]
    assert geometry["wheel_radius_m"] == pytest.approx(0.081)
    assert geometry["wheel_width_m"] == pytest.approx(0.038)
    assert updated["unrelated_value"]["must_survive"] == 17
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["signed_trail_m"] == pytest.approx(0.061)


def test_report_cannot_overwrite_swivel_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_config(tmp_path / "config.yaml")
    original = config_path.read_bytes()
    with pytest.raises(SystemExit) as raised:
        calibration.main(
            [
                "--config",
                str(config_path),
                "sync-event",
                "--caster-event-s",
                "1.0",
                "--car-event-s",
                "1.1",
                "--report",
                str(config_path),
            ]
        )
    assert raised.value.code == 2
    assert "must not overwrite" in capsys.readouterr().err
    assert config_path.read_bytes() == original


def test_roll_sign_cli_reads_known_forward_casterframe_json(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml")
    frames = [
        CasterFrame(
            t=float(index) * 0.1,
            roll_axis_car=np.array([0.0, 1.0, 0.0]),
            omega_roll=2.0,
            omega_spin=0.0,
            r_eff=0.075,
            raw={"phi_dot": -2.0 + 0.02 * index, "roll_valid": True},
        )
        for index in range(8)
    ]
    input_path = save_caster_frames(
        tmp_path / "caster_frames.json", frames, metadata={"maneuver": "forward"}
    )
    report_path = tmp_path / "roll_sign.json"
    assert calibration.main(
        [
            "--config",
            str(config_path),
            "roll-sign",
            "--caster-frames",
            str(input_path),
            "--report",
            str(report_path),
        ]
    ) == 0
    updated = _read_config(config_path)
    assert updated["swivel"]["roll_direction_sign"] == -1.0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["source"] == "raw.phi_dot"
    assert report["sample_count"] == len(frames)


def _write_signal(path: Path, t: np.ndarray, signal: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "flash"])
        writer.writerows(zip(t, signal))


def test_sync_signals_cli_recovers_offset_and_updates_quality(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml")
    t = np.arange(0.0, 6.0, 0.02)
    caster = np.exp(-0.5 * ((t - 2.0) / 0.08) ** 2) + 0.35 * np.exp(
        -0.5 * ((t - 4.1) / 0.14) ** 2
    )
    car = np.exp(-0.5 * ((t - 2.24) / 0.08) ** 2) + 0.35 * np.exp(
        -0.5 * ((t - 4.34) / 0.14) ** 2
    )
    caster_path, car_path = tmp_path / "caster.csv", tmp_path / "car.csv"
    _write_signal(caster_path, t, caster)
    _write_signal(car_path, t, car)
    assert calibration.main(
        [
            "--config",
            str(config_path),
            "sync-signals",
            "--caster-csv",
            str(caster_path),
            "--car-csv",
            str(car_path),
            "--caster-time-column",
            "timestamp",
            "--caster-signal-column",
            "flash",
            "--car-time-column",
            "timestamp",
            "--car-signal-column",
            "flash",
            "--max-offset-s",
            "0.5",
        ]
    ) == 0
    updated = _read_config(config_path)
    assert updated["car_track"]["time_offset_s"] == pytest.approx(0.24, abs=0.021)
    assert updated["sync"]["method"] == "cross_correlation"
    assert updated["sync"]["last_peak_correlation"] > 0.95


def test_psi_zero_cli_uses_rendered_tag_and_calibrated_extrinsics(
    tmp_path: Path,
) -> None:
    render_config = RenderConfig()
    count, fps, psi_zero = 12, 60.0, 0.31
    t = np.arange(count, dtype=float) / fps
    trajectory = SwivelTrajectory(
        t=t,
        phi=np.zeros(count),
        psi=np.full(count, psi_zero),
        car_x=np.zeros(count),
        car_y=np.zeros(count),
        car_theta=np.zeros(count),
    )
    rendered = render_sequence(
        trajectory,
        config=render_config,
        output_dir=tmp_path / "psi_clip",
    )
    config = _base_config()
    config["camera"]["K"] = rendered.truth["K"]
    config["camera"]["T_car_from_cam"] = rendered.truth["T_car_from_cam"]
    config["swivel"]["marker"] = {
        **config["swivel"]["marker"],
        **rendered.truth["marker"],
        "psi0_rad": None,
        "max_reprojection_error_px": 2.0,
    }
    config["swivel"]["geometry"].update(rendered.truth["geometry"])
    config_path = _write_config(tmp_path / "config.yaml", config)
    assert calibration.main(
        [
            "--config",
            str(config_path),
            "psi-zero",
            "--clip",
            str(tmp_path / "psi_clip" / "frames"),
            "--input-type",
            "images",
            "--fps",
            str(fps),
            "--min-detections",
            "5",
        ]
    ) == 0
    updated = _read_config(config_path)
    assert updated["swivel"]["marker"]["psi0_rad"] == pytest.approx(
        psi_zero, abs=np.deg2rad(0.8)
    )


def test_intrinsics_cli_updates_config_through_atomic_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path / "config.yaml")
    fake_image = tmp_path / "checker.png"
    fake_image.write_bytes(b"input path is expanded before the mocked solver")
    expected_K = np.array(
        [[701.0, 0.0, 319.5], [0.0, 699.0, 239.5], [0.0, 0.0, 1.0]]
    )

    def fake_calibration(*args, **kwargs):
        return {
            "K": expected_K,
            "dist": np.zeros(5),
            "rms_reprojection_error_px": 0.2,
            "per_view_error_px": [0.2] * 8,
            "image_size": [640, 480],
            "accepted": [str(fake_image)] * 8,
            "rejected": [],
        }

    monkeypatch.setattr(calibration, "calibrate_intrinsics", fake_calibration)
    assert calibration.main(
        [
            "--config",
            str(config_path),
            "intrinsics",
            "--images",
            str(fake_image),
            "--board-cols",
            "7",
            "--board-rows",
            "5",
        ]
    ) == 0
    updated = _read_config(config_path)
    np.testing.assert_allclose(updated["camera"]["K"], expected_K)
    np.testing.assert_allclose(updated["camera"]["dist"], np.zeros(5))
    assert updated["unrelated_value"]["must_survive"] == 17
