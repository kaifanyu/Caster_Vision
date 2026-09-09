from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from common.rotation import Rx, Rz
from common.caster_frame import CasterFrame, save_caster_frames
from scripts import (
    calibrate_ball_axes,
    calibrate_ball_circle,
    calibrate_ball_direction,
    calibrate_ball_radius,
    inspect_ball_hsv,
)


def _base_config() -> dict:
    return {
        "input": {"ball_clip": "frame.png", "fps_override": 60.0},
        "camera": {
            "K": [[100.0, 0.0, 20.0], [0.0, 100.0, 20.0], [0.0, 0.0, 1.0]],
            "dist": [0, 0, 0, 0, 0],
            "T_car_from_cam": np.eye(4).tolist(),
        },
        "ball": {
            "R_bc": None,
            "R_ball_to_car": None,
            "segment": {"mode": "color", "sentinel": "preserve-me"},
        },
        "track": {},
        "estimate": {},
        "output": {"dir": "out"},
    }


def test_three_point_circle_writes_unified_ball_section(tmp_path: Path) -> None:
    config = _base_config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "frame.png"), frame)
    preview = tmp_path / "preview.png"

    result = calibrate_ball_circle.main(
        [
            "--config",
            str(config_path),
            "--point",
            "10",
            "20",
            "--point",
            "20",
            "10",
            "--point",
            "30",
            "20",
            "--preview",
            str(preview),
        ]
    )

    assert result == 0
    updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "circle" not in updated
    assert updated["ball"]["circle"]["u0"] == pytest.approx(20.0)
    assert updated["ball"]["circle"]["v0"] == pytest.approx(20.0)
    assert updated["ball"]["circle"]["r_px"] == pytest.approx(10.0)
    assert updated["ball"]["segment"]["sentinel"] == "preserve-me"
    assert preview.is_file() and preview.stat().st_size > 0


def test_axis_command_writes_ball_r_bc_from_analytic_pure_motions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _base_config()
    config["ball"]["circle"] = {"u0": 20.0, "v0": 20.0, "r_px": 10.0}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    class FakeSource:
        def __init__(self, name: str) -> None:
            self.path = Path(name)
            self.kind = "video"
            self.size = (40, 40)
            self.fps = 60.0
            self.frame_count = 11

    monkeypatch.setattr(
        calibrate_ball_axes,
        "_open_source",
        lambda path, _kind, _max_frames, _fps: FakeSource(path),
    )
    roll_result = SimpleNamespace(top_increments=[Rx(0.01) for _ in range(10)])
    swivel_result = SimpleNamespace(top_increments=[Rz(0.01) for _ in range(10)])
    results = iter([roll_result, swivel_result])
    monkeypatch.setattr(calibrate_ball_axes, "_run_clip", lambda *args, **kwargs: next(results))
    monkeypatch.setattr(
        calibrate_ball_axes,
        "_pipeline_summary",
        lambda _result: {"frame_count": 11, "step_count": 10},
    )

    result = calibrate_ball_axes.main(
        [
            "--config",
            str(config_path),
            "--roll",
            "pure_roll.mp4",
            "--swivel",
            "pure_swivel.mp4",
        ]
    )

    assert result == 0
    updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    np.testing.assert_allclose(updated["ball"]["R_bc"], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        updated["ball"]["R_ball_to_car"], np.eye(3), atol=1e-12
    )
    assert "frame_calib" not in updated
    report_path = tmp_path / "out" / "ball" / "axis_calibration_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["config_key"] == "ball.R_bc"
    np.testing.assert_allclose(report["R_bc"], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(report["R_ball_to_car"], np.eye(3), atol=1e-12)


def test_ball_to_car_rotation_composes_camera_extrinsics() -> None:
    angle = 0.37
    transform = np.eye(4)
    transform[:3, :3] = Rz(angle)
    R_bc = Rx(-0.21)
    np.testing.assert_allclose(
        calibrate_ball_axes.derive_ball_to_car_rotation(R_bc, transform),
        Rz(angle) @ R_bc,
        atol=1e-12,
    )


def test_known_forward_ball_direction_command_flips_reversed_heading(
    tmp_path: Path,
) -> None:
    config = _base_config()
    config["ball"].update(
        {"roll_direction_sign": -1.0, "direction_sign_calibrated": False}
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    frames_path = save_caster_frames(
        tmp_path / "forward.json",
        [
            CasterFrame(
                t=0.1 * index,
                roll_axis_car=np.array([0.0, 1.0, 0.0]),
                omega_roll=2.0,
                omega_spin=0.0,
                r_eff=1.0,
            )
            for index in range(8)
        ],
        metadata={"R_bc_calibrated": True, "R_ball_to_car_calibrated": True},
    )
    report_path = tmp_path / "direction.json"
    assert calibrate_ball_direction.main(
        [
            "--config",
            str(config_path),
            "--caster-frames",
            str(frames_path),
            "--report",
            str(report_path),
        ]
    ) == 0
    updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert updated["ball"]["roll_direction_sign"] == 1.0
    assert updated["ball"]["direction_sign_calibrated"] is True
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["median_current_heading_x"] == -1.0


def test_hsv_range_handles_red_wrap_and_dark_yoke() -> None:
    red = [np.array([[179, 180, 190], [1, 200, 210]], dtype=np.uint8)]
    color_range = inspect_ball_hsv._range(red)
    assert color_range["lo"][0] > color_range["hi"][0]
    dark_range = inspect_ball_hsv._range(
        [np.array([[20, 10, 15], [80, 25, 30]], dtype=np.uint8)],
        dark_object=True,
    )
    assert dark_range["lo"] == [0, 0, 0]
    assert dark_range["hi"][0] == 179


@pytest.mark.parametrize(
    ("angle_arguments", "expected_angle"),
    [
        (["--revolutions", "4"], 8.0 * np.pi),
        (["--angle-rad", str(8.0 * np.pi)], 8.0 * np.pi),
        (["--phi-start", "1.5", "--phi-end", str(1.5 - 8.0 * np.pi)], 8.0 * np.pi),
    ],
)
def test_ball_radius_command_writes_unified_config_for_each_angle_source(
    tmp_path: Path,
    angle_arguments: list[str],
    expected_angle: float,
) -> None:
    config = _base_config()
    config["ball"]["sentinel"] = "preserve-me"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report_path = tmp_path / "radius.json"

    assert calibrate_ball_radius.main(
        [
            "--config",
            str(config_path),
            "--distance-m",
            "2.0",
            *angle_arguments,
            "--report",
            str(report_path),
        ]
    ) == 0

    updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert updated["ball"]["r_eff_m"] == pytest.approx(2.0 / expected_angle)
    assert updated["ball"]["sentinel"] == "preserve-me"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["config_key"] == "ball.r_eff_m"
    assert report["delta_angle_rad"] == pytest.approx(expected_angle)


def test_ball_radius_command_rejects_ambiguous_angle_sources(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_base_config()), encoding="utf-8")
    with pytest.raises(SystemExit, match="choose exactly one angle source"):
        calibrate_ball_radius.main(
            [
                "--config",
                str(config_path),
                "--distance-m",
                "1.0",
                "--revolutions",
                "2",
                "--angle-rad",
                "12.0",
            ]
        )


def test_ball_radius_report_cannot_overwrite_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_base_config()), encoding="utf-8")
    original = config_path.read_bytes()

    with pytest.raises(SystemExit, match="must not overwrite"):
        calibrate_ball_radius.main(
            [
                "--config",
                str(config_path),
                "--distance-m",
                "1.0",
                "--revolutions",
                "2",
                "--report",
                str(config_path),
            ]
        )

    assert config_path.read_bytes() == original


def test_ball_calibration_artifacts_cannot_overwrite_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_base_config()), encoding="utf-8")
    original = config_path.read_bytes()

    with pytest.raises(SystemExit, match="must not overwrite"):
        calibrate_ball_circle.main(
            [
                "--config",
                str(config_path),
                "--preview",
                str(config_path),
            ]
        )
    assert config_path.read_bytes() == original

    with pytest.raises(SystemExit, match="must not overwrite"):
        calibrate_ball_axes.main(
            [
                "--config",
                str(config_path),
                "--roll",
                "unused-roll.mp4",
                "--swivel",
                "unused-swivel.mp4",
                "--report",
                str(config_path),
            ]
        )
    assert config_path.read_bytes() == original
