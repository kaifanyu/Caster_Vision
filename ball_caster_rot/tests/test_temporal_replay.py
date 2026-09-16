"""Missing temporal poses must not appear as measured motion in diagnostic videos."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ballrot.diagnostics import FrameQuality, HemisphereQuality, write_run_outputs
from ballrot.integrate import decompose_hemispheres
from ballrot.rotation import Rx, Rz
from scripts import simulate_measured


def _series() -> dict[str, np.ndarray]:
    values = {
        "time_s": np.arange(3, dtype=float) / 30.0,
        "frame_index": np.arange(3),
        "alpha_rad": np.array([0.0, 0.1, 0.2]),
        "valid_top": np.array([True, False, True]),
        "valid_bottom": np.array([True, True, False]),
    }
    for shell in ("top", "bottom"):
        for component in ("alpha", "beta", "gamma"):
            values[f"{component}_{shell}_rad"] = np.array([0.0, 0.1, 0.2])
    return values


def _fake_video(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    records = [
        SimpleNamespace(index=index, image=np.zeros((180, 320, 3), dtype=np.uint8))
        for index in range(3)
    ]
    written: list[np.ndarray] = []
    monkeypatch.setattr(simulate_measured, "FrameSource", lambda *args, **kwargs: records)
    monkeypatch.setattr(simulate_measured, "undistort_image", lambda image, *args: image.copy())
    monkeypatch.setattr(
        simulate_measured, "_writer",
        lambda *args: SimpleNamespace(write=lambda image: written.append(image.copy()), release=lambda: None),
    )
    return written


def test_loader_keeps_validity_flags_aligned_after_frame_selection(tmp_path: Path) -> None:
    series = _series()
    path = tmp_path / "results.json"
    path.write_text(json.dumps({
        "metadata": {},
        "frames": {key: value.tolist() for key, value in series.items() if key != "frame_index"},
    }), encoding="utf-8")
    loaded = simulate_measured._measured(path, frames=3, stride=2)
    np.testing.assert_array_equal(loaded["frame_index"], [0, 2])
    np.testing.assert_array_equal(loaded["valid_top"], [True, True])
    np.testing.assert_array_equal(loaded["valid_bottom"], [True, False])


def test_output_export_and_replay_preserve_unresolved_middle_pose(tmp_path: Path) -> None:
    top = np.stack([np.eye(3), np.eye(3), Rx(0.2) @ Rz(0.3)])
    bottom = np.stack([np.eye(3), Rx(0.1) @ Rz(-0.1), Rx(0.2) @ Rz(-0.2)])
    motion = decompose_hemispheres(
        top, bottom, np.eye(3),
        valid_top=np.array([True, False, True]),
        valid_bottom=np.ones(3, dtype=bool),
    )
    good = HemisphereQuality(
        matched_count=30, usable_count=30, inlier_count=29, inlier_ratio=29 / 30,
        mean_residual_deg=0.1, median_residual_deg=0.08, median_fb_error_px=0.2,
        success=True,
    )
    temporal = {
        "config": {"enabled": True},
        "summary": {"top": {"unresolved_frames": [1]}},
        "frames": {"top": [
            {"frame_index": 0, "status": "initial", "valid": True},
            {"frame_index": 1, "status": "unresolved", "valid": False},
            {"frame_index": 2, "status": "recovered", "valid": True, "used_keyframes": [0]},
        ]},
    }
    files = write_run_outputs(
        tmp_path, np.arange(3) / 30.0, motion,
        [FrameQuality(1, HemisphereQuality(), good), FrameQuality(2, good, good)],
        top, bottom, metadata={"fps": 30.0}, temporal=temporal,
    )

    payload = json.loads(files["json"].read_text(encoding="utf-8"))
    assert payload["temporal"] == temporal
    assert payload["frames"]["valid_top"] == [True, False, True]
    assert all(isinstance(flag, bool) for flag in payload["frames"]["valid_top"])
    assert payload["frames"]["valid_bottom"] == [True, True, True]
    for component in ("alpha", "gamma", "beta"):
        assert payload["frames"][f"{component}_top_rad"][1] is None
    with files["csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["top_pose_valid"] for row in rows] == ["True", "False", "True"]
    assert [row["bottom_pose_valid"] for row in rows] == ["True", "True", "True"]
    assert np.isnan(float(rows[1]["beta_top_rad"]))

    loaded = simulate_measured._measured(files["json"], frames=None, stride=1)
    np.testing.assert_array_equal(simulate_measured._pose_validity(loaded, "top"), [True, False, True])
    np.testing.assert_array_equal(simulate_measured._pose_validity(loaded, "bottom"), [True, True, True])
    assert np.isnan(loaded["beta_top_rad"][1])
    _, full = simulate_measured._trajectories(loaded, 30.0)
    assert full.beta_top[1] == full.beta_top[0]
    assert full.beta_top[2] == pytest.approx(0.3)
    assert np.isnan(loaded["beta_top_rad"][1])


def test_legacy_results_infer_missing_pose_from_nonfinite_angles() -> None:
    measured = _series()
    measured.pop("valid_top")
    measured.pop("valid_bottom")
    measured["beta_top_rad"][1] = np.nan
    np.testing.assert_array_equal(simulate_measured._pose_validity(measured, "top"), [True, False, True])
    np.testing.assert_array_equal(simulate_measured._pose_validity(measured, "bottom"), [True, True, True])


def test_explicit_invalid_pose_uses_only_a_display_fallback_without_mutating_data() -> None:
    measured = _series()
    _, full = simulate_measured._trajectories(measured, 30.0)
    for component in ("alpha", "gamma", "beta"):
        np.testing.assert_array_equal(getattr(full, f"{component}_top"), [0.0, 0.0, 0.2])
        np.testing.assert_array_equal(getattr(full, f"{component}_bottom"), [0.0, 0.1, 0.1])
        np.testing.assert_array_equal(measured[f"{component}_top_rad"], [0.0, 0.1, 0.2])


@pytest.mark.parametrize("swap", [False, True])
def test_axes_hide_only_the_invalid_shell_grid_and_probes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, swap: bool
) -> None:
    measured = _series()
    written = _fake_video(monkeypatch)
    grid_colors: list[tuple[int, int, int]] = []
    projected_probes: list[np.ndarray] = []
    notices: list[list[str]] = []
    monkeypatch.setattr(simulate_measured, "_graticule", lambda *args: [np.ones((2, 3))])
    monkeypatch.setattr(simulate_measured, "_probe_points", lambda *args: np.ones((2, 3)))
    monkeypatch.setattr(simulate_measured, "_draw_curve", lambda image, curve, rotation, K, C, radius, color: grid_colors.append(color))
    monkeypatch.setattr(simulate_measured, "_draw_axes", lambda *args: None)

    def project(directions, *args):
        projected_probes.append(directions)
        return np.empty((0, 2)), np.empty(0, dtype=bool)

    monkeypatch.setattr(simulate_measured, "_project", project)
    monkeypatch.setattr(simulate_measured, "_draw_missing_pose_notice", lambda image, missing: notices.append(missing))
    simulate_measured._axes_overlay(
        tmp_path / "clip.mkv", measured, np.eye(3), np.zeros(4),
        (160.0, 90.0, 60.0), np.eye(3), np.array([0.0, 0.0, 3.0]),
        1.0, 30.0, tmp_path / "axes.mp4", 45.0, "mp4v", swap_shells=swap,
    )
    assert len(written) == 3
    assert grid_colors == [(255, 220, 0), (180, 0, 255), (180, 0, 255), (255, 220, 0)]
    assert len(projected_probes) == 4
    assert notices == [[], ["top"], ["bottom"]]


def test_replay_labels_fallbacks_on_each_rendered_panel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    measured = _series()
    written = _fake_video(monkeypatch)
    notices: list[tuple[list[str], bool]] = []
    monkeypatch.setattr(
        simulate_measured, "_draw_missing_pose_notice",
        lambda image, missing, *, display_fallback: notices.append((missing, display_fallback)),
    )
    frames = [np.zeros((180, 320, 3), dtype=np.uint8) for _ in range(3)]
    simulate_measured._replay_video(
        tmp_path / "clip.mkv", measured, np.eye(3), np.zeros(4),
        {"model": frames, "full": frames}, 30.0, tmp_path / "replay.mp4", "mp4v",
    )
    assert len(written) == 3
    assert notices == [([], True), ([], True), (["top"], True), (["top"], True), (["bottom"], True), (["bottom"], True)]


def test_missing_pose_notice_is_visible_and_explains_display_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((180, 640, 3), dtype=np.uint8)
    messages: list[str] = []
    original = simulate_measured.cv2.putText

    def record_text(image, message, *args):
        messages.append(message)
        return original(image, message, *args)

    monkeypatch.setattr(simulate_measured.cv2, "putText", record_text)
    simulate_measured._draw_missing_pose_notice(image, ["top"], display_fallback=True)
    assert "UNRESOLVED: top shell pose" in messages
    assert "Display fallback only; missing motion is unmeasured" in messages
    assert np.count_nonzero(image) > 100
