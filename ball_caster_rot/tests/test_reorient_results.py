"""Reorientation must recover measured matrices, including failures and residuals."""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from ballrot.config import measurement_frame
from ballrot.diagnostics import FrameQuality, HemisphereQuality, _json_clean
from ballrot.integrate import accumulate_increments, decompose_hemispheres
from ballrot.rotation import Rx, Rz
from ballrot.trajectory import angular_rates, rotation_rates
from scripts import reorient_results


def _fixture(failed: bool = False):
    count = 24
    home = Rotation.from_euler("xyz", [19, -28, 36], degrees=True).as_matrix()
    effective = home @ Rx(np.deg2rad(90))
    alpha = np.linspace(0, 0.6, count)
    beta_top = np.linspace(0, 0.7, count)
    beta_bottom = np.linspace(0, -0.35, count)
    absolutes = []
    masks = []
    for name, beta in (("top", beta_top), ("bottom", beta_bottom)):
        truth = np.array([effective @ Rx(a) @ Rz(b) @ effective.T for a, b in zip(alpha, beta)])
        increments = [truth[i] @ truth[i - 1].T for i in range(1, count)]
        if failed:
            for index in ([7, 8] if name == "top" else [14]):
                increments[index - 1] = None
        absolute, valid = accumulate_increments(increments)
        absolutes.append(absolute)
        masks.append(valid)
    old = decompose_hemispheres(*absolutes, home, valid_top=masks[0], valid_bottom=masks[1])
    frames = {"time_s": (np.arange(count) / 30).tolist(), "alpha_rad": old.alpha.tolist()}
    for name in ("top", "bottom"):
        for component in ("alpha", "gamma", "beta"):
            frames[f"{component}_{name}_rad"] = getattr(old, f"{component}_{name}").tolist()
    qualities = [
        asdict(FrameQuality(i, *[
            HemisphereQuality(
                matched_count=80, usable_count=75, inlier_count=70,
                inlier_ratio=70 / 75, mean_residual_deg=0.1, median_residual_deg=0.09,
                median_fb_error_px=0.2, success=bool(mask[i]),
            ) for mask in masks
        ])) for i in range(1, count)
    ]
    payload = _json_clean({
        "metadata": {"R_bc": home.tolist(), "frame_count": count, "fps": 30.0},
        "frames": frames, "quality": qualities,
    })
    return payload, home, absolutes, masks, alpha, beta_top, beta_bottom


def test_full_euler_reconstruction_corrects_nonzero_initial_pose() -> None:
    payload, home, absolutes, masks, alpha, beta_top, beta_bottom = _fixture()
    recovered = reorient_results.reconstruct_camera_rotations(payload)
    assert np.max(np.abs(payload["frames"]["gamma_top_rad"])) > 0.4
    np.testing.assert_allclose(recovered.top_absolute, absolutes[0], atol=1e-12)
    np.testing.assert_allclose(recovered.bottom_absolute, absolutes[1], atol=1e-12)
    corrected = decompose_hemispheres(
        recovered.top_absolute, recovered.bottom_absolute,
        measurement_frame({"R_bc": home, "initial_roll_deg": 90}),
        valid_top=recovered.top_step_valid, valid_bottom=recovered.bottom_step_valid,
    )
    np.testing.assert_allclose(corrected.alpha, alpha, atol=1e-12)
    np.testing.assert_allclose(corrected.beta_top, beta_top, atol=1e-12)
    np.testing.assert_allclose(corrected.beta_bottom, beta_bottom, atol=1e-12)
    np.testing.assert_allclose(corrected.gamma_top, 0, atol=1e-12)
    np.testing.assert_allclose(corrected.gamma_bottom, 0, atol=1e-12)


def test_failed_pairs_hold_prior_matrix_and_preserve_gaps_and_quality() -> None:
    payload, home, absolutes, masks, *_ = _fixture(failed=True)
    original = copy.deepcopy(payload)
    recovered = reorient_results.reconstruct_camera_rotations(payload)
    corrected = decompose_hemispheres(
        recovered.top_absolute, recovered.bottom_absolute, home @ Rx(np.pi / 2),
        valid_top=recovered.top_step_valid, valid_bottom=recovered.bottom_step_valid,
    )
    for i, name in enumerate(("top", "bottom")):
        np.testing.assert_allclose(getattr(recovered, f"{name}_absolute"), absolutes[i], atol=1e-12)
        np.testing.assert_array_equal(getattr(recovered, f"{name}_step_valid"), masks[i])
        assert np.all(np.isnan(getattr(corrected, f"beta_{name}")[~masks[i]]))
    np.testing.assert_array_equal(recovered.top_absolute[7], recovered.top_absolute[6])
    np.testing.assert_array_equal(recovered.top_absolute[8], recovered.top_absolute[6])
    assert _json_clean([asdict(q) for q in recovered.qualities]) == payload["quality"]
    assert payload == original


def _add_temporal_validity(payload):
    """Raw pair success deliberately disagrees with absolute-pose validity."""
    count = payload["metadata"]["frame_count"]
    for name in ("top", "bottom"):
        payload["frames"][f"valid_{name}"] = [True] * count
    for index in (7, 8):
        payload["frames"]["valid_top"][index] = False
        for component in ("alpha", "gamma", "beta"):
            payload["frames"][f"{component}_top_rad"][index] = None
    payload["quality"][8]["top"]["success"] = False  # recovered frame 9
    payload["temporal"] = {
        "config": {"enabled": True},
        "summary": {
            "top": {"unresolved_frames": [7, 8], "status_counts": {"recovered": 1}},
            "bottom": {"unresolved_frames": [], "status_counts": {}},
        },
        "frames": {
            name: [{"frame_index": i, "valid": good,
                    "status": ("unresolved" if not good else
                               "recovered" if name == "top" and i == 9 else "increment")}
                   for i, good in enumerate(payload["frames"][f"valid_{name}"])]
            for name in ("top", "bottom")
        },
    }


def _add_offline_recovery(payload):
    """Offline refinement restores frame 7 but leaves frame 8 unresolved."""
    pristine, *_ = _fixture()
    payload["frames"]["valid_top"][7] = True
    for component in ("alpha", "gamma", "beta"):
        payload["frames"][f"{component}_top_rad"][7] = pristine["frames"][f"{component}_top_rad"][7]
    payload["offline"] = {
        "config": {"enabled": True, "rate_window_s": 0.31, "rate_polynomial_order": 2},
        "summary": {
            "top": {"refined_frames": 20, "recovered_frames": 1, "optimized_components": 1,
                    "rejected_components": 0, "unresolved_frames": [8]},
            "bottom": {"refined_frames": 21, "recovered_frames": 0, "optimized_components": 1,
                       "rejected_components": 0, "unresolved_frames": []},
        },
        "shells": {"top": {"components": [{"accepted": True}]}, "bottom": {"components": []}},
        "reverse_observations": [{"source_frame": 9, "target_frame": 7}],
    }


def test_offline_recovery_preserves_report_and_final_pose_validity() -> None:
    payload, _, absolutes, *_ = _fixture()
    _add_temporal_validity(payload)
    _add_offline_recovery(payload)
    original = copy.deepcopy(payload)
    recovered = reorient_results.reconstruct_camera_rotations(payload)
    assert recovered.top_step_valid[7]
    assert not recovered.top_step_valid[8]
    np.testing.assert_allclose(recovered.top_absolute[7], absolutes[0][7], atol=1e-12)
    np.testing.assert_allclose(recovered.top_absolute[8], absolutes[0][7], atol=1e-12)
    assert recovered.offline == payload["offline"]
    assert recovered.temporal == payload["temporal"]
    assert payload == original


def test_offline_diagnostics_require_a_mapping() -> None:
    payload, *_ = _fixture()
    payload["offline"] = []
    with pytest.raises(ValueError, match="offline diagnostics.*mapping"):
        reorient_results.reconstruct_camera_rotations(payload)


def test_explicit_pose_validity_overrides_pair_success_and_preserves_recovery() -> None:
    payload, _, absolutes, *_ = _fixture()
    _add_temporal_validity(payload)
    original = copy.deepcopy(payload)
    recovered = reorient_results.reconstruct_camera_rotations(payload)
    assert not recovered.top_step_valid[7]  # pair succeeded; temporal pose rejected
    assert recovered.top_step_valid[9]  # pair failed; keyframe recovered full pose
    np.testing.assert_allclose(recovered.top_absolute[7:9], [absolutes[0][6]] * 2, atol=1e-12)
    np.testing.assert_allclose(recovered.top_absolute[9:], absolutes[0][9:], atol=1e-12)
    assert recovered.temporal == payload["temporal"]
    assert payload == original


@pytest.mark.parametrize("invalid", [1, "false", None])
def test_explicit_pose_validity_requires_actual_booleans(invalid) -> None:
    payload, *_ = _fixture()
    _add_temporal_validity(payload)
    payload["frames"]["valid_top"][4] = invalid
    with pytest.raises(ValueError, match="valid_top.*booleans"):
        reorient_results.reconstruct_camera_rotations(payload)


def test_explicit_pose_validity_requires_valid_frame_zero() -> None:
    payload, *_ = _fixture()
    _add_temporal_validity(payload)
    payload["frames"]["valid_top"][0] = False
    with pytest.raises(ValueError, match="frame zero"):
        reorient_results.reconstruct_camera_rotations(payload)


@pytest.mark.parametrize("damage, message", [
    (lambda p: p["metadata"].update(R_bc=np.zeros((3, 3)).tolist()), "proper rotation"),
    (lambda p: p["metadata"].update(frame_count=23), "frame_count"),
    (lambda p: p["frames"]["alpha_top_rad"].__setitem__(0, 0.2), "frame zero"),
    (lambda p: p["frames"]["gamma_bottom_rad"].__setitem__(2, None), "successful frame"),
    (lambda p: p["quality"].pop(), "every frame pair"),
    (lambda p: p["quality"][1].update(frame_index=1), "consecutive"),
])
def test_corrupt_saved_data_is_rejected(damage, message: str) -> None:
    payload, *_ = _fixture()
    damage(payload)
    with pytest.raises(ValueError, match=message):
        reorient_results.reconstruct_camera_rotations(payload)


def test_timestamps_preserve_variable_intervals_and_explicit_override() -> None:
    source = SimpleNamespace(
        records=lambda: iter([
            SimpleNamespace(index=i, timestamp_s=t) for i, t in enumerate([3.0, 3.033, 3.161, 3.194])
        ]),
        frames=lambda: iter([np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(4)]),
    )
    np.testing.assert_allclose(reorient_results._source_times(source, 4, None), [0, .033, .161, .194])
    np.testing.assert_allclose(reorient_results._source_times(source, 4, 60), np.arange(4) / 60)
    with pytest.raises(ValueError, match="decoded 4 frames"):
        reorient_results._source_times(source, 3, None)


def test_explicit_fps_override_bypasses_native_timestamp_validation() -> None:
    def invalid_native_records():
        raise ValueError("invalid native presentation timestamps")

    source = SimpleNamespace(
        records=invalid_native_records,
        frames=lambda: iter([np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(4)]),
    )
    np.testing.assert_allclose(reorient_results._source_times(source, 4, 30), np.arange(4) / 30)


def test_changed_geometry_requires_retracking(tmp_path: Path) -> None:
    source = SimpleNamespace(path=tmp_path / "clip.mkv", size=(1920, 1080))
    config = {
        "camera": {"K": [[1400, 0, 960], [0, 1400, 540], [0, 0, 1]], "dist": [0, 0, 0, 0]},
        "circle": {"u0": 875, "v0": 485, "r_px": 446},
    }
    metadata = {"input": str(source.path), **config["camera"], "circle": [875, 485, 446]}
    reorient_results._same_geometry(config, metadata, source)
    config["circle"]["r_px"] = 445
    with pytest.raises(ValueError, match="geometry changes require retracking"):
        reorient_results._same_geometry(config, metadata, source)


def test_cli_refuses_overwriting_source_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "assumptions:\n  camera_fixed_to_chassis: true\n  ball_center_stationary_in_image: true\n"
        "  two_speckle_colors: true\noutput:\n  dir: out\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="originals are preserved"):
        reorient_results.main([
            "--config", str(config_path), "--results", str(tmp_path / "out" / "results.json")
        ])


@pytest.mark.parametrize("temporal,offline", [(False, False), (True, False), (True, True)])
def test_cli_writes_corrected_results_with_native_times_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], temporal: bool, offline: bool,
) -> None:
    payload, home, _, _, alpha, *_ = _fixture()
    if temporal:
        _add_temporal_validity(payload)
    if offline:
        _add_offline_recovery(payload)
    input_path = tmp_path / "clip.mkv"
    output_dir = tmp_path / "corrected"
    camera = {"K": [[1400, 0, 960], [0, 1400, 540], [0, 0, 1]], "dist": [0, 0, 0, 0]}
    payload["metadata"].update(input=str(input_path), **camera, circle=[875, 485, 446])
    original_path = tmp_path / "results.json"
    original_path.write_text(json.dumps(payload), encoding="utf-8")
    original_bytes = original_path.read_bytes()
    config = {
        "input": {"path": str(input_path)}, "camera": camera,
        "circle": {"u0": 875, "v0": 485, "r_px": 446},
        "frame_calib": {"R_bc": home.tolist(), "initial_roll_deg": 90, "top_shell_sign": -1},
        "output": {"dir": str(output_dir)},
        "assumptions": {
            "camera_fixed_to_chassis": True, "ball_center_stationary_in_image": True,
            "two_speckle_colors": True,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    config_bytes = config_path.read_bytes()
    times = np.concatenate(([0.0], np.cumsum([.128 if i % 7 == 0 else .033 for i in range(23)])))
    source = SimpleNamespace(
        path=input_path, size=(1920, 1080), timing_source="native_video_pts",
        records=lambda: iter([SimpleNamespace(index=i, timestamp_s=t) for i, t in enumerate(times)]),
    )
    monkeypatch.setattr(reorient_results, "FrameSource", lambda *args, **kwargs: source)
    assert reorient_results.main([
        "--config", str(config_path), "--results", str(original_path), "--strict"
    ]) == (2 if temporal else 0)
    result = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    np.testing.assert_allclose(result["frames"]["time_s"], times)
    np.testing.assert_allclose(result["frames"]["alpha_rad"], alpha, atol=1e-12)
    if offline:
        shared_valid = np.asarray(payload["frames"]["valid_top"]) & np.asarray(payload["frames"]["valid_bottom"])
        expected_alpha_rate = angular_rates(alpha, times, shared_valid, window_s=0.31)
    else:
        expected_alpha_rate = np.gradient(alpha, times)
    np.testing.assert_allclose(
        np.asarray(result["frames"]["alpha_velocity_rad_s"], dtype=float),
        expected_alpha_rate, atol=1e-12, equal_nan=True,
    )
    assert result["metadata"]["fps"] == 30.0
    assert result["metadata"]["timing_source"] == "native_video_pts"
    assert result["metadata"]["initial_roll_deg"] == 90
    assert result["metadata"]["top_shell_sign"] == -1
    assert result["quality"] == payload["quality"]
    if temporal:
        assert result["temporal"] == payload["temporal"]
        assert result["summary"]["temporal_tracking"]["top"]["unresolved_frames"] == [7, 8]
        assert result["frames"]["valid_top"] == payload["frames"]["valid_top"]
        if not offline:
            assert "2 unresolved poses" in capsys.readouterr().out
        source_poses = reorient_results.reconstruct_camera_rotations(payload)
        output_poses = reorient_results.reconstruct_camera_rotations(result)
        np.testing.assert_allclose(output_poses.top_absolute, source_poses.top_absolute, atol=1e-12)
    if offline:
        assert result["offline"] == payload["offline"]
        assert result["summary"]["temporal_tracking"] == payload["temporal"]["summary"]
        assert result["summary"]["offline_tracking"]["top"]["unresolved_frames"] == [8]
        assert result["summary"]["offline_tracking"]["top"]["recovered_frames"] == 1
        beta = np.asarray(result["frames"]["beta_top_rad"], dtype=float)
        expected_beta_rate = angular_rates(beta, times, payload["frames"]["valid_top"], window_s=0.31)
        actual_beta_rate = np.asarray(result["frames"]["beta_top_velocity_rad_s"], dtype=float)
        np.testing.assert_allclose(actual_beta_rate, expected_beta_rate, atol=1e-12, equal_nan=True)
        assert np.isnan(actual_beta_rate[8])
        assert result["frames"]["alpha_velocity_rad_s"][8] is None
        with (output_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        for name in ("top", "bottom"):
            expected_omega = rotation_rates(
                getattr(source_poses, f"{name}_absolute"), times,
                payload["frames"][f"valid_{name}"], window_s=0.31,
            )
            actual_omega = np.asarray(result["frames"][f"omega_{name}_camera_rad_s"], dtype=float)
            np.testing.assert_allclose(actual_omega, expected_omega, atol=1e-12, equal_nan=True)
            for component, axis in enumerate("xyz"):
                csv_omega = np.asarray([row[f"omega_{name}_camera_{axis}_rad_s"] for row in csv_rows], dtype=float)
                np.testing.assert_allclose(csv_omega, expected_omega[:, component], atol=1e-12, equal_nan=True)
        assert result["metadata"]["rate_processing"]["shared_roll_support"] == "both shells valid"
        assert "invalid gaps are never bridged" in result["metadata"]["reorientation_note"]
        assert "both shells" in result["metadata"]["reorientation_note"]
        assert "gap-safe rates were recomputed" in capsys.readouterr().out
    assert original_path.read_bytes() == original_bytes
    assert config_path.read_bytes() == config_bytes
