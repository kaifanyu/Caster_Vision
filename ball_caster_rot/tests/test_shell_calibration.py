"""Known tangent geometry, rejected contour evidence and explicit config writes."""
from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from ballrot.rotation import Rx
from ballrot.shell_calibration import calibrate_shell_pivot
from scripts import calibrate_shell_geometry as cli


K = np.array([[900.0, 0, 640], [0, 910.0, 360], [0, 0, 1]])
F = Rx(1.35)
PIVOT = np.array([0.08, -0.04, 3.8])


def silhouette_frames(*, top_sign=1, noise=0.0, alphas=(0.0,)):
    rng = np.random.default_rng(71)
    frames = []
    for frame_number, alpha in enumerate(alphas):
        normal = (F @ Rx(np.deg2rad(alpha)))[:, 2]
        sample = {"frame_index": frame_number * 10, "alpha_deg": alpha}
        for shell, sign in (("top", top_sign), ("bottom", -top_sign)):
            center = PIVOT + sign * 0.1 * normal
            axis = center / np.linalg.norm(center)
            first = np.array([1., 0, 0]) - axis[0] * axis
            first /= np.linalg.norm(first)
            second = np.cross(axis, first)
            phi = np.linspace(0, 2 * np.pi, 120, endpoint=False)
            radial = np.cos(phi)[:, None] * first + np.sin(phi)[:, None] * second
            sin_theta = 1.0 / np.linalg.norm(center)
            rays = np.sqrt(1 - sin_theta**2) * axis + sin_theta * radial
            surface_normals = np.sqrt(center @ center - 1) * rays - center
            rays = rays[sign * (surface_normals @ normal) > 0.2][::2]
            projected = rays @ K.T
            pixels = projected[:, :2] / projected[:, 2:]
            pixels += rng.normal(0, noise, pixels.shape)
            sample[shell] = pixels.tolist()
        frames.append(sample)
    return frames


@pytest.mark.parametrize("top_sign", [-1, 1])
def test_exact_outer_arcs_recover_pivot_with_known_rolls(top_sign):
    report = calibrate_shell_pivot(silhouette_frames(top_sign=top_sign, alphas=(0, 18)), K, F,
                                   top_shell_sign=top_sign, initial_pivot=[0, 0, 3.3])
    assert report["accepted"] and report["status"] == "candidate_passed"
    np.testing.assert_allclose(report["pivot_camera"], PIVOT, atol=1e-8)
    assert report["condition_number"] < 100
    assert all(details["arc_coverage_deg"] > 90 for details in report["per_shell"].values())
    assert max(row["error_px"] for row in report["observations"]) < 1e-7


def test_noisy_contours_and_a_few_outliers_retain_a_supported_candidate():
    frames = silhouette_frames(noise=0.15)
    for shell in ("top", "bottom"):
        frames[0][shell][3][0] += 18
    report = calibrate_shell_pivot(frames, K, F)
    assert report["accepted"]
    np.testing.assert_allclose(report["pivot_camera"], PIVOT, atol=0.006)
    assert all(details["inlier_fraction"] > 0.8 for details in report["per_shell"].values())
    assert sum(not row["inlier"] for row in report["observations"]) >= 2


def test_narrow_arc_cannot_be_written_as_a_good_calibration():
    frames = silhouette_frames()
    for shell in ("top", "bottom"):
        frames[0][shell] = frames[0][shell][5:11]
    report = calibrate_shell_pivot(frames, K, F, initial_pivot=PIVOT)
    assert not report["accepted"]
    assert "top_arc_too_narrow" in report["reasons"]
    assert "bottom_arc_too_narrow" in report["reasons"]


def test_swapped_edge_labels_fail_hemisphere_or_pixel_support():
    frames = silhouette_frames()
    frames[0]["top"], frames[0]["bottom"] = frames[0]["bottom"], frames[0]["top"]
    report = calibrate_shell_pivot(frames, K, F, initial_pivot=PIVOT)
    assert not report["accepted"]
    assert any("inlier_support" in reason for reason in report["reasons"])


def test_both_shells_and_independent_points_are_required():
    frames = silhouette_frames()
    frames[0]["bottom"] = []
    with pytest.raises(ValueError, match="points for bottom"):
        calibrate_shell_pivot(frames, K, F)
    frames = silhouette_frames()
    frames[0]["top"].append(frames[0]["top"][0])
    with pytest.raises(ValueError, match="duplicate"):
        calibrate_shell_pivot(frames, K, F)


def test_later_frames_require_explicit_known_roll():
    frames = silhouette_frames()
    frames[0]["frame_index"] = 20
    del frames[0]["alpha_deg"]
    with pytest.raises(ValueError, match="explicit known alpha_deg"):
        calibrate_shell_pivot(frames, K, F)


def test_separate_narrow_arcs_do_not_become_one_wide_arc():
    frames = silhouette_frames(alphas=(0.0, 18.0))
    for shell in ("top", "bottom"):
        frames[0][shell] = frames[0][shell][:6]
        frames[1][shell] = frames[1][shell][-6:]
    report = calibrate_shell_pivot(frames, K, F, initial_pivot=PIVOT)
    assert not report["accepted"]
    assert all(details["arc_coverage_deg"] < 60 for details in report["per_shell"].values())
    assert all(set(details["arc_coverage_by_frame_deg"]) == {"0", "10"}
               for details in report["per_shell"].values())


def _cli_inputs(tmp_path, monkeypatch, frames):
    config = {
        "input": {"path": "clip.mp4"}, "camera": {"K": K.tolist(), "dist": [0, 0, 0, 0]},
        "circle": {"u0": 640.0, "v0": 360.0, "r_px": 280.0},
        "frame_calib": {"R_bc": F.tolist(), "initial_roll_deg": 0.0},
        "mechanical": {"enabled": True, "geometry": "separated_hemispheres", "gap_fraction": 0.1},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    points_path = tmp_path / "reviewed.json"
    points_path.write_text(json.dumps({"coordinate_system": "undistorted_pixels", "frames": frames}), encoding="utf-8")
    monkeypatch.setattr(cli, "_read_frames", lambda clip, indices, *args: {
        index: np.zeros((720, 1280, 3), np.uint8) for index in indices})
    return config, config_path, points_path


@pytest.mark.parametrize("write", [False, True])
def test_cli_applies_only_explicit_write_and_preserves_camera_circle(tmp_path, monkeypatch, write):
    config, config_path, points_path = _cli_inputs(tmp_path, monkeypatch, silhouette_frames())
    output = tmp_path / "candidate"
    argv = ["--config", str(config_path), "--points", str(points_path), "--output", str(output)]
    if write:
        argv.append("--write")
    assert cli.main(argv) == 0
    updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert updated["camera"] == config["camera"] and updated["circle"] == config["circle"]
    assert ("pivot_camera" in updated["mechanical"]) == write
    without_pivot = {**updated, "mechanical": {key: value for key, value in updated["mechanical"].items()
                                              if key != "pivot_camera"}}
    assert without_pivot == config
    report = json.loads((output / "shell_calibration_report.json").read_text(encoding="utf-8"))
    assert report["config_written"] is write
    assert report["normalized_shell_radius"] == 1.0
    assert (output / "boundary_points.json").is_file()
    assert (output / "frame_000000.png").is_file()


def test_cli_write_cannot_bypass_failed_candidate(tmp_path, monkeypatch):
    frames = silhouette_frames()
    for shell in ("top", "bottom"):
        frames[0][shell] = frames[0][shell][5:11]
    config, config_path, points_path = _cli_inputs(tmp_path, monkeypatch, frames)
    before = config_path.read_bytes()
    output = tmp_path / "rejected"
    assert cli.main(["--config", str(config_path), "--points", str(points_path), "--output", str(output), "--write"]) == 2
    assert config_path.read_bytes() == before
    report = json.loads((output / "shell_calibration_report.json").read_text(encoding="utf-8"))
    assert not report["accepted"] and not report["config_written"]
