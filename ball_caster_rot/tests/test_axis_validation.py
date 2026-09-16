"""Validation observations and coverage must not depend on pose fit success."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ballrot.axis_validation import (ValidationTrajectory, archived_observations, compare_observations,
    default_pairs, distance_from_training, known_angle_errors, load_trajectory, observe_pair, paint_masks, predict_pair)
from ballrot.rotation import Rx, Rz
from ballrot.shell_geometry import surface_camera
from synthetic.generate import default_camera


def trajectory(geometry="separated_hemispheres"):
    camera = default_camera((240, 320))
    F = camera.R_bc
    orientations = np.array([F, F @ Rx(.02) @ Rz(.03)])
    metadata = {"K": camera.K.tolist(), "dist": camera.dist.tolist(), "input": "source.mkv", "R_bc": F.tolist()}
    return ValidationTrajectory(Path("saved.json"), "fused", metadata, np.array([0., .04]),
        {shell: orientations.copy() for shell in ("top", "bottom")},
        {shell: np.ones(2, bool) for shell in ("top", "bottom")},
        {shell: np.array(["vision", "vision"]) for shell in ("top", "bottom")},
        geometry, .1, camera.C, camera.radius)


def points_and_pixels(data, shell="top"):
    points = np.array([[-.2, -.8, .5], [.1, -.9, .4], [.3, -.8, .5]])
    sign = 1 if shell == "top" else -1
    points[:, 2] *= sign
    points /= np.linalg.norm(points, axis=1)[:, None]
    pixels = []
    for orientation in data.orientations[shell]:
        xyz = surface_camera(points, orientation, data.center, data.radius, sign, data.gap, data.geometry)
        h = xyz @ np.asarray(data.metadata["K"]).T
        pixels.append(h[:, :2] / h[:, 2:])
    return points, pixels


@pytest.mark.parametrize("geometry", ["common_sphere_caps", "separated_hemispheres"])
@pytest.mark.parametrize("shell", ["top", "bottom"])
def test_saved_surface_projection_matches_observed_rigid_motion(geometry, shell):
    data = trajectory(geometry)
    _, pixels = points_and_pixels(data, shell)
    predicted, valid, available = predict_pair(data, 0, 1, shell, pixels[0])
    assert available and valid.all()
    np.testing.assert_allclose(predicted, pixels[1], atol=1e-10)


def test_unavailable_and_ray_miss_keep_original_observation_denominator():
    baseline = trajectory()
    candidate = deepcopy(baseline)
    candidate.valid["top"][1] = False
    _, pixels = points_and_pixels(baseline)
    record = {"source_frame": 0, "target_frame": 1, "shell": "top", "diagnostics": {},
              "source_uv": np.vstack((pixels[0], [-5000, -5000])).tolist(),
              "target_uv": np.vstack((pixels[1], [-5000, -5000])).tolist()}
    report = compare_observations(baseline, candidate, [record])
    a, b = report["aggregate_all_observations"].values()
    assert a["observation_count"] == b["observation_count"] == 4
    assert a["projectable_count"] == 3 and a["coverage"] == .75
    assert a["inlier_fraction_all_observations"] == .75
    assert b["coverage"] == b["inlier_fraction_all_observations"] == 0.
    assert b["median_px"] is None
    assert report["aggregate_common_support"]["baseline"]["projectable_count"] == 0
    assert report["pose_available_pair_shell_count"] == {"baseline": 1, "candidate": 0}


def test_wrong_pose_produces_pixel_residuals_without_changing_observations():
    baseline, candidate = trajectory(), trajectory()
    candidate.orientations["top"][1] = candidate.orientations["top"][1] @ Rz(.2)
    _, pixels = points_and_pixels(baseline)
    record = {"source_frame": 0, "target_frame": 1, "shell": "top", "diagnostics": {},
              "source_uv": pixels[0].tolist(), "target_uv": pixels[1].tolist()}
    snapshot = deepcopy(record)
    report = compare_observations(baseline, candidate, [record])
    assert record == snapshot
    assert report["aggregate_common_support"]["baseline"]["median_px"] < 1e-8
    assert report["aggregate_common_support"]["candidate"]["median_px"] > 5.


def textured_translation():
    rng = np.random.default_rng(15)
    image = cv2.GaussianBlur(rng.integers(0, 256, (220, 320), dtype=np.uint8), (3, 3), .5)
    shifted = cv2.warpAffine(image, np.array([[1., 0., 3.], [0., 1., -2.]]), (320, 220))
    return image, shifted


def test_image_only_correspondences_exclude_training_at_both_endpoints():
    source, target = textured_translation()
    training_a, training_b = np.array([[80., 80.], [200., 150.]]), np.array([[100., 100.], [230., 130.]])
    mask = np.ones(source.shape, bool)
    record = observe_pair(source, target, mask, mask, training_a, training_b, exclusion_px=30.)
    first, second = np.asarray(record["source_uv"]), np.asarray(record["target_uv"])
    assert len(first) > 50
    assert np.min(distance_from_training(first, training_a)) >= 30
    assert np.min(distance_from_training(second, training_b)) >= 30
    np.testing.assert_allclose(np.median(second-first, axis=0), [3., -2.], atol=.04)
    assert max(record["fb_error_px"]) <= .75
    assert min(record["patch_ncc"]) >= .75


def test_no_heldout_pixels_does_not_fallback_to_training_points():
    source, target = textured_translation()
    yy, xx = np.mgrid[0:220:20, 0:320:20]
    training = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
    mask = np.ones(source.shape, bool)
    record = observe_pair(source, target, mask, mask, training, training, exclusion_px=30.)
    assert record["source_uv"] == record["target_uv"] == []
    assert record["diagnostics"]["detected_heldout_corners"] == 0
    assert record["diagnostics"]["observation_coverage"] == 0


def test_marker_masks_reject_same_color_background_without_shell_paper():
    image = np.full((240, 320, 3), 30, np.uint8)
    image[65:185, 140:265] = 220
    cv2.fillConvexPoly(image, np.array([[180, 115], [195, 130], [175, 133]]), (0, 0, 220))
    cv2.fillConvexPoly(image, np.array([[80, 115], [95, 130], [75, 133]]), (0, 0, 220))
    masks = paint_masks(image, [160, 120, 100], {
        "top_hsv": {"lo": [0, 90, 45], "hi": [10, 255, 255]},
        "bottom_hsv": {"lo": [60, 40, 45], "hi": [95, 255, 255]}})
    assert masks["top"][125, 185]
    assert not masks["top"][125, 85]


def test_known_angles_compare_orientation_modulo_turns(tmp_path):
    data = trajectory()
    reference = tmp_path / "known.csv"
    reference.write_text("frame_index,alpha_deg,beta_top_deg,beta_bottom_deg\n0,360,720,-360\n", encoding="utf-8")
    report = known_angle_errors(data, reference)
    assert report["shells"]["top"]["max_deg"] < 1e-10
    assert report["shells"]["bottom"]["max_deg"] < 1e-10
    assert "hidden revolution counts are not validated" in report["turn_count_ambiguity"]


def test_known_angle_reference_frame_stays_fixed_when_calibration_changes(tmp_path):
    baseline, candidate = trajectory(), trajectory()
    correction = Rx(np.deg2rad(10.))
    candidate.metadata["R_bc"] = (correction @ np.asarray(baseline.metadata["R_bc"])).tolist()
    for shell in candidate.orientations:
        candidate.orientations[shell] = correction @ candidate.orientations[shell]
    reference = tmp_path / "known.csv"
    reference.write_text("frame_index,alpha_deg,beta_top_deg,beta_bottom_deg\n0,0,0,0\n", encoding="utf-8")
    report = known_angle_errors(candidate, reference, reference_frame=baseline.metadata["R_bc"])
    assert report["shells"]["top"]["max_deg"] == pytest.approx(10.)


def test_fused_loader_uses_saved_geometry_and_per_channel_status(tmp_path):
    camera = default_camera((240, 320))
    data = {"method": "experimental_angle_kalman", "metadata": {
        "R_bc": camera.R_bc.tolist(), "K": camera.K.tolist(), "circle": list(camera.circle),
        "mechanical_model": {"geometry": "separated_hemispheres", "gap_fraction": .15, "pivot_camera": [0., .1, 4.]}
    }, "frames": {"time_s": [0., .04], "alpha_rad": [.1, .2], "alpha_status": ["vision", "predicted"],
        "beta_top_rad": [.2, .3], "beta_top_status": ["vision", "predicted"],
        "beta_bottom_rad": [None, .3], "beta_bottom_status": ["unresolved", "vision"]}}
    path = tmp_path / "results.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_trajectory(path)
    assert loaded.geometry == "separated_hemispheres" and loaded.gap == .15
    np.testing.assert_equal(loaded.center, [0., .1, 4.])
    np.testing.assert_allclose(loaded.orientations["top"][1], camera.R_bc @ Rx(.2) @ Rz(.3))
    assert loaded.valid["bottom"].tolist() == [False, True]
    assert loaded.status["bottom"].tolist() == ["unresolved", "experimental_predicted"]


def test_archives_union_all_methods_and_provenance(tmp_path):
    paths = []
    for index in range(2):
        directory = tmp_path / str(index)
        directory.mkdir()
        values = {}
        for shell in ("top", "bottom"):
            values[f"{shell}_frame_index"] = np.array([0, 1])
            values[f"{shell}_uv"] = np.array([[10.+index, 20.], [30., 40.]])
        np.savez(directory / "offline_observations.npz", **values)
        path = directory / "results.json"
        path.write_text(json.dumps({"metadata": {}}))
        paths.append(path)
    pixels, archives = archived_observations(paths, [0, 1, 2])
    assert len(archives) == 2
    assert pixels[0].tolist() == [[10., 20.], [11., 20.]]
    assert len(pixels[1]) == 1 and len(pixels[2]) == 0


def test_default_pairs_include_cutoffs_without_using_pose_status():
    times = np.arange(557) * .0352
    pairs = default_pairs(times)
    for target in (4.68, 15.31):
        index = int(np.argmin(abs(times-target)))
        assert (index, index+1) in pairs
        assert (0, index) in pairs


def test_cli_exports_empty_support_honestly_and_reuses_observations(monkeypatch, tmp_path):
    from scripts import validate_tracking as cli
    data = trajectory()
    data.metadata["circle"] = [160., 120., 85.]
    monkeypatch.setattr(cli, "load_trajectory", lambda *args: deepcopy(data))
    image = np.full((240, 320, 3), 128, np.uint8)
    monkeypatch.setattr(cli, "FrameSource", lambda *args, **kwargs: iter([
        SimpleNamespace(index=index, image=image.copy()) for index in range(2)]))
    monkeypatch.setattr(cli, "undistort_image", lambda image, *args: image)
    monkeypatch.setattr(cli, "archived_observations", lambda *args: ({0: np.empty((0, 2)), 1: np.empty((0, 2))}, []))
    config = tmp_path / "config.yaml"
    config.write_text("segment:\n  top_hsv: {lo: [0, 90, 45], hi: [10, 255, 255]}\n  bottom_hsv: {lo: [60, 40, 45], hi: [95, 255, 255]}\n")
    args = ["--baseline", "baseline.json", "--candidate", "candidate.json", "--config", str(config),
            "--pairs", "0:1", "--output", str(tmp_path / "first")]
    assert cli.main(args) == 0
    report = json.loads((tmp_path / "first/validation_report.json").read_text())
    heldout = report["observation_sets"]["heldout_base_patches"]
    assert heldout["total_observations"] == 0
    assert "not established" in heldout["holdout_statement"]
    assert report["known_angle_ground_truth"] is None
    assert Path(heldout["contact_sheet"]).is_file()
    args[-1] = str(tmp_path / "second")
    assert cli.main(args + ["--observations", str(tmp_path / "first/validation_observations.json")]) == 0
    reused = json.loads((tmp_path / "second/validation_report.json").read_text())
    assert reused["provenance"]["reused_from"].endswith("validation_observations.json")
