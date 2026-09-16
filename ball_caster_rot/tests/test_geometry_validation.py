import copy
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from ballrot.config import measurement_frame
from ballrot.geometry_validation import (TransferDataset, build_transfer_dataset,
                                        mechanical_orientations, transfer_predictions,
                                        validate_geometry)
from ballrot.offline import SurfaceObservation
from ballrot.rotation import Rx, Rz
from ballrot.shell_geometry import surface_camera


def scene(count=75, points=400):
    rng = np.random.default_rng(28)
    K = np.array([[750., 0, 450], [0, 760., 350], [0, 0, 1.]])
    F = Rx(np.deg2rad(78.))
    C, gap = np.array([.04, -.07, 4.1]), .12
    t = np.linspace(0, 1, count)
    alpha = .45*np.sin(t*3.8)
    beta = {"top": 2.2*t, "bottom": -2.8*t}
    rotations, flags, observed = {}, {}, {}
    for shell, sign in (("top", 1), ("bottom", -1)):
        q = np.array([Rx(a) @ Rz(b) for a, b in zip(alpha, beta[shell])])
        orientations = F @ q
        rotations[shell] = orientations @ F.T
        flags[shell] = np.ones(count, bool)
        local = rng.normal(size=(points, 3))
        local /= np.linalg.norm(local, axis=1)[:, None]
        local[:, 2] = sign*np.abs(local[:, 2])
        observations = []
        for index, orientation in enumerate(orientations):
            xyz = surface_camera(local, orientation, C, 1., sign, gap, "separated_hemispheres")
            normals = local @ orientation.T
            visible = -np.einsum("ij,ij->i", normals, xyz)/np.linalg.norm(xyz, axis=1) > .3
            pixels = xyz @ K.T
            pixels = pixels[:, :2]/pixels[:, 2:]
            for identifier in np.flatnonzero(visible):
                observations.append(SurfaceObservation(index, int(identifier), pixels[identifier]))
        observed[shell] = observations
    return K, F, C, gap, rotations, flags, observed


@pytest.fixture(scope="module")
def synthetic():
    K, F, C, gap, rotations, flags, observations = scene()
    dataset = build_transfer_dataset(observations, rotations, flags, K, C, F, gap,
                                     view_count=18, max_per_view=24)
    return K, F, C, gap, dataset


def test_exact_transfer_and_signed_shell_center(synthetic):
    K, F, C, gap, dataset = synthetic
    prediction, valid, _ = transfer_predictions(dataset, K, C, F, gap)
    target = np.array([row["observed_uv"] for row in dataset.samples])
    assert len(target) > 100 and valid.all()
    np.testing.assert_allclose(prediction, target, atol=1e-9)
    wrong, _, _ = transfer_predictions(dataset, K, C, F, gap, top_shell_sign=-1)
    assert np.median(np.linalg.norm(wrong-target, axis=1)) > 1


def test_selection_reserves_frames_and_landmarks_and_source_validity(synthetic):
    *_, dataset = synthetic
    training = [row for row in dataset.samples if row["split"] == "training"]
    heldout = [row for row in dataset.samples if row["split"] == "heldout"]
    assert training and heldout
    train_ids = {(row["shell"], row["track_id"]) for row in training}
    hold_ids = {(row["shell"], row["track_id"]) for row in heldout}
    hold_frames = {row["target_frame"] for row in heldout}
    assert not train_ids & hold_ids
    assert not any(row["source_frame"] in hold_frames or row["target_frame"] in hold_frames for row in training)
    assert all(dataset.valid[row["shell"]][row["source_frame"]]
               and dataset.valid[row["shell"]][row["target_frame"]] for row in dataset.samples)


def test_candidate_frame_redecomposes_original_camera_rotations(synthetic):
    _, F, _, _, dataset = synthetic
    changed = Rotation.from_rotvec([.025, -.012, .02]).as_matrix() @ F
    orientations, angles = mechanical_orientations(dataset.rotations, dataset.valid, changed)
    original_orientations, _ = mechanical_orientations(dataset.rotations, dataset.valid, F)
    for shell in ("top", "bottom"):
        expected_components = Rotation.from_matrix(changed.T @ dataset.rotations[shell] @ changed).as_euler("XYZ")
        np.testing.assert_allclose(angles[shell], expected_components)
    stale = changed @ F.T @ original_orientations["top"]
    assert np.max(np.abs(orientations["top"]-stale)) > .005


def test_heldout_pixels_do_not_change_geometry_fit(synthetic):
    K, F, C, gap, dataset = synthetic
    seed = C + [.018, -.014, .09]
    report = validate_geometry(dataset, K, seed, F, gap, max_nfev=35)
    assert report["candidate_config_permitted"]
    np.testing.assert_allclose(report["candidate_geometry"]["pivot_camera"], C, atol=1e-3)
    assert abs(report["candidate_geometry"]["gap_fraction"]-gap) < 1e-3
    changed = copy.deepcopy(dataset)
    for row in changed.samples:
        if row["split"] == "heldout":
            row["observed_uv"] = (np.array(row["observed_uv"]) + [40., -25.]).tolist()
    bad = validate_geometry(changed, K, seed, F, gap, max_nfev=35)
    np.testing.assert_allclose(report["candidate_geometry"]["pivot_camera"], bad["candidate_geometry"]["pivot_camera"], atol=1e-10)
    np.testing.assert_allclose(report["candidate_geometry"]["measurement_frame"], bad["candidate_geometry"]["measurement_frame"], atol=1e-10)
    assert report["candidate"]["training"]["median_px"] < report["baseline"]["training"]["median_px"]*.25
    assert bad["candidate"]["heldout"]["median_px"] > 20
    assert not bad["candidate_config_permitted"]
    assert "baseline_predicted_uv" not in dataset.samples[0]


def test_split_leakage_is_rejected(synthetic):
    K, F, C, gap, dataset = synthetic
    data = copy.deepcopy(dataset)
    row = copy.deepcopy(next(row for row in data.samples if row["split"] == "heldout"))
    row["split"] = "training"
    data.samples.append(row)
    with pytest.raises(ValueError, match="leakage"):
        validate_geometry(data, K, C, F, gap)


def test_absent_motion_and_support_cannot_pass_geometry_gates(synthetic):
    K, F, C, gap, data = synthetic
    empty = TransferDataset([], data.rotations, data.valid, {})
    report = validate_geometry(empty, K, C, F, gap)
    assert report["status"] == "candidate_rejected"
    assert not report["solver"]["attempted"]
    assert "geometry_not_observable_without_priors" in report["rejection_reasons"]


def test_perfect_static_pixels_do_not_make_geometry_observable(synthetic):
    K, F, C, gap, original = synthetic
    data = copy.deepcopy(original)
    for shell in ("top", "bottom"):
        data.rotations[shell][:] = np.eye(3)
    for row in data.samples:
        row["observed_uv"] = row["source_uv"]
    report = validate_geometry(data, K, C, F, gap, fit_candidate=False)
    assert report["baseline"]["heldout"]["median_px"] < 1e-8
    assert report["evaluation_only"]
    assert "geometry_not_observable_without_priors" in report["rejection_reasons"]
    assert not report["candidate_config_permitted"]


def test_invalid_rays_are_not_dropped_from_scores(synthetic):
    K, F, C, gap, dataset = synthetic
    data = copy.deepcopy(dataset)
    for row in data.samples:
        row["source_uv"] = [2000., 2000.]
    report = validate_geometry(data, K, C, F, gap, max_nfev=2)
    assert report["baseline"]["heldout"]["count"] > 0
    assert report["baseline"]["heldout"]["physical_fraction"] == 0
    assert report["baseline"]["heldout"]["median_px"] >= 1000
    assert not report["candidate_config_permitted"]


def test_candidate_config_is_separate_and_preserves_initial_roll(tmp_path):
    from scripts.validate_geometry import _candidate_config

    config_path = tmp_path / "config.yaml"
    config = {"input": {"path": "video.mkv"}, "output": {"dir": "out/source"},
              "camera": {"K": np.eye(3).tolist()}, "circle": {"r_px": 250},
              "frame_calib": {"R_bc": np.eye(3).tolist(), "initial_roll_deg": 6.5},
              "mechanical": {"geometry": "separated_hemispheres", "gap_fraction": .1}}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    before = config_path.read_bytes()
    output = tmp_path / "candidate"
    output.mkdir()
    F = Rotation.from_rotvec([.02, -.01, .03]).as_matrix()
    report = {"candidate_config_permitted": False, "candidate_geometry": {
        "pivot_camera": [0., 0., 4.], "gap_fraction": .12, "measurement_frame": F.tolist()}}
    assert _candidate_config(config, config_path, output, report) is None
    assert not list(output.iterdir())
    report["candidate_config_permitted"] = True
    candidate_path = Path(_candidate_config(config, config_path, output, report))
    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    assert config_path.read_bytes() == before
    assert candidate["camera"] == config["camera"] and candidate["circle"] == config["circle"]
    assert candidate["frame_calib"]["initial_roll_deg"] == 6.5
    np.testing.assert_allclose(measurement_frame(candidate["frame_calib"]), F)
    assert candidate["input"]["path"] == str((tmp_path / "video.mkv").resolve())


def test_fixed_evidence_reuse_preserves_partition_and_rejects_missing_poses(synthetic, tmp_path):
    import json
    from types import SimpleNamespace
    from scripts.validate_geometry import _reuse_evidence

    K, F, C, gap, dataset = synthetic
    report = validate_geometry(dataset, K, C, F, gap, fit_candidate=False)
    count = len(dataset.valid["top"])
    times = np.arange(count)*.04
    clip = str(tmp_path / "video.mkv")
    report.update(coordinate_system="undistorted_pixels", source_clip=clip, K=K.tolist(), dist=[0]*5,
                  endpoint_times_s={str(i): float(t) for i, t in enumerate(times)})
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    archive = SimpleNamespace(K=K, frame_count=count, timestamps=times)
    repeated, _ = _reuse_evidence(path, archive, {"input": clip}, np.zeros(5), dataset.rotations, dataset.valid, F)
    for before, after in zip(dataset.samples, repeated.samples):
        for key in ("track_id", "source_frame", "target_frame", "source_uv", "observed_uv", "split"):
            assert before[key] == after[key]
    flags = copy.deepcopy(dataset.valid)
    row = dataset.samples[0]
    flags[row["shell"]][row["target_frame"]] = False
    with pytest.raises(ValueError, match="no valid new source pose"):
        _reuse_evidence(path, archive, {"input": clip}, np.zeros(5), dataset.rotations, flags, F)
    with pytest.raises(ValueError, match="same clip"):
        _reuse_evidence(path, archive, {"input": clip}, np.ones(5), dataset.rotations, dataset.valid, F)


def test_frozen_quality_screen_preserves_pairs_and_bad_top_does_not_corrupt_bottom(synthetic, tmp_path):
    import json
    from types import SimpleNamespace
    from scripts.validate_geometry import _reuse_evidence

    K, F, C, gap, dataset = synthetic
    evidence = validate_geometry(dataset, K, C, F, gap, fit_candidate=False)
    count = len(dataset.valid["top"])
    times = np.arange(count)*.04
    clip = str(tmp_path / "clip.mkv")
    evidence.update(coordinate_system="undistorted_pixels", source_clip=clip, K=K.tolist(), dist=[0]*5,
                    endpoint_times_s={str(i): float(t) for i, t in enumerate(times)})
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    rotations = copy.deepcopy(dataset.rotations)
    components = Rotation.from_matrix(F.T @ rotations["top"] @ F).as_euler("XYZ")
    components[1:, 0] += .7
    components[1:, 1] = .4  # raw-valid but far outside the common 5-degree tilt gate
    rotations["top"] = F @ Rotation.from_euler("XYZ", components).as_matrix() @ F.T
    archive = SimpleNamespace(K=K, frame_count=count, timestamps=times)
    repeated, _ = _reuse_evidence(path, archive, {"input": clip}, np.zeros(5), rotations, dataset.valid, F)
    assert len(repeated.samples) == len(dataset.samples)
    assert repeated.valid["bottom"].all() and not repeated.valid["top"][1:].any()
    report = validate_geometry(repeated, K, C, F, gap, fit_candidate=False)
    assert report["baseline"]["heldout_bottom"]["median_px"] < 1e-8
    assert report["baseline"]["heldout_bottom"]["source_quality_fraction"] == 1.
    assert report["baseline"]["heldout_top"]["source_quality_fraction"] == 0.
    assert report["baseline"]["heldout_top"]["median_px"] >= 1000
    assert "heldout_top_source_pose_quality_unavailable" in report["rejection_reasons"]
    assert report["baseline"]["heldout_top"]["count"] == evidence["baseline"]["heldout_top"]["count"]
