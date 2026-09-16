"""Saved-observation refitting must preserve provenance and measurement gaps."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from ballrot.diagnostics import FrameQuality, HemisphereQuality, _json_clean
from ballrot.integrate import decompose_hemispheres
from ballrot.offline import SurfaceObservation
from ballrot.offline_observations import save_observations
from ballrot.rotation import Rx, Rz
from ballrot.sphere import sphere_pose_from_circle
from scripts import refine_mechanical


def _fixture(tmp_path: Path):
    count = 8
    times = np.array([0, .033, .066, .104, .137, .170, .208, .241])
    frame = Rotation.from_euler("xyz", [45, -15, 30], degrees=True).as_matrix()
    alpha = np.linspace(0, .12, count)
    beta = {"top": np.linspace(0, .2, count), "bottom": np.linspace(0, -.17, count)}
    poses = {name: np.array([frame @ Rx(a) @ Rz(b) @ frame.T for a, b in zip(alpha, values)])
             for name, values in beta.items()}
    valid = {name: np.ones(count, dtype=bool) for name in beta}
    motion = decompose_hemispheres(poses["top"], poses["bottom"], frame,
                                  valid_top=valid["top"], valid_bottom=valid["bottom"])
    frames = {"time_s": times.tolist(), "alpha_rad": motion.alpha.tolist()}
    for name in beta:
        for component in ("alpha", "gamma", "beta"):
            frames[f"{component}_{name}_rad"] = getattr(motion, f"{component}_{name}").tolist()
        frames[f"valid_{name}"] = valid[name].tolist()
    qualities = [asdict(FrameQuality(index, *[
        HemisphereQuality(80, 75, 70, 70 / 75, .1, .09, .2, True) for _ in beta
    ])) for index in range(1, count)]
    K = np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1]], dtype=float)
    circle = (320., 240., 180.)
    center, radius = sphere_pose_from_circle(*circle, K)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_results = source_dir / "results.json"
    payload = {
        "metadata": {"R_bc": frame.tolist(), "frame_count": count, "K": K.tolist(),
                     "dist": [0, 0, 0, 0], "circle": circle,
                     "input": str(tmp_path / "clip.mkv"), "fps": 30,
                     "timing_source": "native_video_pts"},
        "frames": frames, "quality": qualities,
        "temporal": {"config": {"enabled": True}, "summary": {"source": "forward"}},
        "offline": {"config": {"enabled": True, "rate_window_s": .25},
                    "summary": {name: {"unresolved_frames": []} for name in beta}},
        "summary": {"temporal_tracking": {"source": "forward"},
                    "offline_tracking": {name: {"unresolved_frames": []} for name in beta}},
    }
    source_results.write_text(json.dumps(_json_clean(payload)), encoding="utf-8")
    observations = {name: [SurfaceObservation(i, 1, np.array([320 + i, 220.]), 1)
                           for i in range(count)] for name in beta}
    archive_path = source_dir / "offline_observations.npz"
    save_observations(
        archive_path, observations, timestamps=times, K=K, center=center, radius=radius,
        initial_rotations=poses, initial_valid=valid, refined_rotations=poses, refined_valid=valid,
        landmark_sources={1: ("adjacent", 0, 12)},
    )
    config = {
        "input": {"path": str(tmp_path / "clip.mkv")},
        "camera": {"K": K.tolist(), "dist": [0, 0, 0, 0]},
        "circle": dict(zip(("u0", "v0", "r_px"), circle)),
        "frame_calib": {"R_bc": frame.tolist()},
        "mechanical": {"enabled": False, "gap_fraction": .02},
        "offline": {"enabled": True, "rate_window_s": .15},
        "assumptions": {"camera_fixed_to_chassis": True, "ball_center_stationary_in_image": True,
                        "two_speckle_colors": True},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return SimpleNamespace(
        source=source_results, archive=archive_path, payload=payload, config_path=config_path,
        config=config, poses=poses, valid=valid, alpha=alpha, beta=beta, frame=frame, times=times,
    )


def _rewrite_archive(path, mutate):
    with np.load(path, allow_pickle=False) as original:
        arrays = {key: original[key].copy() for key in original.files}
    mutate(arrays)
    np.savez_compressed(path, **arrays)


@pytest.mark.parametrize("mutation,match", [
    (lambda a: a.update(schema_version=np.array(1)), "schema 2"),
    (lambda a: a.pop("landmark_family"), "missing identity"),
    (lambda a: a["landmark_source_frame"].__setitem__(0, 999), "identity provenance"),
    (lambda a: a["top_track_id"].__setitem__(0, 999), "identities"),
    (lambda a: a["timestamps_s"].__setitem__(2, .08), "timestamps.*differs"),
    (lambda a: a["K"].__setitem__((0, 0), 710), "archive K.*differs"),
    (lambda a: a["sphere_center"].__setitem__(0, .1), "sphere_center.*differs"),
    (lambda a: a["top_refined_rotations"].__setitem__(2, Rx(.5)), "refined rotations.*differs"),
    (lambda a: a["top_refined_valid"].__setitem__(2, False), "refined validity.*differs"),
    (lambda a: a.update(top_refined_valid=a["top_refined_valid"].astype(int)), "boolean flags"),
])
def test_mismatched_or_legacy_archives_are_rejected(tmp_path, mutation, match):
    data = _fixture(tmp_path)
    _rewrite_archive(data.archive, mutation)
    with pytest.raises(ValueError, match=match):
        refine_mechanical.load_archive(data.archive, data.payload)


def test_valid_archive_preserves_identity_timestamps_and_raw_arrays(tmp_path):
    data = _fixture(tmp_path)
    loaded = refine_mechanical.load_archive(data.archive, data.payload)
    np.testing.assert_array_equal(loaded.timestamps, data.times)
    np.testing.assert_array_equal(loaded.unconstrained_rotations["top"], data.poses["top"])
    assert loaded.landmark_sources == {1: ("adjacent", 0, 12)}
    assert loaded.observations["bottom"][3].frame_index == 3


def test_refit_refuses_source_overwrite_before_reading_it(tmp_path):
    data = _fixture(tmp_path)
    with pytest.raises(ValueError, match="originals are preserved"):
        refine_mechanical.main([
            "--config", str(data.config_path), "--results", str(data.source),
            "--output", str(data.source.parent),
        ])


def test_configured_geometry_change_requires_new_observations(tmp_path):
    data = _fixture(tmp_path)
    data.config["circle"]["r_px"] += 1
    with pytest.raises(ValueError, match="configured circle.*differs"):
        refine_mechanical._same_geometry(data.config, data.config_path, data.payload["metadata"])


def test_cli_preserves_sources_exports_gaps_and_can_refit_changed_frame(tmp_path, monkeypatch, capsys):
    data = _fixture(tmp_path)
    data.config["frame_calib"]["initial_roll_deg"] = 8
    data.config_path.write_text(yaml.safe_dump(data.config), encoding="utf-8")
    originals = {p: p.read_bytes() for p in (data.source, data.archive, data.config_path)}
    output = tmp_path / "joint"
    expected_frame = data.frame @ Rx(np.deg2rad(8))

    def solver(observations, rotations, valid, K, C, F, *, radius, config, offline_config, top_shell_sign):
        assert config.enabled  # Explicit refit command overrides disabled pipeline toggle.
        assert config.gap_fraction == .02
        assert offline_config.rate_window_s == .15
        np.testing.assert_allclose(F, expected_frame)
        np.testing.assert_array_equal(rotations["top"], data.poses["top"])
        assert len(observations["top"]) == len(data.times)
        final_valid = {name: values.copy() for name, values in valid.items()}
        final_valid["bottom"][3] = False
        poses = {name: np.array([F @ Rx(a) @ Rz(b) @ F.T for a, b in zip(data.alpha, beta)])
                 for name, beta in data.beta.items()}
        diagnostics = {
            "config": asdict(config), "offline_config": asdict(offline_config),
            "model": "shared_roll_independent_spin",
            "summary": {name: {"valid_frames": int(mask.sum()),
                               "unresolved_frames": np.flatnonzero(~mask).tolist()}
                        for name, mask in final_valid.items()},
            "frames": {name: [{"frame_index": i, "status": "accepted" if good else "unresolved"}
                              for i, good in enumerate(mask)] for name, mask in final_valid.items()},
        }
        return SimpleNamespace(rotations=poses, valid=final_valid, alpha=data.alpha,
                               beta=data.beta, diagnostics=diagnostics)

    monkeypatch.setattr(refine_mechanical, "refine_mechanical_trajectory", solver)
    assert refine_mechanical.main([
        "--config", str(data.config_path), "--results", str(data.source), "--output", str(output),
    ]) == 0
    result = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert result["frames"]["valid_bottom"][3] is False
    assert result["frames"]["beta_bottom_rad"][3] is None
    assert result["frames"]["valid_top"][3] is True
    assert result["metadata"]["mechanical_model"]["gap_fraction"] == .02
    np.testing.assert_allclose(result["metadata"]["R_bc"], expected_frame)
    np.testing.assert_array_equal(result["frames"]["time_s"], data.times)
    assert result["quality"] == data.payload["quality"]
    assert result["offline"] == data.payload["offline"]
    assert result["temporal"] == data.payload["temporal"]
    assert result["summary"]["mechanical_tracking"]["bottom"]["unresolved_frames"] == [3]
    assert result["summary"]["offline_tracking"]["bottom"]["unresolved_frames"] == []
    assert result["metadata"]["rate_processing"]["window_s"] == .15
    assert result["unconstrained"]["frames"]["valid_bottom"][3] is True
    saved = refine_mechanical.load_archive(output / "offline_observations.npz", result)
    np.testing.assert_array_equal(saved.unconstrained_rotations["bottom"], data.poses["bottom"])
    assert (output / "mechanical_report.json").is_file()
    assert (output / "results.csv").is_file()
    assert "simulate_measured.py" in capsys.readouterr().out
    for path, before in originals.items():
        assert path.read_bytes() == before


def test_mechanical_source_requires_unconstrained_archive_for_repeated_fit(tmp_path):
    data = _fixture(tmp_path)
    data.payload["mechanical"] = {"config": {"enabled": True}}
    with pytest.raises(ValueError, match="lacks preserved unconstrained"):
        refine_mechanical.load_archive(data.archive, data.payload)
