"""Window fits preserve a measured gauge and never invent a hidden shell spin."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ballrot.mechanical import MechanicalConfig, refine_mechanical_trajectory
from ballrot.offline import OfflineConfig, SurfaceObservation
import ballrot.mechanical_windows as windows_module
from test_mechanical import K, C, F, scene, angular_error, assert_mechanics


def fit(data, initial, valid, *, window=9, overlap=3, observation_provider=None):
    return refine_mechanical_trajectory(
        data, initial, valid, K, C, F,
        config=MechanicalConfig(enabled=True, gap_fraction=.1,
                                window_frames=window, overlap_frames=overlap),
        offline_config=OfflineConfig(enabled=True, max_nfev=80),
        observation_provider=observation_provider)


def test_nonidentity_calibration_and_nonzero_overlap_angles_keep_global_pose():
    data, truth, initial, valid, alpha, beta = scene(count=24, noise=.01)
    result = fit(data, initial, valid)
    assert len(result.diagnostics["windows"]) >= 3
    assert result.diagnostics["status"] == "completed", result.diagnostics
    assert_mechanics(result)
    for name in data:
        assert result.valid[name].all(), result.diagnostics
        assert np.max(angular_error(result.rotations[name], truth[name])) < .07
        assert np.max(np.abs(result.beta[name] - beta[name])) < np.deg2rad(.07)
    assert np.max(np.abs(result.alpha-alpha)) < np.deg2rad(.07)
    assert result.diagnostics["windows"][1]["start_frame"] > 0
    json.dumps(result.diagnostics, allow_nan=False)


def test_bad_middle_images_do_not_erase_earlier_poses_or_observed_later_motion():
    data, truth, initial, valid, _, _ = scene(count=30, noise=.01)
    rng = np.random.default_rng(894)
    for name in data:
        data[name] = [replace(entry, uv=entry.uv+rng.normal(0, 18, 2))
                      if entry.frame_index in (10, 11) else entry for entry in data[name]]
    result = fit(data, initial, valid, window=10, overlap=4)
    for name in data:
        assert result.valid[name][:10].all(), result.diagnostics
        assert not result.valid[name][10:12].any()
        assert result.valid[name][12:].all(), result.diagnostics
        assert np.max(angular_error(result.rotations[name], truth[name])[result.valid[name]]) < .15
    assert_mechanics(result)


def test_disconnected_late_images_do_not_start_a_new_global_reference():
    data, _, initial, valid, _, _ = scene(count=30)
    for name in data:
        data[name] = [replace(entry, track_id=entry.track_id+(1000 if entry.frame_index >= 22 else 0))
                      for entry in data[name] if entry.frame_index < 8 or entry.frame_index >= 22]
    result = fit(data, initial, valid, window=10, overlap=3)
    for name in data:
        assert result.valid[name][:8].all(), result.diagnostics
        assert not result.valid[name][8:].any()
        assert result.diagnostics["gaps"][name][0]["start_frame"] == 8
    assert result.diagnostics["status"] == "completed_with_unresolved_intervals"
    assert result.diagnostics["scan_complete"]
    assert any(attempt["end_frame_exclusive"] == 30 for attempt in result.diagnostics["windows"])
    assert len(result.diagnostics["windows"]) <= 24
    assert all(entry["reason"] != "not_reached_by_measured_window"
               for entries in result.diagnostics["frames"].values() for entry in entries)


def test_same_landmarks_relocalize_after_a_gap_longer_than_the_window():
    data, truth, initial, valid, _, _ = scene(count=36, noise=.01)
    for name in data:
        data[name] = [entry for entry in data[name]
                      if entry.frame_index < 8 or entry.frame_index >= 24]
        valid[name][24:] = False
        initial[name][24:] = np.eye(3)
    result = fit(data, initial, valid, window=10, overlap=3)
    for name in data:
        assert result.valid[name][:8].all(), result.diagnostics
        assert not result.valid[name][8:24].any()
        assert result.valid[name][24:].all(), result.diagnostics
        assert np.max(angular_error(result.rotations[name], truth[name])[24:]) < .15
        assert result.landmarks[name]
    assert result.diagnostics["scan_complete"]
    for component in result.diagnostics["components"]:
        attempt = result.diagnostics["windows"][component["window_index"]]
        assert all(attempt["start_frame"] <= index < attempt["end_frame_exclusive"]
                   for index in component["map_reference_frames"])
    for name in data:
        for entry in result.diagnostics["frames"][name]:
            if "bridge" in entry:
                attempt = result.diagnostics["windows"][entry["window_index"]]
                assert all(attempt["start_frame"] <= link["reference_frame"] < attempt["end_frame_exclusive"]
                           for link in entry["bridge"]["direct_reference_links"])
    assert_mechanics(result)


def test_one_shot_observation_iterators_are_materialized_before_validation():
    data, _, initial, valid, _, _ = scene(count=6)
    expected = fit(data, initial, valid, window=6, overlap=2)
    actual = fit({name: iter(entries) for name, entries in data.items()},
                 initial, valid, window=6, overlap=2)
    for name in data:
        assert np.array_equal(actual.valid[name], expected.valid[name])
        assert np.allclose(actual.rotations[name], expected.rotations[name])


def test_lost_shell_reference_cannot_be_replaced_by_other_shell_roll():
    data, _, initial, valid, _, _ = scene(count=24)
    data["bottom"] = [replace(entry, track_id=entry.track_id+(1000 if entry.frame_index >= 12 else 0))
                      for entry in data["bottom"] if entry.frame_index < 8 or entry.frame_index >= 12]
    result = fit(data, initial, valid)
    assert result.valid["top"].all(), result.diagnostics
    assert result.valid["bottom"][:8].all()
    assert not result.valid["bottom"][8:].any()
    assert np.isnan(result.beta["bottom"][8:]).all()
    assert np.isfinite(result.alpha).all()
    assert_mechanics(result)


def test_window_recovery_requires_observations_and_preserves_missing_shell_flags():
    data, truth, initial, valid, _, _ = scene(count=24)
    for name in data:
        valid[name][[10, 12]] = False
        initial[name][10] = initial[name][9]
        data[name] = [entry for entry in data[name] if entry.frame_index != 12]
    result = fit(data, initial, valid)
    for name in data:
        assert result.valid[name][10], result.diagnostics
        assert result.diagnostics["frames"][name][10]["status"] == "recovered"
        assert angular_error(result.rotations[name], truth[name])[10] < .1
        assert not result.valid[name][12]
    assert_mechanics(result)


def test_no_measured_first_image_cannot_acquire_origin_from_raw_later_poses():
    data, _, initial, valid, _, _ = scene(count=12)
    data = {name: [entry for entry in entries if entry.frame_index != 0]
            for name, entries in data.items()}
    result = fit(data, initial, valid)
    assert not result.valid["top"].any()
    assert not result.valid["bottom"].any()


def _scheduled_fit(monkeypatch, decide, *, count=24, window=10, overlap=3,
                   observation_provider=None):
    """Isolate scheduling from numerical fitting with controllable image support."""
    names = ("top", "bottom")
    data = {name: [SurfaceObservation(index, 7, np.array([float(index), 0.]))
                   for index in range(count)] for name in names}
    initial = {name: np.repeat(np.eye(3)[None], count, axis=0) for name in names}
    valid = {name: np.ones(count, bool) for name in names}
    calls = []

    def batch(observations, poses, initial_valid, *args, config, **kwargs):
        n = len(poses["top"])
        if not config.enabled:
            return SimpleNamespace(rotations=poses, valid=initial_valid,
                                   diagnostics={"summary": {}})
        start = int(observations["top"][0].uv[0])
        call = {"start": start, "end": start+n,
                "references": kwargs["fixed_references"],
                "observation_ids": {name: {(start+entry.frame_index, entry.track_id) for entry in entries}
                                    for name, entries in observations.items()},
                "landmarks": {name: dict(points) for name, points in kwargs["fixed_landmarks"].items()}}
        calls.append(call)
        accepted, maps = decide(call)
        output_valid = {name: np.array([index in accepted[name] for index in range(start, start+n)])
                        for name in names}
        # Distinct values reveal accidental changes to an already published
        # shared roll when only the other shell is newly recovered.
        alpha = np.full(n, float(len(calls))/100)
        beta = {name: np.full(n, float(len(calls))/50) for name in names}
        reports = {name: [{"frame_index": index,
                           "status": "refined" if output_valid[name][index] else "unresolved",
                           "reason": "accepted" if output_valid[name][index] else "final_frame_quality_failed",
                           "observation_count": 100, "inlier_count": 80 if output_valid[name][index] else 40,
                           "inlier_fraction": .8 if output_valid[name][index] else .4,
                           "after": {"median_px": .5 if output_valid[name][index] else 3.},
                           "reprojection": [{"frame": start+index, "attempt": len(calls)}]}
                          for index in range(n)] for name in names}
        if call.get("omit_failed_measurements"):
            for entries in reports.values():
                for entry in entries:
                    if entry["status"] == "unresolved":
                        for key in ("observation_count", "inlier_count", "inlier_fraction", "after", "reprojection"):
                            entry.pop(key)
                        entry["reason"] = "no_measured_overlap_reference"
        return SimpleNamespace(valid=output_valid, alpha=alpha, beta=beta, landmarks=maps,
                               diagnostics={"frames": reports, "status": "completed",
                                            "components": [],
                                            "selected_observation_count": {name: n for name in names}})

    monkeypatch.setattr(windows_module, "_refine_mechanical_batch", batch)
    return fit(data, initial, valid, window=window, overlap=overlap,
               observation_provider=observation_provider), calls


def test_retries_keep_the_forward_target_inside_the_attempt(monkeypatch):
    def decide(call):
        return {name: set(range(7)) for name in ("top", "bottom")}, {}

    result, _ = _scheduled_fit(monkeypatch, decide)
    attempts = result.diagnostics["windows"]
    assert any(attempt["retry"] for attempt in attempts)
    for attempt in attempts:
        for target in attempt["target_frames"].values():
            assert attempt["start_frame"] <= target < attempt["end_frame_exclusive"]
    assert result.diagnostics["scan_complete"]


def test_lagging_shell_keeps_its_reference_during_catch_up(monkeypatch):
    def decide(call):
        bottom = set(range(5))
        # Only an interval retaining frame 4 can supply bottom's continuation.
        if call["start"] == 2:
            assert 4-call["start"] in call["references"]["bottom"]
            bottom.update(range(call["start"], call["end"]))
        return {"top": set(range(call["start"], call["end"])), "bottom": bottom}, {}

    result, calls = _scheduled_fit(monkeypatch, decide, count=18)
    catch_up = [attempt for attempt in result.diagnostics["windows"]
                if attempt["kind"] == "shell_catch_up"]
    assert catch_up and catch_up[0]["target_frames"] == {"bottom": 5}
    assert 4 in catch_up[0]["reference_frames"]["bottom"]
    assert result.valid["bottom"][5:10].all()
    assert np.all(result.alpha[:10] == .01)
    assert result.diagnostics["frames"]["top"][8]["window_index"] == 0
    assert result.diagnostics["frames"]["top"][8]["reprojection"][0]["attempt"] == 1
    assert all("reprojection" not in entry for attempt in result.diagnostics["windows"]
               for frames in attempt["fit"]["frames"].values() for entry in frames)
    assert all(8-call["start"] in call["references"]["alpha"]
               for call in calls[1:] if call["start"] <= 8 < call["end"])


def test_scan_passes_trusted_map_to_later_windows_without_overlap(monkeypatch):
    point = np.array([.2, -.8, .56])

    def decide(call):
        supported = set(range(7))
        if call["start"] >= 14:
            assert not call["references"]["top"]
            assert np.array_equal(call["landmarks"]["top"][7], point)
            supported.update(range(call["start"], call["end"]))
        maps = {name: {7: point.copy()} for name in ("top", "bottom")}
        return {name: supported for name in ("top", "bottom")}, maps

    result, _ = _scheduled_fit(monkeypatch, decide)
    assert not result.valid["top"][7:14].any()
    assert result.valid["top"][14:].all()
    assert result.valid["bottom"][14:].all()
    assert result.diagnostics["trusted_landmark_counts"] == {"top": 1, "bottom": 1}


def test_later_missing_reference_does_not_erase_measured_failure(monkeypatch):
    def decide(call):
        call["omit_failed_measurements"] = call["start"] > 0
        return {name: set(range(7)) for name in ("top", "bottom")}, {}

    result, _ = _scheduled_fit(monkeypatch, decide)
    for name in ("top", "bottom"):
        frame = result.diagnostics["frames"][name][8]
        assert frame["reason"] == "final_frame_quality_failed"
        assert frame["window_index"] == 0
        assert frame["inlier_count"] == 40
        assert frame["reprojection"][0]["attempt"] == 1


def test_image_provider_proposals_enter_fit_with_existing_map_ids(monkeypatch):
    class Provider:
        diagnostics = {"enabled": True}

        def propose(self, start, end, landmarks, alpha, beta, measured, by_frame, reports):
            assert all(9 in landmarks[name] for name in ("top", "bottom"))
            assert reports["top"][0]["reason"] == "accepted"
            return {name: [SurfaceObservation(index, 9, np.array([float(index), 1.]))
                           for index in range(start, end) if not measured[name][index]]
                    for name in ("top", "bottom")}

    def decide(call):
        accepted = {name: set(range(5)) | {index for index, identifier in call["observation_ids"][name]
                                          if identifier == 9} for name in ("top", "bottom")}
        maps = {name: {9: np.array([0., 0., 1.])} for name in accepted}
        return accepted, maps

    result, calls = _scheduled_fit(monkeypatch, decide, count=18, observation_provider=Provider())
    assert not any(identifier == 9 for _, identifier in calls[0]["observation_ids"]["top"])
    assert result.valid["top"].all()
    assert result.valid["bottom"].all()
    assert sum(attempt["rematched_observations"]["top"] for attempt in result.diagnostics["windows"]) == 13
    assert result.diagnostics["image_rematching"]["enabled"]


@pytest.mark.parametrize("settings", [
    {"window_frames": True}, {"window_frames": -1}, {"window_frames": 2, "overlap_frames": 1},
    {"overlap_frames": 3}, {"window_frames": 10}, {"window_frames": 10, "overlap_frames": 10},
    {"window_frames": 10, "overlap_frames": 1.5},
])
def test_window_config_rejects_ambiguous_or_invalid_overlap(settings):
    with pytest.raises(ValueError):
        MechanicalConfig.from_mapping(settings)
