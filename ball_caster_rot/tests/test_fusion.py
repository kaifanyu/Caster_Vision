import json

import numpy as np
import pytest

from ballrot.fusion import visual_angle_observations


def source():
    return {
        "frames": {"time_s": [0, .04, .09]},
        "unconstrained": {"angle_unit": "radians", "frames": {
            "alpha_top": [0, .2, None], "alpha_bottom": [0, .22, .3],
            "beta_top": [0, -.3, None], "beta_bottom": [0, .1, .2],
            "gamma_top": [0, 0, None], "gamma_bottom": [0, 0, 0],
            "valid_top": [True, True, False], "valid_bottom": [True, True, True],
        }},
    }


def test_one_roll_measurement_keeps_noise_floor_and_independent_spins():
    obs = visual_angle_observations(source())
    np.testing.assert_allclose(obs.angles[:, 0], [0, .21, .3])
    assert np.isnan(obs.angles[2, 1])
    assert obs.angles[2, 2] == .2
    assert np.all(obs.std[:, 0] >= np.deg2rad(1.5))
    assert obs.std[1, 0] > obs.std[0, 0]


def test_conflict_rejects_common_roll_without_copying_shell_spin():
    payload = source()
    payload["unconstrained"]["frames"]["alpha_top"][1] = 1
    obs = visual_angle_observations(payload)
    assert np.isnan(obs.angles[1, 0])
    np.testing.assert_allclose(obs.angles[1, 1:], [-.3, .1])
    assert obs.diagnostics["roll_conflict"][1]


def test_tilt_gate_drops_own_shell_and_uses_remaining_roll():
    payload = source()
    payload["unconstrained"]["frames"]["gamma_top"][1] = .2
    obs = visual_angle_observations(payload)
    assert obs.angles[1, 0] == .22
    assert np.isnan(obs.angles[1, 1])
    assert obs.diagnostics["tilt_rejected_top"][1]


def test_preserves_unwrapped_turns_and_source_missing_flags():
    payload = source()
    raw = payload["unconstrained"]["frames"]
    raw["alpha_bottom"][2] = 4 * np.pi + .3
    raw["beta_top"][2] = 123.0  # stale numeric value is not a measurement
    obs = visual_angle_observations(payload)
    assert obs.angles[2, 0] == 4 * np.pi + .3
    assert np.isnan(obs.angles[2, 1])


def test_mechanical_source_preserves_shared_once_and_gaps():
    payload = {"mechanical": {"enabled": True}, "frames": {
        "time_s": [0, .1], "alpha_rad": [0, None],
        "beta_top_rad": [0, None], "beta_bottom_rad": [None, None],
        "valid_top": [True, False], "valid_bottom": [False, False],
    }}
    obs = visual_angle_observations(payload, source="mechanical")
    np.testing.assert_allclose(obs.angles[0, :2], 0)
    assert np.isnan(obs.angles[0, 2]) and np.all(np.isnan(obs.angles[1]))


@pytest.mark.parametrize("kind", ["units", "nonfinite", "timestamps", "flags", "noise"])
def test_invalid_measurement_provenance(kind):
    payload = source()
    options = {}
    if kind == "units":
        payload["unconstrained"]["angle_unit"] = "degrees"
    elif kind == "nonfinite":
        payload["unconstrained"]["frames"]["alpha_top"][1] = None
    elif kind == "timestamps":
        payload["frames"]["time_s"][2] = .04
    elif kind == "flags":
        payload["unconstrained"]["frames"]["valid_top"][0] = 1
    else:
        options["base_std_deg"] = 0
    with pytest.raises(ValueError):
        visual_angle_observations(payload, **options)


def test_cli_separate_schema_and_original_mechanical_validity(tmp_path):
    from scripts.fuse_motion import main

    payload = source()
    payload["metadata"] = {"frame_count": 3}
    payload["mechanical"] = {"enabled": True}
    payload["frames"].update({
        "valid_top": [True, False, False], "valid_bottom": [True, False, False],
        "alpha_rad": [0, None, None],
    })
    original = tmp_path / "original.json"
    original.write_text(json.dumps(payload), encoding="utf-8")
    before = original.read_bytes()
    destination = tmp_path / "fused"
    assert main(["--results", str(original), "--output", str(destination)]) == 0
    assert original.read_bytes() == before
    fused = json.loads((destination / "results.json").read_text(encoding="utf-8"))
    assert fused["method"] == "experimental_angle_kalman"
    assert fused["frames"]["mechanical_valid_top"] == [True, False, False]
    assert fused["frames"]["beta_top_status"][2] == "predicted"
    assert fused["frames"]["beta_top_vision_rad"][2] is None
    assert fused["frames"]["alpha_prior_rad"][0] is None
    assert (destination / "fusion_report.json").exists()
    assert (destination / "results.csv").exists()
    with pytest.raises(ValueError, match="separate output"):
        main(["--results", str(original), "--output", str(tmp_path)])
    assert original.read_bytes() == before


def test_cli_rejects_truncated_mechanical_reference_before_writing(tmp_path):
    from scripts.fuse_motion import main

    payload = source()
    payload["mechanical"] = {"enabled": True}
    payload["frames"].update({"valid_top": [True], "valid_bottom": [True], "alpha_rad": [0]})
    original = tmp_path / "original.json"
    original.write_text(json.dumps(payload), encoding="utf-8")
    destination = tmp_path / "fused"
    with pytest.raises(ValueError, match="mechanical reference"):
        main(["--results", str(original), "--output", str(destination)])
    assert not destination.exists()
