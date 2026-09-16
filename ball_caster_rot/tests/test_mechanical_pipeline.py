"""Integration checks for mechanical validity, provenance, and rate exports."""

import csv
import json

import numpy as np
import pytest

from ballrot.diagnostics import unconstrained_result, write_run_outputs
from ballrot.integrate import mechanical_motion
from ballrot.offline_observations import save_observations
from ballrot.pipeline import run_pipeline
from ballrot.rotation import Rx, Rz
from scripts import reorient_results
from scripts.run import _acceptance


def test_pipeline_does_not_present_unobserved_projected_poses_as_measurements():
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(4)]
    result = run_pipeline(
        frames, K=np.array([[100., 0, 32], [0, 100., 32], [0, 0, 1.]]),
        circle=(32, 32, 23), R_bc=np.eye(3), segment_config={"mode": "equator"},
        temporal_config={"enabled": True}, offline_config={"enabled": True},
        mechanical_config={"enabled": True},
    )
    assert result.mechanical["status"] == "insufficient_observations"
    for name in ("top", "bottom"):
        assert result.initial_valid[name][0]  # forward reference is preserved
        assert result.unconstrained_valid[name][0]
        assert not getattr(result, f"{name}_step_valid").any()
        assert all(value is None for value in getattr(result, f"{name}_increments"))
        assert np.isnan(getattr(result.motion, f"beta_{name}")).all()
    assert np.isnan(result.motion.alpha).all()


def test_pipeline_requires_pixel_evidence_and_calibration_for_mechanics():
    for kwargs in ({"R_bc": np.eye(3)}, {"offline_config": {"enabled": True}}):
        with pytest.raises(ValueError, match="requires offline.enabled=true and calibrated R_bc"):
            run_pipeline([np.zeros((64, 64, 3), np.uint8)],
                         K=np.array([[100., 0, 32], [0, 100., 32], [0, 0, 1.]]),
                         circle=(32, 32, 23), mechanical_config={"enabled": True}, **kwargs)


def test_exports_keep_missing_spin_and_raw_evidence_while_using_shared_roll(tmp_path):
    times = np.arange(9) * .05
    alpha = .3 * times
    beta = {"top": .5 * times, "bottom": -.4 * times}
    valid = {"top": np.ones(9, bool), "bottom": np.ones(9, bool)}
    valid["bottom"][4] = False
    motion = mechanical_motion(alpha, beta, valid)
    poses = {name: np.array([Rx(a) @ Rz(b) for a, b in zip(alpha, beta[name])])
             for name in beta}
    raw = unconstrained_result(motion, poses["top"], poses["bottom"], [])
    report = {"config": {"enabled": True, "gap_fraction": .02},
              "offline_config": {"rate_window_s": .25, "rate_polynomial_order": 2},
              "summary": {}, "frames": {name: [
                  {"frame_index": i, "status": "refined" if valid[name][i] else "unresolved"}
                  for i in range(9)] for name in beta}}
    paths = write_run_outputs(tmp_path, times, motion, [], poses["top"], poses["bottom"],
                              mechanical=report, unconstrained=raw)
    payload = json.loads(paths["json"].read_text())
    assert payload["frames"]["beta_bottom_rad"][4] is None
    assert payload["frames"]["beta_bottom_velocity_rad_s"][4] is None
    assert payload["frames"]["alpha_velocity_rad_s"][4] == pytest.approx(.3)
    assert "either shell" in payload["metadata"]["rate_processing"]["shared_roll_support"]
    assert "not independent accuracy" in payload["summary"]["constraint_metrics_note"]
    assert "unconstrained_checks" in payload["summary"]
    np.testing.assert_allclose(payload["unconstrained"]["top_absolute"], poses["top"])
    with paths["csv"].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[4]["bottom_mechanical_status"] == "unresolved"
    archive = save_observations(
        tmp_path / "observations.npz", {"top": [], "bottom": []}, timestamps=times,
        K=np.eye(3), center=np.array([0, 0, 4]), radius=1,
        initial_rotations=poses, initial_valid=valid,
        refined_rotations=poses, refined_valid=valid,
        unconstrained_rotations=poses, unconstrained_valid=valid)
    with np.load(archive, allow_pickle=False) as arrays:
        assert arrays["schema_version"] == 3
        np.testing.assert_array_equal(arrays["bottom_unconstrained_valid"], valid["bottom"])


def test_acceptance_does_not_treat_forced_axis_agreement_as_independent_evidence():
    quality = {"mean_inlier_residual_deg_max": .1, "inlier_ratio_min": .9,
               "forward_backward_error_px_max": .2, "tracked_count_min": 25,
               "success_rate": 1., "gamma_residual_deg_median": 0.}
    summary = {"top": quality, "bottom": quality,
               "alpha_top_bottom_disagreement_deg_median": 0.,
               "unconstrained_checks": {"top": {**quality, "gamma_residual_deg_median": 8.},
                                        "bottom": quality,
                                        "alpha_top_bottom_disagreement_deg_median": 12.},
               "offline_tracking": {"top": {"unresolved_frames": []}},
               "mechanical_tracking": {"top": {"unresolved_frames": [3]}}}
    passed, failures = _acceptance(summary, 8)
    assert not passed
    assert any("8.0 deg" in message for message in failures)
    assert any("12.0 deg" in message for message in failures)
    assert any("1 unresolved" in message for message in failures)


def test_reorient_requires_mechanical_refit_instead_of_breaking_constraints(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("assumptions:\n  camera_fixed_to_chassis: true\n"
                      "  ball_center_stationary_in_image: true\n  two_speckle_colors: true\n")
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"metadata": {"mechanical_model": {"enabled": True}}}))
    with pytest.raises(ValueError, match="scripts/refine_mechanical.py"):
        reorient_results.main(["--config", str(config), "--results", str(results),
                               "--output", str(tmp_path / "refit")])
