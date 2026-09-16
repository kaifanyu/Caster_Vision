"""Rigid cap separation and faithful rendering of constrained measurements."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ballrot.rotation import Rx, Rz
from scripts import simulate_measured as replay
from synthetic import generate


def measured_series(gap: float = 0.08) -> dict:
    alpha = np.array([0.0, 0.25, 0.6, 0.8])
    result = {
        "time_s": np.arange(4) / 30.0,
        "frame_index": np.arange(4),
        "alpha_rad": alpha,
        "beta_top_rad": np.array([0.0, 0.4, 0.8, 1.2]),
        "beta_bottom_rad": np.array([0.0, -0.3, -0.6, -0.9]),
        "metadata": {"mechanical_model": {
            "enabled": True, "gap_fraction": gap,
            "model": "shared_roll_independent_spin",
        }},
        "mechanical": {"config": {"enabled": True, "gap_fraction": gap}},
    }
    for shell in ("top", "bottom"):
        result[f"alpha_{shell}_rad"] = alpha.copy()
        result[f"gamma_{shell}_rad"] = np.zeros(4)
        result[f"valid_{shell}"] = np.ones(4, dtype=bool)
    return result


@pytest.mark.parametrize("alpha", [-2.1, 0.0, 1.3])
@pytest.mark.parametrize("gap", [0.0, 0.08, 0.3])
def test_cap_geometry_keeps_constant_separation_under_shared_roll_and_counter_spin(alpha, gap):
    radius = 2.7
    center = np.array([0.3, -0.4, 9.0])
    frame = Rx(0.15) @ Rz(-0.37)
    normal = frame @ Rx(alpha)[:, 2]
    rims = []
    for sign, beta in ((1, 1.1), (-1, -2.4)):
        curves = replay._graticule(30.0, sign, gap_fraction=gap)
        points = np.concatenate(curves)
        transform = frame @ Rx(alpha) @ Rz(beta)
        world = center + radius * (points @ transform.T)
        coordinates = (world - center) @ normal
        assert np.min(sign * coordinates) >= radius * gap - 1e-12
        rim = world[np.isclose(sign * points[:, 2], gap, atol=1e-12)]
        assert len(rim) > 30
        np.testing.assert_allclose((rim - center) @ normal, sign * radius * gap, atol=1e-12)
        rims.append(rim)
    separation = (rims[0].mean(axis=0) - rims[1].mean(axis=0)) @ normal
    assert separation == pytest.approx(2.0 * radius * gap, abs=1e-12)


@pytest.mark.parametrize("sign", [-1, 1])
def test_probes_do_not_enter_the_rim_gap(sign):
    camera = generate.default_camera()
    points = replay._probe_points(camera.R_bc, sign, count=30, gap_fraction=0.2)
    assert len(points) == 30
    assert np.all(sign * points[:, 2] >= 0.2)


@pytest.mark.parametrize("swap", [False, True])
def test_missing_spin_keeps_shared_roll_without_creating_a_measurement(swap):
    measured = measured_series()
    measured["valid_top"][1:3] = False
    for component in ("alpha", "gamma", "beta"):
        measured[f"{component}_top_rad"][1:3] = np.nan
    model, full = replay._trajectories(measured, 30.0, swap_shells=swap)
    np.testing.assert_allclose(full.alpha_top, measured["alpha_rad"])
    np.testing.assert_allclose(full.alpha_bottom, measured["alpha_rad"])
    np.testing.assert_array_equal(full.gamma_top, np.zeros(4))
    np.testing.assert_array_equal(full.gamma_bottom, np.zeros(4))
    held = full.beta_bottom if swap else full.beta_top
    np.testing.assert_allclose(held, [0.0, 0.0, 0.0, 1.2])
    assert np.isnan(measured["beta_top_rad"][1:3]).all()
    np.testing.assert_array_equal(replay._pose_validity(measured, "top"), [True, False, False, True])
    np.testing.assert_allclose(model.alpha, full.alpha)


@pytest.mark.parametrize("component", ["alpha_top_rad", "alpha_bottom_rad", "gamma_bottom_rad", "alpha_rad"])
def test_renderer_rejects_impossible_poses_labelled_as_constrained(component):
    measured = measured_series()
    measured[component][2] += 0.05
    with pytest.raises(ValueError, match="violate shared roll"):
        replay._trajectories(measured, 30.0)


def test_saved_gap_cannot_be_changed_only_for_rendering():
    measured = measured_series(0.06)
    model = replay._result_mechanical_model({"mechanical": {"gap_fraction": 0.06}}, measured)
    assert model["gap_fraction"] == 0.06
    with pytest.raises(ValueError, match="gap differs"):
        replay._result_mechanical_model({"mechanical": {"gap_fraction": 0.07}}, measured)
    old_results = {"metadata": {}}
    assert replay._result_mechanical_model({"mechanical": {"enabled": True}}, old_results) is None


def test_loader_checks_saved_constraint_after_frame_selection(tmp_path):
    measured = measured_series()
    payload = {
        "metadata": measured["metadata"], "mechanical": measured["mechanical"],
        "frames": {key: value.tolist() for key, value in measured.items()
                   if isinstance(value, np.ndarray) and key != "frame_index"},
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = replay._measured(path, frames=None, stride=2)
    assert replay._mechanical_model(loaded)["gap_fraction"] == 0.08
    np.testing.assert_array_equal(loaded["frame_index"], [0, 2])
    payload["frames"]["gamma_top_rad"][2] = 0.2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="violate shared roll"):
        replay._measured(path, frames=None, stride=2)


def test_rendered_gap_is_background_and_moves_with_roll_not_spin():
    camera = generate.default_camera((96, 128))
    config = generate.RenderConfig(
        camera=camera, gap_fraction=0.12, draw_yoke=False,
        num_speckles=400, dot_radius_px=3.0,
        degradations=generate.Degradations(glare=True),
    )
    alpha = np.array([0.0, 0.3, -0.4])
    trajectory = generate.Trajectory(alpha, [0.0, 1.2, -0.8], [0.0, -0.9, 1.4])
    rendered = generate.render_sequence(None, trajectory, config)
    masks = []
    for angle, image in zip(alpha, rendered.frames):
        normal = camera.R_bc @ Rx(angle)[:, 2]
        gap = np.abs(config._front_normals @ normal) < config.gap_fraction
        assert gap.sum() > 30
        np.testing.assert_array_equal(
            image[gap], np.broadcast_to(config.background_color_bgr, (gap.sum(), 3))
        )
        masks.append(gap)
    assert not np.array_equal(masks[0], masks[1])
    assert rendered.ground_truth["rim_plane_separation"] == pytest.approx(0.24)
    points, _, _, retained = generate._speckle_attributes(config)
    assert np.all(np.abs(points[retained, 2]) >= config.gap_fraction)


def test_fixed_gap_refuses_independent_roll_in_synthetic_renderer():
    trajectory = generate.Trajectory(
        [0.0, 0.2], [0.0, 0.1], [0.0, -0.1], alpha_bottom=[0.0, 0.3],
    )
    config = generate.RenderConfig(camera=generate.default_camera((64, 80)), gap_fraction=0.1)
    with pytest.raises(ValueError, match="fixed cap gap requires shared roll"):
        generate.render_sequence(None, trajectory, config)


def test_zero_gap_preserves_existing_synthetic_render():
    trajectory = generate.Trajectory([0.0, 0.1], [0.0, 0.2], [0.0, -0.3])
    camera = generate.default_camera((64, 80))
    default = generate.render_sequence(None, trajectory, generate.RenderConfig(camera=camera))
    explicit = generate.render_sequence(None, trajectory, generate.RenderConfig(camera=camera, gap_fraction=0.0))
    for first, second in zip(default.frames, explicit.frames):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("gap", [-0.1, 1.0, float("nan")])
def test_invalid_gap_is_rejected(gap):
    with pytest.raises(ValueError, match="gap_fraction"):
        generate.RenderConfig(gap_fraction=gap)
    with pytest.raises(ValueError, match="gap_fraction"):
        replay._graticule(45.0, 1, gap_fraction=gap)

