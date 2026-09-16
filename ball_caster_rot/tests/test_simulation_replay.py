"""Replay regression gates for per-shell orientation and saved-frame consistency."""

from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ballrot.config import measurement_frame
from scripts import simulate_measured
from synthetic import generate


def _measured() -> dict[str, np.ndarray]:
    t = np.linspace(0.0, 1.0, 5)
    return {
        "alpha_rad": 0.3 * t,
        "alpha_top_rad": 0.4 * t,
        "alpha_bottom_rad": 0.2 * t,
        "beta_top_rad": 0.7 * t,
        "beta_bottom_rad": -0.6 * t,
        "gamma_top_rad": 0.09 * t,
        "gamma_bottom_rad": -0.07 * t,
    }


@pytest.mark.parametrize("swap", [False, True])
def test_full_replay_preserves_each_shell_and_model_uses_shared_roll(swap: bool) -> None:
    measured = _measured()
    model, full = simulate_measured._trajectories(measured, fps=29.0, swap_shells=swap)

    for rendered_name, source_name in (
        ("top", "bottom" if swap else "top"),
        ("bottom", "top" if swap else "bottom"),
    ):
        for component in ("alpha", "gamma", "beta"):
            np.testing.assert_array_equal(
                getattr(full, f"{component}_{rendered_name}"),
                measured[f"{component}_{source_name}_rad"],
            )
        np.testing.assert_array_equal(getattr(model, f"alpha_{rendered_name}"), measured["alpha_rad"])
        np.testing.assert_array_equal(getattr(model, f"gamma_{rendered_name}"), np.zeros(5))
        np.testing.assert_array_equal(
            getattr(model, f"beta_{rendered_name}"), measured[f"beta_{source_name}_rad"]
        )

    # Check matrices as well as fields: swapping must move the entire shell
    # orientation, preserving both the residual and the distinct roll value.
    top_matrices, bottom_matrices = generate._matrix_series(full)
    for actual, source in (
        (top_matrices, "bottom" if swap else "top"),
        (bottom_matrices, "top" if swap else "bottom"),
    ):
        expected = Rotation.from_euler(
            "XYZ", np.column_stack([measured[f"{angle}_{source}_rad"] for angle in ("alpha", "gamma", "beta")])
        ).as_matrix()
        np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_renderer_and_speed_scaling_use_separate_shell_rolls(monkeypatch: pytest.MonkeyPatch) -> None:
    _, full = simulate_measured._trajectories(_measured(), fps=29.0)
    scaled = full.scaled(1.7)
    config = generate.RenderConfig(camera=generate.default_camera((64, 80)), draw_yoke=False)
    observed: list[np.ndarray] = []

    def record_population(image, points, indices, radii, brightness, R_motion, colour, config, rng):
        observed.append(R_motion.copy())

    monkeypatch.setattr(generate, "_draw_population", record_population)
    # An exposure subframe exercises the renderer's actual interpolation path.
    frame_position = 1.5
    generate._render_subframe(
        frame_position, scaled, config,
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
        np.ones(2), np.ones(2), np.array([0]), np.array([1]), np.random.default_rng(42),
    )

    assert len(observed) == 2
    source = _measured()
    for actual, name in zip(observed, ("top", "bottom")):
        xyz = [
            1.7 * np.interp(frame_position, np.arange(5), source[f"{component}_{name}_rad"])
            for component in ("alpha", "gamma", "beta")
        ]
        np.testing.assert_allclose(actual, Rotation.from_euler("XYZ", xyz).as_matrix(), atol=1e-14)


def test_replay_holds_each_shell_independently_across_missing_samples() -> None:
    measured = _measured()
    for component in ("alpha", "gamma", "beta"):
        measured[f"{component}_top_rad"][2] = np.nan
        measured[f"{component}_bottom_rad"][3] = np.nan
    _, full = simulate_measured._trajectories(measured, fps=29.0)
    for component in ("alpha", "gamma", "beta"):
        top = getattr(full, f"{component}_top")
        bottom = getattr(full, f"{component}_bottom")
        assert top[2] == top[1]
        assert bottom[3] == bottom[2]
        assert top[3] == measured[f"{component}_top_rad"][3]
        assert bottom[2] == measured[f"{component}_bottom_rad"][2]
    # Filling replay values must not manufacture measurements in the input.
    assert np.isnan(measured["alpha_top_rad"][2])
    assert np.isnan(measured["alpha_bottom_rad"][3])


@pytest.mark.parametrize("swap", [False, True])
def test_roundtrip_compares_fused_roll_to_the_motion_actually_rendered(
    monkeypatch: pytest.MonkeyPatch, swap: bool
) -> None:
    measured = {
        "alpha_rad": np.array([0.0, 0.1, 0.2]),
        "alpha_top_rad": np.array([0.0, np.nan, 0.2]),
        "alpha_bottom_rad": np.array([0.0, 0.1, 0.2]),
        "beta_top_rad": np.array([0.0, 0.03, 0.06]),
        "beta_bottom_rad": np.array([0.0, -0.02, -0.04]),
        "gamma_top_rad": np.zeros(3),
        "gamma_bottom_rad": np.zeros(3),
    }
    _, full = simulate_measured._trajectories(measured, fps=29.0, swap_shells=swap)
    # At the gap the renderer holds top roll at zero, so perfect tracking
    # reports mean roll 0.05, while the original single-shell result is 0.1.
    result = SimpleNamespace(
        frame_count=3,
        estimates=[SimpleNamespace(top=SimpleNamespace(success=True), bottom=SimpleNamespace(success=True)) for _ in range(2)],
        motion=SimpleNamespace(
            alpha=np.array([0.0, 0.05, 0.2]),
            beta_top=measured["beta_top_rad"],
            beta_bottom=measured["beta_bottom_rad"],
        ),
    )
    monkeypatch.setattr(simulate_measured, "run_pipeline", lambda *args, **kwargs: result)
    report = simulate_measured._roundtrip(
        [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)],
        full, np.eye(3), (16.0, 16.0, 10.0), np.eye(3), {}, 29.0,
        swap_shells=swap,
    )
    for name in ("alpha", "beta_top", "beta_bottom"):
        assert report[name]["rmse_deg"] == pytest.approx(0.0, abs=1e-12)
        assert report[name]["max_abs_deg"] == pytest.approx(0.0, abs=1e-12)


def test_hidden_hemisphere_probe_sampling_finishes_with_no_candidates() -> None:
    # Keep a regression to the former unbounded loop from hanging pytest.
    program = (
        "import numpy as np; from scripts.simulate_measured import _probe_points; "
        "points = _probe_points(np.eye(3), 1); assert points.shape == (0, 3)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        timeout=10, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    visible = simulate_measured._probe_points(np.eye(3), -1)
    assert visible.shape == (14, 3)
    assert (visible[:, 2] < 0).all()
    assert (visible @ np.array([0.0, 0.0, -1.0]) >= 0.45).all()


def _geometry() -> tuple[dict, dict]:
    home = Rotation.from_euler("xyz", [14.0, -33.0, 42.0], degrees=True).as_matrix()
    config = {
        "camera": {"K": [[850.0, 0.0, 320.0], [0.0, 855.0, 240.0], [0.0, 0.0, 1.0]], "dist": [0.01, -0.02, 0.0, 0.0]},
        "circle": {"u0": 320.0, "v0": 240.0, "r_px": 150.0},
        "frame_calib": {"R_bc": home.tolist(), "initial_roll_deg": 87.0},
    }
    metadata = {
        "K": copy.deepcopy(config["camera"]["K"]),
        "dist": copy.deepcopy(config["camera"]["dist"]),
        "circle": [320.0, 240.0, 150.0],
        "R_bc": measurement_frame(config["frame_calib"]).tolist(),
    }
    return config, metadata


def test_replay_accepts_matching_saved_effective_frame() -> None:
    config, metadata = _geometry()
    K, dist, circle, frame = simulate_measured._result_geometry(config, metadata)
    np.testing.assert_array_equal(K, metadata["K"])
    np.testing.assert_array_equal(dist, metadata["dist"])
    assert circle == tuple(metadata["circle"])
    np.testing.assert_allclose(frame, measurement_frame(config["frame_calib"]), atol=1e-14)


@pytest.mark.parametrize("mismatch", ["pose_offset", "saved_home_frame"])
def test_replay_rejects_using_saved_angles_with_an_incompatible_frame(mismatch: str) -> None:
    config, metadata = _geometry()
    if mismatch == "pose_offset":
        config["frame_calib"]["initial_roll_deg"] = 0.0
    else:
        metadata["R_bc"] = copy.deepcopy(config["frame_calib"]["R_bc"])
    with pytest.raises(ValueError, match="initial orientation differs"):
        simulate_measured._result_geometry(config, metadata)


@pytest.mark.parametrize("mismatch", ["K", "dist", "circle"])
def test_replay_rejects_geometry_that_does_not_match_saved_results(mismatch: str) -> None:
    config, metadata = _geometry()
    if mismatch == "K":
        config["camera"]["K"][0][0] += 10.0
    elif mismatch == "dist":
        config["camera"]["dist"][0] += 0.1
    else:
        config["circle"]["r_px"] += 10.0
    with pytest.raises(ValueError, match="camera/circle differs"):
        simulate_measured._result_geometry(config, metadata)
