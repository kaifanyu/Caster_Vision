"""Focused S5 tests for the standalone rendered-ball integration."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from synthetic.ball_integration import ball_case_diagnostics, run_ideal_ball_case
from synthetic.generate_ball import RenderConfig, Trajectory, render_sequence
from synthetic.validate_ball import _pipeline_kwargs


def test_ball_renderer_supports_a_physical_arbitrary_roll_axis() -> None:
    zeros = np.zeros(2)
    trajectory = Trajectory(
        np.array([0.0, 0.12]),
        zeros,
        zeros,
        roll_axis_ball=np.array([0.0, -1.0, 0.0]),
    )
    rendered = render_sequence(
        None,
        trajectory,
        RenderConfig(num_speckles=200),
        return_frames=True,
    )

    assert rendered.frames is not None and len(rendered.frames) == 2
    assert rendered.ground_truth["generator"] == "synthetic.generate_ball"
    np.testing.assert_allclose(
        rendered.ground_truth["roll_axis_ball"], [0.0, -1.0, 0.0]
    )
    increment = np.asarray(rendered.ground_truth["R_top_increment_ball"][1])
    np.testing.assert_allclose(
        Rotation.from_matrix(increment).as_rotvec(),
        [0.0, -0.12, 0.0],
        atol=1e-12,
    )


def test_rendered_ball_runs_production_pipeline_and_adapter() -> None:
    case = run_ideal_ball_case(
        np.array([0.25, 0.08]),
        num_frames=10,
        fps=60.0,
        r_eff=0.075,
        track_config={
            "max_corners": 500,
            "quality": 0.008,
            "min_distance_px": 5,
            "klt_win": 25,
            "max_level": 3,
            "fwd_bwd_err_px": 1.0,
        },
        estimate_config={
            "ransac_iters": 100,
            "ransac_inlier_deg": 2.0,
            "min_inliers": 8,
        },
    )
    diagnostics = ball_case_diagnostics(case)

    assert case.pipeline.frame_count == 10
    assert len(case.frames) == 9
    assert all(frame.raw["device"] == "ball" for frame in case.frames)
    assert all(frame.raw["roll_direction_sign"] == -1.0 for frame in case.frames)
    assert diagnostics["roll_coverage"] >= 0.90
    assert diagnostics["roll_axis_rmse_deg"] < 2.0
    assert diagnostics["roll_rate_mean_relative_error"] < 0.03


def test_ball_oracle_profile_is_independent_and_explicitly_overridable() -> None:
    defaults = _pipeline_kwargs({"track": {"klt_win": 99}})
    assert defaults["track_config"]["klt_win"] == 21
    assert defaults["estimate_config"]["ransac_inlier_deg"] == 1.0

    overridden = _pipeline_kwargs(
        {
            "ball": {
                "synthetic_validation": {
                    "track": {"klt_win": 31},
                    "estimate": {"ransac_iters": 321},
                }
            }
        }
    )
    assert overridden["track_config"]["klt_win"] == 31
    assert overridden["track_config"]["max_corners"] == 400
    assert overridden["estimate_config"]["ransac_iters"] == 321
    assert overridden["estimate_config"]["ransac_inlier_deg"] == 1.0
