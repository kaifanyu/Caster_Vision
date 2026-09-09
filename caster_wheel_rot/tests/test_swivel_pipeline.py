"""Small end-to-end tests for the rendered swivel pipeline."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.run_swivel import _summary
from synthetic.generate_swivel import render_sequence, tracking_trajectory
from swivel.geometry import SwivelGeometry
from swivel.pipeline import run_pipeline


def _geometry(rendered):
    camera, caster = rendered.camera, rendered.geometry
    return SwivelGeometry(
        camera.T_cam_from_car[:3, :3],
        camera.T_cam_from_car[:3, 3],
        caster.swivel_axis_car,
        caster.hub_offset0_car,
        caster.axle0_car,
        caster.wheel_radius_m,
        caster.wheel_width_m,
    )


def test_rendered_clip_runs_tag_roll_and_common_adapter_end_to_end() -> None:
    trajectory = tracking_trajectory(
        9,
        60.0,
        delta_phi_deg=2.0,
        delta_psi_deg=0.4,
    )
    rendered = render_sequence(trajectory)
    result = run_pipeline(
        rendered.frames,
        K=rendered.camera.K,
        geometry=_geometry(rendered),
        marker_size_m=rendered.geometry.tag_size_m,
        fps=60.0,
        r_eff=rendered.geometry.wheel_radius_m,
        track_config={
            "max_corners": 420,
            "quality": 0.005,
            "min_distance_px": 4,
            "klt_win": 25,
            "fwd_bwd_err_px": 1.5,
        },
        estimate_config={"ransac_iters": 100, "ransac_inlier_deg": 2.0, "min_inliers": 8},
        max_off_axis_deg=3.0,
    )

    assert np.mean(result.psi_valid) == 1.0
    assert np.sqrt(np.mean(np.rad2deg(result.psi - trajectory.psi) ** 2)) < 1.0
    recovered = np.array([item.delta_phi for item in result.roll_estimates])
    assert np.sqrt(np.mean(np.rad2deg(recovered - np.diff(trajectory.phi)) ** 2)) < 1.0
    assert all(frame.roll_valid for frame in result.caster_frames)
    assert all(item.valid for item in result.reference_observations)
    assert np.nanmax(np.abs(np.rad2deg(result.reference_phase_error))) < 2.0
    assert len(result.caster_frames) == len(trajectory.t) - 1
    np.testing.assert_allclose(
        [frame.t for frame in result.caster_frames],
        0.5 * (trajectory.t[:-1] + trajectory.t[1:]),
    )


def test_loop_closure_uses_modulo_pose_but_retains_net_turns() -> None:
    tag = SimpleNamespace(valid=True, reprojection_error_px=0.1)
    roll = SimpleNamespace(
        success=True,
        inlier_ratio=0.95,
        off_axis_residual_rad=0.0,
        delta_phi=2.0 * np.pi,
    )
    result = SimpleNamespace(
        tag_observations=[tag, tag, tag],
        roll_estimates=[roll, roll],
        interval_quality=[{"roll_valid": True}, {"roll_valid": True}],
        frame_count=3,
        psi=np.array([0.0, np.pi, 2.0 * np.pi]),
        psi_valid=np.ones(3, dtype=bool),
        phi=np.array([0.0, 2.0 * np.pi, 4.0 * np.pi]),
        phi_valid=np.ones(3, dtype=bool),
    )
    closure = _summary(result, loop_closure=True)["loop_closure"]
    assert closure["psi_closure_error_deg"] == 0.0
    assert closure["phi_closure_error_deg"] == 0.0
    assert closure["phi_net_unwrapped_deg"] == 720.0
    assert closure["phi_net_turns"] == 2.0
