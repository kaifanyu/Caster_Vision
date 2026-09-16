"""Observed feature evidence and distinct separated-shell rendering geometry."""

from types import SimpleNamespace
import json

import numpy as np
import pytest

from ballrot.rotation import Rx
from scripts import simulate_measured as replay


def measured():
    data = {
        "frame_index": np.array([1, 3]), "time_s": np.array([0.04, 0.12]),
        "mechanical": {"frames": {"top": [], "bottom": []}},
    }
    for shell in ("top", "bottom"):
        for component in ("alpha", "gamma", "beta"):
            data[f"{component}_{shell}_rad"] = np.zeros(2)
        data[f"valid_{shell}"] = np.zeros(2, dtype=bool)
        data["mechanical"]["frames"][shell] = [
            {"frame_index": index, "status": "unresolved", "reason": "not_reached"}
            for index in range(4)
        ]
    data["mechanical"]["frames"]["top"][1].update(
        reason="disconnected_inlier_graph",
        reprojection=[
            {"track_id": 105, "observed_uv": [110.0, 180.0], "predicted_uv": [111.0, 180.0],
             "error_px": 1.0, "inlier": True, "visible": True, "anchored": False},
            {"track_id": 206, "observed_uv": [140.0, 220.0], "predicted_uv": [150.0, 221.0],
             "error_px": float(np.sqrt(101)), "inlier": False, "visible": True, "anchored": True},
        ],
    )
    return data


def test_unaccepted_pixels_are_drawn_with_source_ids_and_candidacy(monkeypatch):
    arrows, labels = [], []
    monkeypatch.setattr(replay.cv2, "arrowedLine", lambda *args, **kwargs: arrows.append(args))
    monkeypatch.setattr(replay.cv2, "putText", lambda *args, **kwargs: labels.append(args[1]))
    image = np.zeros((300, 960, 3), dtype=np.uint8)
    summary = replay._draw_reprojection_frame(image, measured(), 0)
    assert [(args[1], args[2]) for args in arrows] == [((110, 180), (111, 180)), ((140, 220), (150, 221))]
    assert arrows[0][3] == (60, 220, 60)
    assert arrows[1][3] == (50, 60, 245)
    assert "T105?" in labels and "T206" in labels
    assert any("top: UNACCEPTED CANDIDATE" in label for label in labels)
    assert any("bottom: NO SAVED PREDICTIONS | not_reached" in label for label in labels)
    assert summary["top"] == {"accepted": False, "available": 2, "shown": 2, "reason": "disconnected_inlier_graph"}
    assert summary["bottom"]["available"] == 0


def test_residual_density_limit_does_not_change_available_counts():
    summary = replay._draw_reprojection_frame(np.zeros((300, 960, 3), dtype=np.uint8), measured(), 0, max_points=1)
    assert summary["top"]["available"] == 2
    assert summary["top"]["shown"] == 1


def test_overlay_reads_original_indices_under_stride_and_preserves_timing(monkeypatch, tmp_path):
    written, positions, released = [], [], []
    records = [SimpleNamespace(index=index, image=np.full((30, 60, 3), index, dtype=np.uint8)) for index in range(4)]
    monkeypatch.setattr(replay, "FrameSource", lambda *args, **kwargs: iter(records))
    monkeypatch.setattr(replay, "undistort_image", lambda image, *args: image.copy())
    monkeypatch.setattr(replay, "_draw_reprojection_frame", lambda image, data, position, limit: positions.append(position))
    writer = SimpleNamespace(write=lambda image: written.append(image.copy()), release=lambda: released.append(True))
    writer_arguments = []
    monkeypatch.setattr(replay, "_writer", lambda *args: writer_arguments.append(args) or writer)
    destination = tmp_path / "reprojection_overlay.mp4"
    assert replay._reprojection_overlay(tmp_path / "source.mp4", measured(), np.eye(3), np.zeros(4), 12.5,
                                        destination, "mp4v") == destination
    assert positions == [0, 1]
    assert [int(frame[0, 0, 0]) for frame in written] == [1, 3]
    assert writer_arguments[0][1:3] == (12.5, (60, 30))
    assert released == [True]


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("geometry", ["common_sphere_caps", "separated_hemispheres"])
def test_shell_projection_uses_rotated_offset_center(sign, geometry):
    frame = Rx(np.pi / 2)
    points = np.array([[0.6, -0.8, 0.0]])
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    center = np.array([0.0, 0.0, 5.0])
    normals = points @ frame.T
    expected_xyz = center + normals
    if geometry == "separated_hemispheres":
        expected_xyz += sign * 0.1 * frame[:, 2]
    expected = expected_xyz @ K.T
    uv, visible = replay._project_shell(points, frame, K, center, 1.0, sign, 0.1, geometry)
    assert visible.tolist() == [True]
    np.testing.assert_allclose(uv, expected[:, :2] / expected[:, 2:])


def test_saved_surface_geometry_cannot_be_changed_only_for_rendering():
    data = {"metadata": {"mechanical_model": {
        "enabled": True, "model": "shared_roll_independent_spin", "gap_fraction": 0.1,
    }}}
    assert replay._mechanical_model(data)["geometry"] == "common_sphere_caps"
    with pytest.raises(ValueError, match="geometry differs"):
        replay._result_mechanical_model({"mechanical": {"geometry": "separated_hemispheres"}}, data)
    data["metadata"]["mechanical_model"]["geometry"] = "separated_hemispheres"
    assert replay._result_mechanical_model({"mechanical": {"geometry": "separated_hemispheres"}}, data)["geometry"] == "separated_hemispheres"


@pytest.mark.parametrize("top_sign", [-1, 1])
@pytest.mark.parametrize("user_swap", [False, True])
@pytest.mark.parametrize("pivot", [None, [0.1, -0.2, 4.2]])
def test_axes_swap_keeps_cap_sign_and_translated_center_together(monkeypatch, tmp_path, top_sign, user_swap, pivot):
    data = measured()
    data["alpha_rad"] = np.zeros(2)
    data["metadata"] = {"top_shell_sign": top_sign, "mechanical_model": {
        "enabled": True, "model": "shared_roll_independent_spin", "geometry": "separated_hemispheres",
        "gap_fraction": 0.1, "pivot_camera": pivot,
    }}
    data["valid_top"][:] = data["valid_bottom"][:] = True
    monkeypatch.setattr(replay, "FrameSource", lambda *args, **kwargs: iter([
        SimpleNamespace(index=1, image=np.zeros((60, 120, 3), dtype=np.uint8)),
    ]))
    monkeypatch.setattr(replay, "undistort_image", lambda image, *args: image)
    monkeypatch.setattr(replay, "_writer", lambda *args: SimpleNamespace(write=lambda image: None, release=lambda: None))
    axes_centers = []
    monkeypatch.setattr(replay, "_draw_axes", lambda *args: axes_centers.append(args[3]))
    monkeypatch.setattr(replay, "_draw_missing_pose_notice", lambda *args: None)
    caps, projected_signs = [], []
    def graticule(step, sign, samples, gap):
        caps.append((sign, gap))
        return [np.array([[0.0, 0.0, float(sign)]])]
    monkeypatch.setattr(replay, "_graticule", graticule)
    monkeypatch.setattr(replay, "_probe_points", lambda *args: np.empty((0, 3)))
    surface_centers = []
    def draw_curve(*args, **kwargs):
        surface_centers.append(args[4])
        projected_signs.append(kwargs)
    monkeypatch.setattr(replay, "_draw_curve", draw_curve)
    monkeypatch.setattr(replay, "_project_shell", lambda *args: (np.empty((0, 2)), np.empty(0, dtype=bool)))
    swap = (top_sign == -1) != user_swap
    replay._axes_overlay(tmp_path / "source.mp4", data, np.eye(3), np.zeros(4), (60, 30, 20),
                         np.eye(3), np.array([0, 0, 5.0]), 1.0, 25.0,
                         tmp_path / "axes.mp4", 45, "mp4v", swap_shells=swap)
    assert caps == [(1, 0.0), (-1, 0.0)]
    expected_sign = -top_sign if user_swap else top_sign
    assert [entry["sign"] for entry in projected_signs] == [expected_sign, -expected_sign]
    assert all(entry["gap_fraction"] == 0.1 and entry["geometry"] == "separated_hemispheres"
               for entry in projected_signs)
    expected_center = [0., 0., 5.] if pivot is None else pivot
    for center in [*axes_centers, *surface_centers]:
        np.testing.assert_allclose(center, expected_center)


def test_residuals_are_a_cli_mode():
    parsed = replay.build_parser().parse_args(["--tests", "residuals", "--max-residuals", "30"])
    assert parsed.tests == ["residuals"] and parsed.max_residuals == 30


def test_camera_setup_preserves_saved_pivot_when_render_size_changes():
    camera = replay.RenderConfig().camera
    pivot = [.2, -.3, 4.5]
    for scale in (1., .5):
        K = np.diag([scale, scale, 1.]) @ camera.K
        circle = tuple(value * scale for value in camera.circle)
        size = tuple(int(value * scale) for value in camera.image_size)
        setup = replay._camera_setup(K, camera.R_bc, circle, size, pivot_camera=pivot)
        np.testing.assert_array_equal(setup.C, pivot)
        assert setup.radius == 1.
        assert setup.circle == circle
    original = replay._camera_setup(camera.K, camera.R_bc, camera.circle, camera.image_size)
    np.testing.assert_allclose(original.C, camera.C)


@pytest.mark.parametrize("requested", [None, [0., 0., 5.1]])
def test_saved_pivot_cannot_be_replaced_only_for_rendering(requested):
    data = {"metadata": {"mechanical_model": {
        "enabled": True, "model": "shared_roll_independent_spin", "pivot_camera": [0., 0., 5.],
    }}}
    assert replay._result_mechanical_model({}, data)["pivot_camera"] == [0., 0., 5.]
    with pytest.raises(ValueError, match="pivot_camera differs"):
        replay._result_mechanical_model({"mechanical": {"pivot_camera": requested}}, data)


def test_legacy_geometry_without_pivot_remains_circle_based():
    data = {"metadata": {"mechanical_model": {
        "enabled": True, "model": "shared_roll_independent_spin",
    }}}
    assert replay._result_mechanical_model({"mechanical": {"pivot_camera": None}}, data)["pivot_camera"] is None
    with pytest.raises(ValueError, match="pivot_camera differs"):
        replay._result_mechanical_model({"mechanical": {"pivot_camera": [0., 0., 5.]}}, data)


@pytest.mark.parametrize("pivot", [[1., 2.], [0., 0., float("nan")], [0., 0., .8]])
def test_invalid_saved_pivot_is_rejected(pivot):
    data = {"metadata": {"mechanical_model": {
        "enabled": True, "model": "shared_roll_independent_spin", "pivot_camera": pivot,
    }}}
    with pytest.raises(ValueError, match="pivot_camera"):
        replay._mechanical_model(data)


@pytest.mark.parametrize("model", [
    {"geometry": "separated_hemispheres"},
    {"geometry": "common_sphere_caps", "pivot_camera": [0., 0., 5.]},
])
def test_roundtrip_explicitly_skips_geometry_unsupported_by_raw_tracker(monkeypatch, model):
    def unsupported(*args, **kwargs):
        pytest.fail("raw common-sphere pipeline must not validate different geometry")
    monkeypatch.setattr(replay, "run_pipeline", unsupported)
    result = replay._roundtrip([], None, np.eye(3), (0., 0., 1.), np.eye(3),
                              {"mechanical": model}, 30.)
    assert result["status"] == "skipped"
    assert "cannot validate" in result["reason"]


def test_main_uses_saved_pivot_for_axes_and_replay_and_skips_invalid_roundtrip(monkeypatch, tmp_path, capsys):
    camera = replay.RenderConfig().camera
    pivot = [.1, -.2, 4.2]
    model = {"enabled": True, "model": "shared_roll_independent_spin", "geometry": "separated_hemispheres",
             "gap_fraction": .1, "pivot_camera": pivot}
    data = measured()
    data["alpha_rad"] = np.zeros(2)
    data["metadata"] = {"K": camera.K.tolist(), "dist": np.zeros(5).tolist(), "R_bc": camera.R_bc.tolist(),
                        "circle": camera.circle, "fps": 30., "mechanical_model": model}
    config_path = tmp_path / "config.yaml"
    config = {"input": {"path": "source.mp4"}, "mechanical": model}
    monkeypatch.setattr(replay, "load_config", lambda *args: (config, config_path))
    monkeypatch.setattr(replay, "_measured", lambda *args: data)
    image = np.zeros((*camera.image_size, 3), dtype=np.uint8)
    monkeypatch.setattr(replay, "FrameSource", lambda *args, **kwargs: iter([SimpleNamespace(image=image)]))
    axis_args, renders = [], []
    monkeypatch.setattr(replay, "_axes_overlay", lambda *args, **kwargs: axis_args.append(args) or args[10])
    monkeypatch.setattr(replay, "render_sequence", lambda _, trajectory, settings: renders.append(settings) or SimpleNamespace(frames=[image, image]))
    monkeypatch.setattr(replay, "_replay_video", lambda *args: args[6])
    monkeypatch.setattr(replay, "run_pipeline", lambda *args, **kwargs: pytest.fail("invalid roundtrip"))
    result_path = tmp_path / "results.json"
    result_path.write_text("{}")
    output = tmp_path / "simulation"
    assert replay.main(["--config", str(config_path), "--results", str(result_path), "--output", str(output),
                        "--tests", "axes", "replay", "roundtrip"]) == 0
    np.testing.assert_array_equal(axis_args[0][6], pivot)
    np.testing.assert_array_equal(renders[0].camera.C, pivot)
    assert renders[0].camera.circle == camera.circle
    report = json.loads((output / "simulation_report.json").read_text())
    assert report["pivot_camera"] == pivot
    assert report["pivot_source"] == "saved_mechanical_override"
    assert report["roundtrip"]["status"] == "skipped"
    assert "roundtrip: skipped" in capsys.readouterr().out
