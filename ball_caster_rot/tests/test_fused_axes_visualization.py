"""Experimental filter pictures must preserve provenance and channel states."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ballrot.rotation import Rx
from scripts import visualize_fused_axes as view
from synthetic.generate import default_camera


def example():
    camera = default_camera((240, 320))
    metadata = {"K": camera.K.tolist(), "dist": camera.dist.tolist(), "circle": list(camera.circle),
                "R_bc": camera.R_bc.tolist(), "input": "unchanged-source.mkv", "fps": 30.,
                "initial_roll_deg": 6.5, "mechanical_model": {
                    "enabled": True, "model": "shared_roll_independent_spin",
                    "geometry": "separated_hemispheres", "gap_fraction": .1, "pivot_camera": None}}
    frames = {"time_s": [0., .04, .08], "mechanical_valid_top": [True, False, False],
              "mechanical_valid_bottom": [False, False, False], "mechanical_alpha_rad": [.05, None, None]}
    for name in view.CHANNELS:
        frames[f"{name}_rad"] = [.1, .2, .3]
        frames[f"{name}_std_rad"] = [.02, .03, .04]
        frames[f"{name}_status"] = ["vision", "vision", "vision"]
        frames[f"{name}_vision_rad"] = [.08, None, .29]
        frames[f"{name}_prediction_age_s"] = [0., .04, .08]
        frames[f"{name}_reinitialized"] = [False, False, False]
    frames["alpha_status"] = ["vision", "predicted", "unresolved"]
    frames["alpha_rad"][2] = frames["alpha_std_rad"][2] = None
    frames["beta_bottom_status"][0] = "unresolved"
    frames["beta_bottom_rad"][0] = frames["beta_bottom_std_rad"][0] = None
    data = {"schema_version": 1, "method": "experimental_angle_kalman", "metadata": metadata,
            "source_results": "original/results.json", "measurement_source": "unconstrained",
            "parameters": {"max_prediction_age_s": .35}, "summary": {}, "frames": frames}
    return data, camera


def sample(data, camera, index):
    return view._sample_geometry(data["frames"], index, camera.K, camera.R_bc, camera.C, camera.radius,
                                 data["metadata"]["mechanical_model"], (0, 0, 320, 240), 1., [320, 240])


def test_axes_use_fused_roll_and_fixed_calibration_with_conditional_band():
    data, camera = example()
    row = sample(data, camera, 0)
    z = next(axis for axis in row["axes"] if axis["name"] == "z")
    h = camera.K @ (camera.C + 1.08 * (camera.R_bc @ Rx(.1))[:, 2])
    np.testing.assert_allclose(z["end"], h[:2] / h[2])
    for endpoint, alpha in zip((row["envelope"][0], row["envelope"][-1]), (.06, .14)):
        h = camera.K @ (camera.C + 1.08 * (camera.R_bc @ Rx(alpha))[:, 2])
        np.testing.assert_allclose(endpoint, h[:2] / h[2])
    assert {grid["shell"] for grid in row["grids"]} == {"top"}
    assert {axis["name"] for axis in row["comparison"]} == {"vision", "mechanical"}


def test_prediction_styles_are_independent_of_original_mechanical_acceptance():
    data, camera = example()
    row = sample(data, camera, 1)
    assert not row["mechanical_top"] and not row["mechanical_bottom"]
    assert [(axis["name"], axis["status"]) for axis in row["axes"]] == [
        ("x", "calibrated"), ("y", "predicted"), ("z", "predicted")]
    assert {grid["shell"] for grid in row["grids"]} == {"top", "bottom"}
    assert all(grid["status"] == "predicted" for grid in row["grids"])
    assert row["comparison"] == []
    assert row["prediction_age_s"]["alpha"] == .04
    assert not row["reinitialized"]["alpha"]
    assert data["frames"]["mechanical_valid_top"] == [True, False, False]


def test_unresolved_shared_roll_hides_current_axes_and_both_shell_grids():
    data, camera = example()
    row = sample(data, camera, 2)
    assert [axis["name"] for axis in row["axes"]] == ["x"]
    assert row["grids"] == row["envelope"] == []
    # A rejected raw visual value is only an explicitly optional reference.
    assert [axis["name"] for axis in row["comparison"]] == ["vision"]


@pytest.mark.parametrize("change", ["missing_angle", "negative_sigma", "bad_status"])
def test_loader_rejects_misleading_supported_states(tmp_path, change):
    data, _ = example()
    if change == "missing_angle":
        data["frames"]["alpha_rad"][0] = None
    elif change == "negative_sigma":
        data["frames"]["alpha_std_rad"][0] = -.1
    else:
        data["frames"]["alpha_status"][0] = "recovered"
    path = tmp_path / "results.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        view._load_results(path)


def test_full_artifact_generation_preserves_source_geometry_and_explicit_states(tmp_path, monkeypatch):
    data, camera = example()
    data["metadata"]["mechanical_model"]["pivot_camera"] = [.1, -.2, 4.3]
    source = tmp_path / "fused.json"
    source.write_text(json.dumps(data))
    saved_text = source.read_text()
    records = [SimpleNamespace(index=i, image=np.full((*camera.image_size, 3), 110, np.uint8)) for i in range(3)]
    monkeypatch.setattr(view, "FrameSource", lambda *args, **kwargs: iter(records))
    video_frames, released = [], []
    writer = SimpleNamespace(isOpened=lambda: True, write=lambda image: video_frames.append(image),
                             release=lambda: released.append(True))
    monkeypatch.setattr(view.cv2, "VideoWriter", lambda *args: writer)
    out = tmp_path / "view"
    assert view.main(["--results", str(source), "--output", str(out)]) == 0
    assert len(video_frames) == 3 and released == [True]
    report = json.loads((out / "fused_axis_inspection_report.json").read_text())
    assert report["pivot"] == [.1, -.2, 4.3]
    assert report["measurement_source"] == "unconstrained"
    assert report["frames_in_video"] == 3
    assert (out / "fused_axes_at_times.png").exists()
    html = (out / "fused_axes_timeline.html").read_text(encoding="utf-8")
    assert "__FUSED_" not in html
    assert '"alpha": "predicted"' in html and '"alpha": "unresolved"' in html
    assert "not a new mechanically recovered or pixel-validated trajectory" in html
    assert source.read_text() == saved_text
