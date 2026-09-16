"""Image rematching proposes observed pixels without inventing landmark IDs."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ballrot.mechanical_rematch import ImageLandmarkRematcher, _match_images
import ballrot.mechanical_rematch as rematch_module
from ballrot.offline import SurfaceObservation
from ballrot.rotation import Rx


def texture(seed=71):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 256, (256, 256), dtype=np.uint8), (3, 3), .6)


def test_lk_measures_shift_instead_of_returning_the_projection():
    first = texture()
    shift = np.array([4., -3.])
    second = cv2.warpAffine(first, np.array([[1., 0., shift[0]], [0., 1., shift[1]]]),
                            (256, 256), borderMode=cv2.BORDER_REFLECT)
    uv = cv2.goodFeaturesToTrack(first, 40, .1, 15).reshape(-1, 2)
    uv = uv[np.all((uv > 25) & (uv < 230), axis=1)]
    matched = _match_images(first, second, uv, uv.copy())
    assert len(matched) >= len(uv)*.8
    for index, pixel, details in matched:
        assert np.linalg.norm(pixel-uv[index]-shift) < .15
        assert details["forward_backward_error_px"] <= 1.
        assert details["patch_similarity"] >= .85
        assert details["prediction_error_px"] > 4.


def test_different_appearance_and_blank_patches_do_not_create_matches():
    first = texture()
    uv = np.array([[x, y] for x in (50., 100., 150., 200.) for y in (50., 100., 150., 200.)])
    assert not _match_images(first, texture(97), uv, uv.copy())
    blank = np.full_like(first, 100)
    assert not _match_images(blank, blank, uv, uv.copy())
    assert not _match_images(first, first, uv, uv.copy(), target_mask=np.zeros_like(first, bool))


def setup_rematcher(tmp_path, *, shift=(4., -3.)):
    first = texture()
    second = cv2.warpAffine(first, np.array([[1., 0., shift[0]], [0., 1., shift[1]]]),
                            (256, 256), borderMode=cv2.BORDER_REFLECT)
    images = tmp_path/"images"
    images.mkdir()
    assert cv2.imwrite(str(images/"frame0.png"), first)
    assert cv2.imwrite(str(images/"frame1.png"), second)
    K = np.array([[180., 0., 128.], [0., 180., 128.], [0., 0., 1.]])
    C, F = np.array([0., 0., 4.]), Rx(np.pi/2)
    points = np.array([[x, -np.sqrt(1-x*x-z*z), z]
                       for x in (-.4, -.2, 0., .2, .4) for z in (.2, .4, .6)])
    homogeneous = (C+points @ F.T) @ K.T
    pixels = homogeneous[:, :2]/homogeneous[:, 2:]
    provider = ImageLandmarkRematcher(images, K, np.zeros(4), C, F, 1., .1,
                                     "common_sphere_caps", 1)
    landmarks = {"top": dict(enumerate(points)), "bottom": {}}
    measured = {"top": np.array([True, False]), "bottom": np.zeros(2, bool)}
    reports = {"top": [{"status": "refined", "reason": "accepted", "reprojection": [
        {"track_id": index, "observed_uv": pixel.tolist(), "inlier": True, "visible": True}
        for index, pixel in enumerate(pixels)]}, {"status": "unresolved"}],
        "bottom": [{"status": "unresolved"}, {"status": "unresolved"}]}
    by_frame = {"top": {0: [SurfaceObservation(0, index, pixel)
                             for index, pixel in enumerate(pixels)]}, "bottom": {}}
    alpha = np.zeros(2)
    beta = {name: np.zeros(2) for name in landmarks}
    return provider, (landmarks, alpha, beta, measured, by_frame, reports), pixels


def test_only_accepted_templates_reobserve_existing_ids_and_preserve_provenance(tmp_path):
    provider, args, pixels = setup_rematcher(tmp_path)
    directory = Path(provider._temporary.name)
    with provider:
        additions = provider.propose(1, 2, *args)
        assert len(additions["top"]) >= 12
        assert not additions["bottom"]
        for entry in additions["top"]:
            assert entry.track_id in args[0]["top"]
            assert entry.frame_index == 1
            assert np.linalg.norm(entry.uv-pixels[entry.track_id]-[4., -3.]) < .2
        assert provider.provenance
        assert all(item["source_frame"] == 0 for item in provider.provenance)
        assert provider.propose(1, 2, *args) == {"top": [], "bottom": []}
        assert provider.diagnostics["added_observation_counts"]["top"] == len(additions["top"])
        json.dumps({"summary": provider.diagnostics, "provenance": provider.provenance}, allow_nan=False)
    assert not directory.exists()


@pytest.mark.parametrize("unsupported", ["not_measured", "failed_frame", "outlier_pixels"])
def test_unaccepted_pixels_never_supply_source_templates(tmp_path, unsupported):
    provider, args, _ = setup_rematcher(tmp_path)
    landmarks, alpha, beta, measured, by_frame, reports = args
    if unsupported == "not_measured":
        measured["top"][0] = False
    elif unsupported == "failed_frame":
        reports["top"][0].update(status="unresolved", reason="final_frame_quality_failed")
    else:
        for item in reports["top"][0]["reprojection"]:
            item["inlier"] = False
    with provider:
        assert provider.propose(1, 2, *args) == {"top": [], "bottom": []}
        assert not provider.provenance
        assert provider.diagnostics["decoded_frames"] == 0


def test_existing_target_id_and_accepted_target_are_not_replaced(tmp_path):
    provider, args, pixels = setup_rematcher(tmp_path)
    args[4]["top"][1] = [SurfaceObservation(1, 0, pixels[0])]
    with provider:
        additions = provider.propose(1, 2, *args)
        assert all(entry.track_id != 0 for entry in additions["top"])
        args[3]["top"][1] = True
        assert provider.propose(1, 2, *args) == {"top": [], "bottom": []}


def test_multiple_ids_cannot_supply_support_at_one_target_corner(tmp_path, monkeypatch):
    provider, args, pixels = setup_rematcher(tmp_path)
    point = pixels[0]+[4., -3.]

    def duplicate_matches(source, target, uv, guesses, **kwargs):
        return [(index, point.copy(), {"patch_similarity": .9+.001*index,
                                      "forward_backward_error_px": .1, "prediction_error_px": 1.})
                for index in range(len(uv))]

    monkeypatch.setattr(rematch_module, "_match_images", duplicate_matches)
    with provider:
        additions = provider.propose(1, 2, *args)
        assert len(additions["top"]) == 1
        assert additions["top"][0].track_id == len(pixels)-1
        assert provider.diagnostics["spatial_collision_rejections"]["top"] == len(pixels)-1


def test_original_nearby_observation_blocks_new_id_at_same_corner(tmp_path, monkeypatch):
    provider, args, pixels = setup_rematcher(tmp_path)
    point = pixels[0]+[4., -3.]
    args[4]["top"][1] = [SurfaceObservation(1, 9999, point+[.5, -.5])]

    def duplicate_matches(source, target, uv, guesses, **kwargs):
        return [(0, point.copy(), {"patch_similarity": .99,
                                  "forward_backward_error_px": .05, "prediction_error_px": 1.})]

    monkeypatch.setattr(rematch_module, "_match_images", duplicate_matches)
    with provider:
        assert provider.propose(1, 2, *args) == {"top": [], "bottom": []}
        assert provider.diagnostics["spatial_collision_rejections"]["top"] == 1
        assert not provider.provenance


def test_sequential_disk_cache_returns_exact_earlier_frame(tmp_path):
    for index in range(12):
        assert cv2.imwrite(str(tmp_path/f"frame{index}.png"), np.full((32, 40, 3), index, np.uint8))
    K = np.array([[80., 0., 20.], [0., 80., 16.], [0., 0., 1.]])
    with ImageLandmarkRematcher(tmp_path, K, np.zeros(4), [0., 0., 4.], np.eye(3),
                               1., .1, "common_sphere_caps", 1) as provider:
        assert np.all(provider._frame(10)[0] == 10)
        assert np.all(provider._frame(2)[0] == 2)
        for index in range(12):
            assert np.all(provider._frame(index)[0] == index)
        assert len(provider._cache) <= 8
        assert provider.diagnostics["decoded_frames"] == 12
