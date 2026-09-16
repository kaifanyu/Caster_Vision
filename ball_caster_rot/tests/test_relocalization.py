"""A landmark map can recover angles, but cannot invent missing pose evidence."""

import json

import cv2
import numpy as np
import pytest

from ballrot.offline import SurfaceObservation
from ballrot.relocalization import estimate_landmark_pose
from ballrot.rotation import Rx, Rz
from synthetic.generate import default_camera


def scene(*, sign=1, geometry="separated_hemispheres", count=80, translation=None):
    rng = np.random.default_rng(912)
    camera = default_camera()
    alpha, beta, gap = .28, 1.25, .14
    orientation = camera.R_bc @ Rx(alpha) @ Rz(beta)
    points = rng.normal(size=(5000, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points = points[sign * points[:, 2] >= (gap if geometry == "common_sphere_caps" else .03)]
    normals = points @ orientation.T
    offsets = points.copy()
    if geometry == "separated_hemispheres":
        offsets[:, 2] += sign * gap
    xyz = camera.C + camera.radius * offsets @ orientation.T
    if translation is not None:
        xyz += translation
    points = points[np.einsum("ij,ij->i", normals, xyz) < -.2][:count]
    normals = points @ orientation.T
    offsets = points.copy()
    if geometry == "separated_hemispheres":
        offsets[:, 2] += sign * gap
    xyz = camera.C + camera.radius * offsets @ orientation.T
    if translation is not None:
        xyz += translation
    h = xyz @ camera.K.T
    pixels = h[:, :2] / h[:, 2:] + rng.normal(0, .04, (len(points), 2))
    mapping = {i: point.copy() for i, point in enumerate(points)}
    observations = [SurfaceObservation(60, i, pixel) for i, pixel in enumerate(pixels)]
    settings = dict(K=camera.K, C=camera.C, F=camera.R_bc, radius=camera.radius,
                    sign=sign, geometry=geometry, gap_fraction=gap,
                    initial_alpha=-.9, initial_beta=-1.7)
    return observations, mapping, settings, (alpha, beta)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("geometry", ["common_sphere_caps", "separated_hemispheres"])
def test_global_map_relocalization_recovers_distant_pose_with_outliers(sign, geometry):
    observations, mapping, settings, truth = scene(sign=sign, geometry=geometry)
    originals = {identifier: point.copy() for identifier, point in mapping.items()}
    rng = np.random.default_rng(44)
    observations[:15] = [SurfaceObservation(60, entry.track_id, entry.uv + rng.uniform(35, 80, 2))
                         for entry in observations[:15]]
    result = estimate_landmark_pose(observations, mapping, **settings)
    assert result is not None
    alpha, beta, diagnostic = result
    error = (np.array([alpha, beta]) - truth + np.pi) % (2 * np.pi) - np.pi
    assert np.max(np.abs(error)) < np.deg2rad(.03)
    assert diagnostic["inlier_count"] == 65
    assert diagnostic["known_observation_count"] == 80
    assert diagnostic["inlier_fraction"] == pytest.approx(65 / 80)
    for identifier, original in originals.items():
        np.testing.assert_array_equal(mapping[identifier], original)
    json.dumps(diagnostic, allow_nan=False)


def test_local_angle_fit_survives_pnp_failure(monkeypatch):
    observations, mapping, settings, truth = scene()
    settings.update(initial_alpha=truth[0] + .08, initial_beta=truth[1] - .12)

    def degenerate(*args, **kwargs):
        raise cv2.error("degenerate sample")

    monkeypatch.setattr(cv2, "solvePnPRansac", degenerate)
    result = estimate_landmark_pose(observations, mapping, **settings)
    assert result is not None
    assert result[2]["initialization"] == "initial_angles"
    np.testing.assert_allclose(result[:2], truth, atol=np.deg2rad(.03))


def test_pnp_translation_does_not_relax_fixed_pivot_model():
    observations, mapping, settings, _ = scene(translation=np.array([.4, -.3, .2]))
    assert estimate_landmark_pose(observations, mapping, **settings) is None


def test_outliers_remain_in_validation_denominator():
    observations, mapping, settings, truth = scene()
    settings.update(initial_alpha=truth[0], initial_beta=truth[1])
    observations[:30] = [SurfaceObservation(60, entry.track_id, entry.uv + [60, -45])
                         for entry in observations[:30]]
    assert estimate_landmark_pose(observations, mapping, **settings) is None


def test_unknown_ids_and_duplicate_measurements_cannot_supply_map_support():
    observations, mapping, settings, _ = scene(count=20)
    known = dict(list(mapping.items())[:8])
    assert estimate_landmark_pose(observations * 5, known, **settings) is None
    assert estimate_landmark_pose(observations, {}, **settings) is None


def test_concentrated_map_is_rejected_by_spread():
    observations, mapping, settings, truth = scene()
    pixels = np.array([entry.uv for entry in observations])
    order = np.argsort(np.linalg.norm(pixels - pixels[0], axis=1))[:12]
    selected = [observations[i] for i in order]
    settings.update(initial_alpha=truth[0], initial_beta=truth[1], min_spread_fraction=.3)
    assert estimate_landmark_pose(selected, mapping, **settings) is None


def test_observations_must_share_one_image():
    observations, mapping, settings, _ = scene()
    observations.append(SurfaceObservation(61, 0, observations[0].uv))
    with pytest.raises(ValueError, match="one image"):
        estimate_landmark_pose(observations, mapping, **settings)
