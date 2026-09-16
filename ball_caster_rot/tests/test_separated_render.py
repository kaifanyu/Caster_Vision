"""Complete separated hemispheres retain their material and displaced silhouette."""

from dataclasses import replace

import numpy as np
import pytest

from ballrot.rotation import Rx, Rz
from synthetic import generate


def config(**kwargs):
    return generate.RenderConfig(
        camera=generate.default_camera((128, 160)), geometry="separated_hemispheres",
        gap_fraction=.2, draw_yoke=False, num_speckles=400, dot_radius_px=2., **kwargs,
    )


def test_separated_halves_keep_complete_hemispheres_including_equatorial_marks():
    settings = config()
    points, _, _, retained = generate._speckle_attributes(settings)
    assert retained.all()
    assert np.any(np.abs(points[retained, 2]) < settings.gap_fraction)
    common = replace(settings, geometry="common_sphere_caps")
    other_points, _, _, cap_retained = generate._speckle_attributes(common)
    np.testing.assert_array_equal(points, other_points)
    assert not cap_retained.all()


def test_separated_render_preserves_surface_outside_the_original_circle():
    settings = config()
    trajectory = generate.Trajectory([0., 0.], [0., 0.], [0., 0.])
    result = generate.render_sequence(None, trajectory, settings)
    image = result.frames[0]
    u, v, radius = settings.camera.circle
    yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
    outside = (xx - u)**2 + (yy - v)**2 > (radius + 2)**2
    surface = np.any(image != settings.background_color_bgr, axis=-1)
    assert np.count_nonzero(surface & outside) > 50
    assert result.ground_truth["geometry"] == "separated_hemispheres"
    assert result.ground_truth["rim_plane_separation"] == pytest.approx(.4)


def test_separated_silhouette_and_gap_depend_on_roll_and_not_independent_spin():
    settings = config(degradations=generate.Degradations(glare=True))
    trajectory = generate.Trajectory([.1, .1, .5], [0., 1.3, 1.3], [0., -2.1, -2.1])
    result = generate.render_sequence(None, trajectory, settings)
    masks = [np.any(image != settings.background_color_bgr, axis=-1) for image in result.frames]
    np.testing.assert_array_equal(masks[0], masks[1])
    assert np.count_nonzero(masks[0] != masks[2]) > 100
    # Translation leaves a real visible opening at this moderate roll.
    u, v, _ = settings.camera.circle
    assert not masks[0][int(v), int(u)]


@pytest.mark.parametrize("sign", [-1, 1])
def test_translated_surface_depth_matches_projected_material_points(sign):
    settings = config()
    camera = settings.camera
    orientation = camera.R_bc @ Rx(.2) @ Rz(-.7)
    depth = generate._hemisphere_depth(camera, orientation, sign, settings.gap_fraction)
    yy, xx = np.nonzero(np.isfinite(depth))
    rays = np.column_stack([xx, yy, np.ones(len(xx))]) @ np.linalg.inv(camera.K).T
    xyz = rays * depth[yy, xx, None]
    expected_center = camera.C + sign * camera.radius * settings.gap_fraction * orientation[:, 2]
    local_points = (xyz - expected_center) @ orientation / camera.radius
    np.testing.assert_allclose(np.linalg.norm(local_points, axis=1), 1., atol=1e-10)
    assert np.min(sign * local_points[:, 2]) >= -1e-10
    normals = (xyz - expected_center) / camera.radius
    assert np.all(np.einsum("ij,ij->i", normals, xyz) < 1e-10)


def test_unknown_render_geometry_is_rejected():
    with pytest.raises(ValueError, match="geometry"):
        generate.RenderConfig(geometry="unknown")
