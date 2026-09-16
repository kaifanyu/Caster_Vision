"""Physical separated-center observations must fit without a false common sphere."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ballrot.mechanical import MechanicalConfig, refine_mechanical_trajectory
from ballrot.offline import OfflineConfig, SurfaceObservation
from ballrot.rotation import Rx, Rz
from ballrot.shell_geometry import shell_centers, surface_camera, unproject_shell
from test_mechanical import K, C, F, angular_error


@pytest.mark.parametrize("sign", [1, -1])
def test_centers_follow_roll_and_not_independent_spin(sign):
    a = .65
    orientations = np.array([F @ Rx(a) @ Rz(b) for b in (-1., 0., 1.3)])
    centers = shell_centers(orientations, C, 1., sign, .1, "separated_hemispheres")
    expected = C + sign * .1 * (F @ Rx(a))[:, 2]
    np.testing.assert_allclose(centers, np.tile(expected, (3, 1)))
    points = np.array([[.2, -np.sqrt(.95), sign*.1]])
    xyz = surface_camera(points, orientations[1:2], C, 1., sign, .1, "separated_hemispheres")
    homogeneous = xyz @ K.T
    normals, valid, _ = unproject_shell(homogeneous[:, :2]/homogeneous[:, 2:], K, C, 1.,
                                       orientations[1:2], sign, .1, "separated_hemispheres")
    assert valid.all()
    np.testing.assert_allclose(normals, points @ orientations[1].T, atol=1e-12)


def test_separated_hemisphere_fit_recovers_motion_and_saves_actual_residuals():
    rng = np.random.default_rng(777)
    count = 16
    alpha = np.linspace(0., .16, count)
    data, truth, initial = {}, {}, {}
    for name, sign in [("top", 1), ("bottom", -1)]:
        x = rng.uniform(-.4, .4, 45)
        z = sign * rng.uniform(.15, .5, 45)
        points = np.column_stack((x, -np.sqrt(1-x*x-z*z), z))
        beta = sign*np.linspace(0., .18, count)
        truth[name] = np.array([F @ Rx(a) @ Rz(b) @ F.T for a, b in zip(alpha, beta)])
        initial[name] = truth[name].copy()
        data[name] = []
        for i, (a, b) in enumerate(zip(alpha, beta)):
            orientation = F @ Rx(a) @ Rz(b)
            # Explicit physical construction independent of the production helper.
            center = C + sign*.1*(F @ Rx(a))[:, 2]
            xyz = center + points @ orientation.T
            image = xyz @ K.T
            uv = image[:, :2]/image[:, 2:] + rng.normal(0, .015, (45, 2))
            data[name].extend(SurfaceObservation(i, j, pixel) for j, pixel in enumerate(uv))
    result = refine_mechanical_trajectory(
        data, initial, {name: np.ones(count, bool) for name in data}, K, C, F,
        config=MechanicalConfig(enabled=True, gap_fraction=.1, geometry="separated_hemispheres",
                                window_frames=9, overlap_frames=3),
        offline_config=OfflineConfig(enabled=True, max_nfev=100))
    for name in data:
        assert result.valid[name].all(), result.diagnostics["summary"]
        assert np.max(angular_error(result.rotations[name], truth[name])) < .1
        assert len(result.landmarks[name]) >= 12
        entry = result.diagnostics["frames"][name][-1]
        assert len(entry["reprojection"]) >= 12
        for item in entry["reprojection"]:
            assert item["error_px"] == pytest.approx(np.linalg.norm(
                np.array(item["observed_uv"]) - item["predicted_uv"]))
        assert entry["bridge"]["connected_to_reference"]
    corrected = refine_mechanical_trajectory(
        data, initial, {name: np.ones(count, bool) for name in data}, K, C*.95, F,
        config=MechanicalConfig(enabled=True, gap_fraction=.1, geometry="separated_hemispheres",
                                pivot_camera=tuple(C), window_frames=9, overlap_frames=3),
        offline_config=OfflineConfig(enabled=True, max_nfev=100))
    for name in data:
        np.testing.assert_allclose(corrected.rotations[name], result.rotations[name], atol=1e-10)
    assert corrected.diagnostics["geometry_calibration"]["source"] == "configured_shell_pivot"


def test_invalid_geometry_is_not_silently_treated_as_common_sphere():
    with pytest.raises(ValueError, match="geometry"):
        MechanicalConfig(geometry="two_unknown_shapes")


@pytest.mark.parametrize("pivot", [[1., 2.], [0., 0., .5], [0., float('nan'), 4.]])
def test_invalid_pivot_is_rejected(pivot):
    with pytest.raises(ValueError, match="pivot_camera"):
        MechanicalConfig(pivot_camera=pivot)
