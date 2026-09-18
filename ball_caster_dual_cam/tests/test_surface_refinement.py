"""Physical projection, robust point recovery and staged-fit acceptance."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dualcam.model import polar_to_points, project, world_points
from dualcam.refinement import (_track_projection, _track_residual_jacobian,
                                refine_surface_points)
from dualcam.solver import _prepare, fit_joint
from tests.test_solver import make_clip, setup_scene


@pytest.mark.parametrize('camera_id,shell,red_sign', [(0, 0, 1), (1, 1, 1), (1, 0, -1)])
def test_conditional_projection_and_jacobian_match_metric_model(camera_id, shell, red_sign):
    cameras, F, C = setup_scene()
    camera = cameras[camera_id]
    q = np.array([[.1, .2, -.4], [.3, -.3, .2], [0., 0., 0.]])
    p = np.array([.8, 1.1])
    sign = red_sign*(1-2*shell)
    A, b = _track_projection(camera, F, C, q, shell, .1, .02, red_sign)
    unit = np.broadcast_to(polar_to_points([p], [sign]), (len(q), 3))
    expected = project(camera, world_points(F, C, q, shell, unit, .1, .02, red_sign))
    uv, weight = expected + [.5, -.3], np.array([.5, 1., 1.5])
    residual, jac = _track_residual_jacobian(p, sign, A, b, uv, weight)
    np.testing.assert_allclose(residual.reshape(-1, 2), (expected-uv)*weight[:, None], atol=1e-10)
    numerical = np.empty_like(jac)
    for i in range(2):
        delta = np.eye(2)[i]*1e-6
        plus = _track_residual_jacobian(p+delta, sign, A, b, uv, weight)[0]
        minus = _track_residual_jacobian(p-delta, sign, A, b, uv, weight)[0]
        numerical[:, i] = (plus-minus)/2e-6
    np.testing.assert_allclose(jac, numerical, atol=1e-6, rtol=1e-6)


def test_full_track_refinement_recovers_from_corrupted_first_observation():
    cameras, F, C = setup_scene()
    clip, q = make_clip(cameras, F, C, 'motion', count=6, frames=16, noise=.02)
    clip['initial_angles'] = q.copy()
    obs = clip['observations']
    obs['uv'][obs['frame'] == 0] += [18., -12.]
    d = _prepare(clip, cameras, F, C, .1, .02, 1, 3)
    original = {k: v.copy() for k, v in d['obs'].items()}
    refined, summary = refine_surface_points(d['obs'], q, d['polar0'], d['signs'],
                                             cameras, F, C, .1, .02, 1,
                                             robust_px=1., max_nfev=100)
    assert summary['cost_after'] < .2*summary['cost_before']
    assert summary['improved_landmarks'] == len(refined)
    assert np.all((refined[:, 1] >= 0) & (refined[:, 1] <= np.pi/2))
    points = polar_to_points(refined, d['signs'])
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1., atol=1e-12)
    xyz = world_points(F, C, q[obs['frame']], obs['shell'], points[d['obs']['landmark']], .1, .02)
    errors = []
    for ci, camera in enumerate(cameras):
        selected = (obs['camera'] == ci) & (obs['frame'] > 0)
        errors.extend(np.linalg.norm(project(camera, xyz[selected])-obs['uv'][selected], axis=1))
    assert np.median(errors) < .2
    for key, value in original.items():
        np.testing.assert_array_equal(value, d['obs'][key])


def test_staged_joint_calibration_recovers_geometry_and_preserves_home():
    cameras, F, C = setup_scene()
    clips = []
    for mode, seed in (('roll', 5), ('swivel', 8)):
        q = np.zeros((12, 3))
        if mode == 'roll':
            q[3:, 0] = np.linspace(.03, .45, 9)
        else:
            q[3:, 1] = np.linspace(.03, .6, 9)
            q[3:, 2] = np.linspace(-.03, -.4, 9)
        clip, _ = make_clip(cameras, F, C, mode, q=q, frames=12, count=8, noise=.04, seed=seed)
        clip['home_hold_s'] = 2/30
        clips.append(clip)
    biased = Rotation.from_rotvec(np.deg2rad([3., -4., 2.])).as_matrix() @ F
    out = fit_joint(clips, cameras, biased, C+[.003, -.002, .006], .1, .02,
                    calibrate_axes=True, refine_pivot=True,
                    options={'surface_refinement_passes': 2})
    assert out['success'], out['diagnostics']
    assert np.rad2deg(Rotation.from_matrix(out['F'] @ F.T).magnitude()) < .8
    assert np.linalg.norm(out['pivot']-C) < .002
    stages = out['diagnostics']['joint_stages']
    assert len(stages) == 2
    assert out['diagnostics']['nfev'] == sum(s['nfev'] for s in stages)
    for stage in stages:
        for points in stage['surface_refinement']:
            assert points['cost_after'] <= points['cost_before'] + 1e-8
    for source, result in zip(clips, out['datasets']):
        np.testing.assert_allclose(result['angles'][:3], 0., atol=1e-12)
        assert len(result['observation_inlier']) == len(source['observations']['uv'])


def test_point_convergence_does_not_bypass_joint_nonconvergence():
    cameras, F, C = setup_scene()
    clip, _ = make_clip(cameras, F, C, 'motion', count=5)
    out = fit_joint([clip], cameras, F, C, .1, .02,
                    options={'surface_refinement_passes': 2, 'max_nfev': 1})
    assert not out['success']
    assert not out['datasets'][0]['valid'].any()
    assert any('did not converge' in s for s in out['diagnostics']['reasons'])
    assert out['diagnostics']['nfev'] == 2


@pytest.mark.parametrize('key,value', [('surface_refinement_passes', -1),
                                       ('surface_refinement_passes', True),
                                       ('surface_refinement_passes', 1.5),
                                       ('surface_refinement_max_nfev', 0),
                                       ('max_nfev', 0)])
def test_invalid_staging_budget_is_rejected(key, value):
    cameras, F, C = setup_scene()
    with pytest.raises(ValueError, match=key):
        fit_joint([], cameras, F, C, .1, .02, options={key: value})
