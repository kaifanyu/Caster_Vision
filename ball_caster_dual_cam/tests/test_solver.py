"""Synthetic image-space checks, including actual oblique second-view geometry."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from dualcam.model import (initialize_surface_point, project, rotation_x,
                           rotation_z, world_points)
from dualcam.solver import fit_joint


def setup_scene():
    K = np.array([[1000., 0., 960.], [0., 1000., 540.], [0., 0., 1.]])
    C = np.array([.02, -.01, .8])
    center2 = np.array([-.38, .04, .05])
    forward = C - center2
    forward /= np.linalg.norm(forward)
    right = np.cross([0., 1., 0.], forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    cameras = [{'K': K, 'R': np.eye(3), 't': np.zeros(3)},
               {'K': K.copy(), 'R': R, 't': -R @ center2}]
    F = Rotation.from_euler('xyz', [72., 8., 15.], degrees=True).as_matrix()
    return cameras, F, C


def make_clip(cameras, F, C, mode, *, noise=.1, count=10, frames=9,
              omit_green=False, outliers=False, seed=5, q=None):
    rng = np.random.default_rng(seed)
    if q is None:
        q = np.zeros((frames, 3))
        if mode in ('roll', 'motion'):
            q[:, 0] = np.linspace(0., .42, frames)
        if mode in ('swivel', 'motion'):
            q[:, 1] = np.linspace(0., .65, frames)
            q[:, 2] = np.linspace(0., -.48, frames)
    rows = []
    for ci, camera in enumerate(cameras):
        camera_center = -camera['R'].T @ camera['t']
        for h in (0, 1):
            if h == 1 and omit_green:
                continue
            track = 0
            attempts = 0
            while track < count:
                attempts += 1
                if attempts > 5000:
                    raise RuntimeError('Could not find visible synthetic landmarks.')
                p = rng.normal(size=3)
                p /= np.linalg.norm(p)
                p[2] = (1 - 2 * h) * abs(p[2])
                xyz = world_points(F, C, q, h, np.tile(p, (frames, 1)), .1, .02)
                centers = world_points(F, C, q, h, np.zeros((frames, 3)), .1, .02)
                normals = (xyz - centers) / .1
                if np.min(np.sum(normals * (camera_center - xyz), axis=1)) < .08:
                    continue
                uv = project(camera, xyz) + rng.normal(0., noise, (frames, 2))
                for f in range(frames):
                    rows.append((ci, h, f, track, uv[f]))
                track += 1
    obs = {key: np.array([row[j] for row in rows], dtype=int)
           for j, key in enumerate(('camera', 'shell', 'frame', 'track'))}
    obs['uv'] = np.array([row[4] for row in rows])
    if outliers:
        ids = rng.choice(len(rows), len(rows) // 25, replace=False)
        obs['uv'][ids] += rng.normal(0., 35., (len(ids), 2))
    return {'times': np.arange(frames) / 30., 'initial_angles': q * .9,
            'mode': mode, 'initial_roll_rad': 0., 'observations': obs}, q


class GeometryTests(unittest.TestCase):
    def test_separated_shell_centers_and_camera_transform(self):
        cameras, F, C = setup_scene()
        q = np.array([.32, .43, -.1])
        p = np.array([.6, .0, .8])
        actual = world_points(F, C, q, 0, p, .1, .02)[0]
        expected = C + F @ rotation_x(q[0]) @ (np.array([0., 0., .01]) + .1 * rotation_z(q[1]) @ p)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        for camera in cameras:
            X = camera['R'] @ expected + camera['t']
            uv = camera['K'] @ X
            np.testing.assert_allclose(project(camera, actual)[0], uv[:2] / uv[2])

    def test_unprojection_recovers_visible_surface_point(self):
        cameras, F, C = setup_scene()
        clip, q = make_clip(cameras, F, C, 'roll', noise=0, count=4)
        obs = clip['observations']
        for ci in (0, 1):
            for h in (0, 1):
                i = np.flatnonzero((obs['camera'] == ci) & (obs['shell'] == h) & (obs['frame'] == 0))[0]
                p = initialize_surface_point(cameras[ci], obs['uv'][i], F, C, q[0], h, .1, .02)
                X = world_points(F, C, q[0], h, p, .1, .02)
                np.testing.assert_allclose(project(cameras[ci], X)[0], obs['uv'][i], atol=1e-8)
                self.assertGreaterEqual((1 - 2 * h) * p[2], 0)


class JointFitTests(unittest.TestCase):
    def test_close_large_image_shells_recover_axes_and_metric_pivot(self):
        # Close webcams give much larger pixel derivatives than the distant
        # fixtures. Recover centimeter-scale pivot bias alongside radian angles.
        C = np.array([.01, -.003, .36])
        center2 = np.array([-.25, .025, .015])
        forward = C - center2
        forward /= np.linalg.norm(forward)
        right = np.cross([0., 1., 0.], forward)
        right /= np.linalg.norm(right)
        R = rotation_z(np.pi) @ np.stack([right, np.cross(forward, right), forward])
        cameras = [
            {'K': np.array([[1430., 0., 960.], [0., 1430., 540.], [0., 0., 1.]]),
             'R': np.eye(3), 't': np.zeros(3)},
            {'K': np.array([[1960., 0., 960.], [0., 1960., 540.], [0., 0., 1.]]),
             'R': R, 't': -R @ center2}]
        F = Rotation.from_euler('xyz', [-78., 3., 2.], degrees=True).as_matrix()
        clips = [make_clip(cameras, F, C, mode, noise=.15, count=12, frames=12,
                           seed=seed)[0] for mode, seed in (('roll', 5), ('swivel', 8))]
        initial = Rotation.from_rotvec(np.deg2rad([3., -4., 2.])).as_matrix() @ F
        out = fit_joint(clips, cameras, initial, C + [.008, -.006, .012],
                        .1, .02, calibrate_axes=True, refine_pivot=True)
        self.assertTrue(out['success'], out['diagnostics'])
        self.assertLess(np.rad2deg(Rotation.from_matrix(out['F'] @ F.T).magnitude()), .8)
        self.assertLess(np.linalg.norm(out['pivot'] - C), .002)

    def test_calibrates_biased_axes_and_pivot_with_noisy_oblique_views(self):
        cameras, F, C = setup_scene()
        roll, qr = make_clip(cameras, F, C, 'roll', noise=.08)
        swivel, qs = make_clip(cameras, F, C, 'swivel', noise=.08, seed=8)
        biased = Rotation.from_rotvec(np.deg2rad([3., -4., 2.])).as_matrix() @ F
        out = fit_joint([roll, swivel], cameras, biased, C + [.003, -.002, .006],
                        .1, .02, calibrate_axes=True, refine_pivot=True)
        self.assertTrue(out['success'], out['diagnostics'])
        error = Rotation.from_matrix(out['F'] @ F.T).magnitude()
        self.assertLess(np.rad2deg(error), .8)
        self.assertLess(np.linalg.norm(out['pivot'] - C), .002)
        for estimated, expected in zip(out['datasets'], (qr, qs)):
            self.assertTrue(estimated['valid'].all())
            self.assertLess(np.rad2deg(np.max(abs(estimated['angles'] - expected))), 1.)

    def test_motion_keeps_independent_shell_spins_and_rejects_outliers(self):
        cameras, F, C = setup_scene()
        clip, q = make_clip(cameras, F, C, 'motion', noise=.1, count=14, outliers=True)
        out = fit_joint([clip], cameras, F, C, .1, .02)
        self.assertTrue(out['success'], out['diagnostics'])
        result = out['datasets'][0]
        self.assertTrue(result['valid'].all())
        self.assertLess(np.rad2deg(np.max(abs(result['angles'] - q))), 2.)
        self.assertLess(result['observation_inlier'].mean(), .99)
        self.assertGreater(result['angles'][-1, 1], 0)
        self.assertLess(result['angles'][-1, 2], 0)

    def test_unseen_shell_has_no_fabricated_spin(self):
        cameras, F, C = setup_scene()
        clip, q = make_clip(cameras, F, C, 'motion', omit_green=True)
        out = fit_joint([clip], cameras, F, C, .1, .02)
        self.assertTrue(out['success'], out['diagnostics'])
        result = out['datasets'][0]
        self.assertFalse(result['valid'][:, 2].any())
        self.assertTrue(np.isnan(result['angles'][:, 2]).all())
        self.assertLess(np.rad2deg(np.nanmax(abs(result['angles'][:, :2] - q[:, :2]))), 1.)

    def test_calibration_rejects_missing_home_or_missing_excitation(self):
        cameras, F, C = setup_scene()
        roll, _ = make_clip(cameras, F, C, 'roll')
        swivel, _ = make_clip(cameras, F, C, 'swivel')
        roll['initial_angles'][0, 0] = .1
        with self.assertRaisesRegex(ValueError, 'home'):
            fit_joint([roll, swivel], cameras, F, C, .1, .02, calibrate_axes=True)
        zero = np.zeros((9, 3))
        roll, _ = make_clip(cameras, F, C, 'roll', noise=0, q=zero)
        swivel, _ = make_clip(cameras, F, C, 'swivel', noise=0, q=zero)
        out = fit_joint([roll, swivel], cameras, F, C, .1, .02, calibrate_axes=True)
        self.assertFalse(out['success'])
        self.assertTrue(any('excursion' in message for message in out['diagnostics']['reasons']))
        self.assertTrue(all(not d['valid'].any() for d in out['datasets']))

    def test_disconnected_late_tracks_do_not_fabricate_absolute_angles(self):
        cameras, F, C = setup_scene()
        clip, q = make_clip(cameras, F, C, 'motion', frames=12)
        obs = clip['observations']
        late = obs['frame'] >= 6
        obs['track'][late] += 1000  # no temporal track bridges the gap to home
        out = fit_joint([clip], cameras, F, C, .1, .02)
        self.assertTrue(out['success'], out['diagnostics'])
        result = out['datasets'][0]
        self.assertTrue(result['valid'][:6].all())
        self.assertFalse(result['valid'][6:].any())
        self.assertTrue(np.isnan(result['angles'][6:]).all())
        self.assertFalse(result['observation_inlier'][late].any())

    def test_calibration_accepts_zero_motion_initialization(self):
        cameras, F, C = setup_scene()
        roll, _ = make_clip(cameras, F, C, 'roll', noise=.08)
        swivel, _ = make_clip(cameras, F, C, 'swivel', noise=.08, seed=8)
        roll['initial_angles'][:] = 0
        swivel['initial_angles'][:] = 0
        biased = Rotation.from_rotvec(np.deg2rad([3., -4., 2.])).as_matrix() @ F
        out = fit_joint([roll, swivel], cameras, biased, C, .1, .02,
                        calibrate_axes=True)
        self.assertTrue(out['success'], out['diagnostics'])
        self.assertLess(np.rad2deg(Rotation.from_matrix(out['F'] @ F.T).magnitude()), .8)

    def test_many_duplicate_corners_do_not_claim_observed_angles(self):
        cameras, F, C = setup_scene()
        clip, _ = make_clip(cameras, F, C, 'motion', count=1, noise=0, omit_green=True)
        source = clip['observations']
        copies = 10
        duplicated = {key: np.concatenate([value.copy() for _ in range(copies)])
                      for key, value in source.items()}
        duplicated['track'] = np.repeat(np.arange(copies), len(source['track']))
        clip['observations'] = duplicated
        out = fit_joint([clip], cameras, F, C, .1, .02)
        self.assertFalse(out['success'])
        self.assertFalse(out['datasets'][0]['valid'].any())
        self.assertTrue(np.isnan(out['datasets'][0]['angles']).all())

    def test_optimizer_nonconvergence_is_not_valid_calibration(self):
        cameras, F, C = setup_scene()
        clip, _ = make_clip(cameras, F, C, 'motion')
        out = fit_joint([clip], cameras, F, C, .1, .02, options={'max_nfev': 1})
        self.assertFalse(out['success'])
        self.assertFalse(out['datasets'][0]['valid'].any())


if __name__ == '__main__':
    unittest.main()
