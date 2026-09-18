"""Cross-view initialization transfers only fresh, observed components."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from dualcam.model import project, world_points
from dualcam.offline import seed_trajectory


def make_data(other_support=True, other_green=True):
    K = np.array([[900., 0., 480.], [0., 900., 270.], [0., 0., 1.]])
    cameras = [{'K': K, 'R': np.eye(3), 't': np.zeros(3)},
               {'K': K.copy(), 'R': np.eye(3), 't': np.array([.04, 0., 0.])}]
    F = Rotation.from_euler('x', 90., degrees=True).as_matrix()
    pivot = np.array([0., 0., .6])
    geometry = {'radius_m': .1, 'gap_m': .02, 'red_shell_sign': 1}
    rng = np.random.default_rng(198)
    points = []
    for h in range(2):
        p = np.column_stack((rng.uniform(-.7, .7, 24), rng.uniform(-1., -.7, 24),
                             rng.uniform(.1, .8, 24)*(1-2*h)))
        points.append(p/np.linalg.norm(p, axis=1)[:, None])
    times = [np.arange(9)/30., np.arange(9)/30.+.01]
    rows = []
    for ci in (0, 1):
        for fi in range(9):
            for h in (0, 1):
                if ci == 1 and fi > 0 and (not other_support or (h == 1 and not other_green)):
                    continue
                uv = project(cameras[ci], world_points(F, pivot, [0., 0., 0.], h,
                                                       points[h], .1, .02))
                for track, pixel in enumerate(uv):
                    # cam0 loses ALL IDs after a long observed prefix.
                    identifier = track+100 if ci == 0 and fi >= 5 else track
                    rows.append((ci, h, fi, identifier, pixel))
    obs = {name: np.asarray([r[i] for r in rows], int)
           for i, name in enumerate(('camera', 'shell', 'frame', 'track'))}
    obs['uv'] = np.asarray([r[4] for r in rows])
    return {'times': times, 'observations': obs}, cameras, F, pivot, geometry


class CrossCameraSeedTests(unittest.TestCase):
    def test_other_view_restarts_new_track_ids_without_fabricating_update(self):
        data, cameras, F, pivot, geometry = make_data()
        seed = seed_trajectory(data, cameras, F, pivot, geometry, 0., progress=None)
        lost = next(i for i, (_, ci, fi) in enumerate(seed['events']) if ci == 0 and fi == 5)
        recovered = next(i for i, (_, ci, fi) in enumerate(seed['events']) if ci == 0 and fi == 6)
        self.assertEqual(seed['components'][lost], [])
        self.assertEqual(seed['components'][recovered], [0, 1, 2])
        new = (seed['keys'][:, 0] == 0) & (seed['keys'][:, 2] >= 100)
        self.assertTrue(np.isfinite(seed['points'][new]).all())
        self.assertGreater(seed['per_camera'][0]['landmarks_seeded_from_other_camera'], 0)

    def test_recent_self_predictions_cannot_promote_new_landmarks(self):
        data, cameras, F, pivot, geometry = make_data(other_support=False)
        seed = seed_trajectory(data, cameras, F, pivot, geometry, 0., progress=None)
        new = (seed['keys'][:, 0] == 0) & (seed['keys'][:, 2] >= 100)
        self.assertTrue(np.isnan(seed['points'][new]).all())
        self.assertEqual(seed['per_camera'][0]['landmarks_seeded_from_other_camera'], 0)

    def test_other_red_shell_cannot_initialize_unobserved_green_spin(self):
        data, cameras, F, pivot, geometry = make_data(other_green=False)
        seed = seed_trajectory(data, cameras, F, pivot, geometry, 0., progress=None)
        new = (seed['keys'][:, 0] == 0) & (seed['keys'][:, 2] >= 100)
        self.assertTrue(np.isfinite(seed['points'][new & (seed['keys'][:, 1] == 0)]).all())
        self.assertTrue(np.isnan(seed['points'][new & (seed['keys'][:, 1] == 1)]).all())

    def test_offline_seed_follows_fast_roll_reversal_with_a_missing_camera_shell(self):
        """A 236 deg/s roll reverses while cam0 loses the green hemisphere."""
        K = np.array([[1100., 0., 480.], [0., 1100., 270.], [0., 0., 1.]])
        pivot = np.array([0., 0., .6])
        eye = np.array([-.24, .04, .03])
        forward = pivot-eye
        forward /= np.linalg.norm(forward)
        right = np.cross([0., 1., 0.], forward)
        right /= np.linalg.norm(right)
        R = np.stack([right, np.cross(forward, right), forward])
        cameras = [{'K': K, 'R': np.eye(3), 't': np.zeros(3)},
                   {'K': K.copy(), 'R': R, 't': -R @ eye}]
        F = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
        geometry = {'radius_m': .1, 'gap_m': .02, 'red_shell_sign': 1}
        rng = np.random.default_rng(509)
        points = rng.normal(size=(2, 160, 3))
        points /= np.linalg.norm(points, axis=2)[..., None]
        points[:, :, 2] = abs(points[:, :, 2])*np.array([1., -1.])[:, None]
        times = [np.arange(40)/30., np.arange(40)/30.+.01]

        def truth(t):
            tau = max(0., t-.1)
            return np.deg2rad([8.+90.*np.sin(np.pi*tau/1.2), 30.*tau, -45.*tau])

        rows = []
        for ci, camera in enumerate(cameras):
            camera_eye = -camera['R'].T @ camera['t']
            previous = [set(), set()]
            identifiers = [{}, {}]
            next_id = [0, 0]
            for fi, t in enumerate(times[ci]):
                angles = truth(t)
                for shell in (0, 1):
                    xyz = world_points(F, pivot, angles, shell, points[shell], .1, .02)
                    center = world_points(F, pivot, angles, shell, np.zeros_like(points[shell]), .1, .02)
                    visible = np.einsum('ij,ij->i', xyz-center, camera_eye-xyz) > .012
                    if ci == 0 and shell == 1 and t > .5:
                        visible[:] = False
                    pixels = project(camera, xyz)
                    for point_id in np.flatnonzero(visible):
                        if point_id not in previous[shell]:
                            identifiers[shell][point_id] = next_id[shell]
                            next_id[shell] += 1
                        pixel = pixels[point_id]+rng.normal(0., .05, 2)
                        rows.append((ci, shell, fi, identifiers[shell][point_id], pixel))
                    previous[shell] = set(np.flatnonzero(visible))
        obs = {name: np.asarray([r[i] for r in rows], int)
               for i, name in enumerate(('camera', 'shell', 'frame', 'track'))}
        obs['uv'] = np.asarray([r[4] for r in rows])
        seed = seed_trajectory({'times': times, 'observations': obs}, cameras, F, pivot,
                               geometry, 8., progress=None, offline_hypotheses=True)
        expected = np.array([truth(event[0]) for event in seed['events']])
        error_deg = np.rad2deg(seed['angles']-expected)
        self.assertLess(np.max(abs(error_deg[:, 0])), 4.)
        self.assertLess(np.max(abs(error_deg[-10:, 1:])), 5.)
        event_times = np.array([e[0] for e in seed['events']])
        after_reversal = seed['angles'][event_times > .85, 0]
        self.assertLess(after_reversal[-1], after_reversal[0]-.6)
        self.assertGreater(seed['per_camera'][1]['metric_adjacent_recovery'], 20)

    def test_offline_hypotheses_retain_future_pixels_without_claiming_new_evidence(self):
        data, cameras, F, pivot, geometry = make_data(other_support=False)
        # Replace all post-loss identities every frame, so no adjacent image
        # correspondence can support a recovered orientation.
        obs = data['observations']
        future = (obs['camera'] == 0) & (obs['frame'] >= 5)
        obs['track'][future] += 1000*obs['frame'][future]
        seed = seed_trajectory(data, cameras, F, pivot, geometry, 0., progress=None,
                               offline_hypotheses=True)
        late = [i for i, (_, ci, fi) in enumerate(seed['events']) if ci == 0 and fi >= 5]
        self.assertTrue(all(seed['components'][i] == [] for i in late))
        # Finite nuisance-point hypotheses retain the observations for the
        # batch solver; they do not change the empty image-support records.
        new = (seed['keys'][:, 0] == 0) & (seed['keys'][:, 2] >= 100)
        self.assertTrue(np.isfinite(seed['points'][new]).all())


if __name__ == '__main__':
    unittest.main()
