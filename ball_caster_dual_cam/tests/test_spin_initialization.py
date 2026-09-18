"""Conditional spin recovery uses both views' actual surface-track increments."""
import unittest

import numpy as np

from dualcam.model import project, world_points
from dualcam.spin_initialization import reseed_spins, conditional_spin_evidence


def synthetic_case():
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
    knots = np.linspace(0., 2.04, 62)
    angles = np.deg2rad(np.column_stack((8.+70.*np.sin(2.*knots),
                                         90.*knots+50.*np.sin(3.*knots), -250.*knots)))
    times = [np.arange(61)/30., np.arange(61)/30.+.01]
    brio_offset = .0065
    rng = np.random.default_rng(904)
    points = rng.normal(size=(2, 100, 3))
    points /= np.linalg.norm(points, axis=2)[..., None]
    points[:, :, 2] = abs(points[:, :, 2])*np.array([1., -1.])[:, None]
    rows = []
    for ci, camera in enumerate(cameras):
        camera_eye = -camera['R'].T @ camera['t']
        for fi, raw_time in enumerate(times[ci]):
            physical_time = raw_time+brio_offset*ci
            q = np.array([np.interp(physical_time, knots, angles[:, k]) for k in range(3)])
            for shell in (0, 1):
                xyz = world_points(F, pivot, q, shell, points[shell], .1, .02)
                center = world_points(F, pivot, q, shell, np.zeros_like(points[shell]), .1, .02)
                visible = np.einsum('ij,ij->i', xyz-center, camera_eye-xyz) > .015
                if ci == 0 and shell == 0 and .8 < raw_time < 1.5:
                    visible[:] = False
                uv = project(camera, xyz)
                for point_id in np.flatnonzero(visible):
                    rows.append((ci, shell, fi, point_id, uv[point_id]+rng.normal(0., .015, 2)))
    obs = {name: np.asarray([r[i] for r in rows], int)
           for i, name in enumerate(('camera', 'shell', 'frame', 'track'))}
    obs['uv'] = np.asarray([r[4] for r in rows])
    obs['weight'] = np.ones(len(rows))
    events = sorted((float(t), ci, fi) for ci, clock in enumerate(times) for fi, t in enumerate(clock))
    return {'times': times, 'observations': obs}, {'events': events}, cameras, F, pivot, geometry, knots, angles, brio_offset


class ConditionalSpinTests(unittest.TestCase):
    def test_recovers_multiple_turns_and_reversal_with_a_camera_shell_occluded(self):
        args = synthetic_case()
        data, _, _, _, _, _, knots, angles, offset = args
        original_pixels = data['observations']['uv'].copy()
        original_angles = angles.copy()
        seed, report = reseed_spins(*args)
        times = np.array([event[0] for event in seed['events']])
        expected = np.column_stack([np.interp(times, knots, angles[:, k]) for k in range(3)])
        error = np.rad2deg(seed['angles']-expected)
        self.assertLess(np.max(abs(error[:, 1:])), 1.)
        np.testing.assert_allclose(seed['angles'][:, 0], expected[:, 0], atol=1e-14)
        self.assertLess(np.rad2deg(seed['angles'][-1, 2]), -360.)
        self.assertEqual(seed['initial_brio_offset_s'], offset)
        self.assertGreater(report['per_shell']['red']['per_camera_edges']['1'], 100)
        self.assertTrue(report['optimizer_hypothesis_only'])
        self.assertTrue(all(not components for components in seed['components']))
        evidence = seed['spin_evidence']
        self.assertGreater(np.mean(evidence['event_supported'][:, 0]), .95)
        self.assertGreater(np.mean(evidence['event_supported'][:, 1]), .95)
        self.assertLessEqual(np.max(abs(evidence['edge_residual_deg'])), 2.)
        self.assertTrue(evidence['event_turn_count_valid'][-1].all())
        for time, camera, frame in evidence['corrected_events']:
            self.assertAlmostEqual(time, data['times'][int(camera)][int(frame)]+offset*(camera == 1))
        np.testing.assert_array_equal(data['observations']['uv'], original_pixels)
        np.testing.assert_array_equal(angles, original_angles)

    def test_missing_shell_stays_explicitly_unsupported_despite_finite_seed(self):
        args = list(synthetic_case())
        obs = args[0]['observations']
        keep = obs['shell'] == 0
        args[0]['observations'] = {key: value[keep] for key, value in obs.items()}
        seed, report = reseed_spins(*args)
        self.assertFalse(report['per_shell']['green']['supported'])
        self.assertEqual(report['per_shell']['green']['edges'], 0)
        np.testing.assert_array_equal(seed['angles'][:, 2], 0.)
        self.assertFalse(seed['spin_evidence']['event_supported'][:, 1].any())
        self.assertFalse(seed['spin_evidence']['event_phase_connected'][:, 1].any())

    def test_outlier_pixels_do_not_create_large_false_spins(self):
        args = list(synthetic_case())
        obs = args[0]['observations']
        # Large mistracks cross colatitude/visibility gates; enough real edges
        # remain in both views to recover a common independent spin curve.
        obs['uv'][::29] += [65., -55.]
        seed, report = reseed_spins(*args)
        knots, angles = args[6], args[7]
        times = np.array([event[0] for event in seed['events']])
        expected = np.column_stack([np.interp(times, knots, angles[:, k]) for k in range(3)])
        self.assertLess(np.max(abs(np.rad2deg(seed['angles'][:, 1:]-expected[:, 1:]))), 2.)
        self.assertLess(report['geometrically_valid_observations'], report['observations'])

    def test_unobserved_shell_interval_breaks_phase_graph_despite_smooth_curve(self):
        args = list(synthetic_case())
        data = args[0]
        obs = data['observations']
        times = np.array([data['times'][ci][fi] for ci, fi in zip(obs['camera'], obs['frame'])])
        # Both cameras lose green for half a second. Earlier/later relative
        # edges are real but cannot establish the later segment's phase.
        keep = ~((obs['shell'] == 1) & (times > .55) & (times < 1.10))
        data['observations'] = {key: value[keep] for key, value in obs.items()}
        seed, report = reseed_spins(*args)
        evidence = seed['spin_evidence']
        event_times = evidence['event_times_s']
        self.assertTrue(evidence['event_supported'][event_times < .4, 1].all())
        self.assertFalse(evidence['event_supported'][event_times > .65, 1].any())
        self.assertFalse(evidence['event_phase_connected'][event_times > 1.2, 1].any())
        self.assertFalse(evidence['event_turn_count_valid'][-1, 1])
        self.assertTrue(evidence['event_supported'][-1, 0])
        self.assertTrue(np.isfinite(seed['angles']).all())
        self.assertTrue(np.any((evidence['edge_shells'] == 1) & ~evidence['edge_phase_connected']))

    def test_interpolation_graph_connectivity_does_not_invent_temporal_rank(self):
        knots = np.array([0., 1/30., 2/30.])
        phase = np.arange(12)*2*np.pi/12
        normals = np.column_stack((.7*np.cos(phase), .7*np.sin(phase), np.full(12, np.sqrt(.51))))
        normals = np.vstack((normals, normals))
        times = np.r_[np.zeros(12), np.full(12, .05)]
        landmark = np.tile(np.arange(12), 2)
        obs = {'shell': np.zeros(24, int), 'camera': np.zeros(24, int)}
        evidence = conditional_spin_evidence(knots, np.zeros((3, 3)),
            [(0., 0, 0), (1/30., 0, 1), (.05, 0, 2)], obs, times, normals,
            landmark, np.arange(12), np.arange(12, 24), np.zeros(12), np.ones(12))
        rank = evidence['measurement_design']['red']
        self.assertEqual(rank['columns'], 2)
        self.assertEqual(rank['rank'], 1)
        self.assertEqual(rank['unique_temporal_equations'], 1)
        self.assertTrue(evidence['event_locally_supported'][:, 0].all())
        self.assertFalse(evidence['event_supported'][1, 0])
        # The observed interpolation functional itself is identifiable even
        # though its two contributing knot values are individually unknown.
        self.assertTrue(evidence['event_supported'][2, 0])


if __name__ == '__main__':
    unittest.main()
