import unittest

import numpy as np

from dualcam.roll_initialization import (_initial_hold_bias, _limited_interpolation,
                                        _separator_direction, reseed_roll)


class RollInitializationTests(unittest.TestCase):
    def test_separator_normal_keeps_physical_color_identity(self):
        angle = .37; normal = np.array([np.sin(angle), np.cos(angle)])
        tangent = np.array([-normal[1], normal[0]])
        cloud = np.array([a*normal+b*tangent for a in (40., 60.) for b in np.linspace(-30, 30, 10)])
        actual, report = _separator_direction(cloud, -cloud)
        self.assertIsNotNone(actual)
        self.assertGreater(actual @ normal, .99999)
        reverse, _ = _separator_direction(-cloud, cloud)
        self.assertLess(actual @ reverse, -.99999)
        self.assertEqual(report['misclassified_fraction'], 0.)

    def test_gaps_are_not_bridged_and_hold_mismatch_is_not_blindly_zeroed(self):
        actual = _limited_interpolation(np.array([0., .1, .5]), np.array([1., 2., 6.]),
                                        np.array([-.01, 0., .05, .1, .3, .5, .51]))
        np.testing.assert_allclose(actual[[1, 2, 3, 5]], [1, 1.5, 2, 6])
        self.assertTrue(np.isnan(actual[[0, 4, 6]]).all())
        times = np.linspace(0., 2.5, 30)
        bias, report = _initial_hold_bias(times, np.full(30, np.deg2rad(90)), np.deg2rad(8))
        self.assertEqual(bias, 0.)
        self.assertFalse(report['applied'])

    def test_two_camera_reseed_corrects_drift_and_reinitializes_landmarks(self):
        F = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
        pivot = np.array([0., 0., 1.])
        camera = {'K': np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]),
                  'R': np.eye(3), 't': np.zeros(3)}
        times = [np.arange(41)*.1, np.arange(40)*.1+.025]
        events = sorted((float(t), ci, fi) for ci, tt in enumerate(times) for fi, t in enumerate(tt))
        def truth(t):
            return np.deg2rad(8.) + .45*max(0., t-2.5)
        rows, identities, keys = [], [], []
        for ci, tt in enumerate(times):
            for sh in (0, 1):
                for track, (a, b) in enumerate((a, b) for a in (40., 60.) for b in np.linspace(-30, 30, 10)):
                    landmark = len(keys); keys.append((ci, sh, track))
                    for fi, t in enumerate(tt):
                        alpha = truth(t); n = np.array([np.sin(alpha), np.cos(alpha)]); tangent = np.array([-n[1], n[0]])
                        uv = np.array([320., 240.]) + (1-2*sh)*a*n + b*tangent
                        rows.append((ci, sh, fi, track, uv, 1.)); identities.append(landmark)
        obs = {k: np.asarray(v, dtype=int if k in ('camera', 'shell', 'frame', 'track') else float)
               for k, v in zip(('camera', 'shell', 'frame', 'track', 'uv', 'weight'), zip(*rows))}
        original = np.array([[truth(t)+np.deg2rad(14)*max(0., t-2.5), .2*t, -.1*t] for t, _, _ in events])
        seed = {'events': events, 'angles': original.copy(), 'keys': np.array(keys),
                'landmark': np.array(identities), 'points': np.full((len(keys), 3), np.nan)}
        actual, report = reseed_roll({'times': times, 'observations': obs}, seed,
                                     [camera, camera], F, pivot, {'radius_m': .2, 'gap_m': .02}, 8.)
        true = np.array([truth(t) for t, _, _ in events])
        self.assertLess(abs(actual['angles'][-1, 0]-true[-1]), np.deg2rad(2.))
        self.assertGreater(report['max_abs_correction_deg'], 15.)
        self.assertTrue(np.isfinite(actual['points']).all())
        np.testing.assert_array_equal(actual['angles'][:, 1:], original[:, 1:])
        np.testing.assert_array_equal(seed['angles'], original)
        self.assertTrue(all(c['initial_hold_alignment']['applied'] for c in report['per_camera']))


if __name__ == '__main__':
    unittest.main()
