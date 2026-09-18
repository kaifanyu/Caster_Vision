"""Independent synthetic evidence for asynchronous metric fusion."""
import unittest

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial.transform import Rotation

from dualcam.model import project, world_points
from dualcam.offline import NativeBundle, event_table, refine_native


def synthetic():
    rng = np.random.default_rng(71)
    K = np.array([[900., 0., 640.], [0., 900., 480.], [0., 0., 1.]])
    cameras = [{'K': K, 'R': np.eye(3), 't': np.zeros(3)},
               {'K': K, 'R': Rotation.from_euler('y', 18, degrees=True).as_matrix(),
                't': np.array([-.18, .015, .03])}]
    F = Rotation.from_euler('x', 78, degrees=True).as_matrix()
    pivot = np.array([0., 0., .6]); geo = {'radius_m': .1, 'gap_m': .02, 'red_shell_sign': 1}
    times = [np.linspace(0, 1., 31), np.linspace(.012, .992, 29)]
    def pose(t): return np.array([.14+.09*np.sin(2*t), -.45*t, .6*t])
    keys, points, rows = [], [], []
    for ci in range(2):
        for h in range(2):
            for k in range(12):
                # Front surface in the rotated caster frame.
                p = np.array([rng.uniform(-.6, .6), -.7, (1-2*h)*rng.uniform(.1, .7)])
                p /= np.linalg.norm(p)
                l = len(points); points.append(p); keys.append((ci, h, k))
                for fi, t in enumerate(times[ci]):
                    if ci == 0 and h == 0 and .35 < t < .8: continue
                    uv = project(cameras[ci], world_points(F, pivot, pose(t), h, p, .1, .02))[0]
                    rows.append((ci, h, fi, k, uv+rng.normal(0, .05, 2), 1., l))
    obs = {key: np.asarray([r[j] for r in rows], dtype=int if j < 4 else float)
           for j, key in enumerate(('camera', 'shell', 'frame', 'track', 'uv', 'weight'))}
    data = {'times': times, 'observations': obs}; events = event_table(data)
    seed = {'events': events, 'angles': np.array([pose(e[0]) for e in events]),
            'points': np.asarray(points), 'keys': np.asarray(keys),
            'landmark': np.array([r[-1] for r in rows])}
    return data, seed, cameras, F, pivot, geo, pose


def synthetic_clock_shift(shift=.023):
    """Independent pixel data with accelerating motion and a known Brio delay."""
    data, seed, cameras, F, pivot, geo, _ = synthetic()
    # Leave room at the clip end so corrected observations remain within the
    # modeled trajectory, rather than testing endpoint clamping as motion.
    data['times'][1] = np.linspace(.012, .962, 29)

    def pose(time):
        return np.array([.14 + .14 * np.sin(2 * np.pi * time),
                         -.3 * time + .18 * np.sin(2 * np.pi * time),
                         .4 * time + .14 * np.sin(3 * np.pi * time)])

    rng = np.random.default_rng(409)
    obs = data['observations']
    for i, (ci, fi, shell) in enumerate(zip(obs['camera'], obs['frame'], obs['shell'])):
        time = data['times'][ci][fi] + shift * (ci == 1)
        point = seed['points'][seed['landmark'][i]]
        obs['uv'][i] = project(cameras[ci], world_points(
            F, pivot, pose(time), shell, point, .1, .02))[0] + rng.normal(0, .03, 2)
    seed['events'] = event_table(data)
    seed['angles'] = np.array([pose(event[0]) for event in seed['events']])
    return data, seed, cameras, F, pivot, geo, pose


class NativeBundleTests(unittest.TestCase):
    def test_analytic_sparse_jacobian_matches_independent_finite_difference(self):
        data, seed, cameras, F, pivot, geo, _ = synthetic()
        problem = NativeBundle(data, seed, cameras, F, pivot, geo, knot_hz=10.)
        expected = approx_derivative(problem.evaluate, problem.x0, method='3-point')
        np.testing.assert_allclose(problem.evaluate(problem.x0, True).toarray(), expected, rtol=2e-5, atol=2e-5)

    def test_jacobian_matches_finite_difference_with_nonzero_time_and_tilt(self):
        data, seed, cameras, F, pivot, geo, _ = synthetic_clock_shift()
        problem = NativeBundle(data, seed, cameras, F, pivot, geo, knot_hz=10.)
        x = problem.x0.copy()
        # Avoid knot boundaries, where piecewise-linear time derivatives change.
        x[problem.timing_id] = .00731
        x[problem.initial_id] = .018
        expected = approx_derivative(problem.evaluate, x, method='3-point')
        actual = problem.evaluate(x, True).toarray()
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        camera0 = np.repeat(problem.obs['camera'] == 0, 2)
        np.testing.assert_array_equal(actual[:2 * len(problem.times), problem.timing_id][camera0], 0.)

    def test_initial_roll_is_a_global_finite_prior_not_a_fixed_gauge(self):
        data, seed, cameras, F, pivot, geo, _ = synthetic()
        problem = NativeBundle(data, seed, cameras, F, pivot, geo)
        delta = np.deg2rad(2.)
        shifted = problem.x0.copy()
        shifted[problem.initial_id] = delta
        original_q, original_points = problem.unpack(problem.x0)
        shifted_q, shifted_points = problem.unpack(shifted)
        np.testing.assert_allclose(shifted_q[:, 0] - original_q[:, 0], delta)
        np.testing.assert_array_equal(shifted_q[:, 1:], original_q[:, 1:])
        np.testing.assert_array_equal(shifted_points, original_points)
        self.assertLess(problem.lower[problem.initial_id], 0.)
        self.assertGreater(problem.upper[problem.initial_id], delta)
        self.assertTrue(np.isfinite(problem.evaluate(shifted)).all())
        self.assertAlmostEqual(problem.evaluate(shifted)[-2], delta / problem.initial_std)

    def test_joint_fit_recovers_known_clock_shift_and_corrects_initial_tilt(self):
        known_shift = .023
        data, seed, cameras, F, pivot, geo, pose = synthetic_clock_shift(known_shift)
        # Add an initial-roll error shared by every causal initialization knot.
        seed['angles'][:, 0] += .01
        problem, x, stages = refine_native(data, seed, cameras, F, pivot, geo,
                                           max_nfev=100, progress=None)
        self.assertLess(abs(x[problem.timing_id] - known_shift), .005)
        initial_roll = problem.unpack(x)[0][0, 0]
        self.assertLess(abs(initial_roll - pose(0.)[0]), np.deg2rad(.2))
        self.assertLess(x[problem.initial_id], -.005)
        self.assertLess(np.median(np.linalg.norm(problem.pixels(x)[0], axis=1)), .2)
        self.assertTrue(stages[-1]['converged'])

    def test_second_camera_constrains_missing_first_camera_shell(self):
        data, seed, cameras, F, pivot, geo, pose = synthetic()
        seed['angles'][5:, 1] += .04
        problem, x, stages = refine_native(data, seed, cameras, F, pivot, geo, max_nfev=50, progress=None)
        truth = np.array([pose(t) for t in problem.event_times])
        estimate = problem.angles_at(x, problem.event_times)
        self.assertLess(np.max(abs(estimate[:, 1]-truth[:, 1])), np.deg2rad(.5))
        self.assertLess(np.median(np.linalg.norm(problem.pixels(x)[0], axis=1)), .2)
        self.assertTrue(all(stage['cost'] >= 0 for stage in stages))

    def test_gap_changes_both_projected_centers(self):
        data, seed, cameras, F, pivot, geo, _ = synthetic()
        a = NativeBundle(data, seed, cameras, F, pivot, geo)
        b = NativeBundle(data, seed, cameras, F, pivot, {**geo, 'gap_m': 0.})
        self.assertGreater(np.median(np.linalg.norm(a.pixels(a.x0)[0]-b.pixels(b.x0)[0], axis=1)), 5.)


if __name__ == '__main__':
    unittest.main()
