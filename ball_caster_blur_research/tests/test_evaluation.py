"""Independent image-consistency tests; the synthetic truth is test-only."""
import copy
import json
import unittest

import numpy as np

from blurtrack.evaluation import compare_reports, landmark_split, score_heldout, subset_data


def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def fixture(offset=.023):
    K = np.array([[900., 0., 640.], [0., 900., 480.], [0., 0., 1.]])
    cameras = [{'K': K, 'R': np.eye(3), 't': np.zeros(3)},
               {'K': K, 'R': ry(.18), 't': np.array([-.18, .015, .03])}]
    F, pivot = rx(np.deg2rad(78)), np.array([0., 0., .6])
    geo = {'radius_m': .1, 'gap_m': .02, 'red_shell_sign': 1}
    knots = np.linspace(0., 1., 101)
    angles = np.column_stack((.14 + .09 * np.sin(3 * knots),
                              -.2 * knots - .15 * knots ** 2,
                              .3 * knots + .12 * knots ** 2))
    times = [np.linspace(0., 1., 31), np.linspace(.012, .962, 29)]
    rows = []
    for ci in range(2):
        for shell in range(2):
            sign = 1 - 2 * shell
            for track in range(10):
                point = np.array([-.4 + track * .08, -.9, sign * (.23 + track % 3 * .08)])
                point /= np.linalg.norm(point)
                for frame, time in enumerate(times[ci]):
                    q = np.array([np.interp(time + offset * (ci == 1), knots, angles[:, k]) for k in range(3)])
                    local = .1 * (rz(q[shell + 1]) @ point) + [0., 0., sign * .01]
                    world = F @ rx(q[0]) @ local + pivot
                    xyz = cameras[ci]['R'] @ world + cameras[ci]['t']
                    uvh = K @ xyz
                    rows.append((ci, shell, frame, track, uvh[:2] / uvh[2]))
    obs = {key: np.array([row[k] for row in rows])
           for k, key in enumerate(('camera', 'shell', 'frame', 'track', 'uv'))}
    obs['weight'] = np.ones(len(rows))
    data = {'times': times, 'observations': obs, 'timestamp_origin_s': 100.}
    return data, cameras, F, pivot, geo, knots, angles, offset


class LandmarkSplitTests(unittest.TestCase):
    def test_whole_tracks_stratified_deterministic_under_reorder_duplicates(self):
        data, *_ = fixture()
        split = landmark_split(data)
        self.assertEqual(len(split['heldout_keys']), 8)
        self.assertTrue(all(v['heldout_tracks'] == 2 and v['training_tracks'] == 8
                            for v in split['strata'].values()))
        obs = data['observations']
        identities = np.column_stack([obs[k] for k in ('camera', 'shell', 'track')])
        for key in np.unique(identities, axis=0):
            selection = split['holdout_mask'][(identities == key).all(axis=1)]
            self.assertEqual(len(np.unique(selection)), 1)
        rows = np.random.default_rng(40).permutation(len(identities))
        rows = np.r_[rows, rows[:100]]
        reordered = {**data, 'observations': {k: v[rows] for k, v in obs.items()}}
        self.assertEqual(split['heldout_keys'], landmark_split(reordered)['heldout_keys'])
        self.assertFalse(np.any(split['train_mask'] & split['holdout_mask']))

    def test_reference_subset_requires_explicit_keys_and_copies_arrays(self):
        data, *_ = fixture()
        split = landmark_split(data)
        reference = subset_data(data, split['holdout_mask'])
        explicit = landmark_split(reference, heldout_keys=split['heldout_keys'])
        self.assertTrue(explicit['holdout_mask'].all())
        self.assertFalse(explicit['train_mask'].any())
        before = data['observations']['uv'].copy()
        reference['observations']['uv'][:] = 0
        np.testing.assert_array_equal(data['observations']['uv'], before)
        with self.assertRaises(ValueError):
            landmark_split(data, heldout_keys=[[0, 0, 99]])


class HeldoutScoringTests(unittest.TestCase):
    def test_fixed_true_trajectory_later_pixels_and_json(self):
        args = fixture()
        report = score_heldout(*args, trajectory_excludes_heldout=True,
                               include_observations=True, windows=[(.5, .8)])
        self.assertEqual(report['heldout_tracks'], 8)
        self.assertEqual(report['initialization_observations'], 24)
        self.assertEqual(report['track_status_counts'], {'initialized': 8})
        self.assertEqual(report['overall']['count'], 4 * 28 + 4 * 26)
        self.assertLess(report['overall']['max_px'], 1e-5)
        self.assertEqual(report['overall']['scored_fraction'], 1.)
        self.assertTrue(all(row['source_frame'] >= 3 for row in report['observations']))
        self.assertFalse(report['accuracy_validated'])
        self.assertTrue(report['trajectory_excludes_heldout_asserted'])
        self.assertGreater(report['windows']['0.5-0.8s']['count'], 0)
        json.dumps(report, allow_nan=False)

    def test_later_outliers_do_not_move_initialized_material_points(self):
        args = list(fixture())
        reference = score_heldout(*args)
        data = copy.deepcopy(args[0])
        later = data['observations']['frame'] >= 3
        data['observations']['uv'][later] += [12., -5.]
        args[0] = data
        corrupted = score_heldout(*args)
        self.assertEqual(corrupted['overall']['count'], reference['overall']['count'])
        self.assertAlmostEqual(corrupted['overall']['median_px'], 13., places=5)
        self.assertEqual(corrupted['overall']['fraction_within_3px_including_failures'], 0.)
        for original, new in zip(reference['tracks'], corrupted['tracks']):
            self.assertEqual(original['initialization_rmse_px'], new['initialization_rmse_px'])

    def test_wrong_later_motion_and_wrong_clock_raise_error_without_refitting(self):
        args = list(fixture())
        angles_before = args[6].copy()
        shifted = args[6].copy()
        shifted[:, 1:] += np.maximum(args[5] - .15, 0)[:, None] * .2
        wrong_motion = score_heldout(*args[:6], shifted, args[7])
        self.assertGreater(wrong_motion['overall']['median_px'], 3.)
        wrong_clock = score_heldout(*args[:-1], 0.)
        self.assertGreater(wrong_clock['per_camera']['brio101']['median_px'], .05)
        self.assertLess(wrong_clock['per_camera']['c920']['max_px'], 1e-5)
        np.testing.assert_array_equal(args[6], angles_before)

    def test_repeated_rows_are_deduplicated_and_conflicts_fail(self):
        args = list(fixture())
        original = score_heldout(*args)
        obs = args[0]['observations']
        args[0] = {**args[0], 'observations': {k: np.concatenate((v, v)) for k, v in obs.items()}}
        doubled = score_heldout(*args)
        self.assertEqual(doubled['overall'], original['overall'])
        self.assertEqual(doubled['duplicate_observations_removed'], 240)
        heldout_rows = np.flatnonzero(landmark_split(args[0])['holdout_mask'])
        args[0]['observations']['uv'][heldout_rows[-1]] += [1., 0.]
        with self.assertRaisesRegex(ValueError, 'Conflicting pixels'):
            score_heldout(*args)

    def test_failed_initialization_and_unknown_trajectory_remain_in_denominator(self):
        args = list(fixture())
        data = copy.deepcopy(args[0])
        split = landmark_split(data)
        key = split['heldout_keys'][0]
        obs = data['observations']
        track = (np.column_stack([obs[k] for k in ('camera', 'shell', 'track')]) == key).all(axis=1)
        obs['uv'][track] = [1e5, 1e5]
        args[0] = data
        report = score_heldout(*args)
        self.assertEqual(report['track_status_counts']['no_valid_initial_observations'], 1)
        self.assertEqual(report['overall']['status_counts']['landmark_init_failed'], 31)
        self.assertLess(report['overall']['scored_fraction'], 1.)
        # Unknown trajectory stays missing; it is not endpoint extrapolation.
        cut = args[5] <= .7
        report = score_heldout(*args[:5], args[5][cut], args[6][cut], args[7])
        self.assertGreater(report['overall']['status_counts']['outside_or_unknown_trajectory'], 0)
        json.dumps(report, allow_nan=False)

    def test_all_zero_weight_reference_produces_null_metrics(self):
        args = list(fixture())
        args[0]['observations']['weight'][:] = 0.
        report = score_heldout(*args)
        self.assertEqual(report['overall']['count'], 0)
        self.assertIsNone(report['overall']['median_px'])
        self.assertEqual(report['overall']['scored_fraction'], 0.)
        self.assertEqual(report['overall']['status_counts'], {'zero_weight_observation': 240})
        self.assertFalse(report['trajectory_excludes_heldout_asserted'])
        json.dumps(report, allow_nan=False)


def motion_report(times, origin=None, phase=False, added=0.):
    report = {'frames': [{'time_s': float(t), 'angles': [t + added, 2 * t + added, -t + added],
                           'status': ['home' if i == 0 else 'vision',
                                      'phase_estimated' if phase and i > 0 else 'vision', 'vision']}
                          for i, t in enumerate(times)]}
    if origin is not None:
        report['timestamp_origin_s'] = origin
    return report


class BranchComparisonTests(unittest.TestCase):
    def test_phase_estimates_excluded_from_strict_coverage_rates_and_disagreement(self):
        times = np.arange(0., .5, .1)
        report = compare_reports({'klt': motion_report(times),
                                  'cotracker': motion_report(times, phase=True, added=2 * np.pi)},
                                 windows=[{'name': 'middle', 'start_s': .1, 'end_s': .3}])
        component = report['branches']['cotracker']['full']['components']['red']
        self.assertEqual(component['strict_supported_events'], 1)
        self.assertEqual(component['strict_rate_intervals'], 0)
        self.assertIsNone(component['strict_rate_min_deg_s'])
        pair = report['pairwise']['klt vs cotracker']['full']['components']['red']
        self.assertEqual(pair['strict_only']['samples'], 1)
        self.assertEqual(pair['all_finite_estimates']['samples'], 5)
        self.assertLess(pair['all_finite_estimates']['max_abs_wrapped_deg'], 1e-10)
        roll = report['branches']['klt']['full']['components']['roll']
        self.assertAlmostEqual(roll['strict_rate_max_deg_s'], np.rad2deg(1.))
        self.assertFalse(report['accuracy_validated'])
        json.dumps(report, allow_nan=False)

    def test_physical_origins_align_rezeroed_report_clocks(self):
        first = motion_report([0., .1, .2, .3], origin=100.)
        second = motion_report([0., .1, .2], origin=100.1, added=.1)
        # Correct q for second's independent component slopes.
        for frame in second['frames']:
            physical = frame['time_s'] + .1
            frame['angles'] = [physical, 2 * physical, -physical]
        result = compare_reports({'first': first, 'second': second})
        self.assertEqual(result['time_alignment'], 'shared absolute recording clock')
        pair = result['pairwise']['first vs second']['full']['components']
        for name in ('roll', 'red', 'green'):
            self.assertLess(pair[name]['strict_only']['max_abs_wrapped_deg'], 1e-10)
        second.pop('timestamp_origin_s')
        result = compare_reports({'first': first, 'second': second})
        self.assertIn('unverified', result['time_alignment'])
        self.assertGreater(result['pairwise']['first vs second']['full']['components']['roll']
                           ['strict_only']['median_abs_wrapped_deg'], 5.)

    def test_long_gaps_and_unresolved_angles_are_not_interpolated(self):
        short = motion_report([0., .1, .2, .3, .4])
        gap = motion_report([0., .4])
        result = compare_reports({'reference': short, 'gap': gap})
        self.assertEqual(result['pairwise']['reference vs gap']['full']['components']['roll']
                         ['all_finite_estimates']['samples'], 2)
        self.assertEqual(result['branches']['gap']['full']['components']['roll']['strict_rate_intervals'], 0)
        short['frames'][1]['angles'][0] = None
        short['frames'][1]['status'][0] = 'unresolved'
        result = compare_reports({'one': short, 'two': motion_report([0., .1, .2, .3, .4])})
        self.assertEqual(result['pairwise']['one vs two']['full']['components']['roll']
                         ['strict_only']['samples'], 4)
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
