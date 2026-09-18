"""Prior-dependent phase estimates remain separate from measured phase."""
import copy
import pickle
import unittest

import numpy as np

from dualcam.offline_finalization import apply_phase_estimates


def case(relative_times=None):
    relative_times = np.asarray(relative_times if relative_times is not None else
                                [0., .03, .06, .09, .12, .15], float)
    physical_times = relative_times+10.2
    q = np.column_stack((.1+.2*relative_times, .3*relative_times, -.4*relative_times))
    frames = []
    for i, time in enumerate(relative_times):
        frames.append({'time_s': float(time), 'camera': ('c920', 'brio101')[i % 2],
                       'source_frame': i//2, 'angles': q[i].copy(),
                       'status': ['home' if i == 0 else 'vision']*3,
                       'turn_count_valid': [True]*3,
                       'angular_velocity': np.array([99., 99., 99.]),
                       'support_cameras': [['c920', 'brio101'], ['c920'], ['brio101']]})
    summary = {'coverage': {name: {'home': 1, 'vision': len(frames)-1}
                            for name in ('roll', 'red', 'green')},
               'optimizer_converged': True}
    evidence = {'event_camera': np.arange(len(frames)) % 2,
                'event_frame': np.arange(len(frames))//2,
                'event_times_s': physical_times.copy(),
                'event_locally_supported': np.ones((len(frames), 2), bool),
                'event_phase_connected': np.ones((len(frames), 2), bool),
                'event_support_cameras': np.ones((len(frames), 2, 2), bool),
                'event_track_counts': np.full((len(frames), 2), 12),
                'event_normal_spread': np.full((len(frames), 2), .2)}
    return frames, summary, evidence, np.eye(3), physical_times, q


class PhaseFinalizationTests(unittest.TestCase):
    def test_short_gap_uses_bundle_curve_with_explicit_unanchored_phase(self):
        args = list(case())
        frames, _, evidence, _, _, q = args
        evidence['event_locally_supported'][2, 0] = False
        evidence['event_phase_connected'][2:, 0] = False
        for frame in frames[2:]:
            frame['angles'][1] = np.nan
            frame['status'][1] = 'unresolved'
        result, summary = apply_phase_estimates(*args)
        np.testing.assert_allclose([f['angles'][1] for f in result], q[:, 1])
        self.assertTrue(all(f['status'][1] == 'phase_estimated' for f in result[2:]))
        self.assertTrue(all(np.isnan(f['strict_angles'][1]) for f in result[2:]))
        self.assertTrue(all(f['strict_status'][1] == 'unresolved' for f in result[2:]))
        self.assertTrue(all(not f['turn_count_valid'][1] for f in result[2:]))
        self.assertTrue(summary['phase_bridge_gaps'][0]['short_bridge_estimated'])
        self.assertEqual(summary['estimated_event_fraction']['red'], 1.)
        self.assertEqual(summary['supported_event_fraction']['red'], 2/6)
        np.testing.assert_allclose([f['angular_velocity'][1] for f in result], .3, atol=1e-12)

    def test_long_gap_keeps_unanchored_tail_null_after_local_tracks_return(self):
        args = list(case([0., .03, .3, .6, .63, .66]))
        frames, _, evidence, _, _, _ = args
        evidence['event_locally_supported'][2:4, 0] = False
        evidence['event_phase_connected'][2:, 0] = False
        for frame in frames[2:]:
            frame['angles'][1] = np.nan
            frame['status'][1] = 'unresolved'
        result, summary = apply_phase_estimates(*args)
        self.assertTrue(all(np.isnan(f['angles'][1]) for f in result[2:]))
        self.assertTrue(all(f['status'][1] == 'unresolved' for f in result[2:]))
        self.assertTrue(all(not f['turn_count_valid'][1] for f in result[2:]))
        self.assertTrue(all(np.isnan(f['angular_velocity'][1]) for f in result[2:]))
        self.assertFalse(summary['phase_bridge_gaps'][0]['short_bridge_estimated'])
        self.assertEqual(summary['estimated_event_fraction']['red'], 2/6)

    def test_rank_unanchored_pixel_fit_cannot_keep_a_measured_phase_label(self):
        args = list(case())
        args[2]['event_phase_connected'][2:, 0] = False
        # Even numerically finite pixel-fit angles are prior dependent when
        # their interpolation functional is outside the measured phase rank.
        result, _ = apply_phase_estimates(*args)
        for frame in result[2:]:
            self.assertEqual(frame['status'][1], 'phase_estimated')
            self.assertEqual(frame['strict_status'][1], 'unresolved')
            self.assertTrue(np.isnan(frame['strict_angles'][1]))
            self.assertEqual(frame['component_measurement_source'][1],
                             'joint_metric_pixels_with_prior_dependent_phase')

    def test_known_components_keys_corrected_times_and_inputs_are_preserved(self):
        args = list(case())
        evidence = args[2]
        evidence['event_phase_connected'][3:, 0] = False
        # Evidence array storage order need not match the result frame order;
        # identity must come from camera/source-frame, with physical times.
        order = np.array([4, 1, 5, 0, 3, 2])
        args[2] = {key: value[order].copy() for key, value in evidence.items()}
        before = pickle.dumps(args)
        original = copy.deepcopy(args[0])
        result, summary = apply_phase_estimates(*args)
        self.assertEqual(pickle.dumps(args), before)
        self.assertEqual([(f['camera'], f['source_frame']) for f in result],
                         [(f['camera'], f['source_frame']) for f in original])
        for old, new in zip(original, result):
            self.assertEqual(new['time_s'], old['time_s'])
            np.testing.assert_array_equal(np.asarray(new['angles'])[[0, 2]], old['angles'][[0, 2]])
            self.assertEqual(new['status'][0], old['status'][0])
            self.assertEqual(new['status'][2], old['status'][2])
            self.assertTrue(new['turn_count_valid'][0])
            self.assertTrue(new['turn_count_valid'][2])
        self.assertEqual(summary['strict_metric_bundle_coverage'], args[1]['coverage'])


if __name__ == '__main__':
    unittest.main()
