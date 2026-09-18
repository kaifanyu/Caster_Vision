"""Physical and observability tests for diagnostic angular-rate extraction."""
import copy
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from blurtrack.rate_analysis import CubicRateCurve, MeasurementDesign, annotate_rates


def fixture(knots, angles, *, query=None, missing_interval=None):
    query = np.asarray(knots if query is None else query)
    first, second = knots[:-1], knots[1:]
    if missing_interval is not None:
        a, b = missing_interval
        keep = (second <= a) | (first >= b)
        first, second = first[keep], second[keep]
    report = {'F': np.eye(3).tolist(), 'frames': [
        {'time_s': float(t), 'angles': [float(np.interp(t, knots, angles[:, k])) for k in range(3)],
         'strict_angles': [float(np.interp(t, knots, angles[:, 0])), None, None],
         'status': ['vision', 'phase_estimated', 'phase_estimated'],
         'strict_status': ['vision', 'unresolved', 'unresolved'],
         'turn_count_valid': [True, False, False]} for t in query]}
    evidence = {'knots': knots, 'event_times_s': query,
                'event_locally_supported': np.ones((len(query), 2), bool),
                'edge_times_first': np.tile(first, 2),
                'edge_times_second': np.tile(second, 2),
                'edge_shells': np.repeat(np.arange(2), len(first)),
                'edge_confidence': np.ones(len(first)*2),
                'edge_residual_deg': np.zeros(len(first)*2),
                'roll_observation_times_s': knots}
    return report, evidence


class RateAnalysisTests(unittest.TestCase):
    def test_cubic_functional_matches_analytic_derivative(self):
        knots = np.linspace(0., 2., 61)
        angles = np.column_stack((.14+.7*knots+.2*knots**2,
                                  -.5+2.3*knots-.3*knots**2,
                                  .8-1.2*knots+.1*knots**2))
        curve = CubicRateCurve(knots, angles)
        query = np.linspace(.007, 1.993, 311)
        _, rates, functional, valid = curve.evaluate(query)
        expected = np.column_stack((.7+.4*query, 2.3-.6*query, -1.2+.2*query))
        self.assertTrue(valid.all())
        np.testing.assert_allclose(rates, curve.spline(query, 1), atol=1e-12)
        np.testing.assert_allclose(rates, expected, atol=1e-12)
        np.testing.assert_allclose(np.asarray(functional.sum(axis=1)), 0., atol=1e-12)

    def test_physical_rig_angular_velocity(self):
        knots = np.linspace(0., 2., 61)
        angles = np.column_stack((.14+.3*knots, .9*knots, -1.7*knots))
        query = np.linspace(.01, 1.99, 109)
        report, evidence = fixture(knots, angles, query=query)
        # A nontrivial proper rig rotation checks the frame convention.
        F = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        report['F'] = F.tolist()
        original = copy.deepcopy(report)
        result = annotate_rates(report, knots, angles, evidence)
        self.assertEqual(report, original)
        for t, frame in zip(query, result['frames']):
            self.assertEqual(frame['rate_status'], ['vision']*3)
            self.assertEqual(frame['phase_status'], ['vision', 'phase_estimated', 'phase_estimated'])
            self.assertEqual(frame['turn_count_valid'], [True, False, False])
            np.testing.assert_allclose(frame['angular_velocity'], [.3, .9, -1.7], atol=1e-12)
            alpha = .14+.3*t
            axis = F @ [0., -np.sin(alpha), np.cos(alpha)]
            np.testing.assert_allclose(frame['angular_velocity_rig_red'], .3*F[:, 0]+.9*axis, atol=1e-12)
            np.testing.assert_allclose(frame['angular_velocity_rig_green'], .3*F[:, 0]-1.7*axis, atol=1e-12)
        self.assertFalse(result['rate_analysis']['pose_fit_modified'])

    def test_disconnected_phase_offset_keeps_local_rates_identifiable(self):
        knots = np.linspace(0., 3., 91)
        angles = np.column_stack((.1+.2*knots, 1.1*knots, -.6*knots))
        report, evidence = fixture(knots, angles, missing_interval=(1., 1.8))
        baseline = annotate_rates(report, knots, angles, evidence, max_prediction_s=0.)
        shifted = angles.copy()
        shifted[knots >= 1.8, 1] += 7.2
        shifted[knots >= 1.8, 2] -= 3.9
        offset_report = annotate_rates(report, knots, shifted, evidence, max_prediction_s=0.)
        for i, t in enumerate(knots):
            if .1 < t < .8 or 2.0 < t < 2.8:
                self.assertEqual(baseline['frames'][i]['rate_status'], ['vision']*3)
                self.assertEqual(offset_report['frames'][i]['rate_status'], ['vision']*3)
                np.testing.assert_allclose(baseline['frames'][i]['angular_velocity'],
                                           offset_report['frames'][i]['angular_velocity'], atol=1e-12)
            if 1.1 < t < 1.7:
                self.assertEqual(offset_report['frames'][i]['rate_status'][1:], ['unresolved']*2)
                self.assertEqual(offset_report['frames'][i]['angular_velocity'][1:], [None, None])
        red_design = baseline['rate_analysis']['measurement_only_design'][1]['components']
        self.assertTrue(any(d['nullity_without_phase_anchor'] == 1 and d['knots'] > 5 for d in red_design))
        self.assertTrue(all(d['regularization_rows'] == 0 for d in red_design))

    def test_missing_local_geometry_does_not_get_vision_from_rank(self):
        knots = np.linspace(0., 1., 31)
        angles = np.column_stack((knots, 2*knots, -knots))
        report, evidence = fixture(knots, angles)
        evidence['event_locally_supported'][(knots > .2) & (knots < .8)] = False
        result = annotate_rates(report, knots, angles, evidence, max_prediction_s=.1)
        center = result['frames'][15]
        self.assertEqual(center['rate_status'], ['vision', 'unresolved', 'unresolved'])
        self.assertIsNone(center['angular_velocity'][1])
        self.assertIsNone(center['angular_velocity_rig_red'])
        self.assertEqual(result['frames'][7]['rate_status'][1:], ['predicted', 'predicted'])

    def test_temporally_connected_graph_does_not_imply_full_rate_rank(self):
        # One off-grid edge touches all three knot unknowns. It leaves an
        # additional changing-phase null mode beyond the constant offset.
        design = MeasurementDesign(csr_matrix([[-.5, .25, .25]]))
        identifiable, projection = design.identifiable(csr_matrix([[-.5, .25, .25], [-1., 1., 0.]]))
        self.assertEqual(identifiable.tolist(), [True, False])
        self.assertGreater(projection[1], .1)
        self.assertEqual(design.diagnostics[0]['nullity_without_phase_anchor'], 2)

    def test_empty_rejected_and_out_of_range_evidence(self):
        knots = np.linspace(0., 1., 31)
        angles = np.column_stack((knots, 2*knots, -knots))
        query = np.r_[-.1, knots, 1.1]
        report, evidence = fixture(knots, angles, query=query)
        evidence['edge_residual_deg'][:] = 3.
        result = annotate_rates(report, knots, angles, evidence)
        for frame in result['frames']:
            self.assertEqual(frame['rate_status'][1:], ['unresolved', 'unresolved'])
        self.assertEqual(result['frames'][0]['angular_velocity'], [None]*3)
        self.assertEqual(result['frames'][-1]['angular_velocity'], [None]*3)
        self.assertIsNone(result['frames'][0]['rate_support_interval_s'][0])

    def test_output_origin_shift_preserves_rates_support_and_phase(self):
        knots = np.linspace(0., 3., 91)
        angles = np.column_stack((.1+.2*knots+.1*knots**2,
                                  1.1*knots-.2*knots**2, -.6*knots+.3*knots**2))
        for explicit_roll_times in (False, True):
            report, evidence = fixture(knots, angles, missing_interval=(1., 1.8))
            for frame in report['frames']:
                frame['visual_components'] = [0, 1, 2]
            if not explicit_roll_times:
                evidence.pop('roll_observation_times_s')
            base = annotate_rates(report, knots, angles, evidence, max_prediction_s=0.)
            for shift in (-.137, .217):
                with self.subTest(explicit_roll_times=explicit_roll_times, shift=shift):
                    shifted_report = copy.deepcopy(report)
                    shifted_report['summary'] = {'output_clock_origin_shift_s': shift}
                    for frame in shifted_report['frames']:
                        frame['time_s'] -= shift
                    shifted = annotate_rates(shifted_report, knots, angles, evidence,
                                             max_prediction_s=0.)
                    self.assertEqual(shifted['rate_analysis']['output_clock_origin_shift_s'], shift)
                    for old, new in zip(base['frames'], shifted['frames']):
                        self.assertEqual(old['rate_status'], new['rate_status'])
                        self.assertEqual(old['phase_status'], new['phase_status'])
                        np.testing.assert_allclose(np.asarray(old['angular_velocity'], float),
                                                   np.asarray(new['angular_velocity'], float),
                                                   atol=1e-12, equal_nan=True)
                        np.testing.assert_allclose(old['angular_velocity_model_interpolant'],
                                                   new['angular_velocity_model_interpolant'], atol=1e-12)
                        for before, after in zip(old['rate_support_interval_s'], new['rate_support_interval_s']):
                            if before is None:
                                self.assertIsNone(after)
                            else:
                                np.testing.assert_allclose(np.asarray(before)-shift, after, atol=1e-12)

    def test_nearby_vision_flag_does_not_fill_actual_roll_measurement_gap(self):
        knots = np.linspace(0., 1., 31)
        angles = np.column_stack((knots, 2*knots, -knots))
        report, evidence = fixture(knots, angles)
        evidence.pop('roll_observation_times_s')
        for frame in report['frames']:
            # The pose status still says vision throughout this artificial gap,
            # reproducing the backend's nearby-event support propagation.
            frame['visual_components'] = [1, 2] if .3 < frame['time_s'] < .7 else [0, 1, 2]
        result = annotate_rates(report, knots, angles, evidence, max_prediction_s=0.)
        self.assertEqual(result['frames'][15]['rate_status'], ['unresolved', 'vision', 'vision'])
        self.assertEqual(result['frames'][4]['rate_status'], ['vision']*3)
        diagnostic = result['rate_analysis']['measurement_only_design'][0]
        self.assertEqual(diagnostic['observation_provenance'], 'strict_roll_with_actual_event_visual_support')
        self.assertLess(diagnostic['accepted_pose_times'], len(knots))


if __name__ == '__main__':
    unittest.main()
