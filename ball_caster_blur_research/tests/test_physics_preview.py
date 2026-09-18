"""Check physical frames, local support and provenance of the diagnostic viewer."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from blurtrack.physics_preview import (
    CASES, build_payload, material_axes, native_trial_allowed, sample_state, sha256,
)


class PhysicsPreviewTests(unittest.TestCase):
    def setUp(self):
        # More than one turn between knots: shortest-path interpolation would
        # give the wrong pose and reverse the meaning of the fitted rate.
        self.payload = {'baseline': {
            'times': [0., .5, 1.],
            'angles': [[.1, 6., -.2], [.3, 10., .4], [.5, 14., 1.]],
        }}
        self.case = {
            'status': 'candidate', 'parameters_rad': [.75, -30.],
            'shell_index': 1, 'center_s': .5, 'start_s': .4, 'end_s': .6,
            'frames': [
                {'camera': 'c920', 'frame': 12, 'time_s': .4},
                {'camera': 'c920', 'frame': 13, 'time_s': .55},
                {'camera': 'brio101', 'frame': 13, 'time_s': .45},
                {'camera': 'brio101', 'frame': 14, 'time_s': .6},
            ],
        }

    def test_candidate_is_not_extrapolated_or_blended_at_support_boundaries(self):
        original = copy.deepcopy(self.payload)
        for t in (.4, .45, .5, .6):
            with self.subTest(time=t):
                state = sample_state(self.payload, self.case, t)
                self.assertTrue(state['trial_active'])
                self.assertAlmostEqual(state['trial'][2], .75 - 30.*(t-.5))
                # Selected shell's independently fitted phase may jump relative
                # to the prior. It must not be blended to hide that difference.
                np.testing.assert_array_equal(state['trial'][:2], state['baseline'][:2])
        for t in (np.nextafter(.4, -np.inf), np.nextafter(.6, np.inf), 0., 1.):
            state = sample_state(self.payload, self.case, t)
            self.assertFalse(state['trial_active'])
            self.assertIsNone(state['trial'])
        self.assertEqual(self.payload, original)

    def test_prior_keeps_unwrapped_motion_and_trial_uses_image_time(self):
        state = sample_state(self.payload, self.case, .45)
        np.testing.assert_allclose(state['baseline'], [.28, 9.6, .34], atol=1e-12)
        self.assertAlmostEqual(state['trial'][2], 2.25)
        # Native camera times are different even when source indices coincide.
        self.assertNotEqual(sample_state(self.payload, self.case, .55)['trial'][2],
                            sample_state(self.payload, self.case, .45)['trial'][2])

    def test_ambiguous_or_missing_candidate_never_acquires_a_trial(self):
        for changes in ({'status': 'ambiguous'}, {'parameters_rad': None},
                        {'status': 'insufficient_texture'}):
            case = dict(self.case, **changes)
            self.assertIsNone(sample_state(self.payload, case, .5)['trial'])
            self.assertFalse(native_trial_allowed(case, 'c920', 12))

    def test_nonfinite_time_is_rejected(self):
        for t in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                sample_state(self.payload, self.case, t)

    def test_native_membership_uses_camera_and_exact_fitted_source_index(self):
        self.assertTrue(native_trial_allowed(self.case, 'c920', 12))
        self.assertTrue(native_trial_allowed(self.case, 'brio101', 14))
        self.assertFalse(native_trial_allowed(self.case, 'brio101', 12))
        self.assertFalse(native_trial_allowed(self.case, 'c920', 14))
        self.assertFalse(native_trial_allowed(self.case, 'c920', 11))
        self.assertFalse(native_trial_allowed(self.case, 'unknown', 12))

    def test_shell_axes_obey_analytic_rotation_and_center_gap(self):
        # Independent closed-form Rx(alpha)Rz(beta), then a nontrivial cyclic
        # world-coordinate permutation so camera/home-frame confusion is caught.
        F = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        pivot = np.array([.2, -.3, .7])
        geometry = {'radius_m': .1, 'gap_m': .02, 'red_shell_sign': 1}
        q = [.23, -.72, 1.17]
        ca, sa = np.cos(q[0]), np.sin(q[0])
        axis = F @ [0., -sa, ca]
        centers = []
        for shell in (0, 1):
            center, axes = material_axes(F, pivot, geometry, q, shell)
            cb, sb = np.cos(q[shell+1]), np.sin(q[shell+1])
            expected = F @ np.array([[cb, -sb, 0.],
                                     [ca*sb, ca*cb, -sa],
                                     [sa*sb, sa*cb, ca]])
            np.testing.assert_allclose(axes, expected, atol=1e-14)
            np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(axes), 1.)
            np.testing.assert_allclose(center, pivot + (1-2*shell)*.01*axis, atol=1e-14)
            centers.append(center)
        np.testing.assert_allclose(centers[0]-centers[1], .02*axis, atol=1e-14)
        self.assertAlmostEqual(np.linalg.norm(centers[0]-centers[1]), .02)

    def test_spin_rotates_material_xy_but_never_roll_axis_or_shell_center(self):
        geometry = {'radius_m': .1, 'gap_m': .02, 'red_shell_sign': -1}
        q = [.4, .2, -.1]
        for shell in (0, 1):
            center, axes = material_axes(np.eye(3), np.zeros(3), geometry, q, shell)
            spun = q.copy()
            spun[shell+1] += np.pi/2
            new_center, new_axes = material_axes(np.eye(3), np.zeros(3), geometry, spun, shell)
            np.testing.assert_array_equal(new_center, center)
            np.testing.assert_allclose(new_axes[:, 2], axes[:, 2], atol=1e-14)
            np.testing.assert_allclose(new_axes[:, 0], axes[:, 1], atol=1e-14)
            np.testing.assert_allclose(new_axes[:, 1], -axes[:, 0], atol=1e-14)
            np.testing.assert_allclose(center, (2*shell-1)*.01*axes[:, 2], atol=1e-14)


class PhysicsPreviewProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        base = self.root/'experiments/hybrid_dual'
        base.mkdir(parents=True)
        geometry = {'radius_m': .1, 'gap_m': .02, 'red_shell_sign': 1}
        self.frames = [
            {'camera': 'c920', 'source_frame': 4, 'time_s': .4},
            {'camera': 'brio101', 'source_frame': 4, 'time_s': .41},
            {'camera': 'c920', 'source_frame': 5, 'time_s': .59},
            {'camera': 'brio101', 'source_frame': 5, 'time_s': .6},
        ]
        report = dict(F=np.eye(3).tolist(), pivot=[0., 0., .4], geometry=geometry,
                      config={}, videos=['c920.avi', 'brio101.avi'], initial_roll_deg=8.,
                      calibration_hashes={'intrinsics': 'fixture'}, frames=self.frames)
        self.report_path = base/'results.json'
        self.report_path.write_text(json.dumps(report))
        self.bundle_path = base/'bundle.npz'
        np.savez(self.bundle_path, knots=[0., 1.], angles=[[.1, 0., 0.], [.2, 1., 2.]])
        self.case_paths = []
        for identifier, label, folder in CASES:
            strict = 'strict' in identifier
            result = dict(status='ambiguous' if strict else 'candidate',
                          best_parameters_rad=None if strict else [.5, 5.],
                          illustrative_parameters_rad=[2., 10.],
                          accuracy_validated=True, turn_count_valid=True,
                          baseline_common_pixel_score={'test': {'rmse': .1, 'pixels': 50}},
                          fitted_common_pixel_score={'test': {'rmse': .125, 'pixels': 50}},
                          metadata=dict(shell='green' if 'green' in identifier else 'red',
                                        center_s=.5, baseline_parameters_rad=[.5, 2.],
                                        fixed_geometry=geometry, frames=[
                                            {'camera': f['camera'], 'frame': f['source_frame'],
                                             'time_s': f['time_s'], 'exposure_s': .008}
                                            for f in self.frames],
                                        atlas_manifest={'baseline_sha256': sha256(self.report_path),
                                                        'baseline_bundle_sha256': sha256(self.bundle_path)}))
            path = self.root/'experiments/blur_physics'/folder/'results.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(result))
            self.case_paths.append(path)

    def test_payload_is_local_unvalidated_and_ambiguous_diagnostics_stay_hidden(self):
        payload = build_payload(self.root)
        self.assertFalse(payload['applied_to_final_trajectory'])
        self.assertEqual(payload['default_case'], 'red_fast')
        self.assertEqual(payload['kind'], 'local_physics_hypothesis_preview')
        self.assertEqual(payload['calibration_hashes'], {'intrinsics': 'fixture'})
        for case in payload['cases']:
            self.assertEqual((case['start_s'], case['end_s']), (.4, .6))
            self.assertFalse(case['accuracy_validated'])
            self.assertFalse(case['turn_count_valid'])
            self.assertEqual(case['source_sha256'], sha256(case['source_result']))
            if 'strict' in case['id']:
                self.assertIsNone(case['parameters_rad'])
                self.assertFalse(case['include_in_video'])
            else:
                self.assertIn('held-out error worsened', case['heldout_decision'])

    def test_changed_report_or_bundle_cannot_reuse_frozen_texture_gauge(self):
        original = self.report_path.read_bytes()
        self.report_path.write_bytes(original+b'\n')
        with self.assertRaisesRegex(ValueError, 'material gauges'):
            build_payload(self.root)
        self.report_path.write_bytes(original)
        np.savez(self.bundle_path, knots=[0., 1.], angles=[[.1, 0., 0.], [.2, 9., 2.]])
        with self.assertRaisesRegex(ValueError, 'material gauges'):
            build_payload(self.root)

    def test_changed_fit_geometry_cannot_be_drawn_on_baseline_geometry(self):
        path = self.case_paths[0]
        result = json.loads(path.read_text())
        result['metadata']['fixed_geometry']['gap_m'] = .03
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, 'geometry differs'):
            build_payload(self.root)

    def test_wrong_native_frame_or_doubled_camera_clock_offset_is_rejected(self):
        path = self.case_paths[0]
        original = json.loads(path.read_text())
        for field, value in (('frame', 99), ('time_s', .41+.00613871341320485)):
            with self.subTest(field=field):
                result = copy.deepcopy(original)
                # Brio's saved timestamp already includes its offset. Applying
                # that correction twice can visibly rotate a fast-moving axis.
                result['metadata']['frames'][1][field] = value
                path.write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError, 'Fit frame timing differs'):
                    build_payload(self.root)
        path.write_text(json.dumps(original))
        build_payload(self.root)


if __name__ == '__main__':
    unittest.main()
