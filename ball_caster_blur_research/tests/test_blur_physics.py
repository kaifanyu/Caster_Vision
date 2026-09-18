"""Physical and identifiability checks against independent synthetic imagery."""
import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from validate_blur_fit import AnalyticScene, make_fixture, coarse_hypotheses, analytic_texture
from blurtrack.blur_physics import ray_shell_normals, BlurFitProblem


class BlurPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_independent_two_camera_intersections_with_gap_and_roll(self):
        for shell in (0, 1):
            scene = AnalyticScene(shell=shell)
            for ci in (0, 1):
                pixels, expected, incidence = scene.pixels(ci, count=200)
                normals, valid, got_incidence = ray_shell_normals(
                    scene.cameras[ci], pixels, scene.F, scene.pivot,
                    scene.alpha, shell, scene.geometry)
                self.assertTrue(valid.all())
                np.testing.assert_allclose(normals, expected, atol=1e-12)
                np.testing.assert_allclose(got_incidence, incidence, atol=1e-12)

    def test_atlas_prediction_matches_independent_61_node_truth(self):
        for shell in (0, 1):
            fixture = make_fixture(noise=0., count=80, shell=shell)
            prediction, valid = fixture['problem'].predict(fixture['truth'])
            self.assertTrue(valid.all())
            self.assertLess(float(np.sqrt(np.mean((prediction-fixture['observed'])**2))), .00015)

    def test_signed_speed_and_phase_recovered_from_temporal_observations(self):
        fixture = make_fixture(noise=.002, count=90, speed_deg_s=-730.)
        truth = fixture['truth']
        fit = fixture['problem'].fit([truth+np.deg2rad([2., 35.]),
                                      [truth[0], -truth[1]]], maxiter=45)
        recovered = np.asarray(fit['best_parameters_rad'])
        error = np.rad2deg(recovered-truth)
        self.assertLess(abs(error[0]), .1)
        self.assertLess(abs(error[1]), 1.)
        self.assertEqual(fit['status'], 'candidate')
        self.assertFalse(fit['accuracy_validated'])
        self.assertFalse(fit['turn_count_valid'])

    def test_single_frame_symmetric_exposure_cannot_determine_direction(self):
        fixture = make_fixture(noise=0., count=100, camera_ids=(0,), one_frame=True)
        problem, truth = fixture['problem'], fixture['truth']
        forwards = problem.predict(truth)[0]
        backwards = problem.predict(truth*np.array([1., -1.]))[0]
        np.testing.assert_allclose(forwards, backwards, atol=1e-7)
        fit = problem.fit([truth+np.deg2rad([1., 20.]),
                           truth*np.array([1., -1.])+np.deg2rad([1., -20.])], maxiter=40)
        self.assertEqual(fit['status'], 'ambiguous')

    def test_coarse_search_recovers_fast_negative_motion_without_truth_seed(self):
        fixture = make_fixture(count=150, speed_deg_s=-1900.)
        seeds, search = coarse_hypotheses(fixture['problem'])
        self.assertEqual(search['scored_hypotheses'], 1176)
        fit = fixture['problem'].fit(seeds, phase_radius_deg=35., speed_radius_deg_s=350., maxiter=60)
        recovered = np.asarray(fit['best_parameters_rad'])
        delta = recovered-fixture['truth']
        phase_error = np.rad2deg(np.arctan2(np.sin(delta[0]), np.cos(delta[0])))
        self.assertLess(abs(phase_error), .1)
        self.assertLess(abs(np.rad2deg(delta[1])), 1.)

    def test_single_frame_speed_exposure_product_is_degenerate(self):
        original = make_fixture(noise=0., count=80, camera_ids=(0,), one_frame=True)
        longer = make_fixture(noise=0., count=80, camera_ids=(0,), one_frame=True, exposure_scale=1.2)
        parameters = original['truth'].copy()
        parameters[1] /= 1.2
        a = original['problem'].predict(original['truth'])[0]
        b = longer['problem'].predict(parameters)[0]
        np.testing.assert_allclose(a, b, atol=2e-7)

    def test_test_pixels_do_not_fit_camera_gain_or_offset(self):
        fixture = make_fixture(count=80)
        problem = fixture['problem']
        before = problem.score(fixture['truth'])
        problem.observed[problem.test] += 1.
        after = problem.score(fixture['truth'])
        self.assertAlmostEqual(before['train']['rmse'], after['train']['rmse'])
        np.testing.assert_array_equal(before['gain'], after['gain'])
        np.testing.assert_array_equal(before['offset'], after['offset'])
        self.assertGreater(after['test']['rmse'], .9)

    def test_textureless_shell_is_ambiguous(self):
        fixture = make_fixture(noise=0., count=60, flat=True)
        fit = fixture['problem'].fit([fixture['truth'], fixture['truth']+np.deg2rad([60., 400.])])
        self.assertFalse(fit['texture_informative'])
        self.assertEqual(fit['status'], 'ambiguous')

    def test_missing_texture_never_becomes_an_observed_pixel(self):
        fixture = make_fixture(noise=0., count=80, missing=True)
        score = fixture['problem'].score(fixture['truth'])
        self.assertLess(score['coverage'], .7)
        self.assertLess(score['covered_pixels'], score['eligible_pixels'])
        fixture['problem'].texture_valid.zero_()
        none = fixture['problem'].fit([fixture['truth']])
        self.assertEqual(none['status'], 'insufficient_texture')
        self.assertIsNone(none['best_parameters_rad'])

    def test_common_heldout_pixels_cannot_rank_candidates_without_common_training_pixels(self):
        # Each phase basin sees a different training region. The only common
        # observed texture belongs to held-out pixels, so candidate ordering
        # cannot use a finite common training objective.
        atlas = AnalyticScene().atlas(width=360, height=90)
        columns = (np.arange(360)+.5)*2*np.pi/360
        atlas['valid'][:, (columns>1.25*np.pi)&(columns<1.75*np.pi)] = False
        rng = np.random.default_rng(92)
        phi = np.repeat([0., np.pi/2, 3*np.pi/2], 100)+rng.uniform(-.01, .01, 300)
        theta = rng.uniform(.2, 1.3, 300)
        normal = np.column_stack((np.cos(phi)*np.sin(theta),
                                  np.sin(phi)*np.sin(theta), np.cos(theta)))
        observations = analytic_texture(phi, theta)[:, 0]-.19
        problem = BlurFitProblem(
            [atlas], np.broadcast_to(normal, (2, 1, 300, 3)).copy(),
            np.ones((2, 1, 300), bool), np.array([[0.], [.1]]), np.ones(1),
            np.zeros(2, int), np.broadcast_to(observations, (2, 300)).copy(),
            np.ones((2, 300), bool), 0.,
            train_mask=np.broadcast_to(np.arange(300)>=100, (2, 300)).copy(), device='cpu')
        fit = problem.fit([[0., 0.], [np.pi, 0.]], phase_radius_deg=1e-8,
                          speed_radius_deg_s=1e-8, maxiter=2)
        self.assertEqual(fit['common_comparison_pixels'], 200)
        self.assertFalse(fit['common_comparison_available'])
        self.assertEqual(fit['status'], 'ambiguous')


if __name__=='__main__': unittest.main()
