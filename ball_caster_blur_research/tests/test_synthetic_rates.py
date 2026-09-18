"""Angular-rate scoring must not silently bridge occlusion or query anchors."""
import unittest

import numpy as np

from scripts.validate_cotracker_gpu import conditional_rate_metrics


class SyntheticRateTests(unittest.TestCase):
    def test_query_frame_interval_excluded_even_with_finite_angle(self):
        result = conditional_rate_metrics(np.deg2rad([100, 1, 2, 3]), np.deg2rad([0, 1, 2, 3]), np.arange(4) * .1)
        self.assertEqual(result["scored_intervals"], 2)
        self.assertEqual(result["candidate_intervals_excluding_query"], 2)
        self.assertAlmostEqual(result["rmse"], 0., places=10)

    def test_missing_estimate_breaks_both_adjacent_intervals(self):
        truth = np.deg2rad([0, 1, 2, 3, 4, 5])
        estimate = truth.copy()
        estimate[3] = np.nan
        result = conditional_rate_metrics(estimate, truth, np.arange(6) * .1)
        self.assertEqual(result["scored_intervals"], 2)
        self.assertEqual(result["candidate_intervals_excluding_query"], 4)
        self.assertEqual(result["fraction_candidate_intervals_scored"], .5)
        self.assertAlmostEqual(result["mae"], 0., places=10)

    def test_shortest_increment_wrap_and_irregular_timestamps(self):
        truth = np.deg2rad([177, 179, 181, 184])
        estimate = np.deg2rad([177, 179, -179, -176])
        result = conditional_rate_metrics(estimate, truth, [0, .1, .3, .6])
        self.assertEqual(result["scored_intervals"], 2)
        self.assertAlmostEqual(result["absolute_error_p95"], 0., places=10)
        self.assertTrue(result["strictly_less_than_180deg_true_increment_assumption"])

    def test_errors_use_same_supported_truth_intervals(self):
        truth = np.deg2rad([0, 10, 30, 60])
        estimate = np.deg2rad([0, 10, 31, 59])
        result = conditional_rate_metrics(estimate, truth, [0, 1, 2, 3])
        # Scored errors are +1 and -2 deg/s; query interval is absent.
        self.assertAlmostEqual(result["signed_error_median"], -.5)
        self.assertAlmostEqual(result["absolute_error_median"], 1.5)
        self.assertAlmostEqual(result["mae"], 1.5)
        self.assertAlmostEqual(result["rmse"], np.sqrt(2.5))
        self.assertAlmostEqual(result["absolute_error_p95"], 1.95)

    def test_aliased_intervals_are_counted_and_not_scored(self):
        truth = np.deg2rad([0, 10, 210, 220])
        result = conditional_rate_metrics(truth, truth, np.arange(4))
        self.assertFalse(result["strictly_less_than_180deg_true_increment_assumption"])
        self.assertEqual(result["supported_intervals_rejected_for_aliasing"], 1)
        self.assertEqual(result["scored_intervals"], 1)
        self.assertAlmostEqual(result["rmse"], 0.)

    def test_no_supported_pairs_returns_null_errors(self):
        result = conditional_rate_metrics([np.nan, 1, np.nan, 3], [0, 1, 2, 3], [0, 1, 2, 3])
        self.assertEqual(result["scored_intervals"], 0)
        self.assertEqual(result["fraction_candidate_intervals_scored"], 0.)
        self.assertIsNone(result["mae"])
        self.assertIsNone(result["rmse"])


if __name__ == "__main__":
    unittest.main()
