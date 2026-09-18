import copy
import unittest

import numpy as np

from blurtrack.camera_ablation import prepare_camera_data


class CameraAblationTests(unittest.TestCase):
    def fixture(self):
        data = {
            "times": [np.array([0., .5, 1.]), np.array([.1, .6, 1.1])],
            "raw_times": [np.array([100., 100.5, 101.]), np.array([100.1, 100.6, 101.1])],
            "provenance": {"nested": {"timestamp_convention": "recorded"}},
            "observations": {"camera": np.array([0, 1, 0, 1]), "frame": np.array([0, 0, 2, 2]),
                             "uv": np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]]), "weight": np.ones(4)},
        }
        knots = np.array([0., .5, 1., 1.5])
        q = np.column_stack((.2 + .1 * knots, 2 + 3 * knots, -1 - 4 * knots))
        return data, knots, q

    def test_brio_corrected_first_knot_and_spin_home_gauge(self):
        data, knots, q = self.fixture()
        prepared, angles, fit_offset, shifts = prepare_camera_data(data, [1], knots, q, .03)
        self.assertEqual(len(prepared["times"][0]), 0)
        np.testing.assert_allclose(prepared["times"][1], [.13, .63, 1.13])
        np.testing.assert_array_equal(prepared["observations"]["frame"], [0, 2])
        np.testing.assert_array_equal(prepared["observations"]["camera"], [1, 1])
        self.assertEqual(fit_offset, 0.)
        np.testing.assert_array_equal(shifts, [0, .03])
        first_event = min(t for times in prepared["times"] for t in times)
        self.assertAlmostEqual(first_event, .13)
        for spin in (1, 2):
            self.assertAlmostEqual(np.interp(first_event, knots, angles[:, spin]), 0., places=12)
            np.testing.assert_allclose(np.diff(angles[:, spin]), np.diff(q[:, spin]))
        np.testing.assert_array_equal(angles[:, 0], q[:, 0])

    def test_c920_only_drops_brio_timeline_without_clock_shift(self):
        data, knots, q = self.fixture()
        prepared, angles, offset, shifts = prepare_camera_data(data, [0], knots, q, .03)
        self.assertEqual(len(prepared["times"][1]), 0)
        np.testing.assert_array_equal(prepared["times"][0], data["times"][0])
        np.testing.assert_array_equal(angles[0, 1:], [0, 0])
        np.testing.assert_array_equal(shifts, [0, 0])
        self.assertEqual(offset, 0.)

    def test_both_cameras_preserve_times_gauge_and_fit_offset(self):
        data, knots, q = self.fixture()
        prepared, angles, offset, shifts = prepare_camera_data(data, [1, 0], knots, q, .03)
        for ci in (0, 1):
            np.testing.assert_array_equal(prepared["times"][ci], data["times"][ci])
        np.testing.assert_array_equal(angles, q)
        np.testing.assert_array_equal(prepared["observations"]["camera"], data["observations"]["camera"])
        np.testing.assert_array_equal(shifts, [0, 0])
        self.assertEqual(offset, .03)

    def test_inputs_raw_times_and_provenance_unchanged_and_copies_independent(self):
        data, knots, q = self.fixture()
        original = copy.deepcopy(data)
        q_original = q.copy()
        prepared, angles, _, _ = prepare_camera_data(data, [1], knots, q, .03)
        for ci in (0, 1):
            np.testing.assert_array_equal(data["times"][ci], original["times"][ci])
            np.testing.assert_array_equal(prepared["raw_times"][ci], original["raw_times"][ci])
        self.assertEqual(prepared["provenance"], original["provenance"])
        np.testing.assert_array_equal(q, q_original)
        prepared["raw_times"][0][0] = -1
        prepared["provenance"]["nested"]["timestamp_convention"] = "changed"
        prepared["observations"]["uv"][0, 0] = -100
        angles[0, 0] = 99
        np.testing.assert_array_equal(data["raw_times"][0], original["raw_times"][0])
        np.testing.assert_array_equal(data["observations"]["uv"], original["observations"]["uv"])
        self.assertEqual(data["provenance"], original["provenance"])
        np.testing.assert_array_equal(q, q_original)

    def test_empty_active_timeline_and_observations_rejected(self):
        data, knots, q = self.fixture()
        data["times"][1] = np.empty(0)
        with self.assertRaises(ValueError):
            prepare_camera_data(data, [1], knots, q, .03)
        data, knots, q = self.fixture()
        data["observations"] = {key: value[:0] for key, value in data["observations"].items()}
        with self.assertRaises(ValueError):
            prepare_camera_data(data, [1], knots, q, .03)


if __name__ == "__main__":
    unittest.main()
