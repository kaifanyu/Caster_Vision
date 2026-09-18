import unittest
import numpy as np
from blurtrack.observations import native_to_crop, crop_to_native, window_ranges, spatial_deduplicate, hybrid_observations, exclude_reference_neighborhood


class ObservationTests(unittest.TestCase):
    def test_crop_mapping_preserves_pixel_centers_and_roundtrips(self):
        bounds = (300, 21, 1350, 1070); shape = (640, 640)
        native = np.array([[300., 21.], [1050.25, 855.75], [1349., 1069.]])
        np.testing.assert_allclose(crop_to_native(native_to_crop(native, bounds, shape), bounds, shape), native, atol=1e-12)
        np.testing.assert_allclose(crop_to_native([-.5, -.5], bounds, shape), [299.5, 20.5])

    def test_windows_cover_every_frame_with_overlap(self):
        for n in (3, 48, 60, 61, 695, 728):
            covered = set()
            for lo, hi in window_ranges(n): covered.update(range(lo, hi))
            self.assertEqual(covered, set(range(n)))
        with self.assertRaises(ValueError): list(window_ranges(100, 60, 60))

    def fixture(self):
        return dict(camera=np.array([0,0,0,1]), shell=np.array([0,0,1,0]), frame=np.zeros(4,int),
                    track=np.arange(4), uv=np.array([[10.,10.],[11.,10.],[10.,10.],[10.,10.]]), weight=np.array([.6,.9,.8,.8]))

    def test_deduplicate_within_same_camera_shell_frame_only(self):
        obs = self.fixture(); result = spatial_deduplicate(obs)
        np.testing.assert_array_equal(result['track'], [1,2,3])
        result = spatial_deduplicate(obs, preferred=np.array([True,False,False,False]))
        np.testing.assert_array_equal(result['track'], [0,2,3])

    def test_hybrid_preserves_klt_and_camera_local_id_namespace(self):
        a = self.fixture(); b = self.fixture(); b['uv'] += [30,0]
        result = hybrid_observations(a,b)
        self.assertEqual(len(result['uv']), 6)
        self.assertGreater(result['track'].max(), a['track'].max())
        overlap = hybrid_observations(a,a)
        self.assertEqual(len(overlap['uv']), 3)

    def test_whole_reference_neighborhood_excluded(self):
        obs = self.fixture(); reference = {k:v[:1] for k,v in obs.items()}
        result = exclude_reference_neighborhood(obs,reference)
        np.testing.assert_array_equal(result['track'], [2,3])


if __name__ == '__main__': unittest.main()
