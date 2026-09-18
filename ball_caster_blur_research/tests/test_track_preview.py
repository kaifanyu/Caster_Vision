import unittest
from pathlib import Path

import numpy as np

from blurtrack.track_preview import _annotate, _select_window


class TrackPreviewTests(unittest.TestCase):
    def test_center_window_and_missing_frame(self):
        entries = [(Path('first.npz'), {'source_start': 0, 'source_end': 59}),
                   (Path('second.npz'), {'source_start': 40, 'source_end': 99})]
        self.assertEqual(_select_window(entries, 44)[0], Path('first.npz'))
        self.assertEqual(_select_window(entries, 55)[0], Path('second.npz'))
        self.assertIsNone(_select_window(entries, 120))

    def test_native_coordinates_candidates_rejections_and_anchor_counts(self):
        frame = np.zeros((80, 100, 3), np.uint8)
        saved = {'tracks_native': np.array([[[20., 20.], [50., 40.], [70., 60.], [85., 20.]]]),
                 'accepted': np.array([[True, True, False, False]]),
                 'shells': np.array([0, 1, 0, 1]),
                 'visibility': np.array([[1., 1., .9, .2]]),
                 'queries_crop': np.array([[0., 1., 1.], [4., 2., 2.], [4., 3., 3.], [4., 4., 4.]])}
        overlay, counts = _annotate(frame, saved, 0, {})
        self.assertEqual(counts['accepted_red'], 1)
        self.assertEqual(counts['accepted_green'], 1)
        self.assertEqual(counts['accepted_query_anchors'], 1)
        self.assertEqual(counts['rejected_model_visible'], 1)
        self.assertGreater(overlay[20, 20, 2], overlay[20, 20, 1])
        self.assertGreater(overlay[40, 50, 1], overlay[40, 50, 2])
        self.assertGreater(overlay[60, 70].sum(), 0)
        self.assertEqual(overlay[20, 85].sum(), 0)
        self.assertEqual(frame.sum(), 0)


if __name__ == '__main__':
    unittest.main()
