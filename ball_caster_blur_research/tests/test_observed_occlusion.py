import unittest
import numpy as np
from blurtrack.observed_occlusion import (RECORDING_BOUNDS,static_yoke_mask,
    observed_shell_mask,occlusion_metadata)

def native(camera,normalized):
    b=np.asarray(RECORDING_BOUNDS[camera]);return np.asarray(normalized)*(b[2:]-b[:2])+b[:2]-.5

class OcclusionTests(unittest.TestCase):
    def test_reviewed_bright_yoke_pixels_excluded(self):
        for camera,points in [('c920',[[.53,.23],[.50,.60],[.48,.95]]),
                              ('brio101',[[.65,.02],[.80,.35],[.86,.70]])]:
            self.assertTrue(static_yoke_mask(camera,native(camera,points)).all())

    def test_clear_shell_reference_pixels_remain(self):
        for camera in ('c920','brio101'):
            self.assertTrue(observed_shell_mask(camera,native(camera,[[.15,.4],[.30,.65],[.5,.1]])).all())

    def test_bounds_do_not_silently_relocate_manual_mask(self):
        with self.assertRaises(ValueError):static_yoke_mask(0,np.zeros((2,2)),[0,0,100,100])
        with self.assertRaises(ValueError):static_yoke_mask('unknown',np.zeros((2,2)))

    def test_native_pixel_margin_dilates_and_preserves_shape(self):
        points=native('c920',[[[.625,.40],[.626,.40]]])
        self.assertFalse(static_yoke_mask('c920',points,margin_px=0).any())
        self.assertTrue(static_yoke_mask('c920',points,margin_px=20).all())
        self.assertEqual(static_yoke_mask('c920',points).shape,(1,2))

    def test_metadata_exposes_data_specific_polygon_provenance(self):
        meta=occlusion_metadata()
        self.assertEqual(len(meta['source_sha256']),64)
        self.assertEqual(meta['reference_frames']['c920'],[231,556])

if __name__=='__main__':unittest.main()
