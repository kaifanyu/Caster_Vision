import unittest
import numpy as np
from blurtrack.texture_atlas import (material_grid,atlas_coordinates,srgb_to_linear,
    linear_to_srgb,visible_projection,robust_fuse,select_diverse_samples,reference_candidates,_sample_reference)

class TextureAtlasTests(unittest.TestCase):
    def test_grid_coordinates_roundtrip_both_shells(self):
        yy,xx=np.mgrid[:12,:32]
        for sign in (-1,1):
            grid=material_grid(12,32,sign)
            np.testing.assert_allclose(np.linalg.norm(grid,axis=-1),1,atol=1e-14)
            np.testing.assert_allclose(atlas_coordinates(grid,12,32,sign),np.stack((xx,yy),axis=-1),atol=1e-12)

    def test_transfer_function_roundtrip(self):
        rgb=np.linspace(0,1,101,dtype=np.float32)
        np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(rgb)),rgb,atol=2e-7)

    def test_front_shell_occludes_back_shell(self):
        camera={'R':np.eye(3),'t':np.zeros(3),'K':np.array([[500,0,320],[0,500,240],[0,0,1]])}
        geometry={'radius_m':.1,'gap_m':.02,'red_shell_sign':1}
        # Camera looks +z: green shell at negative z faces camera; red is rear.
        _,red,_=visible_projection(camera,np.eye(3),np.array([0,0,1]),np.zeros(3),0,
                                    material_grid(16,32,1),geometry)
        _,green,_=visible_projection(camera,np.eye(3),np.array([0,0,1]),np.zeros(3),1,
                                      material_grid(16,32,-1),geometry)
        self.assertFalse(red.any());self.assertTrue(green.any())

    def test_robust_fusion_rejects_corruption_and_leaves_holes(self):
        samples=np.full((4,2,3,3),.4,np.float32);samples[3]=.95
        weights=np.ones((4,2,3),np.float32);weights[:,0,0]=0;weights[1:,0,1]=0
        fused=robust_fuse(samples,weights)
        np.testing.assert_allclose(fused['rgb'][1],.4,atol=1e-7)
        self.assertFalse(fused['valid'][0,0]);self.assertFalse(fused['valid'][0,1])
        self.assertEqual(fused['count'][1,1],3)
        self.assertTrue((fused['rgb'][0,:2]==0).all())

    def test_diversity_selects_different_material_views(self):
        weights=np.zeros((3,2,4),np.float32);weights[0,:,:2]=1;weights[1,:,:2]=.99;weights[2,:,2:]=.8
        self.assertEqual(select_diverse_samples(weights,2),[0,2])

    def test_duplicate_home_views_do_not_crowd_out_lower_quality_new_material(self):
        weights=np.zeros((10,2,4),np.float32);weights[:9,:,:2]=1;weights[9,:,2:]=.02
        views=np.zeros((10,2));views[9]=[.2,.3]
        chosen=select_diverse_samples(weights,10,views)
        self.assertIn(9,chosen);self.assertLessEqual(len(chosen),5)

    def test_references_exclude_failed_phase_and_reserved_control(self):
        frames=[{'camera':'c920','time_s':float(t),'angles':[.1*t,.03*t,0.],
                  'status':['vision','vision','vision'],'source_frame':i}
                 for i,t in enumerate(np.arange(0,12,.1))]
        for frame in frames:
            if frame['time_s']>9:frame['status'][1]='phase_estimated'
        references=reference_candidates(frames,'c920',0,10.5,.0156,450,count=48)
        self.assertTrue(references)
        self.assertTrue(all(f['time_s']<=9 and not 7.7<=f['time_s']<=8.3 for f in references))
        self.assertTrue(all(f['predicted_blur_px']<=6 for f in references))

    def test_bright_yoke_exclusion_applies_to_reference_atlas_samples(self):
        camera={'R':np.eye(3),'t':np.zeros(3),'K':np.array([[500,0,960],[0,500,524],[0,0,1]])}
        geometry={'radius_m':.1,'gap_m':.02,'red_shell_sign':1}
        image=np.full((1080,1920,3),180,np.uint8)
        args=(image,camera,np.eye(3),np.array([0.,0.,1.]),np.zeros(3),1,material_grid(16,32,-1),geometry,0.)
        _,unmasked,_=_sample_reference(*args)
        _,masked,_=_sample_reference(*args,camera_name='c920')
        self.assertGreater(np.count_nonzero(unmasked),0)
        self.assertEqual(np.count_nonzero(masked),0)

if __name__=='__main__':unittest.main()
