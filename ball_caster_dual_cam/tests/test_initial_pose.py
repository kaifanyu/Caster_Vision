"""The starting rim cue must identify tilt without inventing blank-image pose."""
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dualcam.initial_pose import fit_initial_pose


def scene(angles):
    K = np.array([[1000., 0., 960.], [0., 1000., 540.], [0., 0., 1.]])
    pivot = np.array([.02, -.01, .8])
    eye = np.array([-.38, .04, .05])
    forward = pivot-eye
    forward /= np.linalg.norm(forward)
    right = np.cross([0., 1., 0.], forward)
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(forward, right), forward])
    cameras = [{'K': K, 'R': np.eye(3), 't': np.zeros(3)},
               {'K': K.copy(), 'R': R, 't': -R @ eye}]
    F = Rotation.from_euler('xyz', [72., 8., 15.], degrees=True).as_matrix()
    # Compact images keep these image-level regressions inexpensive.
    for camera in cameras:
        camera['K'][:2] *= .5
        camera['K'][[0, 1], [0, 1]] *= 2.5
    config = {'geometry': {'radius_m': .1, 'gap_m': .02, 'red_shell_sign': 1},
              'cameras': {}}
    frames = []
    for ci, name in enumerate(('c920', 'brio101')):
        camera = cameras[ci]
        config['cameras'][name] = {
            'circle': [480., 270., 150.],
            'segment': {'red_hsv': {'lo': [170, 130, 60], 'hi': [12, 255, 255]},
                        'green_hsv': {'lo': [40, 70, 30], 'hi': [100, 255, 255]}}}
        image = np.zeros((540, 960, 3), np.uint8)
        phi = np.linspace(0, 2*np.pi, 4000)
        B = F @ Rotation.from_euler('x', angles[ci], degrees=True).as_matrix()
        eye = -camera['R'].T @ camera['t']
        for sign in (-1, 1):
            center = pivot + .01*sign*B[:, 2]
            normals = np.column_stack((np.cos(phi), np.sin(phi), np.zeros(len(phi)))) @ B.T
            xyz = center + .1*normals
            visible = np.einsum('ij,ij->i', normals, eye-xyz) > 0
            camera_xyz = xyz @ camera['R'].T + camera['t']
            homogeneous = camera_xyz @ camera['K'].T
            uv = homogeneous[:, :2]/homogeneous[:, 2:]
            for point in uv[visible]:
                cv2.circle(image, tuple(np.rint(point).astype(int)), 1, (230, 230, 230), -1)
        frames.append(image)
    return cameras, frames, config, F, pivot


class InitialPoseTests(unittest.TestCase):
    def test_two_images_recover_signed_tilt_instead_of_assuming_seed(self):
        cameras, frames, cfg, F, pivot = scene([-7., -7.])
        result = fit_initial_pose(cameras, frames, cfg, F, pivot, initial_roll_deg=8.)
        self.assertTrue(result['accepted'], result['rejection_reasons'])
        self.assertAlmostEqual(result['initial_roll_deg'], -7., delta=1.)
        self.assertEqual(result['supplied_seed_deg'], 8.)
        self.assertEqual(len(result['rim_support']), 4)

    def test_blank_views_preserve_explicit_seed_and_report_rejection(self):
        cameras, frames, cfg, F, pivot = scene([8., 8.])
        result = fit_initial_pose(cameras, [np.zeros_like(f) for f in frames], cfg, F, pivot)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['initial_roll_deg'], 8.)
        self.assertTrue(result['rejection_reasons'])

    def test_contradictory_camera_rims_cannot_establish_reference(self):
        cameras, frames, cfg, F, pivot = scene([-10., 20.])
        result = fit_initial_pose(cameras, frames, cfg, F, pivot)
        self.assertFalse(result['accepted'])
        self.assertTrue(any('disagree' in reason for reason in result['rejection_reasons']))


if __name__ == '__main__':
    unittest.main()
