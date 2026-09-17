"""Integration checks for camera frames, timing, color identity and provenance."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from dualcam.config import DEFAULT_CONFIG, load_config, load_intrinsics, read_yaml
from dualcam.model import project, rotation_x, rotation_z, world_points
from dualcam.session import SelectedVideo, load_session, pair_timestamps, read_timestamps
from dualcam.solver import fit_joint
from dualcam.tracking import common_increments, initialize_angles, initialize_axes, masks_for, track_session


def ideal_increments(q, F, cameras):
    steps = np.zeros((2, 2, len(q)-1, 3, 3))
    for ci, camera in enumerate(cameras):
        for h in range(2):
            poses = [F @ rotation_x(a) @ rotation_z(b[h]) for a, *b in q]
            for f in range(len(q)-1):
                delta = poses[f+1] @ poses[f].T
                steps[ci, h, f] = camera['R'] @ delta @ camera['R'].T
    return {'times': np.arange(len(q))/30., 'increments': steps,
            'increment_support': np.full((2, 2, len(q)-1), 20, dtype=int)}


class InitializerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.F = Rotation.from_euler('xyz', [64., 13., 27.], degrees=True).as_matrix()
        self.cameras = [{'R': np.eye(3)}, {'R': Rotation.from_euler('xyz', [10., 30., 170.], degrees=True).as_matrix()}]

    def test_both_camera_coordinates_pool_to_same_axis_frame(self):
        q = np.zeros((14, 3))
        q[:, 0] = np.linspace(0., .5, len(q))
        roll = ideal_increments(q, self.F, self.cameras)
        q[:, 0] = 0
        q[:, 1] = np.linspace(0., .55, len(q))
        q[:, 2] = np.linspace(0., .35, len(q))
        swivel = ideal_increments(q, self.F, self.cameras)
        estimated, report = initialize_axes(roll, swivel, self.cameras,
                                             {'geometry': {}, 'tracking': {}})
        np.testing.assert_allclose(estimated, self.F, atol=1e-10)
        self.assertEqual(report['roll']['sample_count'], 52)
        self.assertEqual(report['swivel']['sample_count'], 52)
        roll['increments'][0, 0, 0] = np.nan
        self.assertEqual(len(common_increments(roll, self.cameras)), 51)

    def test_nonzero_roll_and_independent_mixed_motion_are_initialized(self):
        q = np.zeros((15, 3))
        q[:, 0] = .43 + np.linspace(0., .24, len(q))
        q[:, 1] = np.linspace(0., .67, len(q))
        q[:, 2] = np.linspace(0., -.51, len(q))
        tracked = ideal_increments(q, self.F, self.cameras)
        # Camera 2 can supply an increment missing from camera 1.
        tracked['increments'][0, 0, 5] = np.nan
        actual = initialize_angles(tracked, self.cameras, self.F, 'motion', initial_roll_rad=.43)
        np.testing.assert_allclose(actual, q, atol=1e-10)

    def test_red_identity_survives_camera_image_rotation(self):
        frame = np.full((128, 128, 3), 220, np.uint8)
        cv2.rectangle(frame, (48, 80), (80, 104), (0, 0, 255), -1)
        cv2.rectangle(frame, (48, 24), (80, 48), (0, 255, 0), -1)
        camera = read_yaml(DEFAULT_CONFIG)['cameras']['c920']
        masks = masks_for(frame, (64, 64, 61), camera)
        self.assertTrue(masks['top'][92, 64])  # red is physically below green
        self.assertTrue(masks['bottom'][36, 64])
        self.assertFalse(masks['top'][36, 64])
        flipped = masks_for(cv2.rotate(frame, cv2.ROTATE_180), (63, 63, 61), camera)
        self.assertTrue(flipped['top'][35, 63])
        self.assertTrue(flipped['bottom'][91, 63])


class TimestampTests(unittest.TestCase):
    def test_explicit_offset_aligns_clocks_and_preserves_sign(self):
        first = 100 + np.arange(8)/30.
        second = first + .027
        pairs, times, skew = pair_timestamps(first, second, second_offset_s=-.027, max_skew_s=.002)
        np.testing.assert_array_equal(pairs, np.column_stack((np.arange(8), np.arange(8))))
        np.testing.assert_allclose(times, first)
        np.testing.assert_allclose(skew, 0, atol=1e-12)

    def test_pairs_are_monotone_never_reuse_and_respect_max_skew(self):
        first = np.array([0., .010, .025, .040, .055, .070])
        second = np.array([.003, .033, .063])
        pairs, _, skew = pair_timestamps(first, second, max_skew_s=.008)
        self.assertGreater(len(pairs), 0)
        self.assertTrue(np.all(np.diff(pairs, axis=0) > 0))
        self.assertEqual(len(np.unique(pairs[:, 1])), len(pairs))
        self.assertTrue(np.all(abs(skew) <= .008))
        none, _, _ = pair_timestamps([0, 1, 2], [.5, 1.5, 2.5], max_skew_s=.01)
        self.assertEqual(none.shape, (0, 2))

    def test_invalid_times_and_nonconsecutive_csv_indices_are_rejected(self):
        for first in ([0, 0, 1], [0, float('nan'), 1], [1, 0]):
            with self.assertRaises(ValueError):
                pair_timestamps(first, [0., 1.])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'times.csv'
            path.write_text('frame_index,timestamp_s\n0,1.0\n2,1.1\n')
            with self.assertRaisesRegex(ValueError, 'consecutive'):
                read_timestamps(path)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.profile = {'width': 640, 'height': 480, 'fps': 30, 'fourcc': 'MJPG',
                        'exposure_us': 4000, 'gain': 0, 'white_balance_kelvin': 4000,
                        'focus': 35, 'zoom': 100, 'power_line_frequency': 2}
        self.path = self.directory/'intrinsics.yaml'
        self.calibration = {'K': [[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]],
                            'dist': [0., 0., 0., 0., 0.], 'image_size': [640, 480],
                            'capture_profile': self.profile.copy()}
        self.camera = {'capture': self.profile.copy(), 'intrinsics': str(self.path)}

    def write_intrinsics(self):
        self.path.write_text(yaml.safe_dump(self.calibration))

    def create_session(self, status='complete'):
        cameras = {name: {'capture': self.profile.copy()} for name in ('c920', 'brio101')}
        cfg = {'cameras': cameras, 'timing': {'brio_offset_s': -.004, 'max_pair_skew_ms': 3.}}
        metadata = {'status': status, 'cameras': {name: {'requested': self.profile.copy()} for name in cameras}}
        (self.directory/'session.json').write_text(json.dumps(metadata))
        for ci, name in enumerate(cameras):
            with (self.directory/f'{name}_timestamps.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['frame_index', 'timestamp_s'])
                writer.writeheader()
                for i in range(8):
                    writer.writerow({'frame_index': i, 'timestamp_s': 100+i/30+ci*.004})
        return cfg, metadata

    def test_intrinsics_resolution_and_complete_profile_must_match(self):
        self.write_intrinsics()
        info = load_intrinsics(self.camera)
        self.assertTrue(info['profile_verified'])
        self.camera['capture']['width'] = 1280
        with self.assertRaisesRegex(ValueError, 'resolution'):
            load_intrinsics(self.camera)
        self.camera['capture']['width'] = 640
        self.camera['capture']['focus'] = 40
        with self.assertRaisesRegex(ValueError, 'capture_profile'):
            load_intrinsics(self.camera)

    def test_empty_or_incomplete_profile_cannot_claim_verified(self):
        for profile in ({}, {'width': 640, 'height': 480}):
            self.calibration['capture_profile'] = profile
            self.write_intrinsics()
            with self.assertRaises(ValueError):
                load_intrinsics(self.camera)
        self.calibration['capture_profile'] = None
        self.write_intrinsics()
        self.assertFalse(load_intrinsics(self.camera)['profile_verified'])

    def test_session_pairing_and_requested_profile_are_checked(self):
        cfg, metadata = self.create_session()
        session = load_session(self.directory, cfg, max_frames=5)
        self.assertEqual(session['report']['paired_frames'], 5)
        np.testing.assert_allclose(session['times'], np.arange(5)/30)
        cfg['cameras']['c920']['capture']['exposure_us'] = 2000
        with self.assertRaisesRegex(ValueError, 'profile'):
            load_session(self.directory, cfg)
        cfg['cameras']['c920']['capture']['exposure_us'] = 4000
        metadata['cameras']['c920']['requested'] = {}
        (self.directory/'session.json').write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            load_session(self.directory, cfg)

    def test_incomplete_session_is_not_accepted_as_recorded(self):
        cfg, _ = self.create_session(status='recording')
        with self.assertRaises(ValueError):
            load_session(self.directory, cfg)

    def test_brio_focus_option_and_half_frame_pairing_tolerance_are_rejected(self):
        config = read_yaml(DEFAULT_CONFIG)
        config['cameras']['brio101']['capture']['focus'] = 20
        path = self.directory/'rig.yaml'
        path.write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, 'fixed-focus'):
            load_config(path)
        del config['cameras']['brio101']['capture']['focus']
        config['timing']['max_pair_skew_ms'] = 18
        path.write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, 'half a frame'):
            load_config(path)

    def test_video_reads_preserve_frame_indices_and_check_resolution(self):
        path = self.directory/'sample.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (64, 48))
        self.assertTrue(writer.isOpened())
        for i in range(4):
            writer.write(np.full((48, 64, 3), i*40, np.uint8))
        writer.release()
        video = SelectedVideo(path, [64, 48])
        try:
            self.assertLess(abs(float(video.read(0).mean())), 3)
            self.assertLess(abs(float(video.read(2).mean())-80), 3)
            with self.assertRaisesRegex(ValueError, 'increase'):
                video.read(2)
        finally:
            video.close()
        wrong = SelectedVideo(path, [128, 96])
        try:
            with self.assertRaisesRegex(ValueError, 'shape'):
                wrong.read(0)
        finally:
            wrong.close()


class RenderedTrackingTests(unittest.TestCase):
    def test_rendered_two_view_videos_track_and_fit_physical_motion(self):
        from test_solver import setup_scene
        cameras, F, C = setup_scene()
        for camera in cameras:
            camera['K'] = np.array([[760., 0., 320.], [0., 760., 240.], [0., 0., 1.]])
        # The second image is nearly upside down, while the shell labels stay physical.
        turn = rotation_z(np.pi)
        cameras[1]['R'] = turn @ cameras[1]['R']
        cameras[1]['t'] = turn @ cameras[1]['t']
        n = 9
        q = np.column_stack((np.linspace(0., .18, n), np.linspace(0., .25, n),
                             np.linspace(0., -.16, n)))
        rng = np.random.default_rng(21)
        patches = []
        for h in (0, 1):
            for _ in range(40):
                normal = rng.normal(size=3)
                normal /= np.linalg.norm(normal)
                normal[2] = (1-2*h)*abs(normal[2])
                if abs(normal[2]) < .12:
                    continue
                e1 = np.cross(normal, [0., 0., 1.]); e1 /= np.linalg.norm(e1)
                e2 = np.cross(normal, e1)
                theta = np.arange(3)*2*np.pi/3
                triangle = normal + .075*(np.cos(theta)[:, None]*e1 + np.sin(theta)[:, None]*e2)
                triangle /= np.linalg.norm(triangle, axis=1)[:, None]
                patches.append((h, normal, triangle))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cfg = read_yaml(DEFAULT_CONFIG)
            cfg['tracking'].update(max_corners=40, min_distance_px=4, klt_win=15)
            cfg['timing'].update(brio_offset_s=0., max_pair_skew_ms=3.)
            metadata = {'status': 'complete', 'mode': 'motion', 'cameras': {}}
            for ci, name in enumerate(('c920', 'brio101')):
                camera_cfg = cfg['cameras'][name]
                camera_cfg['capture'].update(width=640, height=480)
                camera_cfg['capture']['focus'] = 35 if ci == 0 else None
                K = cameras[ci]['K']
                path = directory/f'{name}.yaml'
                profile = camera_cfg['capture'].copy()
                path.write_text(yaml.safe_dump({'K': K.tolist(), 'dist': [0.]*5,
                                                'image_size': [640, 480], 'capture_profile': profile}))
                camera_cfg['intrinsics'] = str(path)
                center_uv = project(cameras[ci], C)[0]
                camera_cfg['circle'] = [*center_uv, 120.]
                metadata['cameras'][name] = {'requested': profile}
                writer = cv2.VideoWriter(str(directory/f'{name}.avi'),
                                         cv2.VideoWriter_fourcc(*'MJPG'), 30, (640, 480))
                self.assertTrue(writer.isOpened())
                for angles in q:
                    image = np.full((480, 640, 3), 180, np.uint8)
                    origin = -cameras[ci]['R'].T @ cameras[ci]['t']
                    for h, normal, triangle in patches:
                        xyz = world_points(F, C, angles, h, normal, .1, .02)[0]
                        center = world_points(F, C, angles, h, [0., 0., 0.], .1, .02)[0]
                        if (xyz-center) @ (origin-xyz) < .025:
                            continue
                        surface = world_points(F, C, angles, h, triangle, .1, .02)
                        vertices = np.rint(project(cameras[ci], surface)).astype(np.int32)
                        cv2.fillConvexPoly(image, vertices, (20, 20, 220) if h == 0 else (20, 160, 20))
                    writer.write(image)
                writer.release()
                with (directory/f'{name}_timestamps.csv').open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['frame_index', 'timestamp_s'])
                    writer.writerows((i, 100+i/30) for i in range(n))
            (directory/'session.json').write_text(json.dumps(metadata))
            tracked = track_session(directory, cfg, max_frames=n, progress=None)
            obs = tracked['observations']
            self.assertEqual(set(obs['camera']), {0, 1})
            self.assertEqual(set(obs['shell']), {0, 1})
            for ci in (0, 1):
                for h in (0, 1):
                    self.assertGreater(np.sum((obs['camera'] == ci) & (obs['shell'] == h)), 20)
            initial = initialize_angles(tracked, cameras, F, 'motion')
            out = fit_joint([{'times': tracked['times'], 'observations': obs,
                              'initial_angles': initial, 'mode': 'motion'}],
                            cameras, F, C, .1, .02)
            self.assertTrue(out['success'], out['diagnostics'])
            fitted = out['datasets'][0]
            self.assertGreater(fitted['valid'].mean(), .8)
            self.assertLess(np.rad2deg(np.nanmax(abs(fitted['angles']-q))), 3.)


class ObservationIndexTests(unittest.TestCase):
    def test_invalid_camera_and_track_identifiers_are_rejected(self):
        from test_solver import make_clip, setup_scene
        cameras, F, C = setup_scene()
        original, _ = make_clip(cameras, F, C, 'motion', count=4)
        for key, bad in (('camera', -1), ('camera', 2), ('shell', -1), ('track', -1), ('frame', -1)):
            clip = copy.deepcopy(original)
            clip['observations'][key][0] = bad
            with self.assertRaises(ValueError):
                fit_joint([clip], cameras, F, C, .1, .02)
        clip = copy.deepcopy(original)
        del clip['observations']['track']
        with self.assertRaisesRegex(ValueError, 'Missing observation'):
            fit_joint([clip], cameras, F, C, .1, .02)
        clip = copy.deepcopy(original)
        clip['observations']['camera'] = clip['observations']['camera'].astype(float)
        clip['observations']['camera'][0] = .5
        with self.assertRaisesRegex(ValueError, 'integer'):
            fit_joint([clip], cameras, F, C, .1, .02)


if __name__ == '__main__':
    unittest.main()
