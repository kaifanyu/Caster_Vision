"""Scientific replay: time sampling, physical surfaces and occlusion."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from dualcam.model import rotation_x, rotation_z, world_points
from dualcam.offline_render import (RenderState, TrajectorySampler, draw_mesh,
                                    draw_simulation, hemisphere_curves, nearest_frame,
                                    render_offline, visible_surface)


def frame(time, angles, status=None):
    return {'time_s': time, 'angles': angles, 'status': status or ['vision'] * 3,
            'std_rad': [.01] * 3, 'turn_count_valid': [True] * 3}


class ReplaySamplingTests(unittest.TestCase):
    def test_phase_estimates_keep_their_status_through_interpolation(self):
        for start, end in (('vision', 'phase_estimated'), ('phase_estimated', 'vision'),
                           ('phase_estimated', 'predicted'), ('phase_estimated', 'phase_estimated')):
            with self.subTest(start=start, end=end):
                sampler = TrajectorySampler([
                    frame(0., [0., 0., 0.], ['vision', start, 'vision']),
                    frame(.1, [1., 1., 1.], ['vision', end, 'vision'])])
                state = sampler.at(.05)
                self.assertEqual(state.status[1], 'phase_estimated')
                self.assertAlmostEqual(state.angles[1], .5)
                self.assertFalse(state.turn_count_valid[1])
                exact = sampler.at(0. if start == 'phase_estimated' else .1)
                self.assertEqual(exact.status[1], 'phase_estimated')
                self.assertFalse(exact.turn_count_valid[1])
        distant = TrajectorySampler([
            frame(0., [0., 0., 0.], ['vision', 'phase_estimated', 'vision']),
            frame(.2, [1., 1., 1.], ['vision', 'phase_estimated', 'vision'])])
        self.assertEqual(distant.at(.1).status[1], 'unresolved')
        self.assertTrue(np.isnan(distant.at(.1).angles[1]))

    def test_duplicate_events_use_last_combined_state_and_keep_full_turns(self):
        sampler = TrajectorySampler([frame(0, [0, 0, 0]), frame(.1, [1, 2, 3]),
                                     frame(.1, [2, 4 * np.pi, 6])])
        np.testing.assert_allclose(sampler.at(.1).angles, [2, 4 * np.pi, 6])
        np.testing.assert_allclose(sampler.at(.05).angles, [1, 2 * np.pi, 3])

    def test_no_extrapolation_no_gap_fill_no_unresolved_placeholders(self):
        sampler = TrajectorySampler([frame(0, [0, 99, 0], ['vision', 'unresolved', 'vision']),
                                     frame(.1, [1, 2, 3]), frame(.5, [3, 4, 5])])
        self.assertTrue(np.isnan(sampler.at(-.01).angles).all())
        self.assertTrue(np.isnan(sampler.at(.51).angles).all())
        self.assertTrue(np.isnan(sampler.at(.3).angles).all())
        self.assertTrue(np.isnan(sampler.at(.05).angles[1]))
        self.assertFalse(sampler.at(.05).turn_count_valid[1])
        self.assertTrue(np.isnan(sampler.at(0).angles[1]))

    def test_component_prediction_and_conservative_uncertainty(self):
        frames = [frame(0, [0, 0, 0]), frame(.1, [1, 1, 1], ['vision', 'predicted', 'vision'])]
        frames[1]['std_rad'] = [.02, .03, .04]
        state = TrajectorySampler(frames).at(.05)
        self.assertEqual(state.status, ('vision', 'predicted', 'vision'))
        np.testing.assert_allclose(state.std_rad, [.02, .03, .04])

    def test_frame_selection_uses_timestamps_and_is_monotone(self):
        times = np.array([.009, .043, .095, .13])
        selected = [nearest_frame(times, t) for t in np.arange(0, .16, 1 / 30)]
        self.assertEqual(selected, sorted(selected))
        self.assertEqual(nearest_frame(times, .08), 2)
        self.assertEqual(nearest_frame(np.array([0., 1.]), .5), 0)


class HemisphereRenderingTests(unittest.TestCase):
    def test_mesh_has_physical_signed_rims_and_independent_phase(self):
        F = rotation_z(.4) @ rotation_x(.2)
        pivot = np.array([.02, -.03, .45])
        q = np.array([.3, .8, -.2])
        B = F @ rotation_x(q[0])
        for shell, sign in ((0, 1), (1, -1)):
            curves = hemisphere_curves(sign)
            rim = next(points for points, kind in curves if kind == 'rim')
            xyz = world_points(F, pivot, q, shell, rim, .1, .02)
            local = (xyz - pivot) @ B
            np.testing.assert_allclose(local[:, 2], sign * .01, atol=1e-15)
            np.testing.assert_allclose(np.linalg.norm(local[:, :2], axis=1), .1, atol=1e-15)
            for points, _ in curves:
                self.assertTrue(np.all(sign * points[:, 2] >= 0))
                np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1., atol=1e-15)

    def test_other_hemisphere_occludes_far_surface(self):
        # View along local +z: the negative shell is in front and hides +z.
        camera = {'R': np.eye(3), 't': np.zeros(3), 'K': np.eye(3)}
        centers = [np.array([0., 0., .51]), np.array([0., 0., .49])]
        points = np.array([[0., 0., .39], [0., 0., .61]])
        normals = np.array([[0., 0., -1.], [0., 0., 1.]])
        visibility = visible_surface(camera, points, normals, centers, np.eye(3), [1, -1], .1)
        np.testing.assert_array_equal(visibility, [True, False])

    def test_render_finite_states_and_hide_unknown_phase(self):
        F, pivot = np.eye(3), np.array([0., 0., .5])
        geometry = {'radius_m': .1, 'gap_m': .02, 'red_shell_sign': 1}
        known = RenderState(np.array([.2, .4, -.5]), ('vision',) * 3, np.zeros(3), np.ones(3, bool))
        unknown = TrajectorySampler.unknown()
        image = draw_simulation(500, 260, F, pivot, geometry, known)
        self.assertEqual(image.shape, (260, 500, 3))
        self.assertGreater(np.std(image), 15)
        camera = {'K': np.array([[500, 0, 250], [0, 500, 130], [0, 0, 1]]),
                  'R': np.eye(3), 't': np.zeros(3)}
        blank = np.zeros((260, 500, 3), np.uint8)
        self.assertFalse(draw_mesh(blank, camera, F, pivot, geometry, unknown).any())


class ReplayExportTests(unittest.TestCase):
    def test_short_movie_preserves_clock_duration_and_reuses_native_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            videos, frames = [], []
            K = np.array([[160., 0., 80.], [0., 160., 45.], [0., 0., 1.]])
            cameras = [{'K': K, 'R': np.eye(3), 't': np.array([shift, 0., 0.])}
                       for shift in (0., -.1)]
            infos = [{'K': K, 'dist': np.zeros(5), 'image_size': [160, 90]}] * 2
            for name in ('c920', 'brio101'):
                video = root / f'{name}.avi'
                videos.append(str(video))
                writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 15., (160, 90))
                if not writer.isOpened():
                    self.skipTest('OpenCV build has no MJPEG encoder')
                for index in range(3):
                    writer.write(np.full((90, 160, 3), 50 + index * 30, np.uint8))
                    frames.append({**frame(index / 15, [.14 + index * .1, index * .2, -index * .1]),
                                   'camera': name, 'source_frame': index})
                    if index > 0:
                        frames[-1]['status'][1] = 'phase_estimated'
                writer.release()
            report = {'kind': 'fused_motion', 'status': 'completed', 'initial_roll_deg': 8.,
                      'F': np.eye(3), 'pivot': [0., 0., .6], 'videos': videos,
                      'geometry': {'radius_m': .1, 'gap_m': .02, 'red_shell_sign': 1},
                      'frames': sorted(frames, key=lambda value: value['time_s']),
                      'use_video_manifest': False}
            with patch('dualcam.offline_render.load_rig', return_value=(cameras, infos, {})):
                result = render_offline(report, {}, root / 'rendered', fps=30,
                                        progress=None, transcode=False)
                self.assertEqual(result['frames'], 5)
                self.assertAlmostEqual(result['source_duration_s'], 2 / 15)
                self.assertEqual(len(result['stills']), 5)
                capture = cv2.VideoCapture(str(result['movie']))
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 5)
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 30.)
                ok, image = capture.read()
                capture.release()
                self.assertTrue(ok)
                self.assertEqual(image.shape, (900, 1600, 3))
                with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                    render_offline(report, {}, root / 'rendered', progress=None)


if __name__ == '__main__':
    unittest.main()
