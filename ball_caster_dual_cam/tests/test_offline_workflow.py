"""Evidence-based native result support, schema and portable exports."""
import csv
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dualcam.config import write_json
from dualcam.fused_workflow import write_fused_csv
from dualcam.offline import NativeBundle, event_table
from dualcam.offline_render import TrajectorySampler
from dualcam.offline_workflow import run_joint, summarize_solution
from dualcam.visualization import build_viewer_payload
from tests.test_native_bundle import synthetic


def solution(*, drop_both_red=False, duplicate_timestamp=False):
    data, seed, cameras, F, pivot, geometry, pose = synthetic()
    if drop_both_red:
        observations = data['observations']
        times = np.array([data['times'][ci][fi] for ci, fi in
                          zip(observations['camera'], observations['frame'])])
        keep = ~((observations['shell'] == 0) & (times > .3) & (times < .84))
        data['observations'] = {key: value[keep] for key, value in observations.items()}
        seed['landmark'] = seed['landmark'][keep]
    if duplicate_timestamp:
        # Both cameras may have the same receive timestamp. Geometry is updated
        # consistently so this is a real supported simultaneous image event.
        data['times'][1][0] = data['times'][0][0]
        observations = data['observations']
        from dualcam.model import project, world_points
        rows = np.flatnonzero((observations['camera'] == 1) & (observations['frame'] == 0))
        for row in rows:
            observations['uv'][row] = project(cameras[1], world_points(
                F, pivot, pose(0.), observations['shell'][row],
                seed['points'][seed['landmark'][row]], .1, .02))[0]
        seed['events'] = event_table(data)
        seed['angles'] = np.array([pose(event[0]) for event in seed['events']])
    problem = NativeBundle(data, seed, cameras, F, pivot, geometry)
    frames, summary, _ = summarize_solution(problem, problem.x0, data, seed, F, pivot)
    return {'kind': 'fused_motion', 'status': 'completed', 'measurement_source': 'joint_metric',
            'initial_roll_deg': float(np.rad2deg(frames[0]['angles'][0])),
            'F': F, 'pivot': pivot, 'geometry': geometry, 'frames': frames, 'summary': summary}


class OfflineWorkflowTests(unittest.TestCase):
    def test_native_result_is_valid_saved_viewer_and_renderer_input(self):
        report = solution()
        payload = build_viewer_payload(report)
        self.assertEqual(payload['summary']['native_images'], 60)
        self.assertEqual(payload['measurement_source'], 'joint_metric')
        self.assertAlmostEqual(payload['initial_roll_deg'], np.rad2deg(.14))
        self.assertTrue(all(value is None for frame in payload['frames'] for value in frame['std_rad']))
        state = TrajectorySampler(payload['frames']).at(.5)
        self.assertTrue(np.isfinite(state.angles).all())
        self.assertEqual(state.status, ('vision', 'vision', 'vision'))
        for frame in report['frames']:
            self.assertAlmostEqual(np.linalg.norm(frame['caster_quaternion_xyzw']), 1.)
            for quaternion in frame['shell_quaternions_xyzw']:
                self.assertAlmostEqual(np.linalg.norm(quaternion), 1.)
            np.testing.assert_allclose(frame['relative_angles'], frame['angles'] - [.14, 0, 0])

    def test_second_camera_preserves_red_phase_when_first_has_no_red_pixels(self):
        report = solution()
        missing_first = [frame for frame in report['frames']
                         if frame['camera'] == 'c920' and .45 < frame['time_s'] < .7]
        self.assertTrue(missing_first)
        for frame in missing_first:
            self.assertEqual(frame['native_inlier_counts'][1], 0)
            self.assertEqual(frame['support_cameras'][1], ['brio101'])
            self.assertEqual(frame['status'][1], 'vision')
            self.assertTrue(np.isfinite(frame['angles'][1]))
        recovered = report['summary']['supported_by_other_camera_when_this_camera_has_no_nearby_support']
        self.assertGreater(recovered['c920']['red'], 0)
        self.assertEqual(recovered['brio101']['red'], 0)

    def test_missing_in_both_views_stays_null_without_hiding_other_components(self):
        report = solution(drop_both_red=True)
        middle = [frame for frame in report['frames'] if .49 < frame['time_s'] < .65]
        self.assertTrue(middle)
        for frame in middle:
            self.assertEqual(frame['status'][1], 'unresolved')
            self.assertTrue(np.isnan(frame['angles'][1]))
            self.assertFalse(frame['turn_count_valid'][1])
            self.assertIsNone(frame['shell_quaternions_xyzw'][0])
            self.assertTrue(np.isfinite(frame['angles'][[0, 2]]).all())
        self.assertFalse(report['frames'][-1]['turn_count_valid'][1])
        payload = build_viewer_payload(report)
        self.assertTrue(np.isnan(TrajectorySampler(payload['frames']).at(.55).angles[1]))
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'results.json'
            write_json(output, report)
            text = output.read_text(encoding='utf-8')
            self.assertNotIn('NaN', text)
            self.assertNotIn('Infinity', text)
            restored = json.loads(text)
            event = next(frame for frame in restored['frames'] if .5 <= frame['time_s'] < .52)
            self.assertIsNone(event['angles'][1])
            self.assertEqual(event['std_rad'], [None, None, None])
            build_viewer_payload(restored)

    def test_csv_exports_degrees_native_clock_missing_values_and_validity(self):
        report = solution(drop_both_red=True)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'results.csv'
            write_fused_csv(output, report['frames'])
            with output.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(report['frames']))
            self.assertAlmostEqual(float(rows[0]['roll_deg']), np.rad2deg(.14))
            self.assertEqual(rows[0]['camera'], 'c920')
            self.assertEqual(rows[0]['source_frame'], '0')
            missing = next(row for row in rows if .5 <= float(row['time_s']) < .52)
            self.assertEqual(missing['red_deg'], '')
            self.assertEqual(missing['red_status'], 'unresolved')
            self.assertEqual(missing['red_deg_s'], '')
            self.assertEqual(missing['red_std_deg'], '')
            self.assertEqual(missing['red_turn_count_valid'], 'False')
            self.assertNotEqual(missing['roll_deg'], '')
            self.assertNotEqual(missing['green_deg'], '')

    def test_simultaneous_camera_events_retain_supported_velocity(self):
        report = solution(duplicate_timestamp=True)
        self.assertEqual(report['frames'][0]['time_s'], report['frames'][1]['time_s'])
        build_viewer_payload(report)
        # Equal timestamps cannot erase velocities for the rest of the clip.
        self.assertTrue(np.isfinite(report['frames'][-1]['angular_velocity']).all())

    def test_corrected_native_clock_reorders_events_and_preserves_every_source_image(self):
        for offset in (.023, -.023):
            with self.subTest(offset=offset):
                data, seed, cameras, F, pivot, geometry, _ = synthetic()
                problem = NativeBundle(data, seed, cameras, F, pivot, geometry)
                x = problem.x0.copy()
                x[problem.timing_id] = offset
                frames, summary, _ = summarize_solution(problem, x, data, seed, F, pivot)
                expected = sorted((time + offset * (ci == 1), ci, fi)
                                  for time, ci, fi in seed['events'])
                origin = expected[0][0]
                self.assertEqual(len(frames), len(seed['events']))
                self.assertEqual(frames[0]['time_s'], 0.)
                self.assertAlmostEqual(summary['additional_brio_offset_s'], offset)
                self.assertAlmostEqual(summary['output_clock_origin_shift_s'], origin)
                self.assertEqual([(frame['camera'], frame['source_frame']) for frame in frames],
                                 [(('c920', 'brio101')[ci], fi) for _, ci, fi in expected])
                for frame, (corrected, ci, fi) in zip(frames, expected):
                    self.assertAlmostEqual(frame['time_s'], corrected - origin)
                    self.assertAlmostEqual(frame['receive_time_s'], data['times'][ci][fi])
                    self.assertAlmostEqual(frame['camera_time_correction_s'], offset if ci == 1 else 0.)
                    finite = np.isfinite(frame['angles'])
                    np.testing.assert_allclose(frame['angles'][finite],
                                               problem.angles_at(x, [corrected])[0, finite])
                report = {'kind': 'fused_motion', 'status': 'completed',
                          'measurement_source': 'joint_metric',
                          'initial_roll_deg': float(np.rad2deg(frames[0]['angles'][0])),
                          'F': F, 'pivot': pivot, 'geometry': geometry,
                          'frames': frames, 'summary': summary}
                payload = build_viewer_payload(report)
                self.assertEqual(len(payload['frames']), len(seed['events']))
                for name in ('c920', 'brio101'):
                    selected = [frame for frame in frames if frame['camera'] == name]
                    self.assertTrue(np.all(np.diff([frame['time_s'] for frame in selected]) > 0))
                    self.assertEqual([frame['source_frame'] for frame in selected], list(range(len(selected))))


class ResumeWorkflowTests(unittest.TestCase):
    def fixture(self):
        data, seed, cameras, F, pivot, geometry, _ = synthetic()
        data.update(provenance={'source': 'synthetic-native-observations'},
                    timestamp_origin_s=1000., videos=['c920.avi', 'brio101.avi'],
                    report={'synthetic': True}, initial_frames=[])
        seed.update(per_camera=[{}, {}], quality=[None] * len(seed['events']))
        cfg = {'geometry': geometry}
        axes = {'R_bc': F, 'pivot_c920_m': pivot, 'sha256': 'synthetic-axes'}
        return data, seed, cameras, F, pivot, geometry, cfg, axes

    def patches(self, stack, data, cameras, cfg, axes, *, calibration_valid=True):
        prefix = 'dualcam.offline_workflow.'
        stack.enter_context(patch(prefix + 'load_config', return_value=cfg))
        stack.enter_context(patch(prefix + 'load_rig', return_value=(cameras, [], {})))
        stack.enter_context(patch(prefix + 'load_axes', return_value=axes))
        stack.enter_context(patch(prefix + 'collect_native_tracks', return_value=data))
        stack.enter_context(patch(prefix + '_record_provenance'))
        stack.enter_context(patch(prefix + 'calibration_matches', return_value=calibration_valid))
        stack.enter_context(patch(prefix + 'calibration_hashes', return_value={'synthetic': 'calibration'}))

    def test_resume_rejects_changed_provenance_before_loading_saved_parameters(self):
        for change in ('calibration', 'observations', 'axes', 'pivot', 'method'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                root = Path(temp)
                data, _, cameras, F, pivot, _, cfg, axes = self.fixture()
                previous = {'method': 'native_time_joint_metric_bundle',
                            'calibration_hashes': {'synthetic': 'calibration'},
                            'observation_cache_provenance': data['provenance'],
                            'F': F, 'pivot': pivot}
                if change == 'observations':
                    previous['observation_cache_provenance'] = {'source': 'other-video'}
                elif change == 'axes':
                    previous['F'] = np.eye(3)
                elif change == 'pivot':
                    previous['pivot'] = pivot + [.01, 0., 0.]
                elif change == 'method':
                    previous['method'] = 'another-estimator'
                source = root / 'previous.json'
                write_json(source, previous)
                self.patches(stack, data, cameras, cfg, axes, calibration_valid=change != 'calibration')
                fit = stack.enter_context(patch('dualcam.offline_workflow.refine_native'))
                with self.assertRaisesRegex(ValueError, 'inputs or calibration differ'):
                    run_joint('config.yaml', 'session', root / 'output',
                              resume_from=source, progress=None)
                fit.assert_not_called()
                failure = json.loads((root / 'output/results.json').read_text())
                self.assertFalse(failure['success'])
                self.assertEqual(failure['status'], 'failed')

    def test_resumed_conditional_refinement_carries_offset_once_and_rebases_output(self):
        for offset in (.023, -.023):
            with self.subTest(offset=offset), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                root = Path(temp)
                old = root / 'previous'
                old.mkdir()
                data, seed, cameras, F, pivot, geometry, cfg, axes = self.fixture()
                preliminary = NativeBundle(data, seed, cameras, F, pivot, geometry)
                old_x = preliminary.x0.copy()
                old_x[preliminary.timing_id] = offset
                old_frames, old_summary, _ = summarize_solution(preliminary, old_x, data, seed, F, pivot)
                old_summary['optimization_stages'] = [{'converged': True}]
                previous = {'method': 'native_time_joint_metric_bundle',
                            'calibration_hashes': {'synthetic': 'calibration'},
                            'observation_cache_provenance': data['provenance'], 'F': F, 'pivot': pivot,
                            'initial_pose': {'initial_roll_deg': 8., 'accepted': False},
                            'roll_initialization': {'synthetic': True},
                            'summary': old_summary, 'frames': old_frames,
                            'supplied_initial_roll_deg': 8.}
                write_json(old / 'results.json', previous)
                np.savez(old / 'seed.npz', **{key: seed[key] for key in
                         ('angles', 'points', 'landmark', 'keys', 'events')})
                write_json(old / 'seed_diagnostics.json',
                           {'per_camera': seed['per_camera'], 'quality': seed['quality']})
                np.savez(old / 'bundle.npz', knots=preliminary.knots,
                         angles=preliminary.unpack(old_x)[0], additional_brio_offset_s=offset)
                self.patches(stack, data, cameras, cfg, axes)

                from dualcam.spin_initialization import reseed_spins as real_conditional
                conditional_calls = []

                def conditional(data_arg, seed_arg, cameras_arg, F_arg, pivot_arg, geometry_arg,
                                knots_arg, angles_arg, offset_arg):
                    self.assertAlmostEqual(offset_arg, offset)
                    np.testing.assert_allclose(knots_arg, preliminary.knots)
                    if not conditional_calls:
                        np.testing.assert_allclose(angles_arg, preliminary.unpack(old_x)[0])
                    conditional_calls.append(True)
                    self.assertEqual(seed_arg['events'], seed['events'])
                    return real_conditional(data_arg, seed_arg, cameras_arg, F_arg, pivot_arg,
                                             geometry_arg, knots_arg, angles_arg, offset_arg)

                def fitted(data_arg, seed_arg, cameras_arg, F_arg, pivot_arg, geometry_arg, **kwargs):
                    # The actual bundle class must receive the fitted clock
                    # offset without shifting its raw native event grid twice.
                    self.assertEqual(seed_arg['events'], seed['events'])
                    problem = NativeBundle(data_arg, seed_arg, cameras_arg, F_arg, pivot_arg, geometry_arg)
                    self.assertAlmostEqual(problem.x0[problem.timing_id], offset)
                    self.assertAlmostEqual(problem.initial_alpha, np.deg2rad(8.))
                    return problem, problem.x0.copy(), [{'converged': True}]

                spin = stack.enter_context(patch('dualcam.spin_initialization.reseed_spins', side_effect=conditional))
                fit = stack.enter_context(patch('dualcam.offline_workflow.refine_native', side_effect=fitted))
                report = run_joint('config.yaml', 'session', root / 'output',
                                   resume_from=old / 'results.json', progress=None)
                self.assertEqual(spin.call_count, 2)
                self.assertEqual(fit.call_count, 1)
                origin = min(min(data['times'][0]), min(data['times'][1]) + offset)
                self.assertAlmostEqual(report['timestamp_origin_s'], 1000. + origin)
                self.assertAlmostEqual(report['summary']['additional_brio_offset_s'], offset)
                self.assertAlmostEqual(report['summary']['initial_joint_fit']['additional_brio_offset_s'], offset)
                self.assertEqual(len(report['frames']), len(seed['events']))
                for frame in report['frames']:
                    ci = ('c920', 'brio101').index(frame['camera'])
                    raw = data['times'][ci][frame['source_frame']]
                    self.assertAlmostEqual(frame['time_s'], raw + offset * (ci == 1) - origin)
                build_viewer_payload(report)
                self.assertTrue((root / 'output/orientation_3d.html').is_file())


if __name__ == '__main__':
    unittest.main()
