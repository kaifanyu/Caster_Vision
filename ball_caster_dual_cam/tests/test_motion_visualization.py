"""Saved state validity and self-contained 3D export regressions."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dualcam.model import rotation_x, rotation_z
from dualcam.visualization import build_viewer_payload, write_motion_viewer


def example_report():
    return {
        'kind': 'fused_motion', 'status': 'completed', 'measurement_source': 'rotation',
        'F': (rotation_z(.4) @ rotation_x(.2)).tolist(), 'pivot': [.01, -.02, .4],
        'geometry': {'radius_m': .1, 'gap_m': .02, 'red_shell_sign': -1},
        'initial_roll_deg': 0.,
        'frames': [
            {'time_s': 0., 'camera': 'c920', 'source_frame': 0,
             'angles': [0., 0., 0.], 'status': ['home'] * 3,
             'std_rad': [.01] * 3, 'age_s': [0.] * 3, 'turn_count_valid': [True] * 3},
            {'time_s': .03, 'camera': 'c920', 'source_frame': 1,
             'angles': [.2, 99., -.4], 'status': ['vision', 'unresolved', 'predicted'],
             'std_rad': [.02, float('nan'), .05], 'age_s': [0., 1., .03],
             'turn_count_valid': [True, False, False]},
            {'time_s': .03, 'camera': 'brio101', 'source_frame': 0,
             'angles': [.21, .3, None], 'status': ['vision', 'vision', 'unresolved'],
             'std_rad': [.01, .02, float('inf')], 'age_s': [0., 0., None],
             'turn_count_valid': [True, False, False]},
        ],
        'summary': {'per_camera': {'c920': {'visual_updates': 1}, 'brio101': {'visual_updates': 1}},
                    'native_images': 3},
    }


class PayloadTests(unittest.TestCase):
    def test_keeps_equal_native_times_masks_unknowns_and_does_not_modify_source(self):
        report = example_report()
        payload = build_viewer_payload(report)
        self.assertEqual(len(payload['frames']), 3)
        self.assertEqual([f['camera'] for f in payload['frames']], ['c920', 'c920', 'brio101'])
        self.assertEqual(payload['frames'][1]['angles'], [.2, None, -.4])
        self.assertEqual(payload['frames'][1]['status'], ['vision', 'unresolved', 'predicted'])
        self.assertIsNone(payload['frames'][1]['std_rad'][1])
        self.assertIsNone(payload['frames'][2]['std_rad'][2])
        self.assertEqual(report['frames'][1]['angles'][1], 99.)
        self.assertEqual(payload['summary']['coverage']['green'], {'home': 1, 'predicted': 1, 'unresolved': 1})
        payload['summary']['per_camera']['c920']['visual_updates'] = 77
        self.assertEqual(report['summary']['per_camera']['c920']['visual_updates'], 1)
        json.dumps(payload, allow_nan=False)

    def test_preserves_saved_frame_geometry_and_full_turn_values(self):
        report = example_report()
        report['frames'][1]['angles'][0] = 2 * np.pi + .2
        report['frames'][1]['turn_count_valid'][0] = False
        payload = build_viewer_payload(report)
        F = np.asarray(payload['F'])
        q = payload['frames'][1]['angles'][0]
        np.testing.assert_allclose((F @ rotation_x(q))[:, 0], F[:, 0])
        np.testing.assert_allclose((F @ rotation_x(q))[:, 2], F @ [0., -np.sin(.2), np.cos(.2)])
        self.assertEqual(payload['pivot'], report['pivot'])
        self.assertEqual(payload['red_shell_sign'], -1)
        self.assertGreater(q, 2 * np.pi)
        self.assertFalse(payload['frames'][1]['turn_count_valid'][0])

    def test_supported_angles_must_exist_and_be_finite(self):
        for value in (None, float('nan'), float('inf'), '0.2', True):
            with self.subTest(value=value):
                report = example_report()
                report['frames'][1]['angles'][0] = value
                with self.assertRaisesRegex(ValueError, 'angle must be a finite'):
                    build_viewer_payload(report)
        report = example_report()
        del report['frames'][1]['angles']
        with self.assertRaisesRegex(ValueError, 'angles must contain'):
            build_viewer_payload(report)

    def test_invalid_rotation_geometry_status_and_uncertainty_fail(self):
        changes = [
            ('F', np.diag([1., 1., -1.]).tolist()),
            ('F', (np.eye(3) * 2).tolist()), ('F', [[1., 0., 0.]]),
            ('F', [[float('nan')] * 3] * 3),
            ('pivot', [0., 0., float('inf')]),
            ('geometry', {'radius_m': 0., 'gap_m': 0.}),
            ('geometry', {'radius_m': .1, 'gap_m': .2}),
            ('geometry', {'radius_m': .1, 'gap_m': .02, 'red_shell_sign': 0}),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                report = example_report(); report[key] = value
                with self.assertRaises(ValueError):
                    build_viewer_payload(report)
        for key, value in [('status', ['measured'] * 3), ('std_rad', [-1.] * 3),
                           ('age_s', [-1.] * 3), ('turn_count_valid', [1, 0, 0])]:
            with self.subTest(key=key):
                report = example_report(); report['frames'][1][key] = value
                with self.assertRaises(ValueError):
                    build_viewer_payload(report)

    def test_rejects_reversed_events_and_duplicate_camera_frames_or_times(self):
        for change in ({'time_s': -.1}, {'source_frame': 0}, {'time_s': 0.}, {'camera': 'other'}):
            report = example_report(); report['frames'][1].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_viewer_payload(report)
        report = example_report(); report['frames'].reverse()
        with self.assertRaises(ValueError):
            build_viewer_payload(report)

    def test_rejects_other_result_formats_and_empty_events(self):
        for report in ({'kind': 'axes'}, {**example_report(), 'frames': []}):
            with self.assertRaises(ValueError):
                build_viewer_payload(report)


class ExportTests(unittest.TestCase):
    def test_standalone_export_escapes_data_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'results.json'; output = root / 'viewer.html'
            report = example_report()
            report['warnings'] = ['</script><script>alert(1)</script>\u2028\u2029']
            source.write_text(json.dumps(report), encoding='utf-8')
            exported = write_motion_viewer(source, output)
            self.assertEqual(exported, output)
            html = output.read_text(encoding='utf-8')
            self.assertNotIn('__MOTION_DATA__', html)
            self.assertNotIn('<script>alert(1)</script>', html)
            self.assertIn('\\u003c/script>', html)
            self.assertIn('\\u2028\\u2029', html)
            self.assertNotRegex(html, r'<script\b[^>]*\bsrc\s*=')
            self.assertNotRegex(html, r'<link\b[^>]*\bhref\s*=')
            with self.assertRaisesRegex(ValueError, 'new or empty'):
                write_motion_viewer(source, output)
            self.assertEqual(output.read_text(encoding='utf-8'), html)
            self.assertEqual(json.loads(source.read_text())['warnings'], report['warnings'])

    def test_accepts_empty_output_and_invalid_payload_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'results.json'; output = root / 'viewer.html'
            report = example_report(); source.write_text(json.dumps(report))
            output.touch(); write_motion_viewer(source, output)
            self.assertGreater(output.stat().st_size, 1000)
            output.unlink(); report['F'] = [[0.] * 3] * 3
            source.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                write_motion_viewer(source, output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
