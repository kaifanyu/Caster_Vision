import copy
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from blurtrack.exports import export_rates_csv, write_artifacts


def report_fixture():
    return {'timestamp_origin_s': 1000., 'rate_analysis': {
        'method': 'local_C1_cubic_Hermite_diagnostic_derivative', 'grid_spacing_s': 1/30},
        'frames': [{'time_s': 2., 'receive_time_s': 1.994, 'camera': 'brio101', 'source_frame': 59,
                    'angular_velocity': [1., 2., 3.],
                    'rate_status': ['vision', 'vision', 'predicted'],
                    'phase_status': ['vision', 'phase_estimated', 'phase_estimated'],
                    'turn_count_valid': [True, False, False],
                    'rate_support_interval_s': [[1.93, 2.06], [1.93, 2.06], None],
                    'rate_source': ['conditional_metric_roll_poses', 'conditional_metric_spin_edges', 'conditional_metric_spin_edges'],
                    'rate_relative_null_projection': [0., 1e-16, .3],
                    'angular_velocity_rig_red': [1., .1, 2.],
                    'angular_velocity_rig_red_status': 'vision',
                    'angular_velocity_rig_green': [1., .2, 3.],
                    'angular_velocity_rig_green_status': 'predicted'}]}


class RateExportTests(unittest.TestCase):
    def export(self, report):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'nested'/'rates.csv'
            self.assertEqual(export_rates_csv(path, report), path)
            with path.open(newline='', encoding='utf-8') as stream:
                return list(csv.DictReader(stream))

    def test_local_rate_and_global_phase_are_independent(self):
        report = report_fixture()
        original = copy.deepcopy(report)
        row = self.export(report)[0]
        self.assertEqual(report, original)
        self.assertEqual(row['time_s'], '2.0')
        self.assertEqual(row['timestamp_s'], '1002.0')
        self.assertEqual(row['source_frame'], '59')
        self.assertEqual(row['red_spin_rate_status'], 'vision')
        self.assertEqual(row['red_spin_phase_status'], 'phase_estimated')
        self.assertEqual(row['red_spin_turn_count_valid'], 'False')
        self.assertEqual(row['red_spin_strict_angular_velocity_rad_s'], '2.0')
        self.assertAlmostEqual(float(row['red_spin_strict_angular_velocity_deg_s']), math.degrees(2.))
        self.assertEqual(row['red_spin_rate_support_start_s'], '1.93')
        self.assertEqual(row['red_angular_velocity_rig_valid'], 'True')
        self.assertEqual(row['red_strict_angular_velocity_rig_z_rad_s'], '2.0')
        self.assertEqual(row['green_spin_angular_velocity_rad_s'], '3.0')
        self.assertEqual(row['green_spin_strict_angular_velocity_rad_s'], '')
        self.assertEqual(row['green_spin_rate_support_start_s'], '')
        self.assertEqual(row['green_angular_velocity_rig_valid'], 'False')
        self.assertEqual(row['green_strict_angular_velocity_rig_z_rad_s'], '')
        self.assertEqual(row['green_angular_velocity_rig_z_rad_s'], '3.0')

    def test_unresolved_and_nonfinite_values_are_blank(self):
        report = report_fixture()
        report.pop('timestamp_origin_s')
        frame = report['frames'][0]
        frame['rate_status'][1] = 'unresolved'
        frame['angular_velocity'][2] = float('nan')
        frame['angular_velocity_rig_green'][1] = float('inf')
        row = self.export(report)[0]
        self.assertEqual(row['timestamp_s'], '')
        self.assertEqual(row['red_spin_angular_velocity_rad_s'], '')
        self.assertEqual(row['red_spin_strict_angular_velocity_rad_s'], '')
        self.assertEqual(row['red_spin_rate_support_end_s'], '')
        self.assertEqual(row['green_spin_angular_velocity_rad_s'], '')
        self.assertEqual(row['red_angular_velocity_rig_status'], 'unresolved')
        self.assertEqual(row['red_angular_velocity_rig_x_rad_s'], '')
        self.assertEqual(row['green_angular_velocity_rig_status'], 'unresolved')
        self.assertNotIn('nan', ','.join(row.values()).lower())
        self.assertNotIn('inf', ','.join(row.values()).lower())

    def test_predicted_roll_cannot_validate_rig_vector(self):
        report = report_fixture()
        report['frames'][0]['rate_status'][0] = 'predicted'
        row = self.export(report)[0]
        self.assertEqual(row['red_angular_velocity_rig_status'], 'predicted')
        self.assertEqual(row['red_angular_velocity_rig_valid'], 'False')
        self.assertEqual(row['red_strict_angular_velocity_rig_x_rad_s'], '')
        self.assertEqual(row['red_angular_velocity_rig_x_rad_s'], '1.0')

    def test_unannotated_report_cannot_overwrite_existing_export(self):
        report = report_fixture()
        del report['frames'][0]['rate_status']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'rates.csv'
            path.write_text('completed earlier export', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'rate_status'):
                export_rates_csv(path, report)
            self.assertEqual(path.read_text(encoding='utf-8'), 'completed earlier export')

    def test_empty_report_has_schema_and_unknown_turns_stay_blank(self):
        self.assertEqual(self.export({'frames': []}), [])
        report = report_fixture()
        del report['frames'][0]['turn_count_valid']
        row = self.export(report)[0]
        self.assertEqual(row['red_spin_turn_count_valid'], '')

    def test_write_artifacts_handles_saved_json_nulls_in_legacy_csvs(self):
        report = report_fixture()
        frame = report['frames'][0]
        frame.update(angles=[.2, None, None], strict_angles=[.2, None, None],
                     angular_velocity=[1., None, None], strict_angular_velocity=[1., None, None],
                     std_rad=[None]*3, status=['vision', 'unresolved', 'unresolved'],
                     strict_status=['vision', 'unresolved', 'unresolved'],
                     rate_status=['vision', 'unresolved', 'unresolved'],
                     caster_quaternion_xyzw=None, swivel_axis_c920=None)
        original_frame = copy.deepcopy(frame)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            # Exercise actual JSON and both actual legacy CSV writers. HTML
            # rendering is unrelated to null normalization and has a wider
            # report schema, so this bounded regression substitutes it.
            with patch('dualcam.visualization.write_motion_viewer',
                       side_effect=lambda source,target: target.write_text('viewer')) as viewer:
                write_artifacts(output, report)
                write_artifacts(output, report)
            self.assertEqual(viewer.call_count,2)
            self.assertEqual(viewer.call_args.args[0],output/'results.json')
            self.assertEqual((output/'orientation_3d.html').read_text(),'viewer')
            saved = json.loads((output/'results.json').read_text(encoding='utf-8'))
            self.assertIsNone(saved['frames'][0]['angular_velocity'][1])
            self.assertEqual(report['frames'][0], original_frame)
            for name in ('results.csv', 'strict_results.csv'):
                with (output/name).open(newline='', encoding='utf-8') as stream:
                    row = list(csv.DictReader(stream))[0]
                self.assertEqual(row['red_deg'], '')
                self.assertEqual(row['red_deg_s'], '')
                self.assertEqual(row['green_deg_s'], '')
                self.assertEqual(row['roll_std_deg'], '')
                self.assertAlmostEqual(float(row['roll_deg_s']), math.degrees(1.))
            with (output/'angular_rates.csv').open(newline='', encoding='utf-8') as stream:
                row = list(csv.DictReader(stream))[0]
            self.assertEqual(row['red_spin_strict_angular_velocity_rad_s'], '')
            self.assertEqual(row['red_spin_rate_status'], 'unresolved')


if __name__ == '__main__':
    unittest.main()
