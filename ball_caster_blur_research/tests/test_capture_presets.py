"""Validate trial preset integration without opening or changing a camera."""
import copy
from pathlib import Path
import unittest

import blurtrack
from dualcam.capture import (V4L2Camera, check_intrinsics_profile,
                             load_capture_config, validate_requested)
from dualcam.config import load_config


class CapturePresetTests(unittest.TestCase):
    def test_presets_preserve_geometry_and_only_change_exposure(self):
        baseline, _ = load_capture_config(blurtrack.BASELINE / 'config/rig.yaml')
        for filename, exposure in [('capture_2ms.yaml', 2000),
                                   ('capture_1ms.yaml', 1000),
                                   ('capture_0p5ms.yaml', 500)]:
            config, path = load_capture_config(blurtrack.ROOT / 'config' / filename)
            resolved = load_config(path)
            for name in ('c920', 'brio101'):
                with self.subTest(preset=filename, camera=name):
                    actual = config['cameras'][name]['capture']
                    expected = copy.deepcopy(baseline['cameras'][name]['capture'])
                    expected['exposure_us'] = exposure
                    self.assertEqual(actual, expected)
                    validate_requested(actual)
                    intrinsics = Path(resolved['cameras'][name]['intrinsics'])
                    self.assertEqual(intrinsics, blurtrack.BASELINE / 'calibration' / f'{name}.yaml')
                    profile = check_intrinsics_profile(path, config['cameras'][name])
                    self.assertTrue(profile['intrinsics_sha256'])
                    # Imported calibration metadata is honestly still unknown.
                    self.assertFalse(profile['intrinsics_capture_profile_verified'])
            for section in ('geometry', 'timing', 'tracking', 'solver'):
                self.assertEqual(config[section], baseline[section])
            for section in ('stereo', 'axes'):
                self.assertEqual(Path(resolved[section]['path']),
                                 blurtrack.BASELINE / 'calibration' / f'{section}.yaml')

    def test_recorder_maps_units_and_sets_manual_controls_without_hardware(self):
        for filename, expected_units in [('capture_2ms.yaml', 20),
                                         ('capture_1ms.yaml', 10),
                                         ('capture_0p5ms.yaml', 5)]:
            config, _ = load_capture_config(blurtrack.ROOT / 'config' / filename)
            for name in ('c920', 'brio101'):
                with self.subTest(preset=filename, camera=name):
                    # Mock advertised controls, not a claim of live device support.
                    int_control = {'type': 'int', 'menu': {}, 'min': 0,
                                   'max': 10000, 'step': 1, 'raw': ''}
                    controls = {key: copy.deepcopy(int_control) for key in
                                ['gain', 'white_balance_temperature', 'power_line_frequency',
                                 'exposure_time_absolute', 'exposure_dynamic_framerate',
                                 'white_balance_automatic', 'brightness', 'contrast',
                                 'saturation', 'sharpness', 'backlight_compensation']}
                    controls['auto_exposure'] = {
                        'type': 'menu', 'menu': {1: 'Manual Mode', 3: 'Aperture Priority Mode'},
                        'min': 0, 'max': 3, 'step': 1, 'raw': ''}
                    if name == 'c920':
                        controls.update({key: copy.deepcopy(int_control) for key in
                                         ['focus_automatic_continuous', 'focus_absolute', 'zoom_absolute']})
                    plan, missing = V4L2Camera(name, config['cameras'][name]).plan_controls(controls)
                    self.assertFalse(missing)
                    self.assertEqual(plan[0], ('auto_exposure', 1))
                    planned = dict(plan)
                    self.assertEqual(planned['exposure_time_absolute'], expected_units)
                    self.assertEqual(planned['exposure_dynamic_framerate'], 0)
                    self.assertEqual(planned['white_balance_automatic'], 0)
                    if name == 'c920':
                        self.assertEqual(planned['focus_automatic_continuous'], 0)
                        self.assertEqual(planned['focus_absolute'], 50)
                    else:
                        self.assertFalse(any('focus' in key for key in planned))


if __name__ == '__main__':
    unittest.main()
