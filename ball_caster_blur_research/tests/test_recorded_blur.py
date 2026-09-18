"""Real-recording exports must preserve uncertainty after photometric fitting."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_recorded_blur import gate_export, verify_atlas


def result_fixture(status='candidate', comparable=True):
    return dict(status=status, common_comparison_available=comparable,
                best_parameters_rad=[.4, 12.], candidates=[dict(parameters_rad=[.4, 12.])],
                accuracy_validated=False, turn_count_valid=False)


class RecordedBlurExportTests(unittest.TestCase):
    def test_incomparable_hypotheses_have_only_an_illustrative_rate(self):
        result = result_fixture(comparable=False)
        candidates = copy.deepcopy(result['candidates'])
        gate_export(result)
        self.assertIsNone(result['best_parameters_rad'])
        self.assertIsNone(result['recovered_rate_deg_s'])
        self.assertEqual(result['illustrative_parameters_rad'], [.4, 12.])
        self.assertEqual(result['candidates'], candidates)
        self.assertIn('Too few', result['rejection_reason'])

    def test_ambiguous_direction_or_texture_is_not_exported_as_best(self):
        result = result_fixture(status='ambiguous')
        candidates = copy.deepcopy(result['candidates'])
        gate_export(result)
        self.assertIsNone(result['best_parameters_rad'])
        self.assertIsNone(result['recovered_rate_deg_s'])
        self.assertEqual(result['illustrative_parameters_rad'], [.4, 12.])
        self.assertEqual(result['candidates'], candidates)

    def test_comparable_candidate_still_is_not_a_validated_recovery(self):
        result = result_fixture()
        gate_export(result)
        self.assertEqual(result['best_parameters_rad'], [.4, 12.])
        self.assertIsNone(result['recovered_rate_deg_s'])
        self.assertFalse(result['accuracy_validated'])
        self.assertFalse(result['turn_count_valid'])
        self.assertIn('Photometric candidate only', result['rejection_reason'])

    def test_empty_fit_has_no_exported_rate(self):
        result = dict(best_parameters_rad=None, candidates=[], status='insufficient_texture')
        gate_export(result)
        self.assertIsNone(result['best_parameters_rad'])
        self.assertIsNone(result['recovered_rate_deg_s'])
        self.assertNotIn('illustrative_parameters_rad', result)

    def test_gate_is_idempotent_and_preserves_rejected_diagnostics(self):
        result = result_fixture(status='ambiguous')
        gate_export(result)
        first = copy.deepcopy(result)
        gate_export(result)
        self.assertEqual(result, first)


class AtlasPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='blur_preflight_')
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.baseline = root/'baseline'; self.baseline.mkdir()
        self.atlas = root/'atlas'; self.atlas.mkdir()
        self.F = np.eye(3); self.pivot = np.array([0., 0., .4])
        self.axes = root/'axes.yaml'; self.axes.write_text('fixture axes')
        self.cfg = dict(axes=dict(path=str(self.axes)), geometry=dict(radius_m=.1, gap_m=.02))
        report = dict(F=self.F.tolist(), pivot=self.pivot.tolist())
        (self.baseline/'results.json').write_text(json.dumps(report))
        (self.baseline/'bundle.npz').write_bytes(b'original test trajectory bytes')
        self.texture = self.atlas/'atlas_c920_red.npz'
        self.texture.write_bytes(b'original test atlas bytes')
        digest = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        self.manifest = dict(baseline_sha256=digest(self.baseline/'results.json'),
                             baseline_bundle_sha256=digest(self.baseline/'bundle.npz'),
                             axes_sha256=digest(self.axes), calibration_hashes={'fixture':'calibration-hash'},
                             geometry=self.cfg['geometry'],
                             atlases=[dict(camera='c920', shell='red', sha256=digest(self.texture))])
        (self.atlas/'atlas_manifest.json').write_text(json.dumps(self.manifest))
        calibration = patch('fit_recorded_blur.calibration_hashes', return_value={'fixture':'calibration-hash'})
        self.mock_calibration = calibration.start()
        self.addCleanup(calibration.stop)

    def verify(self):
        return verify_atlas(self.atlas, self.cfg, self.baseline, self.F, self.pivot)

    def test_unchanged_inputs_pass_without_rewriting_manifest(self):
        before = (self.atlas/'atlas_manifest.json').read_bytes()
        self.assertEqual(self.verify(), self.manifest)
        self.assertEqual((self.atlas/'atlas_manifest.json').read_bytes(), before)

    def test_modified_atlas_pixels_are_rejected(self):
        self.texture.write_bytes(b'replaced texture')
        with self.assertRaisesRegex(ValueError, 'atlas hash mismatch'): self.verify()

    def test_modified_trajectory_bundle_is_rejected(self):
        (self.baseline/'bundle.npz').write_bytes(b'different trajectory')
        with self.assertRaisesRegex(ValueError, 'trajectory bundle hash differs'): self.verify()

    def test_modified_calibration_or_axes_are_rejected(self):
        self.mock_calibration.return_value = {'fixture':'different-camera-calibration'}
        with self.assertRaisesRegex(ValueError, 'calibration/geometry differs'): self.verify()
        self.mock_calibration.return_value = {'fixture':'calibration-hash'}
        self.axes.write_text('replaced axes')
        with self.assertRaisesRegex(ValueError, 'axes hash differs'): self.verify()


if __name__=='__main__': unittest.main()
