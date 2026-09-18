"""Workflow provenance, failure preservation, and image-fit integration."""
from __future__ import annotations

import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from dualcam.config import load_config, load_rig, sha256, write_yaml
from dualcam.workflow import (WorkflowError, calibrate_axes, calibration_hashes,
                              calibration_matches, load_axes, run_motion)
from scripts.calibrate_axes import main as axes_main
from tests.test_solver import make_clip, setup_scene


def rig_files(root):
    cameras, F, pivot = setup_scene()
    cfg = {"cameras": {}, "stereo": {"path": "stereo.yaml"}, "axes": {"path": "axes.yaml"},
           "geometry": {"radius_m": .1, "gap_m": .02, "red_shell_sign": 1},
           "timing": {"brio_offset_s": 0., "max_pair_skew_ms": 12., "verified": True},
           "tracking": {"min_track_length": 3}, "solver": {}}
    for idx, name in enumerate(("c920", "brio101")):
        profile = {"width": 1920, "height": 1080, "fps": 30, "fourcc": "MJPG"}
        cfg["cameras"][name] = {"intrinsics": f"{name}.yaml", "capture": profile}
        write_yaml(root / f"{name}.yaml", {"K": cameras[idx]["K"], "dist": [0] * 5, "image_size": [1920, 1080], "capture_profile": profile})
    write_yaml(root / "stereo.yaml", {"R_21": cameras[1]["R"], "t_21_m": cameras[1]["t"],
                                      "intrinsics_sha256": {name: sha256(root / f"{name}.yaml") for name in ("c920", "brio101")}})
    write_yaml(root / "rig.yaml", cfg)
    loaded = load_config(root / "rig.yaml")
    axes = {"R_bc": F, "pivot_c920_m": pivot, "radius_m": .1, "gap_m": .02,
            "red_shell_sign": 1, "timing": cfg["timing"], "calibration_hashes": calibration_hashes(loaded)}
    write_yaml(root / "axes.yaml", axes)
    return root / "rig.yaml", loaded, cameras, F, pivot


def tracked_clip(clip, name):
    n = len(clip["times"])
    return {**clip, "session": name, "pairs": np.column_stack([np.arange(n), np.arange(n)]),
            "session_report": {"timing_verified": True}, "session_mode": clip["mode"],
            "increments": np.tile(np.eye(3), (2, 2, n-1, 1, 1)),
            "increment_support": np.ones((2, 2, n-1), int) * 10,
            "circles": np.array([[960, 540, 125], [960, 540, 125]])}


class WorkflowTests(unittest.TestCase):
    def test_calibration_provenance_survives_linux_windows_newline_conversion(self):
        for source_eol, target_eol in ((b"\n", b"\r\n"), (b"\r\n", b"\n")):
            with self.subTest(source_eol=source_eol), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _, cfg, cameras, F, _ = rig_files(root)
                paths = [root / f"{name}.yaml" for name in ("c920", "brio101", "stereo")]

                def convert(path, newline):
                    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", newline))

                for path in paths[:2]:
                    convert(path, source_eol)
                stereo = yaml.safe_load(paths[2].read_text())
                stereo["intrinsics_sha256"] = {name: sha256(root / f"{name}.yaml")
                                                for name in ("c920", "brio101")}
                write_yaml(paths[2], stereo)
                convert(paths[2], source_eol)
                saved_hashes = calibration_hashes(cfg)
                axes_path = root / "axes.yaml"
                axes = yaml.safe_load(axes_path.read_text())
                axes["calibration_hashes"] = saved_hashes
                write_yaml(axes_path, axes)
                axes_before = axes_path.read_bytes()

                for path in paths:
                    convert(path, target_eol)
                self.assertTrue(all(saved_hashes[name] != digest
                                    for name, digest in calibration_hashes(cfg).items()))
                loaded, _, _ = load_rig(cfg)
                np.testing.assert_allclose(loaded[1]["R"], cameras[1]["R"])
                np.testing.assert_allclose(load_axes(cfg)["R_bc"], F)
                # Rendering uses this same comparison for saved result hashes.
                self.assertTrue(calibration_matches(cfg, saved_hashes))
                self.assertEqual(axes_path.read_bytes(), axes_before)

                # A real intrinsic change must still fail after conversion.
                data = yaml.safe_load(paths[0].read_text())
                data["K"][0][0] += 1
                write_yaml(paths[0], data)
                with self.assertRaisesRegex(ValueError, "Stereo calibration is stale"):
                    load_rig(cfg)
                with self.assertRaisesRegex(ValueError, "Axis calibration is stale"):
                    load_axes(cfg)
                self.assertFalse(calibration_matches(cfg, saved_hashes))

    def test_incomplete_or_invalid_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, cfg, _, _, _ = rig_files(root)
            hashes = calibration_hashes(cfg)
            for expected in (None, {}, [], {**hashes, "extra": "0" * 64},
                             {**hashes, "c920": None}):
                with self.subTest(expected=expected):
                    self.assertFalse(calibration_matches(cfg, expected))
            stereo_path = root / "stereo.yaml"
            stereo = yaml.safe_load(stereo_path.read_text())
            stereo["intrinsics_sha256"].pop("brio101")
            write_yaml(stereo_path, stereo)
            with self.assertRaisesRegex(ValueError, "Stereo calibration is stale"):
                load_rig(cfg)

    def test_stale_stereo_intrinsics_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, cfg, _, _, _ = rig_files(root)
            with (root / "c920.yaml").open("a") as stream:
                stream.write("# calibration changed\n")
            with self.assertRaisesRegex(ValueError, "Stereo calibration is stale"):
                load_rig(cfg)

    def test_axes_reject_changed_hash_geometry_or_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, cfg, _, _, _ = rig_files(root)
            self.assertEqual(load_axes(cfg)["sha256"], sha256(root / "axes.yaml"))
            cfg["geometry"]["radius_m"] = .11
            with self.assertRaisesRegex(ValueError, "geometry.radius_m"):
                load_axes(cfg)
            cfg["geometry"]["radius_m"] = .1
            cfg["timing"]["brio_offset_s"] = .01
            with self.assertRaisesRegex(ValueError, "time offset"):
                load_axes(cfg)
            cfg["timing"]["brio_offset_s"] = 0
            with (root / "stereo.yaml").open("a") as stream:
                stream.write("# extrinsics changed\n")
            with self.assertRaisesRegex(ValueError, "Axis calibration is stale"):
                load_axes(cfg)

    def test_failed_axis_fit_keeps_existing_calibration_and_cli_returns_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, cfg, cameras, F, pivot = rig_files(root)
            before = (root / "axes.yaml").read_bytes()
            roll, _ = make_clip(cameras, F, pivot, "roll", count=4, frames=5)
            swivel, _ = make_clip(cameras, F, pivot, "swivel", count=4, frames=5)
            clips = [tracked_clip(roll, "roll"), tracked_clip(swivel, "swivel")]
            rejected = {"success": False, "F": F, "pivot": pivot, "datasets": [],
                        "diagnostics": {"reasons": ["rank deficient calibration"]}}
            with patch("dualcam.workflow.track_session", side_effect=clips), \
                 patch("dualcam.workflow.initialize_axes", return_value=(F, {})), \
                 patch("dualcam.workflow.initialize_pivot", return_value=pivot), \
                 patch("dualcam.workflow.fit_joint", return_value=rejected):
                code = axes_main(["--config", str(config_path), "--roll", "roll", "--swivel", "swivel", "--output", str(root / "failed_fit"),
                                  "--max-nfev", "321", "--surface-refinement-passes", "2"])
            self.assertEqual(code, 2)
            self.assertEqual((root / "axes.yaml").read_bytes(), before)
            report = json.loads((root / "failed_fit/report.json").read_text())
            self.assertFalse(report["success"])
            self.assertEqual(report["status"], "rejected")
            self.assertEqual(report['config']['solver']['max_nfev'], 321)
            self.assertEqual(report['config']['solver']['surface_refinement_passes'], 2)
            self.assertEqual(load_config(config_path)['solver'], {})
            self.assertTrue((root / "failed_fit/roll_tracks.npz").exists())
            self.assertTrue((root / "failed_fit/swivel_tracks.npz").exists())

    def test_successful_axes_are_loadable_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, cfg, cameras, F, pivot = rig_files(root)
            roll, _ = make_clip(cameras, F, pivot, "roll", count=4, frames=5)
            swivel, _ = make_clip(cameras, F, pivot, "swivel", count=4, frames=5)
            accepted = {"success": True, "F": F, "pivot": pivot, "datasets": [], "diagnostics": {"reasons": []}}
            with patch("dualcam.workflow.track_session", side_effect=[tracked_clip(roll, "roll"), tracked_clip(swivel, "swivel")]), \
                 patch("dualcam.workflow.initialize_axes", return_value=(F, {})), \
                 patch("dualcam.workflow.initialize_pivot", return_value=pivot), \
                 patch("dualcam.workflow.fit_joint", return_value=accepted) as fit:
                report = calibrate_axes(config_path, "roll", "swivel", root / "accepted")
            self.assertTrue(report["success"])
            fitted = fit.call_args
            self.assertTrue(fitted.kwargs["calibrate_axes"])
            self.assertTrue(fitted.kwargs["refine_pivot"])
            self.assertEqual(fitted.kwargs["options"]["red_sign"], 1)
            np.testing.assert_allclose(load_axes(cfg)["R_bc"], F)
            self.assertIn("home_reference", yaml.safe_load((root / "axes.yaml").read_text()))

    def test_motion_pipeline_real_solver_exports_missing_spin_as_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, _, cameras, F, pivot = rig_files(root)
            clip, expected = make_clip(cameras, F, pivot, "motion", noise=.03, count=6, frames=6, omit_green=True)
            tracked = tracked_clip(clip, "synthetic_motion")
            with patch("dualcam.workflow.track_session", return_value=tracked), \
                 patch("dualcam.workflow.initialize_angles", return_value=clip["initial_angles"]):
                report = run_motion(config_path, "motion", root / "motion_result", initial_roll_deg=0)
            self.assertTrue(report["success"], report["fit"]["diagnostics"])
            with (root / "motion_result/results.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row["beta_green_rad"] == "" and row["valid_green"] == "0" for row in rows))
            self.assertTrue(all(row["valid_alpha"] == "1" for row in rows))
            result = json.loads((root / "motion_result/results.json").read_text())
            self.assertEqual(result["fit"]["datasets"][0]["angles"][0][2], None)
            self.assertEqual(result["axes_sha256"], sha256(root / "axes.yaml"))
            np.testing.assert_allclose(report["fit"]["datasets"][0]["angles"][:, :2], expected[:, :2], atol=.02)

    def test_wrong_recording_mode_leaves_axes_untouched_and_saves_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, _, cameras, F, pivot = rig_files(root)
            clip, _ = make_clip(cameras, F, pivot, "motion", count=4, frames=5)
            before = (root / "axes.yaml").read_bytes()
            with patch("dualcam.workflow.track_session", return_value=tracked_clip(clip, "wrong")):
                with self.assertRaisesRegex(WorkflowError, "Expected a --mode roll"):
                    calibrate_axes(config_path, "wrong", "swivel", root / "wrong_mode")
            self.assertEqual((root / "axes.yaml").read_bytes(), before)
            self.assertEqual(json.loads((root / "wrong_mode/report.json").read_text())["status"], "failed")

    def test_calibration_clip_replay_requires_opt_in_and_does_not_force_pure_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, _, cameras, F, pivot = rig_files(root)
            # Deliberately label mixed motion as roll: replay must recover spins,
            # not report the zero spins imposed during pure-roll calibration.
            clip, expected = make_clip(cameras, F, pivot, "motion", noise=.03, count=6, frames=6)
            tracked = tracked_clip(clip, "recorded_roll")
            tracked["session_mode"] = "roll"
            with patch("dualcam.workflow.track_session", return_value=tracked), \
                 patch("dualcam.workflow.initialize_angles", return_value=clip["initial_angles"]):
                with self.assertRaisesRegex(WorkflowError, "allow-calibration-clip"):
                    run_motion(config_path, "roll", root / "without_opt_in", initial_roll_deg=0)
                report = run_motion(config_path, "roll", root / "replay", initial_roll_deg=0,
                                    allow_calibration_clip=True)
            self.assertTrue(report["success"], report["fit"]["diagnostics"])
            self.assertEqual(report["recording_mode"], "roll")
            self.assertEqual(report["evaluation"], "calibration_clip_consistency_check")
            self.assertTrue(any("not independent" in warning for warning in report["warnings"]))
            self.assertGreater(np.max(np.abs(expected[:, 1:])), .1)
            np.testing.assert_allclose(report["fit"]["datasets"][0]["angles"], expected, atol=.02)
            self.assertEqual(tracked["session_mode"], "roll")

    def test_calibration_clip_opt_in_still_rejects_checkerboard_recordings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, _, cameras, F, pivot = rig_files(root)
            clip, _ = make_clip(cameras, F, pivot, "motion", count=4, frames=5)
            tracked = tracked_clip(clip, "board")
            tracked["session_mode"] = "checkerboard"
            with patch("dualcam.workflow.track_session", return_value=tracked):
                with self.assertRaisesRegex(WorkflowError, "checkerboard"):
                    run_motion(config_path, "board", root / "rejected_board", initial_roll_deg=0,
                               allow_calibration_clip=True)

    def test_unverified_timing_and_missing_profiles_are_reported_without_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, cfg, cameras, F, pivot = rig_files(root)
            raw_cfg = yaml.safe_load(config_path.read_text())
            raw_cfg["timing"]["verified"] = False
            write_yaml(config_path, raw_cfg)
            for name in ("c920", "brio101"):
                path = root / f"{name}.yaml"
                data = yaml.safe_load(path.read_text())
                data["capture_profile"] = None
                write_yaml(path, data)
            stereo = yaml.safe_load((root / "stereo.yaml").read_text())
            stereo["intrinsics_sha256"] = {name: sha256(root / f"{name}.yaml") for name in ("c920", "brio101")}
            write_yaml(root / "stereo.yaml", stereo)
            roll, _ = make_clip(cameras, F, pivot, "roll", count=4, frames=5)
            swivel, _ = make_clip(cameras, F, pivot, "swivel", count=4, frames=5)
            accepted = {"success": True, "F": F, "pivot": pivot, "datasets": [], "diagnostics": {"reasons": []}}
            stdout = io.StringIO()
            with patch("dualcam.workflow.track_session", side_effect=[tracked_clip(roll, "roll"), tracked_clip(swivel, "swivel")]), \
                 patch("dualcam.workflow.initialize_axes", return_value=(F, {})), \
                 patch("dualcam.workflow.initialize_pivot", return_value=pivot), \
                 patch("dualcam.workflow.fit_joint", return_value=accepted), redirect_stdout(stdout):
                report = calibrate_axes(config_path, "roll", "swivel", root / "unverified")
            self.assertTrue(report["success"])
            self.assertEqual(len(report["warnings"]), 3)
            self.assertFalse(report["provenance"]["timing_verified_by_operator"])
            self.assertEqual(report["provenance"]["intrinsics_capture_profile_verified"], {"c920": False, "brio101": False})
            self.assertIn("Host receive timestamps", stdout.getvalue())
            self.assertIn("Fitting shared axes and pivot", stdout.getvalue())
            saved = json.loads((root / "unverified/report.json").read_text())
            self.assertEqual(saved["warnings"], report["warnings"])

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "results"
            output.mkdir()
            original = output / "important.txt"
            original.write_text("preserve me")
            with self.assertRaisesRegex(ValueError, "new or empty"):
                run_motion("unused", "unused", output, initial_roll_deg=0)
            self.assertEqual(original.read_text(), "preserve me")
            self.assertFalse((output / "results.json").exists())


if __name__ == "__main__":
    unittest.main()
