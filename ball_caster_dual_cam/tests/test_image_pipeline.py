"""Axis calibration bootstrap from real decoded, rendered webcam AVI files."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
from ballrot.track import PersistentKLTTracker

from dualcam.config import load_config, load_rig
from dualcam.model import project, rotation_z, world_points
from dualcam.solver import fit_joint
from dualcam.tracking import (initialize_angles, initialize_axes, initialize_pivot,
                              track_session)


NAMES = ("c920", "brio101")
RADIUS, GAP = .1, .02


def render_calibration_sessions(root):
    """Create paused-home pure roll/swivel clips; no camera or frontend mocks."""
    K = np.array([[650., 0., 320.], [0., 650., 240.], [0., 0., 1.]])
    pivot = np.array([0., 0., .55])
    center2 = np.array([-.26, .025, .015])
    forward = pivot - center2
    forward /= np.linalg.norm(forward)
    right = np.cross([0., 1., 0.], forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = rotation_z(np.pi) @ np.stack([right, down, forward])
    cameras = [{"K": K, "R": np.eye(3), "t": np.zeros(3)},
               {"K": K.copy(), "R": R, "t": -R @ center2}]
    F = Rotation.from_euler("xyz", [90., 8., 15.], degrees=True).as_matrix()
    frames = 14
    roll = np.zeros((frames, 3))
    roll[:, 0] = np.linspace(0., .30, frames)
    swivel = np.zeros((frames, 3))
    swivel[:, 1] = np.linspace(0., .40, frames)
    swivel[:, 2] = np.linspace(0., .33, frames)
    motions = {"roll": roll, "swivel": swivel}
    all_angles = np.vstack(list(motions.values()))
    rng = np.random.default_rng(218)
    triangles = []
    for shell in (0, 1):
        selected, previous_uv = [], [[], []]
        for _ in range(12000):
            p = rng.normal(size=3)
            p /= np.linalg.norm(p)
            p[2] = (1-2*shell) * abs(p[2])
            if abs(p[2]) < .14:
                continue
            tangent = np.cross(p, [0., 0., 1.])
            tangent /= np.linalg.norm(tangent)
            bitangent = np.cross(p, tangent)
            phases = np.array([.2, 2.1, 4.3])
            vertices = p + .070*(np.cos(phases)[:, None]*tangent + np.sin(phases)[:, None]*bitangent)
            vertices /= np.linalg.norm(vertices, axis=1)[:, None]
            projected, good = [], True
            for ci, camera in enumerate(cameras):
                origin = -camera["R"].T @ camera["t"]
                points = world_points(F, pivot, all_angles, shell, np.tile(p, (len(all_angles), 1)), RADIUS, GAP)
                centers = world_points(F, pivot, all_angles, shell, np.zeros((len(all_angles), 3)), RADIUS, GAP)
                if np.min(np.sum((points-centers)/RADIUS*(origin-points), axis=1)) < .09:
                    good = False
                    break
                uv = project(camera, points)[0]
                if previous_uv[ci] and np.min(np.linalg.norm(np.array(previous_uv[ci])-uv, axis=1)) < 14:
                    good = False
                    break
                projected.append(uv)
            if good:
                selected.append(vertices)
                for ci, uv in enumerate(projected):
                    previous_uv[ci].append(uv)
            if len(selected) == 12:
                break
        if len(selected) != 12:
            raise RuntimeError(f"Could not render enough patches on shell {shell}: {len(selected)}")
        triangles.append(selected)
    cfg = {"cameras": {}, "stereo": {"path": "stereo.yaml"}, "axes": {"path": "axes.yaml"},
           "geometry": {"radius_m": RADIUS, "gap_m": GAP, "red_shell_sign": 1,
                        "R_bc_seed": None, "pivot_c920_m": None},
           "timing": {"brio_offset_s": 0., "max_pair_skew_ms": 8., "verified": True},
           "tracking": {"max_corners": 24, "quality": .005, "min_distance_px": 6,
                        "klt_win": 17, "pyramid_levels": 3, "fwd_bwd_err_px": .8,
                        "min_axis_step_deg": .2}}
    for ci, (name, camera) in enumerate(zip(NAMES, cameras)):
        profile = {"width": 640, "height": 480, "fps": 30, "fourcc": "MJPG", "exposure_us": 4000,
                   "gain": 0, "white_balance_kelvin": 4000, "power_line_frequency": 2}
        if name == "c920":
            profile.update(focus=55, zoom=100)
        center_uv = project(camera, pivot)[0]
        depth = (camera["R"] @ pivot + camera["t"])[2]
        # Approximate enclosing sphere, deliberately not the true individual shell center.
        circle = [*center_uv.tolist(), float(650*.11/depth)]
        cfg["cameras"][name] = {
            "device": f"synthetic-{name}", "intrinsics": f"{name}.yaml", "capture": profile, "circle": circle,
            "segment": {"red_hsv": {"lo": [170, 65, 40], "hi": [12, 255, 255]},
                        "green_hsv": {"lo": [40, 45, 30], "hi": [100, 255, 255]},
                        "grow_px": 4, "separation_px": 2, "morphology_px": 3, "boundary_margin_px": 1}}
        (root / f"{name}.yaml").write_text(yaml.safe_dump({
            "image_size": [640, 480], "K": K.tolist(), "dist": [0.]*5, "capture_profile": profile}))
    for mode, angles in motions.items():
        directory = root / mode
        directory.mkdir()
        metadata = {"status": "complete", "mode": mode, "cameras": {}, "config_snapshot": cfg}
        for ci, (name, camera) in enumerate(zip(NAMES, cameras)):
            metadata["cameras"][name] = {"requested": cfg["cameras"][name]["capture"]}
            writer = cv2.VideoWriter(str(directory/f"{name}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30., (640, 480))
            if not writer.isOpened():
                raise RuntimeError("OpenCV MJPG encoding is required for this integration test.")
            try:
                for q in angles:
                    image = np.full((480, 640, 3), 215, dtype=np.uint8)
                    for shell, polygons in enumerate(triangles):
                        for points in polygons:
                            uv = project(camera, world_points(F, pivot, q, shell, points, RADIUS, GAP))
                            color = (25, 30, 220) if shell == 0 else (25, 185, 40)
                            cv2.fillConvexPoly(image, np.rint(uv*256).astype(np.int32), color,
                                               lineType=cv2.LINE_AA, shift=8)
                    writer.write(image)
            finally:
                writer.release()
            with (directory/f"{name}_timestamps.csv").open("w", newline="") as stream:
                table = csv.writer(stream)
                table.writerow(["frame_index", "timestamp_s", "read_start_s", "read_end_s"])
                table.writerows((f, f/30., f/30., f/30.) for f in range(frames))
        (directory/"session.json").write_text(json.dumps(metadata))
    (root/"stereo.yaml").write_text(yaml.safe_dump({"R_21": R.tolist(), "t_21_m": cameras[1]["t"].tolist()}))
    (root/"rig.yaml").write_text(yaml.safe_dump(cfg))
    return load_config(root/"rig.yaml"), F, pivot, motions


class ImageAxisCalibrationTests(unittest.TestCase):
    def test_unpaired_images_preserve_tracks_and_paired_rotation_interval(self):
        cv2.setNumThreads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg, truth_F, _, motions = render_calibration_sessions(root)
            # Simulate a host-time pairing hole while both videos retain all
            # images. The first motion increment spans nine native intervals.
            with (root/"roll/brio101_timestamps.csv").open("w", newline="") as stream:
                table = csv.writer(stream)
                table.writerow(["frame_index", "timestamp_s"])
                table.writerows((i, i/30 + (.015 if 1 <= i <= 8 else 0)) for i in range(14))
            calls = []
            original = PersistentKLTTracker.track_pair
            def count_pairs(tracker, *args, **kwargs):
                calls.append(1)
                return original(tracker, *args, **kwargs)
            with patch.object(PersistentKLTTracker, "track_pair", count_pairs):
                tracked = track_session(root/"roll", cfg, progress=None)
            selected = np.array([0, 9, 10, 11, 12, 13])
            np.testing.assert_array_equal(tracked["pairs"], np.column_stack([selected, selected]))
            self.assertEqual(len(calls), 2 * 13)
            obs = tracked["observations"]
            self.assertEqual(set(obs["frame"]), set(range(len(selected))))
            for ci in (0, 1):
                branch = obs["camera"] == ci
                home_ids = set(obs["track"][branch & (obs["frame"] == 0)])
                next_ids = set(obs["track"][branch & (obs["frame"] == 1)])
                self.assertGreaterEqual(len(home_ids & next_ids), 8)
            cameras, _, _ = load_rig(cfg)
            initial = initialize_angles(tracked, cameras, truth_F, "roll")
            np.testing.assert_allclose(initial[:, 0], motions["roll"][selected, 0], atol=.05)

    def test_axis_bootstrap_and_refinement_from_two_rendered_pure_motion_clips(self):
        cv2.setNumThreads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg, truth_F, truth_C, motions = render_calibration_sessions(root)
            cameras, _, _ = load_rig(cfg)
            tracked = {mode: track_session(root/mode, cfg, max_frames=20, progress=None)
                       for mode in motions}
            for clip in tracked.values():
                obs = clip["observations"]
                for ci in (0, 1):
                    for shell in (0, 1):
                        selected = (obs["camera"] == ci) & (obs["shell"] == shell)
                        counts = [len(np.unique(obs["frame"][selected & (obs["track"] == key)]))
                                  for key in np.unique(obs["track"][selected])]
                        self.assertGreaterEqual(sum(count >= 3 for count in counts), 6)
            initial_F, report = initialize_axes(tracked["roll"], tracked["swivel"], cameras, cfg)
            self.assertEqual(report["method"], "two_camera_rotation_vector_PCA_initializer_only")
            initial_C = initialize_pivot(tracked["roll"], cameras, cfg)
            datasets = [{"times": clip["times"], "mode": mode, "observations": clip["observations"],
                         "initial_angles": initialize_angles(clip, cameras, initial_F, mode)}
                        for mode, clip in tracked.items()]
            out = fit_joint(datasets, cameras, initial_F, initial_C, RADIUS, GAP,
                            calibrate_axes=True, refine_pivot=True)
            self.assertTrue(out["success"], out["diagnostics"])
            error_deg = np.rad2deg(Rotation.from_matrix(out["F"] @ truth_F.T).magnitude())
            self.assertLess(error_deg, 3.)
            self.assertLess(np.linalg.norm(out["pivot"]-truth_C), .004)
            for result, expected in zip(out["datasets"], motions.values()):
                self.assertGreater(result["valid"].mean(), .9)
                self.assertLess(np.rad2deg(np.nanmax(abs(result["angles"]-expected))), 3.)


if __name__ == "__main__":
    unittest.main()
