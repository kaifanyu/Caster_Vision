"""Independent color/KLT branches with shared timing and camera-local track IDs."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ballrot.calibration import detect_circle_auto
from ballrot.estimate import solve_hemisphere_increment
from ballrot.integrate import calibrate_ball_frame
from ballrot.segment import segment_frame
from ballrot.sphere import sphere_pose_from_circle
from ballrot.track import KLTConfig, PersistentKLTTracker
from .config import CAMERA_NAMES, load_intrinsics, rotation
from .session import SelectedVideo, load_session


def masks_for(frame, circle, config):
    segmentation = config["segment"]
    masks = segment_frame(
        frame, circle, mode="color", top_hsv=segmentation["red_hsv"],
        bottom_hsv=segmentation["green_hsv"],
        grow_px=segmentation.get("grow_px", 5),
        separation_px=segmentation.get("separation_px", 5),
        morph_kernel=segmentation.get("morphology_px", 3),
        paint_support={"enabled": True, "min_saturation": 40, "min_value": 30,
                       "max_distance_px": segmentation.get("grow_px", 5)},
    )
    margin = int(segmentation.get("boundary_margin_px", 1))
    if margin < 0:
        raise ValueError("boundary_margin_px must be nonnegative")
    result = {}
    for name in ("top", "bottom"):
        mask = masks[name].astype(np.uint8)
        if margin:
            mask = cv2.erode(mask, np.ones((2*margin+1, 2*margin+1), np.uint8))
        result[name] = mask.astype(bool)
    return result


def choose_circle(frame, camera_cfg):
    configured = camera_cfg.get("circle")
    if configured is None:
        u, v, r, _ = detect_circle_auto(frame)
        return (u, v, r)
    circle = np.asarray(configured, dtype=float)
    if circle.shape != (3,) or not np.isfinite(circle).all() or circle[2] <= 0:
        raise ValueError("camera.circle must be [u0,v0,r_px] in undistorted image pixels")
    if not (0 <= circle[0] < frame.shape[1] and 0 <= circle[1] < frame.shape[0]):
        raise ValueError("Configured circle center is outside the captured image")
    return tuple(circle)


def track_session(path, cfg, *, max_frames=180, progress=print):
    session = load_session(path, cfg, max_frames=max_frames)
    intrinsics = [load_intrinsics(cfg["cameras"][name]) for name in CAMERA_NAMES]
    n = len(session["times"])
    increments = np.full((2, 2, n-1, 3, 3), np.nan)
    support = np.zeros((2, 2, n-1), dtype=int)
    rows = []
    circles = []
    for camera_id, name in enumerate(CAMERA_NAMES):
        info = intrinsics[camera_id]
        video = SelectedVideo(session["path"]/f"{name}.avi", info["image_size"])
        mapx, mapy = cv2.initUndistortRectifyMap(info["K"], info["dist"], None, info["K"],
                                              tuple(info["image_size"]), cv2.CV_32FC1)
        tracker = PersistentKLTTracker(KLTConfig.from_mapping(cfg.get("tracking", {})))
        collected = {}
        previous = previous_masks = None
        paired_indices = {int(source): i for i, source in enumerate(session["pairs"][:, camera_id])}
        previous_paired_points = None
        try:
            # Keep identities through unpaired source frames. Jumping directly
            # between selected pairs can destroy home tracks across a timing gap.
            for source_index in range(int(session["pairs"][0, camera_id]),
                                      int(session["pairs"][-1, camera_id]) + 1):
                frame_index = paired_indices.get(source_index)
                raw = video.read(source_index)
                frame = cv2.remap(raw, mapx, mapy, cv2.INTER_LINEAR)
                if previous is None:
                    circle = choose_circle(frame, cfg["cameras"][name])
                    circles.append(circle)
                    C, _ = sphere_pose_from_circle(*circle, info["K"])
                masks = masks_for(frame, circle, cfg["cameras"][name])
                if previous is None:
                    tracker.initialize(frame, masks)
                else:
                    matches = tracker.track_pair(previous, frame, previous_masks, masks)
                    for shell, old_name in enumerate(("top", "bottom")):
                        match = matches[old_name]
                        if match.count:
                            # Pixels survive only the measured forward/backward LK test.
                            # The approximate sphere is NOT used to reject pixel tracks.
                            for identifier, uv0, uv1, error in zip(match.track_ids, match.uv_prev,
                                                                 match.uv_curr, match.fb_error):
                                weight = 1/np.sqrt(1+float(error)**2)
                                for fi, uv in ((paired_indices.get(source_index-1), uv0),
                                               (frame_index, uv1)):
                                    if fi is not None:
                                        key = (shell, fi, int(identifier))
                                        collected[key] = (camera_id, shell, fi, int(identifier), uv, weight)
                if frame_index is not None:
                    points = []
                    for shell, old_name in enumerate(("top", "bottom")):
                        uv, identifiers = tracker.points(old_name)
                        current = {int(identifier): point for identifier, point in zip(identifiers, uv)}
                        points.append(current)
                        if previous_paired_points is not None:
                            prior = previous_paired_points[shell]
                            common = sorted(prior.keys() & current.keys())
                            if len(common) >= 8:
                                # Endpoint correspondences retain IDs only if
                                # every intermediate KLT step survived. This is
                                # the rotation BETWEEN paired frames, not just
                                # the last native-frame increment.
                                estimate = solve_hemisphere_increment(
                                    np.asarray([prior[k] for k in common]),
                                    np.asarray([current[k] for k in common]), info["K"], C,
                                    ransac_iters=120, min_inliers=8, rng=7+frame_index,
                                )
                                if estimate.success:
                                    increments[camera_id, shell, frame_index-1] = estimate.R
                                    support[camera_id, shell, frame_index-1] = estimate.inlier_count
                    previous_paired_points = points
                previous, previous_masks = frame, masks
                if progress and frame_index is not None and (frame_index+1) % 30 == 0:
                    progress(f"{name}: tracked {frame_index+1}/{n} paired frames", flush=True)
        finally:
            video.close()
        rows.extend(collected.values())
    if not rows:
        raise ValueError("No surviving tracks. Review circles, color masks, focus and exposure with preview.py")
    columns = list(zip(*rows))
    observations = {key: np.asarray(values, dtype=int if key in ("camera", "shell", "frame", "track") else float)
                    for key, values in zip(("camera", "shell", "frame", "track", "uv", "weight"), columns)}
    return {"observations": observations, "times": session["times"],
            "increments": increments, "increment_support": support,
            "circles": np.asarray(circles), "pairs": session["pairs"], "session": str(session["path"]),
            "session_report": {**session["report"],
                               "tracking_image_policy": "Track every native image between first and last selected pair; fit only paired observations.",
                               "tracked_source_frames": (session["pairs"][-1]-session["pairs"][0]+1).tolist()},
            "session_mode": session["metadata"].get("mode"),
            "intrinsics": intrinsics}


def common_increments(tracked, cameras, shell=None):
    """Rotate raw initializer evidence into C920 coordinates; never average Euler angles."""
    increments = tracked["increments"]
    result = []
    for ci, camera in enumerate(cameras):
        R = np.asarray(camera["R"])
        for si in range(2) if shell is None else (shell,):
            for dR in increments[ci, si]:
                if np.isfinite(dR).all():
                    result.append(R.T @ dR @ R)
    return result


def initialize_axes(roll, swivel, cameras, cfg, *, roll_sign=1, swivel_sign=1):
    seed = cfg["geometry"].get("R_bc_seed")
    if seed is not None:
        return rotation(seed, "geometry.R_bc_seed"), {"method": "explicit_user_seed"}
    F, report = calibrate_ball_frame(
        common_increments(roll, cameras), common_increments(swivel, cameras),
        roll_sign=roll_sign, swivel_sign=swivel_sign,
        min_step_deg=cfg.get("tracking", {}).get("min_axis_step_deg", .2),
    )
    if min(report["roll"]["sample_count"], report["swivel"]["sample_count"]) < 8:
        raise ValueError("Too few moving increments for axes; record clear pure-motion sweeps")
    if report["raw_axis_separation_deg"] < 60:
        raise ValueError("Roll and swivel initializer axes are too close; inspect motion purity and masks")
    return F, {"method": "two_camera_rotation_vector_PCA_initializer_only", **report}


def initialize_pivot(tracked, cameras, cfg):
    seed = cfg["geometry"].get("pivot_c920_m")
    if seed is not None:
        value = np.asarray(seed, float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("geometry.pivot_c920_m must be a finite 3-vector in meters")
        return value
    geo = cfg["geometry"]
    enclosing_radius = geo["radius_m"] + geo["gap_m"]/2
    seeds = []
    for ci, camera in enumerate(cameras):
        normalized, _ = sphere_pose_from_circle(*tracked["circles"][ci], camera["K"])
        seeds.append(camera["R"].T @ (enclosing_radius*normalized-camera["t"]))
    # Bootstrap only. The separated-hemisphere calibration refines this pivot.
    return np.mean(seeds, axis=0)


def initialize_angles(tracked, cameras, F, mode, initial_roll_rad=0.):
    n = len(tracked["times"])
    angles = np.zeros((n, 3))
    angles[:, 0] = initial_roll_rad
    poses = [np.eye(3), np.eye(3)]
    effective = F @ Rotation.from_rotvec([initial_roll_rad, 0, 0]).as_matrix()
    for i in range(1, n):
        candidates = []
        for shell in range(2):
            vecs, weights = [], []
            for ci, camera in enumerate(cameras):
                dR = tracked["increments"][ci, shell, i-1]
                if np.isfinite(dR).all():
                    transformed = camera["R"].T @ dR @ camera["R"]
                    vecs.append(Rotation.from_matrix(transformed).as_rotvec())
                    weights.append(max(1, tracked["increment_support"][ci, shell, i-1]))
            if vecs:
                increment = Rotation.from_rotvec(np.average(vecs, axis=0, weights=weights)).as_matrix()
                poses[shell] = increment @ poses[shell]
                xyz = Rotation.from_matrix(effective.T @ poses[shell] @ effective).as_euler("XYZ")
                candidates.append(xyz[0])
                angles[i, shell+1] = xyz[2]
            else:
                angles[i, shell+1] = angles[i-1, shell+1]
        angles[i, 0] = initial_roll_rad + (np.arctan2(np.sin(candidates).sum(), np.cos(candidates).sum())
                                          if candidates else angles[i-1, 0]-initial_roll_rad)
    angles = np.unwrap(angles, axis=0)
    if mode == "roll":
        angles[:, 1:] = 0
    elif mode == "swivel":
        angles[:, 0] = initial_roll_rad
    elif mode != "motion":
        raise ValueError("mode must be roll, swivel or motion")
    angles[0] = [initial_roll_rad, 0, 0]
    return angles


def save_tracks(path, tracked):
    np.savez_compressed(path, **tracked["observations"], times=tracked["times"],
                        increments=tracked["increments"], increment_support=tracked["increment_support"],
                        circles=tracked["circles"])
