"""Deterministic swivel-caster renderer with real ArUco imagery.

The renderer is intentionally geometric rather than photorealistic.  It runs
the same tag detector, KLT tracker, ray-plane lift, and metrics code used for
real clips while retaining exact per-frame truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from common.rotation import Rz


Array = np.ndarray


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _unit(value: Sequence[float]) -> Array:
    vector = np.asarray(value, dtype=float)
    return vector / np.linalg.norm(vector)


def look_at_transform(camera_center_car: Array, target_car: Array) -> Array:
    """Return a car->OpenCV-camera transform for a z-up world."""

    center = np.asarray(camera_center_car, dtype=float).reshape(3)
    forward = _unit(np.asarray(target_car, dtype=float).reshape(3) - center)
    right = _unit(np.cross(forward, np.array([0.0, 0.0, 1.0])))
    down = _unit(np.cross(forward, right))
    R_cam_from_car = np.stack([right, down, forward])
    transform = np.eye(4)
    transform[:3, :3] = R_cam_from_car
    transform[:3, 3] = -R_cam_from_car @ center
    return transform


@dataclass(frozen=True)
class SyntheticCamera:
    width: int = 640
    height: int = 480
    K: Array = field(
        default_factory=lambda: np.array(
            [[720.0, 0.0, 319.5], [0.0, 720.0, 239.5], [0.0, 0.0, 1.0]]
        )
    )
    T_cam_from_car: Array = field(
        default_factory=lambda: look_at_transform(
            np.array([0.04, -0.40, 0.34]), np.array([0.14, 0.0, 0.085])
        )
    )

    @property
    def T_car_from_cam(self) -> Array:
        return np.linalg.inv(self.T_cam_from_car)

    @property
    def R_cam_from_car(self) -> Array:
        return self.T_cam_from_car[:3, :3]

    @property
    def camera_center_car(self) -> Array:
        return self.T_car_from_cam[:3, 3]


@dataclass(frozen=True)
class SyntheticGeometry:
    swivel_axis_car: Array = field(
        default_factory=lambda: np.array([0.18, 0.0, 0.16])
    )
    hub_offset0_car: Array = field(
        default_factory=lambda: np.array([-0.055, 0.0, -0.080])
    )
    axle0_car: Array = field(default_factory=lambda: np.array([0.0, -1.0, 0.0]))
    wheel_radius_m: float = 0.075
    wheel_width_m: float = 0.034
    tag_size_m: float = 0.040
    tag_plate_size_m: float = 0.062
    tag_height_m: float = 0.020

    @property
    def trail_m(self) -> float:
        return abs(float(self.hub_offset0_car[0]))

    @property
    def heading0_car(self) -> Array:
        return _unit(np.cross([0.0, 0.0, 1.0], self.axle0_car))

    def hub_car(self, psi: float) -> Array:
        return self.swivel_axis_car + Rz(float(psi)) @ self.hub_offset0_car

    def axle_car(self, psi: float) -> Array:
        return _unit(Rz(float(psi)) @ self.axle0_car)


@dataclass(frozen=True)
class SwivelTrajectory:
    t: Array
    phi: Array
    psi: Array
    car_x: Array
    car_y: Array
    car_theta: Array

    def __post_init__(self) -> None:
        values = [np.asarray(getattr(self, name), dtype=float) for name in (
            "t", "phi", "psi", "car_x", "car_y", "car_theta"
        )]
        n = len(values[0])
        if n < 2 or any(value.shape != (n,) for value in values):
            raise ValueError("trajectory arrays must share shape (N,), N >= 2")
        if np.any(np.diff(values[0]) <= 0.0):
            raise ValueError("trajectory time must increase")
        for name, value in zip(("t", "phi", "psi", "car_x", "car_y", "car_theta"), values):
            object.__setattr__(self, name, value)


def tracking_trajectory(
    num_frames: int = 91,
    fps: float = 60.0,
    *,
    delta_phi_deg: float = 3.0,
    delta_psi_deg: float = 1.0,
) -> SwivelTrajectory:
    t = np.arange(num_frames, dtype=float) / fps
    phase = np.arange(num_frames, dtype=float)
    phi = np.deg2rad(delta_phi_deg) * phase
    # A trend plus sinusoid exercises simultaneous swivel and roll without
    # leaving the camera's useful face for the baseline gate.
    psi = np.deg2rad(delta_psi_deg) * phase + np.deg2rad(12.0) * np.sin(
        2.0 * np.pi * phase / max(num_frames - 1, 1)
    )
    return SwivelTrajectory(t, phi, psi, 0.20 * t, np.zeros_like(t), np.zeros_like(t))


def canonical_trajectory(
    case: str,
    *,
    num_frames: int = 121,
    fps: float = 60.0,
    geometry: SyntheticGeometry | None = None,
) -> SwivelTrajectory:
    geom = geometry or SyntheticGeometry()
    t = np.arange(num_frames, dtype=float) / fps
    name = case.strip().casefold().replace("-", "_")
    if name == "straight":
        speed = 0.25
        return SwivelTrajectory(
            t,
            speed * t / geom.wheel_radius_m,
            np.zeros_like(t),
            speed * t,
            np.zeros_like(t),
            np.zeros_like(t),
        )
    if name in {"spin", "spin_in_place"}:
        yaw_rate = 0.9
        # Deliberately locked/misaligned wheel: an obvious scrub ground truth.
        return SwivelTrajectory(
            t,
            np.zeros_like(t),
            np.zeros_like(t),
            np.zeros_like(t),
            np.zeros_like(t),
            yaw_rate * t,
        )
    if name in {"circle", "constant_radius_circle"}:
        speed, yaw_rate = 0.25, 0.45
        radius = speed / yaw_rate
        theta = yaw_rate * t
        x = radius * np.sin(theta)
        y = radius * (1.0 - np.cos(theta))
        demand = np.arctan2(yaw_rate * geom.swivel_axis_car[0], speed)
        psi = np.full_like(t, demand + np.deg2rad(8.0))
        contact_speed = np.hypot(speed, yaw_rate * geom.swivel_axis_car[0])
        # Only the component along the deliberately offset wheel heading rolls;
        # the lateral component is analytic scrub.
        phi = contact_speed * np.cos(np.deg2rad(8.0)) * t / geom.wheel_radius_m
        return SwivelTrajectory(t, phi, psi, x, y, theta)
    raise ValueError("case must be straight, spin_in_place, or circle")


@dataclass(frozen=True)
class Degradations:
    pixel_noise_sigma: float = 0.0
    motion_blur: bool = False
    glare: bool = False
    speckle_fraction: float = 1.0


@dataclass(frozen=True)
class RenderConfig:
    camera: SyntheticCamera = field(default_factory=SyntheticCamera)
    geometry: SyntheticGeometry = field(default_factory=SyntheticGeometry)
    marker_dictionary: str = "DICT_4X4_50"
    marker_id: int = 0
    dot_count_per_face: int = 520
    dot_radius_px: int = 2
    seed: int = 24681357
    degradations: Degradations = field(default_factory=Degradations)


@dataclass
class RenderResult:
    frames: list[Array]
    truth: dict[str, Any]
    camera: SyntheticCamera
    geometry: SyntheticGeometry


def _aruco_dictionary(name: str) -> Any:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco is unavailable; install opencv-contrib-python")
    code = getattr(cv2.aruco, name, None)
    if code is None:
        raise ValueError(f"unknown ArUco dictionary {name!r}")
    return cv2.aruco.getPredefinedDictionary(code)


def _marker_image(dictionary: Any, marker_id: int, pixels: int = 192) -> Array:
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, int(marker_id), pixels)
    if hasattr(dictionary, "generateImageMarker"):
        return dictionary.generateImageMarker(int(marker_id), pixels)
    image = np.empty((pixels, pixels), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, int(marker_id), pixels, image, 1)
    return image


def _transform_points(points_car: Array, T_cam_from_car: Array) -> Array:
    points = np.asarray(points_car, dtype=float)
    return points @ T_cam_from_car[:3, :3].T + T_cam_from_car[:3, 3]


def _project(points_car: Array, camera: SyntheticCamera) -> tuple[Array, Array]:
    points_cam = _transform_points(points_car, camera.T_cam_from_car)
    homogeneous = points_cam @ camera.K.T
    valid = points_cam[:, 2] > 1e-8
    uv = np.full((len(points_cam), 2), np.nan)
    uv[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:3]
    valid &= (
        (uv[:, 0] >= -20)
        & (uv[:, 0] < camera.width + 20)
        & (uv[:, 1] >= -20)
        & (uv[:, 1] < camera.height + 20)
    )
    return uv, valid


def _axis_rotation(axis: Array, angle: float) -> Array:
    return Rotation.from_rotvec(_unit(axis) * float(angle)).as_matrix()


def _speckle_population(config: RenderConfig) -> tuple[Array, Array, Array]:
    rng = np.random.default_rng(config.seed)
    count = config.dot_count_per_face
    fraction = float(np.clip(config.degradations.speckle_fraction, 0.05, 1.0))
    count = max(16, int(round(count * fraction)))
    inner = 0.22 * config.geometry.wheel_radius_m
    outer = 0.95 * config.geometry.wheel_radius_m
    radii = np.sqrt(rng.uniform(inner**2, outer**2, size=(2, count)))
    angles = rng.uniform(-np.pi, np.pi, size=(2, count))
    brightness = rng.integers(155, 256, size=(2, count))
    return radii, angles, brightness


def _draw_wheel(
    image: Array,
    phi: float,
    psi: float,
    config: RenderConfig,
    population: tuple[Array, Array, Array],
) -> dict[str, Any]:
    geom, camera = config.geometry, config.camera
    hub = geom.hub_car(psi)
    axle = geom.axle_car(psi)
    heading = _unit(np.cross([0.0, 0.0, 1.0], axle))
    view_scores: dict[str, float] = {}
    visible_faces: list[int] = []
    theta_boundary = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    for face_index, sign in enumerate((-1, 1)):
        center = hub + sign * 0.5 * geom.wheel_width_m * axle
        normal = sign * axle
        view = _unit(camera.camera_center_car - center)
        score = float(normal @ view)
        view_scores[str(sign)] = score
        if score <= 0.0:
            continue
        visible_faces.append(sign)
        boundary = center + geom.wheel_radius_m * (
            np.cos(theta_boundary)[:, None] * heading
            + np.sin(theta_boundary)[:, None] * np.array([0.0, 0.0, 1.0])
        )
        uv, valid = _project(boundary, camera)
        if np.count_nonzero(valid) < 8:
            continue
        polygon = np.rint(uv[valid]).astype(np.int32)
        cv2.fillConvexPoly(image, polygon, (42, 45, 48), cv2.LINE_AA)
        inner = center + 0.19 * geom.wheel_radius_m * (
            np.cos(theta_boundary)[:, None] * heading
            + np.sin(theta_boundary)[:, None] * np.array([0.0, 0.0, 1.0])
        )
        inner_uv, inner_valid = _project(inner, camera)
        if np.count_nonzero(inner_valid) >= 8:
            cv2.fillConvexPoly(
                image, np.rint(inner_uv[inner_valid]).astype(np.int32), (82, 85, 88), cv2.LINE_AA
            )

        radii, angles, brightness = population
        local_radial = radii[face_index, :, None] * (
            np.cos(angles[face_index])[:, None] * geom.heading0_car
            + np.sin(angles[face_index])[:, None] * np.array([0.0, 0.0, 1.0])
        )
        spun = local_radial @ _axis_rotation(geom.axle0_car, phi).T
        points = geom.swivel_axis_car + (
            geom.hub_offset0_car
            + sign * 0.5 * geom.wheel_width_m * geom.axle0_car
            + spun
        ) @ Rz(psi).T
        dot_uv, dot_valid = _project(points, camera)
        for point, okay, value in zip(dot_uv, dot_valid, brightness[face_index]):
            if okay:
                shade = int(value)
                cv2.circle(image, tuple(np.rint(point).astype(int)), config.dot_radius_px, (shade, shade, shade), -1, cv2.LINE_AA)

        # Bold absolute phase reference at a known material angle.
        reference_local = 0.72 * geom.wheel_radius_m * geom.heading0_car
        reference = geom.swivel_axis_car + Rz(psi) @ (
            geom.hub_offset0_car
            + sign * 0.5 * geom.wheel_width_m * geom.axle0_car
            + _axis_rotation(geom.axle0_car, phi) @ reference_local
        )
        reference_uv, reference_valid = _project(reference[None, :], camera)
        if reference_valid[0]:
            cv2.circle(image, tuple(np.rint(reference_uv[0]).astype(int)), 5, (30, 30, 245), -1, cv2.LINE_AA)
            cv2.circle(image, tuple(np.rint(reference_uv[0]).astype(int)), 5, (255, 255, 255), 1, cv2.LINE_AA)
    return {"hub_car": hub, "axle_car": axle, "view_scores": view_scores, "visible_faces": visible_faces}


def _draw_fork(image: Array, psi: float, config: RenderConfig) -> None:
    geom, camera = config.geometry, config.camera
    # A slim yoke bar gives realistic occlusion without covering the sidewall.
    local = np.array(
        [[-0.070, -0.026, 0.005], [0.025, -0.026, 0.005], [0.025, 0.026, 0.005], [-0.070, 0.026, 0.005]]
    )
    local[:, 2] += geom.swivel_axis_car[2]
    local[:, :2] += geom.swivel_axis_car[:2]
    center = geom.swivel_axis_car.copy()
    points = center + (local - center) @ Rz(psi).T
    uv, valid = _project(points, camera)
    if np.all(valid):
        cv2.fillConvexPoly(image, np.rint(uv).astype(np.int32), (70, 74, 80), cv2.LINE_AA)


def _draw_tag(image: Array, psi: float, config: RenderConfig, marker: Array) -> Array:
    geom, camera = config.geometry, config.camera
    quiet = int(
        round(
            0.5
            * marker.shape[0]
            * (geom.tag_plate_size_m / geom.tag_size_m - 1.0)
        )
    )
    plate_pixels = marker.shape[0] + 2 * quiet
    plate = np.full((plate_pixels, plate_pixels), 255, dtype=np.uint8)
    plate[quiet : quiet + marker.shape[0], quiet : quiet + marker.shape[1]] = marker
    center = geom.swivel_axis_car + np.array([0.0, 0.0, geom.tag_height_m])
    half = 0.5 * geom.tag_plate_size_m
    local_corners = np.array([[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]])
    corners_car = center + local_corners @ Rz(psi).T
    destination, valid = _project(corners_car, camera)
    if not np.all(valid):
        return np.full((4, 2), np.nan)
    source = np.array([[0.0, 0.0], [plate_pixels - 1.0, 0.0], [plate_pixels - 1.0, plate_pixels - 1.0], [0.0, plate_pixels - 1.0]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(source, destination.astype(np.float32))
    warped = cv2.warpPerspective(plate, H, (camera.width, camera.height), flags=cv2.INTER_NEAREST, borderValue=255)
    mask = cv2.warpPerspective(np.full_like(plate, 255), H, (camera.width, camera.height), flags=cv2.INTER_NEAREST, borderValue=0)
    image[mask > 0] = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)[mask > 0]
    # Return true marker (not plate) corners for diagnostics.
    marker_half = 0.5 * geom.tag_size_m
    marker_local = np.array([[-marker_half, marker_half, 0.0], [marker_half, marker_half, 0.0], [marker_half, -marker_half, 0.0], [-marker_half, -marker_half, 0.0]])
    marker_uv, _ = _project(center + marker_local @ Rz(psi).T, camera)
    return marker_uv


def _render_pose(
    phi: float,
    psi: float,
    config: RenderConfig,
    population: tuple[Array, Array, Array],
    marker: Array,
) -> tuple[Array, dict[str, Any]]:
    camera = config.camera
    image = np.full((camera.height, camera.width, 3), (222, 224, 226), dtype=np.uint8)
    wheel = _draw_wheel(image, phi, psi, config, population)
    _draw_fork(image, psi, config)
    tag_corners = _draw_tag(image, psi, config, marker)
    if config.degradations.glare:
        cv2.ellipse(image, (360, 285), (55, 18), -18, 0, 360, (252, 252, 252), -1, cv2.LINE_AA)
    return image, {**wheel, "tag_corners_uv": tag_corners}


def render_sequence(
    trajectory: SwivelTrajectory | None = None,
    *,
    config: RenderConfig | None = None,
    output_dir: str | Path | None = None,
) -> RenderResult:
    cfg = config or RenderConfig()
    motion = trajectory or tracking_trajectory()
    dictionary = _aruco_dictionary(cfg.marker_dictionary)
    marker = _marker_image(dictionary, cfg.marker_id)
    population = _speckle_population(cfg)
    frames: list[Array] = []
    per_frame: list[dict[str, Any]] = []
    rng = np.random.default_rng(cfg.seed + 991)
    for index, (phi, psi) in enumerate(zip(motion.phi, motion.psi)):
        if cfg.degradations.motion_blur and index > 0:
            subframes = []
            subtruth = None
            for blend in (0.72, 0.86, 1.0):
                p = (1.0 - blend) * motion.phi[index - 1] + blend * phi
                s = (1.0 - blend) * motion.psi[index - 1] + blend * psi
                subframe, subtruth = _render_pose(float(p), float(s), cfg, population, marker)
                subframes.append(subframe.astype(np.float32))
            frame = np.rint(np.mean(subframes, axis=0)).astype(np.uint8)
            detail = subtruth or {}
        else:
            frame, detail = _render_pose(float(phi), float(psi), cfg, population, marker)
        sigma = float(cfg.degradations.pixel_noise_sigma)
        if sigma > 0.0:
            # Spatially varying sub-pixel warp models localization error in
            # detected corners/speckles (sigma is genuinely measured in px).
            # A coarse field keeps the image physical enough for KLT while
            # preventing a harmless whole-frame translation from gaming the
            # robustness gate.
            camera = cfg.camera
            coarse_shape = (max(3, camera.height // 64), max(4, camera.width // 64))
            dx = cv2.resize(
                rng.normal(0.0, sigma, coarse_shape).astype(np.float32),
                (camera.width, camera.height),
                interpolation=cv2.INTER_CUBIC,
            )
            dy = cv2.resize(
                rng.normal(0.0, sigma, coarse_shape).astype(np.float32),
                (camera.width, camera.height),
                interpolation=cv2.INTER_CUBIC,
            )
            grid_x, grid_y = np.meshgrid(
                np.arange(camera.width, dtype=np.float32),
                np.arange(camera.height, dtype=np.float32),
            )
            frame = cv2.remap(
                frame,
                grid_x + dx,
                grid_y + dy,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(222, 224, 226),
            )
            frame = np.clip(
                frame.astype(float) + rng.normal(0.0, 0.75 * sigma, frame.shape),
                0,
                255,
            ).astype(np.uint8)
        frames.append(frame)
        per_frame.append(_json_safe(detail))

    truth = {
        "schema": "synthetic-swivel-v1",
        "K": cfg.camera.K,
        "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
        "T_cam_from_car": cfg.camera.T_cam_from_car,
        "T_car_from_cam": cfg.camera.T_car_from_cam,
        "image_size": [cfg.camera.width, cfg.camera.height],
        "marker": {"dictionary": cfg.marker_dictionary, "id": cfg.marker_id, "size_m": cfg.geometry.tag_size_m},
        "geometry": {
            "swivel_axis_car": cfg.geometry.swivel_axis_car,
            "hub_offset0_car": cfg.geometry.hub_offset0_car,
            "axle0_car": cfg.geometry.axle0_car,
            "wheel_radius_m": cfg.geometry.wheel_radius_m,
            "wheel_width_m": cfg.geometry.wheel_width_m,
            "trail_m": cfg.geometry.trail_m,
        },
        "trajectory": {
            "t": motion.t,
            "phi": motion.phi,
            "psi": motion.psi,
            "car_x": motion.car_x,
            "car_y": motion.car_y,
            "car_theta": motion.car_theta,
        },
        "frames": per_frame,
        "degradations": _json_safe(cfg.degradations.__dict__),
    }
    result = RenderResult(frames, _json_safe(truth), cfg.camera, cfg.geometry)
    if output_dir is not None:
        destination = Path(output_dir).expanduser().resolve()
        frame_dir = destination / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames):
            if not cv2.imwrite(str(frame_dir / f"frame_{index:05d}.png"), frame):
                raise OSError(f"failed to write synthetic frame {index}")
        (destination / "ground_truth.json").write_text(
            json.dumps(result.truth, indent=2), encoding="utf-8"
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "out" / "synthetic" / "baseline")
    parser.add_argument("--case", choices=("tracking", "straight", "spin_in_place", "circle"), default="tracking")
    parser.add_argument("--frames", type=int, default=91)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--delta-phi-deg", type=float, default=3.0)
    parser.add_argument("--delta-psi-deg", type=float, default=1.0)
    parser.add_argument("--noise-px", type=float, default=0.0)
    parser.add_argument("--blur", action="store_true")
    parser.add_argument("--glare", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.frames < 2 or args.fps <= 0:
        raise SystemExit("--frames must be >=2 and --fps must be positive")
    trajectory = (
        tracking_trajectory(args.frames, args.fps, delta_phi_deg=args.delta_phi_deg, delta_psi_deg=args.delta_psi_deg)
        if args.case == "tracking"
        else canonical_trajectory(args.case, num_frames=args.frames, fps=args.fps)
    )
    cfg = RenderConfig(degradations=Degradations(args.noise_px, args.blur, args.glare, 1.0))
    result = render_sequence(trajectory, config=cfg, output_dir=args.output)
    print(f"Wrote {len(result.frames)} frames and ground_truth.json to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
