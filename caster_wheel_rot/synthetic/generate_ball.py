"""Render deterministic, two-colour speckled ball-caster sequences.

The renderer is intentionally simple: material speckles are fixed unit vectors
on each hemisphere, each hemisphere follows ``R_axis(alpha) @ Rz(beta)`` (with
the original +x roll axis as the default), and the same pinhole geometry
consumed by the measurement pipeline projects them.  All angles written to
``ground_truth.json`` are radians.

The module can be used as a library::

    result = generate_sequence("out/synthetic/default")

or as a command line utility::

    python synthetic/generate_ball.py --output out/synthetic/ball/default
    python synthetic/generate_ball.py --output out/synthetic/ball/calibration \
        --mode calibration-suite
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


Array = np.ndarray


def _rx(angle: float) -> Array:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(angle: float) -> Array:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(angle: float) -> Array:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fibonacci_sphere(n: int) -> Array:
    """Return ``n`` repeatable, nearly uniform unit directions.

    The half-sample offset avoids points exactly on either pole or the equator,
    which makes the top/bottom material assignment unambiguous.
    """

    if n < 1:
        raise ValueError("n must be at least 1")
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * i
    points = np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
        axis=1,
    )
    return points


def project(
    points_ball_t0: Array,
    R_motion: Array,
    R_bc: Array,
    C: Array,
    radius: float,
    K: Array,
) -> tuple[Array, Array]:
    """Project rotated material directions and report front-facing points.

    Rows represent vectors.  ``R_motion`` acts in ball coordinates and
    ``R_bc`` maps ball coordinates to camera coordinates.  Thus the returned
    correspondences obey the pipeline convention ``b = R @ a``.
    """

    points = np.asarray(points_ball_t0, dtype=np.float64)
    R_motion = np.asarray(R_motion, dtype=np.float64)
    R_bc = np.asarray(R_bc, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_ball_t0 must have shape (N, 3)")
    if R_motion.shape != (3, 3) or R_bc.shape != (3, 3) or K.shape != (3, 3):
        raise ValueError("R_motion, R_bc, and K must be 3x3")
    if C.shape != (3,) or radius <= 0:
        raise ValueError("C must have shape (3,) and radius must be positive")

    p = points @ R_motion.T
    n_cam = p @ R_bc.T
    x_cam = C + float(radius) * n_cam
    facing = (np.einsum("ij,ij->i", x_cam, n_cam) < 0.0) & (x_cam[:, 2] > 0.0)
    homogeneous = x_cam @ K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = homogeneous[:, :2] / homogeneous[:, 2:3]
    facing &= np.isfinite(uv).all(axis=1)
    return uv, facing


@dataclass(frozen=True)
class Degradations:
    """Optional, independently controllable robustness degradations.

    ``pixel_noise_std_px`` jitters projected speckle centres, in pixels.
    ``motion_blur_samples`` averages evenly spaced exposure sub-frames; set it
    above one to enable blur.  ``density_scale`` is the retained fraction of
    material speckles, and ``speed_scale`` multiplies every scripted angle.
    """

    pixel_noise_std_px: float = 0.0
    motion_blur_samples: int = 1
    motion_blur_fraction: float = 0.8
    glare: bool = False
    glare_strength: float = 0.78
    density_scale: float = 1.0
    speed_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.pixel_noise_std_px < 0:
            raise ValueError("pixel_noise_std_px must be non-negative")
        if self.motion_blur_samples < 1:
            raise ValueError("motion_blur_samples must be at least 1")
        if not 0.0 <= self.motion_blur_fraction <= 1.0:
            raise ValueError("motion_blur_fraction must be in [0, 1]")
        if not 0.0 <= self.glare_strength <= 1.0:
            raise ValueError("glare_strength must be in [0, 1]")
        if not 0.0 < self.density_scale <= 1.0:
            raise ValueError("density_scale must be in (0, 1]")
        if self.speed_scale <= 0:
            raise ValueError("speed_scale must be positive")


@dataclass(frozen=True)
class CameraSetup:
    """Known pinhole camera and scale-free sphere geometry."""

    image_size: tuple[int, int]  # (height, width)
    K: Array
    R_bc: Array
    C: Array
    radius: float
    circle: tuple[float, float, float]  # (u0, v0, r_px)
    dist: Array = field(default_factory=lambda: np.zeros(5, dtype=np.float64))

    def __post_init__(self) -> None:
        height, width = self.image_size
        if height < 16 or width < 16:
            raise ValueError("image_size must be (height, width), both >= 16")
        K = np.asarray(self.K, dtype=np.float64)
        R_bc = np.asarray(self.R_bc, dtype=np.float64)
        C = np.asarray(self.C, dtype=np.float64)
        dist = np.asarray(self.dist, dtype=np.float64).reshape(-1)
        if K.shape != (3, 3) or R_bc.shape != (3, 3) or C.shape != (3,):
            raise ValueError("K/R_bc/C shapes must be (3,3)/(3,3)/(3,)")
        if not np.allclose(R_bc.T @ R_bc, np.eye(3), atol=1e-10):
            raise ValueError("R_bc must be orthonormal")
        if not np.isclose(np.linalg.det(R_bc), 1.0, atol=1e-10):
            raise ValueError("R_bc must be a proper rotation")
        if self.radius <= 0 or self.circle[2] <= 0:
            raise ValueError("sphere and projected-circle radii must be positive")
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "R_bc", R_bc)
        object.__setattr__(self, "C", C)
        object.__setattr__(self, "dist", dist)


def default_camera(
    image_size: tuple[int, int] = (480, 640),
    circle_radius_px: float | None = None,
) -> CameraSetup:
    """Build a centred camera with a mildly tilted, known ball frame."""

    height, width = image_size
    r_px = float(circle_radius_px if circle_radius_px is not None else min(height, width) * 0.32)
    u0, v0 = width / 2.0, height / 2.0
    focal = float(max(height, width) * 1.05)
    K = np.array([[focal, 0.0, u0], [0.0, focal, v0], [0.0, 0.0, 1.0]])

    # At zero tilt: ball +x points image-right, +z points image-up, and +y
    # points away from the camera.  A small camera-frame tilt prevents the
    # calibration harness from succeeding only in a perfectly aligned case.
    aligned = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    R_bc = _rz(np.deg2rad(4.0)) @ _ry(np.deg2rad(-6.0)) @ aligned

    # Exact centred pinhole relation: r_image = f*r/sqrt(d^2-r^2).
    radius = 1.0
    distance = radius * np.sqrt(1.0 + (focal / r_px) ** 2)
    C = np.array([0.0, 0.0, distance])
    return CameraSetup(
        image_size=image_size,
        K=K,
        R_bc=R_bc,
        C=C,
        radius=radius,
        circle=(u0, v0, r_px),
    )


@dataclass(frozen=True)
class Trajectory:
    """Absolute ball-frame DOF values at each nominal video frame.

    ``roll_axis_ball`` is fixed within one clip but may differ between clips,
    matching the direction-free response of an ideal spherical caster.
    """

    alpha: Array
    beta_top: Array
    beta_bottom: Array
    fps: float = 60.0
    name: str = "custom"
    roll_axis_ball: Array = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )

    def __post_init__(self) -> None:
        alpha = np.asarray(self.alpha, dtype=np.float64).reshape(-1)
        beta_top = np.asarray(self.beta_top, dtype=np.float64).reshape(-1)
        beta_bottom = np.asarray(self.beta_bottom, dtype=np.float64).reshape(-1)
        roll_axis = np.asarray(self.roll_axis_ball, dtype=np.float64).reshape(-1)
        if len(alpha) < 2:
            raise ValueError("a trajectory needs at least two frames")
        if len(alpha) != len(beta_top) or len(alpha) != len(beta_bottom):
            raise ValueError("alpha, beta_top, and beta_bottom lengths must match")
        if self.fps <= 0 or not np.isfinite(self.fps):
            raise ValueError("fps must be positive and finite")
        if not all(np.isfinite(values).all() for values in (alpha, beta_top, beta_bottom)):
            raise ValueError("trajectory values must be finite")
        if roll_axis.shape != (3,) or not np.all(np.isfinite(roll_axis)):
            raise ValueError("roll_axis_ball must be a finite 3-vector")
        axis_norm = float(np.linalg.norm(roll_axis))
        if axis_norm <= 1e-12:
            raise ValueError("roll_axis_ball must be non-zero")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta_top", beta_top)
        object.__setattr__(self, "beta_bottom", beta_bottom)
        object.__setattr__(self, "roll_axis_ball", roll_axis / axis_norm)

    @property
    def num_frames(self) -> int:
        return len(self.alpha)

    @property
    def time_s(self) -> Array:
        return np.arange(self.num_frames, dtype=np.float64) / self.fps

    def scaled(self, factor: float) -> "Trajectory":
        if factor <= 0:
            raise ValueError("trajectory scale must be positive")
        return replace(
            self,
            alpha=self.alpha * factor,
            beta_top=self.beta_top * factor,
            beta_bottom=self.beta_bottom * factor,
            name=self.name if np.isclose(factor, 1.0) else f"{self.name}_speed_{factor:g}x",
        )


def _piecewise_linear(x: Array, knots: Sequence[tuple[float, float]]) -> Array:
    xp = np.asarray([item[0] for item in knots], dtype=np.float64)
    fp = np.asarray([item[1] for item in knots], dtype=np.float64)
    return np.interp(x, xp, fp)


def scripted_trajectory(num_frames: int = 121, fps: float = 60.0) -> Trajectory:
    """Trajectory with ramps, smooth oscillations, reversals, and independent spin."""

    if num_frames < 8:
        raise ValueError("scripted trajectory needs at least eight frames")
    phase = np.linspace(0.0, 1.0, num_frames)

    alpha_ramp = _piecewise_linear(
        phase,
        [(0.0, 0.0), (0.28, 10.0), (0.56, -8.0), (0.78, 13.0), (1.0, 0.0)],
    )
    alpha_deg = alpha_ramp + 3.0 * np.sin(4.0 * np.pi * phase)

    top_ramp = _piecewise_linear(
        phase,
        [(0.0, 0.0), (0.24, 30.0), (0.48, -18.0), (0.76, 42.0), (1.0, 16.0)],
    )
    beta_top_deg = top_ramp + 5.0 * np.sin(6.0 * np.pi * phase)

    bottom_ramp = _piecewise_linear(
        phase,
        [(0.0, 0.0), (0.20, -22.0), (0.43, 27.0), (0.70, -38.0), (1.0, 9.0)],
    )
    beta_bottom_deg = bottom_ramp + 4.0 * np.sin(4.0 * np.pi * phase + np.pi)

    return Trajectory(
        alpha=np.deg2rad(alpha_deg),
        beta_top=np.deg2rad(beta_top_deg),
        beta_bottom=np.deg2rad(beta_bottom_deg),
        fps=fps,
        name="scripted",
    )


def pure_roll_trajectory(
    num_frames: int = 61,
    fps: float = 60.0,
    total_angle_deg: float = 32.0,
) -> Trajectory:
    """Calibration motion about ball +x, shared by both hemispheres."""

    alpha = np.linspace(0.0, np.deg2rad(total_angle_deg), num_frames)
    zeros = np.zeros(num_frames, dtype=np.float64)
    return Trajectory(alpha, zeros, zeros, fps=fps, name="pure_roll")


def pure_swivel_trajectory(
    num_frames: int = 61,
    fps: float = 60.0,
    total_angle_deg: float = 42.0,
) -> Trajectory:
    """Calibration motion about ball +z with zero roll.

    Both hemispheres rotate together here so either colour population can be
    used to estimate the same swivel axis.
    """

    beta = np.linspace(0.0, np.deg2rad(total_angle_deg), num_frames)
    zeros = np.zeros(num_frames, dtype=np.float64)
    return Trajectory(zeros, beta, beta.copy(), fps=fps, name="pure_swivel")


def speed_sweep_trajectory(
    delta_deg_per_frame: float,
    num_frames: int = 41,
    fps: float = 60.0,
) -> Trajectory:
    """Pure-swivel test whose inter-frame rotation is exactly ``delta``.

    The direction reverses in blocks so high-speed sweeps do not accumulate
    many full turns.  Top and bottom use offset reversal schedules, retaining
    their independence while both exercise the requested angular step.
    """

    if delta_deg_per_frame <= 0:
        raise ValueError("delta_deg_per_frame must be positive")
    if num_frames < 4:
        raise ValueError("speed-sweep trajectory needs at least four frames")
    step = np.deg2rad(delta_deg_per_frame)
    block = max(2, (num_frames - 1) // 6)
    sample = np.arange(num_frames - 1)
    top_sign = np.where((sample // block) % 2 == 0, 1.0, -1.0)
    bottom_sign = np.where(((sample + block // 2) // block) % 2 == 0, -1.0, 1.0)
    beta_top = np.concatenate([[0.0], np.cumsum(step * top_sign)])
    beta_bottom = np.concatenate([[0.0], np.cumsum(step * bottom_sign)])
    zeros = np.zeros(num_frames, dtype=np.float64)
    return Trajectory(
        zeros,
        beta_top,
        beta_bottom,
        fps=fps,
        name=f"speed_sweep_{delta_deg_per_frame:g}deg_per_frame",
    )


@dataclass(frozen=True)
class RenderConfig:
    """Appearance, output, and repeatability settings for one sequence."""

    camera: CameraSetup = field(default_factory=default_camera)
    num_speckles: int = 1800
    dot_radius_px: float = 2.8
    seed: int = 24681357
    top_color_bgr: tuple[int, int, int] = (48, 58, 232)
    bottom_color_bgr: tuple[int, int, int] = (225, 194, 38)
    ball_color_bgr: tuple[int, int, int] = (184, 187, 192)
    background_color_bgr: tuple[int, int, int] = (36, 39, 44)
    yoke_color_bgr: tuple[int, int, int] = (18, 20, 23)
    yoke_width_px: int = 24
    draw_yoke: bool = True
    degradations: Degradations = field(default_factory=Degradations)
    frame_prefix: str = "frame_"
    image_extension: str = ".png"

    def __post_init__(self) -> None:
        if self.num_speckles < 20:
            raise ValueError("num_speckles must be at least 20")
        if self.dot_radius_px <= 0:
            raise ValueError("dot_radius_px must be positive")
        if self.yoke_width_px < 0:
            raise ValueError("yoke_width_px must be non-negative")
        if not self.frame_prefix or any(ch in self.frame_prefix for ch in "/\\"):
            raise ValueError("frame_prefix must be a non-empty filename prefix")
        ext = self.image_extension.lower()
        if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            raise ValueError(f"unsupported image extension: {self.image_extension}")
        for name in (
            "top_color_bgr",
            "bottom_color_bgr",
            "ball_color_bgr",
            "background_color_bgr",
            "yoke_color_bgr",
        ):
            colour = getattr(self, name)
            if len(colour) != 3 or any(int(channel) < 0 or int(channel) > 255 for channel in colour):
                raise ValueError(f"{name} must contain three values in [0, 255]")


@dataclass
class RenderResult:
    """Paths and in-memory metadata returned by :func:`render_sequence`."""

    output_dir: Path | None
    frame_paths: list[Path]
    ground_truth_path: Path | None
    ground_truth: dict[str, Any]
    frames: list[Array] | None = None


def _axis_rotation(axis: Array, angle: float) -> Array:
    """Active axis-angle rotation used by ideal-ball comparison cases."""

    vector = np.asarray(axis, dtype=np.float64).reshape(3).copy()
    vector /= np.linalg.norm(vector)
    x, y, z = vector
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    sine, cosine = np.sin(float(angle)), np.cos(float(angle))
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


def _motion_matrix(
    alpha: float,
    beta: float,
    roll_axis_ball: Array = np.array([1.0, 0.0, 0.0]),
) -> Array:
    return _axis_rotation(roll_axis_ball, float(alpha)) @ _rz(float(beta))


def _interpolate(values: Array, frame_position: float) -> float:
    position = float(np.clip(frame_position, 0.0, len(values) - 1.0))
    lo = int(np.floor(position))
    hi = min(lo + 1, len(values) - 1)
    fraction = position - lo
    return float((1.0 - fraction) * values[lo] + fraction * values[hi])


def _base_image(config: RenderConfig) -> Array:
    camera = config.camera
    height, width = camera.image_size
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[...] = np.asarray(config.background_color_bgr, dtype=np.uint8)
    u0, v0, r_px = camera.circle
    cv2.circle(
        image,
        (int(round(u0)), int(round(v0))),
        int(round(r_px)),
        config.ball_color_bgr,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    return image


def _speckle_attributes(config: RenderConfig) -> tuple[Array, Array, Array, Array]:
    """Return points, radii, brightness scales, and a stable density mask."""

    rng = np.random.default_rng(config.seed)
    # Normalised isotropic Gaussian samples are uniform on the unit sphere and
    # look like a genuinely random sprayed/painted speckle pattern.  The public
    # fibonacci_sphere helper remains useful for geometry tests that need even
    # deterministic coverage without sampling variance.
    points = rng.normal(size=(config.num_speckles, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    radii = config.dot_radius_px * rng.uniform(0.72, 1.28, config.num_speckles)
    brightness = rng.uniform(0.90, 1.08, config.num_speckles)
    order = rng.permutation(config.num_speckles)
    keep_n = max(20, int(round(config.num_speckles * config.degradations.density_scale)))
    density_mask = np.zeros(config.num_speckles, dtype=bool)
    density_mask[order[:keep_n]] = True
    return points, radii, brightness, density_mask


def _scaled_colour(colour: tuple[int, int, int], scale: float) -> tuple[int, int, int]:
    values = np.clip(np.rint(np.asarray(colour, dtype=np.float64) * scale), 0, 255).astype(int)
    return int(values[0]), int(values[1]), int(values[2])


def _draw_population(
    image: Array,
    points: Array,
    indices: Array,
    radii: Array,
    brightness: Array,
    R_motion: Array,
    colour: tuple[int, int, int],
    config: RenderConfig,
    rng: np.random.Generator,
) -> None:
    camera = config.camera
    if len(indices) == 0:
        return
    uv, facing = project(
        points[indices],
        R_motion,
        camera.R_bc,
        camera.C,
        camera.radius,
        camera.K,
    )
    selected = indices[facing]
    uv = uv[facing]
    if config.degradations.pixel_noise_std_px > 0.0:
        uv = uv + rng.normal(0.0, config.degradations.pixel_noise_std_px, uv.shape)

    height, width = camera.image_size
    in_image = (
        (uv[:, 0] >= -config.dot_radius_px * 2.0)
        & (uv[:, 0] < width + config.dot_radius_px * 2.0)
        & (uv[:, 1] >= -config.dot_radius_px * 2.0)
        & (uv[:, 1] < height + config.dot_radius_px * 2.0)
    )
    for point_index, point_uv in zip(selected[in_image], uv[in_image]):
        dot_radius = max(1, int(round(radii[point_index])))
        cv2.circle(
            image,
            (int(round(point_uv[0])), int(round(point_uv[1]))),
            dot_radius,
            _scaled_colour(colour, brightness[point_index]),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )


def _apply_ball_mask(image: Array, config: RenderConfig) -> None:
    """Clip antialiased dots at the known spherical silhouette."""

    height, width = config.camera.image_size
    u0, v0, r_px = config.camera.circle
    yy, xx = np.ogrid[:height, :width]
    outside = (xx - u0) ** 2 + (yy - v0) ** 2 > r_px**2
    image[outside] = np.asarray(config.background_color_bgr, dtype=np.uint8)


def _apply_glare(image: Array, config: RenderConfig) -> None:
    if not config.degradations.glare or config.degradations.glare_strength <= 0.0:
        return
    height, width = config.camera.image_size
    u0, v0, r_px = config.camera.circle
    yy, xx = np.mgrid[:height, :width]
    glare_u = u0 - 0.28 * r_px
    glare_v = v0 - 0.30 * r_px
    sigma_u = 0.18 * r_px
    sigma_v = 0.11 * r_px
    gaussian = np.exp(
        -0.5 * (((xx - glare_u) / sigma_u) ** 2 + ((yy - glare_v) / sigma_v) ** 2)
    )
    inside = (xx - u0) ** 2 + (yy - v0) ** 2 <= r_px**2
    alpha = config.degradations.glare_strength * gaussian * inside
    image_float = image.astype(np.float64)
    image_float += alpha[..., None] * (255.0 - image_float)
    image[:] = np.clip(np.rint(image_float), 0, 255).astype(np.uint8)


def _apply_yoke(image: Array, config: RenderConfig) -> tuple[int, int, int, int] | None:
    if not config.draw_yoke or config.yoke_width_px <= 0:
        return None
    u0, v0, r_px = config.camera.circle
    half_width = config.yoke_width_px / 2.0
    x0 = int(round(u0 - half_width))
    x1 = int(round(u0 + half_width))
    y0 = int(round(v0 - 1.08 * r_px))
    y1 = int(round(v0 + 1.08 * r_px))
    cv2.rectangle(image, (x0, y0), (x1, y1), config.yoke_color_bgr, thickness=-1)
    return x0, y0, x1, y1


def _render_subframe(
    frame_position: float,
    trajectory: Trajectory,
    config: RenderConfig,
    points: Array,
    radii: Array,
    brightness: Array,
    top_indices: Array,
    bottom_indices: Array,
    rng: np.random.Generator,
) -> Array:
    image = _base_image(config)
    alpha = _interpolate(trajectory.alpha, frame_position)
    beta_top = _interpolate(trajectory.beta_top, frame_position)
    beta_bottom = _interpolate(trajectory.beta_bottom, frame_position)
    _draw_population(
        image,
        points,
        top_indices,
        radii,
        brightness,
        _motion_matrix(alpha, beta_top, trajectory.roll_axis_ball),
        config.top_color_bgr,
        config,
        rng,
    )
    _draw_population(
        image,
        points,
        bottom_indices,
        radii,
        brightness,
        _motion_matrix(alpha, beta_bottom, trajectory.roll_axis_ball),
        config.bottom_color_bgr,
        config,
        rng,
    )
    _apply_ball_mask(image, config)
    return image


def _render_frame(
    frame_index: int,
    trajectory: Trajectory,
    config: RenderConfig,
    points: Array,
    radii: Array,
    brightness: Array,
    top_indices: Array,
    bottom_indices: Array,
) -> Array:
    degradation = config.degradations
    samples = degradation.motion_blur_samples
    if samples == 1 or degradation.motion_blur_fraction == 0.0:
        offsets = np.array([0.0])
    else:
        half_exposure = 0.5 * degradation.motion_blur_fraction
        offsets = np.linspace(-half_exposure, half_exposure, samples)

    accumulated = np.zeros((*config.camera.image_size, 3), dtype=np.float64)
    # A frame-local RNG makes output independent of whether callers render all
    # frames at once or later add random-access rendering.
    for sample_index, offset in enumerate(offsets):
        sample_seed = np.random.SeedSequence([config.seed, frame_index, sample_index, 9157])
        rng = np.random.default_rng(sample_seed)
        accumulated += _render_subframe(
            frame_index + float(offset),
            trajectory,
            config,
            points,
            radii,
            brightness,
            top_indices,
            bottom_indices,
            rng,
        ).astype(np.float64)
    image = np.clip(np.rint(accumulated / len(offsets)), 0, 255).astype(np.uint8)
    _apply_glare(image, config)
    _apply_yoke(image, config)
    return image


def _matrix_series(trajectory: Trajectory) -> tuple[list[Array], list[Array]]:
    top = [
        _motion_matrix(alpha, beta, trajectory.roll_axis_ball)
        for alpha, beta in zip(trajectory.alpha, trajectory.beta_top)
    ]
    bottom = [
        _motion_matrix(alpha, beta, trajectory.roll_axis_ball)
        for alpha, beta in zip(trajectory.alpha, trajectory.beta_bottom)
    ]
    return top, bottom


def _increments(absolute: Sequence[Array]) -> list[Array]:
    identity = np.eye(3)
    return [identity] + [current @ previous.T for previous, current in zip(absolute[:-1], absolute[1:])]


def _camera_matrices(matrices_ball: Sequence[Array], R_bc: Array) -> list[Array]:
    return [R_bc @ matrix @ R_bc.T for matrix in matrices_ball]


def _to_lists(values: Iterable[Array]) -> list[Any]:
    return [np.asarray(value).tolist() for value in values]


def _rotation_angles(matrices: Sequence[Array]) -> Array:
    cosines = np.array([(np.trace(matrix) - 1.0) / 2.0 for matrix in matrices])
    return np.arccos(np.clip(cosines, -1.0, 1.0))


def _ground_truth(
    trajectory: Trajectory,
    config: RenderConfig,
    frame_names: list[str],
    points: Array,
    retained_mask: Array,
) -> dict[str, Any]:
    camera = config.camera
    top_ball, bottom_ball = _matrix_series(trajectory)
    top_cam = _camera_matrices(top_ball, camera.R_bc)
    bottom_cam = _camera_matrices(bottom_ball, camera.R_bc)
    top_inc_ball = _increments(top_ball)
    bottom_inc_ball = _increments(bottom_ball)
    top_inc_cam = _increments(top_cam)
    bottom_inc_cam = _increments(bottom_cam)
    yoke_box = None
    if config.draw_yoke and config.yoke_width_px > 0:
        u0, v0, r_px = camera.circle
        half = config.yoke_width_px / 2.0
        yoke_box = [u0 - half, v0 - 1.08 * r_px, u0 + half, v0 + 1.08 * r_px]

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "generator": "synthetic.generate_ball",
        "sequence_name": trajectory.name,
        "angle_units": "radians",
        "fps": trajectory.fps,
        "num_frames": trajectory.num_frames,
        "image_size": {"height": camera.image_size[0], "width": camera.image_size[1]},
        "frame_files": frame_names,
        # Frequently consumed quantities are duplicated at the top level to
        # keep validators simple; camera also groups them semantically.
        "K": camera.K.tolist(),
        "dist": camera.dist.tolist(),
        "R_bc": camera.R_bc.tolist(),
        "C": camera.C.tolist(),
        "radius": camera.radius,
        "circle": {"u0": camera.circle[0], "v0": camera.circle[1], "r_px": camera.circle[2]},
        "camera": {
            "K": camera.K.tolist(),
            "dist": camera.dist.tolist(),
            "R_bc": camera.R_bc.tolist(),
            "C": camera.C.tolist(),
            "radius": camera.radius,
            "circle": {
                "u0": camera.circle[0],
                "v0": camera.circle[1],
                "r_px": camera.circle[2],
            },
        },
        "alpha": trajectory.alpha.tolist(),
        "beta_top": trajectory.beta_top.tolist(),
        "beta_bottom": trajectory.beta_bottom.tolist(),
        "roll_axis_ball": trajectory.roll_axis_ball.tolist(),
        "R_top": _to_lists(top_ball),
        "R_bottom": _to_lists(bottom_ball),
        "R_top_ball": _to_lists(top_ball),
        "R_bottom_ball": _to_lists(bottom_ball),
        "R_top_camera": _to_lists(top_cam),
        "R_bottom_camera": _to_lists(bottom_cam),
        "R_top_increment_ball": _to_lists(top_inc_ball),
        "R_bottom_increment_ball": _to_lists(bottom_inc_ball),
        "R_top_increment_camera": _to_lists(top_inc_cam),
        "R_bottom_increment_camera": _to_lists(bottom_inc_cam),
        "appearance": {
            "top_color_bgr": list(config.top_color_bgr),
            "bottom_color_bgr": list(config.bottom_color_bgr),
            "ball_color_bgr": list(config.ball_color_bgr),
            "background_color_bgr": list(config.background_color_bgr),
            "yoke_color_bgr": list(config.yoke_color_bgr),
            "dot_radius_px": config.dot_radius_px,
            "num_speckles_material": config.num_speckles,
            "num_speckles_retained": int(retained_mask.sum()),
            "num_top_retained": int((retained_mask & (points[:, 2] > 0.0)).sum()),
            "num_bottom_retained": int((retained_mask & (points[:, 2] < 0.0)).sum()),
            "yoke_bbox_xyxy": yoke_box,
        },
        "degradations": {
            "pixel_noise_std_px": config.degradations.pixel_noise_std_px,
            "motion_blur_samples": config.degradations.motion_blur_samples,
            "motion_blur_fraction": config.degradations.motion_blur_fraction,
            "glare": config.degradations.glare,
            "glare_strength": config.degradations.glare_strength,
            "density_scale": config.degradations.density_scale,
            "speed_scale": config.degradations.speed_scale,
        },
        "motion_summary": {
            "max_top_increment_deg": float(np.rad2deg(_rotation_angles(top_inc_ball)).max()),
            "max_bottom_increment_deg": float(
                np.rad2deg(_rotation_angles(bottom_inc_ball)).max()
            ),
            "median_top_increment_deg": float(
                np.rad2deg(np.median(_rotation_angles(top_inc_ball[1:])))
            ),
            "median_bottom_increment_deg": float(
                np.rad2deg(np.median(_rotation_angles(bottom_inc_ball[1:])))
            ),
        },
        "seed": config.seed,
        "frames": [],
    }
    for index in range(trajectory.num_frames):
        metadata["frames"].append(
            {
                "index": index,
                "time_s": index / trajectory.fps,
                "file": frame_names[index],
                "alpha": float(trajectory.alpha[index]),
                "beta_top": float(trajectory.beta_top[index]),
                "beta_bottom": float(trajectory.beta_bottom[index]),
                "R_top": top_ball[index].tolist(),
                "R_bottom": bottom_ball[index].tolist(),
                "R_top_camera": top_cam[index].tolist(),
                "R_bottom_camera": bottom_cam[index].tolist(),
            }
        )
    return metadata


def render_sequence(
    output_dir: str | Path | None,
    trajectory: Trajectory,
    config: RenderConfig | None = None,
    *,
    return_frames: bool = False,
) -> RenderResult:
    """Render one sequence and optionally save PNGs plus ground-truth JSON.

    Passing ``output_dir=None`` performs a memory-only render; in that case
    frames are returned even if ``return_frames`` is false.  Existing files
    with this renderer's exact frame prefix are replaced so repeat runs cannot
    accidentally leave stale trailing frames.
    """

    config = config or RenderConfig()
    trajectory = trajectory.scaled(config.degradations.speed_scale)
    output_path = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    frames_dir: Path | None = None
    ground_truth_path: Path | None = None
    if output_path is not None:
        frames_dir = output_path / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        pattern = f"{config.frame_prefix}*{config.image_extension}"
        for stale_frame in frames_dir.glob(pattern):
            if stale_frame.is_file():
                stale_frame.unlink()
        ground_truth_path = output_path / "ground_truth.json"

    points, radii, brightness, retained = _speckle_attributes(config)
    top_indices = np.flatnonzero((points[:, 2] > 0.0) & retained)
    bottom_indices = np.flatnonzero((points[:, 2] < 0.0) & retained)
    if min(len(top_indices), len(bottom_indices)) < 10:
        raise ValueError("density leaves too few speckles in one hemisphere")

    keep_frames = return_frames or output_path is None
    memory_frames: list[Array] | None = [] if keep_frames else None
    frame_paths: list[Path] = []
    frame_names: list[str] = []
    digits = max(6, len(str(trajectory.num_frames - 1)))
    for frame_index in range(trajectory.num_frames):
        image = _render_frame(
            frame_index,
            trajectory,
            config,
            points,
            radii,
            brightness,
            top_indices,
            bottom_indices,
        )
        filename = f"{config.frame_prefix}{frame_index:0{digits}d}{config.image_extension}"
        relative_name = str(Path("frames") / filename).replace("\\", "/")
        frame_names.append(relative_name)
        if frames_dir is not None:
            frame_path = frames_dir / filename
            if not cv2.imwrite(str(frame_path), image):
                raise OSError(f"failed to write synthetic frame: {frame_path}")
            frame_paths.append(frame_path)
        if memory_frames is not None:
            memory_frames.append(image)

    truth = _ground_truth(trajectory, config, frame_names, points, retained)
    if ground_truth_path is not None:
        with ground_truth_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(truth, handle, indent=2, allow_nan=False)
            handle.write("\n")
    return RenderResult(output_path, frame_paths, ground_truth_path, truth, memory_frames)


def generate_sequence(
    output_dir: str | Path | None,
    *,
    num_frames: int = 121,
    fps: float = 60.0,
    seed: int = 24681357,
    noise_px: float = 0.0,
    motion_blur: bool | int = False,
    glare: bool = False,
    density_scale: float = 1.0,
    speed_scale: float = 1.0,
    camera: CameraSetup | None = None,
    return_frames: bool = False,
) -> RenderResult:
    """Convenience wrapper for the standard scripted validation clip.

    ``motion_blur=True`` uses five exposure samples; passing an integer selects
    an exact sample count.  The explicit keyword names mirror Stage-D sweeps.
    """

    if isinstance(motion_blur, bool):
        blur_samples = 5 if motion_blur else 1
    else:
        blur_samples = int(motion_blur)
    degradation = Degradations(
        pixel_noise_std_px=noise_px,
        motion_blur_samples=blur_samples,
        glare=glare,
        density_scale=density_scale,
        speed_scale=speed_scale,
    )
    config = RenderConfig(
        camera=camera or default_camera(),
        seed=seed,
        degradations=degradation,
    )
    trajectory = scripted_trajectory(num_frames=num_frames, fps=fps)
    return render_sequence(output_dir, trajectory, config, return_frames=return_frames)


def generate_robustness_case(
    output_dir: str | Path | None,
    *,
    num_frames: int = 121,
    fps: float = 60.0,
    seed: int = 24681357,
    noise_px: float = 0.0,
    motion_blur: bool | int = False,
    glare: bool = False,
    density_scale: float = 1.0,
    speed_scale: float = 1.0,
    delta_deg_per_frame: float | None = None,
    camera: CameraSetup | None = None,
    return_frames: bool = False,
) -> RenderResult:
    """Render one Stage-D case with all degradations named as sweep axes.

    If ``delta_deg_per_frame`` is supplied, an exact-step swivel trajectory is
    used (and ``speed_scale`` may still multiply it).  Otherwise this uses the
    standard mixed scripted motion.
    """

    if isinstance(motion_blur, bool):
        blur_samples = 5 if motion_blur else 1
    else:
        blur_samples = int(motion_blur)
    degradation = Degradations(
        pixel_noise_std_px=noise_px,
        motion_blur_samples=blur_samples,
        glare=glare,
        density_scale=density_scale,
        speed_scale=speed_scale,
    )
    config = RenderConfig(
        camera=camera or default_camera(),
        seed=seed,
        degradations=degradation,
    )
    trajectory = (
        speed_sweep_trajectory(delta_deg_per_frame, num_frames=num_frames, fps=fps)
        if delta_deg_per_frame is not None
        else scripted_trajectory(num_frames=num_frames, fps=fps)
    )
    return render_sequence(output_dir, trajectory, config, return_frames=return_frames)


def generate_calibration_sequences(
    output_root: str | Path,
    *,
    num_frames: int = 61,
    fps: float = 60.0,
    seed: int = 24681357,
    camera: CameraSetup | None = None,
) -> dict[str, RenderResult]:
    """Render isolated pure-roll and pure-swivel axis-calibration clips."""

    root = Path(output_root).expanduser().resolve()
    config = RenderConfig(camera=camera or default_camera(), seed=seed)
    roll = render_sequence(root / "pure_roll", pure_roll_trajectory(num_frames, fps), config)
    swivel = render_sequence(
        root / "pure_swivel",
        pure_swivel_trajectory(num_frames, fps),
        config,
    )
    return {"pure_roll": roll, "pure_swivel": swivel}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="sequence output directory")
    parser.add_argument(
        "--mode",
        choices=("scripted", "pure-roll", "pure-swivel", "speed-sweep", "calibration-suite"),
        default="scripted",
    )
    parser.add_argument("--frames", type=int, default=None, help="number of frames")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=24681357)
    parser.add_argument("--noise-px", type=float, default=0.0)
    parser.add_argument(
        "--motion-blur-samples",
        type=int,
        default=1,
        help="1 disables blur; 5 is a useful Stage-D test",
    )
    parser.add_argument("--glare", action="store_true")
    parser.add_argument("--density-scale", type=float, default=1.0)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument(
        "--delta-deg-per-frame",
        type=float,
        default=5.0,
        help="exact angular step used by --mode speed-sweep",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mode == "calibration-suite":
        count = args.frames if args.frames is not None else 61
        results = generate_calibration_sequences(
            args.output,
            num_frames=count,
            fps=args.fps,
            seed=args.seed,
        )
        summary = {
            name: {
                "frames": len(result.frame_paths),
                "ground_truth": str(result.ground_truth_path),
            }
            for name, result in results.items()
        }
    else:
        if args.frames is not None:
            count = args.frames
        elif args.mode == "scripted":
            count = 121
        elif args.mode == "speed-sweep":
            count = 41
        else:
            count = 61
        degradation = Degradations(
            pixel_noise_std_px=args.noise_px,
            motion_blur_samples=args.motion_blur_samples,
            glare=args.glare,
            density_scale=args.density_scale,
            speed_scale=args.speed_scale,
        )
        config = RenderConfig(seed=args.seed, degradations=degradation)
        if args.mode == "scripted":
            trajectory = scripted_trajectory(count, args.fps)
        elif args.mode == "pure-roll":
            trajectory = pure_roll_trajectory(count, args.fps)
        elif args.mode == "speed-sweep":
            trajectory = speed_sweep_trajectory(args.delta_deg_per_frame, count, args.fps)
        else:
            trajectory = pure_swivel_trajectory(count, args.fps)
        result = render_sequence(args.output, trajectory, config)
        summary = {
            "sequence": trajectory.name,
            "frames": len(result.frame_paths),
            "ground_truth": str(result.ground_truth_path),
            "frames_dir": str(result.output_dir / "frames") if result.output_dir else None,
        }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
