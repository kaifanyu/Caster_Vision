"""Manifest-driven, device-independent caster comparison report.

The preferred input is a precomputed ``CasterFrame`` JSON stream.  That path
keeps device-specific tracking outside this module and guarantees that ball
and swivel runs pass through the exact same :func:`metrics.compute_metrics`
call.  Raw clips are also accepted: imports of device pipelines and adapters
are intentionally deferred until such a row is encountered.

Paths inside the YAML manifest are resolved relative to the manifest itself.
An example manifest is::

    output_dir: out/comparison
    metrics:
      lag_max_s: 1.0
    runs:
      - maneuver: straight
        device: ball
        caster_frames: data/ball_straight.json
        car_track: data/ball_straight_track.csv
      - maneuver: straight
        device: swivel
        caster_frames: data/swivel_straight.json
        car_track: data/swivel_straight_track.csv

Each run may instead provide ``clip`` plus a project configuration using the
``config`` key.  Precomputed streams remain the stable interchange format for
repeatable comparisons and CI.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import inspect
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:  # Support ``python compare/run_comparison.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from common.caster_frame import CasterFrame, load_caster_frames
from common.kinematics import CarTrack, load_car_track
from metrics.metrics import MetricConfig, MetricResult, compute_metrics


_SUMMARY_METRICS = (
    "mean_abs_alignment_deg",
    "total_scrub_m",
    "scrub_fraction",
    "rolling_efficiency",
    "mean_slip",
    "mean_abs_slip",
    "swivel_lag_s",
    "settling_time_s",
    "shimmy_frequency_hz",
    "shimmy_amplitude_deg",
    "alignment_valid_duration_s",
    "scrub_valid_duration_s",
    "slip_valid_duration_s",
)


@dataclass(frozen=True)
class ComparisonReport:
    """Locations and serialization-ready rows produced by one harness run."""

    output_dir: Path
    summary_rows: tuple[dict[str, Any], ...]
    plot_paths: dict[str, Path]
    run_directories: dict[str, Path]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "summary_rows": [dict(row) for row in self.summary_rows],
            "plot_paths": {key: str(value) for key, value in self.plot_paths.items()},
            "run_directories": {
                key: str(value) for key, value in self.run_directories.items()
            },
        }


@dataclass(frozen=True)
class _ScoredRun:
    run_id: str
    maneuver: str
    device: str
    source_kind: str
    caster_source: str
    car_track_source: str
    result: MetricResult
    run_dir: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return dict(value)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(output.get(key), Mapping):
            output[key] = _deep_merge(output[key], value)
        else:
            output[key] = copy.deepcopy(value)
    return output


def _path(base: Path, value: str | Path, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty path")
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.strip().casefold()).strip("-")
    return result or "run"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, allow_nan=False), encoding="utf-8"
    )
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    return value


def _write_series_csv(path: Path, result: MetricResult) -> Path:
    series = result.series
    columns: dict[str, np.ndarray] = {
        "t_s": series.t,
        "rolling_heading_x": series.rolling_heading_car[:, 0],
        "rolling_heading_y": series.rolling_heading_car[:, 1],
        "heading_angle_rad": series.heading_angle_rad,
        "demand_heading_angle_rad": series.demand_heading_angle_rad,
        "contact_velocity_x_mps": series.contact_velocity_car[:, 0],
        "contact_velocity_y_mps": series.contact_velocity_car[:, 1],
        "contact_speed_mps": series.contact_speed_mps,
        "v_long_mps": series.v_long_mps,
        "v_lat_x_mps": series.v_lat_car_mps[:, 0],
        "v_lat_y_mps": series.v_lat_car_mps[:, 1],
        "scrub_speed_mps": series.scrub_speed_mps,
        "v_roll_mps": series.v_roll_mps,
        "alignment_error_rad": series.alignment_error_rad,
        "slip_ratio": series.slip_ratio,
        "confidence": series.confidence,
        "track_valid": series.track_valid,
        "heading_valid": series.heading_valid,
        "roll_valid": series.roll_valid,
        "alignment_valid": series.alignment_valid,
        "slip_valid": series.slip_valid,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index in range(len(series.t)):
            writer.writerow([_csv_value(values[index]) for values in columns.values()])
    return path


def _config_for_run(
    manifest: Mapping[str, Any], row: Mapping[str, Any], manifest_dir: Path
) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for value in (manifest.get("config"), row.get("config")):
        if value is None:
            continue
        if isinstance(value, Mapping):
            loaded = dict(value)
        else:
            loaded = _load_yaml(_path(manifest_dir, value, "config"))
        combined = _deep_merge(combined, loaded)
    return combined


def _car_track_spec(
    manifest: Mapping[str, Any], row: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    value = row.get("car_track", manifest.get("car_track"))
    if value is None:
        raise ValueError("each run needs car_track (or a top-level default)")
    options: dict[str, Any] = {}
    if isinstance(manifest.get("car_track_options"), Mapping):
        options.update(manifest["car_track_options"])
    if isinstance(value, Mapping):
        if "path" not in value:
            raise ValueError("a car_track mapping needs a path")
        options.update({key: item for key, item in value.items() if key != "path"})
        value = value["path"]
    if isinstance(row.get("car_track_options"), Mapping):
        options.update(row["car_track_options"])
    return value, options


def _load_track_for_run(
    manifest: Mapping[str, Any],
    row: Mapping[str, Any],
    project_config: Mapping[str, Any],
    manifest_dir: Path,
) -> tuple[CarTrack, Path]:
    value, manifest_options = _car_track_spec(manifest, row)
    track_path = _path(manifest_dir, value, "car_track")
    project_options = project_config.get("car_track", {})
    if not isinstance(project_options, Mapping):
        project_options = {}
    # Project config's car_track.time_offset_s is defined by calibration as
    # an offset added to *caster* timestamps by each device pipeline.  It must
    # not also shift the car track here.  A manifest may still request an
    # explicit track offset through car_track_options.time_offset_s.
    options = {
        key: item
        for key, item in project_options.items()
        if key != "time_offset_s"
    }
    options.update(manifest_options)
    return (
        load_car_track(
            track_path,
            time_offset_s=float(options.get("time_offset_s", 0.0)),
            smooth_window=int(
                options.get("smooth_window", options.get("savgol_window", 9))
            ),
            polyorder=int(options.get("polyorder", options.get("savgol_polyorder", 2))),
            theta_unit=str(options.get("theta_unit", "rad")),
        ),
        track_path,
    )


def _offset_frames(frames: Sequence[CasterFrame], offset_s: float) -> list[CasterFrame]:
    if not np.isfinite(offset_s):
        raise ValueError("caster_time_offset_s must be finite")
    if offset_s == 0.0:
        return list(frames)
    return [
        CasterFrame(
            t=frame.t + offset_s,
            roll_axis_car=frame.roll_axis_car,
            omega_roll=frame.omega_roll,
            omega_spin=frame.omega_spin,
            r_eff=frame.r_eff,
            raw=dict(frame.raw),
        )
        for frame in frames
    ]


def _camera_setup(config: Mapping[str, Any], image_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    from common.config import camera_matrix, distortion_coefficients

    camera = config.get("camera", {})
    if not isinstance(camera, Mapping):
        raise ValueError("config.camera must be a mapping")
    K, _ = camera_matrix(camera, image_shape)
    return K, distortion_coefficients(camera)


def _raw_ball_frames(
    clip_path: Path,
    config: Mapping[str, Any],
    run_dir: Path,
) -> list[CasterFrame]:
    """Run the present Approach-A pipeline without importing it at startup."""

    from ball.adapter import adapt_ball_pipeline
    from ball.pipeline import run_pipeline
    from common.config import configured_circle
    from common.io_frames import open_frame_source

    input_config = config.get("input", {})
    ball_config = config.get("ball", {})
    if not isinstance(input_config, Mapping) or not isinstance(ball_config, Mapping):
        raise ValueError("config.input and config.ball must be mappings")
    camera_config = config.get("camera", {})
    if not isinstance(camera_config, Mapping) or camera_config.get("K") is None:
        raise ValueError("raw ball comparison requires calibrated camera.K")
    fps_override = input_config.get("fps_override")
    image_fps = None if fps_override is None else float(fps_override)
    source = open_frame_source(
        clip_path,
        input_type=str(input_config.get("type", "auto")),
        max_frames=input_config.get("max_frames"),
        image_fps=image_fps,
    )
    fps = float(image_fps or source.fps or 0.0)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("raw ball comparison has no reliable FPS; set input.fps_override")
    width, height = source.size
    K, dist = _camera_setup(config, (height, width, 3))
    circle_config = ball_config.get("circle", config.get("circle", {}))
    circle = configured_circle(circle_config if isinstance(circle_config, Mapping) else {})
    R_bc_value = ball_config.get("R_bc")
    if R_bc_value is None and isinstance(config.get("frame_calib"), Mapping):
        R_bc_value = config["frame_calib"].get("R_bc")
    if R_bc_value is None:
        raise ValueError("raw ball comparison requires ball.R_bc in the run config")
    R_bc = np.asarray(R_bc_value, dtype=float)
    R_ball_to_car_value = ball_config.get("R_ball_to_car")
    if R_ball_to_car_value is None:
        raise ValueError("raw ball comparison requires calibrated ball.R_ball_to_car")
    if not bool(ball_config.get("direction_sign_calibrated", False)):
        raise ValueError("raw ball comparison requires calibrated ball direction sign")
    overlay = None
    output_config = config.get("output", {})
    if isinstance(output_config, Mapping) and bool(output_config.get("save_overlay_video", False)):
        overlay = run_dir / "tracking_overlay.mp4"
    timestamped_source = (
        (record.image, record.index / fps, record.source) for record in source
    )
    pipeline_result = run_pipeline(
        timestamped_source,
        K=K,
        dist=dist,
        circle=circle,
        R_bc=R_bc,
        segment_config=ball_config.get("segment", config.get("segment", {})),
        track_config=config.get("track", {}),
        estimate_config=config.get("estimate", {}),
        fps=fps,
        overlay_path=overlay,
    )
    radius = ball_config.get("r_eff_m", ball_config.get("r_eff"))
    if radius is None:
        raise ValueError("raw ball comparison requires calibrated ball.r_eff_m")
    frames = adapt_ball_pipeline(
        pipeline_result,
        R_bc=R_bc,
        R_ball_to_car=np.asarray(R_ball_to_car_value, dtype=float),
        r_eff=float(radius),
        roll_direction_sign=float(ball_config.get("roll_direction_sign", -1.0)),
    )
    track_config = config.get("car_track", {})
    caster_offset = (
        float(track_config.get("time_offset_s", 0.0))
        if isinstance(track_config, Mapping)
        else 0.0
    )
    return _offset_frames(frames, caster_offset)


def _invoke_lazy_swivel_pipeline(
    clip_path: Path,
    config: Mapping[str, Any],
    run_dir: Path,
) -> Any:
    """Invoke a high-level swivel clip function when the optional module exists.

    Accepted entry-point names are ``run_clip``, ``run_swivel_pipeline``, and
    ``run_pipeline``.  Keyword injection is deliberately signature based so a
    missing high-level API gives a useful error instead of a cryptic call deep
    in OpenCV.
    """

    try:
        module = importlib.import_module("swivel.pipeline")
    except ModuleNotFoundError as exc:
        if exc.name == "swivel.pipeline":
            raise RuntimeError(
                "raw swivel rows require swivel.pipeline; precompute a "
                "CasterFrame JSON with scripts/run_swivel.py in this build"
            ) from exc
        raise
    candidates = {
        "clip": clip_path,
        "clip_path": clip_path,
        "path": clip_path,
        "input_path": clip_path,
        "config": dict(config),
        "output_dir": run_dir,
    }
    failures: list[str] = []
    for name in ("run_clip", "run_swivel_pipeline", "run_pipeline"):
        function = getattr(module, name, None)
        if not callable(function):
            continue
        signature = inspect.signature(function)
        kwargs = {key: value for key, value in candidates.items() if key in signature.parameters}
        missing = [
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and parameter.name not in kwargs
        ]
        if missing:
            failures.append(f"{name} requires unsupported arguments {missing}")
            continue
        return function(**kwargs)
    detail = "; ".join(failures) or "no recognized high-level entry point"
    raise RuntimeError(f"cannot dispatch raw swivel clip through swivel.pipeline: {detail}")


def _raw_swivel_frames(
    clip_path: Path,
    config: Mapping[str, Any],
    run_dir: Path,
) -> list[CasterFrame]:
    from swivel.adapter import SwivelSeries, adapt_swivel_series

    output = _invoke_lazy_swivel_pipeline(clip_path, config, run_dir)
    if isinstance(output, Sequence) and all(isinstance(item, CasterFrame) for item in output):
        return list(output)
    candidate = getattr(output, "caster_frames", None)
    if candidate is not None and isinstance(candidate, Sequence) and all(
        isinstance(item, CasterFrame) for item in candidate
    ):
        return list(candidate)
    series = output if isinstance(output, SwivelSeries) else getattr(output, "series", None)
    if not isinstance(series, SwivelSeries):
        raise TypeError(
            "swivel pipeline must return CasterFrames, SwivelSeries, or an object with .series"
        )
    swivel_config = config.get("swivel", {})
    geometry = swivel_config.get("geometry", {}) if isinstance(swivel_config, Mapping) else {}
    if not isinstance(swivel_config, Mapping) or not isinstance(geometry, Mapping):
        raise ValueError("config.swivel and swivel.geometry must be mappings")
    radius = swivel_config.get("r_eff_m", swivel_config.get("r_eff"))
    if radius is None:
        raise ValueError("raw swivel comparison requires calibrated swivel.r_eff_m")
    kwargs: dict[str, Any] = {
        "r_eff": float(radius),
        "axle0_car": np.asarray(
            geometry.get("axle0_car", geometry.get("axle_zero_car", [0.0, -1.0, 0.0])),
            dtype=float,
        ),
        "roll_direction_sign": float(swivel_config.get("roll_direction_sign", 1.0)),
        "heading_direction_sign": float(swivel_config.get("heading_direction_sign", 1.0)),
    }
    if "swivel_axis_car" in geometry and (
        "hub_offset0_car" in geometry or "hub_offset_zero_car" in geometry
    ):
        kwargs["swivel_axis_car"] = np.asarray(geometry["swivel_axis_car"], dtype=float)
        kwargs["hub_offset0_car"] = np.asarray(
            geometry.get("hub_offset0_car", geometry.get("hub_offset_zero_car")), dtype=float
        )
    return adapt_swivel_series(series, **kwargs)


def _caster_frames_for_run(
    row: Mapping[str, Any],
    device: str,
    project_config: Mapping[str, Any],
    manifest_dir: Path,
    run_dir: Path,
) -> tuple[list[CasterFrame], str, str]:
    precomputed = row.get("caster_frames", row.get("frames"))
    clip = row.get("clip")
    if precomputed is not None and clip is not None:
        raise ValueError("a run must provide caster_frames or clip, not both")
    if precomputed is not None:
        source = _path(manifest_dir, precomputed, "caster_frames")
        frames, metadata = load_caster_frames(source)
        if bool(metadata.get("tracking_only", False)):
            raise ValueError(
                f"precomputed stream is tracking-only and cannot be compared: {source}"
            )
        false_flags = (
            "trusted_measurement",
            "K_approximated",
            "R_bc_calibrated",
            "R_ball_to_car_calibrated",
            "psi0_calibrated",
            "geometry_calibrated",
            "direction_sign_calibrated",
            "r_eff_calibrated",
        )
        for key in false_flags:
            if key not in metadata:
                continue
            rejected = bool(metadata[key]) if key == "K_approximated" else not bool(metadata[key])
            if rejected:
                raise ValueError(
                    f"precomputed stream metadata rejects comparison ({key}={metadata[key]!r}): {source}"
                )
        if metadata.get("clocks_synchronized") is False and not bool(
            row.get("clocks_synchronized", False)
        ):
            raise ValueError(
                "precomputed stream says clocks are unsynchronized; calibrate sync or "
                "set clocks_synchronized: true on the row after external alignment"
            )
        source_kind = "precomputed"
    elif clip is not None:
        assumptions = project_config.get("assumptions", {})
        if not isinstance(assumptions, Mapping) or not bool(
            assumptions.get("camera_fixed_to_chassis", False)
        ):
            raise ValueError(
                "raw-clip comparison requires assumptions.camera_fixed_to_chassis=true"
            )
        if device == "ball" and not bool(
            assumptions.get("ball_center_stationary_in_image", False)
        ):
            raise ValueError(
                "raw ball comparison requires "
                "assumptions.ball_center_stationary_in_image=true"
            )
        clocks_confirmed = bool(
            assumptions.get("car_and_caster_clocks_synchronized", False)
        ) or bool(row.get("clocks_synchronized", False))
        if not clocks_confirmed:
            raise ValueError(
                "raw-clip comparison requires confirmed clock synchronization: set "
                "assumptions.car_and_caster_clocks_synchronized=true after calibration "
                "or clocks_synchronized: true on the manifest row"
            )
        source = _path(manifest_dir, clip, "clip")
        frames = (
            _raw_ball_frames(source, project_config, run_dir)
            if device == "ball"
            else _raw_swivel_frames(source, project_config, run_dir)
        )
        source_kind = "raw_clip"
    else:
        raise ValueError("each run needs caster_frames (preferred) or clip")
    if len(frames) < 2:
        raise ValueError(f"run {row.get('maneuver')!r}/{device} produced fewer than two frames")
    offset = float(row.get("caster_time_offset_s", 0.0))
    return _offset_frames(frames, offset), source_kind, str(source)


def _metric_values(
    manifest: Mapping[str, Any],
    row: Mapping[str, Any],
    project_config: Mapping[str, Any],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for candidate in (
        project_config.get("metrics"),
        manifest.get("metrics"),
        row.get("metrics"),
    ):
        if isinstance(candidate, Mapping):
            values.update(candidate)
    aliases = {
        "speed_epsilon_mps": "min_contact_speed_mps",
        "slip_speed_epsilon_mps": "min_longitudinal_speed_mps",
        "max_lag_s": "lag_max_s",
        "shimmy_min_hz": "shimmy_min_frequency_hz",
        "shimmy_max_hz": "shimmy_max_frequency_hz",
    }
    for old, new in aliases.items():
        if old in values and new not in values:
            values[new] = values[old]
    return values


def _arm_for_run(
    manifest: Mapping[str, Any],
    row: Mapping[str, Any],
    project_config: Mapping[str, Any],
) -> np.ndarray:
    project_track = project_config.get("car_track", {})
    candidates = (
        row.get("r_arm_car"),
        row.get("r_arm_car_m"),
        manifest.get("r_arm_car"),
        manifest.get("r_arm_car_m"),
        project_track.get("r_arm_car_m") if isinstance(project_track, Mapping) else None,
    )
    value = next((item for item in candidates if item is not None), [0.0, 0.0])
    arm = np.asarray(value, dtype=float).reshape(-1)
    if arm.shape != (2,) or not np.all(np.isfinite(arm)):
        raise ValueError("r_arm_car must contain two finite values")
    return arm


def _steady_mask(
    timestamps: np.ndarray,
    row: Mapping[str, Any],
    metric_values: Mapping[str, Any],
) -> np.ndarray | None:
    segment = row.get("steady_segment_s", metric_values.get("shimmy_segment_s"))
    if segment is None:
        return None
    values = np.asarray(segment, dtype=float).reshape(-1)
    if values.shape != (2,) or not np.all(np.isfinite(values)) or values[1] <= values[0]:
        raise ValueError("steady_segment_s/shimmy_segment_s must be [start, end]")
    return (timestamps >= values[0]) & (timestamps <= values[1])


def _write_run_artifacts(scored: _ScoredRun) -> None:
    scored.run_dir.mkdir(parents=True, exist_ok=True)
    _write_series_csv(scored.run_dir / "series.csv", scored.result)
    _write_json(
        scored.run_dir / "series.json",
        {
            "schema": "caster-metric-series-v1",
            "run": {
                "run_id": scored.run_id,
                "maneuver": scored.maneuver,
                "device": scored.device,
                "source_kind": scored.source_kind,
                "caster_source": scored.caster_source,
                "car_track": scored.car_track_source,
            },
            "series": scored.result.series.to_dict(),
        },
    )
    _write_json(scored.run_dir / "summary.json", scored.result.summary.to_dict())


def _plot_maneuver(path: Path, maneuver: str, runs: Sequence[_ScoredRun]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    for run in runs:
        series = run.result.series
        elapsed = series.t - series.t[0]
        label = run.device if sum(item.device == run.device for item in runs) == 1 else run.run_id
        axes[0].plot(elapsed, np.rad2deg(series.alignment_error_rad), label=label)
        axes[1].plot(elapsed, series.scrub_speed_mps, label=label)
        axes[2].plot(elapsed, series.slip_ratio, label=label)
    axes[0].set_ylabel("alignment error (deg)")
    axes[1].set_ylabel("scrub speed (m/s)")
    axes[2].set_ylabel("longitudinal slip")
    axes[2].set_xlabel("elapsed time (s)")
    axes[0].set_title(maneuver)
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.45)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="best")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _summary_row(scored: _ScoredRun, output_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": scored.run_id,
        "maneuver": scored.maneuver,
        "device": scored.device,
        "source_kind": scored.source_kind,
        "caster_source": scored.caster_source,
        "car_track": scored.car_track_source,
        "series_csv": str((scored.run_dir / "series.csv").relative_to(output_dir)),
        "series_json": str((scored.run_dir / "series.json").relative_to(output_dir)),
    }
    row.update(scored.result.summary.to_dict())
    return row


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    metadata = [
        "run_id",
        "maneuver",
        "device",
        "source_kind",
        "caster_source",
        "car_track",
        "series_csv",
        "series_json",
    ]
    remaining = [key for key in rows[0] if key not in metadata]
    header = metadata + remaining
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in header})
    return path


def _markdown_value(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value).replace("|", "\\|")


def _write_summary_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    columns = (
        ("maneuver", "Maneuver"),
        ("device", "Device"),
        ("mean_abs_alignment_deg", "Mean |align| (deg)"),
        ("total_scrub_m", "Total scrub (m)"),
        ("mean_abs_slip", "Mean |slip|"),
        ("swivel_lag_s", "Lag (s)"),
        ("shimmy_frequency_hz", "Shimmy (Hz)"),
        ("rolling_efficiency", "Longitudinal efficiency"),
    )
    lines = [
        "# Caster comparison summary",
        "",
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_markdown_value(row.get(key)) for key, _ in columns) + " |"
        )
    lines.extend(
        [
            "",
            "Rolling efficiency is the longitudinal-distance fraction; it is not assumed to equal `1 - scrub_fraction`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_comparison(
    manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> ComparisonReport:
    """Score every manifest row and write plots plus CSV/JSON/Markdown reports."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _load_yaml(manifest_file)
    manifest_dir = manifest_file.parent
    raw_runs = manifest.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("manifest must contain a non-empty top-level runs list")
    defaults = manifest.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, Mapping):
        raise ValueError("manifest.defaults must be a mapping")
    if bool(manifest.get("require_pairs", False)):
        devices_by_maneuver: dict[str, set[str]] = {}
        for raw_row in raw_runs:
            if not isinstance(raw_row, Mapping):
                continue
            preview_row = _deep_merge(defaults, raw_row)
            preview_maneuver = str(preview_row.get("maneuver", "")).strip()
            preview_device = str(preview_row.get("device", "")).strip().casefold()
            if preview_maneuver and preview_device in {"ball", "swivel"}:
                devices_by_maneuver.setdefault(preview_maneuver, set()).add(
                    preview_device
                )
        incomplete = {
            maneuver: sorted({"ball", "swivel"} - devices)
            for maneuver, devices in devices_by_maneuver.items()
            if devices != {"ball", "swivel"}
        }
        if incomplete:
            detail = "; ".join(
                f"{maneuver}: missing {', '.join(missing)}"
                for maneuver, missing in sorted(incomplete.items())
            )
            raise ValueError(
                f"manifest requires paired devices for every maneuver ({detail})"
            )
    destination_value = output_dir if output_dir is not None else manifest.get("output_dir", "out/comparison")
    destination = _path(manifest_dir, destination_value, "output_dir")
    destination.mkdir(parents=True, exist_ok=True)

    scored_runs: list[_ScoredRun] = []
    used_ids: set[str] = set()
    ordinal_by_pair: dict[tuple[str, str], int] = {}
    for row_index, raw_row in enumerate(raw_runs):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"runs[{row_index}] must be a mapping")
        row = _deep_merge(defaults, raw_row)
        maneuver = str(row.get("maneuver", "")).strip()
        device = str(row.get("device", "")).strip().casefold()
        if not maneuver:
            raise ValueError(f"runs[{row_index}].maneuver must be non-empty")
        if device not in {"ball", "swivel"}:
            raise ValueError(f"runs[{row_index}].device must be ball or swivel")
        pair = (maneuver, device)
        ordinal_by_pair[pair] = ordinal_by_pair.get(pair, 0) + 1
        default_id = f"{_slug(maneuver)}-{device}-{ordinal_by_pair[pair]}"
        run_id = _slug(str(row.get("id", row.get("run_id", default_id))))
        if run_id in used_ids:
            raise ValueError(f"duplicate run id after normalization: {run_id}")
        used_ids.add(run_id)
        run_dir = destination / "runs" / run_id
        project_config = _config_for_run(manifest, row, manifest_dir)
        frames, source_kind, caster_source = _caster_frames_for_run(
            row, device, project_config, manifest_dir, run_dir
        )
        car_track, track_path = _load_track_for_run(
            manifest, row, project_config, manifest_dir
        )
        metric_values = _metric_values(manifest, row, project_config)
        metric_config = MetricConfig.from_value(metric_values)
        timestamps = np.asarray([frame.t for frame in frames])
        steady = _steady_mask(timestamps, row, metric_values)
        step_time = row.get("step_time_s")
        result = compute_metrics(
            frames,
            car_track,
            _arm_for_run(manifest, row, project_config),
            config=metric_config,
            step_time_s=None if step_time is None else float(step_time),
            steady_mask=steady,
        )
        scored = _ScoredRun(
            run_id=run_id,
            maneuver=maneuver,
            device=device,
            source_kind=source_kind,
            caster_source=caster_source,
            car_track_source=str(track_path),
            result=result,
            run_dir=run_dir,
        )
        _write_run_artifacts(scored)
        scored_runs.append(scored)

    grouped: dict[str, list[_ScoredRun]] = {}
    for scored in scored_runs:
        grouped.setdefault(scored.maneuver, []).append(scored)
    plot_paths = {
        maneuver: _plot_maneuver(
            destination / "plots" / f"{_slug(maneuver)}.png", maneuver, runs
        )
        for maneuver, runs in grouped.items()
    }
    summary_rows = tuple(_summary_row(item, destination) for item in scored_runs)
    _write_summary_csv(destination / "summary.csv", summary_rows)
    _write_json(
        destination / "summary.json",
        {
            "schema": "caster-comparison-v1",
            "manifest": str(manifest_file),
            "runs": summary_rows,
            "plots": {
                maneuver: str(path.relative_to(destination))
                for maneuver, path in plot_paths.items()
            },
        },
    )
    _write_summary_markdown(destination / "summary.md", summary_rows)
    return ComparisonReport(
        output_dir=destination,
        summary_rows=summary_rows,
        plot_paths=plot_paths,
        run_directories={item.run_id: item.run_dir for item in scored_runs},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score paired ball/swivel CasterFrame streams from a YAML manifest."
    )
    parser.add_argument("manifest_path", nargs="?", help="YAML manifest path")
    parser.add_argument("--manifest", dest="manifest_option", help="YAML manifest path")
    parser.add_argument(
        "--output",
        help="override output directory (resolved relative to the manifest)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = args.manifest_option or args.manifest_path
    if not manifest:
        raise SystemExit("a manifest path is required (positional or --manifest)")
    report = run_comparison(manifest, output_dir=args.output)
    print(f"Scored {len(report.summary_rows)} runs")
    print(f"Report: {report.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ComparisonReport", "build_parser", "main", "run_comparison"]
