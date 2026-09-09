from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from common.caster_frame import CasterFrame, save_caster_frames
from compare.run_comparison import main, run_comparison


def _write_track(path: Path, t: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "x", "y", "theta", "vx_world", "vy_world", "omega"])
        for time in t:
            writer.writerow([time, time, 0.0, 0.0, 1.0, 0.0, 0.0])


def _common_frames(t: np.ndarray, device: str) -> list[CasterFrame]:
    output = []
    for index, time in enumerate(t):
        device_raw = (
            {"device": "ball", "alpha": index * 0.1, "beta1": 11 + index, "beta2": -7}
            if device == "ball"
            else {"device": "swivel", "phi": index * 4.0, "psi": 99 - index}
        )
        output.append(
            CasterFrame(
                t=float(time),
                # z cross (0,-1,0) = +x, the specified calibrated sign.
                roll_axis_car=np.array([0.0, -1.0, 0.0]),
                omega_roll=2.0,
                omega_spin=0.0,
                r_eff=0.5,
                raw={
                    **device_raw,
                    "heading_valid": True,
                    "roll_valid": True,
                    "confidence": 1.0,
                },
            )
        )
    return output


def test_precomputed_end_to_end_uses_identical_metrics_and_writes_report(
    tmp_path: Path,
) -> None:
    t = np.linspace(0.0, 2.0, 21)
    save_caster_frames(tmp_path / "ball_frames.json", _common_frames(t, "ball"))
    save_caster_frames(tmp_path / "swivel_frames.json", _common_frames(t, "swivel"))
    _write_track(tmp_path / "track.csv", t)
    manifest = {
        "output_dir": "report",
        "car_track_options": {"smooth_window": 1},
        "runs": [
            {
                "id": "straight-ball",
                "maneuver": "straight line",
                "device": "ball",
                "caster_frames": "ball_frames.json",
                "car_track": "track.csv",
            },
            {
                "id": "straight-swivel",
                "maneuver": "straight line",
                "device": "swivel",
                "caster_frames": "swivel_frames.json",
                "car_track": "track.csv",
            },
        ],
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    report = run_comparison(manifest_path)

    assert report.output_dir == (tmp_path / "report").resolve()
    assert len(report.summary_rows) == 2
    ball, swivel = report.summary_rows
    # Device-only raw alpha/beta versus phi/psi fields never reach metrics.
    for field in (
        "mean_abs_alignment_deg",
        "total_scrub_m",
        "scrub_fraction",
        "rolling_efficiency",
        "mean_slip",
        "mean_abs_slip",
    ):
        assert ball[field] == pytest.approx(swivel[field], nan_ok=True)
    assert ball["mean_abs_alignment_deg"] == pytest.approx(0.0)
    assert ball["total_scrub_m"] == pytest.approx(0.0)
    assert ball["rolling_efficiency"] == pytest.approx(1.0)
    assert ball["mean_slip"] == pytest.approx(0.0)

    expected = [
        "summary.csv",
        "summary.json",
        "summary.md",
        "plots/straight-line.png",
        "runs/straight-ball/series.csv",
        "runs/straight-ball/series.json",
        "runs/straight-ball/summary.json",
        "runs/straight-swivel/series.csv",
        "runs/straight-swivel/series.json",
        "runs/straight-swivel/summary.json",
    ]
    for relative in expected:
        artifact = report.output_dir / relative
        assert artifact.is_file(), relative
        assert artifact.stat().st_size > 0

    summary = json.loads((report.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "caster-comparison-v1"
    assert len(summary["runs"]) == 2
    series = json.loads(
        (report.output_dir / "runs/straight-ball/series.json").read_text(encoding="utf-8")
    )
    assert series["schema"] == "caster-metric-series-v1"
    assert len(series["series"]["t"]) == len(t)
    assert "alpha" not in series["series"] and "phi" not in series["series"]
    markdown = (report.output_dir / "summary.md").read_text(encoding="utf-8")
    assert "straight line" in markdown and "Longitudinal efficiency" in markdown


def test_cli_accepts_manifest_option_and_relative_output_override(
    tmp_path: Path,
) -> None:
    t = np.linspace(0.0, 1.0, 11)
    save_caster_frames(tmp_path / "frames.json", _common_frames(t, "ball"))
    _write_track(tmp_path / "track.csv", t)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "runs": [
                    {
                        "maneuver": "straight",
                        "device": "ball",
                        "caster_frames": "frames.json",
                        "car_track": "track.csv",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert main(["--manifest", str(manifest_path), "--output", "cli-report"]) == 0
    assert (tmp_path / "cli-report/summary.csv").is_file()


def test_precomputed_tracking_only_or_unsynchronized_metadata_is_rejected(
    tmp_path: Path,
) -> None:
    t = np.linspace(0.0, 1.0, 11)
    track_path = tmp_path / "track.csv"
    _write_track(track_path, t)
    frames = _common_frames(t, "ball")

    def run_with(metadata: dict, **row_extra) -> None:
        frame_path = save_caster_frames(
            tmp_path / "frames_flagged.json", frames, metadata=metadata
        )
        manifest_path = tmp_path / "flagged.yaml"
        manifest_path.write_text(
            yaml.safe_dump(
                {
                    "runs": [
                        {
                            "maneuver": "straight",
                            "device": "ball",
                            "caster_frames": frame_path.name,
                            "car_track": track_path.name,
                            **row_extra,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        run_comparison(manifest_path, output_dir="flagged-report")

    with pytest.raises(ValueError, match="tracking-only"):
        run_with({"tracking_only": True, "trusted_measurement": False})
    with pytest.raises(ValueError, match="unsynchronized"):
        run_with({"trusted_measurement": True, "clocks_synchronized": False})
    # An externally synchronized legacy/exported stream can make the assertion
    # explicitly at the manifest boundary.
    run_with(
        {"trusted_measurement": True, "clocks_synchronized": False},
        clocks_synchronized=True,
    )


def test_manifest_rejects_ambiguous_or_duplicate_runs(tmp_path: Path) -> None:
    t = np.array([0.0, 1.0, 2.0])
    save_caster_frames(tmp_path / "frames.json", _common_frames(t, "ball"))
    _write_track(tmp_path / "track.csv", t)
    manifest_path = tmp_path / "bad.yaml"
    row = {
        "id": "same",
        "maneuver": "straight",
        "device": "ball",
        "caster_frames": "frames.json",
        "car_track": "track.csv",
    }
    manifest_path.write_text(
        yaml.safe_dump({"runs": [row, dict(row)]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate run id"):
        run_comparison(manifest_path)


def test_manifest_can_require_complete_ball_swivel_pairs(tmp_path: Path) -> None:
    manifest_path = tmp_path / "unpaired.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "require_pairs": True,
                "runs": [
                    {
                        "maneuver": "straight",
                        "device": "ball",
                        "caster_frames": "unused.json",
                        "car_track": "unused.csv",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing swivel"):
        run_comparison(manifest_path)


@pytest.mark.parametrize(
    ("assumptions", "message"),
    [
        ({}, "camera_fixed_to_chassis"),
        ({"camera_fixed_to_chassis": True}, "ball_center_stationary_in_image"),
        (
            {
                "camera_fixed_to_chassis": True,
                "ball_center_stationary_in_image": True,
            },
            "clock synchronization",
        ),
    ],
)
def test_raw_comparison_requires_declared_physical_assumptions_and_sync(
    tmp_path: Path, assumptions: dict, message: str
) -> None:
    manifest_path = tmp_path / "raw.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "runs": [
                    {
                        "maneuver": "straight",
                        "device": "ball",
                        "clip": "missing.mp4",
                        "car_track": "missing.csv",
                        "config": {"assumptions": assumptions},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        run_comparison(manifest_path)
