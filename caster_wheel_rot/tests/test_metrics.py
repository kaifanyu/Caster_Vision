from __future__ import annotations

import json

import numpy as np
import pytest

from common.caster_frame import CasterFrame
from common.kinematics import CarTrack
from metrics.metrics import MetricConfig, compute_metrics, signed_angle


def _track(
    t: np.ndarray,
    *,
    vx: np.ndarray | float = 1.0,
    vy: np.ndarray | float = 0.0,
    omega: np.ndarray | float = 0.0,
    theta: np.ndarray | float = 0.0,
) -> CarTrack:
    t = np.asarray(t, dtype=float)

    def values(value: np.ndarray | float) -> np.ndarray:
        return np.broadcast_to(np.asarray(value, dtype=float), t.shape).copy()

    vx_array, vy_array = values(vx), values(vy)
    # Position is metadata for metrics because the already-derived velocities
    # are authoritative.  A cumulative approximation keeps it plausible.
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    if len(t) > 1:
        dt = np.diff(t)
        x[1:] = np.cumsum(0.5 * (vx_array[:-1] + vx_array[1:]) * dt)
        y[1:] = np.cumsum(0.5 * (vy_array[:-1] + vy_array[1:]) * dt)
    return CarTrack(
        t=t,
        x=x,
        y=y,
        theta=values(theta),
        vx_world=vx_array,
        vy_world=vy_array,
        omega=values(omega),
    )


def _axis_for_heading(heading: np.ndarray) -> np.ndarray:
    """Invert the specified heading convention h = z cross roll_axis."""

    h = np.asarray(heading, dtype=float)
    h = h / np.linalg.norm(h)
    return np.array([h[1], -h[0], 0.0])


def _frames(
    t: np.ndarray,
    heading: np.ndarray,
    *,
    speed: np.ndarray | float = 1.0,
    radius: float = 0.5,
    raw: list[dict] | None = None,
) -> list[CasterFrame]:
    t = np.asarray(t, dtype=float)
    headings = np.asarray(heading, dtype=float)
    if headings.shape == (2,):
        headings = np.broadcast_to(headings, (len(t), 2))
    speeds = np.broadcast_to(np.asarray(speed, dtype=float), t.shape)
    raw_values = raw if raw is not None else [{} for _ in t]
    return [
        CasterFrame(
            t=time,
            roll_axis_car=_axis_for_heading(headings[index]),
            omega_roll=float(abs(speeds[index]) / radius),
            omega_spin=0.0,
            r_eff=radius,
            raw=dict(raw_values[index]),
        )
        for index, time in enumerate(t)
    ]


def test_signed_angle_and_straight_run_have_analytic_values() -> None:
    assert signed_angle(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(
        np.pi / 2
    )
    t = np.linspace(0.0, 2.0, 21)
    result = compute_metrics(_frames(t, np.array([1.0, 0.0])), _track(t))

    assert np.nanmax(np.abs(result.series.alignment_error_rad)) < 1e-12
    assert np.max(result.series.scrub_speed_mps) < 1e-12
    assert np.nanmax(np.abs(result.series.slip_ratio)) < 1e-12
    assert result.summary.total_scrub_m == pytest.approx(0.0, abs=1e-12)
    assert result.summary.scrub_fraction == pytest.approx(0.0, abs=1e-12)
    assert result.summary.rolling_efficiency == pytest.approx(1.0, abs=1e-12)
    assert result.summary.mean_slip == pytest.approx(0.0, abs=1e-12)


def test_spin_in_place_has_known_contact_scrub_and_masks_slip() -> None:
    t = np.linspace(0.0, 2.0, 21)
    track = _track(t, vx=0.0, vy=0.0, omega=2.0, theta=2.0 * t)
    frames = _frames(t, np.array([1.0, 0.0]), speed=0.0)
    result = compute_metrics(frames, track, r_arm_car=np.array([0.5, 0.0]))

    np.testing.assert_allclose(
        result.series.contact_velocity_car,
        np.tile([0.0, 1.0], (len(t), 1)),
        atol=1e-12,
    )
    np.testing.assert_allclose(result.series.scrub_speed_mps, 1.0, atol=1e-12)
    np.testing.assert_allclose(result.series.alignment_error_rad, np.pi / 2, atol=1e-12)
    assert result.summary.total_scrub_m == pytest.approx(2.0)
    assert result.summary.slip_valid_count == 0
    assert np.isnan(result.summary.mean_slip)


def test_forty_five_degree_efficiency_is_not_one_minus_scrub_fraction() -> None:
    t = np.linspace(0.0, 3.0, 31)
    angle = np.pi / 4
    heading = np.array([np.cos(angle), np.sin(angle)])
    # Match roll speed to the longitudinal component so slip itself is zero.
    result = compute_metrics(
        _frames(t, heading, speed=np.cos(angle)),
        _track(t),
    )

    assert result.summary.mean_abs_alignment_rad == pytest.approx(angle)
    assert result.summary.scrub_fraction == pytest.approx(np.sin(angle))
    assert result.summary.rolling_efficiency == pytest.approx(np.cos(angle))
    assert result.summary.rolling_efficiency != pytest.approx(
        1.0 - result.summary.scrub_fraction
    )


def test_roll_dropout_is_masked_and_gap_is_not_integrated_across() -> None:
    t = np.arange(5.0)
    raw = [{}, {}, {"roll_valid": False}, {}, {}]
    result = compute_metrics(
        _frames(t, np.array([1.0, 0.0]), raw=raw),
        _track(t),
    )

    np.testing.assert_array_equal(result.series.slip_valid, [True, True, False, True, True])
    assert result.summary.slip_valid_count == 4
    # Valid intervals are [0,1] and [3,4], not one bridge from t=1 to t=3.
    assert result.summary.slip_valid_duration_s == pytest.approx(2.0)
    # Heading-only metrics remain valid throughout a roll-only dropout.
    assert result.summary.scrub_valid_duration_s == pytest.approx(4.0)


def test_raw_heading_and_generic_relative_contact_velocity_are_honored() -> None:
    t = np.array([0.0, 1.0, 2.0])
    raw = [
        {
            "rolling_heading_car": [1.0, 0.0],
            "contact_velocity_rel_car": [1.0, 0.0],
            "contact_offset_car": [0.2, -0.1],
        }
        for _ in t
    ]
    # The axis-derived heading is +y; raw keeps the observed +x heading even
    # though the chassis and caster roll rate are stationary.
    frames = _frames(t, np.array([0.0, 1.0]), speed=0.0, raw=raw)
    result = compute_metrics(frames, _track(t, vx=0.0, vy=0.0))

    np.testing.assert_allclose(
        result.series.rolling_heading_car, np.tile([1.0, 0.0], (len(t), 1))
    )
    np.testing.assert_allclose(
        result.series.contact_velocity_car, np.tile([1.0, 0.0], (len(t), 1))
    )
    np.testing.assert_allclose(result.series.alignment_error_rad, 0.0)


def test_confidence_zero_and_out_of_track_range_are_invalid() -> None:
    track_t = np.array([0.0, 1.0, 2.0])
    frame_t = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    raw = [{}, {}, {"confidence": 0.0}, {}, {}]
    result = compute_metrics(
        _frames(frame_t, np.array([1.0, 0.0]), raw=raw),
        _track(track_t),
    )
    np.testing.assert_array_equal(result.series.track_valid, [False, True, True, True, False])
    np.testing.assert_array_equal(result.series.heading_valid, [False, True, False, True, False])
    assert result.summary.alignment_valid_count == 2


def test_positive_lag_means_response_occurs_after_demand() -> None:
    dt = 0.02
    t = np.arange(0.0, 12.0, dt)
    known_lag = 0.24
    frequency = 0.37
    amplitude = np.deg2rad(20.0)
    demand = amplitude * np.sin(2 * np.pi * frequency * t)
    response = amplitude * np.sin(2 * np.pi * frequency * (t - known_lag))
    velocity = np.column_stack([np.cos(demand), np.sin(demand)])
    headings = np.column_stack([np.cos(response), np.sin(response)])
    result = compute_metrics(
        _frames(t, headings),
        _track(t, vx=velocity[:, 0], vy=velocity[:, 1]),
        config=MetricConfig(lag_max_s=0.6, lag_min_samples=50),
    )

    assert result.summary.swivel_lag_s > 0.0
    assert result.summary.swivel_lag_s == pytest.approx(known_lag, abs=dt)


def test_welch_recovers_known_shimmy_frequency_and_amplitude() -> None:
    dt = 0.01
    t = np.arange(0.0, 10.0, dt)
    frequency = 2.0
    amplitude_deg = 3.0
    angle = np.deg2rad(amplitude_deg) * np.sin(2 * np.pi * frequency * t)
    headings = np.column_stack([np.cos(angle), np.sin(angle)])
    result = compute_metrics(
        _frames(t, headings),
        _track(t),
        config=MetricConfig(
            shimmy_nperseg=500,
            shimmy_min_samples=100,
            shimmy_min_frequency_hz=0.5,
        ),
        steady_mask=np.ones(len(t), dtype=bool),
    )

    assert result.summary.shimmy_frequency_hz == pytest.approx(frequency, abs=0.11)
    assert result.summary.shimmy_amplitude_deg == pytest.approx(amplitude_deg, rel=0.08)


def test_shimmy_max_frequency_excludes_out_of_band_peak() -> None:
    dt = 0.01
    t = np.arange(0.0, 10.0, dt)
    angle = np.deg2rad(2.0) * np.sin(2 * np.pi * 2.0 * t)
    angle += np.deg2rad(8.0) * np.sin(2 * np.pi * 30.0 * t)
    headings = np.column_stack([np.cos(angle), np.sin(angle)])
    result = compute_metrics(
        _frames(t, headings),
        _track(t),
        config=MetricConfig(
            shimmy_nperseg=500,
            shimmy_min_samples=100,
            shimmy_min_frequency_hz=0.5,
            shimmy_max_frequency_hz=20.0,
        ),
    )
    assert result.summary.shimmy_frequency_hz == pytest.approx(2.0, abs=0.11)


def test_settling_time_and_json_safe_serialization() -> None:
    dt = 0.01
    t = np.arange(0.0, 3.0 + dt, dt)
    initial = np.deg2rad(20.0)
    tau = 0.5
    response = initial * np.exp(-t / tau)
    headings = np.column_stack([np.cos(response), np.sin(response)])
    result = compute_metrics(
        _frames(t, headings),
        _track(t),
        config=MetricConfig(
            settling_threshold_deg=5.0,
            settling_dwell_s=0.2,
            step_time_s=0.0,
        ),
    )

    expected = tau * np.log(4.0)
    assert result.summary.settling_time_s == pytest.approx(expected, abs=2 * dt)
    # Non-applicable or nonfinite values are represented as JSON null rather
    # than leaking NumPy scalars or non-standard NaN tokens.
    encoded = json.dumps(result.to_dict(), allow_nan=False)
    assert '"series"' in encoded and '"summary"' in encoded
