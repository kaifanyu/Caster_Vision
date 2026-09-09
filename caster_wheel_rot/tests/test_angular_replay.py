"""Measurement gaps, phase corrections and moving-wheel replay regressions."""
import csv

import numpy as np
import pytest

from scripts.simulate_measured_swivel import phase_spokes
from swivel.angular import angular_intervals, replay_phases, write_angular_csv
from swivel.geometry import SwivelGeometry


def sample():
    return dict(timestamps_s=[0., .1, .3, .4], phi_rad=[0., .2, 4., 4.3],
                psi_rad=[0., .1, 2., 2.2], phi_valid=[True, True, False, False],
                psi_valid=[True]*4, reference_corrected=[False, False, True, False],
                interval_quality=[
                    dict(roll_valid=True, heading_valid=True, delta_phi_rad=.2),
                    dict(roll_valid=False, heading_valid=False, delta_phi_rad=None),
                    dict(roll_valid=True, heading_valid=True, delta_phi_rad=.3)])


def test_rates_use_interval_dt_and_measured_increment_not_corrected_phase(tmp_path):
    data = sample()
    data["phi_rad"][-1] = 100.  # Accumulated correction must not enter speed.
    rows = angular_intervals(data)
    assert rows[0]["phi_dot_rad_s"] == pytest.approx(2.)
    assert rows[2]["phi_dot_rad_s"] == pytest.approx(3.)
    assert rows[2]["psi_dot_rad_s"] == pytest.approx(2.)
    assert rows[1]["phi_dot_rad_s"] is None
    assert rows[1]["psi_dot_rad_s"] is None
    assert not rows[2]["accumulated_roll_complete"]
    write_angular_csv(tmp_path / "motion.csv", rows)
    with (tmp_path / "motion.csv").open() as stream:
        saved = list(csv.DictReader(stream))
    assert saved[1]["phi_dot_rad_s"] == ""


def test_legacy_corrected_phase_is_not_used_as_velocity():
    data = sample()
    data["phi_valid"] = [True]*4
    data["interval_quality"][1] = dict(roll_valid=True, heading_valid=True)
    assert angular_intervals(data)[1]["phi_dot_rad_s"] is None


@pytest.mark.parametrize("times", [[0., 0., .2, .3], [0., .2, .1, .3], [0., .1, None, .3]])
def test_bad_timestamps_rejected(times):
    data = sample()
    data["timestamps_s"] = times
    with pytest.raises(ValueError, match="timestamps"):
        angular_intervals(data)


def test_replay_reanchors_at_gaps_and_face_switches():
    rows = angular_intervals(sample())
    phase, segments = replay_phases(rows, [True]*4, [1]*4)
    np.testing.assert_allclose(phase, [0., .2, 0., .3])
    assert segments.tolist() == [0, 0, 1, 1]
    phase, segments = replay_phases(rows, [True, False, True, True], [1, 0, 1, -1])
    assert np.isnan(phase[1])
    assert phase[3] == 0.
    assert segments.tolist() == [0, -1, 1, 2]


def test_replay_180_swivel_moves_hub_and_switches_face():
    geometry = SwivelGeometry(np.eye(3), [2., 0., 3.], [0., 0., 0.],
                              [0., -.045, -.184], [1., 0., 0.], .0515, .049)
    np.testing.assert_allclose(geometry.hub_center_car(0), [0., -.045, -.184])
    np.testing.assert_allclose(geometry.hub_center_car(np.pi), [0., .045, -.184], atol=1e-12)
    assert geometry.visible_face_sign(0) == -geometry.visible_face_sign(np.pi)
    # A quarter roll turns the first spoke from +z toward -y about +x.
    spoke = phase_spokes(geometry, 0., np.pi/2, 1)[0]
    np.testing.assert_allclose(spoke-geometry.face_center_car(0., 1),
                               [0., -.85*.0515, 0.], atol=1e-12)
