"""Known optical events, independently sampled images and receive-time delays."""
import csv
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest
import yaml

from dualcam.timing import brightness_signal, estimate_offset


EDGES = np.cumsum([2., .6, .9, .5, .8, .7, 1.1, .4, .9, .8, .6, 1.2, .5, .8, .7, .9])


def signal(times, exposure=.01, edges=EDGES):
    samples = times[:, None]+np.linspace(-exposure/2, exposure/2, 41)
    on = np.searchsorted(edges, samples) % 2
    return 35+175*on.mean(axis=1)


def test_delay_sign_and_magnitude_from_independent_camera_samples():
    a = np.arange(0, 18, 1/30)
    b = np.arange(.011, 18, 1/29.9)
    result = estimate_offset(a, signal(a, .0156), b+.030, signal(b, .008))
    assert result['success'], result['reasons']
    assert abs(result['brio_offset_s']+.030) < .008
    assert result['matched_edges'] == len(EDGES)
    assert len(result['per_edge_offset_bracket_s']) == len(EDGES)


def test_drift_is_not_hidden_by_one_median_offset():
    t = np.arange(0, 18, 1/30)
    result = estimate_offset(t, signal(t), t+.02+.007*t, signal(t))
    assert not result['success']
    assert result['estimated_delay_change_ms'] > 50


def test_missing_pulse_and_periodic_signal_are_rejected():
    t = np.arange(0, 18, 1/30)
    with pytest.raises(ValueError, match='different pulse'):
        estimate_offset(t, signal(t), t, signal(t, edges=np.delete(EDGES, [6, 7])))
    periodic = np.arange(2, 14, .5)
    with pytest.raises(ValueError, match='irregular'):
        estimate_offset(t, signal(t, edges=periodic), t, signal(t, edges=periodic))
    periodic = np.array([[x, x+.4] for x in range(2, 14)]).ravel()
    with pytest.raises(ValueError, match='irregular'):
        estimate_offset(t, signal(t, edges=periodic), t, signal(t, edges=periodic))


def test_flat_or_incomplete_signal_cannot_produce_an_offset():
    t = np.arange(0, 18, 1/30)
    with pytest.raises(ValueError, match='contrast'):
        estimate_offset(t, np.full(len(t), 35), t, signal(t))
    values = signal(t)
    values[:10] = 210
    with pytest.raises(ValueError, match='OFF'):
        estimate_offset(t, values, t, signal(t))


def test_brightness_extraction_uses_raw_csv_times_and_selected_patch():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root/'session.json').write_text(json.dumps({'cameras': {}}))
        writer = cv2.VideoWriter(str(root/'c920.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 30, (64, 48))
        assert writer.isOpened()
        for value in (30, 190, 30):
            frame = np.zeros((48, 64, 3), np.uint8)
            frame[8:24, 8:24] = value
            writer.write(frame)
        writer.release()
        with (root/'c920_timestamps.csv').open('w', newline='') as stream:
            out = csv.writer(stream)
            out.writerow(['frame_index', 'timestamp_s'])
            out.writerows(enumerate([.013, .051, .083]))
        times, values = brightness_signal(root, 'c920', [10, 10, 10, 10])
        np.testing.assert_allclose(times, [.013, .051, .083])
        np.testing.assert_allclose(values, [30, 190, 30], atol=3)
        with pytest.raises(ValueError, match='ROI'):
            brightness_signal(root, 'c920', [60, 0, 10, 10])


def test_timing_cli_reads_two_recordings_without_modifying_rig_or_timestamps(tmp_path):
    from dualcam.config import DEFAULT_CONFIG
    from scripts.calibrate_timing import main
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    directory = tmp_path/'recording'
    directory.mkdir()
    metadata = {'status': 'complete', 'mode': 'timing', 'cameras': {}}
    original_csv = {}
    for ci, name in enumerate(('c920', 'brio101')):
        profile = cfg['cameras'][name]['capture']
        profile.update(width=64, height=48)
        metadata['cameras'][name] = {'requested': profile}
        optical_time = np.arange(ci*.011, 18, 1/(30-ci*.1))
        brightness = signal(optical_time)
        writer = cv2.VideoWriter(str(directory/f'{name}.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 30, (64, 48))
        assert writer.isOpened()
        for value in brightness:
            writer.write(np.full((48, 64, 3), round(value), np.uint8))
        writer.release()
        path = directory/f'{name}_timestamps.csv'
        with path.open('w', newline='') as stream:
            out = csv.writer(stream)
            out.writerow(['frame_index', 'timestamp_s'])
            out.writerows(enumerate(optical_time+ci*.030))
        original_csv[path] = path.read_bytes()
    (directory/'session.json').write_text(json.dumps(metadata))
    config = tmp_path/'rig.yaml'
    config.write_text(yaml.safe_dump(cfg))
    original_config = config.read_bytes()
    output = tmp_path/'timing'
    status = main(['--config', str(config), '--session', str(directory), '--output', str(output),
                   '--c920-roi', '8', '8', '32', '24', '--brio101-roi', '8', '8', '32', '24'])
    assert status == 0
    report = json.loads((output/'report.json').read_text())
    assert abs(report['brio_offset_s']+.030) < .008
    assert yaml.safe_load((output/'timing_suggestion.yaml').read_text())['timing']['verified'] is False
    assert config.read_bytes() == original_config
    assert all(path.read_bytes() == data for path, data in original_csv.items())
