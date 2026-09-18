"""Declared home intervals anchor actual images, not fabricated motion priors."""
import numpy as np
import pytest

from dualcam.solver import fit_joint
from dualcam.tracking import check_home_hold
from tests.test_solver import make_clip, setup_scene


def paused_clip():
    cameras, F, C = setup_scene()
    q = np.zeros((12, 3))
    q[4:, 1] = np.linspace(.04, .4, 8)
    clip, _ = make_clip(cameras, F, C, 'swivel', frames=12, count=10, noise=.03, q=q)
    return cameras, F, C, clip, q


def test_home_interval_checks_pixels_and_rejects_moving_interval():
    _, _, _, clip, _ = paused_clip()
    result = check_home_hold(clip, .1)
    assert result['frames'] == 4
    assert max(x['max_median_displacement_px'] for x in result['per_camera_shell']) < .2
    with pytest.raises(ValueError, match='moves'):
        check_home_hold(clip, .3)
    with pytest.raises(ValueError, match='end before'):
        check_home_hold(clip, .4)
    with pytest.raises(ValueError, match='too few'):
        check_home_hold(clip, .1, min_points=20)


def test_multiple_home_images_anchor_motion_when_first_image_has_no_red_tracks():
    cameras, F, C, clip, truth = paused_clip()
    obs = clip['observations']
    keep = ~((obs['frame'] == 0) & (obs['shell'] == 0))
    clip['observations'] = {k: v[keep] for k, v in obs.items()}
    clip['home_hold_s'] = .1
    result = fit_joint([clip], cameras, F, C, .1, .02)
    assert result['success'], result['diagnostics']
    out = result['datasets'][0]
    assert out['valid'][1:, 1].all()
    np.testing.assert_allclose(out['angles'][1:4, 1], 0, atol=1e-12)
    np.testing.assert_allclose(out['angles'][4:, 1], truth[4:, 1], atol=.01)
    assert out['support_diagnostics'][1]['supported_home_frames'] == 3
    # Without the declaration those other images are not known reference poses.
    clip['home_hold_s'] = 0.
    result = fit_joint([clip], cameras, F, C, .1, .02)
    assert not result['datasets'][0]['valid'][:, 1].any()


def test_home_hold_cannot_anchor_every_frame_or_accept_nan_duration():
    cameras, F, C, clip, _ = paused_clip()
    for invalid in (-1., float('nan'), 1.):
        clip['home_hold_s'] = invalid
        with pytest.raises(ValueError, match='home_hold_s'):
            fit_joint([clip], cameras, F, C, .1, .02)
