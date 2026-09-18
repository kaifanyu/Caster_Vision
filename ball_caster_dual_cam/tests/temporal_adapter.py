"""Exercise ported image-level tests through the production native-view tracker."""
from types import SimpleNamespace
import numpy as np
from ballrot.segment import segment_frame
from ballrot.temporal import safe_tracking_masks
from dualcam.fusion import IncrementProposal


def run_pipeline(frames, *, K, circle, R_bc, segment_config, track_config,
                 estimate_config, temporal_config):
    camera = {'K': K, 'R': np.eye(3), 't': np.zeros(3)}
    tracker = IncrementProposal(camera, R_bc, circle, np.zeros(3), track_config, temporal_config)
    valid, poses, increments, matches = [], [], [], []
    for i, item in enumerate(frames):
        image, time = item if isinstance(item, tuple) else (item, i/30)
        mask = segment_frame(image, circle, **segment_config)
        mask = safe_tracking_masks(mask, temporal_config.get('boundary_margin_px', 3))
        tracker.observe(image, {'top': mask.top, 'bottom': mask.bottom}, i, time)
        valid.append(tracker.valid.copy())
        poses.append([t.pose.copy() for t in tracker.temporal])
        if i:
            increments.append(tracker.last_increments.copy())
            matches.append(tracker.last_matches)
    valid, poses = np.array(valid), np.array(poses)
    return SimpleNamespace(top_step_valid=valid[:,0], bottom_step_valid=valid[:,1],
                           top_absolute=poses[:,0], bottom_absolute=poses[:,1],
                           top_increments=[d[0] for d in increments],
                           motion=SimpleNamespace(alpha_top=np.where(valid[:,0], 0., np.nan)),
                           matches=matches, temporal={'frames':dict(zip(('top','bottom'),
                                                        [t.history for t in tracker.temporal]))})
