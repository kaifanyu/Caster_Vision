"""Conservative static-yoke exclusions for the existing fixed-camera run.

Polygons were manually reviewed on undistorted C920 frames 231/556 and Brio
frames 239/574 (about 8.0/19.2 seconds). They cover the definite stationary
outer yoke, including reflected bright sections missed by a brightness gate.
They are NOT general shell segmentation, and do not model the cross shaft,
rim, moving shadow, or scene occluders. They can remove a small amount of true
shell near the yoke. A changed camera mount/calibration requires new review.

Coordinates below are normalized edge coordinates of the stated crop bounds.
The x/y pixel-center convention is (u-x0+.5)/width, (v-y0+.5)/height. Both the
measured image and the texture reference must apply the same exclusion.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

RECORDING_BOUNDS = {'c920':(506,69,1415,978), 'brio101':(355,46,1403,1080)}
YOKE_POLYGONS = {
    'c920': [[.406,1.], [.421,.83], [.435,.61], [.448,.42], [.454,.325],
             [.475,.252], [.500,.207], [.529,.195], [.560,.207], [.588,.244],
             [.606,.300], [.612,.39], [.610,.60], [.607,.82], [.598,1.]],
    'brio101': [[.57,0.], [.71,0.], [.79,.115], [.85,.23], [.895,.36],
                [.925,.49], [.931,.60], [.914,.70], [.884,.76], [.85,.787],
                [.818,.778], [.787,.755], [.772,.70], [.772,.61], [.766,.50],
                [.747,.395], [.721,.295], [.684,.20], [.646,.11]],
}


def camera_name(camera):
    if isinstance(camera,(int,np.integer)) and camera in (0,1):
        return ('c920','brio101')[int(camera)]
    if camera not in RECORDING_BOUNDS:
        raise ValueError('The reviewed yoke mask only supports c920/cam0 or brio101/cam1')
    return camera


def static_yoke_mask(camera, uv, bounds=None, *, margin_px=5.):
    """True means excluded. Native undistorted uv may have any leading shape.

    ``bounds`` validates the caller's declared crop; polygon location remains
    anchored to reviewed native image coordinates. Passing different bounds
    raises rather than silently moving the physical yoke mask.
    """
    name=camera_name(camera); anchor=np.asarray(RECORDING_BOUNDS[name],float)
    if bounds is not None and not np.array_equal(np.asarray(bounds),anchor):
        raise ValueError('Crop differs from manually reviewed recording bounds; review yoke coordinates before use')
    if not np.isfinite(margin_px) or margin_px<0:raise ValueError('margin_px must be finite and nonnegative')
    uv=np.asarray(uv,float)
    if uv.shape[-1:]!=(2,) or not np.isfinite(uv).all():raise ValueError('uv must contain finite native pixel pairs')
    points=uv.reshape(-1,2); vertices=np.asarray(YOKE_POLYGONS[name])* (anchor[2:]-anchor[:2])+anchor[:2]-.5
    inside=np.zeros(len(points),bool); near=np.zeros(len(points),bool)
    x,y=points.T
    for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
        dy=b[1]-a[1]
        if abs(dy)>1e-12:
            crosses=((a[1]>y)!=(b[1]>y)) & (x<(b[0]-a[0])*(y-a[1])/dy+a[0])
            inside ^= crosses
        edge=b-a
        fraction=np.clip(((points-a)@edge)/(edge@edge),0.,1.)
        distance=np.linalg.norm(points-a-fraction[:,None]*edge,axis=1)
        near |= distance <= margin_px
    return (inside|near).reshape(uv.shape[:-1])


def observed_shell_mask(camera, uv, bounds=None, *, margin_px=5.):
    """True means eligible with respect to the reviewed yoke only."""
    return ~static_yoke_mask(camera,uv,bounds,margin_px=margin_px)


def occlusion_metadata(margin_px=5.):
    return {'kind':'manually_reviewed_static_yoke_exclusion', 'margin_native_px':margin_px,
            'bounds':RECORDING_BOUNDS, 'normalized_polygons':YOKE_POLYGONS,
            'reference_frames':{'c920':[231,556],'brio101':[239,574]},
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Existing fixed camera mount, original calibrated undistorted pixels; definite outer yoke only. Cross shaft, rim and other occluders require independent masks. No accuracy certificate.'}
