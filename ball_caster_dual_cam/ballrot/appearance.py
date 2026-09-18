"""Appearance verification reused from ball_caster_rot mechanical_rematch."""
import cv2
import numpy as np

def _patch_similarity(first, second, source_uv, target_uv, size=15):
    half = size//2+1
    for image, point in ((first, source_uv), (second, target_uv)):
        if not (half <= point[0] < image.shape[1]-half and half <= point[1] < image.shape[0]-half):
            return None
    a = cv2.getRectSubPix(first, (size, size), tuple(map(float, source_uv))).astype(float)
    b = cv2.getRectSubPix(second, (size, size), tuple(map(float, target_uv))).astype(float)
    if min(a.std(), b.std()) < 5.:
        return None
    a -= a.mean()
    b -= b.mean()
    return float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b)))
