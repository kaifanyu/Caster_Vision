"""Conditional surface-landmark fits for the shared-axis bundle solver."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .model import initialize_surface_point, points_to_polar, polar_to_points


def _track_projection(camera, F, pivot, angles, shell, radius, gap, red_sign):
    """Homogeneous pixel coordinates A_i @ unit_point + b_i for one track."""
    alpha, beta = angles[:, 0], angles[:, 1 + shell]
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    zeros = np.zeros(len(alpha))
    rotations = np.stack((cb, -sb, zeros,
                          ca*sb, ca*cb, -sa,
                          sa*sb, sa*cb, ca), axis=1).reshape(-1, 3, 3)
    matrix = camera['K'] @ camera['R'] @ F
    A = radius * np.einsum('ij,njk->nik', matrix, rotations)
    offset = red_sign * (1 - 2*shell) * gap / 2
    centers = np.column_stack((zeros, -sa*offset, ca*offset))
    b = centers @ matrix.T + camera['K'] @ (camera['R'] @ pivot + camera['t'])
    return A, b


def _track_residual_jacobian(polar, sign, A, b, uv, sqrt_weight):
    azimuth, theta = polar
    ca, sa, ct, st = np.cos(azimuth), np.sin(azimuth), np.cos(theta), np.sin(theta)
    point = np.array([ca*st, sa*st, sign*ct])
    derivative = np.array([[-sa*st, ca*ct], [ca*st, sa*ct], [0., -sign*st]])
    homogeneous = A @ point + b
    depth = homogeneous[:, 2]
    unclamped = np.abs(depth) >= 1e-9
    depth = np.where(unclamped, depth, np.copysign(1e-9, depth))
    prediction = homogeneous[:, :2] / depth[:, None]
    dh = A @ derivative
    dz = dh[:, 2] * unclamped[:, None]
    jac = (dh[:, :2] - prediction[:, :, None]*dz[:, None, :]) / depth[:, None, None]
    return ((prediction-uv)*sqrt_weight[:, None]).ravel(), (jac*sqrt_weight[:, None, None]).reshape(-1, 2)


def _soft_l1_cost(residual, scale):
    return float(scale * np.sum(np.hypot(scale, residual) - scale))


def refine_surface_points(obs, angles, polar, signs, cameras, F, pivot,
                          radius, gap, red_sign, *, robust_px, max_nfev):
    """Refine prepared landmarks with axes, pivot and every frame pose fixed.

    Keep all observations, weights and signed-hemisphere bounds. Compare both
    the current point and a fresh visible-ray seed; retain only nonincreasing
    robust-cost updates. This is initialization, not a calibration acceptance
    test: a subsequent full joint solve and its quality gates are required.
    """
    refined = np.asarray(polar, float).copy()
    before = after = 0.
    evaluations = converged = improved = 0
    # Sorting once avoids rescanning every observation for each landmark.
    order = np.argsort(obs['landmark'], kind='stable')
    groups = np.split(order, np.flatnonzero(np.diff(obs['landmark'][order])) + 1)
    for ids in groups:
        first = ids[np.argmin(obs['frame'][ids])]
        landmark = int(obs['landmark'][first])
        ci, shell = int(obs['camera'][first]), int(obs['shell'][first])
        camera = cameras[ci]
        q, uv = angles[obs['frame'][ids]], obs['uv'][ids]
        weights = np.sqrt(obs['weight'][ids])
        A, b = _track_projection(camera, F, pivot, q, shell, radius, gap, red_sign)

        def evaluate(p):
            return _track_residual_jacobian(p, signs[landmark], A, b, uv, weights)

        current = refined[landmark].copy()
        best, best_cost = current, _soft_l1_cost(evaluate(current)[0], robust_px)
        before += best_cost
        point = initialize_surface_point(camera, obs['uv'][first], F, pivot,
                                         angles[obs['frame'][first]], shell, radius, gap, red_sign)
        ray_seed = points_to_polar(point)[0]
        starts = [current]
        if np.linalg.norm(polar_to_points([current], [signs[landmark]])[0] - point) > 1e-6:
            starts.append(ray_seed)
        solved = False
        for start in starts:
            fit = least_squares(lambda p: evaluate(p)[0], start,
                                jac=lambda p: evaluate(p)[1],
                                bounds=([-np.inf, 0.], [np.inf, np.pi/2]),
                                loss='soft_l1', f_scale=robust_px, max_nfev=max_nfev,
                                ftol=1e-8, xtol=1e-8, gtol=1e-8)
            evaluations += fit.nfev
            if np.isfinite(fit.cost) and fit.cost <= best_cost:
                best, best_cost, solved = fit.x, float(fit.cost), bool(fit.success)
        improved += int(best_cost < _soft_l1_cost(evaluate(current)[0], robust_px) - 1e-9)
        converged += int(solved)
        refined[landmark] = best
        after += best_cost
    return refined, {'landmarks': len(groups), 'improved_landmarks': improved,
                     'converged_landmarks': converged, 'point_nfev': evaluations,
                     'cost_before': before, 'cost_after': after}
