"""Native-time, two-view motion fusion with measured keyframe recovery.

One angle/rate state represents common roll and the two independent shell spins.
Each camera updates it at its own timestamp; no nearest-pair images are dropped.
The image fit uses fixed, previously anchored material points. A prediction may
seed a pixel search but may never create new reference points on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ballrot.track import KLTConfig, PersistentKLTTracker, detect_features, track_features
from ballrot.estimate import solve_hemisphere_increment
from ballrot.sphere import sphere_pose_from_circle
from ballrot.temporal import TemporalConfig, TemporalRotationTracker
from .model import initialize_surface_point, project, rotation_x, rotation_z, world_points


@dataclass(frozen=True)
class FusionConfig:
    max_keyframes: int = 16
    keyframe_interval: int = 8
    keyframe_attempts: int = 3
    min_points: int = 8
    max_points: int = 80
    max_nfev: int = 80
    outlier_px: float = 3.
    min_inlier_fraction: float = .7
    max_rmse_px: float = 2.
    min_spread: float = .025
    patch_correlation: float = .75
    measurement_floor_deg: float = .3
    accel_noise_deg_s2: float = 240.
    innovation_gate_sigma: float = 5.
    max_prediction_s: float = .35
    max_angle_std_deg: float = 10.
    rotation_measurement_std_deg: float = 1.5
    max_off_mechanism_deg: float = 10.
    max_shell_roll_disagreement_deg: float = 3.

    def __post_init__(self):
        integer = {'max_keyframes', 'keyframe_interval', 'keyframe_attempts',
                   'min_points', 'max_points', 'max_nfev'}
        for field in fields(self):
            value = getattr(self, field.name)
            if (isinstance(value, (bool, np.bool_)) or not np.isfinite(value)
                    or value <= 0 or (field.name in integer and int(value) != value)):
                raise ValueError(f'fusion.{field.name} must be positive and finite'
                                 + (' and an integer' if field.name in integer else ''))
        if self.min_points < 3 or self.max_points < self.min_points:
            raise ValueError('fusion needs at least three points and max_points >= min_points')
        if self.min_inlier_fraction > 1 or self.patch_correlation > 1:
            raise ValueError('fusion fractions must be <= 1')

    @classmethod
    def from_mapping(cls, values=None):
        values=dict(values or {})
        unknown=set(values)-{f.name for f in fields(cls)}
        if unknown:
            raise ValueError('Unknown fusion settings: '+', '.join(sorted(unknown)))
        return cls(**values)


class SharedMotionFilter:
    """Six-state Kalman filter with correlated visual angle measurements.

    Missing data never update the filter. Reacquisition after the prediction
    horizon can restore orientation, but cannot establish missed whole turns.
    Covariance is conditional on calibration and mapped-point accuracy.
    """
    def __init__(self, initial_angles, timestamp, config=None):
        self.cfg = config or FusionConfig()
        q = np.asarray(initial_angles, float)
        if q.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(timestamp):
            raise ValueError('Need finite initial angles and timestamp')
        self.x = np.r_[q, np.zeros(3)]
        self.P = np.diag(np.r_[np.full(3, np.deg2rad(.3)**2),
                               np.full(3, np.deg2rad(180.)**2)])
        self.time = float(timestamp)
        self.last_visual = np.full(3, float(timestamp))
        self.turns_observed = np.ones(3, bool)

    def predict(self, timestamp):
        dt = float(timestamp)-self.time
        if not np.isfinite(dt) or dt < 0:
            raise ValueError('Fusion events must have finite, nondecreasing times')
        A = np.eye(6); A[:3, 3:] = np.eye(3)*dt
        G = np.vstack((np.eye(3)*(.5*dt*dt), np.eye(3)*dt))
        self.x = A @ self.x
        self.P = A @ self.P @ A.T + np.deg2rad(self.cfg.accel_noise_deg_s2)**2*(G @ G.T)
        self.time = float(timestamp)
        self.turns_observed &= self.time-self.last_visual <= self.cfg.max_prediction_s
        return self.x[:3].copy()

    def update(self, components, angles, covariance):
        ids = np.asarray(components, int)
        z, R = np.asarray(angles, float), np.asarray(covariance, float)
        if (ids.ndim != 1 or not len(ids) or len(set(ids)) != len(ids)
                or np.any((ids < 0) | (ids > 2)) or z.shape != ids.shape
                or R.shape != (len(ids), len(ids)) or not np.isfinite(z).all()
                or not np.isfinite(R).all() or not np.allclose(R, R.T)
                or np.min(np.linalg.eigvalsh(R)) <= 0):
            raise ValueError('Invalid visual measurement or covariance')
        H = np.eye(6)[ids]
        innovation = z-self.x[ids]
        S = H @ self.P @ H.T + R
        if float(innovation @ np.linalg.solve(S, innovation)) > self.cfg.innovation_gate_sigma**2*len(ids):
            return False
        stale = self.time-self.last_visual[ids] > self.cfg.max_prediction_s
        if stale.any():
            # Do not interpret a recovered orientation as a measured gap rate.
            velocities = ids[stale]+3
            self.x[velocities] = 0.
            self.P[velocities, :] = 0.; self.P[:, velocities] = 0.
            self.P[velocities, velocities] = np.deg2rad(180.)**2
            S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.x += K @ innovation
        A = np.eye(6)-K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        self.P = (self.P+self.P.T)*.5
        self.last_visual[ids] = self.time
        return True

    def snapshot(self, measured=()):
        std = np.sqrt(np.maximum(0, np.diag(self.P)[:3]))
        age = self.time-self.last_visual
        usable = (age <= self.cfg.max_prediction_s) & (std <= np.deg2rad(self.cfg.max_angle_std_deg))
        status = np.where(usable, 'predicted', 'unresolved')
        for component in measured:
            if usable[component]:
                status[component] = 'vision'
        return {'angles': np.where(usable, self.x[:3], np.nan),
                'angular_velocity': np.where(usable, self.x[3:], np.nan),
                'std_rad': std, 'age_s': age, 'status': status,
                'turn_count_valid': self.turns_observed.copy() & usable}


def visible_projection(camera, F, pivot, angles, shell, points, radius, gap, red_sign=1):
    xyz = world_points(F, pivot, angles, shell, points, radius, gap, red_sign)
    center = world_points(F, pivot, angles, shell, np.zeros_like(points), radius, gap, red_sign)
    eye = -camera['R'].T @ camera['t']
    depth = (xyz @ camera['R'].T + camera['t'])[:, 2]
    visible = (depth > 0) & (np.sum((xyz-center)*(eye-xyz), axis=1) > 0)
    return project(camera, xyz), visible


def fit_visual_pose(camera, F, pivot, radius, gap, points, shells, uv, seed,
                    config=None, red_sign=1, *, allow_partial=True):
    """Fit one native image against established landmarks, with no motion prior.

    Only observed shell spins participate. A camera may update common roll and
    one shell while the other view supplies the other shell at its next event.
    """
    cfg = config or FusionConfig()
    points, shells, uv, seed = map(np.asarray, (points, shells, uv, seed))
    present = [h for h in (0, 1) if np.count_nonzero(shells == h) >= cfg.min_points]
    selected = np.isin(shells, present)
    accepted = np.zeros(len(points), bool)
    if not present:
        return {'success': False, 'reason': 'too few mapped points', 'inlier': accepted}
    ids = np.array([0]+[1+h for h in present])

    def residual(values, mask=selected):
        q = seed.astype(float).copy(); q[ids] = values
        xyz = world_points(F, pivot, q, shells[mask], points[mask], radius, gap, red_sign)
        return (project(camera, xyz)-uv[mask]).ravel()

    fit = least_squares(residual, seed[ids], loss='soft_l1', f_scale=1.,
                        max_nfev=cfg.max_nfev, ftol=1e-7, xtol=1e-7, gtol=1e-7)
    q = seed.astype(float).copy(); q[ids] = fit.x
    prediction, facing = visible_projection(camera, F, pivot, q, shells, points, radius, gap, red_sign)
    errors = np.linalg.norm(prediction-uv, axis=1)
    accepted = selected & facing & (errors <= cfg.outlier_px)
    reason = None if fit.success else 'pixel optimizer did not converge'
    for h in present:
        candidate = selected & (shells == h); good = accepted & candidate
        spread = 0.
        if good.sum() >= cfg.min_points:
            p = points[good]-points[good].mean(axis=0)
            spread = np.linalg.svd(p, compute_uv=False)[1]/np.sqrt(good.sum())
        if (good.sum() < cfg.min_points or good.sum()/candidate.sum() < cfg.min_inlier_fraction
                or spread < cfg.min_spread):
            reason = 'insufficient inliers or spatial coverage'
    rmse = float(np.sqrt(np.mean(errors[accepted]**2))) if accepted.any() else None
    if rmse is None or rmse > cfg.max_rmse_px:
        reason = 'pixel RMSE exceeds limit'
    # Use the actual accepted pixel Jacobian, not a prior-regularized Jacobian.
    J = np.column_stack([(residual(fit.x+np.eye(len(ids))[i]*1e-6, accepted)
                           -residual(fit.x-np.eye(len(ids))[i]*1e-6, accepted))/2e-6
                          for i in range(len(ids))])
    singular = np.linalg.svd(J, compute_uv=False)
    if len(singular) != len(ids) or singular[-1] < 1e-3*singular[0]:
        reason = 'visual pose is rank deficient'
    covariance = (np.linalg.pinv(J.T @ J)*max(1., (rmse or 1.)**2)
                  + np.eye(len(ids))*np.deg2rad(cfg.measurement_floor_deg)**2)
    result = {'success': reason is None, 'reason': reason, 'angles': q,
            'components': ids, 'covariance': covariance, 'inlier': accepted,
            'observations': int(selected.sum()), 'inliers': int(accepted.sum()),
            'inlier_fraction': float(accepted.sum()/selected.sum()), 'rmse_px': rmse,
            'nfev': fit.nfev}
    if reason is not None and allow_partial and len(present) == 2:
        # Occlusion/blur of one shell must not erase real evidence for the
        # other. Retain exactly the components the successful subfit observes.
        candidates = []
        for h in present:
            use = shells == h
            candidate = fit_visual_pose(camera, F, pivot, radius, gap, points[use], shells[use], uv[use],
                                        seed, cfg, red_sign, allow_partial=False)
            if candidate['success']:
                inlier = np.zeros(len(points), bool); inlier[use] = candidate['inlier']
                candidate['inlier'] = inlier
                candidate['partial_shell_update'] = True
                candidates.append(candidate)
        if candidates:
            result = max(candidates, key=lambda item: item['inliers'])
    return result


@dataclass
class Keyframe:
    gray: np.ndarray
    uv: np.ndarray
    points: np.ndarray
    shells: np.ndarray
    identifiers: np.ndarray
    angles: np.ndarray
    index: int


def patch_similarity(first, second, uv0, uv1):
    a = cv2.getRectSubPix(first, (11, 11), tuple(map(float, uv0))).astype(float).ravel()
    b = cv2.getRectSubPix(second, (11, 11), tuple(map(float, uv1))).astype(float).ravel()
    a -= a.mean(); b -= b.mean()
    return float(a @ b/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-9))


class KeyframeView:
    """Bounded camera-local reference bank; unknown points cannot relocalize.

    All matches pass forward/backward LK and appearance checks, followed by a
    fixed-landmark geometric fit. Only visually accepted poses promote images.
    """
    def __init__(self, camera, F, pivot, radius, gap, tracking=None, config=None, red_sign=1):
        self.camera, self.F, self.pivot = camera, F, pivot
        self.radius, self.gap, self.red_sign = radius, gap, red_sign
        self.cfg = config or FusionConfig()
        self.klt = KLTConfig.from_mapping(tracking)
        self.bank = []; self.latest = None; self.next_id = 0

    def observe(self, gray, masks, seed):
        if self.latest is None:
            return None
        def distance(key):
            difference = np.arctan2(np.sin(key.angles-seed), np.cos(key.angles-seed))
            return float(np.linalg.norm(difference))
        keys = [self.latest]+sorted([k for k in self.bank if k is not self.latest], key=distance)[:self.cfg.keyframe_attempts]
        matches = {}; recovered = 0
        for key in keys:
            guesses, visible = visible_projection(self.camera, self.F, self.pivot, seed, key.shells,
                                                   key.points, self.radius, self.gap, self.red_sign)
            for h, name in enumerate(('top', 'bottom')):
                take = np.flatnonzero(visible & (key.shells == h)
                                      & ~np.isin(key.identifiers, list(matches)))
                if not len(take):
                    continue
                direct = track_features(key.gray, gray, key.uv[take], curr_mask=masks[name],
                                        config=self.klt, initial_uv_curr=guesses[take])
                for j, pixel in zip(direct.source_indices, direct.uv_curr):
                    i = take[j]
                    if patch_similarity(key.gray, gray, key.uv[i], pixel) < self.cfg.patch_correlation:
                        continue
                    # Multiple references cannot contribute the same physical
                    # pixel twice and inflate geometric support.
                    if any(np.linalg.norm(pixel-v[0]) < 3 for v in matches.values()):
                        continue
                    matches[int(key.identifiers[i])] = (pixel, key.points[i], h)
                    recovered += int(key is not self.latest)
        if not matches:
            return None
        identifiers = np.array(list(matches), int)
        uv, points, shells = zip(*matches.values())
        return {'uv': np.asarray(uv), 'points': np.asarray(points),
                'shells': np.asarray(shells), 'identifiers': identifiers,
                'keyframe_matches': recovered}

    def promote(self, gray, masks, angles, index, observation=None, inlier=None, components=(0, 1, 2)):
        """Call only for a known initial pose or an accepted visual pose."""
        rows = []
        for h, name in enumerate(('top', 'bottom')):
            if 0 not in components or 1+h not in components:
                continue
            mask = masks[name].astype(np.uint8).copy()*255
            if observation is not None:
                selected = np.flatnonzero(inlier & (observation['shells'] == h))
                for i in selected[:self.cfg.max_points]:
                    pixel = observation['uv'][i]
                    rows.append((pixel, observation['points'][i], h, observation['identifiers'][i]))
                    cv2.circle(mask, tuple(np.rint(pixel).astype(int)), int(self.klt.min_distance_px), 0, -1)
            remaining = self.cfg.max_points-sum(row[2] == h for row in rows)
            for uv in detect_features(gray, mask, self.klt)[:max(0, remaining)]:
                p = initialize_surface_point(self.camera, uv, self.F, self.pivot, angles,
                                             h, self.radius, self.gap, self.red_sign)
                predicted, facing = visible_projection(self.camera, self.F, self.pivot, angles,
                                                       np.array([h]), p[None], self.radius, self.gap, self.red_sign)
                if not facing[0] or np.linalg.norm(predicted[0]-uv) > 1.:
                    continue
                rows.append((uv, p, h, self.next_id)); self.next_id += 1
        if not rows:
            return
        uv, points, shells, identifiers = zip(*rows)
        key = Keyframe(gray.copy(), np.asarray(uv), np.asarray(points), np.asarray(shells),
                       np.asarray(identifiers), np.asarray(angles).copy(), index)
        self.latest = key
        if not self.bank or index-self.bank[-1].index >= self.cfg.keyframe_interval:
            self.bank.append(key)
            if len(self.bank) > self.cfg.max_keyframes:
                # Keep the initial home reference; bound memory for long clips.
                del self.bank[1]


class IncrementProposal:
    """Native rotations with accepted-image anchors and separate search guesses.

    `measurement` exposes only the temporal tracker's accepted image estimates.
    Its approximate sphere model is explicitly distinct from a metric landmark
    fit. Failed predictions may seed a search, but cannot become measurements.
    """
    def __init__(self, camera, F, circle, initial_angles, tracking=None, temporal=None):
        self.camera, self.F = camera, F
        self.center, _ = sphere_pose_from_circle(*circle, camera['K'])
        self.tracker = PersistentKLTTracker(KLTConfig.from_mapping(tracking))
        self.previous = self.masks = None
        self.q = np.asarray(initial_angles, float).copy()
        self.initial_angles = self.q.copy()
        klt = KLTConfig.from_mapping(tracking)
        options = TemporalConfig.from_mapping({'enabled': True, 'motion_recovery_enabled': True,
                                               **dict(temporal or {})})
        self.temporal = [TemporalRotationTracker(camera['K'], self.center, 1., circle[2], klt,
                                                 {'ransac_iters': 80, 'min_inliers': options.min_inliers},
                                                 options, seed=7+h) for h in (0, 1)]
        self.min_inliers = options.min_inliers
        self.valid = np.zeros(2, bool); self.records = [None, None]
        self.gamma = np.zeros(2)
        self.last_matches = None
        self.last_increments = [None, None]
        self.anchor(self.q)

    def anchor(self, q):
        self.q = np.asarray(q, float).copy()
        self.poses = [self.F @ rotation_x(q[0]) @ rotation_z(q[1+h]) for h in (0, 1)]

    def observe(self, frame, masks, index, timestamp=0.):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self.previous is None:
            self.tracker.initialize(frame, masks)
            for h, name in enumerate(('top', 'bottom')):
                uv, ids = self.tracker.points(name)
                self.temporal[h].initialize(gray, masks[name], uv, ids, timestamp_s=timestamp)
                self.valid[h] = True
        else:
            matches = self.tracker.track_pair(self.previous, frame, self.masks, masks)
            self.last_matches = matches
            self.last_increments = [None, None]
            rolls = []
            for h, name in enumerate(('top', 'bottom')):
                m = matches[name]
                result = solve_hemisphere_increment(m.uv_prev, m.uv_curr, self.camera['K'],
                                                    self.center, ransac_iters=80, min_inliers=self.min_inliers, rng=7+index)
                self.last_increments[h] = result.R if result.success else None
                uv, ids = self.tracker.points(name)
                pose, valid, record = self.temporal[h].update(index, gray, masks[name], m, result,
                                                            uv, ids, timestamp_s=timestamp)
                self.valid[h] = valid; self.records[h] = record
                R = self.camera['R']
                # The temporal pose is an observed orientation relative to the
                # original camera frame. Failed predictions only seed searches.
                proposal = pose if valid else self.temporal[h].prediction
                world = R.T @ proposal @ R @ self.F @ rotation_x(self.initial_angles[0])
                self.poses[h] = world
                euler = Rotation.from_matrix(self.F.T @ world).as_euler('XYZ')
                self.gamma[h] = euler[1]
                if valid:
                    rolls.append(euler[0])
                self.q[1+h] += np.arctan2(np.sin(euler[2]-self.q[1+h]), np.cos(euler[2]-self.q[1+h]))
            if rolls:
                a = np.arctan2(np.sin(rolls).sum(), np.cos(rolls).sum())
                self.q[0] += np.arctan2(np.sin(a-self.q[0]), np.cos(a-self.q[0]))
        self.previous, self.masks = frame, masks
        return self.q.copy()

    def measurement(self, predicted, max_gamma_deg=10., max_roll_disagreement_deg=3., std_deg=1.5):
        eligible = self.valid & (abs(np.rad2deg(self.gamma)) <= max_gamma_deg)
        shells = np.flatnonzero(eligible)
        if not len(shells):
            return None
        rolls = np.array([Rotation.from_matrix(self.F.T @ self.poses[h]).as_euler('XYZ')[0] for h in shells])
        if len(rolls) == 2 and abs(np.rad2deg(np.arctan2(np.sin(rolls[0]-rolls[1]), np.cos(rolls[0]-rolls[1])))) > max_roll_disagreement_deg:
            return None
        # Select the locally continuous branch for orientation. The filter
        # separately invalidates whole-turn counts after an unobserved gap.
        q = self.q.copy()
        q[0] = np.arctan2(np.sin(rolls).sum(), np.cos(rolls).sum())
        q = predicted + np.arctan2(np.sin(q-predicted), np.cos(q-predicted))
        ids = np.array([0]+[1+int(h) for h in shells])
        return {'success': True, 'angles': q, 'components': ids,
                'covariance': np.eye(len(ids))*np.deg2rad(std_deg)**2,
                'model': 'enclosing_sphere_rotation', 'gamma_deg': np.rad2deg(self.gamma),
                'tracking_status': [None if r is None else r['status'] for r in self.records]}
