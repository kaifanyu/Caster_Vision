"""Native-time metric reconstruction of two independently spinning hemispheres.

Every pixel observation is evaluated at its own camera timestamp. Both cameras
constrain one continuous angle trajectory and camera-local material landmarks;
there is no requirement to match paint features between cameras. The offline
bundle solve uses an analytic sparse Jacobian and the physical separated centers.
"""
from __future__ import annotations

from collections import Counter
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

from .fusion import FusionConfig, SharedMotionFilter, fit_visual_pose
from .model import initialize_surface_point, points_to_polar, polar_to_points


def event_table(data):
    return sorted((float(t), ci, fi) for ci, times in enumerate(data['times'])
                  for fi, t in enumerate(times))


def seed_trajectory(data, cameras, F, pivot, geometry, initial_roll_deg, progress=print,
                    *, offline_hypotheses=False):
    """Incremental metric landmark fit used only to initialize the batch solve.

    Newly visible points enter the map only following a supported image fit.
    No enclosing-sphere rotations enter this estimator. With offline_hypotheses,
    an unanchored adjacent-frame fit may initialize an optimizer hypothesis.
    Such hypotheses are not final measurements: accepted pixel connectivity and
    residuals must be re-established by the full batch solve.
    """
    obs = data['observations']; events = event_table(data)
    options = FusionConfig(min_points=6, min_inlier_fraction=.5, outlier_px=4.,
                           max_rmse_px=2.8, measurement_floor_deg=.35,
                           max_prediction_s=.25, innovation_gate_sigma=12.,
                           accel_noise_deg_s2=2000. if offline_hypotheses else 240.)
    # These looser gates supply a nonlinear optimizer's starting guess only.
    # Final pixel inliers still require the strict three-pixel batch test.
    hypothesis_options = FusionConfig(min_points=6, min_inlier_fraction=.35,
        outlier_px=8., max_rmse_px=5., measurement_floor_deg=.5, min_spread=.015)
    state = SharedMotionFilter(np.deg2rad([initial_roll_deg, 0., 0.]), events[0][0], options)
    radius, gap, sign = geometry['radius_m'], geometry['gap_m'], geometry.get('red_shell_sign', 1)
    keys = np.column_stack([obs[k] for k in ('camera', 'shell', 'track')])
    unique, landmark = np.unique(keys, axis=0, return_inverse=True)
    points = np.full((len(unique), 3), np.nan)
    groups = {(ci, fi): np.flatnonzero((obs['camera'] == ci) & (obs['frame'] == fi))
              for _, ci, fi in events}
    poses, accepted, quality = [], [], []
    counters = [Counter(), Counter()]
    last_camera_support = np.full((2, 3), -np.inf)
    previous_rows = [None, None]; previous_pose = [None, None]; previous_time = [-np.inf, -np.inf]
    previous_support = [set(), set()]
    for ei, (time, ci, fi) in enumerate(events):
        prediction = state.predict(time)
        rows = groups[ci, fi]; ids = landmark[rows]
        mapped = np.isfinite(points[ids]).all(axis=1)
        fit = None; supported = []
        if fi == 0:
            # Both streams start within a fraction of a frame of the same pose.
            q = np.deg2rad([initial_roll_deg, 0., 0.]); supported = [0, 1, 2]
        elif mapped.sum() >= options.min_points:
            take = rows[mapped]
            fit = fit_visual_pose(cameras[ci], F, pivot, radius, gap,
                                  points[landmark[take]], obs['shell'][take], obs['uv'][take],
                                  prediction, options, sign)
            q = prediction
        else:
            q = prediction; counters[ci]['too_few_mapped_points'] += 1
        if fi > 0:
            # Re-linearize surviving adjacent IMAGE correspondences on the true
            # moving shell centers when an old fixed map ceases to fit. This is
            # measured relative motion, not a prediction-only relocalization.
            # The past pose must still be anchored by recent visual evidence.
            if (offline_hypotheses or fit is None or not fit['success']) and previous_rows[ci] is not None:
                old = {int(landmark[i]): i for i in previous_rows[ci]}
                current, prior, local = [], [], []
                for i in rows:
                    h = int(obs['shell'][i]); key = int(landmark[i])
                    evidence_time = max(last_camera_support[:, h+1].max(), -np.inf)
                    anchored_previous = 0 in previous_support[ci] and h+1 in previous_support[ci]
                    if key not in old or (not offline_hypotheses and
                        (not anchored_previous or previous_time[ci]-evidence_time > .1)):
                        continue
                    j = old[key]
                    p = initialize_surface_point(cameras[ci], obs['uv'][j], F, pivot,
                                                 previous_pose[ci], h, radius, gap, sign)
                    current.append(i); prior.append(j); local.append(p)
                if len(current) >= options.min_points:
                    current = np.asarray(current, int)
                    incremental = fit_visual_pose(cameras[ci], F, pivot, radius, gap,
                                                  np.asarray(local), obs['shell'][current], obs['uv'][current],
                                                  previous_pose[ci] if offline_hypotheses else prediction,
                                                  hypothesis_options if offline_hypotheses else options, sign)
                    if incremental['success']:
                        fit = incremental; counters[ci]['metric_adjacent_recovery'] += 1
                        if offline_hypotheses and not previous_support[ci]:
                            counters[ci]['unanchored_optimizer_hypotheses'] += 1
            if fit is not None and fit['success']:
                components = fit['components']
                if state.update(components, fit['angles'][components], fit['covariance']):
                    supported = list(components); counters[ci]['visual_updates'] += 1
                else:
                    counters[ci]['innovation_rejections'] += 1
            elif fit is not None:
                counters[ci][fit['reason']] += 1
            q = state.x[:3].copy()
        last_camera_support[ci, supported] = time
        # A camera that loses all paint tracks can rebuild its local map from
        # a fresh observation of the SAME components in the other view. This
        # creates an initialization, not another measurement of that pose.
        shared_support = time-last_camera_support[1-ci] <= .07
        promotion = set(supported)
        if shared_support[0]:
            for component in (1, 2):
                if shared_support[component]: promotion.update((0, component))
        # Keep old landmarks fixed here. Their locations are refined jointly
        # from every observation in the offline bundle adjustment below.
        for i in rows:
            h = int(obs['shell'][i]); l = landmark[i]
            if np.isfinite(points[l]).all() or 0 not in promotion or h+1 not in promotion:
                continue
            points[l] = initialize_surface_point(cameras[ci], obs['uv'][i], F, pivot, q,
                                                 h, radius, gap, sign)
            if h+1 not in supported: counters[ci]['landmarks_seeded_from_other_camera'] += 1
        poses.append(q); accepted.append(supported)
        previous_rows[ci], previous_pose[ci], previous_time[ci] = rows, q.copy(), time
        previous_support[ci] = set(supported)
        quality.append(None if fit is None else {k: fit[k] for k in
                       ('success', 'reason', 'inliers', 'rmse_px') if k in fit})
        if progress and (ei+1) % 150 == 0:
            progress(f'Metric seed {ei+1}/{len(events)} images; accepted '
                     f'cam0={counters[0]["visual_updates"]}, cam1={counters[1]["visual_updates"]}', flush=True)
    if offline_hypotheses:
        # A poor causal initializer must not delete valid future observations
        # from an offline problem. Give remaining tracks nuisance-point seeds;
        # only the batch residual/support graph can accept their final poses.
        lookup = {(ci, fi): ei for ei, (_, ci, fi) in enumerate(events)}
        for i, l in enumerate(landmark):
            if not np.isfinite(points[l]).all():
                ci, h, fi = (int(obs[k][i]) for k in ('camera', 'shell', 'frame'))
                points[l] = initialize_surface_point(cameras[ci], obs['uv'][i], F, pivot,
                    poses[lookup[ci, fi]], h, radius, gap, sign)
    return {'events': events, 'angles': np.asarray(poses), 'components': accepted,
            'landmark': landmark, 'keys': unique, 'points': points, 'quality': quality,
            'per_camera': counters}


class NativeBundle:
    """Analytic sparse reprojection problem on an irregular time grid.

    Columns contain angle knots followed by two spherical coordinates per
    feature. Red/green phase at the first knot is the only fixed angle gauge.
    Initial roll has a finite prior, so an approximate 8-degree start can improve.
    """
    def __init__(self, data, seed, cameras, F, pivot, geometry, *, knot_hz=30.,
                 acceleration_std_deg_s2=500., initial_std_deg=3., mode='motion'):
        if mode not in ('motion', 'roll', 'swivel'):
            raise ValueError('mode must be motion, roll or swivel')
        self.mode = mode; self.F = np.asarray(F); self.pivot = np.asarray(pivot)
        self.cameras = cameras; self.geometry = geometry
        events = seed['events']; self.event_times = np.asarray([e[0] for e in events])
        if not len(events) or np.any(np.diff(self.event_times) < 0):
            raise ValueError('Need ordered native events')
        self.knots = np.linspace(self.event_times[0], self.event_times[-1],
                                 max(3, int(np.ceil((self.event_times[-1]-self.event_times[0])*knot_hz))+1))
        self.q0 = np.column_stack([np.interp(self.knots, self.event_times, seed['angles'][:, k]) for k in range(3)])
        self.q0[0, 1:] = 0.
        if mode == 'roll': self.q0[:, 1:] = 0.
        if mode == 'swivel': self.q0[:, 0] = self.q0[0, 0]
        original = data['observations']
        keep = np.isfinite(seed['points'][seed['landmark']]).all(axis=1)
        counts = np.bincount(seed['landmark'], minlength=len(seed['points']))
        keep &= counts[seed['landmark']] >= 3
        if keep.sum() < 20:
            raise ValueError('Too few initialized metric observations for offline refinement')
        self.original_rows = np.flatnonzero(keep)
        self.obs = {k: np.asarray(v)[keep] for k, v in original.items()}
        used, self.landmark = np.unique(seed['landmark'][keep], return_inverse=True)
        self.keys = seed['keys'][used]
        self.signs = geometry.get('red_shell_sign', 1)*(1-2*self.keys[:, 1])
        self.polar0 = points_to_polar(seed['points'][used])
        self.polar0[:, 1] = np.clip(self.polar0[:, 1], 1e-5, np.pi/2-1e-5)
        self.times = np.asarray([data['times'][ci][fi] for ci, fi in zip(self.obs['camera'], self.obs['frame'])])
        self.left, self.fraction = self.interpolation(self.times)
        active = np.ones_like(self.q0, bool); active[0] = False
        if mode == 'roll': active[:, 1:] = False
        if mode == 'swivel': active[:, 0] = False
        self.active = active; self.angle_ids = np.full(active.shape, -1, int)
        self.angle_ids[active] = np.arange(active.sum())
        self.nactive = int(active.sum())
        self.initial_id = self.nactive
        self.timing_id = self.nactive+1
        self.nangle = self.nactive+2
        # A global initial-roll coordinate lets the solver correct a common
        # pose bias without needing hundreds of correlated knot steps.
        self.x0 = np.r_[self.q0[active], 0., 0., self.polar0.ravel()]
        self.x0[self.timing_id] = float(seed.get('initial_brio_offset_s', 0.))
        self.lower = np.full(len(self.x0), -np.inf); self.upper = -self.lower
        self.lower[self.nangle+1::2] = 1e-7
        self.upper[self.nangle+1::2] = np.pi/2-1e-7
        self.lower[self.initial_id], self.upper[self.initial_id] = np.deg2rad([-25., 25.])
        self.lower[self.timing_id], self.upper[self.timing_id] = -.08, .08
        self.sqrt_weight = np.sqrt(self.obs.get('weight', np.ones(len(self.times))))
        self.accel_std = np.deg2rad(acceleration_std_deg_s2)
        self.initial_std = np.deg2rad(initial_std_deg)
        self.initial_alpha = float(seed.get('initial_roll_prior_rad', self.q0[0, 0]))
        self.matrix = np.asarray([c['K'] @ c['R'] @ self.F for c in cameras])[self.obs['camera']]
        self.shift = np.asarray([c['K'] @ (c['R'] @ self.pivot+c['t']) for c in cameras])[self.obs['camera']]
        self.pixel_scale = np.ones(len(self.times))

    def interpolation(self, times):
        left = np.clip(np.searchsorted(self.knots, times, side='right')-1, 0, len(self.knots)-2)
        fraction = (times-self.knots[left])/(self.knots[left+1]-self.knots[left])
        return left, np.clip(fraction, 0., 1.)

    def unpack(self, x):
        q = self.q0.copy(); q[self.active] = x[:self.nactive]
        q[:, 0] += x[self.initial_id]
        return q, x[self.nangle:].reshape(-1, 2)

    def observation_interpolation(self, x):
        return self.interpolation(self.times+x[self.timing_id]*(self.obs['camera'] == 1))

    def angles_at(self, x, times):
        q, _ = self.unpack(x); left, fraction = self.interpolation(np.asarray(times))
        return q[left]*(1-fraction[:, None])+q[left+1]*fraction[:, None]

    def pixels(self, x, jacobian=False):
        q, polar = self.unpack(x); left, f = self.observation_interpolation(x)
        angles = q[left]*(1-f[:, None])+q[left+1]*f[:, None]
        shell = self.obs['shell']; sign = self.signs[self.landmark]
        az, theta = polar[self.landmark].T
        spin = az+angles[np.arange(len(angles)), 1+shell]
        cp, sp, ct, st = np.cos(spin), np.sin(spin), np.cos(theta), np.sin(theta)
        ca, sa = np.cos(angles[:, 0]), np.sin(angles[:, 0])
        radius, gap = self.geometry['radius_m'], self.geometry['gap_m']
        px, py, pz = radius*cp*st, radius*sp*st, sign*(radius*ct+gap/2)
        local = np.column_stack((px, ca*py-sa*pz, sa*py+ca*pz))
        homogeneous = np.einsum('nij,nj->ni', self.matrix, local)+self.shift
        depth = homogeneous[:, 2]
        depth = np.where(abs(depth) < 1e-9, 1e-9, depth)
        prediction = homogeneous[:, :2]/depth[:, None]
        error = prediction-self.obs['uv']
        if not jacobian: return error, local, homogeneous[:, 2]
        # d(local)/d(alpha, beta=azimuth, colatitude)
        deriv = np.stack((np.column_stack((np.zeros(len(px)), -local[:, 2], local[:, 1])),
                          np.column_stack((-py, ca*px, sa*px)),
                          np.column_stack((radius*cp*ct, ca*radius*sp*ct+sa*sign*radius*st,
                                           sa*radius*sp*ct-ca*sign*radius*st))), axis=2)
        dh = self.matrix @ deriv
        J = (dh[:, :2]-prediction[:, :, None]*dh[:, 2, None, :])/depth[:, None, None]
        return error, J

    def evaluate(self, x, jacobian=False):
        q, _ = self.unpack(x); m = len(self.times)
        weight = self.sqrt_weight*self.pixel_scale
        error, pixel_jac = self.pixels(x, True)
        dt0 = np.diff(self.knots)[:-1]; dt1 = np.diff(self.knots)[1:]
        duration = .5*(dt0+dt1)
        coeff = np.column_stack((1/dt0, -(1/dt0+1/dt1), 1/dt1))/(duration*self.accel_std)[:, None]
        accel = coeff[:, 0, None]*q[:-2]+coeff[:, 1, None]*q[1:-1]+coeff[:, 2, None]*q[2:]
        prior = (q[0, 0]-self.initial_alpha)/self.initial_std
        residual = np.r_[(error*weight[:, None]).ravel(), accel.ravel(), prior, x[self.timing_id]/.05]
        if not jacobian: return residual
        rows, cols, values = [], [], []
        rr = np.arange(2*m).reshape(m, 2)
        def add(r, c, v):
            r, c, v = np.broadcast_arrays(r, c, v); valid = c >= 0
            rows.append(r[valid]); cols.append(c[valid]); values.append(v[valid])
        left, f = self.observation_interpolation(x)
        for knot, fraction in ((left, 1-f), (left+1, f)):
            for derivative, component in ((0, np.zeros(m, int)), (1, 1+self.obs['shell'])):
                add(rr, self.angle_ids[knot, component, None],
                    pixel_jac[:, :, derivative]*(weight*fraction)[:, None])
        for k in (0, 1):
            add(rr, (self.nangle+2*self.landmark+k)[:, None], pixel_jac[:, :, 1+k]*weight[:, None])
        add(rr, self.initial_id, pixel_jac[:, :, 0]*weight[:, None])
        velocity = (q[left+1]-q[left])/(self.knots[left+1]-self.knots[left])[:, None]
        corrected_times = self.times+x[self.timing_id]*(self.obs['camera'] == 1)
        timing_weight = weight*(self.obs['camera'] == 1)*(corrected_times > self.knots[0])*(corrected_times < self.knots[-1])
        jt = (pixel_jac[:, :, 0]*velocity[:, 0, None]+pixel_jac[:, :, 1]*velocity[np.arange(m), 1+self.obs['shell'], None])
        add(rr, self.timing_id, jt*timing_weight[:, None])
        ar = np.arange(accel.size).reshape(-1, 3)+2*m
        for k in range(3):
            add(ar, self.angle_ids[k:k+len(accel)], coeff[:, k, None])
        add(np.array([len(residual)-2]), np.array([self.initial_id]), np.array([1/self.initial_std]))
        add(np.array([len(residual)-1]), np.array([self.timing_id]), np.array([1/.05]))
        return coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(residual), len(x))).tocsr()

    def visibility(self, x):
        _, local, depth = self.pixels(x)
        q = self.angles_at(x, self.times+x[self.timing_id]*(self.obs['camera'] == 1))
        offset = self.signs[self.landmark]*self.geometry['gap_m']/2
        centers_local = np.column_stack((np.zeros(len(q)), -np.sin(q[:, 0])*offset, np.cos(q[:, 0])*offset))
        xyz = local @ self.F.T+self.pivot
        centers = centers_local @ self.F.T+self.pivot
        eyes = np.asarray([-c['R'].T @ c['t'] for c in self.cameras])[self.obs['camera']]
        return (depth > 0) & (np.sum((xyz-centers)*(eyes-xyz), axis=1) > 0)


def refine_native(data, seed, cameras, F, pivot, geometry, *, max_nfev=120, progress=print, mode='motion'):
    problem = NativeBundle(data, seed, cameras, F, pivot, geometry, mode=mode)
    x = problem.x0.copy(); stages = []
    # Radians have physical, comparable scales. Inverse-Jacobian scaling makes
    # nearly polar/short surface tracks take enormous steps and stalls the
    # common pose (also observed in the original axis-calibration solver).
    parameter_scale = np.ones(len(x)); parameter_scale[problem.timing_id] = .03
    for stage in range(2):
        if progress:
            progress(f'Joint metric bundle {stage+1}/2: {len(problem.times)} observations, '
                     f'{len(problem.knots)} trajectory knots, {len(problem.keys)} landmarks', flush=True)
        result = least_squares(problem.evaluate, x, jac=lambda x: problem.evaluate(x, True),
                               bounds=(problem.lower, problem.upper), loss='soft_l1', f_scale=1.5,
                               x_scale=parameter_scale, max_nfev=max_nfev, ftol=2e-5, xtol=1e-7, gtol=1e-6,
                               tr_options={'atol': 1e-6, 'btol': 1e-6, 'maxiter': 300})
        x = result.x
        error = np.linalg.norm(problem.pixels(x)[0], axis=1)
        visible = problem.visibility(x)
        stages.append({'nfev': result.nfev, 'cost': result.cost, 'converged': result.success,
                       'message': result.message, 'optimality': result.optimality,
                       'median_error_px': float(np.median(error)),
                       'initial_roll_deg': float(np.rad2deg(problem.unpack(x)[0][0, 0])),
                       'additional_brio_offset_s': float(x[problem.timing_id]),
                       'fraction_within_3px': float(np.mean((error < 3)&visible))})
        if progress: progress(str(stages[-1]), flush=True)
        # Gross mistracks must not pull the final trajectory. Keep a small
        # robust weight so a feature can recover without changing residual size.
        problem.pixel_scale = np.where((error <= 5.) & visible, 1., .03)
    return problem, x, stages
