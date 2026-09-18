"""Conditional shell-spin initialization from native material-track increments.

Given a refined common roll, ray/sphere intersections supply each material
point's azimuth in the unspun caster frame. Differences along a camera-local
track eliminate its unknown material phase. These are optimizer hypotheses;
the subsequent full image-space bundle fit decides final support and accuracy.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from .model import initialize_surface_point


def _interpolation(knots, times):
    left = np.clip(np.searchsorted(knots, times, side='right')-1, 0, len(knots)-2)
    fraction = np.clip((times-knots[left])/(knots[left+1]-knots[left]), 0., 1.)
    return left, fraction


def _unspun_normals(data, cameras, F, pivot, geometry, knots, angles, offset):
    obs = data['observations']
    camera_id = np.asarray(obs['camera'], int)
    shell = np.asarray(obs['shell'], int)
    times = np.array([data['times'][c][f] for c, f in zip(camera_id, obs['frame'])])
    times += offset*(camera_id == 1)
    alpha = np.interp(times, knots, angles[:, 0])
    ca, sa = np.cos(alpha), np.sin(alpha)
    sign = geometry.get('red_shell_sign', 1)*(1-2*shell)
    radius, gap = geometry['radius_m'], geometry['gap_m']
    offset_local = np.column_stack((np.zeros(len(times)), -sa*sign*gap/2, ca*sign*gap/2))
    centers = offset_local @ F.T+pivot
    normals = np.full((len(times), 3), np.nan)
    incidence = np.zeros(len(times))
    hit = np.zeros(len(times), bool)
    for ci, camera in enumerate(cameras):
        rows = np.flatnonzero(camera_id == ci)
        rays = np.column_stack((obs['uv'][rows], np.ones(len(rows)))) @ np.linalg.inv(camera['K']).T
        rays = rays @ np.asarray(camera['R'])
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        eye = -np.asarray(camera['R']).T @ np.asarray(camera['t'])
        delta = eye-centers[rows]
        b = np.einsum('ij,ij->i', delta, rays)
        discriminant = b*b-np.einsum('ij,ij->i', delta, delta)+radius*radius
        distance = -b-np.sqrt(np.maximum(discriminant, 0.))
        ok = (discriminant >= 0) & (distance > 0)
        world_normal = (eye+distance[:, None]*rays-centers[rows])/radius
        home_normal = world_normal @ F
        # Row-vector inverse of Rx(alpha): q = Rx(alpha).T @ n_home.
        local = np.column_stack((home_normal[:, 0],
                                 ca[rows]*home_normal[:, 1]+sa[rows]*home_normal[:, 2],
                                 -sa[rows]*home_normal[:, 1]+ca[rows]*home_normal[:, 2]))
        normals[rows] = local
        incidence[rows] = -np.einsum('ij,ij->i', world_normal, rays)
        hit[rows] = ok
    xy = np.linalg.norm(normals[:, :2], axis=1)
    signed = sign*normals[:, 2]
    valid = (hit & np.isfinite(normals).all(axis=1) & (signed >= 0.)
             & (xy > .2) & (incidence > .25)
             & (times >= knots[0]) & (times <= knots[-1]))
    phi = np.arctan2(normals[:, 1], normals[:, 0])
    theta = np.arccos(np.clip(signed, -1., 1.))
    return times, normals, phi, theta, incidence, xy, valid


def _spin_curve(knots, time0, time1, delta, confidence):
    """Robust sparse relative-angle fit with beta at the first knot fixed zero."""
    n = len(knots)
    if not len(delta):
        return np.zeros(n), {'edges': 0, 'supported': False, 'iterations': 0,
                             'note': 'No pixel-derived spin edges; zero is an unobserved optimizer seed.'}, np.empty(0)
    left0, f0 = _interpolation(knots, time0)
    left1, f1 = _interpolation(knots, time1)
    rows = np.repeat(np.arange(len(delta)), 4)
    cols = np.column_stack((left0, left0+1, left1, left1+1)).ravel()
    values = np.column_stack((-(1-f0), -f0, 1-f1, f1)).ravel()
    A = coo_matrix((values, (rows, cols)), shape=(len(delta), n)).tocsr()[:, 1:]
    dt0, dt1 = np.diff(knots)[:-1], np.diff(knots)[1:]
    acceleration_std = np.deg2rad(3000.)
    coefficients = np.column_stack((1/dt0, -(1/dt0+1/dt1), 1/dt1))
    coefficients /= (.5*(dt0+dt1)*acceleration_std)[:, None]
    D = coo_matrix((coefficients.ravel(),
                    (np.repeat(np.arange(n-2), 3),
                     np.column_stack((np.arange(n-2), np.arange(1, n-1), np.arange(2, n))).ravel())),
                   shape=(n-2, n)).tocsr()[:, 1:]
    # A tiny ridge regularizes entirely unobserved sections without pretending
    # they contain measurements. Final pixel support must establish validity.
    regularization = D.T @ D+diags(np.full(n-1, 1e-9))
    base = np.maximum(np.asarray(confidence), .01)/np.deg2rad(.5)**2
    robust = np.ones(len(delta)); x = np.zeros(n-1)
    for iteration in range(12):
        weight = base*robust
        weighted = A.multiply(weight[:, None])
        new_x = spsolve((A.T @ weighted+regularization).tocsc(), A.T @ (weight*delta))
        residual = A @ new_x-delta
        centered = residual-np.median(residual)
        scale = max(np.deg2rad(.25), 1.4826*float(np.median(abs(centered))))
        cutoff = max(np.deg2rad(.75), 2.5*scale)
        new_robust = np.minimum(1., cutoff/np.maximum(abs(residual), 1e-12))
        change = float(np.max(abs(new_x-x)))
        x, robust = new_x, new_robust
        if iteration >= 2 and change < 1e-5:
            break
    residual = A @ x-delta
    report = {'edges': len(delta), 'supported': True, 'iterations': iteration+1,
              'median_edge_error_deg': float(np.rad2deg(np.median(abs(residual)))),
              'p95_edge_error_deg': float(np.rad2deg(np.percentile(abs(residual), 95))),
              'edge_fraction_within_2deg': float(np.mean(abs(residual) < np.deg2rad(2.))),
              'start_spin_deg': 0., 'end_spin_deg': float(np.rad2deg(x[-1])),
              'acceleration_prior_std_deg_s2': 3000.}
    return np.r_[0., x], report, residual


def _surface_support(targets, times, rows, landmark, normals, max_distance_s=.05):
    """Count distinct observed tracks and their 2-D normal spread near times."""
    rows = np.asarray(rows, int)
    rows = rows[np.argsort(times[rows])]
    observed_times = times[rows]
    counts = np.zeros(len(targets), int)
    spread = np.zeros(len(targets))
    camera_support_rows = []
    for i, time in enumerate(targets):
        lo, hi = np.searchsorted(observed_times, [time-max_distance_s, time+max_distance_s])
        nearby = rows[lo:hi]
        if not len(nearby):
            camera_support_rows.append(np.empty(0, int))
            continue
        # The nearest observation represents each independent material track;
        # repeated frames of one corner cannot inflate support or spatial rank.
        nearest = nearby[np.argsort(abs(times[nearby]-time))]
        _, first = np.unique(landmark[nearest], return_index=True)
        selected = nearest[first]
        counts[i] = len(selected)
        if len(selected) >= 3:
            xy = normals[selected, :2]
            singular = np.linalg.svd(xy-xy.mean(axis=0), compute_uv=False)
            spread[i] = singular[1]/np.sqrt(len(selected))
        camera_support_rows.append(selected)
    return counts, spread, camera_support_rows


def _support_gaps(times, supported):
    missing = np.flatnonzero(~supported)
    if not len(missing):
        return []
    groups = np.split(missing, np.flatnonzero(np.diff(missing) > 1)+1)
    result = []
    for group in groups:
        lo, hi = int(group[0]), int(group[-1])
        before = float(times[lo-1]) if lo else None
        after = float(times[hi+1]) if hi+1 < len(times) else None
        result.append({'first_unresolved_s': float(times[lo]), 'last_unresolved_s': float(times[hi]),
                       'preceding_support_s': before, 'next_support_s': after,
                       'bracket_duration_s': after-before if before is not None and after is not None else None,
                       'unresolved_events': len(group)})
    return result


def conditional_spin_evidence(knots, curves, events, obs, observation_times,
                              normals, landmark, first, second, edge_residual,
                              confidence, *, min_tracks=6, min_spread=.025,
                              max_support_distance_s=.05, max_turn_gap_s=.15):
    """Strict component evidence from accepted relative azimuth observations.

    An accepted edge has geometric/ray gates already applied by the caller and
    at most two degrees of angular residual. Its nonzero interpolation nodes
    connect a phase graph only where nearby *distinct* observed tracks have
    sufficient normal-plane spread. Acceleration/ridge priors never create
    graph edges. A later disconnected segment therefore retains unknown phase.
    """
    knots, curves = np.asarray(knots), np.asarray(curves)
    first, second = np.asarray(first, int), np.asarray(second, int)
    edge_residual = np.asarray(edge_residual)
    accepted = (np.isfinite(edge_residual) & (abs(edge_residual) <= np.deg2rad(2.))
                & np.isfinite(confidence) & (np.asarray(confidence) > 0))
    first, second = first[accepted], second[accepted]
    residual = edge_residual[accepted]
    times = np.asarray(observation_times)
    event_times = np.array([event[0] for event in events], float)
    event_counts = np.zeros((len(events), 2), int)
    event_spread = np.zeros((len(events), 2))
    event_connected = np.zeros((len(events), 2), bool)
    event_supported = np.zeros((len(events), 2), bool)
    event_locally_supported = np.zeros((len(events), 2), bool)
    event_cameras = np.zeros((len(events), 2, 2), bool)
    knot_supported = np.zeros((len(knots), 2), bool)
    knot_connected = np.zeros((len(knots), 2), bool)
    edge_graph_connected = np.zeros(len(first), bool)
    measurement_design = {}
    for shell in (0, 1):
        selected = np.flatnonzero(obs['shell'][first] == shell)
        rows = np.unique(np.r_[first[selected], second[selected]])
        count, spread, _ = _surface_support(knots, times, rows, landmark, normals,
                                           max_support_distance_s)
        quality = (count >= min_tracks) & (spread >= min_spread)
        knot_supported[:, shell] = quality
        parent = np.arange(len(knots))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return int(i)

        left0, f0 = _interpolation(knots, times[first[selected]])
        left1, f1 = _interpolation(knots, times[second[selected]])
        graph_nodes = []
        observed_nodes = np.zeros(len(knots), bool)
        for j in range(len(selected)):
            coefficients = {}
            for node, value in ((left0[j], -(1-f0[j])), (left0[j]+1, -f0[j]),
                                (left1[j], 1-f1[j]), (left1[j]+1, f1[j])):
                coefficients[int(node)] = coefficients.get(int(node), 0.)+value
            nodes = [node for node, value in coefficients.items() if abs(value) > 1e-9]
            if len(nodes) < 2 or not quality[nodes].all():
                graph_nodes.append([])
                continue
            anchor = root(nodes[0])
            for node in nodes[1:]:
                parent[root(node)] = anchor
            observed_nodes[nodes] = True
            graph_nodes.append(nodes)
        connected = np.zeros(len(knots), bool)
        if quality[0] and observed_nodes[0]:
            anchor = root(0)
            connected = np.array([observed_nodes[i] and root(i) == anchor for i in range(len(knots))])
        # Shared interpolation nodes alone are not a rank proof: repeated
        # tracks at identical two times supply only one temporal equation.
        # Audit the deduplicated measurement design of the anchored component,
        # with beta[0] fixed, and exclude every regularization equation.
        columns = np.flatnonzero(connected & (np.arange(len(knots)) != 0))
        component_edges = [edge for edge, nodes in zip(selected, graph_nodes)
                           if nodes and connected[nodes].all()]
        rank_report = {'columns': len(columns), 'rank': 0, 'nullity': len(columns),
                       'unique_temporal_equations': 0, 'full_column_rank': False,
                       'rank_tolerance': 1e-7, 'regularization_rows': 0}
        nullspace = np.zeros((len(knots), 0))
        if len(columns) and component_edges:
            pairs = np.unique(np.column_stack((times[first[component_edges]],
                                               times[second[component_edges]])), axis=0)
            A = np.zeros((len(pairs), len(knots)))
            a, f = _interpolation(knots, pairs[:, 0])
            b, g = _interpolation(knots, pairs[:, 1])
            rr = np.arange(len(pairs))
            for cc, vv in ((a, -(1-f)), (a+1, -f), (b, 1-g), (b+1, g)):
                np.add.at(A, (rr, cc), vv)
            R, permutation = qr(A[:, columns], mode='r', pivoting=True, check_finite=False)
            diagonal = abs(np.diag(R))
            tolerance = max(1e-7, (float(diagonal.max()) if len(diagonal) else 0.)*1e-7)
            rank = int(np.count_nonzero(diagonal > tolerance))
            rank_report.update(rank=rank, nullity=len(columns)-rank,
                               unique_temporal_equations=len(pairs),
                               full_column_rank=rank == len(columns), rank_tolerance=tolerance,
                               min_qr_pivot=float(diagonal.min()) if len(diagonal) else 0.)
            if rank < len(columns):
                nullity = len(columns)-rank
                basis = np.vstack((-solve_triangular(R[:rank, :rank], R[:rank, rank:len(columns)]),
                                   np.eye(nullity)))
                original = np.empty_like(basis)
                original[permutation] = basis
                orthogonal = qr(original, mode='economic', check_finite=False)[0]
                nullspace = np.zeros((len(knots), nullity))
                nullspace[columns] = orthogonal
        else:
            connected[:] = False
        measurement_design[('red', 'green')[shell]] = rank_report
        graph_groups = {}
        for node in np.flatnonzero(observed_nodes):
            graph_groups.setdefault(root(int(node)), []).append(int(node))
        rank_report['graph_components'] = [
            {'first_knot': min(group), 'last_knot': max(group), 'knots': len(group),
             'start_s': float(knots[min(group)]), 'end_s': float(knots[max(group)]), 'contains_zero': 0 in group}
            for group in graph_groups.values()]
        identifiable = np.linalg.norm(nullspace, axis=1) <= 1e-7
        knot_connected[:, shell] = connected & identifiable
        rank_report['individually_identifiable_knots'] = int(np.count_nonzero(connected & identifiable))
        for edge, nodes in zip(selected, graph_nodes):
            edge_graph_connected[edge] = bool(nodes and connected[nodes].all() and identifiable[nodes].all())
        count, spread, support_rows = _surface_support(event_times, times, rows, landmark, normals,
                                                       max_support_distance_s)
        event_counts[:, shell], event_spread[:, shell] = count, spread
        left, fraction = _interpolation(knots, event_times)
        phase_known = ((connected[left] | (fraction >= 1.-1e-9))
                       & (connected[left+1] | (fraction <= 1e-9)))
        null_projection = ((1-fraction[:, None])*nullspace[left]+fraction[:, None]*nullspace[left+1])
        phase_known &= np.linalg.norm(null_projection, axis=1) <= 1e-7
        local_quality = (count >= min_tracks) & (spread >= min_spread)
        event_locally_supported[:, shell] = local_quality
        supported = phase_known & local_quality
        event_connected[:, shell] = phase_known
        event_supported[:, shell] = supported
        for i, support in enumerate(support_rows):
            for ci in np.unique(obs['camera'][support]):
                event_cameras[i, shell, int(ci)] = True
    turn_valid = np.zeros_like(event_supported)
    for shell in (0, 1):
        whole_turns = True
        last_supported = float(event_times[0])
        for i, time in enumerate(event_times):
            if time-last_supported > max_turn_gap_s:
                whole_turns = False
            if event_supported[i, shell]:
                last_supported = float(time)
            turn_valid[i, shell] = whole_turns and event_supported[i, shell]
    return {'source': 'conditional_metric_azimuth', 'knots': knots.copy(),
            'spin_curves': curves[:, 1:].copy(), 'event_times_s': event_times,
            'corrected_events': np.asarray(events, float),
            'event_camera': np.array([event[1] for event in events], int),
            'event_frame': np.array([event[2] for event in events], int),
            'event_supported': event_supported, 'event_phase_connected': event_connected,
            'event_locally_supported': event_locally_supported,
            'event_turn_count_valid': turn_valid, 'event_track_counts': event_counts,
            'event_normal_spread': event_spread, 'event_support_cameras': event_cameras,
            'knot_supported': knot_supported, 'knot_phase_connected': knot_connected,
            'edge_observation_first': first, 'edge_observation_second': second,
            'edge_times_first': times[first], 'edge_times_second': times[second],
            'edge_shells': np.asarray(obs['shell'])[first],
            'edge_cameras': np.asarray(obs['camera'])[first], 'edge_landmarks': landmark[first],
            'edge_residual_deg': np.rad2deg(residual),
            'edge_confidence': np.asarray(confidence)[accepted],
            'edge_phase_connected': edge_graph_connected,
            'observation_unit_normals': np.asarray(normals).copy(),
            'observation_times_s': times.copy(),
            'measurement_design': measurement_design,
            'policy': {'max_angular_residual_deg': 2., 'min_independent_tracks': min_tracks,
                       'min_2d_normal_spread': min_spread,
                       'max_support_distance_s': max_support_distance_s,
                       'max_turn_gap_s': max_turn_gap_s,
                       'regularization_creates_support': False}}


def reseed_spins(data, seed, cameras, F, pivot, geo, knots, angles, brio_offset_s=0.):
    """Return a full-bundle seed and diagnostics using conditional spin edges.

    ``knots, angles`` describe the previously refined physical-time trajectory.
    The returned event samples remain on the original raw event clock; native
    Brio observations and landmark initialization use the additional fitted
    ``brio_offset_s``. No existing calibration file or input array is modified.
    """
    knots, angles = np.asarray(knots, float), np.asarray(angles, float)
    F, pivot = np.asarray(F, float), np.asarray(pivot, float)
    if (knots.ndim != 1 or len(knots) < 3 or angles.shape != (len(knots), 3)
            or not np.isfinite(knots).all() or not np.isfinite(angles).all()
            or np.any(np.diff(knots) <= 0) or not np.isfinite(brio_offset_s)):
        raise ValueError('Need finite increasing knots, (N,3) angles, and timing offset')
    obs = data['observations']
    keys, landmark = np.unique(np.column_stack([obs[k] for k in ('camera', 'shell', 'track')]),
                               axis=0, return_inverse=True)
    times, normals, phi, theta, incidence, xy, valid = _unspun_normals(
        data, cameras, F, pivot, geo, knots, angles, brio_offset_s)
    # Only immediate observations in the original track order form edges;
    # rejected intermediate points do not silently bridge an evidence gap.
    order = np.lexsort((times, landmark))
    first, second = order[:-1], order[1:]
    same = landmark[first] == landmark[second]
    dt = times[second]-times[first]
    delta = np.arctan2(np.sin(phi[second]-phi[first]), np.cos(phi[second]-phi[first]))
    good = (same & valid[first] & valid[second] & (dt > 0) & (dt <= .15)
            & (abs(theta[second]-theta[first]) <= .12) & (abs(delta) <= np.pi/2))
    first, second, delta = first[good], second[good], delta[good]
    source_weight = np.asarray(obs.get('weight', np.ones(len(times))))
    confidence = (np.sqrt(source_weight[first]*source_weight[second])
                  *np.minimum(incidence[first], incidence[second])
                  *np.minimum(xy[first], xy[second])**2)
    curves = angles.copy()
    shell_reports = {}
    edge_residual = np.empty(len(delta))
    for shell, name in enumerate(('red', 'green')):
        selected = obs['shell'][first] == shell
        curve, details, residual = _spin_curve(knots, times[first[selected]], times[second[selected]],
                                              delta[selected], confidence[selected])
        curves[:, shell+1] = curve
        edge_residual[selected] = residual
        details['per_camera_edges'] = {str(ci): int(np.count_nonzero(selected & (obs['camera'][first] == ci)))
                                       for ci in range(2)}
        shell_reports[name] = details
    events = [tuple(e) for e in seed['events']]
    event_times = np.array([e[0] for e in events])
    new_angles = np.column_stack([np.interp(event_times, knots, curves[:, k]) for k in range(3)])
    physical_angles = np.column_stack([np.interp(times, knots, curves[:, k]) for k in range(3)])
    points = np.full((len(keys), 3), np.nan)
    # Prefer an accepted, well-conditioned ray for each nuisance landmark.
    # Retain an explicit geometric fallback for unobserved/rejected tracks so
    # the final batch can assess their pixels rather than deleting future data.
    fallback_count = 0
    groups = np.split(order, np.flatnonzero(np.diff(landmark[order]))+1)
    for l, candidates in enumerate(groups):
        accepted = candidates[valid[candidates]]
        if len(accepted):
            i = accepted[np.argmax(incidence[accepted]*xy[accepted])]
            h = int(obs['shell'][i]); beta = physical_angles[i, h+1]
            cb, sb = np.cos(beta), np.sin(beta)
            p = normals[i]
            points[l] = [cb*p[0]+sb*p[1], -sb*p[0]+cb*p[1], p[2]]
        else:
            i = candidates[np.argmin(times[candidates])]
            ci, h = int(obs['camera'][i]), int(obs['shell'][i])
            points[l] = initialize_surface_point(cameras[ci], obs['uv'][i], F, pivot,
                                                 physical_angles[i], h, geo['radius_m'], geo['gap_m'],
                                                 geo.get('red_shell_sign', 1))
            fallback_count += 1
    per_camera = [{'conditional_spin_edges': int(np.count_nonzero(obs['camera'][first] == ci))}
                  for ci in range(2)]
    result = {'events': events, 'angles': new_angles, 'points': points, 'keys': keys,
              'landmark': landmark, 'per_camera': per_camera,
              'quality': [None]*len(events), 'components': [[] for _ in events],
              'initial_brio_offset_s': float(brio_offset_s)}
    corrected_events = sorted((float(t)+brio_offset_s*(int(ci) == 1), int(ci), int(fi))
                               for t, ci, fi in events)
    evidence = conditional_spin_evidence(knots, curves, corrected_events, obs, times, normals,
                                         landmark, first, second, edge_residual, confidence)
    result['spin_evidence'] = evidence
    report = {'method': 'conditional_roll_ray_sphere_azimuth_edge_IRLS',
              'optimizer_hypothesis_only': True, 'observations': len(times),
              'geometrically_valid_observations': int(valid.sum()), 'accepted_relative_edges': len(delta),
              'landmark_fallback_count': fallback_count, 'additional_brio_offset_s': float(brio_offset_s),
              'per_shell': shell_reports,
              'angular_evidence': {'accepted_edges_within_2deg': len(evidence['edge_shells']),
                  'supported_event_fraction': dict(zip(('red', 'green'),
                      np.mean(evidence['event_supported'], axis=0).tolist())),
                  'phase_connected_event_fraction': dict(zip(('red', 'green'),
                      np.mean(evidence['event_phase_connected'], axis=0).tolist())),
                  'turn_count_valid_at_end': evidence['event_turn_count_valid'][-1].tolist(),
                  'measurement_design': evidence['measurement_design'],
                  'locally_supported_event_fraction': dict(zip(('red', 'green'),
                      np.mean(evidence['event_locally_supported'], axis=0).tolist())),
                  'local_support_gaps': {name: _support_gaps(evidence['event_times_s'],
                      evidence['event_locally_supported'][:, shell]) for shell, name in enumerate(('red', 'green'))},
                  'anchored_support_gaps': {name: _support_gaps(evidence['event_times_s'],
                      evidence['event_supported'][:, shell]) for shell, name in enumerate(('red', 'green'))},
                  'policy': evidence['policy']},
              'note': 'Pixel-derived relative spin increments initialize a final metric bundle fit. Roll and calibration are conditional inputs; regularization does not establish missing support or whole turns.'}
    return result, report
