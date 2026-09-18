"""Diagnostic cubic angular rates with measurement-only observability checks.

The metric pose fit is left unchanged. A local C1 cubic Hermite interpolant on
a 30 Hz grid supplies analytic derivatives; this is a rate estimator, not a new
continuous-exposure fit or a source of additional temporal bandwidth.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.linalg import svd
from scipy.sparse import coo_matrix, csr_matrix


def _interpolation_matrix(knots, times):
    times = np.asarray(times, float)
    left = np.clip(np.searchsorted(knots, times, side='right')-1, 0, len(knots)-2)
    fraction = np.clip((times-knots[left])/(knots[left+1]-knots[left]), 0., 1.)
    matrix = coo_matrix((np.column_stack((1-fraction, fraction)).ravel(),
                         (np.repeat(np.arange(len(times)), 2),
                          np.column_stack((left, left+1)).ravel())),
                        shape=(len(times), len(knots))).tocsr()
    matrix.eliminate_zeros()
    return matrix


def _derivative_matrix(times):
    """Three-sample quadratic slopes, including nonuniform endpoint spacing."""
    rows, cols, values = [], [], []
    for i, time in enumerate(times):
        indices = np.arange(max(0, min(i-1, len(times)-3)), max(0, min(i-1, len(times)-3))+3)
        x = times[indices]
        for j in range(3):
            others = [k for k in range(3) if k != j]
            denominator = (x[j]-x[others[0]])*(x[j]-x[others[1]])
            value = (2*time-x[others[0]]-x[others[1]])/denominator
            rows.append(i); cols.append(indices[j]); values.append(value)
    return coo_matrix((values, (rows, cols)), shape=(len(times), len(times))).tocsr()


@dataclass
class CubicRateCurve:
    """Linear-in-knot-values interpolant, exposing its derivative functional."""
    knots: np.ndarray
    angles: np.ndarray
    knot_hz: float = 30.

    def __post_init__(self):
        self.knots = np.asarray(self.knots, float)
        self.angles = np.asarray(self.angles, float)
        if (self.knots.ndim != 1 or len(self.knots) < 3
                or self.angles.shape != (len(self.knots), 3)
                or not np.isfinite(self.knots).all() or not np.isfinite(self.angles).all()
                or np.any(np.diff(self.knots) <= 0)
                or not np.isfinite(self.knot_hz) or self.knot_hz <= 0):
            raise ValueError('Need finite increasing knots, (N,3) unwrapped angles and positive knot_hz')
        start, end = self.knots[[0, -1]]
        self.grid = start + np.arange(int(np.floor((end-start)*self.knot_hz))+1)/self.knot_hz
        if end-self.grid[-1] > 1e-9:
            self.grid = np.r_[self.grid, end]
        else:
            self.grid[-1] = end
        if len(self.grid) < 3:
            self.grid = np.linspace(start, end, 3)
        self.sample_matrix = _interpolation_matrix(self.knots, self.grid)
        self.slope_matrix = _derivative_matrix(self.grid)
        values = self.sample_matrix @ self.angles
        self.spline = CubicHermiteSpline(self.grid, values, self.slope_matrix @ values,
                                         extrapolate=False)

    def functional(self, times):
        """Rows map original metric angle knots to interpolant d(angle)/dt."""
        times = np.asarray(times, float)
        valid = np.isfinite(times) & (times >= self.knots[0]) & (times <= self.knots[-1])
        clipped = np.clip(np.nan_to_num(times, nan=self.knots[0]), self.grid[0], self.grid[-1])
        left = np.clip(np.searchsorted(self.grid, clipped, side='right')-1, 0, len(self.grid)-2)
        step = self.grid[left+1]-self.grid[left]; u = (clipped-self.grid[left])/step
        values = np.column_stack(((6*u*u-6*u)/step, (-6*u*u+6*u)/step))
        direct = coo_matrix((values.ravel(), (np.repeat(np.arange(len(times)), 2),
                                               np.column_stack((left, left+1)).ravel())),
                            shape=(len(times), len(self.grid))).tocsr()
        slope = coo_matrix((np.column_stack((3*u*u-4*u+1, 3*u*u-2*u)).ravel(),
                             (np.repeat(np.arange(len(times)), 2),
                              np.column_stack((left, left+1)).ravel())),
                            shape=(len(times), len(self.grid))).tocsr()
        functional = (direct+slope @ self.slope_matrix) @ self.sample_matrix
        functional.data[abs(functional.data) < 1e-12] = 0.
        functional.eliminate_zeros()
        return functional.tocsr(), valid

    def evaluate(self, times):
        functional, valid = self.functional(times)
        rates = np.asarray(functional @ self.angles)
        rates[~valid] = np.nan
        return self.spline(times), rates, functional, valid


class MeasurementDesign:
    """Nullspaces of actual temporal equations, without phase or motion priors.

    A component's arbitrary constant phase is deliberately retained. A rate is
    observable if its functional annihilates every measurement null mode.
    """
    def __init__(self, matrix, *, relative_rank_tolerance=1e-8):
        matrix = csr_matrix(matrix, dtype=float)
        matrix.sum_duplicates(); matrix.eliminate_zeros()
        self.nknots = matrix.shape[1]
        parent = np.arange(self.nknots)

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return int(i)

        for row in range(matrix.shape[0]):
            nodes = matrix.indices[matrix.indptr[row]:matrix.indptr[row+1]]
            if len(nodes):
                anchor = root(nodes[0])
                for node in nodes[1:]:
                    parent[root(node)] = anchor
        roots = np.array([root(i) for i in range(self.nknots)])
        self.components, self.diagnostics = [], []
        for component in np.unique(roots):
            columns = np.flatnonzero(roots == component)
            block = matrix[:, columns]
            rows = np.flatnonzero(np.asarray(block.getnnz(axis=1)).ravel() > 0)
            if not len(rows):
                null = np.eye(len(columns)); rank = 0; singular = np.empty(0)
            else:
                dense = block[rows].toarray()
                # Identical track timestamps repeat the same temporal equation.
                # They improve precision but cannot manufacture temporal rank.
                dense = np.unique(np.round(dense, decimals=12), axis=0)
                _, singular, vh = svd(dense, full_matrices=True, check_finite=False)
                threshold = max(1e-12, singular[0]*relative_rank_tolerance)
                rank = int(np.sum(singular > threshold)); null = vh[rank:].T.copy()
            self.components.append((columns, null))
            self.diagnostics.append({'first_knot': int(columns[0]), 'last_knot': int(columns[-1]),
                                     'knots': len(columns), 'measurement_rank': rank,
                                     'nullity_without_phase_anchor': int(null.shape[1]),
                                     'largest_singular_value': float(singular[0]) if len(singular) else None,
                                     'smallest_retained_singular_value': float(singular[rank-1]) if rank else None,
                                     'regularization_rows': 0})

    def identifiable(self, functionals, *, relative_projection_tolerance=1e-7):
        functionals = csr_matrix(functionals)
        norm = np.sqrt(np.asarray(functionals.multiply(functionals).sum(axis=1)).ravel())
        projected_squared = np.zeros(functionals.shape[0])
        for columns, null in self.components:
            if null.shape[1]:
                projection = functionals[:, columns] @ null
                projected_squared += np.sum(np.asarray(projection)**2, axis=1)
        relative = np.sqrt(projected_squared)/np.maximum(norm, 1e-12)
        return (norm > 1e-12) & (relative <= relative_projection_tolerance), relative


def _edge_design(knots, evidence, shell):
    times0 = np.asarray(evidence.get('edge_times_first', []), float)
    times1 = np.asarray(evidence.get('edge_times_second', []), float)
    shells = np.asarray(evidence.get('edge_shells', []), int)
    if not (times0.shape == times1.shape == shells.shape):
        raise ValueError('Spin evidence temporal-edge arrays must have equal lengths')
    confidence = np.asarray(evidence.get('edge_confidence', np.ones(len(times0))), float)
    residual = np.asarray(evidence.get('edge_residual_deg', np.zeros(len(times0))), float)
    if confidence.shape != times0.shape or residual.shape != times0.shape:
        raise ValueError('Spin edge confidence/residual arrays must match temporal-edge lengths')
    accepted = ((shells == shell) & np.isfinite(times0) & np.isfinite(times1)
                & (times1 > times0) & (times1-times0 <= .150000001)
                & (times0 >= knots[0]) & (times1 <= knots[-1])
                & np.isfinite(confidence) & (confidence > 0)
                & np.isfinite(residual) & (abs(residual) <= 2.))
    pairs = np.unique(np.column_stack((times0[accepted], times1[accepted])), axis=0)
    if not len(pairs):
        return csr_matrix((0, len(knots))), accepted
    return (_interpolation_matrix(knots, pairs[:, 1])
            - _interpolation_matrix(knots, pairs[:, 0])), accepted


def _local_support(query_times, evidence, shell, accepted, *, min_tracks=6, min_spread=.025):
    """Use previously validated local geometry, never accumulated phase flags."""
    native_times = np.asarray(evidence.get('event_times_s', []), float)
    local = evidence.get('event_locally_supported')
    if local is not None and len(native_times):
        local = np.asarray(local, bool)
        if native_times.ndim != 1 or local.shape != (len(native_times), 2):
            raise ValueError('Spin event support must have shape (N,2) matching event_times_s')
        local = local[:, shell]
        order = np.argsort(native_times); native_times, local = native_times[order], local[order]
        right = np.clip(np.searchsorted(native_times, query_times), 0, len(native_times)-1)
        left = np.maximum(0, right-1)
        nearest = np.where(abs(query_times-native_times[left]) <= abs(query_times-native_times[right]), left, right)
        close = abs(query_times-native_times[nearest]) <= .02
        return local[nearest] & close
    # Synthetic tests and collectors can supply counts/spread directly without
    # any phase-connectivity flag; require both so priors cannot create support.
    counts = evidence.get('event_track_counts'); spread = evidence.get('event_normal_spread')
    if counts is not None and spread is not None and len(native_times):
        temporary = dict(evidence)
        temporary['event_locally_supported'] = ((np.asarray(counts) >= min_tracks)
                                                & (np.asarray(spread) >= min_spread))
        return _local_support(query_times, temporary, shell, accepted)
    return np.zeros(len(query_times), bool)


def _prediction_labels(times, measured, valid, max_prediction_s):
    status = np.full(len(times), 'unresolved', dtype=object)
    status[measured & valid] = 'vision'
    observed = np.sort(times[measured & valid])
    if len(observed) and max_prediction_s > 0:
        right = np.clip(np.searchsorted(observed, times), 0, len(observed)-1)
        left = np.maximum(0, right-1)
        distance = np.minimum(abs(times-observed[left]), abs(times-observed[right]))
        predicted = (~measured) & valid & (distance <= max_prediction_s)
        status[predicted] = 'predicted'
    return status


def annotate_rates(report, knots, angles, phase_evidence, *, knot_hz=30., max_prediction_s=.15):
    """Return a copied report with separate rate, phase and turn-count fields.

    Input evidence must come from accepted visible metric observations. This
    function cannot infer visibility or physical validity from tracker scores.
    Pose angles/status and turn-count flags remain unchanged. Rates are rad/s;
    rig angular-velocity vectors use the C920 optical frame specified by F.
    Knots and evidence use the original native-relative clock. Report times
    have subtracted summary.output_clock_origin_shift_s; support intervals are
    converted back into that same report-relative clock on output.
    """
    if not np.isfinite(max_prediction_s) or max_prediction_s < 0:
        raise ValueError('max_prediction_s must be finite and nonnegative')
    result = copy.deepcopy(report); frames = result.get('frames', [])
    curve = CubicRateCurve(knots, angles, knot_hz)
    evidence_knots = np.asarray(phase_evidence.get('knots', knots), float)
    if evidence_knots.shape != curve.knots.shape or not np.allclose(evidence_knots, curve.knots, atol=1e-10, rtol=0):
        raise ValueError('Rate curve and temporal evidence must use the same metric knots')
    origin_shift = float(report.get('summary', {}).get('output_clock_origin_shift_s', 0.))
    if not np.isfinite(origin_shift):
        raise ValueError('output_clock_origin_shift_s must be finite')
    times = np.asarray([f['time_s'] for f in frames], float)+origin_shift
    # Subtracting then restoring an origin can move an exact endpoint a few
    # floating-point ulps beyond its domain. This is numerical tolerance only.
    for endpoint in curve.knots[[0, -1]]:
        times[np.isclose(times, endpoint, atol=1e-10, rtol=0)] = endpoint
    poses, rates, functional, in_range = curve.evaluate(times)
    status = np.full((len(frames), 3), 'unresolved', dtype=object)
    sources = ['conditional_metric_roll_poses', 'conditional_metric_spin_edges', 'conditional_metric_spin_edges']
    measured_by_component, design_reports, projections = [], [], []

    # Roll observations are accepted metric pose values, not conditional spin
    # edges. The backend's near-in-time "vision" flag can include interpolated
    # support, so prefer actual event visual_components when these are present.
    # The caller can supply stricter native-relative observation times instead.
    roll_times = phase_evidence.get('roll_observation_times_s')
    roll_provenance = 'explicit_native_relative_roll_observation_times'
    if roll_times is None:
        has_visual_flags = any('visual_components' in f for f in frames)
        roll_provenance = ('strict_roll_with_actual_event_visual_support' if has_visual_flags
                           else 'legacy_conditional_strict_roll_pose_samples')
        roll_times = []
        for frame in frames:
            value = frame.get('strict_angles', frame.get('angles', [None]*3))[0]
            if (frame.get('strict_status', frame.get('status', ['unresolved']*3))[0] in ('home', 'vision')
                    and value is not None and np.isfinite(value)
                    and (not has_visual_flags or 0 in frame.get('visual_components', []))):
                roll_times.append(frame['time_s']+origin_shift)
    roll_times = np.asarray(roll_times, float)
    for endpoint in curve.knots[[0, -1]]:
        roll_times[np.isclose(roll_times, endpoint, atol=1e-10, rtol=0)] = endpoint
    roll_times = np.unique(roll_times[np.isfinite(roll_times) & (roll_times >= curve.knots[0]) & (roll_times <= curve.knots[-1])])
    roll_design = MeasurementDesign(_interpolation_matrix(curve.knots, roll_times))
    roll_measured, roll_projection = roll_design.identifiable(functional)
    measured_by_component.append(roll_measured); projections.append(roll_projection)
    design_reports.append({'component': 'roll', 'source': sources[0], 'components': roll_design.diagnostics,
                           'observation_provenance': roll_provenance,
                           'accepted_pose_times': len(roll_times),
                           'assumption': 'Strict pixel-supported roll poses are conditional angle measurements; pose-fit priors/calibration can still bias their values.'})
    for shell in (0, 1):
        matrix, accepted = _edge_design(curve.knots, phase_evidence, shell)
        design = MeasurementDesign(matrix)
        identifiable, relative_null = design.identifiable(functional)
        local = _local_support(times, phase_evidence, shell, accepted)
        measured_by_component.append(identifiable & local); projections.append(relative_null)
        design_reports.append({'component': ('red_spin', 'green_spin')[shell], 'source': sources[shell+1],
                               'accepted_edges': int(accepted.sum()), 'unique_temporal_equations': matrix.shape[0],
                               'components': design.diagnostics})
    measured = np.column_stack(measured_by_component) & in_range[:, None]
    for component in range(3):
        status[:, component] = _prediction_labels(times, measured[:, component], in_range, max_prediction_s)
    F = np.asarray(report.get('F', np.eye(3)), float)
    if (F.shape != (3, 3) or not np.isfinite(F).all()
            or not np.allclose(F.T @ F, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(F), 1., atol=1e-6, rtol=0)):
        raise ValueError('Report F must be a finite proper (3,3) rig-frame rotation')
    intervals = []
    for i in range(len(frames)):
        nodes = functional.indices[functional.indptr[i]:functional.indptr[i+1]]
        intervals.append([float(curve.knots[nodes.min()]-origin_shift),
                          float(curve.knots[nodes.max()]-origin_shift)] if len(nodes) and in_range[i] else None)
    for i, frame in enumerate(frames):
        frame['phase_status'] = copy.deepcopy(frame.get('phase_status', frame.get('status', ['unresolved']*3)))
        frame.setdefault('turn_count_valid', [False]*3)
        frame['angular_velocity_model_interpolant'] = [float(v) if np.isfinite(v) else None for v in rates[i]]
        frame['angular_velocity'] = [float(rates[i, k]) if status[i, k] != 'unresolved' and np.isfinite(rates[i, k]) else None for k in range(3)]
        frame['rate_status'] = status[i].tolist()
        frame['rate_support_interval_s'] = [intervals[i] if measured[i, k] else None for k in range(3)]
        frame['rate_source'] = sources.copy()
        frame['rate_relative_null_projection'] = [float(projections[k][i]) for k in range(3)]
        # The vector is the derivative of the diagnostic cubic orientation.
        # The existing reported pose itself is deliberately not overwritten.
        alpha = poses[i, 0]
        axis = F @ np.array([0., -np.sin(alpha), np.cos(alpha)]) if np.isfinite(alpha) else np.full(3, np.nan)
        for component, shell in ((1, 'red'), (2, 'green')):
            valid_vector = status[i, 0] != 'unresolved' and status[i, component] != 'unresolved' and np.isfinite(rates[i, [0, component]]).all()
            frame['angular_velocity_rig_'+shell] = (rates[i, 0]*F[:, 0]+rates[i, component]*axis).tolist() if valid_vector else None
            frame['angular_velocity_rig_'+shell+'_status'] = ('vision' if measured[i, 0] and measured[i, component] else 'predicted') if valid_vector else 'unresolved'
    widths = [v[1]-v[0] for v in intervals if v is not None]
    result['rate_analysis'] = {
        'method': 'local_C1_cubic_Hermite_diagnostic_derivative', 'units': 'radians_per_second',
        'pose_fit_modified': False, 'knot_hz': float(knot_hz), 'grid_spacing_s': 1./knot_hz,
        'output_clock_origin_shift_s': origin_shift,
        'time_coordinate_note': 'Spline/evidence evaluation adds the output clock origin shift to frame time_s; exported support bounds use frame time_s coordinates.',
        'nominal_sampling_nyquist_hz': knot_hz/2, 'maximum_derivative_stencil_s': max(widths, default=0.),
        'bandwidth_note': 'Fixed-rate resampling and cubic interpolation do not add temporal bandwidth or establish anti-aliasing; rates are windowed model estimates, not instantaneous ground truth.',
        'support_interval_note': 'Per-frame intervals bound original pose knots in the local derivative stencil, not an uncertainty interval or the full span of measurement equations used for its nullspace test.',
        'phase_invariance': 'Spin rate functionals are tested against every measurement null mode, including arbitrary offsets of disconnected phase components. No phase anchor or motion-prior row establishes rate support.',
        'max_prediction_s': float(max_prediction_s), 'measurement_only_design': design_reports,
        'coverage': {name: {label: int(np.sum(status[:, c] == label)) for label in ('vision', 'predicted', 'unresolved')}
                     for c, name in enumerate(('roll', 'red_spin', 'green_spin'))},
        'rig_vector_frame': 'C920 optical frame; F maps calibrated home axes into this frame',
        'rig_vector_orientation': 'Uses the diagnostic cubic roll and its derivative; saved metric pose angles remain unchanged.',
        'uncertainty_note': 'Observability is conditional on accepted point identities, the calibrated geometry, roll and timing. It is not an angular-accuracy certificate or covariance estimate.'}
    return result
