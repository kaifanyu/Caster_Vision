"""Free-motion roll/swivel control comparisons; all outputs are research-only.

Reconstruct the historical roll export with its matching NativeBundle, then fit
the swivel KLT seed in unconstrained motion mode. Every comparison branch uses
the same prepared trajectory initialization, fixed calibration, motion prior,
finite zero-degree initial-roll prior and frozen per-control timing estimate.
These are consistency controls, never independent physical angle ground truth.
"""
from __future__ import annotations

import os
for _thread_variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '1'

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from blurtrack import BASELINE, ROOT
from blurtrack.evaluation import landmark_split, score_heldout
from blurtrack.metric import initialize_metric, refine
from blurtrack.observations import hybrid_observations
from dualcam.config import load_config, load_rig, write_json
from dualcam.native_observations import load_native_tracks
from dualcam.offline import NativeBundle
from dualcam.offline_workflow import summarize_solution
from dualcam.workflow import load_axes, calibration_hashes


def _arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def _load_seed(path):
    seed = _arrays(path)
    seed['events'] = [(float(t), int(ci), int(fi)) for t, ci, fi in seed['events']]
    count = len(seed['events'])
    seed.update(per_camera=[{}, {}], quality=[None] * count, components=[[] for _ in range(count)])
    seed['initial_roll_prior_rad'] = 0.
    return seed


def reconstruct_roll(native, cameras, F, pivot, geo):
    source = BASELINE / 'out/roll_validation_final'
    seed = _load_seed(source / 'seed.npz')
    saved = _arrays(source / 'bundle.npz')
    problem = NativeBundle(native, seed, cameras, F, pivot, geo, mode='motion', acceleration_std_deg_s2=500.)
    x = saved['x']
    if x.shape != problem.x0.shape or not np.isfinite(x).all():
        raise ValueError('Historical roll parameter vector does not match current NativeBundle')
    if np.any(x < problem.lower) or np.any(x > problem.upper):
        raise ValueError('Historical roll parameters violate current bounds')
    for key, expected in [('knots', problem.knots), ('keys', problem.keys),
                          ('landmark', problem.landmark), ('original_rows', problem.original_rows)]:
        np.testing.assert_allclose(saved[key], expected, atol=1e-12, rtol=0,
                                   err_msg=f'Historical roll {key} mismatch')
    errors = np.linalg.norm(problem.pixels(x)[0], axis=1)
    np.testing.assert_allclose(errors, saved['pixel_error'], atol=1e-7, rtol=1e-9,
                               err_msg='Historical roll pixel residuals cannot be reproduced')
    np.testing.assert_array_equal(problem.visibility(x), saved['visible'])
    np.testing.assert_allclose(x[problem.timing_id], saved['additional_brio_offset_s'], atol=1e-12, rtol=0)
    # The old export stored event poses, despite also exporting a knot vector.
    np.testing.assert_allclose(problem.angles_at(x, saved['events'][:, 0]), saved['angles'],
                               atol=1e-9, rtol=0, err_msg='Historical event-angle semantics changed')
    angles, _ = problem.unpack(x)
    diagnostics = {'reconstruction_validated': True, 'source': str(source.resolve()),
                   'parameter_count': len(x), 'knot_count': len(problem.knots),
                   'historical_event_angle_rows': len(saved['angles']),
                   'checks': ['parameter shape and bounds', 'knots', 'landmark keys/index mapping',
                              'source observation rows', 'pixel residuals', 'visibility',
                              'additional Brio offset', 'historical native event poses'],
                   'additional_brio_offset_s': float(x[problem.timing_id]),
                   'prior_initial_roll_deg': 0., 'mode': 'motion; all three components free'}
    return problem.knots, angles, float(x[problem.timing_id]), diagnostics


def prepare(control, native, cameras, F, pivot, geo, output, max_nfev):
    directory = output / f'{control}_baseline'
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / 'bundle.npz'
    provenance = {'native_provenance': native['provenance'], 'geometry': geo,
                  'F': np.asarray(F).tolist(), 'pivot': np.asarray(pivot).tolist(),
                  'initial_roll_prior_deg': 0., 'acceleration_std_deg_s2': 500.,
                  'mode': 'motion; all three components free'}
    if artifact.is_file():
        report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
        if report['provenance'] != provenance:
            raise ValueError(f'{control}: prepared control provenance changed')
        saved = _arrays(artifact)
        return saved['knots'], saved['angles'], float(saved['additional_brio_offset_s'])
    if control == 'roll':
        knots, angles, offset, report = reconstruct_roll(native, cameras, F, pivot, geo)
    else:
        seed = _load_seed(BASELINE / 'out/swivel_validation_seed/seed.npz')
        problem, x, stages = refine(native, seed, cameras, F, pivot, geo,
                                    max_nfev=max_nfev, freeze_timing=False)
        knots, (angles, _) = problem.knots, problem.unpack(x)
        offset = float(x[problem.timing_id])
        report = {'source': str(BASELINE / 'out/swivel_validation_seed/seed.npz'),
                  'optimization_stages': stages, 'optimizer_converged': bool(stages[-1]['converged'])}
    report.update(accuracy_validated=False, provenance=provenance,
                  caveat='Full-data conventional-track fit; common conditional initialization, not ground truth.')
    np.savez_compressed(artifact, knots=knots, angles=angles, additional_brio_offset_s=offset)
    write_json(directory / 'report.json', report)
    print(f'Prepared {control}: {len(knots)} knots, Brio offset {offset:+.6f}s', flush=True)
    return knots, angles, offset


def _trajectory_metrics(control, knots, angles, home_hold_s=2.5):
    degrees = np.rad2deg(angles)
    relative = degrees - degrees[0]
    home = knots <= knots[0] + home_hold_s
    target = 0 if control == 'roll' else 1
    other = [k for k in range(3) if k != target]
    labels = ('roll', 'red_spin', 'green_spin')
    return {'angle_order': list(labels), 'expected_moving_component': labels[target],
            'expectation_source': 'named manual control clip; no encoder or independent motion truth',
            'initial_angles_deg': degrees[0].tolist(), 'final_relative_angles_deg': relative[-1].tolist(),
            'range_deg': np.ptp(degrees, axis=0).tolist(),
            'home_interval_s': [float(knots[0]), float(min(knots[-1], knots[0] + home_hold_s))],
            'home_range_deg': np.ptp(degrees[home], axis=0).tolist(),
            'home_max_abs_relative_deg': np.max(np.abs(relative[home]), axis=0).tolist(),
            'cross_component_max_abs_relative_deg': {labels[k]: float(np.max(np.abs(relative[:, k]))) for k in other},
            'scope': 'Unrestricted fitted curve, including prior-dependent estimates; separate support coverage is reported.'}


def fit_control(control, source, native, learned, cameras, F, pivot, cfg, prepared, output, max_nfev):
    directory = output / f'{control}_fit_{source}'
    directory.mkdir(parents=True, exist_ok=True)
    knots, q, offset = prepared
    if source == 'klt':
        obs = {key: value.copy() for key, value in native['observations'].items()}
    else:
        if learned['provenance'].get('base_native_provenance') != native['provenance']:
            raise ValueError('Learned control cache has different native provenance')
        for ci in range(2):
            np.testing.assert_array_equal(learned['times'][ci], native['times'][ci])
        obs = (hybrid_observations(native['observations'], learned['observations']) if source == 'hybrid'
               else {key: value.copy() for key, value in learned['observations'].items()})
    data = {**native, 'observations': obs}
    branch_provenance = {'source': source, 'native_provenance': native['provenance'],
                         'learned_provenance': None if source == 'klt' else learned['provenance'],
                         'calibration_hashes': calibration_hashes(cfg), 'max_nfev': max_nfev,
                         'initial_roll_prior_deg': 0., 'acceleration_std_deg_s2': 500.,
                         'frozen_additional_brio_offset_s': offset}
    report_path = directory / 'results.json'
    if report_path.is_file():
        previous = json.loads(report_path.read_text(encoding='utf-8'))
        if previous['experiment'] != branch_provenance:
            raise ValueError(f'{control}/{source}: existing branch provenance differs')
        return previous
    seed = initialize_metric(data, cameras, F, pivot, cfg['geometry'], knots, q, offset, initial_prior_deg=0.)
    problem, x, stages = refine(data, seed, cameras, F, pivot, cfg['geometry'], max_nfev=max_nfev,
                                freeze_timing=True, acceleration_std_deg_s2=500.)
    angles, polar = problem.unpack(x)
    frames, summary, inlier = summarize_solution(problem, x, data, seed, F, pivot)
    errors = np.linalg.norm(problem.pixels(x)[0], axis=1)
    visible = problem.visibility(x)
    split = landmark_split(native)
    evaluation = score_heldout(native, cameras, F, pivot, cfg['geometry'], problem.knots, angles,
                               float(x[problem.timing_id]), heldout_keys=split['heldout_keys'],
                               trajectory_excludes_heldout=False, reference_name=f'{control}_common_KLT_tracks')
    summary.update(optimization_stages=stages, optimizer_converged=bool(stages[-1]['converged']))
    report = {'report_kind': 'fused_motion', 'method': f'{source}_unconstrained_control_bundle',
              'control': control, 'accuracy_validated': False, 'experiment': branch_provenance,
              'initialization_scope': 'Shared full-data conventional control solution; conditional refinement only.',
              'fit_constraints': 'No prescribed roll-only or swivel-only constraint. All three components are free.',
              'control_metrics': _trajectory_metrics(control, problem.knots, angles),
              'pixel_metrics': {'observations': len(errors), 'median_px': float(np.median(errors)),
                                'p95_px': float(np.percentile(errors, 95)),
                                'fraction_visible_within_3px': float(np.mean(visible & (errors <= 3.))),
                                'metric_inliers': int(inlier.sum())},
              'evaluation': evaluation, 'summary': summary, 'frames': frames,
              'timestamp_origin_s': native['timestamp_origin_s'] + summary['output_clock_origin_shift_s'],
              'F': F, 'pivot': pivot, 'geometry': cfg['geometry'], 'config': cfg,
              'videos': native['videos'], 'use_video_manifest': True}
    np.savez_compressed(directory / 'bundle.npz', knots=problem.knots, angles=angles,
                        optimized_parameters=x, surface_polar=polar,
                        additional_brio_offset_s=x[problem.timing_id], landmark_keys=problem.keys,
                        original_observation_rows=problem.original_rows,
                        observation_inlier=inlier, pixel_error=errors, visible=visible)
    with (directory / 'angles.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['native_time_s', 'roll_deg', 'red_spin_deg', 'green_spin_deg'])
        writer.writerows(np.column_stack((problem.knots, np.rad2deg(angles))))
    write_json(report_path, report)
    write_json(directory / 'control_metrics.json', {key: report[key] for key in
               ('control', 'method', 'accuracy_validated', 'control_metrics', 'pixel_metrics', 'summary')})
    print(f'Finished {control}/{source}: {report["control_metrics"]}', flush=True)
    return report


def write_comparison(output, combined):
    """Export a portable table and fitted-curve figure with explicit scope."""
    fields = ['control', 'source', 'roll_range_deg', 'red_range_deg', 'green_range_deg',
              'max_home_drift_deg', 'max_cross_component_deg', 'median_pixel_error_px',
              'visible_within_3px_fraction', 'reference_median_px', 'reference_p95_px',
              'reference_scored_fraction', 'reference_within_3px_including_failures']
    table = []
    for key, report in sorted(combined.items()):
        control, source = key.split('/')
        motion, pixels = report['control_metrics'], report['pixel_metrics']
        reference = report['reference_metrics']
        table.append([control, source, *motion['range_deg'], max(motion['home_max_abs_relative_deg']),
                      max(motion['cross_component_max_abs_relative_deg'].values()),
                      pixels['median_px'], pixels['fraction_visible_within_3px'], reference['median_px'],
                      reference['p95_px'], reference['scored_fraction'],
                      reference['fraction_within_3px_including_failures']])
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(table)
    lines = ['# Unconstrained control comparison', '',
             'All three angles are free. Branches share physical calibration, a finite 0-degree initial-roll prior, '
             '500 deg/s² acceleration prior, and the same fitted per-control clock offset. The conventional '
             'control solution initializes each branch; these are conditional consistency checks, not independent accuracy measurements.', '',
             '| Control | Source | Roll / red / green range (deg) | Max home drift (deg) | Max cross-component motion (deg) | Median pixel residual (px) | Visible within 3 px |',
             '|---|---|---|---:|---:|---:|---:|']
    for row in table:
        control, source, roll, red, green, home, cross, median, fraction = row[:9]
        lines.append(f'| {control} | {source} | {roll:.3f} / {red:.3f} / {green:.3f} | {home:.3f} | {cross:.3f} | {median:.3f} | {fraction:.1%} |')
    lines.extend(['', 'Home drift uses the first 2.5 seconds relative to its initial fitted pose. '
                  'Cross-component motion assumes roll is the moving component in the roll clip, and red spin in the swivel clip. '
                  'These expectations come from the named manual clips, not an encoder. Metrics cover the unrestricted fitted curve; '
                  'individual results.json files retain separate image support and unresolved statuses.', '',
                  'Pixel residuals within each branch use its own tracker observations and are not a like-for-like accuracy ranking. '
                  'The common KLT reference is also evaluated in each results.json, explicitly as descriptive consistency because '
                  'its full-data fit and initializer were not held out.', '',
                  '| Control | Source | Common-reference median (px) | p95 (px) | Scored coverage | Within 3 px, including failures |',
                  '|---|---|---:|---:|---:|---:|'])
    for row in table:
        median, p95, coverage, within = row[9:]
        number = lambda value: 'missing' if value is None else f'{value:.3f}'
        percent = lambda value: 'missing' if value is None else f'{value:.1%}'
        lines.append(f'| {row[0]} | {row[1]} | {number(median)} | {number(p95)} | {percent(coverage)} | {percent(within)} |')
    lines.extend(['',
                  '[Fitted relative curves](control_curves.png) · [CSV table](comparison.csv) · [Machine-readable summary](comparison.json)'])
    (output / 'COMPARISON.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'klt': '#2876b8', 'cotracker': '#dc8b21', 'hybrid': '#29935f'}
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex='col', constrained_layout=True)
    for col, control in enumerate(('roll', 'swivel')):
        for source in ('klt', 'cotracker', 'hybrid'):
            if f'{control}/{source}' not in combined:
                continue
            saved = _arrays(output / f'{control}_fit_{source}/bundle.npz')
            relative = np.rad2deg(saved['angles'] - saved['angles'][0])
            t = saved['knots'] - saved['knots'][0]
            for k in range(3):
                axes[k, col].plot(t, relative[:, k], label=source, color=colors[source], linewidth=1.3)
        for k, component in enumerate(('Roll', 'Red spin', 'Green spin')):
            ax = axes[k, col]
            ax.axvspan(0., 2.5, color='#c5c5c5', alpha=.2)
            ax.set_ylabel(f'{component} relative angle (deg)')
            ax.grid(alpha=.2)
        axes[0, col].set_title(f'{control.capitalize()} control: all motion components free')
        axes[2, col].set_xlabel('Native data time relative to first knot (s)')
        if axes[0, col].lines:
            axes[0, col].legend(loc='best')
    fig.suptitle('Fitted control curves: shared conventional initialization, no independent angle ground truth', fontsize=12)
    fig.savefig(output / 'control_curves.png', dpi=160)
    fig.savefig(output / 'control_curves.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls', nargs='+', choices=('roll', 'swivel'), default=['roll', 'swivel'])
    parser.add_argument('--sources', nargs='+', choices=('klt', 'cotracker', 'hybrid'), default=['klt', 'cotracker', 'hybrid'])
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/controls')
    parser.add_argument('--max-nfev', type=int, default=120)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(BASELINE / 'config/rig.yaml')
    cameras, _, _ = load_rig(cfg)
    axes = load_axes(cfg)
    F, pivot = axes['R_bc'], axes['pivot_c920_m']
    results = {}
    for control in args.controls:
        native = load_native_tracks(BASELINE / 'out' / f'{control}_native_cache')
        prepared = prepare(control, native, cameras, F, pivot, cfg['geometry'], args.output, args.max_nfev)
        if args.prepare_only:
            continue
        learned_path = args.output / f'{control}_cotracker/native_tracks.npz'
        learned = load_native_tracks(learned_path) if learned_path.is_file() else None
        for source in args.sources:
            if source != 'klt' and learned is None:
                print(f'SKIP {control}/{source}: collection incomplete at {learned_path}', flush=True)
                continue
            report = fit_control(control, source, native, learned, cameras, F, pivot, cfg, prepared,
                                  args.output, args.max_nfev)
            results[f'{control}/{source}'] = {key: report[key] for key in
                ('method', 'accuracy_validated', 'control_metrics', 'pixel_metrics', 'initialization_scope')}
            results[f'{control}/{source}']['reference_metrics'] = report['evaluation']['overall']
    if results:
        summary_path = args.output / 'comparison.json'
        combined = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else {}
        combined.update(results)
        write_json(summary_path, combined)
        write_comparison(args.output, combined)


if __name__ == '__main__':
    main()
