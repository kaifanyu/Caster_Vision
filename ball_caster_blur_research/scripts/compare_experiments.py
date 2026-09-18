"""Compare completed fits without modifying trajectories or baseline artifacts.

Run again as branches finish: python scripts/compare_experiments.py
All intervals use original native-relative corrected time (report time plus
its output-origin shift), matching the conditional consistency evaluator.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blurtrack import BASELINE

COMPONENTS = ('roll', 'red_spin', 'green_spin')
LABELS = ('Roll', 'Red spin', 'Green spin')
WINDOWS = ((10.6, 12.), (18.6, 19.7), (20.9, 21.5))
DEFAULT_BRANCHES = ('baseline', 'cotracker_dual', 'cotracker_c920', 'cotracker_brio101',
                    'hybrid_dual', 'klt_dual_holdout', 'cotracker_dual_holdout', 'hybrid_dual_holdout')
DISPLAY = {'baseline': 'Original KLT baseline', 'cotracker_dual': 'CoTracker dual',
           'cotracker_c920': 'CoTracker C920', 'cotracker_brio101': 'CoTracker Brio101',
           'cotracker_c920_native': 'CoTracker C920 native',
           'cotracker_brio101_native': 'CoTracker Brio101 native',
           'hybrid_dual': 'Hybrid dual', 'klt_dual_holdout': 'KLT dual [holdout]',
           'cotracker_dual_holdout': 'CoTracker dual [holdout]',
           'hybrid_dual_holdout': 'Hybrid dual [holdout]'}


def _finite(value):
    return value is not None and np.isfinite(value)


def _fractions(values):
    counts = dict(Counter(str(v) for v in values))
    return {'counts': counts, 'fractions': {k: v/len(values) for k, v in counts.items()}}


def _spans(times, mask, max_gap=.15):
    """Event-cell intervals for plotted status spans; never bridge long gaps."""
    times = np.asarray(times, float); mask = np.asarray(mask, bool)
    if not len(times):
        return []
    order = np.argsort(times, kind='stable'); times, mask = times[order], mask[order]
    bounds = np.r_[times[0], .5*(times[:-1]+times[1:]), times[-1]]
    spans = []
    for i in np.flatnonzero(mask):
        left = max(bounds[i], times[i]-max_gap/2)
        right = min(bounds[i+1], times[i]+max_gap/2)
        if spans and left-spans[-1][1] <= 1e-10:
            spans[-1][1] = float(right)
        else:
            spans.append([float(left), float(right)])
    return spans


def summarize_report(report):
    frames = report.get('frames', [])
    offset = float(report.get('summary', {}).get('output_clock_origin_shift_s', 0.))
    times = np.asarray([float(f['time_s'])+offset for f in frames])
    rate_annotated = bool(report.get('rate_analysis')) and all('rate_status' in f for f in frames)
    components = {}
    for k, name in enumerate(COMPONENTS):
        phase = [f.get('phase_status', f.get('status', ['unresolved']*3))[k] for f in frames]
        strict = [f.get('strict_status', f.get('status', ['unresolved']*3))[k] in ('home', 'vision')
                  and _finite(f.get('strict_angles', f.get('angles', [None]*3))[k]) for f in frames]
        local = [k in f.get('visual_components', []) for f in frames]
        rate_status = [f['rate_status'][k] for f in frames] if rate_annotated else None
        rates = np.asarray([f.get('angular_velocity', [None]*3)[k] for f in frames], float)
        rate_vision = np.asarray([s == 'vision' for s in rate_status]) & np.isfinite(rates) if rate_annotated else np.zeros(len(frames), bool)
        phase_estimated = np.asarray(phase) == 'phase_estimated'
        components[name] = {
            'phase': _fractions(phase), 'strict_phase_supported_fraction': float(np.mean(strict)) if frames else None,
            'actual_local_pixel_support_fraction': float(np.mean(local)) if frames and any('visual_components' in f for f in frames) else None,
            'phase_estimated_fraction': float(np.mean(phase_estimated)) if frames else None,
            'phase_estimated_spans_s': _spans(times, phase_estimated),
            'rate': _fractions(rate_status) if rate_annotated else None,
            'rate_vision_fraction': float(np.mean(rate_vision)) if frames and rate_annotated else None,
            'turn_count_valid_at_end': bool(frames[-1].get('turn_count_valid', [False]*3)[k]) if frames else None,
            'windows': {}}
        for start, end in WINDOWS:
            selected = (times >= start) & (times <= end)
            observed = selected & rate_vision
            values = np.rad2deg(rates[observed])
            components[name]['windows'][f'{start:g}-{end:g}'] = {
                'start_s': start, 'end_s': end, 'native_events': int(selected.sum()),
                'rate_vision_events': int(observed.sum()) if rate_annotated else None,
                'rate_vision_fraction': float(observed.sum()/selected.sum()) if selected.any() and rate_annotated else None,
                'vision_rate_min_deg_s': float(values.min()) if len(values) else None,
                'vision_rate_max_deg_s': float(values.max()) if len(values) else None,
                'vision_rate_p95_abs_deg_s': float(np.percentile(abs(values), 95)) if len(values) else None,
                'phase_estimated_fraction': float(np.mean(phase_estimated[selected])) if selected.any() else None}
    return {'native_events': len(frames), 'time_domain_native_relative_s': [float(times.min()), float(times.max())] if len(times) else None,
            'coverage_denominator': 'Emitted native frame events in this branch; mono native refits contain only the active camera.',
            'native_event_camera_counts': dict(Counter(f.get('camera', 'unknown') for f in frames)),
            'output_clock_origin_shift_s': offset, 'rates_annotated': rate_annotated,
            'rate_evidence_note': ('Saved rate annotation; accepted visible metric evidence, conditional on physical fit.' if rate_annotated else
                                  'Comparable rate annotation unavailable. Original baseline derivatives and differently gated phase evidence are not substituted.'),
            'components': components,
            'rate_analysis': report.get('rate_analysis'),
            'optimizer_converged': report.get('summary', {}).get('optimizer_converged'),
            'final_fit_inlier_rmse_px': report.get('summary', {}).get('inlier_rmse_px'),
            'camera_ids': report.get('experiment', {}).get('camera_ids'),
            'heldout_reference_excluded': bool(report.get('experiment', {}).get('heldout_reference_excluded', False))}


def _read_json(path):
    content = path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def load_comparison(experiments, baseline_path, branch_names):
    branches, raw = {}, {}
    for name in branch_names:
        path = baseline_path if name == 'baseline' else experiments/name/'results.json'
        source_branch = name
        if name in ('cotracker_c920', 'cotracker_brio101'):
            preferred = experiments/(name+'_native')/'results.json'
            if preferred.exists():
                try:
                    preferred_report, _ = _read_json(preferred)
                    if preferred_report.get('status') == 'completed' and preferred_report.get('frames'):
                        path, source_branch = preferred, name+'_native'
                except (OSError, json.JSONDecodeError):
                    pass
        entry = {'label': DISPLAY.get(source_branch, source_branch), 'source_branch': source_branch,
                 'results_path': str(path.resolve())}
        if source_branch in ('cotracker_c920', 'cotracker_brio101'):
            entry['label'] += ' (legacy gauge)'
            entry['phase_gauge_note'] = 'Legacy mono output: shared cam0 initial gauge may precede the first active-camera observation. Phase coverage is not a fair native-time camera ablation.'
        elif source_branch in ('cotracker_c920_native', 'cotracker_brio101_native'):
            entry['phase_gauge_note'] = 'Native mono refit: phase gauge at the first observed active-camera image; rates/windows aligned to the common physical clock.'
        try:
            report, digest = _read_json(path)
        except (OSError, json.JSONDecodeError) as error:
            branches[name] = {**entry, 'available': False, 'reason': str(error)}
            continue
        if report.get('status') != 'completed' or not report.get('frames'):
            branches[name] = {**entry, 'available': False, 'reason': 'No completed nonempty report yet'}
            continue
        entry.update(available=True, results_sha256=digest, **summarize_report(report))
        evaluation_path = path.parent/'evaluation.json'
        evaluation = report.get('evaluation')
        if evaluation_path.exists():
            try:
                evaluation, entry['evaluation_sha256'] = _read_json(evaluation_path)
                entry['evaluation_path'] = str(evaluation_path.resolve())
            except (OSError, json.JSONDecodeError) as error:
                entry['evaluation_read_error'] = str(error)
        if evaluation:
            # Both the scorer assertion and the trajectory configuration must
            # agree; filenames containing "holdout" are never sufficient.
            excluded = bool(evaluation.get('trajectory_excludes_heldout_asserted')) and entry['heldout_reference_excluded']
            entry['evaluation_group'] = 'conditional_track_holdout' if excluded else 'full_data_descriptive'
            entry['evaluation'] = {k: evaluation.get(k) for k in (
                'metric', 'validation_scope', 'reference_name', 'split_seed', 'heldout_tracks',
                'overall', 'per_camera', 'per_shell', 'windows', 'warnings', 'initialization_caveat')}
            entry['evaluation']['trajectory_excludes_heldout_asserted'] = bool(evaluation.get('trajectory_excludes_heldout_asserted'))
        else:
            entry['evaluation_group'] = 'not_scored'
            entry['evaluation'] = None
        branches[name] = entry; raw[name] = report
    return branches, raw


def _format(value, *, percent=False, decimals=2):
    if value is None or not np.isfinite(value):
        return 'N/A'
    return f'{100*value:.1f}%' if percent else f'{value:.{decimals}f}'


def write_markdown(path, comparison):
    branches = comparison['branches']
    available = {k: v for k, v in branches.items() if v['available']}
    missing = [k for k, v in branches.items() if not v['available']]
    lines = ['# Experiment comparison', '', f"Generated {comparison['generated_utc']}.", '',
             'These are conditional consistency and observability comparisons, not angular accuracy measurements. '
             'Full-data and held-out evaluations use separate tables. A low scored-pixel error must be read with scored coverage and the failure-inclusive 3 px fraction.', '',
             'Missing or incomplete branches: '+(', '.join(missing) if missing else 'none')+'.', '',
             '## Full-data descriptive consistency', '',
             'The reference pixels were not excluded from pose fitting in this group; these rows are not held-out accuracy tests.', '']
    for group, title in (('full_data_descriptive', None), ('conditional_track_holdout', 'Conditional held-out-track consistency')):
        if title:
            lines += ['## '+title, '', 'Whole reference tracks were excluded from this refinement, but every branch shares a full-data baseline initializer. '
                      'The reference tracker and learned tracker also share source images. This is conditional validation, not an independent ground-truth test.', '']
        lines += ['| Branch | Median px | p95 px | Within 3 px / all candidates | Scored / candidates | Scored pixels |',
                  '|---|---:|---:|---:|---:|---:|']
        for name, item in available.items():
            if item['evaluation_group'] != group:
                continue
            stats = item['evaluation']['overall']
            lines.append(f"| {item['label']} | {_format(stats.get('median_px'))} | {_format(stats.get('p95_px'))} | "
                         f"{_format(stats.get('fraction_within_3px_including_failures'), percent=True)} | "
                         f"{_format(stats.get('scored_fraction'), percent=True)} | {stats.get('count', 0)} |")
        lines += ['']
    reference = available.get('klt_dual_holdout')
    if reference and reference['evaluation_group'] == 'conditional_track_holdout':
        reference_stats = reference['evaluation']['overall']
        lines += ['Relative to the KLT holdout control (positive pixel deltas are larger errors):', '']
        for name, item in available.items():
            if name == 'klt_dual_holdout' or item['evaluation_group'] != 'conditional_track_holdout':
                continue
            stats = item['evaluation']['overall']
            needed = ('median_px', 'p95_px', 'scored_fraction', 'fraction_within_3px_including_failures')
            if all(stats.get(k) is not None and reference_stats.get(k) is not None for k in needed):
                delta = {k: stats[k]-reference_stats[k] for k in needed}
                lines.append(f"- {item['label']}: median {delta['median_px']:+.2f} px; p95 {delta['p95_px']:+.2f} px; "
                             f"scored coverage {100*delta['scored_fraction']:+.1f} percentage points; "
                             f"failure-inclusive 3 px fraction {100*delta['fraction_within_3px_including_failures']:+.1f} percentage points.")
        lines += ['', 'These descriptive differences do not establish statistical significance or physical angular accuracy.', '']
    unscored = [v['label'] for v in available.values() if v['evaluation_group'] == 'not_scored']
    if unscored:
        lines += ['No comparable reference-pixel score available for: '+', '.join(unscored)+'.', '']
    lines += ['## Phase and local measurement coverage', '',
              'Fractions use each branch’s emitted native events: both timelines for dual fits and only the active timeline for native mono refits. '
              'Event counts are not independent sample counts and raw mono/dual counts are not compared. '
              'Strict phase uses `strict_status` home/vision and finite strict angles. Local pixels require the actual event’s `visual_components`. '
              'Rate vision means the derivative functional passes the conditional temporal evidence test. Missing baseline rate annotation is N/A, not zero.', '',
              '| Branch / component | Strict phase | Phase estimated | Actual local pixels | Rate vision | Final whole turns valid |',
              '|---|---:|---:|---:|---:|---:|']
    for item in available.values():
        for component, metrics in item['components'].items():
            lines.append(f"| {item['label']} / {component} | {_format(metrics['strict_phase_supported_fraction'], percent=True)} | "
                         f"{_format(metrics['phase_estimated_fraction'], percent=True)} | {_format(metrics['actual_local_pixel_support_fraction'], percent=True)} | "
                         f"{_format(metrics['rate_vision_fraction'], percent=True)} | {metrics['turn_count_valid_at_end']} |")
    lines += ['', 'Mono phase origins:', '']
    for item in available.values():
        if item.get('phase_gauge_note'):
            lines.append(f"- {item['label']}: {item['phase_gauge_note']}")
    lines += ['', '## Fast-motion windows', '',
              'Reported speeds below use rate-vision events only and degrees/second. A derivative may remain observable inside a later phase component even after its accumulated phase becomes uncertain.', '',
              '| Window s | Branch / component | Rate vision | Signed min | Signed max | p95 absolute | Phase estimated |',
              '|---|---|---:|---:|---:|---:|---:|']
    for start, end in WINDOWS:
        window = f'{start:g}-{end:g}'
        for item in available.values():
            if not item['rates_annotated']:
                continue
            for component, metrics in item['components'].items():
                w = metrics['windows'][window]
                lines.append(f"| {window} | {item['label']} / {component} | {_format(w['rate_vision_fraction'], percent=True)} | "
                             f"{_format(w['vision_rate_min_deg_s'], decimals=1)} | {_format(w['vision_rate_max_deg_s'], decimals=1)} | "
                             f"{_format(w['vision_rate_p95_abs_deg_s'], decimals=1)} | {_format(w['phase_estimated_fraction'], percent=True)} |")
    lines += ['', '## Interpretation limits', '']
    lines += ['- '+note for note in comparison['limitations']]
    lines += ['', '![Coverage and fast-motion rate comparison](comparison.png)', '',
              'Reproduce with `python scripts/compare_experiments.py`; rerunning updates the report as branches finish. Source paths and SHA-256 hashes are recorded in `comparison.json`.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def plot_comparison(path, comparison, raw):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    available = [(name, item) for name, item in comparison['branches'].items() if item['available']]
    fig, axes = plt.subplots(4, 3, figsize=(17.5, 16), gridspec_kw={'height_ratios': [1.12, 1, 1, 1]})
    fig.subplots_adjust(left=.13, right=.97, top=.90, bottom=.09, wspace=.35, hspace=.51)
    fig.suptitle('Dual-camera tracking: phase, local evidence, and diagnostic angular rates', fontsize=16, y=.985)
    fig.text(.5, .955, 'Conditional comparisons only; native event coverage is not accuracy. N/A baseline rates were not re-estimated with differently gated evidence.',
             ha='center', fontsize=10)
    heatmaps = [('strict_phase_supported_fraction', 'Strict phase support (%)'),
                ('actual_local_pixel_support_fraction', 'Actual local pixel support (%)'),
                ('rate_vision_fraction', 'Local rate vision support (%)')]
    for col, (key, title) in enumerate(heatmaps):
        ax = axes[0, col]
        values = np.asarray([[np.nan if item['components'][c][key] is None else 100*item['components'][c][key]
                              for c in COMPONENTS] for _, item in available])
        cmap = plt.get_cmap('Blues').copy(); cmap.set_bad('#eeeeee')
        ax.imshow(values, vmin=0, vmax=100, cmap=cmap, aspect='auto')
        ax.set_xticks(range(3), LABELS, fontsize=9)
        ax.set_yticks(range(len(available)), [item['label'] for _, item in available] if col == 0 else ['']*len(available), fontsize=8)
        ax.set_title(title, fontsize=11, pad=10)
        for i in range(len(available)):
            for j in range(3):
                value = values[i, j]
                ax.text(j, i, 'N/A' if np.isnan(value) else f'{value:.1f}', ha='center', va='center',
                        color='white' if value > 60 else '#222222', fontsize=9)
    rate_branches = [(name, item) for name, item in available if item['rates_annotated']]
    palette = ['#2374AB', '#E58B25', '#8D4BB6', '#248557', '#555555', '#CB4550', '#8B671B']
    colors = {name: palette[i % len(palette)] for i, (name, _) in enumerate(rate_branches)}
    for row, (start, end) in enumerate(WINDOWS, 1):
        for col, component in enumerate(COMPONENTS):
            ax = axes[row, col]; values_for_bounds = []
            for lane, (name, item) in enumerate(rate_branches):
                report = raw[name]; shift = item['output_clock_origin_shift_s']
                frames = sorted(report['frames'], key=lambda f: f['time_s'])
                times = np.asarray([f['time_s']+shift for f in frames])
                rates = np.rad2deg(np.asarray([f['angular_velocity'][col] for f in frames], float))
                statuses = np.asarray([f['rate_status'][col] for f in frames])
                inside = (times >= start) & (times <= end)
                values_for_bounds.extend(rates[inside & np.isfinite(rates)].tolist())
                linestyle = '--' if item['evaluation_group'] == 'conditional_track_holdout' else '-'
                # NaNs explicitly break lines at unsupported events. Also
                # break long native timestamp holes rather than drawing across.
                vision = rates.copy(); vision[statuses != 'vision'] = np.nan
                prediction = rates.copy(); prediction[statuses != 'predicted'] = np.nan
                if len(times) > 1:
                    breaks = np.r_[False, np.diff(times) > .15]
                    vision[breaks] = np.nan; prediction[breaks] = np.nan
                ax.plot(times[inside], vision[inside], color=colors[name], lw=1.25, ls=linestyle)
                ax.plot(times[inside], prediction[inside], color=colors[name], lw=1., ls=':', alpha=.65)
                for left, right in item['components'][component]['phase_estimated_spans_s']:
                    left, right = max(start, left), min(end, right)
                    if right > left:
                        ax.axvspan(left, right, ymin=.012+lane*.018, ymax=.027+lane*.018,
                                   color=colors[name], alpha=.65, linewidth=0)
            if values_for_bounds:
                low, high = min(values_for_bounds), max(values_for_bounds)
                scale = max(high-low, 10.)
                ax.set_ylim(low-.25*scale, high+.08*scale)
            ax.set_xlim(start, end)
            ax.axhline(0, color='#bbbbbb', lw=.6, zorder=0)
            ax.grid(alpha=.2)
            ax.set_title(f'{LABELS[col]}: {start:g}–{end:g} s', fontsize=11)
            ax.set_xlabel('Original native-relative corrected time (s)', fontsize=8)
            ax.set_ylabel('Diagnostic rate (deg/s)', fontsize=9)
            ax.tick_params(labelsize=8)
    handles = [Line2D([0], [0], color=colors[name], lw=2,
                      ls='--' if item['evaluation_group'] == 'conditional_track_holdout' else '-',
                      label=item['label']) for name, item in rate_branches]
    if handles:
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .037), ncol=4, fontsize=9, frameon=False)
    fig.text(.5, .019, 'Thin colored lanes at panel bottoms mark phase_estimated spans (same branch order as legend). '
             'Solid/dashed curves: rate vision; dotted fragments: predicted.', ha='center', fontsize=8)
    fig.savefig(path, dpi=160, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', type=Path, default=ROOT/'experiments')
    parser.add_argument('--baseline', type=Path, default=BASELINE/'out/orientation_final/results.json')
    parser.add_argument('--output', type=Path, default=ROOT/'reports')
    parser.add_argument('--branches', nargs='+', default=list(DEFAULT_BRANCHES))
    args = parser.parse_args()
    branches, raw = load_comparison(args.experiments, args.baseline, args.branches)
    comparison = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'accuracy_validated': False,
                  'clock': 'original native-relative corrected time; report time_s + output_clock_origin_shift_s',
                  'windows_s': [list(w) for w in WINDOWS], 'branches': branches,
                  'limitations': [
                      'Full-data reference consistency is descriptive. Only rows with both exclusion assertions are grouped as conditional track holdout.',
                      'All refinements share the accepted full-data baseline initializer, including monocular and held-out branches; no independent from-scratch accuracy claim follows.',
                      'Reference pixels share source images with learned observations. Camera geometry, roll, clock bias, track identity and exposure blur can bias all branches together.',
                      'Pixel median and p95 apply to scored observations only; scored coverage and the 3 px fraction over all candidates include failure consequences.',
                      'Local rate can be identified in a disconnected phase component. Phase-estimated spans do not imply every later local angular rate is unobserved.',
                      'Rate vision is a conditional temporal nullspace test, not a covariance or accuracy bound. Roll additionally assumes locally pixel-supported pose samples.',
                      'The 30 Hz local cubic derivative is a diagnostic interpolant of saved metric knots. It does not increase temporal bandwidth or correct exposure integration.',
                      'Native camera events are correlated samples. Coverage fractions and plotted comparisons are descriptive, without statistical significance claims.',
                      'Coverage fractions use each branch\'s emitted events. Native mono refits use active-camera timestamps only and anchor phase at their first observed image. Raw native-event counts cannot be compared as independent trials.',
                      'When native mono refits are still pending, legacy mono results are labeled explicitly; their shared cam0 phase gauge can precede the first active-camera image and invalidate camera-benefit conclusions from phase coverage.',
                      'Dotted predicted-rate fragments can show large prior/interpolant excursions in unsupported intervals; they are excluded from the rate-vision speed tables and are not measured peak speeds.',
                      'The original baseline has no identically filtered rate annotation here; use the separately labeled KLT holdout control for comparable rate diagnostics.']}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    write_markdown(args.output/'comparison.md', comparison)
    if raw:
        plot_comparison(args.output/'comparison.png', comparison, raw)
    print(json.dumps({'completed_branches': list(raw), 'unavailable_branches': [k for k, v in branches.items() if not v['available']],
                      'outputs': [str(args.output/name) for name in ('comparison.json', 'comparison.md', 'comparison.png')]}, indent=2))


if __name__ == '__main__':
    main()
