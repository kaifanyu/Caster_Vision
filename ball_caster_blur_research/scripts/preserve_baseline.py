"""Snapshot the accepted pipeline without changing it or duplicating videos."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT.parent / 'ball_caster_dual_cam'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory():
    roots = ('ballrot', 'dualcam', 'scripts', 'config', 'calibration', 'docs', 'tests')
    files = [p for directory in roots for p in (PIPELINE / directory).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and 'archive' not in p.parts
             and p.suffix.lower() in ('.py', '.yaml', '.yml', '.json', '.md', '.html', '.txt', '.ini', '.sh')]
    files += [p for p in PIPELINE.iterdir() if p.is_file()
              and (p.suffix.lower() in ('.md', '.txt', '.ini', '.toml') or p.name == '.gitignore')]
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    destination = ROOT / 'baseline'
    manifest_path = destination / 'manifest.json'
    if args.verify:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        changed = [r['path'] for r in manifest['source_files'] + manifest['reference_artifacts']
                   if not (PIPELINE/r['path']).is_file() or digest(PIPELINE/r['path']) != r['sha256']]
        added = sorted(set(p.relative_to(PIPELINE).as_posix() for p in inventory())
                       - {r['path'] for r in manifest['source_files']})
        if changed or added:
            raise SystemExit(f'Baseline differs: changed/missing={changed}; new source={added}')
        print(f'Baseline unchanged: {len(manifest["source_files"])} source files and '
              f'{len(manifest["reference_artifacts"])} reference artifacts.')
        return
    if manifest_path.exists():
        raise SystemExit('Baseline already exists. Use --verify; do not overwrite the accepted snapshot.')
    destination.mkdir(parents=True, exist_ok=True)
    files = inventory()
    archive = destination / 'accepted_pipeline_source.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(PIPELINE).as_posix())
    def record(path):
        return {'path': path.relative_to(PIPELINE).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest(path)}
    artifacts = [PIPELINE / 'out/orientation_final' / name for name in
                 ('results.json', 'results.csv', 'strict_results.csv', 'bundle.npz',
                  'orientation_3d.html', 'replay/combined_tracking.mp4')]
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PIPELINE,
                            capture_output=True, text=True, check=False)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'pipeline': str(PIPELINE),
                'git_head': commit.stdout.strip(), 'note': 'Source snapshot includes uncommitted accepted work. '
                'Videos and output artifacts are referenced in place; none were changed or duplicated.',
                'source_archive': archive.name, 'source_archive_sha256': digest(archive),
                'source_files': [record(p) for p in files],
                'reference_artifacts': [record(p) for p in artifacts]}
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(f'Saved {len(files)} source files in {archive} ({archive.stat().st_size:,} bytes).')


if __name__ == '__main__':
    main()
