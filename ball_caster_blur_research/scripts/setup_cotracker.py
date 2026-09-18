"""Provision pinned official assets, or verify existing assets with --check."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='No downloads or git changes')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'model_manifest.json').read_text())
    repo = ROOT/'third_party/co-tracker'
    checkpoint = ROOT/'checkpoints/scaled_offline.pth'
    if not repo.exists():
        if args.check: raise SystemExit('Missing official repository; run without --check to provision')
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git','clone',manifest['repository'],str(repo)], check=True)
        subprocess.run(['git','-C',str(repo),'checkout','--detach',manifest['revision']], check=True)
    revision = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if revision != manifest['revision']:
        raise SystemExit('Existing checkout has a different revision; retain it and provision a separate pinned checkout')
    changes = subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).splitlines()
    changes = [line for line in changes if not (line.startswith('?? ') and line.endswith('/__pycache__/'))]
    if changes:
        raise SystemExit('Official checkout contains modifications; verify before inference')
    if not checkpoint.exists():
        if args.check: raise SystemExit('Missing checkpoint; run without --check to provision')
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        partial = checkpoint.with_suffix('.download')
        urllib.request.urlretrieve(manifest['checkpoint_url'], partial)
        if digest(partial) != manifest['checkpoint_sha256']:
            raise SystemExit('Checkpoint checksum mismatch; downloaded bytes were not installed')
        partial.replace(checkpoint)
    if digest(checkpoint) != manifest['checkpoint_sha256']:
        raise SystemExit('Checkpoint checksum mismatch')
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('Assets verified but CUDA is unavailable in this Python interpreter')
    report = dict(python=sys.version, executable=sys.executable, repository_revision=revision,
                  checkpoint_sha256=manifest['checkpoint_sha256'], cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(),
                  dependencies={name:importlib.metadata.version(name) for name in
                      ('torch','torchvision','numpy','scipy','opencv-python-headless','einops','PyYAML','matplotlib','imageio')})
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/environment.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
