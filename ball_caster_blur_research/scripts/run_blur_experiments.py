"""Reproduce the existing-recording physical-blur comparison matrix."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STRICT=[('red19_strict','red',19.2,'atlas_masked',[]),
        ('green11_strict','green',11.35,'atlas_masked',[]),
        ('red8_control','red',8.,'atlas_masked',[])]
SENSITIVITY=[('red19_atlas12','red',19.2,'atlas_12px_masked',[]),
             ('green11_atlas12','green',11.35,'atlas_12px_masked',[]),
             ('red21_atlas12','red',21.2,'atlas_12px_masked',[]),
             ('red19_quad17','red',19.2,'atlas_12px_masked',['--quadrature','17']),
             ('red19_exposure08','red',19.2,'atlas_12px_masked',['--exposure-scale','0.8']),
             ('red19_exposure12','red',19.2,'atlas_12px_masked',['--exposure-scale','1.2']),
             ('red19_c920','red',19.2,'atlas_12px_masked',['--cameras','c920']),
             ('red19_brio','red',19.2,'atlas_12px_masked',['--cameras','brio101']),
             ('red19_side192','red',19.2,'atlas_12px_masked',['--side','192'])]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',required=True,help='Fresh experiment folder prefix; existing results are refused')
    p.add_argument('--group',choices=['all','strict','sensitivity'],default='all')
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if not args.prefix or Path(args.prefix).name!=args.prefix:raise ValueError('Prefix must be a folder name, not a path')
    cases=STRICT+SENSITIVITY if args.group=='all' else STRICT if args.group=='strict' else SENSITIVITY
    for name,shell,center,atlas,extra in cases:
        folder=ROOT/'experiments/blur_physics'/f'{args.prefix}_{name}'
        if (folder/'results.json').exists():raise ValueError(f'Existing results: {folder}; choose a fresh prefix')
    for name,shell,center,atlas,extra in cases:
        folder=ROOT/'experiments/blur_physics'/f'{args.prefix}_{name}'
        print(f'RUN {folder.name}',flush=True)
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts/fit_recorded_blur.py'),'--shell',shell,'--center',str(center),
            '--atlas',str(ROOT/'experiments/blur_physics'/atlas),'--output',str(folder),'--device',args.device,*extra],check=True,cwd=ROOT)


if __name__=='__main__':main()
