#!/usr/bin/env python3
"""Retune a saved shared trajectory without changing the underlying image evidence."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dualcam.config import write_json
from dualcam.fusion import FusionConfig
from dualcam.fused_workflow import refilter_rotation_report, write_fused_csv
from dualcam.workflow import _output_directory


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--accel-noise-deg-s2',type=float)
    p.add_argument('--measurement-std-deg',type=float)
    p.add_argument('--max-prediction-s',type=float)
    args=p.parse_args(argv)
    try:
        original=json.loads(args.results.read_text())
        cfg=FusionConfig.from_mapping(original.get('fusion_config'))
        overrides={k:v for k,v in [('accel_noise_deg_s2',args.accel_noise_deg_s2),
                                    ('rotation_measurement_std_deg',args.measurement_std_deg),
                                    ('max_prediction_s',args.max_prediction_s)] if v is not None}
        result=refilter_rotation_report(original,replace(cfg,**overrides))
        result['refilter_source']=str(args.results.resolve())
        output=_output_directory(args.output)
        write_json(output/'results.json',result)
        write_fused_csv(output/'results.csv',result['frames'])
    except (OSError,ValueError,TypeError) as exc:
        print(f'Refilter failed: {exc}',file=sys.stderr)
        return 2
    print(f'Trajectory: {output/"results.csv"}\nCoverage: {result["summary"]["coverage"]}')
    return 0 if result['success'] else 2


if __name__=='__main__':
    raise SystemExit(main())
