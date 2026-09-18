"""Create a reviewable index of completed physical-blur experiments."""
from pathlib import Path
import argparse
import html
import json
import math


def number(value, digits=3):
    return 'unavailable' if value is None else f'{value:.{digits}f}'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'experiments/blur_physics')
    p.add_argument('--pattern',default='final_*')
    args=p.parse_args()
    results=[]
    for folder in sorted(args.root.glob(args.pattern)):
        path=folder/'results.json'
        if path.is_file():results.append((folder,json.loads(path.read_text())))
    if not results:raise ValueError('No completed experiment results found')
    header=['Experiment','Status','Candidate rate (deg/s)','Prior rate (deg/s)',
            'Common hypothesis pixels','Held-out prior RMSE','Held-out candidate RMSE']
    rows=[];cards=[];records=[]
    for folder,r in results:
        m=r['metadata'];best=r.get('best_parameters_rad')
        speed=None if best is None else math.degrees(best[1])
        prior=math.degrees(m['baseline_parameters_rad'][1])
        old=r.get('baseline_common_pixel_score',{}).get('test',{})
        new=r.get('fitted_common_pixel_score',{}).get('test',{})
        row=[folder.name,r['status'],number(speed,2),number(prior,2),str(r.get('common_comparison_pixels',0)),number(old.get('rmse'),5),number(new.get('rmse'),5)]
        rows.append(row)
        record=dict(experiment=folder.name,evidence=f'{folder.name}/results.json',comparison=f'{folder.name}/comparison.png',
            status=r['status'],candidate_rate_deg_s=speed,prior_rate_deg_s=prior,
            common_hypothesis_pixels=r.get('common_comparison_pixels',0),common_training_pixels=r.get('common_training_pixels'),
            common_heldout_pixels=r.get('common_heldout_pixels'),prior_rmse=old.get('rmse'),candidate_rmse=new.get('rmse'),
            prior_comparison_heldout_pixels=old.get('pixels',0),camera_support=r.get('camera_support'),
            runtime_s=r.get('total_runtime_s'),gpu=r.get('gpu'),accuracy_validated=False)
        records.append(record)
        name=html.escape(folder.name)
        detail=f"{r.get('rejection_reason','No independent angular ground truth.')} Held-out comparison uses {old.get('pixels',0)} pixels."
        cards.append(f'<article id="{name}"><h2>{name}</h2><p>{html.escape(detail)}</p>'
            f'<p><a href="{name}/results.json">Full evidence JSON</a></p>'
            f'<img src="{name}/comparison.png" alt="Measured and exposure-rendered images for {name}" loading="lazy"></article>')
    intro=('The accepted trajectory is unchanged. These are local, constant-speed, fixed-roll experiments on existing recordings. '
           'Candidate means a photometric hypothesis among searched basins, not a validated physical rate. Ambiguous cases export no chosen rate. '
           'The 12-pixel-reference atlas contains existing blur and can bias speed. '
           'RMSE is linear-light color contrast on identical held-out pixels for each prior/candidate pair; supports differ between experiments, '
           'so RMSE values across rows must not be ranked as though measured on the same pixels.')
    table='| '+' | '.join(header)+' |\n|'+'|'.join(['---']*len(header))+'|\n'
    for row in rows:table+='| '+' | '.join(row)+' |\n'
    markdown='# Physical blur experiment evidence\n\n'+intro+'\n\n'+table+'\n'
    for folder,r in results:
        markdown+=f'- [{folder.name} comparison]({folder.name}/comparison.png), [evidence]({folder.name}/results.json)\n'
    (args.root/'summary.md').write_text(markdown,encoding='utf-8')
    (args.root/'summary.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    head=''.join('<th>'+html.escape(x)+'</th>' for x in header)
    body=''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
    page+='<title>Physical blur experiments</title><style>body{font:16px/1.5 system-ui,sans-serif;margin:32px auto;padding:0 20px;max-width:1500px;color:#243244;background:#f5f7fa}h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{padding:10px;border:1px solid #dce2e9;text-align:left}th{background:#e8eef5}article{margin-top:32px;padding:20px;background:white;border:1px solid #dce2e9;border-radius:8px}img{width:100%;height:auto}a{color:#155bb1}.scroll{overflow-x:auto}</style>'
    page+='<h1>Physical blur experiments on the existing recordings</h1><p>'+html.escape(intro)+'</p>'
    page+='<p><a href="../../BLUR_FIT_RESULTS.md">Interpretation and limitations</a> · <a href="../../CAMERA_SETTINGS.md">Recording settings</a></p>'
    if (args.root.parent/'physics_axis_preview/orientation_3d.html').exists():
        page+='<p><a href="../physics_axis_preview/orientation_3d.html">Open simulated-axis comparison</a> · <a href="../physics_axis_preview/physics_axes.mp4">Two-camera axis-overlay video</a></p>'
    page+='<div class="scroll"><table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div>'
    page+=''.join(cards)+'</html>'
    (args.root/'index.html').write_text(page,encoding='utf-8')
    print(f'Wrote {len(results)} completed experiments to {args.root}/index.html')


if __name__=='__main__':main()
