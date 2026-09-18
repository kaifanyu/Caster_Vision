"""Build frozen measured textures from supported pre-failure native frames."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blurtrack import ROOT, BASELINE
from blurtrack.texture_atlas import build_atlases

def render_reference_contacts(output,config_path):
    """Show measured source views alongside the atlas; no inferred texture."""
    import json
    import cv2
    import numpy as np
    from dualcam.config import load_config,load_rig
    from dualcam.session import SelectedVideo
    from blurtrack.observations import crop_bounds
    out=Path(output);manifest=json.loads((out/'atlas_manifest.json').read_text())
    cfg=load_config(config_path);_,intrinsics,_=load_rig(cfg)
    report=json.loads(Path(manifest['baseline']).read_text())
    for ci in (0,1):
        records=[entry for entry in manifest['atlases'] if entry['camera_id']==ci]
        refs={}
        for entry in records:
            selected=np.linspace(0,len(entry['references'])-1,min(6,len(entry['references']))).round().astype(int)
            entry['contact_references']=[entry['references'][i] for i in selected]
            refs.update({r['source_frame']:None for r in entry['contact_references']})
        info=intrinsics[ci];name=records[0]['camera']
        maps=cv2.initUndistortRectifyMap(info['K'],info['dist'],None,info['K'],tuple(info['image_size']),cv2.CV_32FC1)
        bounds=crop_bounds(cfg['cameras'][name]['circle'],info['image_size'])
        video=SelectedVideo(report['videos'][ci],info['image_size'])
        try:
            for index in sorted(refs):
                frame=cv2.remap(video.read(index),*maps,cv2.INTER_LINEAR)
                x0,y0,x1,y1=bounds
                refs[index]=cv2.resize(frame[y0:y1,x0:x1],(360,360))
        finally:video.close()
        for entry in records:
            canvas=np.full((1105,1080,3),25,np.uint8)
            for i,r in enumerate(entry['contact_references']):
                x=(i%3)*360;y=(i//3)*420
                canvas[y+60:y+420,x:x+360]=refs[r['source_frame']]
                cv2.putText(canvas,f'{name} {entry["shell"]} #{r["source_frame"]}',(x+8,y+22),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
                cv2.putText(canvas,f't={r["time_s"]:.3f}s; pred blur {r["predicted_blur_px"]:.1f}px',(x+8,y+46),cv2.FONT_HERSHEY_SIMPLEX,.52,(240,240,240),1,cv2.LINE_AA)
            atlas=cv2.imread(str(Path(entry['path']).with_suffix('.png')))
            canvas[865:1105]=cv2.resize(atlas,(1080,240),interpolation=cv2.INTER_NEAREST)
            cv2.putText(canvas,f'Measured atlas; dark cells unknown; coverage {100*entry["valid_fraction"]:.1f}%',(8,858),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1,cv2.LINE_AA)
            cv2.imwrite(str(out/f'references_{name}_{entry["shell"]}.jpg'),canvas)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-cache',default=str(BASELINE/'out/joint_native_cache'))
    p.add_argument('--baseline',default=str(ROOT/'experiments/hybrid_dual/results.json'))
    p.add_argument('--output',default=str(ROOT/'experiments/blur_physics/atlas'))
    p.add_argument('--config',default=str(BASELINE/'config/rig.yaml'))
    p.add_argument('--height',type=int,default=192)
    p.add_argument('--width',type=int,default=512)
    p.add_argument('--references',type=int,default=24)
    p.add_argument('--max-blur-px',type=float,default=6.)
    p.add_argument('--preview-only',action='store_true')
    args=p.parse_args()
    if not args.preview_only:build_atlases(args.native_cache,args.baseline,args.output,config_path=args.config,
                   height=args.height,width=args.width,max_references=args.references,max_blur_px=args.max_blur_px)
    render_reference_contacts(args.output,args.config)

if __name__=='__main__':main()
