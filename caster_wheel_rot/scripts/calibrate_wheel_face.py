#!/usr/bin/env python
"""Review a moving wheel outline and fit shared geometry from manual annotations.

annotate: C selects face-center placement; R selects perimeter clicks; U undoes
a rim click; Enter accepts a frame with >=5 perimeter clicks and a center;
Esc cancels without overwriting annotations. Frames are undistorted first.
fit: camera, swivel zero, axle direction and width stay fixed unless --fit-width
is explicitly requested with diverse poses. --fit-radius is likewise optional.
preview: render the current geometry/mask over a complete clip, without fitting.
mask: click a polygon around one marker and its carrier; store it in that
marker's coordinates so the exclusion follows the fork on subsequent frames.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares

from common.rotation import Rz
from swivel.geometry import SwivelGeometry, project_camera_points
from swivel.masking import wheel_tracking_mask
from swivel.roll import _disk_to_image_homography
from swivel.tag import ArucoTagTracker, MultiArucoTagTracker


def tracker_from_config(cfg, K, geometry):
    marker = cfg['swivel']['marker']
    kwargs = dict(dictionary=marker.get('dictionary','DICT_4X4_50'),
                  R_car_from_camera=geometry.R_car_from_camera,
                  max_reprojection_error_px=marker.get('max_reprojection_error_px',2.))
    if marker.get('zero_rotations'):
        return MultiArucoTagTracker(K,marker['size_m'],zero_rotations=marker['zero_rotations'],**kwargs)
    return ArucoTagTracker(K,marker['size_m'],marker_id=marker.get('id',0),
                          R_tag_to_car_zero=Rz(marker['psi0_rad']),**kwargs)


def fit_geometry(geometry, K, annotations, *, fit_radius=False, fit_width=False,
                 max_rms_px=3., max_adjustment_m=.025):
    if not annotations or any(not a.get('reviewed',False) for a in annotations):
        raise ValueError('fit requires reviewed annotations, not predicted template points')
    for a in annotations:
        points = np.asarray(a['perimeter_uv'],float)
        center = np.asarray(a['face_center_uv'],float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points)<5 or not np.isfinite(points).all() or center.shape!=(2,) or not np.isfinite(center).all():
            raise ValueError('each pose needs a finite face center and at least five perimeter points')
        if int(a['face_sign']) not in [-1,1] or not np.isfinite(a['psi_rad']):
            raise ValueError('invalid face sign or swivel angle')
    angles = np.array([a['psi_rad'] for a in annotations])
    spread = np.max(np.abs(np.angle(np.exp(1j*(angles[:,None]-angles[None,:])))))
    if (fit_radius or fit_width) and (len(annotations)<3 or spread<np.deg2rad(30)):
        raise ValueError('fitting radius/width requires >=3 poses spanning >=30 degrees')
    if fit_width and len({int(a['face_sign']) for a in annotations}) < 2:
        raise ValueError('fitting width requires annotations of both faces')
    initial = list(geometry.hub_offset_zero_car)
    if fit_radius: initial.append(geometry.wheel_radius)
    if fit_width: initial.append(geometry.wheel_width)
    initial = np.asarray(initial)
    low, high = initial-max_adjustment_m, initial+max_adjustment_m
    if fit_radius: low[3] = max(.001,low[3])
    if fit_width: low[-1] = max(0.,low[-1])
    def unpack(x):
        changes = {'hub_offset_zero_car':x[:3]}
        offset = 3
        if fit_radius:
            changes['wheel_radius'] = x[offset];offset+=1
        if fit_width: changes['wheel_width'] = x[offset]
        return replace(geometry,**changes)
    def residual(x):
        g = unpack(x); errors=[]
        for a in annotations:
            H, _ = _disk_to_image_homography(K,g,float(a['psi_rad']),int(a['face_sign']))
            inv = np.linalg.inv(H)
            conic = inv.T @ np.diag([1.,1.,-g.wheel_radius**2]) @ inv
            points = np.column_stack([a['perimeter_uv'],np.ones(len(a['perimeter_uv']))])
            cp = points @ conic
            signed = np.sum(cp*points,axis=1)/np.maximum(2*np.linalg.norm(cp[:,:2],axis=1),1e-12)
            errors.extend(signed)
            center,_ = g.face_plane_camera(a['psi_rad'],a['face_sign'])
            uv, valid = project_camera_points(center,K)
            if not valid: return np.full(sum(len(b['perimeter_uv'])+2 for b in annotations),1e6)
            errors.extend(uv-np.asarray(a['face_center_uv']))
        return np.asarray(errors)
    result = least_squares(residual,initial,bounds=(low,high),loss='soft_l1',f_scale=2.,max_nfev=400)
    rms = float(np.sqrt(np.mean(residual(result.x)**2)))
    rank = int(np.linalg.matrix_rank(result.jac))
    if not result.success or rms>max_rms_px or rank<len(initial):
        raise ValueError(f'geometry fit rejected: success={result.success}, RMS={rms:.3f}px, rank={rank}/{len(initial)}')
    if np.any(np.minimum(result.x-low,high-result.x)<1e-6):
        raise ValueError('geometry fit reached adjustment bound; review calibration and measurements')
    return unpack(result.x), {'rms_px':rms,'before_rms_px':float(np.sqrt(np.mean(residual(initial)**2))),
        'pose_count':len(annotations),'jacobian_rank':rank,'hub_offset0_car':result.x[:3].tolist(),
        'wheel_radius_m':unpack(result.x).wheel_radius,'wheel_width_m':unpack(result.x).wheel_width,
        'note':'Residual is annotation agreement, not independent angle accuracy.'}


def frames_with_pose(clip,cfg):
    g=SwivelGeometry.from_mapping(cfg);K=np.array(cfg['camera']['K']);dist=np.array(cfg['camera']['dist'])
    tracker=tracker_from_config(cfg,K,g);cap=cv2.VideoCapture(str(clip));maps=None
    if not cap.isOpened():raise ValueError(f'cannot open {clip}')
    fps=cap.get(cv2.CAP_PROP_FPS)
    try:
        index=0
        while True:
            ok,im=cap.read()
            if not ok:break
            if maps is None:maps=cv2.initUndistortRectifyMap(K,dist,None,K,(im.shape[1],im.shape[0]),cv2.CV_32FC1)
            im=cv2.remap(im,*maps,cv2.INTER_LINEAR)
            yield index,im,tracker.track(im),fps
            index+=1
    finally:cap.release()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'config.yaml')
    sub=p.add_subparsers(dest='command',required=True)
    ann=sub.add_parser('annotate');ann.add_argument('--clip',type=Path,required=True)
    ann.add_argument('--frames',type=int,nargs='+',required=True);ann.add_argument('--output',type=Path,required=True)
    fit=sub.add_parser('fit');fit.add_argument('--annotations',type=Path,required=True)
    fit.add_argument('--output-config',type=Path,required=True);fit.add_argument('--report',type=Path,required=True)
    fit.add_argument('--fit-radius',action='store_true');fit.add_argument('--fit-width',action='store_true')
    fit.add_argument('--max-rms-px',type=float,default=3.)
    prev=sub.add_parser('preview');prev.add_argument('--clip',type=Path,required=True)
    prev.add_argument('--output',type=Path,required=True);prev.add_argument('--stride',type=int,default=1)
    mask_parser=sub.add_parser('mask');mask_parser.add_argument('--clip',type=Path,required=True)
    mask_parser.add_argument('--frame',type=int,required=True);mask_parser.add_argument('--marker-id',type=int,required=True)
    mask_parser.add_argument('--output-config',type=Path,required=True)
    args=p.parse_args(argv);cv2.setNumThreads(2)
    cfg=yaml.safe_load(args.config.read_text());g=SwivelGeometry.from_mapping(cfg);K=np.array(cfg['camera']['K'])
    if args.command=='mask':
        target=None
        for index,im,obs,_ in frames_with_pose(args.clip,cfg):
            if index==args.frame:
                target=(im,obs);break
        if target is None:raise ValueError('mask frame lies outside the clip')
        im,obs=target
        corners=obs.detected_corners.get(args.marker_id)
        if corners is None:raise ValueError('requested marker is not detected in this frame')
        points=[];scale=min(1.,1200/im.shape[1]);name='Carrier polygon: click corners / U undo / Enter accept / Esc cancel'
        def mask_click(event,x,y,flags,param):
            if event==cv2.EVENT_LBUTTONDOWN:points.append([x/scale,y/scale])
        cv2.namedWindow(name);cv2.setMouseCallback(name,mask_click)
        try:
            while True:
                shown=im.copy();cv2.polylines(shown,[np.rint(corners).astype(np.int32)],True,(0,255,255),2)
                if len(points)>1:cv2.polylines(shown,[np.rint(points).astype(np.int32)],True,(255,0,255),2)
                cv2.imshow(name,cv2.resize(shown,None,fx=scale,fy=scale));key=cv2.waitKey(20)&255
                if key==27:raise KeyboardInterrupt('mask annotation cancelled')
                if key==ord('u') and points:points.pop()
                if key in [10,13] and len(points)>=3:break
        finally:cv2.destroyWindow(name)
        canonical=np.array([[-.5,.5],[.5,.5],[.5,-.5],[-.5,-.5]],np.float32)
        H=cv2.getPerspectiveTransform(np.asarray(corners,np.float32),canonical)
        polygon=cv2.perspectiveTransform(np.asarray(points,np.float32)[None],H)[0]
        if not np.isfinite(polygon).all():raise ValueError('carrier projection is degenerate')
        values=cfg['swivel'].setdefault('mask',{});values['enabled']=True
        values.setdefault('marker_polygons',{})[str(args.marker_id)]=polygon.tolist()
        for key in ['swivel_clip','ball_clip','car_track']:
            if cfg.get('input',{}).get(key):cfg['input'][key]=str((args.config.resolve().parent/cfg['input'][key]).resolve())
        args.output_config.parent.mkdir(parents=True,exist_ok=True)
        tmp=args.output_config.with_suffix(args.output_config.suffix+'.tmp');tmp.write_text(yaml.safe_dump(cfg,sort_keys=False));tmp.replace(args.output_config)
        return 0
    if args.command=='fit':
        data=json.loads(args.annotations.read_text())
        if data.get('coordinate_space')!='undistorted_pixels':raise ValueError('annotations must use undistorted pixels')
        if not np.allclose(data.get('K'),K):raise ValueError('annotation intrinsics differ from configuration')
        fitted,report=fit_geometry(g,K,data['poses'],fit_radius=args.fit_radius,fit_width=args.fit_width,max_rms_px=args.max_rms_px)
        updated=copy.deepcopy(cfg);geo=updated['swivel']['geometry'];geo['hub_offset0_car']=fitted.hub_offset_zero_car.tolist()
        geo['wheel_radius_m']=float(fitted.wheel_radius);geo['wheel_width_m']=float(fitted.wheel_width)
        for key in ['swivel_clip','ball_clip','car_track']:
            if updated.get('input',{}).get(key):updated['input'][key]=str((args.config.resolve().parent/updated['input'][key]).resolve())
        for path,text in [(args.output_config,yaml.safe_dump(updated,sort_keys=False)),(args.report,json.dumps(report,indent=2))]:
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text);tmp.replace(path)
        print(json.dumps(report,indent=2));return 0
    args.output.parent.mkdir(parents=True,exist_ok=True)
    poses=[];writer=None
    try:
        for index,im,obs,fps in frames_with_pose(args.clip,cfg):
            if args.command=='annotate' and index not in args.frames:continue
            if args.command=='preview' and index%max(1,args.stride):continue
            shown=im.copy()
            if obs.valid:
                mask,pr=wheel_tracking_mask(im,g,K,obs.psi,inner_fraction=cfg['swivel']['geometry'].get('sidewall_inner_fraction',.2),config=cfg['swivel'].get('mask'),observation=obs)
                shown[mask]=(shown[mask]*.75+np.array([0,180,0])*.25).astype(np.uint8)
                cv2.polylines(shown,[np.rint(pr.boundary_uv).astype(np.int32)],True,(255,255,0),2)
            cv2.putText(shown,f'frame {index} tag={obs.valid}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
            if args.command=='preview':
                if writer is None:
                    writer=cv2.VideoWriter(str(args.output),cv2.VideoWriter_fourcc(*'mp4v'),fps/max(1,args.stride),(im.shape[1],im.shape[0]))
                    if not writer.isOpened():raise OSError('could not open preview video')
                writer.write(shown);continue
            if not obs.valid:raise ValueError(f'frame {index} has no accepted fork pose; choose a visible calibrated tag')
            state={'mode':'rim','rim':[],'center':None};scale=min(1.,1200/im.shape[1]);name='Wheel: C center / R rim / U undo / Enter accept / Esc cancel'
            def click(event,x,y,flags,param):
                if event==cv2.EVENT_LBUTTONDOWN:
                    point=[x/scale,y/scale]
                    if state['mode']=='center':state['center']=point
                    else:state['rim'].append(point)
            cv2.namedWindow(name);cv2.setMouseCallback(name,click)
            while True:
                display=shown.copy()
                for uv in state['rim']:cv2.circle(display,tuple(np.rint(uv).astype(int)),5,(0,0,255),-1)
                if state['center'] is not None:cv2.drawMarker(display,tuple(np.rint(state['center']).astype(int)),(0,0,255),cv2.MARKER_CROSS,20,2)
                cv2.imshow(name,cv2.resize(display,None,fx=scale,fy=scale));key=cv2.waitKey(20)&255
                if key==27:raise KeyboardInterrupt('annotation cancelled; existing file preserved')
                if key==ord('c'):state['mode']='center'
                if key==ord('r'):state['mode']='rim'
                if key==ord('u') and state['rim']:state['rim'].pop()
                if key in [10,13] and len(state['rim'])>=5 and state['center'] is not None:break
            cv2.destroyWindow(name)
            poses.append({'frame':index,'psi_rad':obs.psi,'face_sign':g.visible_face_sign(obs.psi),
                          'face_center_uv':state['center'],'perimeter_uv':state['rim'],'reviewed':True})
        if args.command=='annotate':
            if len(poses)!=len(set(args.frames)):raise ValueError('some requested frames were outside the video')
            data={'clip':str(args.clip.resolve()),'coordinate_space':'undistorted_pixels','K':K.tolist(),'poses':poses}
            tmp=args.output.with_suffix(args.output.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(args.output)
    finally:
        if writer is not None:writer.release()
        if args.command=='annotate':cv2.destroyAllWindows()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
