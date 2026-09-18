"""Direct, exposure-integrated photometric tests on the separated hemispheres.

Geometry/roll and measured texture are fixed. Only local shell phase and angular
speed are fitted. Photometric consistency is not a physical accuracy certificate.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn.functional as functional


def ray_shell_normals(camera, pixels, F, pivot, alpha, shell, geometry):
    """Exact visible outer-shell ray intersections in the unspun carrier frame."""
    pixels = np.asarray(pixels, float)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError('pixels must be Nx2')
    ca,sa=np.cos(alpha),np.sin(alpha)
    carrier=np.asarray(F)@np.array([[1.,0,0],[0,ca,-sa],[0,sa,ca]])
    R=np.asarray(camera['R']); eye=-R.T@np.asarray(camera['t'])
    rays=np.column_stack((pixels,np.ones(len(pixels))))@np.linalg.inv(camera['K']).T@R
    rays/=np.linalg.norm(rays,axis=1,keepdims=True)
    radius=geometry['radius_m']; gap=geometry['gap_m']; sign=geometry.get('red_shell_sign',1)
    depth=np.full(len(pixels),np.inf); owner=np.full(len(pixels),-1)
    normals=np.zeros((len(pixels),3)); incidence=np.zeros(len(pixels))
    for h in (0,1):
        center=np.asarray(pivot)+carrier@np.array([0.,0.,sign*(1-2*h)*gap/2])
        delta=eye-center; b=rays@delta; disc=b*b-delta@delta+radius*radius
        distance=-b-np.sqrt(np.maximum(disc,0.))
        world=(eye+rays*distance[:,None]-center)/radius
        local=world@carrier
        visible=(disc>0)&(distance>0)&(local[:,2]*sign*(1-2*h)>0)&(distance<depth)
        owner[visible]=h; depth[visible]=distance[visible]
        normals[visible]=local[visible]
        incidence[visible]=-np.sum(world[visible]*rays[visible],axis=1)
    valid=(owner==shell)&(incidence>.25)&(abs(normals[:,2])>.04)&(abs(normals[:,2])<.995)
    return normals,valid,incidence


def color_signal(rgb,shell=0):
    rgb=np.asarray(rgb)
    return rgb[...,shell]-(rgb[...,(shell+1)%3]+rgb[...,(shell+2)%3])/2


class BlurFitProblem:
    """GPU texture lookup and exposure quadrature; SciPy optimizes two angles.

    Atlas bins are centered at phi=(x+.5)*2pi/W, theta=(y+.5)*pi/(2H).
    Inputs normals are unspun carrier normals [frame,sample,pixel,3].
    Observations are linear-light red/green contrast, not restored images.
    """
    def __init__(self,atlases,normals,geometry_valid,sample_times,sample_weights,
                 camera_ids,observed,pixel_valid,reference_time_s,shell=0,
                 train_mask=None,device='cuda'):
        self.device=torch.device(device)
        if self.device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable')
        self.shell=shell; self.reference_time_s=float(reference_time_s)
        normals=np.asarray(normals,float); self.shape=normals.shape[:3]
        if normals.ndim!=4 or normals.shape[-1]!=3 or shell not in (0,1):
            raise ValueError('Need normals[F,S,N,3] and shell 0 or 1')
        self.nframes,self.samples,self.npixels=self.shape
        weights=np.asarray(sample_weights,float)
        if weights.shape!=(self.samples,) or np.any(weights<0) or not np.isclose(weights.sum(),1.):
            raise ValueError('Quadrature weights must be nonnegative and sum to one')
        def tensor(a):return torch.as_tensor(a,dtype=torch.float32,device=self.device)
        self.weights=tensor(weights)[None,:,None]
        self.dt=tensor(np.asarray(sample_times)-reference_time_s)
        if self.dt.shape!=(self.nframes,self.samples):raise ValueError('sample_times shape mismatch')
        self.phi=tensor(np.arctan2(normals[...,1],normals[...,0]))
        theta=np.arccos(np.clip(abs(normals[...,2]),0,1))
        self.geometry_valid=torch.as_tensor(geometry_valid,dtype=torch.bool,device=self.device)
        if tuple(self.geometry_valid.shape)!=self.shape:raise ValueError('geometry_valid shape mismatch')
        self.camera_ids=np.asarray(camera_ids,int)
        self.atlas_shape=np.asarray(atlases[0]['valid']).shape
        H,W=self.atlas_shape
        textures=[];valid=[]
        for atlas in atlases:
            if np.asarray(atlas['rgb']).shape!=(H,W,3) or np.asarray(atlas['valid']).shape!=(H,W):
                raise ValueError('All atlases must have equal dimensions')
            value=color_signal(atlas['rgb'],shell)
            textures.append(np.concatenate((value,value[:,:1]),axis=1))
            mask=np.asarray(atlas['valid'],float)
            valid.append(np.concatenate((mask,mask[:,:1]),axis=1))
        self.texture=tensor(np.asarray(textures)[self.camera_ids,None])
        self.texture_valid=tensor(np.asarray(valid)[self.camera_ids,None])
        self.v=tensor(2*(theta/(np.pi/2)*H-.5)/(H-1)-1)
        self.observed=tensor(observed)
        mask=np.asarray(pixel_valid,bool)&np.asarray(geometry_valid,bool).all(axis=1)
        if mask.shape!=(self.nframes,self.npixels):raise ValueError('pixel_valid shape mismatch')
        self.pixel_valid=torch.as_tensor(mask,device=self.device)
        if train_mask is None:
            train_mask=np.broadcast_to(np.arange(self.npixels)%4!=0,mask.shape)
        self.train=torch.as_tensor(np.asarray(train_mask,bool),device=self.device)&self.pixel_valid
        self.test=(~torch.as_tensor(np.asarray(train_mask,bool),device=self.device))&self.pixel_valid
        self.sample_times=np.asarray(sample_times)

    def _predict(self,parameters):
        phase=parameters[0]+parameters[1]*self.dt
        W=self.atlas_shape[1]
        u=torch.remainder((self.phi-phase[:,:,None])/(2*np.pi)*W-.5,W)
        grid=torch.stack((2*u/W-1,self.v),dim=-1)
        samples=functional.grid_sample(self.texture,grid,mode='bilinear',padding_mode='zeros',align_corners=True)[:,0]
        support=functional.grid_sample(self.texture_valid,grid,mode='bilinear',padding_mode='zeros',align_corners=True)[:,0]
        prediction=torch.sum(samples*self.weights,dim=1)
        valid=support.min(dim=1).values*self.geometry_valid.all(dim=1)
        return prediction,valid

    def predict(self,parameters):
        with torch.inference_mode():
            prediction,valid=self._predict(torch.as_tensor(parameters,dtype=torch.float32,device=self.device))
        return prediction.cpu().numpy(),(valid>.995).cpu().numpy()

    def _residual(self,prediction,mask):
        """Fit per-frame gain/offset to TRAIN pixels only, then apply everywhere."""
        fit=mask&self.train; weights=fit.float(); count=weights.sum(dim=1).clamp_min(1.)
        pm=(prediction*weights).sum(dim=1)/count
        om=(self.observed*weights).sum(dim=1)/count
        variance=((prediction-pm[:,None])**2*weights).sum(dim=1)/count
        covariance=((prediction-pm[:,None])*(self.observed-om[:,None])*weights).sum(dim=1)/count
        gain=(covariance/(variance+1e-7)).clamp(.5,2.)
        offset=(om-gain*pm).clamp(-.25,.25)
        residual=gain[:,None]*prediction+offset[:,None]-self.observed
        return residual,gain,offset,variance

    def score(self,parameters,mask=None):
        with torch.inference_mode():
            prediction,valid=self._predict(torch.as_tensor(parameters,dtype=torch.float32,device=self.device))
            eligible=self.pixel_valid if mask is None else self.pixel_valid&torch.as_tensor(mask,dtype=torch.bool,device=self.device)
            covered=eligible&(valid>.995)
            residual,gain,offset,variance=self._residual(prediction,covered)
            def metrics(use):
                values=residual[use]
                return {'pixels':int(use.sum()),'rmse':float(torch.sqrt((values**2).mean())) if len(values) else None,
                        'robust_loss':float((torch.sqrt(1+(values/.035)**2)-1).mean()*.035**2) if len(values) else None}
            training=metrics(covered&self.train);testing=metrics(covered&self.test)
            coverage=float(covered.sum()/eligible.sum().clamp_min(1))
            objective=(training['robust_loss'] if training['robust_loss'] is not None else 1.)+.02*(1-coverage)
            return dict(train=training,test=testing,coverage=coverage,eligible_pixels=int(eligible.sum()),
                        covered_pixels=int(covered.sum()),objective=objective,
                        gain=gain.cpu().tolist(),offset=offset.cpu().tolist(),
                        texture_variance=variance.cpu().tolist())

    def fit(self,seeds_radians,phase_radius_deg=20.,speed_radius_deg_s=200.,maxiter=60):
        started=time.perf_counter(); candidates=[]
        for seed in np.atleast_2d(seeds_radians):
            # Fixed material pixel set for each basin. Unknown texels never
            # contribute an ordinary image measurement; a coverage barrier
            # prevents candidates hiding their errors in unknown texture.
            _,valid=self.predict(seed)
            mask=self.pixel_valid&torch.as_tensor(valid,device=self.device)
            if int((mask&self.train).sum())<30:continue
            radii=np.deg2rad([phase_radius_deg,speed_radius_deg_s])
            def objective(z):
                p=torch.tensor(z,dtype=torch.float32,device=self.device,requires_grad=True)
                pred,support=self._predict(p)
                residual,_,_,_=self._residual(pred,mask)
                selected=mask&self.train
                data=(torch.sqrt(1+(residual/.035)**2)-1)*.035**2
                loss=data[selected].mean()+.05*((1-support)[selected]**2).mean()
                grad=torch.autograd.grad(loss,p)[0]
                return float(loss.detach()),grad.detach().cpu().numpy().astype(float)
            solution=minimize(objective,np.asarray(seed,float),jac=True,method='L-BFGS-B',
                bounds=list(zip(seed-radii,seed+radii)),options={'maxiter':maxiter,'ftol':1e-11,'gtol':1e-7,'maxls':25})
            params=solution.x
            candidates.append(dict(parameters_rad=params.tolist(),phase_deg=float(np.rad2deg(params[0])%360),
                speed_deg_s=float(np.rad2deg(params[1])),converged=bool(solution.success),iterations=solution.nit,
                message=str(solution.message),score=self.score(params),seed_rad=np.asarray(seed).tolist()))
        if not candidates:return {'best_parameters_rad':None,'candidates':[],'status':'insufficient_texture','runtime_s':time.perf_counter()-started}
        # A symmetric isolated exposure cannot choose time direction. Explicitly
        # evaluate the reversed path instead of trusting local solver progress
        # to reveal this exact ambiguity.
        centers=np.sum(self.sample_times*self.weights.detach().cpu().numpy()[0,:,0],axis=1)
        direction_ambiguous=False
        if np.ptp(centers)<1e-8:
            nominal=min(candidates,key=lambda c:c['score']['objective'])
            p=np.asarray(nominal['parameters_rad']);reverse=np.array([p[0]+2*p[1]*(centers[0]-self.reference_time_s),-p[1]])
            a,av=self.predict(p);b,bv=self.predict(reverse);use=av&bv&self.pixel_valid.cpu().numpy()
            if use.any() and np.max(abs(a[use]-b[use]))<2e-6:
                direction_ambiguous=abs(p[1])>1e-4
                candidates.append(dict(parameters_rad=reverse.tolist(),phase_deg=float(np.rad2deg(reverse[0])%360),
                    speed_deg_s=float(np.rad2deg(reverse[1])),converged=True,iterations=0,
                    message='Exact symmetric-exposure reversed-path hypothesis',score=self.score(reverse),seed_rad=reverse.tolist()))
        # Compare all refined hypotheses on exactly the same visible material
        # pixels. If intersection collapses, report an unresolved comparison.
        common=np.asarray(self.pixel_valid.cpu()).copy()
        for c in candidates:common&=self.predict(c['parameters_rad'])[1]
        common_train=int((common&self.train.cpu().numpy()).sum())
        common_test=int((common&self.test.cpu().numpy()).sum())
        enough=int(common.sum())>=max(50,int(self.pixel_valid.sum())//10) and common_train>=30
        if enough:
            for c in candidates:c['common_score']=self.score(c['parameters_rad'],common)
            candidates.sort(key=lambda c:c['common_score']['train']['robust_loss'] if c['common_score']['train']['robust_loss'] is not None else float('inf'))
        else:candidates.sort(key=lambda c:c['score']['objective'])
        best=candidates[0]
        alternatives=[c for c in candidates[1:] if abs(c['speed_deg_s']-best['speed_deg_s'])>30 or
            abs((c['phase_deg']-best['phase_deg']+180)%360-180)>8]
        def cost(c):return c.get('common_score',c['score'])['train']['robust_loss']
        gap=None if not alternatives or cost(best) is None else min(cost(c) for c in alternatives)/max(cost(best),1e-9)-1
        textured=max(best['score']['texture_variance'],default=0)>.00005
        finite_cost=cost(best) is not None and np.isfinite(cost(best))
        status='candidate' if enough and finite_cost and textured and best['converged'] and not direction_ambiguous and (gap is None or gap>.02) else 'ambiguous'
        return dict(best_parameters_rad=best['parameters_rad'],candidates=candidates,status=status,
                    common_comparison_pixels=int(common.sum()),common_comparison_available=enough,
                    common_training_pixels=common_train,common_heldout_pixels=common_test,
                    relative_cost_gap_to_distinct_hypothesis=gap,texture_informative=textured,
                    direction_ambiguous=direction_ambiguous,
                    accuracy_validated=False,turn_count_valid=False,
                    runtime_s=time.perf_counter()-started,device=str(self.device))
