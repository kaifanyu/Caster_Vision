import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from blurtrack.metric import initialize_metric, refine, exclude_reference_tracks


def physical_fixture():
    rng = np.random.default_rng(972)
    F = Rotation.from_euler('x',78,degrees=True).as_matrix()
    pivot = np.array([0.,0.,.6]); geo = dict(radius_m=.1,gap_m=.02,red_shell_sign=1)
    K = np.array([[900.,0,640],[0,900.,480],[0,0,1]])
    cameras = [dict(K=K,R=np.eye(3),t=np.zeros(3)),
               dict(K=K,R=Rotation.from_euler('y',18,degrees=True).as_matrix(),t=np.array([-.18,.015,.03]))]
    times = [np.linspace(0,1,31),np.linspace(.01,.99,29)]
    def pose(t): return np.array([.14+.12*t,-.5*t,.7*t])
    rows = []
    for ci in (0,1):
        for h in (0,1):
            for track in range(12):
                p = np.array([rng.uniform(-.6,.6),-.7,(1-2*h)*rng.uniform(.12,.6)])
                p /= np.linalg.norm(p)
                for fi,t in enumerate(times[ci]):
                    if ci==0 and h==1 and .3<t<.8: continue
                    alpha,red,green = pose(t)
                    B = F@Rotation.from_rotvec([alpha,0,0]).as_matrix()
                    center = pivot+B@np.array([0,0,(1-2*h)*.01])
                    P = center+.1*B@Rotation.from_rotvec([0,0,(red,green)[h]]).apply(p)
                    image = K@(cameras[ci]['R']@P+cameras[ci]['t'])
                    uv = image[:2]/image[2]+rng.normal(0,.03,2)
                    rows.append([ci,h,fi,track,*uv,1.])
    rows=np.asarray(rows)
    obs={k:rows[:,i].astype(int) for i,k in enumerate(('camera','shell','frame','track'))}
    obs.update(uv=rows[:,4:6],weight=rows[:,6])
    data=dict(times=times,observations=obs)
    knots=np.linspace(0,1,31); truth=np.array([pose(t) for t in knots])
    return data,cameras,F,pivot,geo,knots,truth


class MetricTests(unittest.TestCase):
    def test_brio_only_fit_anchors_phase_at_first_corrected_observation(self):
        from blurtrack.camera_ablation import prepare_camera_data
        data,cameras,F,pivot,geo,knots,truth=physical_fixture()
        data['times'][1]=data['times'][1]-.006
        working,guess,offset,shifts=prepare_camera_data(data,(1,),knots,truth,.006)
        first=working['times'][1][0]
        seed=initialize_metric(working,cameras,F,pivot,geo,knots,guess,offset,
                               np.rad2deg(np.interp(first,knots,truth[:,0])))
        problem,x,stages=refine(working,seed,cameras,F,pivot,geo,max_nfev=60,progress=None)
        self.assertAlmostEqual(problem.knots[0],first)
        self.assertEqual(len(seed['events']),len(data['times'][1]))
        recovered=problem.unpack(x)[0]
        expected=np.column_stack([np.interp(problem.knots,knots,truth[:,k]) for k in range(3)])
        expected[:,1:]-=expected[0,1:]
        self.assertLess(np.max(abs(np.rad2deg(recovered-expected))),.15)
        self.assertTrue(stages[-1]['converged'])

    def test_reference_exclusion_removes_whole_identity(self):
        obs = dict(camera=np.array([0,0,0,1]), shell=np.zeros(4,int),
                   track=np.array([1,1,2,1]),frame=np.array([0,1,0,0]),
                   uv=np.array([[10.,10.],[90.,90.],[50.,50.],[10.,10.]]),weight=np.ones(4))
        ref = {k:v[:1] for k,v in obs.items()}
        kept = exclude_reference_tracks(obs,ref)
        self.assertEqual(len(kept['uv']),2)
        np.testing.assert_array_equal(kept['track'],[2,1])

    def test_metric_fit_recovers_known_three_angle_motion_despite_camera0_green_gap(self):
        data,cameras,F,pivot,geo,knots,truth=physical_fixture()
        guess=truth.copy();guess[:,1:]+=np.deg2rad(1.5)*np.sin(np.pi*knots[:,None])
        seed=initialize_metric(data,cameras,F,pivot,geo,knots,guess,0.,np.rad2deg(truth[0,0]))
        problem,x,stages=refine(data,seed,cameras,F,pivot,geo,max_nfev=60,progress=None)
        recovered=problem.unpack(x)[0]
        expected=np.column_stack([np.interp(problem.knots,knots,truth[:,k]) for k in range(3)])
        self.assertLess(np.max(abs(np.rad2deg(recovered-expected))),.15)
        self.assertAlmostEqual(x[problem.timing_id],0.)
        self.assertTrue(stages[-1]['converged'])

    def test_material_ids_are_camera_and_shell_local_and_inputs_unchanged(self):
        data,cameras,F,pivot,geo,knots,truth=physical_fixture()
        uv=data['observations']['uv'].copy()
        seed=initialize_metric(data,cameras,F,pivot,geo,knots,truth,.006,8.)
        self.assertEqual(len(seed['keys']),48)
        np.testing.assert_array_equal(uv,data['observations']['uv'])
        self.assertTrue(np.isfinite(seed['points']).all())
        np.testing.assert_allclose(np.linalg.norm(seed['points'],axis=1),1.)


if __name__=='__main__':unittest.main()
