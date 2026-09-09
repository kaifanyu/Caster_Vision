"""Regressions for real-footage failures: wrong surfaces, tag switches and gaps."""
from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common.rotation import Rz
from common.track import track_features, KLTConfig
from scripts.calibrate_swivel_tags import robust_rotation_mean
from scripts.calibrate_wheel_face import fit_geometry
from swivel.geometry import SwivelGeometry, project_camera_points
from swivel.integration import integrate_roll
from swivel.masking import wheel_tracking_mask
from swivel.reference import ReferenceObservation
from swivel.tag import MultiArucoTagTracker, TagObservation, marker_object_points
from swivel.roll import RollEstimator, RollEstimate
from swivel.pipeline import run_pipeline
from swivel.adapter import adapt_swivel_series


def geometry():
    return SwivelGeometry(np.array([[1.,0,0],[0,0,-1],[0,1,0]]),np.array([0.,0.,2.]),
                          np.zeros(3),np.array([-.1,0.,0.]),np.array([0.,-1.,0.]),.3,.06)


K = np.array([[700.,0,320],[0,700.,240],[0,0,1.]])


def references(angles):
    return [ReferenceObservation(a is not None, np.nan if a is None else np.deg2rad(a)) for a in angles]


def test_missing_increment_does_not_become_a_complete_absolute_angle():
    phi, complete, segment, _, _ = integrate_roll(np.deg2rad([10,0,10]),[True,False,True],references([None]*4))
    np.testing.assert_allclose(np.rad2deg(phi),[0,10,10,20])
    assert complete.tolist() == [True,True,False,False]
    assert segment.tolist() == [0,0,1,1]


def test_reference_correction_propagates_to_future_frames():
    phi, complete, _, _, corrected = integrate_roll(np.deg2rad([10,0,10]),[True,False,True],references([0,10,20,30]),max_step_deg=30)
    np.testing.assert_allclose(np.rad2deg(phi),[0,10,20,30],atol=1e-12)
    assert complete.all()
    assert corrected.tolist() == [False,False,True,False]


def test_reference_cannot_restore_unseen_turn_count_after_long_gap():
    _, complete, _, _, corrected = integrate_roll(np.zeros(11),np.zeros(11,bool),references([0]+[None]*10+[20]),max_gap_frames=5,max_step_deg=30)
    assert not complete[-1] and not corrected.any()
    # Nor can a newly seen mark define frame-zero phase after an earlier loss.
    _, complete, _, _, _ = integrate_roll(np.deg2rad([0,10]),[False,True],references([None,10,20]))
    assert not complete[-1]


def test_multitag_switch_preserves_yaw_with_different_marker_mountings():
    cam = np.diag([1.,-1.,-1.])
    zeros = {0:np.eye(3),1:Rz(.7) @ Rotation.from_euler('x',.1).as_matrix()}
    tracker = MultiArucoTagTracker(K,.4,zero_rotations=zeros,R_car_from_camera=cam.T)
    def detection(key, angle):
        pts = marker_object_points(.4) @ (cam@Rz(angle)@zeros[key]).T + [0,0,2]
        uv,_=project_camera_points(pts,K)
        return [uv.astype(np.float32)[None]],np.array([[key]]),[]
    with patch.object(tracker.backend,'detect',side_effect=[detection(0,np.deg2rad(179)),detection(1,np.deg2rad(181))]):
        first=tracker.track(np.zeros((480,640,3),np.uint8));second=tracker.track(np.zeros((480,640,3),np.uint8))
    assert first.valid and second.valid
    assert np.rad2deg(second.psi-first.psi)==pytest.approx(2,abs=.05)
    assert second.marker_id==1


def test_multitag_rejects_uncalibrated_id_and_conflicting_yaw():
    tracker=MultiArucoTagTracker(K,.4,zero_rotations={0:np.eye(3),1:np.eye(3)})
    with patch.object(tracker.backend,'detect',return_value=([np.zeros((1,4,2),np.float32)],np.array([[37]]),[])):
        assert not tracker.track(np.zeros((480,640,3),np.uint8)).valid
    with patch.object(tracker.backend,'detect',return_value=([],None,[])), \
         patch.object(tracker.trackers[0],'track_corners',return_value=TagObservation(True,0,psi=0,psi_wrapped=0,reprojection_error_px=.1)), \
         patch.object(tracker.trackers[1],'track_corners',return_value=TagObservation(True,1,psi=.5,psi_wrapped=.5,reprojection_error_px=.1)):
        result=tracker.track(np.zeros((480,640,3),np.uint8))
    assert not result.valid and 'disagree' in result.failure_reason


def test_rotation_calibration_rejects_pose_outliers():
    rng=np.random.default_rng(3);truth=Rz(.8)
    matrices=[truth@Rotation.from_rotvec(rng.normal(0,.002,3)).as_matrix() for _ in range(20)]
    matrices.extend([Rz(-1.5),Rz(2.5)])
    mean,report=robust_rotation_mean(matrices)
    assert report['retained']==20
    assert (Rotation.from_matrix(mean.T@truth)).magnitude()<.003
    with pytest.raises(ValueError):robust_rotation_mean(matrices[:2])


def test_white_face_mask_preserves_small_speckles_and_excludes_fork():
    im=np.full((480,640,3),220,np.uint8)
    cv2.circle(im,(260,180),2,(0,0,0),-1)
    cv2.rectangle(im,(300,150),(340,220),(20,20,20),-1)
    mask,_=wheel_tracking_mask(im,geometry(),K,0,inner_fraction=0,margin_px=0,
        config={'enabled':True,'white_face':{'enabled':True,'close_radius_px':5},'support_margin_px':0})
    assert mask[180,260] and not mask[180,320]


def test_marker_border_and_carrier_are_removed_from_mask():
    im=np.full((480,640,3),220,np.uint8)
    corners=np.array([[260,180],[300,180],[300,220],[260,220]],float)
    obs=TagObservation(True,0,corners_uv=corners,detected_corners={0:corners})
    mask,_=wheel_tracking_mask(im,geometry(),K,0,inner_fraction=0,margin_px=0,
        config={'enabled':True,'support_margin_px':0,
                'marker_polygons':{'0':[[-.8,-.8],[.8,-.8],[.8,1.5],[-.8,1.5]]}},observation=obs)
    assert not mask[175,280] and not mask[200,280] and mask[275,250]


def test_motion_prediction_follows_trail_face_and_roll():
    g=geometry();est=RollEstimator(g,K)
    first,second=g.radial_basis_zero();vectors=np.array([.2*first,.15*second])
    a=.1;b=.4;delta=.2;face=g.visible_face_sign(a)
    center,_=g.face_plane_camera(a,face)
    pixels,_=project_camera_points(vectors@g.R_camera_from_fork(a).T+center,K)
    projected,valid=est.predict_pixels(pixels,a,b,face,delta)
    center_b,_=g.face_plane_camera(b,face)
    moved=vectors@Rotation.from_rotvec(g.axle_zero_car*delta).as_matrix().T
    expected,_=project_camera_points(moved@g.R_camera_from_fork(b).T+center_b,K)
    assert valid.all();np.testing.assert_allclose(projected,expected,atol=1e-9)


def test_geometry_fit_recovers_shared_hub_offset_across_swivel():
    truth=geometry();wrong=replace(truth,hub_offset_zero_car=truth.hub_offset_zero_car+[.009,-.006,.007])
    annotations=[]
    for angle in [0.,.5,2.7]:
        face=truth.visible_face_sign(angle)
        points=truth.sidewall_boundary_car(angle,face,samples=24)
        uv,_=project_camera_points(truth.car_points_to_camera(points),K)
        center,_=truth.face_plane_camera(angle,face);center_uv,_=project_camera_points(center,K)
        annotations.append({'reviewed':True,'psi_rad':angle,'face_sign':face,'perimeter_uv':uv.tolist(),'face_center_uv':center_uv.tolist()})
    fitted,report=fit_geometry(wrong,K,annotations)
    np.testing.assert_allclose(fitted.hub_offset_zero_car,truth.hub_offset_zero_car,atol=1e-6)
    assert report['rms_px']<1e-4
    annotations[0]['reviewed']=False
    with pytest.raises(ValueError):fit_geometry(wrong,K,annotations)


def test_pipeline_reference_repair_does_not_create_false_interval_velocity():
    def estimate(degrees):
        valid=degrees is not None
        return RollEstimate(np.deg2rad(degrees) if valid else np.nan,np.eye(3) if valid else None,
            np.eye(3) if valid else None,np.ones(8,bool),np.ones(8,bool),np.zeros(8),0.,1,1.,1.,'mock')
    with patch('swivel.pipeline.ArucoTagTracker') as tag,patch('swivel.pipeline.RollEstimator') as roll,patch('swivel.pipeline.detect_reference_phase') as ref:
        tag.return_value.track.return_value=TagObservation(True,0,psi=0,psi_wrapped=0,reprojection_error_px=0)
        roll.return_value.estimate.side_effect=[estimate(10),estimate(None),estimate(10)]
        ref.side_effect=references([0,10,20,30])
        result=run_pipeline([np.zeros((480,640,3),np.uint8)]*4,K=K,geometry=geometry(),marker_size_m=.4,reference_dot_config={'enabled':True,'max_step_deg':30},fps=30)
    np.testing.assert_allclose(np.rad2deg(result.phi),[0,10,20,30],atol=1e-10)
    frames=adapt_swivel_series(result.series,r_eff=.3)
    assert not frames[1].raw['roll_valid']
    assert frames[2].omega_roll==pytest.approx(np.deg2rad(10)*30)


def test_initial_klt_prediction_is_validated():
    im=np.zeros((100,100),np.uint8)
    with pytest.raises(ValueError,match='initial_uv_curr'):
        track_features(im,im,np.array([[50,50]],float),initial_uv_curr=np.array([[np.nan,0]]))


def test_reference_color_is_not_removed_by_white_material_mask():
    from synthetic.generate_swivel import render_sequence, tracking_trajectory
    from tests.test_swivel_pipeline import _geometry
    rendered=render_sequence(tracking_trajectory(3,60,delta_phi_deg=2,delta_psi_deg=.2))
    result=run_pipeline(rendered.frames,K=rendered.camera.K,geometry=_geometry(rendered),
        marker_size_m=rendered.geometry.tag_size_m,reference_dot_config={'enabled':True},
        mask_config={'enabled':True,'white_face':{'enabled':True,'max_saturation':50}},fps=60)
    assert all(o.valid for o in result.reference_observations)


def test_circular_fallback_can_pass_pipeline_with_pixel_evidence():
    from synthetic.generate_swivel import render_sequence, tracking_trajectory
    from tests.test_swivel_pipeline import _geometry
    rendered=render_sequence(tracking_trajectory(3,60,delta_phi_deg=2,delta_psi_deg=.2))
    with patch('swivel.roll.ransac_kabsch',return_value=None):
        result=run_pipeline(rendered.frames,K=rendered.camera.K,geometry=_geometry(rendered),
            marker_size_m=rendered.geometry.tag_size_m,reference_dot_config={'enabled':False},fps=60)
    assert all(q['roll_valid'] for q in result.interval_quality)
    assert all(e.method=='circular' for e in result.roll_estimates)
    np.testing.assert_allclose(np.rad2deg(np.diff(result.phi)),2,atol=.2)


def test_prediction_respects_distorted_pixel_coordinates():
    g=geometry();dist=np.array([.2,-.1,0.,0.,0.]);est=RollEstimator(g,K,dist=dist)
    first,_=g.radial_basis_zero();center,_=g.face_plane_camera(0,1)
    point=.2*first@g.R_camera_from_fork(0).T+center
    pixels,_=cv2.projectPoints(point[None],np.zeros(3),np.zeros(3),K,dist)
    predicted,valid=est.predict_pixels(pixels.reshape(-1,2),0,0,1,0)
    assert valid.all();np.testing.assert_allclose(predicted,pixels.reshape(-1,2),atol=1e-5)
