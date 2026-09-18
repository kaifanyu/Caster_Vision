"""Image evidence, native clocks, and shared-state fusion regressions."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from dualcam.fusion import (FusionConfig, IncrementProposal, KeyframeView, SharedMotionFilter,
                            fit_visual_pose, visible_projection)
from dualcam.fused_workflow import motion_inputs, run_fused_motion, render_fused, refilter_rotation_report
from dualcam.config import load_config, write_yaml
from dualcam.model import rotation_x, rotation_z
from dualcam.session import SelectedVideo
from ballrot.track import TrackMatches
from tests.test_solver import setup_scene
from tests.test_workflow import rig_files


class SharedFilterTests(unittest.TestCase):
    def test_refilter_does_not_create_missing_visual_evidence_or_mutate_input(self):
        frames=[{'time_s':t,'camera':'c920','source_frame':i,
                 'visual':None if i!=1 else {'success':True,'model':'enclosing_sphere_rotation',
                                             'components':[0,1],'angles':[.01,.02,0]}}
                for i,t in enumerate([0.,.03,.8])]
        original={'kind':'fused_motion','measurement_source':'rotation','F':np.eye(3).tolist(),
                  'initial_roll_deg':0.,'frames':frames,'summary':{'per_camera':{}}}
        result=refilter_rotation_report(original,FusionConfig())
        self.assertEqual(result['frames'][1]['status'].tolist(),['vision','vision','predicted'])
        self.assertEqual(result['frames'][-1]['status'].tolist(),['unresolved']*3)
        self.assertNotIn('status',original['frames'][1])
        self.assertFalse(result['frames'][-1]['filter_update_accepted'])
        with self.assertRaises(ValueError):
            refilter_rotation_report({**original,'measurement_source':'metric'},FusionConfig())

    def test_two_native_camera_streams_update_one_roll_and_independent_spins(self):
        filt = SharedMotionFilter([0., 0., 0.], 0.)
        events = sorted([(i/30, [0, 1]) for i in range(1, 61)]
                        +[(i/30+.016, [0, 2]) for i in range(1, 61)])
        for t, ids in events:
            filt.predict(t)
            truth = np.array([.7*t, -.2*t, .4*t])
            self.assertTrue(filt.update(ids, truth[ids], np.eye(2)*1e-5))
        np.testing.assert_allclose(filt.x[:3], truth, atol=.006)
        np.testing.assert_allclose(filt.x[3:], [.7, -.2, .4], atol=.02)

    def test_one_shell_does_not_invent_the_other_spin_and_long_gap_hides_predictions(self):
        filt = SharedMotionFilter([0., 0., 0.], 0.)
        for i in range(1, 31):
            filt.predict(i/30)
            filt.update([0, 1], [.1*i/30, .2*i/30], np.eye(2)*1e-4)
        snap = filt.snapshot([0, 1])
        self.assertEqual(snap['status'].tolist(), ['vision', 'vision', 'unresolved'])
        self.assertFalse(snap['turn_count_valid'][2])
        filt.predict(2.)
        self.assertTrue(np.isnan(filt.snapshot()['angles']).all())
        # A real recovered orientation cannot establish the missed turn count.
        self.assertTrue(filt.update([0, 1], [.2, .4], np.eye(2)*1e-4))
        self.assertFalse(filt.snapshot([0, 1])['turn_count_valid'][:2].any())

    def test_bad_camera_innovation_is_rejected_without_resetting_other_measurements(self):
        filt = SharedMotionFilter([0., 0., 0.], 0.)
        filt.predict(.03)
        self.assertTrue(filt.update([0, 2], [.01, .02], np.eye(2)*1e-5))
        before = filt.x.copy()
        self.assertFalse(filt.update([0, 1], [2., 2.], np.eye(2)*1e-5))
        np.testing.assert_array_equal(filt.x, before)
        with self.assertRaises(ValueError):
            filt.predict(.02)
        with self.assertRaises(ValueError):
            filt.update([0], [0], [[-1]])

    def test_full_revolution_is_not_wrapped_into_a_false_velocity_jump(self):
        filt = SharedMotionFilter([0., 0., 0.], 0.)
        for i in range(1, 181):
            t = i/30; filt.predict(t)
            self.assertTrue(filt.update([0, 1, 2], [1.4*t, 0., 0.], np.eye(3)*1e-4))
        self.assertGreater(filt.x[0], 2*np.pi)
        self.assertTrue(filt.snapshot([0, 1, 2])['turn_count_valid'].all())


class VisualFusionTests(unittest.TestCase):
    def test_camera_rotation_measurements_share_one_calibrated_frame(self):
        cameras, F, pivot = setup_scene()
        q0 = np.array([.4,0,0]); truth=np.array([6.5,.3,-.2])
        frame=np.zeros((32,32,3),np.uint8); masks={'top':np.ones((32,32),bool),'bottom':np.ones((32,32),bool)}
        for camera in cameras:
            tracker=IncrementProposal(camera,F,(960,540,100),q0)
            tracker.tracker.initialize(frame,masks)
            tracker.previous=frame; tracker.masks=masks
            temporal=[]
            for h in (0,1):
                world=F @ rotation_x(truth[0]) @ rotation_z(truth[1+h])
                relative=camera['R'] @ world @ (F @ rotation_x(q0[0])).T @ camera['R'].T
                temporal.append(SimpleNamespace(prediction=relative,
                    update=lambda *args, pose=relative, **kwargs:(pose,True,{'status':'keyframe'})))
            tracker.temporal=temporal
            empty=TrackMatches.empty()
            with patch.object(tracker.tracker,'track_pair',return_value={'top':empty,'bottom':empty}):
                tracker.observe(frame,masks,1,.03)
            measurement=tracker.measurement(truth+.01)
            self.assertIsNotNone(measurement)
            np.testing.assert_allclose(measurement['angles'],truth,atol=1e-10)
            tracker.valid[:]=False
            self.assertIsNone(tracker.measurement(truth))

    def scene(self):
        cameras, F, pivot = setup_scene()
        rng = np.random.default_rng(94)
        shells = np.repeat([0, 1], 1000)
        points = rng.normal(size=(2000, 3)); points /= np.linalg.norm(points, axis=1)[:, None]
        points[:, 2] = abs(points[:, 2])*(1-2*shells)
        return cameras, F, pivot, points, shells

    def test_both_oblique_views_recover_the_same_mixed_motion_from_known_points(self):
        cameras, F, pivot, points, shells = self.scene()
        truth = np.array([.12, -.1, .2])
        for camera in cameras:
            uv, visible = visible_projection(camera, F, pivot, truth, shells, points, .1, .02)
            take = np.concatenate([np.flatnonzero(visible & (shells == h))[:25] for h in (0, 1)])
            result = fit_visual_pose(camera, F, pivot, .1, .02, points[take], shells[take], uv[take], truth+.03)
            self.assertTrue(result['success'], result)
            np.testing.assert_allclose(result['angles'], truth, atol=1e-5)
            self.assertGreater(np.linalg.eigvalsh(result['covariance']).min(), 0)

    def test_unobserved_shell_is_not_a_visual_measurement_and_wrong_matches_fail(self):
        cameras, F, pivot, points, shells = self.scene()
        q = np.array([.1, .2, .3]); camera = cameras[0]
        uv, visible = visible_projection(camera, F, pivot, q, shells, points, .1, .02)
        take = np.flatnonzero(visible & (shells == 0))[:30]
        good = fit_visual_pose(camera, F, pivot, .1, .02, points[take], shells[take], uv[take], q+.01)
        self.assertTrue(good['success'], good)
        self.assertEqual(good['components'].tolist(), [0, 1])
        bad_uv = uv[take].copy(); bad_uv[:20] += np.random.default_rng(3).normal(0, 50, (20, 2))
        bad = fit_visual_pose(camera, F, pivot, .1, .02, points[take], shells[take], bad_uv, q)
        self.assertFalse(bad['success'])

    def test_keyframe_recovers_original_landmark_ids_after_an_unmatched_image(self):
        camera = {'K': np.array([[300.,0,160],[0,300,120],[0,0,1.]]), 'R': np.eye(3), 't': np.zeros(3)}
        F = np.diag([1.,-1.,-1.]); pivot = np.array([0.,0.,.5])
        gray = np.zeros((240,320), np.uint8)
        rng = np.random.default_rng(18)
        gray[75:165,115:205] = rng.integers(0,256,(90,90),dtype=np.uint8)
        mask = np.zeros_like(gray, bool); mask[80:160,120:200] = True
        masks = {'top':mask, 'bottom':np.zeros_like(mask)}
        view = KeyframeView(camera,F,pivot,.1,.02, config=replace(FusionConfig(),max_points=30))
        view.promote(gray,masks,np.zeros(3),0)
        ids = view.latest.identifiers.copy()
        blank = np.zeros_like(gray)
        self.assertIsNone(view.observe(blank,masks,np.zeros(3)))
        recovered = view.observe(gray,masks,np.zeros(3))
        self.assertIsNotNone(recovered)
        self.assertGreaterEqual(len(recovered['identifiers']),8)
        self.assertTrue(np.isin(recovered['identifiers'],ids).all())
        self.assertEqual(view.latest.index,0)  # failed/predicted images never promoted


class NativeInputTests(unittest.TestCase):
    def test_two_mp4_files_produce_one_trajectory_and_both_overlays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config_path,cfg,_,_,_=rig_files(root)
            actual=load_config()
            for name in ('c920','brio101'):
                cfg['cameras'][name]['circle']=[960,540,200]
                cfg['cameras'][name]['segment']=actual['cameras'][name]['segment']
            write_yaml(config_path,cfg)
            frame=np.full((1080,1920,3),220,np.uint8)
            for side in (-1,1):
                color=(0,0,255) if side==1 else (0,180,0)
                for y in (25,55,85,115):
                    for x in (-105,-70,-35,0,35,70,105):
                        center=np.array([960+x,540+side*y])
                        cv2.fillConvexPoly(frame,np.array([[-6,6],[6,6],[0,-6]])+center,color)
            paths=[]; stamps=[]
            for ci,name in enumerate(('c920','brio101')):
                video=root/f'{name}.mp4'; paths.append(video)
                writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30.,(1920,1080))
                if not writer.isOpened(): self.skipTest('MP4 encoder unavailable in this OpenCV build')
                for _ in range(3): writer.write(frame)
                writer.release()
                timestamp=root/f'{name}.csv'; stamps.append(timestamp)
                timestamp.write_text('frame_index,timestamp_s\n'+''.join(f'{i},{i/30+ci*.016}\n' for i in range(3)))
            report=run_fused_motion(config_path,root/'result',initial_roll_deg=0,
                                    videos=paths,timestamps=stamps,progress=None)
            self.assertTrue(report['success'],report['summary'])
            self.assertEqual(report['summary']['native_images'],6)
            self.assertEqual(report['measurement_source'],'rotation')
            self.assertTrue((root/'result/results.csv').is_file())
            self.assertEqual(len(report['frames']),6)
            render_fused(report,cfg,root/'overlays')
            for name in ('c920','brio101'):
                video=cv2.VideoCapture(str(root/f'overlays/{name}_axes.avi'))
                try:
                    self.assertTrue(video.isOpened())
                    self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)),3)
                finally: video.release()

    def test_explicit_video_path_can_ignore_an_unrelated_adjacent_session_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/'c920.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),30.,(32,32))
            self.assertTrue(writer.isOpened())
            for _ in range(3): writer.write(np.zeros((32,32,3),np.uint8))
            writer.release()
            (root/'session.json').write_text('{"cameras":{"c920":{"video_segments":[{"path":"unrelated.avi","start_frame":0,"frame_count":3}]}}}')
            source=SelectedVideo(path,[32,32],use_manifest=False)
            try: self.assertEqual(source.read(2).shape,(32,32,3))
            finally: source.close()

    def test_external_mp4s_keep_all_staggered_frames_and_use_declared_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); paths=[]
            for ci in range(2):
                path=root/f'{ci}.csv'; paths.append(path)
                path.write_text('frame_index,timestamp_s\n'+''.join(f'{i},{i/30+ci*.02}\n' for i in range(10)))
            cfg={'timing':{'brio_offset_s':-.004}}
            inputs=motion_inputs(cfg,videos=[root/'c920.mp4',root/'brio.mp4'],timestamps=paths)
            self.assertEqual(len(inputs['events']),20)
            self.assertAlmostEqual(inputs['events'][1][0],.016)
            self.assertEqual([e[2] for e in inputs['events'] if e[1]==0],list(range(10)))
            self.assertTrue(str(inputs['video_paths'][0]).endswith('.mp4'))
            with self.assertRaises(ValueError):
                motion_inputs(cfg,videos=[root/'a'],timestamps=paths)


if __name__ == '__main__':
    unittest.main()
