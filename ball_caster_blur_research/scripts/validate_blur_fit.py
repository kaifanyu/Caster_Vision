"""Independent synthetic validation for the direct exposure-integrated fitter.

Truth uses NumPy analytic texture evaluation and 61-point Gauss-Legendre
integration, never the fitted atlas sampler. Cameras, shell geometry, roll,
exposure and timestamps are known here. Real-video accuracy is not certified.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def analytic_texture(phi, theta, shell=0, *, flat=False, perturbation=0.):
    """Linear-light RGB, aperiodic within one turn and independent of the atlas."""
    phi, theta = np.broadcast_arrays(np.asarray(phi), np.asarray(theta))
    if flat:
        signal = np.full(phi.shape, .2)
    else:
        signal = (.2 + .075*np.sin(5*phi+2.7*theta+.3)
                  + .05*np.cos(11*phi-1.4*theta)
                  + .04*np.sin(19*phi+5*theta)
                  + .025*np.cos(2*phi+theta))
        signal += perturbation*np.sin(17*phi-7*theta+.9)
    rgb = np.full(phi.shape+(3,), .19)
    rgb[..., shell] += signal  # shell 0 red, shell 1 green
    return rgb


def look_at(origin, target):
    forward = np.asarray(target)-np.asarray(origin)
    forward /= np.linalg.norm(forward)
    right = np.cross([0., 1., 0.], forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack((right, down, forward))
    return R, -R@origin


@dataclass
class AnalyticScene:
    """Independent full-ray scene: radius 100 mm, gap 20 mm, carrier tilt 8 deg."""
    size: int = 160
    shell: int = 0

    def __post_init__(self):
        self.geometry = dict(radius_m=.1, gap_m=.02, red_shell_sign=1)
        self.F = Rotation.from_euler('y', 90, degrees=True).as_matrix()
        self.alpha = np.deg2rad(8.)
        self.B = self.F@Rotation.from_rotvec([self.alpha, 0, 0]).as_matrix()
        self.pivot = np.array([0., 0., .45])
        self.K = np.array([[self.size*1.35, 0, (self.size-1)/2],
                           [0, self.size*1.35, (self.size-1)/2], [0, 0, 1.]])
        self.cameras = [dict(K=self.K, R=np.eye(3), t=np.zeros(3))]
        R, t = look_at(np.array([.25, -.035, .03]), self.pivot)
        self.cameras.append(dict(K=self.K, R=R, t=t))

    def intersections(self, ci, pixels):
        """World-space ray/sphere roots, retaining only the nearest hemisphere."""
        camera = self.cameras[ci]
        rays = np.column_stack((pixels, np.ones(len(pixels))))@np.linalg.inv(camera['K']).T@camera['R']
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        origin = -camera['R'].T@camera['t']
        closest = np.full(len(pixels), np.inf)
        owner = np.full(len(pixels), -1, int)
        normals = np.zeros((len(pixels), 3))
        incidence = np.zeros(len(pixels))
        for shell, sign in enumerate((1, -1)):
            center = self.pivot+self.B[:, 2]*(sign*.01)
            offset = origin-center
            projection = rays@offset
            determinant = projection**2 - (offset@offset-.1**2)
            depth = -projection-np.sqrt(np.maximum(determinant, 0.))
            world_normal = (origin+depth[:, None]*rays-center)/.1
            normal = world_normal@self.B
            keep = ((determinant>0) & (depth>0) & (normal[:, 2]*sign>=0) & (depth<closest))
            closest[keep] = depth[keep]
            owner[keep] = shell
            normals[keep] = normal[keep]
            incidence[keep] = -(world_normal[keep]*rays[keep]).sum(axis=1)
        return normals, owner, incidence

    def pixels(self, ci, count=700, seed=721):
        yy, xx = np.indices((self.size, self.size))
        pixels = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
        normals, owner, incidence = self.intersections(ci, pixels)
        sign = 1-2*self.shell
        keep = (owner==self.shell) & (incidence>.35) & (normals[:, 2]*sign>.08) & (normals[:, 2]*sign<.96)
        selected = np.flatnonzero(keep)
        if count is not None and len(selected)>count:
            selected = np.random.default_rng(seed+ci).choice(selected, count, replace=False)
        return pixels[selected], normals[selected], incidence[selected]

    def atlas(self, width=720, height=180, *, flat=False, perturbation=0., missing=False):
        phi = (np.arange(width)+.5)*2*np.pi/width
        theta = (np.arange(height)+.5)*(np.pi/2)/height
        pp, tt = np.meshgrid(phi, theta)
        rgb = analytic_texture(pp, tt, self.shell, flat=flat, perturbation=perturbation)
        valid = np.ones((height, width), bool)
        if missing:
            valid[:, :int(width*.8)] = False
        return dict(rgb=rgb.astype(np.float32), valid=valid)

    def signal(self, normals, phase, speed, center_s, exposure_s, *, samples=61, flat=False):
        nodes, weights = np.polynomial.legendre.leggauss(samples)
        phi = np.arctan2(normals[:, 1], normals[:, 0])
        theta = np.arccos(np.clip(normals[:, 2]*(1-2*self.shell), -1, 1))
        value = np.zeros(len(normals))
        for node, weight in zip(nodes, weights):
            spin = phase+speed*(center_s+node*exposure_s/2)
            rgb = analytic_texture(phi-spin, theta, self.shell, flat=flat)
            other = [k for k in range(3) if k!=self.shell]
            value += (rgb[:, self.shell]-.5*rgb[:, other].sum(axis=-1))*(weight/2)
        return value


def make_fixture(*, quadrature=9, speed_deg_s=850., phase_deg=37., camera_ids=(0, 1),
                 exposure_scale=1., noise=.002, flat=False, missing=False,
                 perturbation=0., one_frame=False, count=700, device='cpu', shell=0):
    from blurtrack.blur_physics import BlurFitProblem
    scene = AnalyticScene(shell=shell)
    phase, speed = np.deg2rad([phase_deg, speed_deg_s])
    native_times = np.array([-.095, -.056, -.013, .026, .069, .112]) if not one_frame else np.array([0.])
    pixels, normals, observations, ids, sample_times = [], [], [], [], []
    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    rng = np.random.default_rng(1503)
    for ci in camera_ids:
        uv, n, _ = scene.pixels(ci, count=count)
        exposure = (.0156, .008)[ci]
        for t in native_times + (0.004 if ci==1 and not one_frame else 0.):
            truth = scene.signal(n, phase, speed, t, exposure, flat=flat)
            # A mild per-frame camera response is a nuisance, not motion truth.
            if noise:
                truth = truth*(.94+.1*rng.random())+rng.normal(0, .003)
            truth += rng.normal(0, noise, len(n))
            pixels.append(uv); normals.append(np.repeat(n[None], quadrature, axis=0))
            observations.append(truth); ids.append(ci)
            sample_times.append(t+nodes*exposure*exposure_scale/2)
    # Both cameras deliberately use the same number of independent pixels.
    count_min = min(len(o) for o in observations)
    normals = np.array([n[:, :count_min] for n in normals])
    observed = np.array([o[:count_min] for o in observations])
    train_mask = np.ones(observed.shape, bool)
    train_mask[:, ::5] = False
    atlas = scene.atlas(flat=flat, perturbation=perturbation, missing=missing)
    atlases = [atlas, atlas]
    problem = BlurFitProblem(atlases, normals, np.ones(normals.shape[:-1], bool),
                            np.asarray(sample_times), weights/2, np.asarray(ids),
                            observed, np.ones(observed.shape, bool), 0., shell=shell,
                            train_mask=train_mask, device=device)
    return dict(scene=scene, problem=problem, truth=np.array([phase, speed]),
                pixels=np.array([p[:count_min] for p in pixels]), observed=observed,
                train_mask=train_mask, ids=np.asarray(ids), atlas=atlas,
                sample_times=np.asarray(sample_times), sample_weights=weights/2)


def jsonable(value):
    if isinstance(value, dict): return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return jsonable(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def coarse_hypotheses(problem, *, phase_step_deg=15, speed_step_deg_s=100., count=6):
    """Explicit finite search domain, no access to synthetic motion truth.

    This is a basin search, not proof that all aliases have been excluded.
    Nuisance gains/offsets are still fit only to training pixels.
    """
    scored = []
    for phase in np.arange(0., 360., phase_step_deg):
        for speed in np.arange(-2400., 2400.1, speed_step_deg_s):
            parameters = np.deg2rad([phase, speed])
            scored.append((problem.score(parameters)['objective'], parameters))
    scored.sort(key=lambda item: item[0])
    selected = []
    for _, candidate in scored:
        separated = all(abs(np.rad2deg(candidate[1]-previous[1]))>=150 or
                        abs(np.rad2deg(np.arctan2(np.sin(candidate[0]-previous[0]),
                                                 np.cos(candidate[0]-previous[0]))))>=20
                        for previous in selected)
        if separated: selected.append(candidate)
        if len(selected)==count: break
    return np.asarray(selected), dict(phase_domain_deg=[0., 360.],
                                     speed_domain_deg_s=[-2400., 2400.],
                                     phase_step_deg=phase_step_deg,
                                     speed_step_deg_s=speed_step_deg_s,
                                     scored_hypotheses=len(scored))


def render_truth_contact(output):
    """Display independently rendered sharp/blurred imagery, never a restored frame."""
    size = 224
    scenes = [AnalyticScene(size=size, shell=shell) for shell in (0, 1)]
    yy, xx = np.indices((size, size))
    pixels = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
    panel = np.full((2*(size+36), 2*size, 3), 246, np.uint8)
    for ci in (0, 1):
        normals, owner, _ = scenes[0].intersections(ci, pixels)
        for column, exposure in enumerate((0., (.0156, .008)[ci])):
            rgb = np.full((len(pixels), 3), .055)
            for shell in (0, 1):
                selected = owner==shell
                signal = scenes[shell].signal(normals[selected], np.deg2rad(37.),
                    np.deg2rad((850., -730.)[shell]), .035, exposure)
                rgb[selected] = .19
                rgb[selected, shell] += signal
            display = np.rint(np.clip(rgb, 0, 1)**(1/2.2)*255).astype(np.uint8)
            row = ci*(size+36)
            panel[row+36:row+36+size, column*size:(column+1)*size] = display.reshape(size, size, 3)
            label = f"cam{ci}: {'sharp' if column==0 else str(round(exposure*1000, 1))+' ms exposure'}"
            cv2.putText(panel, label, (column*size+6, row+24), cv2.FONT_HERSHEY_SIMPLEX, .48, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.imwrite(str(output/'independent_truth_contact.png'), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def write_summary(output, report):
    lines = ['# Direct blur fit: independent synthetic validation', '',
             'Truth: 100 mm radius, 20 mm gap, 8 degree carrier tilt; two virtual calibrated views, '
             '15.6/8 ms exposure, analytic linear-light texture with 61-node exposure integration. '
             'Fit: sampled atlas and 9-node quadrature, except the explicit 17-node case. '
             '20% of pixels are held out from both motion and photometric nuisance fitting.', '',
             '| Case | Phase error (deg) | Signed speed error (deg/s) | Status |',
             '|---|---:|---:|---|']
    for case in report['cases']:
        lines.append(f"| {case['name']} | {case['phase_error_deg']:.4f} | {case['speed_error_deg_s']:.4f} | {case['fit']['status']} |")
    lines += ['', 'The sparse-start fast negative-spin failure is intentionally retained. '
              'A candidate is only the best basin searched, not proof of global uniqueness; '
              'the coarse-search case demonstrates why searching distinct phase/speed hypotheses matters.', '',
              'Missing texture and a textureless shell should remain ambiguous. A single exposure '
              'cannot determine rotation direction: the mirrored motion integrates to the same image. '
              'Exposure and speed are also coupled through their product in an isolated frame.', '',
              'These tests verify the mathematical implementation under known calibration and texture. '
              'They do not measure physical accuracy on real recorded footage, which also contains '
              'shading, occlusion, compression, exposure uncertainty, and an imperfect reconstructed atlas.']
    (output/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/blur_fit_synthetic')
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--maxiter', type=int, default=60)
    parser.add_argument('--points', type=int, default=700)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(description=__doc__, device=args.device, radius_m=.1, gap_m=.02,
                  carrier_roll_deg=8., truth_quadrature=61,
                  exposures_s=[.0156, .008], cases=[],
                  limitations=['Known fixed geometry, roll and clocks; independently rendered analytic texture.',
                               'No spatial shading, occluder mismatch, sensor rolling shutter, compression, or real texture reconstruction error except explicit atlas-mismatch case.',
                               'Good synthetic recovery does not establish physical accuracy on recorded footage.',
                               'The candidate flag distinguishes only hypotheses searched; it cannot certify a global optimum.'])
    report['source_sha256'] = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (Path(__file__), ROOT/'blurtrack/blur_physics.py')}
    report['torch_version'] = torch.__version__
    if args.device=='cuda':
        torch.cuda.reset_peak_memory_stats()
        report['gpu']=torch.cuda.get_device_name()
    cases = [('dual_noisy', {}), ('c920_noisy', dict(camera_ids=(0,))),
             ('brio_noisy', dict(camera_ids=(1,))),
             ('dual_fast_negative_sparse_seeds', dict(speed_deg_s=-1900.)),
             ('dual_fast_negative_coarse_search', dict(speed_deg_s=-1900., coarse_search=True)),
             ('green_independent_spin', dict(shell=1, speed_deg_s=-730.)),
             ('exposure_under20', dict(exposure_scale=.8)),
             ('exposure_over20', dict(exposure_scale=1.2)),
             ('quadrature17', dict(quadrature=17)),
             ('atlas_mismatch', dict(perturbation=.012)),
             ('missing_atlas', dict(missing=True)),
             ('flat_texture', dict(flat=True)),
             ('single_frame_direction', dict(one_frame=True, camera_ids=(0,)))]
    for name, options in cases:
        started = time.perf_counter()
        fixture = make_fixture(device=args.device, count=args.points,
                               **{k:v for k,v in options.items() if k!='coarse_search'})
        truth = fixture['truth']
        # Deliberately displaced starts and both signs; truth is never a seed.
        seeds = np.array([[truth[0]+np.deg2rad(dp), truth[1]*factor]
                          for dp, factor in [(9., .82), (-12., 1.16), (0., -.9)]])
        search = None
        if options.get('coarse_search'):
            seeds, search = coarse_hypotheses(fixture['problem'])
        result = fixture['problem'].fit(seeds, phase_radius_deg=35.,
                                         speed_radius_deg_s=350., maxiter=args.maxiter)
        estimate = (np.asarray(result['best_parameters_rad']) if result['best_parameters_rad'] is not None
                    else np.full(2, np.nan))
        phase_error = np.rad2deg(np.arctan2(np.sin(estimate[0]-truth[0]), np.cos(estimate[0]-truth[0])))
        entry = dict(name=name, options=options, truth_deg=np.rad2deg(truth),
                     estimate_deg=np.rad2deg(estimate), phase_error_deg=phase_error,
                     speed_error_deg_s=float(np.rad2deg(estimate[1]-truth[1])),
                     elapsed_s=time.perf_counter()-started, fit=result,
                     score_at_truth=fixture['problem'].score(truth), search=search,
                     parameter_recovery_within_1deg_5deg_s=bool(abs(phase_error)<1 and
                                                               abs(np.rad2deg(estimate[1]-truth[1]))<5))
        report['cases'].append(entry)
        print(json.dumps(jsonable({k:v for k,v in entry.items() if k not in ('fit', 'score_at_truth')})), flush=True)
        (args.output/'validation.json').write_text(json.dumps(jsonable(report), indent=2), encoding='utf-8')
    if args.device=='cuda':
        report['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
        report['peak_gpu_reserved_bytes']=torch.cuda.max_memory_reserved()
    report['total_case_time_s']=sum(case['elapsed_s'] for case in report['cases'])
    (args.output/'validation.json').write_text(json.dumps(jsonable(report), indent=2), encoding='utf-8')
    write_summary(args.output, report)
    render_truth_contact(args.output)
    return report


if __name__=='__main__': main()
