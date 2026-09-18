# CoTracker3 implementation and validation

The follow-on physical blur experiment is now implemented separately. Its GPU
exposure renderer, frozen measured atlas, ambiguity/export gates and validation
are documented in [BLUR_FIT_RESULTS.md](BLUR_FIT_RESULTS.md). Capture presets and
the exact recorder/YAML behavior are in [CAMERA_SETTINGS.md](CAMERA_SETTINGS.md).

This is a separate research implementation. The accepted source and final outputs
in `../ball_caster_dual_cam` remain frozen. The pretrained CoTracker3 model supplies
image correspondences; the existing calibrated physical solver fits this caster.
The network was not fine-tuned on these recordings.

## Implementation plan and delivered architecture

1. **Pin and verify the official model.** `model_manifest.json` records the exact
   upstream commit and SHA-256 of `scaled_offline.pth`. Inference loads local tensor
   weights with `weights_only=True`; it does not use a mutable Torch Hub download.
2. **Track each camera on the GPU.** Decode native frames, undistort using each
   camera's original intrinsics, crop around the caster, and resize on the CPU.
   Process 60-frame offline windows with a 40-frame stride. Select up to 32 paint
   corners per shell at the first, middle, and last frames, plus the official
   support grid. Reverse inference supplies observations before each query.
   This is overlapping offline inference, not one full-recording attention pass.
3. **Preserve measurement coordinates and identity.** Invert both crop and resize
   mappings back to original undistorted camera pixels. Keep native frame times
   and camera/shell/window-local IDs. Window tracks are not silently stitched;
   the shared physical trajectory couples their separate nuisance landmarks.
   Raw visibility and confidence are retained. Query anchors are real detected
   image coordinates, explicitly distinguished from independent neural tests.
4. **Fit the two-shell mechanism.** With `q=[alpha,beta_red,beta_green]`, the shell
   orientation is `F Rx(alpha) Rz(beta_shell)`. The two centers are displaced by
   `+/-gap/2` along the rolled swivel axis. Radius is 100 mm, gap is 20 mm. Both
   cameras constrain one trajectory, with separate material maps and spins.
   Cross-camera feature matching is unnecessary. The approximate 8-degree
   initial roll is a finite prior, not a hard reset of the geometry.
5. **Reject physically inconsistent evidence.** Use paint masks, network scores,
   spatial deduplication, robust reprojection, hemisphere visibility, minimum
   track persistence, and final 3-pixel inlier tests. These reduce errors; they
   cannot prove every retained material identity. Confidence is not covariance.
   Reverse inference is not an independent forward/backward cycle-error test;
   photometric patch checks remain future work. Exposure-integral residuals are
   now tested in the separate bounded experiment linked above; they are not
   automatically added to this full-run CoTracker fit.
6. **Keep rate and phase validity separate.** Export diagnostic local cubic
   derivatives on the 30 Hz pose grid. Measurement-only temporal rank and local
   geometry tests support spin rates, including components with unknown constant
   phase. Unsupported short intervals are predictions; longer ones are null.
   Roll support is conditional on actual accepted visual roll poses. The cubic
   curve does not increase sampling bandwidth or remove exposure blur.

CoTracker uses CUDA. The calibrated sparse nonlinear fit and exports use CPU
SciPy/OpenCV. The run uses native camera timestamps with the accepted additional
Brio offset fixed at 6.1387 ms, so camera/observation comparisons do not retune time.

## Validation plan and scope

- Unit tests cover coordinate transforms, raw model output, reverse-time mapping,
  camera/shell identity, real separated-hemisphere pose recovery, held-out track
  exclusion, timing, rate null modes, and missing-data exports.
- Six GPU synthetic tests use the actual 0.2 gap/radius ratio, known 8-degree
  carrier tilt, independent shell spins, foreground occlusion, and sharp/8 ms/
  15.6 ms images. Ground-truth visibility is explicitly an oracle in this
  controlled test. Query frames are excluded. This tests correspondence and
  conditional spin estimates, not real-camera calibration accuracy.
- The entire actual run is processed from both cameras. CoTracker C920-only,
  Brio-only, fused CoTracker, and fused KLT+CoTracker branches use the same physical
  parameters and initial trajectory. These are **conditional refinements**, not
  independent reconstructions from scratch: the accepted full-data trajectory
  initializes all fits, but supplies no trajectory residual in the objective.
- Conditional held-out checks reserve whole conventional tracks. Learned tracks
  overlapping any reference pixel within 6 px are excluded in their entirety.
  Spatial overlap is only a cross-algorithm identity proxy, and the shared
  full-data initializer prevents a fully independent generalization claim.
  Reference point coordinates are fitted only to early samples; later pixels
  are scored with failed tracks retained in coverage denominators.
- Both `data/roll` and `data/swivel` are processed through the GPU tracker and
  unconstrained physical fits. Cross-talk and home drift are consistency checks,
  since these recordings do not supply synchronized encoder ground truth.

See [RUN_RESULTS.md](RUN_RESULTS.md) for the measured outcome and acceptance
decision. Exposure-integrated image fitting, rolling-shutter estimation,
calibration uncertainty, and a smoothness/peak-speed sweep are not implemented
by this CoTracker stage. An independent reference is still needed to validate
real peak angular velocity and recovered turn counts.

## Reproduce on this Windows computer

The tested interpreter is
`C:\Users\aishi\AppData\Local\Programs\Python\Python310\python.exe`.
Exact package versions and CUDA/GPU details are in
[reports/environment.json](reports/environment.json). No global environment was
changed for this experiment.

From this directory, using PowerShell:

```powershell
$py = 'C:\Users\aishi\AppData\Local\Programs\Python\Python310\python.exe'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
& $py scripts/setup_cotracker.py --check
& $py -m unittest discover -s tests -v
& $py scripts/track_cotracker.py --output experiments/cotracker_native
& $py scripts/fit_cotracker.py --source hybrid --output experiments/hybrid_repeat --render
& $py scripts/fit_cotracker.py --source cotracker --cameras c920 --output experiments/c920_repeat
& $py scripts/fit_cotracker.py --source cotracker --cameras brio101 --output experiments/brio_repeat
& $py scripts/fit_cotracker.py --source cotracker --holdout --output experiments/cotracker_holdout_repeat
& $py scripts/fit_cotracker.py --source klt --holdout --output experiments/klt_holdout_repeat
& $py scripts/fit_cotracker.py --source hybrid --holdout --output experiments/hybrid_holdout_repeat
& $py scripts/compare_experiments.py
& $py scripts/preserve_baseline.py --verify
```

The auxiliary validation commands are:

```powershell
& $py scripts/validate_cotracker_gpu.py --output experiments/gpu_validation_repeat
& $py scripts/track_cotracker.py --session ../ball_caster_dual_cam/data/roll --native-cache ../ball_caster_dual_cam/out/roll_native_cache --output experiments/controls/roll_cotracker
& $py scripts/track_cotracker.py --session ../ball_caster_dual_cam/data/swivel --native-cache ../ball_caster_dual_cam/out/swivel_native_cache --output experiments/controls/swivel_cotracker
& $py scripts/validate_controls.py
```

Monocular fits retain only their active event stream and put the phase gauge at
the first actual corrected observation. Brio's fixed clock correction is absorbed
into its working time array, with the original receive times preserved in exports.
This avoids penalizing a single camera for the other camera's earlier start.
The combined video renderer requires both cameras; mono results have HTML viewers.

Choose a new fit output directory for each rerun. The tracker validates input,
code, checkpoint and configuration provenance when reusing cached windows.
`scripts/refresh_exports.py experiments/hybrid_dual` refreshes rate diagnostics
and HTML/CSV from a saved fit without reoptimizing poses; `--render` also creates
a replay in an empty `replay` directory. It currently targets run-session fits.

For a clean environment (optional; not needed on this tested computer):

```powershell
py -3.10 -m venv .venv
& .venv/Scripts/python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
& .venv/Scripts/python.exe -m pip install -r requirements-inference.txt
& .venv/Scripts/python.exe scripts/setup_cotracker.py
```

The official model is under its upstream CC-BY-NC-4.0 license; see
`third_party/co-tracker/LICENSE.md`. Assets and large experiment caches are
ignored by Git. Provisioning is explicit; normal tracking does not download code.

## Output semantics

- `results.json`, `results.csv`: estimated pose, camera support, phase status.
- `strict_results.csv`: legacy strictly phase-supported pose export.
- `angular_rates.csv`: the authoritative new rate export, with per-component
  `rate_status`, `phase_status`, `turn_count_valid`, strict measured-rate columns,
  derivative support intervals, and C920-frame angular-velocity vectors.
- `orientation_3d.html`: offline viewer of the two moving hemispheres and gap.
- `replay/combined_tracking.mp4`: both native camera views with the fitted
  geometry and one shared simulated caster. Illustrated meridians are render
  aids, not additional observed paint features.
- `bundle.npz`, `strict_rate_evidence.npz`, `evaluation.json`: saved physical
  parameters, rate-support evidence, and conditional image-consistency audit.

For shell `s`, physical angular velocity in the C920 frame is
`omega_s = alpha_dot F e_x + beta_s_dot F Rx(alpha) e_z`.
The swivel rate `beta_s_dot` is a scalar and is different from this combined
vector during simultaneous roll and swivel. Blank strict rate fields mean the
rate is not supported as a measurement; a finite estimate does not validate
global phase or whole turns.
