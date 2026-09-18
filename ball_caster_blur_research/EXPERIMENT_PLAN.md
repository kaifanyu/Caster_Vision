# Proposed experiments: recover fast swivel motion without changing the baseline

Status: CoTracker3 stage implemented and GPU-tested, 2026-09-18. See
[IMPLEMENTATION.md](IMPLEMENTATION.md) for the delivered design, reproducible
commands and validation scope; [RUN_RESULTS.md](RUN_RESULTS.md) records results.
The broader stages below remain a roadmap wherever not explicitly delivered.

## Preserve the accepted result

The active pipeline and completed replay remain in `../ball_caster_dual_cam`.
`baseline/accepted_pipeline_source.zip` captures its current source/configuration,
including uncommitted changes; `baseline/manifest.json` hashes 97 source files and
six final artifacts. Videos and results are referenced in place. Every experiment
must write under this new folder, with its own config, code revision and checkpoint
hash. Verify the baseline with `python scripts/preserve_baseline.py --verify`.

## Stage 1: small, controlled learned-tracker comparison

Use the same native images, camera calibration, gap/radius, initial-roll prior,
timing policy and joint backend in all branches. Do not let geometry or smoothing
changes obscure the effect of a different observation source.

| Branch | Input observations | Purpose |
|---|---|---|
| A | Accepted KLT tracks | Frozen reference |
| B | CoTracker3 offline, original images | Test material-point continuity around fast swivel |
| C | Best measured combination of A and B | Preserve strong conventional tracks and add verified new evidence |
| D | SEA-RAFT or WAFT on original images | Test distributed local motion for angular-rate estimation |
| E, optional | TAPNext/BootsTAPIR after compatibility check | Independent point-tracker comparison if B remains unreliable |
| F, optional | RVRT followed by the best tracker | Determine whether restoration helps physical estimation, not just appearance |

Start with common physical-time windows approximately 10.6-12.0 s, 18.6-19.7 s
and 20.9-21.5 s, plus a sharp stationary interval and a sharp moving control.
Select native frames independently in each camera using their corrected times.
Do not force synchronization by duplicating, interpolating or dropping images.
Use `data/roll` and `data/swivel` as cross-talk checks; their commanded motion is
a consistency constraint, not encoder ground truth.

The current computer has an NVIDIA RTX 4060 with about 8 GiB total VRAM. The full
1920x1080 float32 input tensor alone would occupy 16.11 GiB for C920 and 16.87 GiB
for Brio, before model activations. Process one camera/window at a time; crop on
CPU around the caster and test memory use before increasing duration/point count.
The tested scaled-offline checkpoint used about 3.00 GB allocated / 4.28 GB
reserved CUDA memory for 60 frames and 192 queried points, with a 16-frame CNN
feature chunk. Thirty-five run windows used 115.34 seconds of model inference,
excluding CPU decoding, masking, fitting and export.

For CoTracker, use points on observed paint interiors/edges in sharp anchor
images before and after each failure. Keep query time, camera, shell and identity
explicit. Evaluate forward and backward tracking with overlapping windows. Match
window identities using actual overlapping image/appearance evidence; do not
silently assign identical material identity because two predicted curves are close.
Preserve the crop/resize transform exactly so predicted coordinates map back to
the original undistorted calibration pixels. Maintain independent shell angles.

Occluded network predictions can initialize a hypothesis but are not visual
measurements. Require shell masks, sphere visibility, spatial spread, raw-patch
agreement where meaningful, bidirectional consistency and robust stereo geometry.
Network visibility/confidence is not an angular standard deviation. Multiple
trackers using the same pixels are correlated evidence and must not be counted
twice as independent factors.

## Stage 2: estimate velocity as a physical trajectory quantity

Use a continuous-time angle curve, initially cubic splines with analytic
derivatives, evaluated at each native exposure time. Solve trajectory parameters
from physical image residuals; do not directly differentiate noisy 2D network
tracks or fill the video with AI-generated intermediate frames. Test several
smoothness weights and report peak attenuation and timing lag. The prior must not
win merely by producing a smooth rate curve.

Keep these three outputs separate:

- `angular_velocity_status`: local angular increment/derivative evidence over a
  stated time support and effective bandwidth.
- `phase_status`: whether accumulated phase is connected to the initial reference
  or depends on a gap prior.
- `turn_count_valid`: whether full revolutions remain identifiable.

If a later track component determines `beta(t)+constant`, its derivative can be
known while absolute phase is unknown. Conversely, a smooth derivative within a
fully unobserved gap is still a prediction. Apply observability checks to the
derivative functional, excluding the smoothing prior, rather than copying the
angle-validity flag into every future rate estimate.

For shell `s`, `R_s = F Rx(alpha) Rz(beta_s)`. The rig-frame physical angular
velocity is `omega_s = alpha_dot * F e_x + beta_s_dot * F Rx(alpha) e_z`.
Export both the shell's swivel rate `beta_s_dot` and this vector when needed;
they are different quantities during simultaneous roll and swivel.

## Stage 3: fit the measured blur with the known geometry

Implemented as a bounded local experiment on the existing recordings. See
[BLUR_FIT_RESULTS.md](BLUR_FIT_RESULTS.md) and the
[measured-versus-rendered evidence index](experiments/blur_physics/index.html).
The delivered fitter freezes roll/geometry/texture and fits one shell's phase and
constant rate across five native frames per camera. It does not yet jointly
optimize a full continuous-time three-angle trajectory or certify real rates.
Ambiguous intervals remain unresolved rather than replacing the accepted replay.

This is the principal targeted accuracy experiment after obtaining a stable
initial trajectory. Build an observed surface texture atlas from sharp parts of
the existing run/calibration clips, keeping unknown surface regions masked. Do
not use the illustrative meridians from the replay as if they were real paint.

For camera `c` and frame `k`, model a pixel as an exposure integral:

```text
predicted_I_ck(x) = 1 / tau_c * integral(render_c(texture, q(t), x),
                             t_center_ck - tau_c/2, t_center_ck + tau_c/2)
q(t) = [alpha(t), beta_red(t), beta_green(t)]
```

Implement the integral with temporal quadrature and check convergence when its
sample count is increased. Retain both displaced hemisphere centers and the gap.
Mask the yoke, background, grazing surfaces and uncertain texture. Begin with
fixed measured exposures, geometry and calibrated camera poses; allow limited
per-frame photometric gain/offset. Exclude saturated pixels. Gamma/compression,
moving shadows and specularities require robust residuals and may violate an
ideal radiance average. Exposure midpoint/clock alignment and any rolling-shutter
line timing must be distinguished and independently checked.

Use the existing three-angle estimate plus learned correspondences to initialize
the solve. Compare the rendered blur directly with the original captured images.
Sharp rendered frames may illustrate the solution, but they are inferred outputs.
Keep multiple direction/turn hypotheses when the raw images do not discriminate
between them. Avoid freeing the entire texture/geometry so much that it absorbs
incorrect motion.

This design adapts principles from [Motion-from-Blur](https://github.com/rozumden/MotionFromBlur)
and [MBA-VO](https://github.com/ethliup/MBA-VO); neither repository already provides
this particular two-shell, two-camera estimator.

## Decide by physical error, not smoothness

1. **Known synthetic motion:** render unique shell textures at high temporal
   resolution, integrate the actual exposure durations and sample actual timestamp
   patterns. Vary speed, reversals, yoke occlusion, noise, JPEG artifacts and timing
   errors. Avoid evaluating only on the same texture/rendering assumptions used by
   the estimator; also degrade independently captured sharp high-speed footage.
2. **Existing recordings:** score independently annotated visible landmarks,
   held-out image patches and camera-by-camera agreement. Measure nominal pure-roll
   spin cross-talk and return-to-home consistency. These cannot alone establish
   true instantaneous rate in jointly blurred intervals.
3. **Independent reference:** repeat fast swivels with a synchronized encoder or
   high-speed short-exposure reference camera. Use a sensor appropriate to each
   independently moving degree of freedom. This is needed to claim physical rate
   accuracy, peak speed recovery and timing accuracy on real motion.

Report angular error, local-rate MAE/RMSE, peak-speed bias, peak timing lag,
whole-turn errors, correct rejection of ambiguous frames, coverage, compute time
and GPU memory. Plot errors and support against exposure/blur severity. Cluster
uncertainty by track/time block/camera and perturb calibration/timing; treating all
neural samples as independent understates uncertainty.

Accept a new branch only if it improves difficult intervals without degrading
sharp controls or masking uncertainty. Pin the software and weights, preserve the
same input frame mapping, and retain all unsuccessful or ambiguous cases in the
evaluation. A visually smoother overlay or larger accepted-track count alone is
not a sufficient acceptance criterion.
