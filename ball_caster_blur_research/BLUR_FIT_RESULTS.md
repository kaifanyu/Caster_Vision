# Physical blur fitting on the existing recordings

The new experiment is implemented in this research folder. It uses the existing
15.6 ms C920 and 8 ms Brio recordings; no camera setting or accepted trajectory
was changed. It fits the original blurred measurements with the actual separated
hemispheres, rather than using an AI-restored image as a new measurement.

Open the [experiment index](experiments/blur_physics/index.html) for measured
images, rendered hypotheses, residuals and the numerical comparisons. The
[machine-readable summary](experiments/blur_physics/summary.json) retains support
counts and individual evidence-file links. Recording instructions and complete
trial YAML files are in [CAMERA_SETTINGS.md](CAMERA_SETTINGS.md).

The local fits also have an [interactive simulated-axis comparison](experiments/physics_axis_preview/orientation_3d.html)
and a [two-camera axis-overlay video](experiments/physics_axis_preview/physics_axes.mp4).
These show separate local hypotheses beside the prior, with independent red/green
material X/Y axes; they do not replace or extend the full-run trajectory.

## Measured outcome and decision

**Keep the accepted trajectory. This experiment has not demonstrated reliable
additional recovery of the fast blurred motion in these recordings.**

The final runs use the reviewed yoke exclusions in both texture references and
target images. Older pilot folders are preserved but excluded from the final
index. Both final atlas manifests include texture, trajectory bundle and axes
hashes; the runner checks these before fitting.

| Window | Evidence from final tests | Decision |
|---|---|---|
| Red 19.2 s, strict atlas | Zero common pixels across competing hypotheses | No selected rate |
| Green 11.35 s, strict atlas | Only 59 common pixels; below the support requirement | No selected rate |
| Red 19.2 s, broader atlas | -256.30 deg/s, effectively the prior -256.30; held-out RMSE 0.03660 -> 0.03661 | No demonstrated improvement |
| Green 11.35 s, broader atlas | +101.73 deg/s versus prior +242.72; held-out RMSE 0.01698 -> 0.02136 | Reject proposed change; roughly 26% worse holdout error |
| Red 21.2 s, broader atlas | +49.71 deg/s versus prior +51.99; held-out RMSE 0.06721 -> 0.06731 | No demonstrated improvement |
| Reserved low-motion red control, 8.0 s | +6.41 deg/s versus prior +5.53; held-out RMSE 0.12242 -> 0.12087 | Small photometric improvement, not a fast-motion validation |

RMSE is unitless linear-light color contrast, not angular error. The control's
remaining residual is substantial; a successful optimizer does not make the
fixed-texture appearance model a perfect description of real lighting.

The strict measured atlas covers about 19.1%/52.1% of red and 7.2%/7.0% of green
for C920/Brio respectively. These are atlas cell fractions, not equal-area surface
fractions. Broader references increase red coverage to 20.6%/67.3% and green to
7.2%/41.8%, at the cost of including pre-blurred texture. Unknown regions are
never invented. At red 19.2 s, only 130 of the fused candidate's 2,871 supported
pixels come from C920; both cameras are included, but the usable texture evidence
is mostly from Brio. Pixel counts alone are not an information/covariance measure.
The corresponding C920-only trial is ambiguous (53 common hypothesis pixels).
Brio alone yields about -256.11 deg/s. Fusing both views is still the appropriate
architecture for complementary visibility; here it cannot create missing C920
texture or independently verify a Brio-dominated numerical fit.

The 17-node red fit is approximately -256.77 deg/s; changing assumed exposures
by -20%/+20% gives approximately -257.20/-257.05 deg/s. These small changes do not
establish that a missed peak was recovered. The assumption perturbations do not
alter the source recordings, and no such result is promoted to the full replay.
Doubling each image-grid dimension from 96 to 192 yields -257.52 deg/s, with
held-out RMSE 0.03393 -> 0.03395; it also does not improve the prior comparison.

Validation: **89 CPU tests passed**, **12 final real-video trials** and
**13 independent synthetic GPU cases** ran on the RTX 4060. The final real trials
took about 317 seconds in total including video decoding, with a maximum of
140.5 MiB allocated by PyTorch. The deliberately failing sparse-start synthetic case remains
documented. [Baseline verification](experiments/blur_physics/baseline_verify.txt)
confirmed 97 source files and six reference artifacts remain unchanged.

## What is implemented

- Exact calibrated ray intersections with the two 100 mm-radius outer
  hemispheres, displaced by +/-10 mm along the rolled axis. The gap, camera
  intrinsics/extrinsics and existing initial-roll solution are retained.
- Frozen measured per-camera texture atlases. A material point follows
  `F Rx(alpha) Rz(beta_shell)`. Atlas coordinates come from the frozen hybrid
  trajectory; this is conditional texture registration, not independent truth.
  Unknown cells remain masked. References avoid the tested sharp-control window
  and stop before the relevant phase failures (red 18.5 s, green 10.5 s).
- A strict reference atlas (predicted smear <=6 pixels) and a separate sensitivity
  atlas (<=12 pixels). Neither threshold means perfectly sharp. The broader atlas
  contains visibly smeared references and can bias fitted speed downward.
- Five native frames per selected camera, with their separate timestamps and
  exposures. At each frame, GPU texture sampling integrates a local constant
  swivel rate over nine Gauss-Legendre exposure nodes. The frozen, time-varying
  roll trajectory is evaluated at each node; each shell is fitted separately.
- Robust linear-light color-contrast residuals, per-frame limited gain/offset,
  and a spatial block holdout (one in four blocks before visibility/texture
  masking; actual counts are reported). Held-out pixels do not fit
  either motion or those gain/offset parameters. Frames are undistorted before
  comparison; unknown texture, grazing regions and clipping are excluded.
- Manually reviewed masks for the stationary outer yokes, in the same native
  coordinates for references and target images. These masks are specific to this
  recording/mount and deliberately do not claim to model every shaft, reflection,
  moving shadow or other occluder.
- 1,176 coarse phase/speed hypotheses (15 degree phase increments, 100 degree/s
  speed increments within +/-2,400 degree/s), followed by up to eight local starts
  including the prior. This finite search does not prove global uniqueness or
  rule out faster motion. Local fits use bounds of +/-20 degrees and +/-220
  degree/s around each seed.
- Fair ranking on a common pixel set across refined hypotheses, minimum common
  training support and explicit direction/texture ambiguity checks. A symmetric
  isolated exposure includes its exactly reversed path as an alternative.
  Ambiguous results keep diagnostic hypotheses but export no chosen rate CSV.
- Frozen texture/calibration/trajectory hash checks, input arrays, raw candidate
  scores, per-camera support, held-out residuals, GPU information and plots.
  No candidate is automatically inserted into the full-run orientation export.

## How to interpret the evidence

`candidate` means a converged photometric hypothesis among the searched basins.
It does not establish a physically accurate rate. `recovered_rate_deg_s` remains
null for all real cases because no independent angular reference is available.
`accuracy_validated` and `turn_count_valid` remain false. A local rate may become
identifiable without recovering the earlier whole-turn count; this implementation
does not certify either from a good-looking overlay.

Use the same-pixel **prior versus candidate** holdout comparison within each row.
Different atlas, camera or quadrature trials can have different valid support;
their RMSE values must not be treated as measurements on identical pixels.
The image index shows each rendered hypothesis on its own available texture
support so missing regions remain visible. Numerical comparisons use identical
pixels, and report their count explicitly.

The strict atlas can have too little material coverage to compare plausible
phases fairly. Allowing blurrier references may return a numerical rate, but it
also changes the measurement model. That tradeoff is a reason to retain both
experiments, not a reason to select the smoother-looking answer.

## Independent GPU validation

[13 controlled cases](experiments/blur_fit_synthetic_checked/README.md) use an
independent NumPy analytic texture and ray renderer with 61 exposure samples,
known two-camera geometry, 8 degree roll and independent shell spin. The fitter
uses a sampled atlas and nine nodes. Nominal dual-camera speed error is about
0.016 degree/s for an 850 degree/s truth. The dense-search case recovers about
-1900.05 degree/s for -1900 degree/s truth.

A sparse-start failure at that negative speed is retained: it selects the wrong
local minimum near +1629 degree/s. This demonstrates why `candidate` is not a
global accuracy certificate. Missing texture, flat texture and a single-exposure
direction tie remain ambiguous. There are also camera, exposure, texture-mismatch
and 17-node quadrature sensitivity cases.

These validate the implementation under known synthetic conditions. They do not
measure errors against true angular velocity in the webcams. Real limitations
include reference blur, appearance changing with illumination/roll, uncalibrated
camera response and compression, imperfect support masks, uncertain geometry,
receive-time versus exposure-midpoint timing, and unmodeled rolling shutter.
The local constant-rate model cannot resolve arbitrary acceleration within an
exposure or throughout a long interval where both views lost texture.

## Reproduce

From `Caster_Vision/ball_caster_blur_research`, using the installed CUDA Python:

```powershell
& 'C:\Users\aishi\AppData\Local\Programs\Python\Python310\python.exe' scripts/build_blur_atlas.py --output experiments/blur_physics/new_atlas
& 'C:\Users\aishi\AppData\Local\Programs\Python\Python310\python.exe' scripts/fit_recorded_blur.py --shell red --center 19.2 --atlas experiments/blur_physics/new_atlas --output experiments/blur_physics/new_red19
& 'C:\Users\aishi\AppData\Local\Programs\Python\Python310\python.exe' scripts/validate_blur_fit.py --device cuda --output experiments/new_blur_validation
& 'C:\Users\aishi\AppData\Local\Programs\Python\Python310\python.exe' -m unittest discover -s tests
& 'C:\Users\aishi\AppData\Local\Programs\Python\Python310\python.exe' scripts/preserve_baseline.py --verify
```

Use a fresh output directory for a new fit. The runner also accepts
`--cameras c920`, `--cameras brio101`, `--quadrature 17` and
`--exposure-scale 0.8` / `1.2`. Exposure scaling changes only a modeling assumption
for the already-recorded AVI; it neither changes a camera nor creates a new
short-exposure recording. `--initial PHASE_DEG RATE_DEG_S` narrows the search for
diagnostics and should not replace the broad search when testing ambiguity.

For better capture, use the supplied exposure presets with the explicit Linux
recorder command in [the camera guide](CAMERA_SETTINGS.md). Add continuous diffuse
light, verify the actual streaming exposure, and preserve lens/mount calibration.
A slow, well-lit sweep of each shell over its full material rotation would also
provide a much more complete, less blurred atlas for this physical fitter.

The existing roll/swivel calibration clips are also potential sharper reference
sources before recording anything new: their controlled trajectories have much
less predicted smear (roughly 1.5 pixels C920 / 0.9 pixels Brio at the 95th
percentile). However, each clip independently sets its shell phases to zero.
Those atlases cannot be mixed with the run atlas without first fitting and
validating constant per-shell material-phase offsets. The control excursions are
only about 42 degrees of roll and 41 degrees of red swivel, with no comparable
green swivel sweep; they would add partial coverage, not a complete texture map.
Cross-session material registration is a specific next software experiment and
has not been silently assumed in the delivered fits.

The complete delivered comparison matrix can be repeated with
`python scripts/run_blur_experiments.py --prefix new_trial --group all` after the
masked atlases exist. It uses the full phase/speed search in each sensitivity
case. `python scripts/summarize_blur_fit.py --pattern 'new_trial_*'` generates the
review index for that prefix. Use a fresh prefix to retain previous results.
