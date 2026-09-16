# Ball-caster rotation measurement

This project measures the three rotational degrees of freedom of a two-piece,
random-speckled ball caster from video:

- `alpha`: shared roll of the assembly about ball `+x`
- `beta_top`: independent top-hemisphere spin about ball `+z`
- `beta_bottom`: independent bottom-hemisphere spin about ball `+z`

It uses classical computer vision: HSV/equator segmentation, forward-backward
KLT tracking, ray-sphere unprojection, and RANSAC/Kabsch rotation fitting. No
GPU or deep-learning model is required.

For the current footage, [rerun with paint-supported masks and overlapping
offline fits](documents/NEW_VIDEO_WORKFLOW.md#rerun-with-paint-support-and-interval-recovery).
This retracks the main clip into a separate output directory, then generates
the tracking, axis and replay overlays. Recovery requires actual image overlap;
unsupported intervals remain marked missing.

The code and Python environment are prepared. Real footage and real camera/
axis calibration values are deliberately left for you in `data/` and
`config.yaml`.

## Read this before recording

The measurement model has two hard requirements:

1. The camera must be rigidly fixed to the robot chassis, not fixed in the
   room while the robot moves.
2. The ball center and projected ball circle must remain stationary in the
   image. The ball may rotate, but its center may not translate.

`scripts/run.py` stops if either corresponding flag in `config.yaml` is false.
Those flags are a declaration by you; the program cannot prove the assumptions
from one clip. If the circle visibly shifts between the first and last frame,
stop--the rotation-only geometry is not applicable as written.

The remaining operating assumptions are:

- Inter-frame motion is small enough for KLT, typically below roughly 10
  degrees/frame. The synthetic Stage D result gives the measured limit for
  this implementation.
- The yoke is excluded from tracking.
- Two distinct speckle colors are preferred. A configurable equator split is
  available for one-color footage.
- Camera intrinsics and the fixed ball-to-camera axis orientation are needed
  for trustworthy angles.
- The surface has dense, non-repeating, high-contrast speckles and limited
  glare/motion blur.

## Prepared environment

In this workspace, `.venv` has already been created with Python 3.10 and the
pinned packages in `requirements.txt`. The numeric convention tests currently
pass. `setup.ps1` is safe to rerun and recreates/repairs the environment when
needed.

Open PowerShell and start from the project root:

```powershell
Set-Location 'C:\Users\kyu\Documents\Upenn\Caster_Vision\ball_caster_rot'
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
.\.venv\Scripts\Activate.ps1
python --version
python -m pytest -q
```

The execution-policy change applies only to that PowerShell process. If
`.venv` must be recreated, `setup.ps1` expects the `py` launcher to provide
Python 3.10 (`py -3.10`).

## Run synthetic validation first

The synthetic suite is the ground-truth oracle. Run the shorter plumbing smoke
first:

```powershell
python .\scripts\validate_synthetic.py --config .\config.example.yaml --quick
```

Then run the full release/acceptance suite before using real footage:

```powershell
python .\scripts\validate_synthetic.py --config .\config.example.yaml
```

`--quick` still touches Stages A-E and all requested robustness sweep values,
but uses shorter clips. It is a smoke test, not the final correctness gate.
The default full run must report `OVERALL: PASS`:

- Stage A: numeric geometry and rotation-convention tests
- Stage B: per-frame top/bottom rotation accuracy
- Stage C: end-to-end `alpha`, `beta_top`, and `beta_bottom` decomposition
- Stage D: noise, speed, blur, glare, and speckle-density characterization
- Stage E: automatic circle and pure-motion axis calibration against truth

Artifacts are written to `out/synthetic/`:

- `validation_report.json`
- `stage_c_angles.png`
- `stage_d_robustness.png`
- `stage_e_angles.png`

The prepared environment's full run passed all gates. Its measured summary is:

- Stage B median top/bottom increment error: `0.045 / 0.057 deg`; p95:
  `0.081 / 0.118 deg`
- Stage C RMSE (`alpha / beta_top / beta_bottom`):
  `0.161 / 0.459 / 0.870 deg`
- Stage D characterized limit: `5 deg/frame` (the 10 deg/frame case was the
  first to cross the coverage/error criterion); noise through 1 px stayed
  below 2 deg RMSE
- Stage E RMSE with estimated circle and axes:
  `0.143 / 0.421 / 0.832 deg`

These values are recorded in `out/synthetic/validation_report.json`; rerun the
full gate after changing tracking/geometry code or dependency versions.

The report states `delta_max_deg_per_frame`. For an expected peak caster speed
of `omega` degrees/second, the minimum characterized frame rate is
`omega / delta_max`; use additional safety margin in a real recording.

Useful optional arguments are:

```powershell
python .\scripts\validate_synthetic.py --config .\config.example.yaml --output .\out\synthetic --seed 24681357
```

Do not move on to real-data interpretation if the full synthetic gate fails.

## Input slots

The prepared folders and config slots are:

| Purpose | Default location/config key |
| --- | --- |
| Main caster recording | `data/clip.mp4` / `input.path` |
| Checkerboard photographs | `data/camera_calibration/` |
| Pure roll calibration | `data/axis_calibration/pure_roll.mp4` |
| Pure swivel calibration | `data/axis_calibration/pure_swivel.mp4` |
| Camera matrix/distortion | `camera.K`, `camera.dist` |
| Fixed ball circle | `circle.u0`, `circle.v0`, `circle.r_px` |
| Ball-to-camera axes | `frame_calib.R_bc` |

Put the main video at `data/clip.mp4`, or change this block in `config.yaml`:

```yaml
input:
  path: data/clip.mp4
  type: auto
  max_frames: null
  fps_override: null
```

Videos, an image-sequence directory, a single image, and image globs are
accepted. For an image sequence, specify its frame rate:

```yaml
input:
  path: "data/frames/*.png"
  type: images
  max_frames: null
  fps_override: 120.0
```

Config paths are resolved relative to `config.yaml`. Keep forward slashes in a
quoted YAML glob. Video FPS metadata is used unless `fps_override` is set.

## Calibration order

For a trustworthy measurement, use this order:

1. Camera `K` and distortion coefficients
2. HSV/yoke masks
3. Fixed projected ball circle
4. Ball-to-camera orientation `R_bc`
5. Main measurement clip

If camera calibration changes, repeat circle and axis calibration because the
pipeline tracks on undistorted frames.

### 1. Enter or calibrate `K` and `dist`

If you already have OpenCV calibration values for the exact lens settings and
recording resolution, paste them into `config.yaml`:

```yaml
camera:
  K: [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
  dist: [k1, k2, p1, p2, k3]
  fov_deg: 60.0
```

Replace every symbolic value with a number. OpenCV distortion vectors with 4,
5, 8, 12, or 14 coefficients are supported. Values calibrated at a different
resolution, crop, zoom, focus, or lens setting should not be reused blindly.

To calibrate from a checkerboard, place at least 10 sharp images with varied
positions and tilts in `data/camera_calibration/`. Keep resolution, focus,
zoom, and lens settings identical to the caster recording. For a board with
9 by 6 **inner corners** and 25 mm squares, run:

```powershell
python .\scripts\calibrate_camera.py --images .\data\camera_calibration --board-cols 9 --board-rows 6 --square-size 25 --config .\config.yaml
```

Change `9`, `6`, and `25` to match your board. The script writes `camera.K`
and `camera.dist`, annotated accepted images, and
`out/camera_calibration/calibration_report.json`. Retake the set if detections
are sparse or RMS reprojection error exceeds about 1 pixel.

If `camera.K` remains `null`, the pipeline estimates a pinhole camera from
image width and `fov_deg`. That mode is explicitly warned as untrusted and is
only suitable for a smoke run.

### 2. Tune color and yoke masks

The default HSV ranges match the synthetic renderer, not necessarily your
paint and lighting. With the real clip in place, run:

```powershell
python .\scripts\inspect_hsv.py --config .\config.yaml --clip .\data\clip.mp4 --frames 12 --write
```

`--frames` decides how many evenly spaced frames of the clip are loaded for
sampling; samples accumulate across every frame you visit. **Sample each
hemisphere on several frames.** A shell that is oblique and shadowed early in
a clip is frontal and bright later, and a range fitted only to the dark
version silently drops most of that shell for the rest of the run. A
value-capped `bottom_hsv` fitted from one frame is the most common cause of
"the detection video misses a lot of labels".

In the window:

- Press `T`, `B`, or `Y` to choose the top, bottom, or yoke class.
- Click several patches of the active class.
- `N`/`.` and `P`/`,` step through frames; `0` returns to the first.
- `SPACE` toggles the processed tracking masks: growth, separation, yoke
  exclusion, and temporal boundary margin are included when a circle is set.
  Unsampled classes retain their configured ranges in the preview.
- `U` undoes the last click; `R` clears the active class.
- `Enter` prints and writes the suggested ranges; `Q`/Esc cancels.

The status line counts clicks *and* distinct frames per class, and the script
warns if a hemisphere was sampled on fewer than three frames. OpenCV hue is
`0..179`; wrapped red ranges where `lo.H > hi.H` are supported.
Preview and saved ranges both use the requested `--hue-margin` and
`--sv-margin`. Without a circle or both shell ranges, preview shows only
available raw HSV colors. Sample painted markings for the shell classes;
white interior rims and holes do not follow the outer-sphere model.

#### Sparse markings: grow the mask off the paint

Colour thresholding only makes the *coloured* pixels trackable. That is fine
for dense speckle, but a shell decorated with sparse printed marks (triangles,
stickers, dots) leaves most of its surface unusable and starves KLT, even
though the plain surface between the marks carries plenty of scuff-and-dirt
texture. `segment.grow_px` dilates each colour mask so that surrounding
surface becomes trackable too:

```yaml
segment:
  grow_px: 17.0            # dilate each colour mask by this radius, in pixels
  separation_px: 9.0       # extra band kept clear of the other hemisphere
  yoke_dilate_px: 5.0      # widen the yoke exclusion after thresholding
  color_wins_over_yoke: true   # a pixel that matched a speckle colour is not yoke
```

Growth can never carry a feature across the seam: a grown region is discarded
wherever it comes within `separation_px` of the other hemisphere's grown
region. Set `grow_px` to roughly half the gap between marks and confirm in
`tracking_overlay.mp4` that no track sits on the wrong shell. Leave it at `0`
for dense speckle, where it buys nothing.

`color_wins_over_yoke` fixes a conflict that is easy to miss: the yoke range
is a catch-all for dark pixels and is subtracted *before* the hemispheres, so
a loose one also deletes shaded marks near the limb. Keep `yoke_hsv.hi[2]`
(the value cap) as low as the yoke allows.

For one-color footage, use the equator fallback:

```yaml
assumptions:
  two_speckle_colors: false

segment:
  mode: equator
  equator: {angle_deg: 0.0, offset_px: 0.0, deadband_px: 3.0}
```

`angle_deg` tilts the dividing line, `offset_px` shifts it vertically from the
circle center, and `deadband_px` rejects seam features. In equator mode the
yoke threshold is especially important. Set `yoke_hsv.enabled: false` only if
the yoke cannot enter either usable feature region, and verify that decision
in the tracking overlay.

### 3. Calibrate the projected circle

Do this after `K/dist`, because the script fits the undistorted first frame.
Start with automatic fitting:

```powershell
python .\scripts\calibrate_circle.py --config .\config.yaml --clip .\data\clip.mp4
```

Inspect `out/circle_preview.png`. The green ring must follow the true ball
silhouette, not the yoke or a shadow. If it misses, use the interactive
three-point fit:

```powershell
python .\scripts\calibrate_circle.py --config .\config.yaml --clip .\data\clip.mp4 --manual
```

Click three widely separated silhouette points, press `R` to reset, and press
Enter to accept. The script writes `circle.u0`, `circle.v0`, and `circle.r_px`
to `config.yaml`. Leaving all three values `null` makes the real runner attempt
automatic fitting, but a reviewed fixed calibration is safer.

### 4. Enter or calibrate `R_bc`

`R_bc` maps ball-coordinate vectors into camera coordinates. Its columns are
the camera-coordinate directions of ball `+x`, `+y`, and `+z`:

```yaml
frame_calib:
  R_bc:
    - [r11, r12, r13]
    - [r21, r22, r23]
    - [r31, r32, r33]
```

You may paste a manually measured matrix, but it must be orthonormal with
determinant `+1`. A wrong `R_bc` can produce smooth-looking but incorrectly
labeled angles.

The preferred method uses two labeled clips. First calibrate `K/dist`, the
circle, and HSV masks. Then record both clips from the same rigid camera setup
and marked home pose:

- `pure_roll.mp4`: isolated, monotonic motion about ball `+x`; no swivel
- `pure_swivel.mp4`: isolated, monotonic motion about ball `+z`; no roll

For each clip, use one smooth 60-90 degree sweep in one direction. Avoid
stationary lead-in/tail footage, reversals, pauses, and regripping during the
sweep. Keep the camera, caster housing, circle, focus, zoom, crop, and
resolution unchanged across both calibration clips and the main clip.

Run:

```powershell
python .\scripts\calibrate_axes.py --config .\config.yaml --roll .\data\axis_calibration\pure_roll.mp4 --swivel .\data\axis_calibration\pure_swivel.mp4 --min-axis-step-deg 0.20
```

`--min-axis-step-deg 0.20` excludes increments smaller than 0.20 degrees from
the common-axis fit and its purity/direction diagnostics. Near zero rotation,
the axis direction is not observable and subpixel tracking jitter can point it
almost anywhere. This motion floor removes those noise-dominated estimates; it
does not relax the 15 degree axis-spread limit or hide off-axis motion above
the floor. The default is 0.20 degrees, so the explicit option documents the
threshold used for this calibration and can be omitted on later runs.

The JSON report records `raw_step_count`, `rejected_low_motion_count`,
`retained_step_count`, `sample_count`, and `min_axis_step_deg` for both clips.
A result is usable only when it prints `PASS` and retains at least
`--min-valid-steps` (default `8`) increments for each axis. If too few steps
remain, record a smoother/faster sweep instead of lowering the motion floor
into the tracking-noise range.

Imagery determines an axis line but not its sign. If either clip moved in the
negative right-hand-rule direction, rerun with the appropriate flag:

```powershell
python .\scripts\calibrate_axes.py --config .\config.yaml --roll .\data\axis_calibration\pure_roll.mp4 --swivel .\data\axis_calibration\pure_swivel.mp4 --min-axis-step-deg 0.20 --roll-sign -1 --swivel-sign -1
```

Use only the sign flags that apply. The script checks axis spread, PCA
consistency, axis separation, valid-step count, and motion direction. It
writes `frame_calib.R_bc` only on `PASS`; its JSON report goes to
`<output.dir>/axis_calibration_report.json`.

## Approximate smoke run versus calibrated measurement

These are intentionally different standards.

### Approximate smoke run

An approximate run can verify decoding, segmentation, tracking, and output
generation. It is **not** a trustworthy angle measurement. For this mode:

- `camera.K` may remain `null`, using `fov_deg`.
- The circle may remain null for automatic fitting.
- HSV/yoke masks still need to match the footage.
- `scripts/run.py` still requires some proper 3x3 `R_bc`. If you temporarily
  use the identity matrix only to exercise the software, all decomposed axes
  and angles must be treated as uncalibrated unless the physical frames truly
  coincide.

Temporary plumbing-only placeholder:

```yaml
frame_calib:
  R_bc: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
```

Run the smoke without strict acceptance:

```powershell
python .\scripts\run.py --config .\config.yaml
```

The startup banner will say `FOV approximation (UNTRUSTED)` when applicable.
Do not report `alpha/beta` from this run as measurements.

After axis calibration passes and writes `frame_calib.R_bc`, use that same
non-strict command for the first baseline prediction of `data/clip.mp4`:

```powershell
python .\scripts\run.py --config .\config.yaml
```

This writes the baseline angles, velocities, diagnostics, and tracking overlay
to `out/real/` while reporting failed self-consistency checks as warnings.
Inspect `out/real/tracking_overlay.mp4`, `angles.png`, `diagnostics.png`, and
the summary printed in the terminal before treating the result as meaningful.
Once the baseline looks correct, repeat with `--strict` for the acceptance
exit code.

### Trustworthy calibrated measurement

Treat a real run as a calibrated measurement only after all of the following:

- Full synthetic Stages A-E report `OVERALL: PASS`.
- `K/dist` were calibrated for the recording setup and resolution.
- The saved circle preview is correct and the circle stays stationary.
- `R_bc` is measured or the pure-motion calibration passes.
- The tracking overlay shows the correct hemisphere colors and no yoke tracks.
- Real-run self-consistency diagnostics meet the targets below.

Then run strict mode:

```powershell
python .\scripts\run.py --config .\config.yaml --strict
```

`--strict` returns a nonzero status when the implemented self-consistency
checks miss their targets. Use `--no-overlay` only when codec/output issues
prevent writing the debug video:

```powershell
python .\scripts\run.py --config .\config.yaml --strict --no-overlay
```

## Persistent tracking and nearby keyframe correction

The current `config.yaml` enables temporal tracking for the measurement run:

```yaml
temporal:
  enabled: true
  boundary_margin_px: 3
```

`run.py` keeps feature identities between frames and replenishes lost tracks.
It also attempts direct image matches to nearby saved images, called keyframes,
every three frames by default and when an adjacent estimate or previous pose
is unreliable, provided image sharpness is sufficient. Tracked or predicted
positions initialize these matches.
Accepted observations from one or more nearby keyframes jointly fit the current
orientation, reducing reliance on a long chain of frame-to-frame rotations.
Earlier saved poses are not re-optimized by this temporal pass; the optional
offline refinement below revisits them after tracking finishes.
A separate snapshot of the latest accepted frame provides a nearby recovery
reference between scheduled keyframes, without replacing the older references.

The temporal estimator checks inlier count, inlier ratio, angular residual,
image sharpness, and feature spread before accepting a pose. Sharpness is
compared with recent image history to accommodate changing visible texture and
lighting; it is a quality indicator, not a definitive blur detector. When an
adjacent pair is weak, a direct match to an earlier good keyframe can recover the current
orientation, including motion across the gap. If that match is also unreliable,
the frame remains invalid. A later recovery does not turn unresolved frames
into measured poses, and the estimator does not interpolate them as measured
truth. The axes overlay hides that shell's grid and probes and displays an
`UNRESOLVED` notice. Replay labels the display fallback for a missing pose;
the fallback rendering is not a measurement.

`results.json` adds a `temporal` object containing the effective `config`, a
`summary`, and per-frame `frames` diagnostics. Each shell records its status
(`initial`, `increment`, `keyframe`, `recovered`, or `unresolved`), quality
metrics, rejection reasons, and keyframe references. These records distinguish
an accepted correction from an unresolved tracking interval. The angle series
also records `valid_top` and `valid_bottom` per frame.

This is a local correction: keyframes have bounded lifetimes and require
visible texture and overlapping views. It does not provide global loop closure
or repair an incorrect circle, camera model, or moving ball center. It can
reduce gradual tracking drift, but improved accuracy on real footage still
needs comparison with visible marks or an independent motion reference.

Set both `temporal.enabled: false` and `offline.enabled: false` to use the
original independent frame-pair tracker for a baseline comparison.
Configurations without either section also retain that behavior. Axis calibration continues to use its existing
frame-pair estimator; this measurement-tracking update alone does not require
repeating a passing axis calibration.

### Check boundaries across the clip

Temporal tracking erodes the usable masks by `boundary_margin_px` to keep
feature centers farther from mask boundaries. The default is a conservative
extra 3-pixel margin; it does not guarantee that every KLT window at every
pyramid level avoids the yoke, opening, or silhouette. Increasing the margin
can exclude contamination but also remove useful features, especially with
sparse markings. Inspect the tracking overlay after changing it.

Check the circle and HSV masks near the beginning, middle, and end, and at
blurred or partly obscured moments. Additional HSV samples help when a shell's
color falls outside the saved range as lighting changes. They do not provide
orientation anchors or automatically update the circle over time. If image
geometry changed between recordings, refit the circle and review the dependent
calibration; if the center moves within one recording, the current fixed-center
model is unsuitable. Stable geometry and masks do not require annotating every
frame before tracking.

To regenerate both overlays after enabling temporal tracking:

```powershell
.\.venv\Scripts\python.exe .\scripts\run.py --config .\config.yaml
.\.venv\Scripts\python.exe .\scripts\simulate_measured.py --config .\config.yaml --tests axes replay --stride 1 --render-width 640
```

The first command writes the tracking overlay and new measurements. The second
uses those measurements for the axes overlay and replay. Both write beneath
`output.dir`; preserve an earlier run or select a different output directory
in a copied config if you want a side-by-side comparison.

## Offline trajectory refinement

The current measurement config enables a second pass after temporal tracking:

```yaml
temporal:
  enabled: true
offline:
  enabled: true
  backward_window_frames: 12
  backward_stride: 3
  rate_window_s: 0.25
  rate_polynomial_order: 2
```

The tracker supplies persistent feature observations and accepted direct
keyframe matches. It also performs actual image matching backward from later
sharp frames into a bounded 12-frame buffer. Backward matching runs every three
frames (`backward_stride: 3`), matching intervening nearby frames and sampling
farther valid references at that stride;
weak earlier frames can be retried within the buffer
when their texture remains usable. These are fresh image observations, not
interpolated poses.

Direct keyframe and backward matches use a fixed template identity: the exact
source image and source feature define a landmark. Those identities are kept
separate from adjacent-frame persistent KLT tracks, so a drifted snapshot in
a later keyframe is not silently merged into the same material landmark.
Spatial selection reserves adjacent observations to connect intervening frames,
then admits fixed-template evidence (weight 2) in coherent landmark bundles.
Each admitted landmark retains observations in at least three distinct frames;
frame occupancy and a spatial grid balance coverage under the per-frame cap.
This prevents a keyframe from keeping long tracks while dropping the reference
observations needed by shorter reverse matches in intervening frames. Selection
does not rank observations by agreement with the initial pose estimate.

Offline refinement jointly adjusts camera-relative shell orientations and
feature locations on the sphere to reduce their pixel
reprojection errors: the differences between observed markings and their
predicted image positions. Observations from later images can therefore
correct earlier poses as well as the current pose. The calibrated camera,
ball center, and sphere radius remain fixed during this fit.

The current settings retain tracks seen in at least three frames and select
up to 100 observations per frame for fitting. A connected fit must converge,
not increase its robust pixel-error objective, preserve visibility and spatial
coverage, keep the final inlier graph connected to its anchor, and pass the
per-frame checks. Defaults require at least 12 inliers,
70% inlier support, and at most 2-pixel error for an observation to count as an
inlier. Changes to previously valid poses are limited to 5 degrees. A rejected
fit retains that component's original poses and validity; it is recorded as a
rejection, not silently presented as a refined result. Each component retains
its earliest trusted pose as its reference, so an error already in that anchor
can remain.

This independent stage retains each shell's general 3D rotation. It does not enforce
shared roll, zero gamma, or no slip. The resulting orientations still flow
through the calibrated caster-angle decomposition, and shell disagreement and
gamma remain diagnostics. The mechanical stage below then fits the shared
mechanism when enabled. Neither stage measures ground-relative contact slip.

Rates use native measurement timestamps and local polynomial fits within
contiguous valid intervals, with a 0.25-second quadratic fitting window by
default. They do not bridge missing measurements. Shared-alpha rates require
both shells to be valid in independent mode, preventing a change in the observed shell from
creating an artificial derivative spike. A single shell can still support an
alpha angle while its shared-alpha rate is missing. Each beta rate uses its
own shell's validity, and insufficient local samples leave rates missing.

Camera-frame angular-velocity vectors describe the full camera-relative
rotation, separately from the decomposed caster-angle rates. JSON stores
`frames.omega_top_camera_rad_s` and `frames.omega_bottom_camera_rad_s` as
`N x 3` arrays. CSV has separate x/y/z columns, for example
`omega_top_camera_x_rad_s`. These are spatial angular velocities satisfying
`dR/dt = skew(omega) @ R`, not world-relative contact-slip velocities.
Rate processing is recorded in `metadata.rate_processing` in `results.json`.

The 0.25-second window can attenuate brief steering transients. Compare results
with smaller and larger windows and use the same settings for both caster
designs. A smoother rate curve is not evidence of more accurate motion. These
rate fits do not smooth or replace the saved pose trajectory. Keep
`input.fps_override: null` to use native video timestamps.

Inspect `summary.offline_tracking` and the `offline` object in `results.json`
for final counts and refinement diagnostics, and the main
`frames.valid_top` / `frames.valid_bottom` flags for final pose
validity. The `temporal` diagnostics and feature `tracking_overlay.mp4` describe
the initial tracking pass; they are not post-refinement residuals or final
validity. Regenerate `simulation/axes_overlay.mp4` from the new results to view
the refined trajectory. `offline_observations.npz` preserves feature IDs,
undistorted pixel observations, weights, timestamps, fixed geometry, and the
initial/refined rotations and validity for later audits; retain it together
with the config and results. Observation archive schema version 2 also saves
`landmark_id`, `landmark_family`, `landmark_source_frame`, and
`landmark_source_track_id`, making fixed-template and adjacent-track
observations distinguishable.

Unresolved intervals without sufficient observation connections remain
invalid; fitting separate components does not establish their missing relative
motion. Joint fitting cannot recover information absent from the footage,
correct the fixed geometry, or
establish absolute accuracy without an independent reference. Repeated texture
and correlated tracking errors can also survive refinement. Compare equivalent
runs with `offline.enabled: false` and `true` (with `mechanical.enabled: false`), preserving separate output
directories, before using the results for quantitative design comparisons.
The two commands above run refinement and then render its results; no separate
offline command is needed.

## Joint mechanical fit and fixed shell gap

The current config also enables:

```yaml
mechanical:
  enabled: true
  gap_fraction: 0.10  # 20 mm gap / 200 mm ball diameter
```

This stage fits image observations jointly with exactly three motion
coordinates: shared roll `alpha`, independent `beta_top` and `beta_bottom`.
For the calibrated initial frame `F`, camera-relative shell rotation is
`F @ Rx(alpha) @ Rz(beta_shell) @ F.T`. Independent sideways tilt is excluded.
The two shell spin axes agree, while their outward cap directions are opposite.
There is no equal-spin or no-slip constraint.

`gap_fraction` is the physical separation between the rim planes divided by
ball diameter. The current measured dimensions are a 20 mm gap and 100 mm
ball radius, giving `20 / (2 * 100) = 0.10`. The fit uses a normalized radius;
there is no separate physical-radius setting for rotation estimation. Keep
`circle.r_px` in pixels. Zero means ideal touching rims. Cap landmarks, rendered
surfaces, grids, and probes respect this fixed 3D gap. Its projected pixel width
can change with perspective. The sphere center stays fixed, and this is not a
full collision model of the yoke or shell openings.

`run.py` performs the fit when enabled; it requires both temporal and offline
processing. To reuse saved observations without tracking again:

```powershell
.\.venv\Scripts\python.exe .\scripts\refine_mechanical.py --config .\config.yaml --results .\out\offline_verified\results.json --output .\out\mechanical_verified
.\.venv\Scripts\python.exe .\scripts\simulate_measured.py --config .\config.yaml --results .\out\mechanical_verified\results.json --output .\out\mechanical_verified\simulation --tests axes replay --stride 1 --render-width 640
```

That example uses the saved 120-frame run. Substitute another `results.json`
with its sibling schema-2-or-later `offline_observations.npz` for another run.
The refit validates camera geometry and landmark provenance and preserves its
source directory. Changes to starting roll or gap require refitting; changes
to intrinsics, distortion, or the sphere circle require retracking. Use
`refine_mechanical.py` to change the starting frame of constrained results;
`reorient_results.py` only re-expresses unconstrained results.

The optimizer uses the existing robust pixel loss and quality gates. Shared
roll is anchored once per connected joint trajectory; each shell component
has a separate spin reference. Per-frame acceptance requires enough well-spread
inliers connected to that reference. The 5-degree fit-correction limit is
relative to the mechanically projected starting guess; the separate projection
change from the unconstrained pose is reported. Rejected measurements remain
invalid, without falling back to independent poses under a constrained label.
Bad observations can still affect a robust fit, and disconnected components
inherit their input reference rather than recovering unseen motion.

Inspect `mechanical_report.json`, `results.json.mechanical`, and
`summary.mechanical_tracking`. CSV includes separate mechanical statuses.
`results.json.unconstrained` preserves the independent poses and checks;
schema 3 observation archives also preserve their matrices and validity.
Zero gamma and matching roll are imposed by the model, not independent proof
of accuracy. The runner's self-consistency checks use the preserved independent
estimates instead. Pixel residuals are fit diagnostics, not ground truth.

One visible shell can measure shared roll and its rate. A hidden shell's spin
remains missing. The axes overlay hides invalid shells, and replay explicitly
labels any held spin used for display. Each shell's swivel rate uses only its
own valid intervals; shared roll rates use either valid shell in this joint mode.
Neither rates nor missing poses are filled across unsupported intervals.

## Outputs and acceptance checks

The default real output directory is `out/real/`:

- `results.csv`: angles in radians/degrees, angular velocities, and per-frame
  quality columns
- `results.json`: metadata, complete time series, quality summary, and temporal
  correction / offline refinement diagnostics when enabled
- `offline_observations.npz`: saved feature observations when offline refinement
  is enabled
- `angles.png`
- `angular_velocities.png`
- `diagnostics.png`
- `tracking_overlay.mp4` unless disabled

All reported angles are accumulated relative to frame 0, which is treated as
identity. Start the main clip at a physically marked home pose if that zero
should coincide with the caster's mechanical zero.

The strict runner checks every frame pair for each hemisphere:

- mean Kabsch inlier residual at or below `0.5 deg`
- RANSAC inlier ratio at or above `0.7`
- per-frame median forward-backward tracking error at or below `1 px`
- minimum tracked count at or above `estimate.min_inliers` (default `8`)
- successful rotation solves on every frame pair (with temporal tracking,
  inspect the additional validity and recovery diagnostics; the original
  frame-pair checks still report weak intervals)
- median absolute `gamma` residual at or below `1 deg`

It also requires median top/bottom roll disagreement at or below `1 deg`, and
that the clip's own swivel axis agrees with `R_bc` to within `5 deg` (see
below).

### The swivel-axis cross-check

Every other check above is blind to one failure: a caster that was re-seated,
re-gripped, or otherwise re-oriented between axis calibration and the
measurement. Tracking stays clean, residuals stay small, and every reported
angle is silently expressed in the wrong frame.

`summary.swivel_axis_check` closes that gap. The two shells share the roll but
spin independently about the swivel axis, so `R_bottom^T @ R_top` is a
rotation about the swivel axis alone -- in any frame, without reference to
`R_bc`. Differencing it between consecutive frames leaves increments whose
common axis *is* the swivel axis in camera coordinates. The runner compares
that measured axis with `R_bc[:, 2]` and reports:

- `status: ok` -- the clip confirms the calibrated frame.
- `status: mismatch` -- `R_bc` does not describe this footage. Recalibrate the
  axes without disturbing the mount, or re-record the measurement clip in the
  pose the calibration clips were shot in. Do not trust `alpha` or `beta`.
- `status: not observable` -- the shells never spun far enough apart for this
  clip to see its own axis, so `R_bc` stays unverified against this footage.

`pca_explained_ratio` says how well a single axis explains the differential
motion. A high ratio with a large `disagreement_deg` is the signature of a
correctly measured but wrongly referenced clip; a low ratio means there was
not enough clean differential motion to conclude anything.

Also inspect the plots and JSON rather than relying only on the exit code:

- Valid solutions and tracked counts should remain healthy on every frame,
  not merely at the median.
- `gamma_top` and `gamma_bottom` should remain small without an upward trend.
- Roll recovered from top and bottom should agree, ideally within about 1
  degree on a clean clip.
- If a clip physically returns to its marked start pose, end-to-start rotation
  should be only a few degrees. That number is loop-closure drift only for a
  true return-to-start clip.

Real-video checks are self-consistency evidence, not external ground truth.
Passing synthetic data proves the implementation; it does not compensate for
bad real calibration, a moving camera, blur, or a violated motion model.

## Verifying measured angles by re-simulation

`scripts/simulate_measured.py` plays the measured angles back and checks them
three ways, weakest assumption first:

```powershell
python .\scripts\simulate_measured.py --config .\config.yaml --stride 4
```

- `axes` draws the calibrated ball axes, a ball-fixed graticule, and numbered
  ball-fixed probe rings on the **real** footage, rotated by the measured
  per-frame orientation. The large arrows are fixed initial reference axes;
  the graticule and rings move. Rings start at virtual surface locations, not
  detected paint marks. Check whether each stays fixed relative to neighboring
  visible texture, ignoring yoke/opening occlusions. Sliding indicates an error
  in the predicted surface motion. This test uses no synthetic rendering.
- `replay` renders the measured trajectory with the synthetic ball renderer
  using the real `K`, circle, and `R_bc`, beside the real clip. It renders the
  modeled `Rx(alpha) @ Rz(beta)` manifold and the full measured orientation
  separately, so the difference between the two panels is exactly the motion
  the caster model cannot represent.
- `roundtrip` re-measures the rendered sequence with the same pipeline and
  compares the recovered angles with the ones fed in. A small round-trip error
  means the estimator and the angle conventions are self-consistent; it does
  **not** validate `R_bc`, the circle, or the physical motion model. A large
  round-trip error can also result from loss of tracking or motion outside the
  estimator's characterized range; it does not uniquely identify a code bug.

Read them together. A clean round trip establishes consistency on that
rendered sequence; a sliding real overlay still needs camera geometry,
tracking quality, and accumulated motion to be checked.

Select individual tests with `--tests axes`, and use `--stride`/`--frames` to
keep rendering time reasonable on long clips. Outputs go to
`<output.dir>/simulation/`.

### Clips that start away from the calibrated home pose

Keep `frame_calib.R_bc` as the home calibration. Set the clip's starting roll
relative to that home using `frame_calib.initial_roll_deg` (default `0.0`).
Use a physically measured, signed angle; the current config explicitly keeps
zero until a different starting roll is established. The software does not
automatically choose an offset to minimize gamma or shell disagreement.
The runner uses `R_bc @ Rx(initial_roll_deg)` for decomposition and saves that
effective frame in the results. Reported `alpha` remains relative to the clip's
start. Adding an offset to angles decomposed in the wrong frame would leave
roll/spin mixing uncorrected. This setting assumes the mount and roll axis did
not change; a re-seated camera requires recalibration.

A starting-pose mismatch can tilt the rendered shell boundary and mix roll
and spin during decomposition. It is different from accumulated tracking
error. If `A` is the camera-relative orientation and `F` the initial frame,
the full overlay renders `F @ (F.T @ A @ F) = A @ F`. Correcting `F` changes
the grid's reference placement; it does not repair errors in `A`. An initial
offset and tracking drift can both be present in the same recording.

`frame_calib.top_shell_sign` identifies which geometric cap carries the top
HSV class: `1` for +z, `-1` for -z. It controls replay colour/motion assignment;
it does not change the tracked colour labels or angular sign convention.
The full replay preserves each shell's measured alpha, gamma, and beta. The
model panel uses shared alpha and zero gamma. Synthetic speckles are an
illustration, so their individual positions need not match the real paint.

**Historical September 10 example; not an offset recommendation for the current
clip.** For that recording, `config.upright.aligned.yaml` copies the passed
home calibration and sets a **visually estimated** `initial_roll_deg: 90.0`
and `top_shell_sign: -1`. Its outputs use a separate directory. Those old config
files are absent from the current checkout. With the historical files restored,
these commands re-express those saved rotations and render that recording:

```powershell
.\.venv\Scripts\python.exe .\scripts\reorient_results.py --config .\config.upright.aligned.yaml --results .\out\real_20260910\results.json
.\.venv\Scripts\python.exe .\scripts\simulate_measured.py --config .\config.upright.aligned.yaml --tests axes replay --stride 1 --render-width 640
```

Reorientation reconstructs the original full rotations, preserves failed
steps, checks the input/camera/circle, and refuses to overwrite the source
results. A pose change cannot recover motion lost during tracking failures.
Changing the initial pose in YAML requires reorientation or a tracking rerun
before simulation; the simulator rejects a frame mismatch. Omit
`--render-width` when requesting the optional `roundtrip` test.

To also regenerate the feature tracking overlay from scratch, run
`scripts/run.py --config .\config.upright.aligned.yaml` before simulation.
The original feature overlay remains valid because reorientation does not
alter the image correspondences.

Video measurement now uses normalized native presentation timestamps unless
`input.fps_override` explicitly requests retiming. Results record the timing
source; wholly unavailable timestamps warn and fall back to nominal FPS,
while malformed or mixed timing is rejected. Diagnostic MP4s still use a
constant frame rate; use `time_s` in CSV/JSON as the measurement clock. These
changes retain the existing `Rx(alpha) @ Rz(beta)` convention; they do not
implement the additional PDF contact/velocity model.

## Plotting saved motion

To visualize the saved measurement with missing motion explicitly marked:

```powershell
.\.venv\Scripts\python.exe .\scripts\plot_motion_history.py --results .\out\real_20260910_aligned\results.json --output .\out\motion_history_20260910 --with-clip
```

This writes a five-page PDF, PNG/SVG figures, a CSV, and per-shell rotation
matrices/quaternions. Camera-relative motion is shown separately from
calibration-dependent roll/spin. Interval rates use the saved timestamps;
failed intervals remain missing. `--with-clip` also verifies source timestamps
and adds actual video snapshots. Read the September 10 interpretation in
[out/motion_history_20260910/README.md](out/motion_history_20260910/README.md).

## Recording advice

- Bolt the camera to the chassis and disable electronic stabilization.
- Lock resolution, zoom, focus, and preferably exposure after calibration.
- Use bright diffuse light and a short exposure; a global-shutter camera is
  preferable for fast rotation.
- Use a matte surface and dense, irregular speckles. Keep top/bottom colors
  well separated in hue and brightness.
- Make the ball large in frame without cropping its silhouette.
- Keep per-frame rotation below the Stage D `delta_max`; use a 1.5-2x frame-rate
  safety margin when possible.
- Avoid lossy re-encoding and preserve reliable FPS/timestamps.
- Record a separate loop-closure clip that returns to a physically marked home
  pose.
- For axis calibration, isolate one monotonic motion at a time. Wobble or mixed
  roll/swivel invalidates the common-axis estimate.

## Troubleshooting

| Symptom | Likely cause and action |
| --- | --- |
| No/too few tracks | Tune HSV, add speckles/light, check yoke mask, or use equator mode for one-color footage. |
| High residual and low inlier ratio | Inspect `tracking_overlay.mp4`; look for blur, glare, yoke leakage, wrong HSV labels, or excessive rotation/frame. |
| Forward-backward error above 1 px | Raise FPS, shorten exposure, improve texture, or reduce inter-frame speed. |
| Point count collapses near the limb | Add speckle density/coverage; cautiously raise `limb_cull_deg` or improve the viewpoint. |
| Top/bottom roll disagree | Recheck `R_bc` and hemisphere segmentation contamination. |
| `gamma` grows over time | Recheck `R_bc`, circle stability, accumulated drift, mount wobble, and whether motion violates the `Rx(alpha) @ Rz(beta)` model. Check `swivel_axis_check` first: a large `disagreement_deg` explains a large `gamma` outright. |
| `swivel_axis_check` reports `mismatch` | Check the initial roll relative to the calibrated home pose and accumulated tracking drift. A known starting roll can be set with `initial_roll_deg`; a changed camera/roll-axis relationship needs new calibration. |
| Detection misses most of one hemisphere | Its HSV range was fitted on one frame, usually a shaded one. Re-run `inspect_hsv.py --frames 12` and sample that shell bright, shaded, frontal, and near the limb; check `yoke_hsv.hi[2]` is not eating it. |
| Few tracks despite correct colour masks | Sparse markings on plain surface. Raise `segment.grow_px` so the untextured surface between marks becomes trackable, and raise `track.max_corners`. |
| Circle preview misses the ball | Rerun `calibrate_circle.py --manual`; use widely separated silhouette points. |
| Checkerboard RMS above 1 px | Retake sharp views spanning the image with more varied tilts; verify inner-corner counts. |
| Axis calibration has high spread despite strong PCA/tracking | Inspect the report's raw/rejected/retained counts. Use `--min-axis-step-deg 0.20` to reject noise-dominated near-stationary increments; re-record if fewer than eight clean moving steps remain. |
| Axis calibration fails for other reasons | Ensure clips start at home, contain isolated monotonic motion, have correct signs, and provide at least eight valid retained steps. |
| Recovered sign is reversed | Rerun axis calibration with `--roll-sign -1` and/or `--swivel-sign -1`. |
| Image sequence has no FPS | Set `input.fps_override` or pass `--fps` to axis calibration. |
| Overlay codec fails | Run with `--no-overlay` to diagnose the rest, or choose a supported four-character `output.overlay_codec`. |

For interactive HSV and circle tools, run from a normal Windows desktop
session with OpenCV GUI access. The circle tool also supports three supplied
`--point U V` arguments when interactive clicking is unavailable.
