# New-video inspection and calibration workflow

## Rerun with paint support and interval recovery

The current `config.yaml` enables these changes. Keep the existing three MKV
recordings and calibrated starting frame. Retrack **the main clip** to collect
observations with the new mask; refitting the old observation archive cannot
remove tracks that came from an exposed inner surface.

From `ball_caster_rot` in PowerShell:

```powershell
.\.venv\Scripts\python.exe .\scripts\run.py --config .\config.yaml --output-dir .\out\real_recovered --strict
.\.venv\Scripts\python.exe .\scripts\simulate_measured.py --config .\config.yaml --results .\out\real_recovered\results.json --output .\out\real_recovered\simulation --tests axes replay --stride 1 --render-width 640
```

`--output-dir` is relative to the working directory. These commands preserve
`out/real`. The first writes `tracking_overlay.mp4`, measurements and diagnostic
reports; the second writes `simulation/axes_overlay.mp4` and
`simulation/replay.mp4`. Run them separately: `--strict` may return exit code 2
after writing results if measurements remain unresolved. The simulation can
still show that diagnostic result; missing poses are not valid measurements.

The new settings are:

```yaml
segment:
  paint_support:
    enabled: true
    min_saturation: 30
    min_value: 40
    max_distance_px: 17
offline:
  graph_recovery_enabled: true
  graph_recovery_max_step_deg: 30.0
  window_frames: 60
  window_overlap_frames: 20
mechanical:
  window_frames: 60
  overlap_frames: 20
```

- **Paint support:** each image supplies high-confidence colored seeds before
  mask growth or yoke arbitration. Tracking stays within 17 pixels of that
  support, excluding isolated gray inner-disc/rim pixels accepted by the old
  broad green HSV range. This follows visible paint without assuming the
  starting roll. It is not a full semantic rim/yoke detector: paint reflections
  or unpainted boundaries immediately beside a mark can still contaminate a
  tracking window. `inspect_hsv.py` previews the same processed masks.
- **Image-supported recovery:** nearby forward and backward track observations
  must connect a lost frame to at least two already anchored frames, agree on
  rotation, and pass pixel residual, spread and visibility checks. Recovery
  can extend through successive overlaps. A blank image or disconnected set
  of later tracks remains unresolved. The held display pose is never an anchor.
- **Local offline fits:** independent and mechanical fits use overlapping
  60-frame windows and smaller retries when needed. Accepted overlap poses
  stay fixed in the original camera reference. The independent fit can prune
  explicitly failed image observations and refit its neighbors. Mechanical
  fitting joins only through accepted overlap, retaining shared roll and each
  shell's own observed swivel reference. It cannot replace a missing swivel
  measurement with the other shell's roll.

Inspect `results.json.offline.shells.<shell>.graph_recovery`, offline `windows`,
and `mechanical_report.json` (`windows`, `gaps`, `summary`). Forward tracking,
recovered initialization and final mechanical validity are separate stages.
More accepted frames do not establish angular accuracy without ground truth.

These changes do not require rerunning pure-roll/pure-swivel axis calibration
just to try the main clip. Revisit it if the camera/mechanism reference changes
or its diagnostic evidence is poor. Initial orientation and the physical
common-center sphere/cap assumption are unchanged; verify those separately.

### Checks on the existing recordings

The implementation passed the full 341-test suite; subsequent mechanical
iterator/retry fixes passed 34 focused tests. In a controlled saved-observation
check of the first 120 frames, windowed mechanical fitting accepted **56 top /
51 bottom** poses, compared with **44 / 38** in the previous single fit. Pixel
quality limits stayed unchanged. It stopped after the last supported interval;
this is additional usable coverage, not a demonstrated angular-accuracy gain.

Sequentially decoded mask samples retain hundreds of exterior corners while
removing the known inner-disc false corners in frames 0 and 3. Exact frames 133
and 438 also show motion blur: better masks alone do not make those steps
trustworthy. The new graph checks correctly reject their old saved observations
at the existing 2-pixel reprojection limit. Do not relax that limit solely to
make an overlay continue through a gap.

The final support distance is **17 pixels**. An 8-pixel trial removed useful
texture and caused an earlier loss at frame 62. The 17-pixel setting still
eliminates the known gray-disc false corners, while an 85-frame forward check
retained 83 top and 79 bottom poses and recovered after the brief blurred
interval. This is a tracking check, not a final mechanical accuracy result.
The saved `out/rim_recovery_validation/run160` trial used the superseded
8-pixel setting; use the commands above to generate results with the final
configuration.

## Previous full-run failure

For the latest 557-frame run with no moving axes, see
[the failure review](LATEST_RUN_FAILURE_REVIEW.md). Its mechanical solver did
not converge, and the forward tracker also lost its reference during the clip;
the masks were not empty. A 200-evaluation refit alone did not fix the failure.
After repairing observation selection, the 200-evaluation fit converged in 91
evaluations, but only 40 top and 36 bottom poses passed out of 557. The current
config retains that budget per fit and now uses the local processing above.

## Shared roll, independent swivel, and a configurable fixed gap

The current config enables a joint mechanical fit **after** the independent
offline fit. Both shells share roll `alpha`, each retains its own swivel
`beta`, and independent sideways tilt `gamma` is excluded. The fit adjusts
these three coordinates and surface landmarks against observed image pixels.
It does not assume equal shell spins or no slip.

```yaml
mechanical:
  enabled: true
  gap_fraction: 0.10  # 20 mm gap / (2 * 100 mm ball radius)
```

`gap_fraction = measured rim-plane gap / ball diameter`, with both dimensions
in the same units. The measured gap is **20 mm** and ball radius is **100 mm**,
so the current setting is `20 / 200 = 0.10`. The rotation pipeline uses a
normalized sphere radius; there is no separate physical-radius setting for
this fit. Keep `circle.r_px` as the fitted image radius in pixels. Physical
radius is still needed for future linear-speed/slip calculations. Zero would
mean ideal touching rims. The
distance remains fixed in 3D; its pixel width can change with viewing angle.
This spherical-cap model keeps the common center fixed; it does not translate
the shells apart or model the yoke's full shape.

### Apply the constraint to saved observations without tracking again

For the saved **120-frame** run currently available in this workspace:

```powershell
.\.venv\Scripts\python.exe .\scripts\refine_mechanical.py --config .\config.yaml --results .\out\offline_verified\results.json --output .\out\mechanical_verified
.\.venv\Scripts\python.exe .\scripts\simulate_measured.py --config .\config.yaml --results .\out\mechanical_verified\results.json --output .\out\mechanical_verified\simulation --tests axes replay --stride 1 --render-width 640
```

For another run, replace the results path. Its sibling
`offline_observations.npz` must use schema 2 or later and retain landmark
source identities. The script checks that camera/circle/video geometry
matches, and writes a separate output directory. It can refit a corrected
`initial_roll_deg` or shell assignment without repeating image tracking.
Changing camera intrinsics, distortion, or circle geometry requires retracking.

For the **full clip**, use the two commands in the next section with the
current config; `run.py` now includes mechanical fitting. `input.max_frames`
controls the measured length, while simulator `--frames` only limits playback.
After changing the gap or starting roll, refit and regenerate the simulation;
changing YAML alone does not change saved measurements.

### Interpret the constrained result

- `frames.valid_top` / `valid_bottom` are final measurement validity.
- `mechanical_report.json` and `results.json.mechanical` record image residuals,
  per-frame reasons, and `summary` counts. CSV adds `top_mechanical_status` and
  `bottom_mechanical_status`.
- `results.json.unconstrained` preserves the independent poses and diagnostics.
  Schema 3 observation archives preserve forward, unconstrained, and final
  rotations/validity for later refits.
- A failed mechanical pose stays unresolved; the independent pose is **not**
  substituted as a valid constrained result. The axes overlay hides that
  shell. Replay may hold its spin for display, while sharing the observed roll,
  with an explicit missing-measurement label.
- Shared roll and zero gamma are now imposed facts, so their agreement is not
  an accuracy test. The runner checks the preserved independent estimates.
  Each connected trajectory still inherits its trusted starting reference.
- A visible shell can support common roll and its rate; the hidden shell's
  independent swivel and swivel rate remain missing. Rates never bridge
  intervals where their required measurements are missing.

The fixed mechanical geometry removes incompatible shell boundaries. It does
not guarantee that the grid follows the painted marks, correct a wrong initial
frame, or measure ground slip without independent ground-relative motion.
`tracking_overlay.mp4` still shows the forward feature-tracking stage;
`axes_overlay.mp4` and `replay.mp4` use the final fitted results.

**Mechanical validation before entering the measured gap, September 15:**
278 automated tests passed. With the earlier zero-gap setting, the saved
120-frame recording was refitted with unchanged 2-pixel / 70% support gates:
34 top-shell poses and 19 bottom-shell poses passed. The remaining poses are
unresolved; this is not a validated full measurement trajectory. Median selected
pixel residual changed from 1.427 to 1.373 pixels. The audit also found saved
bottom-shell tracks on the upper shell/yoke, incompatible with the required cap.
Review the broad green HSV mask and yoke exclusion before retracking; forcing
those points onto the bottom cap would bias the motion. A configurable gap
does not fix mislabeled observations. The preview files are under
`out/mechanical_verified/simulation/`. These are zero-gap results; refit and
rerender with the current `0.10` setting before using the measured geometry.

## Rerun the calibrated clip with temporal and offline refinement

The latest isolated-motion axis calibration has passed. The measurement
pipeline now supports persistent tracks and nearby keyframe correction,
enabled by `temporal.enabled: true` in the current `config.yaml`. Offline
trajectory refinement is enabled with `offline.enabled: true` and runs after
the initial tracking pass. This software update alone does not require another
circle fit, HSV annotation, or axis
calibration. The preparation steps below remain useful for new recordings or
an actual setup change.

From the project root, run:

```powershell
.\.venv\Scripts\python.exe .\scripts\run.py --config .\config.yaml
.\.venv\Scripts\python.exe .\scripts\simulate_measured.py --config .\config.yaml --tests axes replay --stride 1 --render-width 640
```

The first command writes the measurements and `tracking_overlay.mp4`; the
second writes `simulation/axes_overlay.mp4` and `simulation/replay.mp4` beneath
`output.dir` (currently `out/real`). These replace the derived files there.
For comparison, preserve the previous output or use a copied config with a
different `output.dir`. Disable `mechanical.enabled` before disabling either
upstream stage. Disable both `offline.enabled` and `temporal.enabled`
in that copy to run the original frame-pair tracker. Disable only
`offline.enabled` to compare against temporal tracking alone.

### What the correction does

- Features retain their identities across frames, and lost tracks are
  replenished with new identities.
- Direct image matching to nearby keyframes is attempted every three frames
  by default and when an adjacent estimate or previous pose is unreliable,
  provided sharpness is sufficient. It uses tracked or predicted positions
  as starting guesses. Accepted matches
  from one or more overlapping keyframes refine the current orientation.
  This initial temporal pass does not re-optimize earlier saved poses; the
  offline pass below does.
- Inlier count, match agreement, residual, sharpness, and spatial spread gate
  acceptance. Sharpness is compared with recent image history as texture and
  lighting change. An earlier good keyframe can recover the current
  orientation after a weak adjacent pair, including the intervening motion.
- If recovery is not supported, the affected shell's frame remains invalid.
  Later recovery does not fill earlier unresolved frames with invented
  measurements. The axes overlay hides its moving grid and probes and shows
  `UNRESOLVED`; replay explicitly labels the display fallback for a missing
  pose. Check diagnostics as well as playback.

Inspect `results.json` under `temporal`: it contains the effective `config`,
`summary`, and per-frame `frames` records. Per-shell statuses are `initial`,
`increment`, `keyframe`, `recovered`, and `unresolved`, with quality metrics,
rejection reasons, and keyframe references. The existing consistency checks
still apply; a correction is not evidence of ground-truth accuracy. The angle
series also includes `valid_top` and `valid_bottom` flags. With offline
refinement enabled, these final frame flags take precedence over the initial
temporal statuses when interpreting measurement coverage.

Keyframes stay nearby and require overlapping, sufficiently sharp texture.
This is not global loop closure and cannot fix a wrong camera model, a bad
circle, or movement of the ball center. `calibrate_axes.py` keeps its existing
frame-pair path, independent of these measurement options.

### What the offline pass adds

After tracking, the pipeline jointly refines shell poses and persistent
feature locations on the spherical surface using pixel reprojection errors.
It uses persistent observations and direct matches to nearby keyframes, so
later images can improve earlier poses. Camera calibration, ball center, and
sphere radius stay fixed. This is image-based trajectory refinement, not just
smoothing the reported angles.

It also matches actual images backward from later sharp frames into a bounded
12-frame buffer (`offline.backward_window_frames: 12`). Backward matching runs
every three frames (`backward_stride: 3`), matching intervening nearby frames
and sampling farther valid references at that stride; weak earlier frames can
be retried when there is enough usable texture. These observations are checked
for agreement and quality before they enter the fit. They do not synthesize
new frames or interpolate unseen motion.

Direct keyframe and backward matches identify a landmark by its exact source
image and source feature, separately from adjacent-frame persistent KLT
tracks. This prevents drifted snapshots in different keyframes from being
merged under one assumed surface point. Selection admits observations of a
landmark together across at least three frames, reserves adjacent tracks, and
balances spatial coverage and frame occupancy before spending the remaining
budget on fixed-template evidence (weight 2). This preserves reverse-template
connections to intervening frames. It does not rank observations by agreement
with the initial pose estimate.

The current fit uses tracks observed in at least three frames and selects up
to 100 observations per frame. Each connected fit must converge, not increase the
robust pixel objective, preserve visibility/spatial coverage, keep the final
inlier graph connected to its anchor, and pass the
per-frame checks: at least 12 inliers and 70% inlier support, with a 2-pixel
inlier threshold. Changes to previously valid poses are limited to 5 degrees.
If these checks fail, that component keeps its original poses and validity,
and the rejection is reported. Each component inherits its earliest trusted
input pose; fitting cannot independently correct that reference.

This independent stage keeps each shell's general 3D rotation. Shared roll,
zero gamma, and no-slip mechanics are not imposed in this step; the existing
caster kinematics still decompose the refined rotations afterward. The optional
mechanical stage above then fits shared roll and independent swivel. Shell-roll
disagreement and gamma therefore remain useful model checks. Independent
ground-relative motion and contact geometry are still needed to measure slip.

Angular rates use native timestamps and local polynomial fitting within
contiguous valid intervals. The current settings are
`offline.rate_window_s: 0.25` and `offline.rate_polynomial_order: 2`.
They do not connect across missing observations. Shared-alpha rates require
both shells to be valid in the independent mode, so switching the visible shell cannot produce a
spurious rate spike. Each beta rate uses its own shell's validity. A rate can
therefore be missing even when an angle is available; short valid intervals
may not have enough samples for a rate fit.

Camera-frame angular-velocity vectors are saved as `N x 3` JSON arrays
`frames.omega_top_camera_rad_s` and `frames.omega_bottom_camera_rad_s`, with
separate x/y/z CSV columns such as `omega_top_camera_x_rad_s`. These measure
full camera-relative rotation and are distinct from caster-angle derivatives
and world-relative slip velocity. `metadata.rate_processing` records the rate method
and settings in `results.json`.

CSV columns `top_offline_status` and `bottom_offline_status` distinguish refined
or recovered poses from original forward estimates retained after a rejected fit.

A 0.25-second window can attenuate short steering transients. Test sensitivity
to smaller and larger windows, and use the same window for both caster
designs. These fits affect rates, not the saved pose trajectory. Keep
`input.fps_override: null` so the time axis remains the native recording times.

Inspect `results.json` under `summary.offline_tracking` and `offline` for the
final counts and refinement outcome, and the main
`frames.valid_top` / `frames.valid_bottom` flags for final validity. Retain
`offline_observations.npz` with the config and results for future audits. It
contains feature IDs, undistorted pixel observations, weights, timestamps,
fixed geometry, and initial/refined rotations and validity.
Archive schema version 2 also records `landmark_id`, `landmark_family`,
`landmark_source_frame`, and `landmark_source_track_id` for auditing each
observation's template origin.
The `temporal` quality records and feature tracking video describe the initial
pass; they are not post-refinement fit statistics. The regenerated axes overlay
uses the final refined poses.

Unresolved intervals without sufficient observation connections stay invalid;
separate fitted components do not establish their missing relative motion.
This pass cannot repair
a wrong fixed camera/circle model, recover invisible texture, or guarantee
ground-truth accuracy. Use an independent known-angle or encoder reference
before interpreting small differences between caster designs.

### Independent-stage validation before adding the mechanical fit

**Validation on September 15:** the implementation passed 213 automated tests,
including known synthetic rotations, missing images, and timestamp/rate checks.
On the first 120 frames of the current recording, the offline fit rejected all
trajectory components and retained the forward poses. The first failing top
frame was 35 (24/36 observations within 2 pixels); bottom components failed at
frames 1 (6/21) and 76 (13/28). The required fraction is 70%. Increasing the
observation budget to 400 or 800 per frame also did not pass. These are fit
consistency results, not measured angular accuracy. The new rate processing
and explicit missing-data handling still apply, but improved orientation
accuracy has not been demonstrated on this recording. The audited results
are in `out/offline_verified/results.json`; a 30-frame sampled replay and axes
overlay are in `out/offline_final_validation/simulation_smoke`.

### Check the starting pose separately

Keep `frame_calib.R_bc` as the calibrated mechanical-home frame. The current
`frame_calib.initial_roll_deg` is explicitly `0.0`; if this clip starts at a
different roll, supply its physically measured signed angle relative to that
same home. Do not choose an offset just because it reduces gamma or makes the
simulation look better. This correction assumes an unchanged mount and roll
axis; a changed camera/caster relationship requires calibration review.

The measurement frame becomes `R_bc @ Rx(initial_roll_deg)`. Reported alpha
still measures change from the first frame. A wrong initial frame can mix
roll/spin and tilt the simulator's starting shell boundary, while accumulating
tracking errors cause a separate trajectory error. Correcting the reference
does not remove those accumulated errors. Offline refinement addresses the
observed trajectory and does not automatically determine the mechanical home.

The simulator's speckles and numbered grid probes are virtual surface points,
not registered copies of the painted marks. Compare whether the moving grid
stays fixed relative to nearby real texture, rather than expecting a synthetic
dot to start on a particular painted mark. After changing initial roll, rerun
the two commands above; changing YAML and rendering old measurements alone is
rejected because their coordinate frames would disagree.

The approximately 90-degree starting roll in the historical notes below
belongs to the September 10 recording. It is not a measured correction or a
recommended setting for the current clip.

### Do more frames need annotation?

Spot-check the circle and masks at the beginning, middle, and end, plus frames
with changing light, blur, or occlusion. Add HSV samples if those checks show
that real markings disappear from a mask or the yoke enters it. More annotation
alone does not correct orientation drift or a changing circle. If the camera
geometry changed between clips, refit and review calibration; if the center
moves within a clip, the fixed-center model needs a different recording or a
geometry-model change.

Temporal tracking automatically adds `temporal.boundary_margin_px: 3` pixels
of mask erosion to reduce boundary contamination. This extra margin is
conservative and does not guarantee that every tracking window avoids an
occluder at all pyramid levels. Increase it only while checking that enough
well-distributed features remain. Stable masks and geometry do not require
annotating every frame before generating the overlays.

## Preparation for a new recording or changed setup

Updated September 15, 2026 for the current checkout. Use `config.yaml` for the commands in this section. Its `input.path` now points to `data/upright/clip.mkv`, and `input.fps_override` remains null so the pipeline uses native video timestamps.

The current upright measurement and calibration clips have already been prepared, and the latest axis calibration passed. The older `config.upright.review.yaml` and `config.upright.aligned.yaml` files are absent. For replacement footage, review whether the saved camera, HSV, circle, and axis values in `config.yaml` still apply; changing `input.path` alone does not establish that. Repeat only the preparation steps required by the changed setup.

### 1. Convert all three original videos to upright MKVs

Run this PowerShell block. Each command rotates the original pixels by 180 degrees and writes a lossless FFV1 MKV. Existing output files are skipped, including the current main-clip copy; skipping does not verify their contents. Use this only for the three original upside-down recordings, and keep the originals.

```powershell
Set-Location 'C:\Users\kyu\Documents\Upenn\Caster_Vision\ball_caster_rot'
New-Item -ItemType Directory -Force -Path .\data\upright | Out-Null

if (-not (Test-Path -LiteralPath .\data\upright\clip.mkv)) {
    ffmpeg -hide_banner -nostdin -n -noautorotate -i .\data\clip.mp4 -map 0:v:0 -vf "format=bgr0,hflip,vflip" -fps_mode passthrough -enc_time_base:v demux -c:v ffv1 -level 3 -coder 1 -context 1 -g 1 -pix_fmt bgr0 -slicecrc 1 -an .\data\upright\clip.mkv
    if ($LASTEXITCODE -ne 0) { throw 'Main-clip conversion failed. Inspect the output before retrying.' }
}

if (-not (Test-Path -LiteralPath .\data\upright\pure_swivel.mkv)) {
    ffmpeg -hide_banner -nostdin -n -noautorotate -i .\data\axis_calibration\pure_swivel.mp4 -map 0:v:0 -vf "format=bgr0,hflip,vflip" -fps_mode passthrough -enc_time_base:v demux -c:v ffv1 -level 3 -coder 1 -context 1 -g 1 -pix_fmt bgr0 -slicecrc 1 -an .\data\upright\pure_swivel.mkv
    if ($LASTEXITCODE -ne 0) { throw 'Pure-swivel conversion failed. Inspect the output before retrying.' }
}

if (-not (Test-Path -LiteralPath .\data\upright\pure_roll.mkv)) {
    ffmpeg -hide_banner -nostdin -n -noautorotate -i .\data\axis_calibration\pure_roll.mp4 -map 0:v:0 -vf "format=bgr0,hflip,vflip" -fps_mode passthrough -enc_time_base:v demux -c:v ffv1 -level 3 -coder 1 -context 1 -g 1 -pix_fmt bgr0 -slicecrc 1 -an .\data\upright\pure_roll.mkv
    if ($LASTEXITCODE -ne 0) { throw 'Pure-roll conversion failed. Inspect the output before retrying.' }
}
```

The commands preserve frame timing apart from Matroska timestamp rounding; they do not synthesize frames or force 60 fps. A failed conversion can leave a partial file, so inspect or move that file before rerunning its command. Open all three copies and check the beginning, middle, and end for the correct orientation and a stationary ball center. Correct the camera-image orientation consistently; do not rotate individual clips just to put the red shell at screen-top.

Optional decoded-frame-count check for the copies (the September 10 inspection found 551 main, 167 swivel, and 144 roll frames in the originals):

```powershell
foreach ($clip in @('clip', 'pure_swivel', 'pure_roll')) {
    ffprobe -v error -select_streams v:0 -count_frames -show_entries format=filename:stream=codec_name,width,height,nb_read_frames -of json ".\data\upright\$clip.mkv"
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect upright $clip.mkv" }
}
```

### 2. Confirm the camera calibration and input paths

`config.yaml` now contains:

```yaml
input:
  path: data/upright/clip.mkv
  type: auto
  max_frames: null
  fps_override: null
```

There is one measurement input in the YAML. The two axis-calibration paths are mandatory `--roll` and `--swivel` arguments to `calibrate_axes.py`, supplied in step 5 below; they are not separate YAML input keys.

Reuse `camera.K` and `camera.dist` only if they describe the same lens settings, resolution, crop, and final upright image orientation. If rotating the footage restores the orientation used for camera calibration, keep the camera values as they are. If it changes the calibrated orientation, update the full camera model or recalibrate before proceeding. All three clips must use the same camera setup and image convention.

Use the project Python for annotation. If the environment has not been set up, run `.\setup.ps1` once; it installs the pinned dependencies and runs the tests. No environment activation is needed for the commands below. Run the interactive annotation tools from your desktop PowerShell session.

### 3. Fit and save the circle on the upright main clip

For a recording with changed image geometry, fit the circle before color annotation because the HSV preview uses the configured circle to limit its masks. This tool reads `input.path` and undistorts the image before displaying it:

```powershell
.\.venv\Scripts\python.exe .\scripts\calibrate_circle.py --config .\config.yaml --manual --preview .\out\upright_annotation\circle_preview.png
```

Click three widely separated points on the spherical exterior, avoiding the yoke, seam opening, and shadows. Press `R` to reset, Enter to accept and save the circle into `config.yaml`, or `Q`/Esc to cancel. Check that the same circle describes later frames and both calibration clips; do not force one circle to fit different camera positions. Omitting `--manual` runs an automatic fit and saves it immediately.

### 4. Annotate colors and save HSV thresholds

```powershell
.\.venv\Scripts\python.exe .\scripts\inspect_hsv.py --config .\config.yaml --frames 12 --write
```

- `T`: sample the red physical shell (`top`).
- `B`: sample the green physical shell (`bottom`).
- `Y`: sample the dark yoke.
- Click to collect samples; `N`/`P` moves between frames. Sample each shell in at least three frames, including bright and shaded regions.
- Space toggles the preview, `U` undoes the last sample for the active class, and `R` resets that class.
- Enter writes thresholds to `config.yaml`; all three classes need samples. `Q`/Esc cancels without saving.

Keep shell identity attached to color even when the shells swap positions on screen. Saving also enables the yoke mask. The sampler previews raw HSV thresholds, not the final mask growth or yoke exclusion; check for wrong-shell or yoke tracks in the tracking overlay in step 6. The sampler may load fewer than 12 frames because OpenCV can overestimate MKV frame counts; confirm that the loaded frames cover the relevant poses.

### 5. Calibrate axes using both upright motion clips

Both calibration clips must start from the same mechanical home pose without disturbing the camera/caster mount. The selected interval must contain isolated, monotonic roll or swivel motion, without regripping, reversals, or long pauses. A numerical `PASS` cannot establish these physical prerequisites. Once the camera, circle, and color masks are ready, run:

```powershell
.\.venv\Scripts\python.exe .\scripts\calibrate_axes.py --config .\config.yaml --roll .\data\upright\pure_roll.mkv --swivel .\data\upright\pure_swivel.mkv --min-axis-step-deg 0.20 --max-frames 120 --report .\out\upright_annotation\axis_calibration_first120_report.json
```

The first-120-frame interval comes from the previous inspection of these particular recordings; do not assume it is appropriate for replacement videos. The command saves `frame_calib.R_bc` into `config.yaml` only on `PASS`. If it fails, inspect the report and fix the calibration before tracking. Use `--roll-sign -1` or `--swivel-sign -1` only when the recorded physical sweep is negative in the code's right-hand convention. Both axes are estimated from the red/top shell, so it must visibly move in both clips.

### 6. Run tracking after calibration passes

The main clip's initial mechanical pose must also be consistent with the axis-calibration reference. Measure a different starting roll relative to the same home and set `initial_roll_deg` as explained above. The approximate quarter-turn mismatch in the historical inspection belongs to the September 10 recording, not necessarily the current clip. `initial_roll_deg` adjusts the 3D reference frame, not the video pixels. Do not assume that a 180-degree video correction fixes this separate pose issue.

```powershell
.\.venv\Scripts\python.exe .\scripts\run.py --config .\config.yaml
```

This uses the upright main clip and writes to `output.dir` (currently `out/real`). Move any results you want to retain before running it, because it replaces the derived files there. Inspect the tracking overlay and quality summary before interpreting the results; failed tracking and consistency warnings still matter.

## Historical inspection and replay notes (September 10, 2026)

The remainder records the earlier run, including its commands, measurements, and now-absent configuration files. Statements that calibration or conversion was completed refer to that run; they do not mean the current checkout is ready. Follow the current steps above for preparation and annotation. Historical replay commands require restoring their referenced configurations and results first. In particular, the approximate 90-degree pose setting and the older rate/tracking limitations below describe that recording and software version, not a recommended pose or the enabled offline pass today.

Reviewed September 10, 2026. The three uploaded videos can be rotated 180 degrees without synthesizing frames. Upright FFV1 copies are in `data/upright/`. The original MP4 files and existing calibration/results are preserved.

**The HSV, circle, and isolated-motion axis calibration are complete. Initial-pose correction and native video timestamp handling are now implemented.** The corrected main-clip results improve substantially but still fail the existing consistency targets. They remain diagnostic measurements, and the updated PDF kinematics are still not implemented. See [UPDATED_KINEMATICS_REVIEW.md](UPDATED_KINEMATICS_REVIEW.md) for the PDF convention comparison.

`config.upright.review.yaml` retains the user's completed calibration and original run settings. `config.upright.aligned.yaml` copies that calibration into a separate configuration with `initial_roll_deg: 90.0`, `top_shell_sign: -1`, and output directory `out/real_20260910_aligned`. The 90-degree starting roll is an approximate visual alignment, not a new measured mechanical-home calibration. Use the aligned configuration for the corrected replay commands below; there is no need to repeat the completed calibration steps.

### What the files actually contain

All three videos are 1920 x 1080. Sequential decoding recovered every reported frame. There is no rotation metadata in the source streams; the upside-down orientation is in the pixels.

| Original file | Decoded frames | Average FPS | Duration | Largest interval between frames |
|---|---:|---:|---:|---:|
| `data/axis_calibration/pure_roll.mp4` | 144 | 30.000 | 4.800 s | 33.333 ms |
| `data/axis_calibration/pure_swivel.mp4` | 167 | 22.587 | 7.394 s | 255.833 ms |
| `data/clip.mp4` | 551 | 23.220 | 23.729 s | 128.033 ms |

The latter two have variable frame timing. Their nominal `r_frame_rate=60/1` does not mean 60 captured frames per second. Most intervals are approximately 32 or 48 ms, with several much longer gaps. Metadata and every original presentation timestamp are saved in `out/inspection_20260910/video_metadata.json` and `original_video_timing.json`. The last frame timestamp is slightly shorter than the stream duration because the final frame has a display duration too.

Final upright-copy validation is saved in `out/inspection_20260910/upright_verification.json`: all three decoded frame counts match, all timestamps are strictly increasing, and five sampled frames per clip (including the final frame) are pixel-exact 180-degree rotations. Maximum timestamp rounding is 0.433 ms. The three copies total approximately 619 MB.

The older `out/real/results.json` contains **328 frames from earlier footage**, not these 551 frames. Its median roll disagreement is 20.34 degrees, median top/bottom gamma residuals are 24.56/11.06 degrees, and swivel-axis disagreement is 31.26 degrees. These fail the existing acceptance targets. The previous circle `(837.846, 687.418, 374.133)` belonged to that scene. The new circle has already been fitted and saved in both upright configurations.

The saved original main-clip measurements are under `out/real_20260910/`; the reoriented measurements are under `out/real_20260910_aligned/`. Unit tests cover the initial-pose correction, preservation of failed tracking steps, and native timestamp behavior. These software checks do not establish the accuracy of the real recording or implement the PDF convention.

### Rotation and 60 fps

The copies were made with the following command for each source, changing the input/output paths as needed:

```powershell
ffmpeg -hide_banner -nostdin -n -noautorotate -i .\data\clip.mp4 -map 0:v:0 -vf "format=bgr0,hflip,vflip" -fps_mode passthrough -enc_time_base:v demux -c:v ffv1 -level 3 -coder 1 -context 1 -g 1 -pix_fmt bgr0 -slicecrc 1 -an .\data\upright\clip.mkv
```

The output names are `clip.mkv`, `pure_roll.mkv`, and `pure_swivel.mkv`, all under `data/upright/`. `-n` refuses to overwrite an existing copy. Two flips produce a true 180-degree rotation; a vertical flip alone mirrors the geometry. RGB FFV1 avoids another lossy encode and avoids YUV range conversion altering the decoded colors. These copies contain video only. Matroska stores timestamps on a millisecond grid, so the original exact timestamps remain available in the inspection JSON. `-enc_time_base:v demux` matters: passthrough alone allowed this FFmpeg build to round timestamps to 1/30 s and produce duplicate times during the initial inspection; the final copies correct that.

FFmpeg's [`fps` filter](https://ffmpeg.org/ffmpeg-filters.html#fps) duplicates or drops frames. Its [`minterpolate` filter](https://ffmpeg.org/ffmpeg-filters.html#minterpolate) estimates intermediate frames; [RIFE](https://github.com/hzwer/ECCV2022-RIFE) also supports learned frame interpolation. These can produce a smoother 60 fps viewing copy, for example:

```powershell
ffmpeg -n -i .\data\upright\clip.mkv -vf "minterpolate=fps=60:mi_mode=mci" -c:v libx264 -crf 18 -pix_fmt yuv420p -an .\data\upright\clip_preview_60fps.mp4
```

That optional command has not been run. Synthesized frames are estimates, not new measurements of the shell. My assessment for this speckle/KLT pipeline is that they can introduce correlated errors or distort marks, especially at occlusions and the shell seam. Evaluate any experimental interpolation against independent ground truth; do not infer improved accuracy from a smoother overlay or more frames.

Do not set `input.fps_override: 60` on the original footage. It changes the assigned time and angular rates without adding frames. For greater measurement accuracy, improve lighting/exposure and texture, reduce physical speed, and avoid capture/export gaps. The README's characterized 5 degrees/frame is a benchmark-specific limit, not a guarantee: at a uniform 29 fps it corresponds to 145 degrees/s before any safety margin, but a 256 ms gap reaches 5 degrees at only about 20 degrees/s.

### Native timing is now used for measurements

`ballrot/io_frames.py` now reads presentation timestamps immediately after sequential video decoding and normalizes the first frame to zero. `scripts/run.py` passes timestamped `source.records()` into the pipeline when `input.fps_override` is null. Saved-result reorientation also reads the source video times and recalculates the rates. Both paths record timing provenance in the results metadata.

Native timestamps must be finite and strictly increasing. When a backend reports repeated zero timestamps, the reader can warn and use nominal FPS, recorded as `fps_fallback`; other invalid or non-increasing timestamps are rejected. The current upright main clip reports `native_video_pts`. An explicit FPS override deliberately assigns uniform times and should be left null for this recording.

The reoriented 551-frame output spans **23.696 s** from first to last frame, compared with **18.333 s** in the original run that used the upright MKV's nominal 30 fps. Correct timing fixes the time base; it does not repair missing or inaccurate rotations. Existing velocity gradients span valid samples across failed pairs, so rates near those gaps require particular care.

Tracking overlays and simulated videos are still encoded at a constant diagnostic frame rate. Their frame correspondence can be inspected, but playback duration and apparent playback speed are not the recording's measurement clock. Use the saved `time_s` values for timing.

### Calibration, in order

Use the project Python explicitly; the system Python environment has incompatible scientific-package versions.

```powershell
Set-Location 'C:\Users\kyu\Documents\Upenn\Caster_Vision\ball_caster_rot'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe .\scripts\validate_synthetic.py --config .\config.example.yaml --output .\out\synthetic_20260910_full
```

The full validation run for these initial-pose changes completed with **OVERALL: PASS**, Stages A-E, in [out/synthetic_initial_pose_validation/validation_report.json](out/synthetic_initial_pose_validation/validation_report.json). All **86 unit tests** also passed. These checks validate the implementation against controlled data; the current real clip still reports consistency warnings.

The following steps document the calibration procedure for future recordings or an actual setup change. **They have already been completed for the current upright clip.** `config.upright.review.yaml` points to the upright main clip and `out/real_20260910`; it now contains the user's HSV thresholds, fitted circle, and passing `R_bc`. Following the user's confirmation that the camera setup has not changed, it reuses `camera.K`, `camera.dist`, and `camera.fov_deg` from `config.baseline.yaml`, assuming the corrected upright imagery matches the calibration orientation. Preserve these saved values unless a new calibration is needed.

1. **Confirm the recording assumptions.** The camera must be rigid relative to the caster housing/chassis, and the ball center and silhouette must remain stationary in the image. Check the beginning, middle, and end of each clip, and that the common circle also describes both calibration clips. The current model cannot compensate for a moving center. Keep the physical shell identity attached to its color: the existing settings label red as `top` and green as `bottom`, even when their positions on screen swap.

2. **Reuse the confirmed camera intrinsics in the final image orientation.** The user confirmed the camera setup has not changed, and the supplied K/dist have been restored in the upright config. Skip new checkerboard calibration when lens settings, resolution, crop, and final pixel orientation also match. If the 180-degree correction restores the orientation in which K was calibrated, do not transform K a second time. If it changes the calibrated orientation, transform the full camera model correctly or recalibrate. Do not rotate the main video alone while leaving calibration imagery/axes in another orientation.

   Only if a new camera calibration is needed, capture 10-20 sharp checkerboard images at varied positions and tilts, at 1920 x 1080 with fixed lens settings. Use the same upright image convention. Substitute the board's actual inner-corner counts and square size in this example:

   ```powershell
   .\.venv\Scripts\python.exe .\scripts\calibrate_camera.py --images .\data\camera_calibration --board-cols 9 --board-rows 6 --square-size 25 --config .\config.upright.review.yaml --preview-dir .\out\camera_calibration_20260910
   ```

   Inspect the accepted images and report; aim for reprojection RMS below about 1 pixel. The script warns, rather than refusing to write K, if RMS is high. Repeat circle and axis calibration whenever K/dist changes.

3. **Tune masks across lighting and pose changes.**

   ```powershell
   .\.venv\Scripts\python.exe .\scripts\inspect_hsv.py --config .\config.upright.review.yaml --clip .\data\upright\clip.mkv --frames 12 --write
   ```

   Select `T`, `B`, or `Y` and click red, green, or black-yoke samples. Use `N`/`P` to visit frames; sample each shell in at least three different frames, including bright and shaded regions. Space previews the masks; Enter writes them. The current `grow_px: 17` can help sparse triangular markings, but check that growth does not reach the other shell or yoke. Do not classify by screen-top/screen-bottom.

   OpenCV estimates MKV frame count from duration and the 30 fps header, so this sampler can load fewer than the requested 12 frames. Check how many frames actually loaded and that the samples cover the relevant poses; the verified decoded counts are in the table above.

4. **Fit the circle after undistortion.**

   ```powershell
   .\.venv\Scripts\python.exe .\scripts\calibrate_circle.py --config .\config.upright.review.yaml --clip .\data\upright\clip.mkv --preview .\out\real_20260910\circle_preview.png
   ```

   Inspect the silhouette fit. If wrong, repeat with `--manual` and click three widely separated points on the spherical exterior, avoiding the yoke, seam opening, and shadows. Review the same ring on later frames and both axis clips. Do not force one circle to fit visibly different camera positions.

5. **Establish a consistent axis reference and mechanical home.** The sampled main clip starts roughly a quarter turn away from the calibration clips' initial assembly pose. This is visual evidence of a reference mismatch, not a measured 90-degree calibration. For future acquisition, start the main and pure-motion clips at the same marked home pose without disturbing the mount. The current main clip can now use `frame_calib.initial_roll_deg` to express rotations in the corrected initial frame, as described below. Adding a scalar angle offset to already incorrect decompositions would not fix them.

   Use isolated, monotonic 60-90 degree sweeps without regripping, long pauses, or reversals. The current calibrator estimates both axes from the **red/top shell**, so that shell must visibly move in the swivel clip. The full current roll clip failed because of late axis deviations. The user has already run the following passing calibration on the first 120 frames of both clips; this command is recorded for reproducibility, not a required rerun:

   ```powershell
   .\.venv\Scripts\python.exe .\scripts\calibrate_axes.py --config .\config.upright.review.yaml --roll .\data\upright\pure_roll.mkv --swivel .\data\upright\pure_swivel.mkv --min-axis-step-deg 0.20 --max-frames 120 --report .\out\real_20260910\axis_calibration_first120_report.json
   ```

   The passing report has roll/swivel p95 axis spreads of **13.052/8.066 degrees**, **83/67** retained increments, and raw axis separation **89.161 degrees**. It saved `frame_calib.R_bc` successfully. The aligned config keeps that same home calibration. Use `--roll-sign -1` and/or `--swivel-sign -1` only when the known physical motion is negative in the **current code's** right-hand convention. No spin sign can be established from an unlabeled axis line alone. For the PDF, separately measure the mechanical initial tilt relative to vertical and the physical ball radius in metres.

### Correct the initial pose and generate the replay

The new `frame_calib.initial_roll_deg` sets the measurement frame to `R_bc_home @ Rx(initial_roll_deg)` (with the angle converted to radians). `R_bc` in the YAML remains the saved home calibration. Output metadata stores both `R_bc_home` and the effective `R_bc`. Reported motion starts at zero relative to the first frame; the configured initial roll is not automatically an absolute mechanical tilt for the PDF contact equations.

`frame_calib.top_shell_sign: -1` places the red/top shell on the opposite rendered cap for this recording. This controls the physical shell assignment in the replay, while HSV tracking labels remain attached to their original colors. It does not change the calibration-axis signs or recover missing motion.

Reorient the existing measurements, then render the corrected replay:

```powershell
.\.venv\Scripts\python.exe .\scripts\reorient_results.py --config .\config.upright.aligned.yaml --results .\out\real_20260910\results.json
.\.venv\Scripts\python.exe .\scripts\simulate_measured.py --config .\config.upright.aligned.yaml --tests axes replay --stride 1 --render-width 640
```

Reorientation has already been completed for the 551 saved frames. It reconstructs the full measured camera rotations from the old per-shell XYZ angles, including the out-of-model gamma component, then decomposes them in the corrected frame. It preserves the original run and its per-frame quality records. It refuses to overwrite the source result directory or accept changed camera/circle geometry or a different input. Geometry changes require retracking.

The aligned results improve these diagnostics:

| Metric | Original frame | Approximate 90-degree correction |
|---|---:|---:|
| Median absolute top gamma | 40.384 deg | 11.098 deg |
| Median absolute bottom gamma | 35.347 deg | 11.345 deg |
| Median shell-roll disagreement | 54.572 deg | 18.517 deg |
| Swivel-axis disagreement | 81.901 deg | 8.200 deg |

**Consistency remains `WARN`.** The saved recording contains 9 failed top-shell pairs and 35 failed bottom-shell pairs; reorientation preserves every one of these failures. A failed pair holds its previous accumulated matrix and leaves its reported shell angles missing. Later orientations can remain inaccurate because the lost motion has not been recovered. The 90-degree alignment improves the frame convention but does not make the current recording a validated kinematic measurement.

Reorientation does not regenerate a feature-tracking overlay. The original overlay is still at `out/real_20260910/tracking_overlay.mp4`. To generate a fresh feature overlay in the aligned output directory, run the full tracking pipeline, then rerun the simulation command above:

```powershell
.\.venv\Scripts\python.exe .\scripts\run.py --config .\config.upright.aligned.yaml
```

This overwrites the derived aligned measurement files, keeping the original run intact. It also uses the corrected frame and native timestamps. Retracking with identical masks and tracking parameters does not itself resolve failed steps. `--strict` is optional for either measurement command and exits with code 2 when the existing consistency targets are missed. `run.py` has no `--clip` or `--output` option; change the YAML instead.

Use `--frames 120` on `simulate_measured.py` for a shorter visual preview. Keep `--stride 1` to display every frame. `--render-width 640` reduces diagnostic rendering cost and is for axes/replay only; omit it for the separate synthetic `roundtrip` test. Synthetic round-trip agreement does not validate the real clip's calibration or recover missing tracking.

| Output under `out/real_20260910_aligned/` | Contents |
|---|---|
| `results.csv` | Frame/time, alpha, both beta values in radians/degrees, three rates in rad/s, separate shell alpha/gamma, inlier ratios, residuals, tracked counts |
| `results.json` | Metadata, angular series, per-frame solve success and quality, consistency summary |
| `angles.png`, `angular_velocities.png`, `diagnostics.png` | Angle, rate, and tracking-quality plots |
| `tracking_overlay.mp4` | Feature matches and inlier counts; generated only by a full `run.py` run |
| `simulation/axes_overlay.mp4`, `simulation/replay.mp4`, `simulation/simulation_report.json` | Geometric overlay, rendered replay, selected simulation comparisons |

The current strict targets include successful solutions on every frame pair, at least eight tracked points, inlier ratio >=0.7, maximum per-frame mean inlier residual <=0.5 degrees, forward/backward error <=1 pixel, median absolute gamma <=1 degree, median shell-roll disagreement <=1 degree, and observable swivel-axis agreement within 5 degrees. A failed increment loses accumulated motion. An unobservable swivel axis or approximate camera does not automatically fail strict mode, so a zero exit code does not establish calibration.

**“All tracking details” requires another export.** These historical CSV/JSON files do not save per-feature pixel correspondences, individual errors/inlier masks, or the incremental and accumulated rotation matrices held in `PipelineResult`. The original tracker used for those runs redetected features for each pair without persistent identities. The temporal mode described above now maintains identities and adds correction diagnostics; it does not turn these historical outputs into complete track exports. A complete export should preserve those arrays, native timestamps, units, calibration snapshot, convention, and validity of the accumulated trajectory.

### Updated kinematics status

The implementation still uses `Rx(alpha) @ Rz(beta_h)`. The PDF defines the rod along +y and an angular-velocity sign corresponding to `Ry(alpha) @ Rz(-beta_h)`. This can be reconciled with a proper frame conversion and beta sign change, provided the reference pose is correct; changing an axis label alone is insufficient.

For PDF angles and a known contacting shell, the ideal no-slip relations are `vx=R*alpha_dot`, `vy=R*sin(alpha_mechanical)*beta_dot_contact`, and effective radius magnitude `R*abs(sin(alpha_mechanical))`. Those are model predictions and are **not implemented in current standard outputs**. Contact identity, physical radius, mechanical home, and the local signed-angle convention are needed. Actual slip additionally requires independently measured chassis velocity; forces and impulses require physical parameters such as inertia and load. The separate [kinematics review](UPDATED_KINEMATICS_REVIEW.md) lists equations, conversion, singularities, and implementation requirements.
