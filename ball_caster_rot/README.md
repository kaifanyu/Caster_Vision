# Ball-caster rotation measurement

This project measures the three rotational degrees of freedom of a two-piece,
random-speckled ball caster from video:

- `alpha`: shared roll of the assembly about ball `+x`
- `beta_top`: independent top-hemisphere spin about ball `+z`
- `beta_bottom`: independent bottom-hemisphere spin about ball `+z`

It uses classical computer vision: HSV/equator segmentation, forward-backward
KLT tracking, ray-sphere unprojection, and RANSAC/Kabsch rotation fitting. No
GPU or deep-learning model is required.

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
- `SPACE` toggles a live preview of the mask the current samples would give.
- `U` undoes the last click; `R` clears the active class.
- `Enter` prints and writes the suggested ranges; `Q`/Esc cancels.

The status line counts clicks *and* distinct frames per class, and the script
warns if a hemisphere was sampled on fewer than three frames. OpenCV hue is
`0..179`; wrapped red ranges where `lo.H > hi.H` are supported.

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

## Outputs and acceptance checks

The default real output directory is `out/real/`:

- `results.csv`: angles in radians/degrees, angular velocities, and per-frame
  quality columns
- `results.json`: metadata, complete time series, and quality summary
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
- successful rotation solves on every frame pair (a missed increment makes
  later absolute angles incomplete)
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
  per-frame orientation. Each numbered ring is pinned to one point of the
  physical shell, so if the rotation is right the ring stays on the same
  speckle for the whole clip. A ring that slides off shows the error directly,
  in pixels, with no model in the loop. This test uses no synthetic rendering.
- `replay` renders the measured trajectory with the synthetic ball renderer
  using the real `K`, circle, and `R_bc`, beside the real clip. It renders the
  modeled `Rx(alpha) @ Rz(beta)` manifold and the full measured orientation
  separately, so the difference between the two panels is exactly the motion
  the caster model cannot represent.
- `roundtrip` re-measures the rendered sequence with the same pipeline and
  compares the recovered angles with the ones fed in. A small round-trip error
  means the estimator and the angle conventions are self-consistent; it does
  **not** validate `R_bc`, the circle, or the physical motion model. A large
  round-trip error is a code or convention bug.

Read them together. A clean round trip beside a sliding `axes` overlay means
the code is right and the calibration or the footage is not.

Select individual tests with `--tests axes`, and use `--stride`/`--frames` to
keep rendering time reasonable on long clips. Outputs go to
`<output.dir>/simulation/`.

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
| `swivel_axis_check` reports `mismatch` | The ball was re-oriented relative to the camera between axis calibration and this clip. Redo axis calibration in the measurement pose, or re-record the clip in the calibration pose. Per-frame residuals cannot detect this. |
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
