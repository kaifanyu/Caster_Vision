# Caster wheel rotation and ball-vs-swivel comparison

This repository implements the complete synthetic-first workflow in
`swivel_caster_and_comparison_SPEC.md` and incorporates the existing Approach-A
ball-caster pipeline. It can:

- recover a standard caster's swivel angle `psi` from a fork-mounted ArUco tag;
- recover wheel roll `phi` from random sidewall texture with KLT, ray-plane
  lifting, and constrained RANSAC/Kabsch;
- recover the ball caster's two-hemisphere rotations with the existing
  ray-sphere pipeline;
- adapt both devices to the same `CasterFrame` record;
- compute alignment, scrub, slip, lag, settling, shimmy, and rolling efficiency
  without device-specific logic in the metric layer;
- run paired ball/swivel experiments from a manifest and produce common plots,
  CSV, JSON, and Markdown reports; and
- validate conventions, vision recovery, metrics, robustness, and both adapters
  on deterministic synthetic data.

No GPU or learned model is required. Python 3.10, NumPy/SciPy, OpenCV contrib,
Matplotlib, and PyYAML are pinned in `requirements.txt`.

Real results are only trustworthy after calibration. The example configuration
deliberately leaves `camera.K`, `camera.T_car_from_cam`, both effective radii,
`swivel.marker.psi0_rad`, `ball.R_bc`, and `ball.R_ball_to_car` unset.

## The markings you need

### Standard swivel caster wheel

Mark both flat sidewalls, not the tread.

1. Add a dense, random, non-repeating field of small high-contrast speckles to
   **both wheel faces**. A regular dot grid, evenly spaced ring, molded lettering,
   or repeated pattern is ambiguous after rotation and should not be the only
   texture.
2. Put the small dots in the usable annulus, approximately `0.20 R` to `0.90 R`
   from the hub. Keep paint off the bearing/hub, tread, rim transition, and any
   area the fork can hide or rub.
3. At the intended camera distance, make most dots about 3-8 pixels across.
   That is often roughly 1-3 mm physically, but pixel size is the real criterion.
   Mix dot sizes and shapes. Aim for roughly 80-150 dots per face and at least
   30 clearly visible corners in a typical frame. The estimator's eight-inlier
   minimum is a rejection floor, not a marking target.
4. Put **one larger red reference dot on each face**, at the same material angle
   on the wheel. Make it unique and about 3-4 times the small-dot diameter
   (often 6-10 mm). The default detector accepts OpenCV red hue ranges 0-12 and
   168-179 with saturation at least 80 and value at least 70; tune these values
   in `swivel.reference_dot.hsv_ranges` under the actual lighting. Its configured
   area gate is 8-1000 pixels squared.
5. Use matte, rubber-compatible paint markers or a durable matte coating. Clean
   a test patch first and verify adhesion without weakening the wheel. Avoid
   glossy tape, thick blobs that can contact the floor, and reflective paint.

The small speckles provide accurate inter-frame roll. The single large dot is
an absolute phase cross-check and repairs short missed intervals. It cannot
determine how many complete turns occurred during a long fully hidden gap, so
FPS and visibility still matter.

### Fork/swivel housing

The wheel texture does not measure fork yaw by itself. Mount a `DICT_4X4_50`,
ID 0 ArUco marker on a flat, rigid top plate that rotates with the fork. Do not
put it on the chassis.

- The default black square is exactly 40 mm, excluding its white border.
- Leave at least a 10 mm clean white quiet border on every side.
- Print at 100%/Actual Size, measure both the 40 mm square and supplied 20 mm
  check bar with calipers, and put the measured black-square size in
  `swivel.marker.size_m`.
- Keep it planar, matte, unobstructed by bolts/cables, and visible through the
  full swivel range.

Generate the print sheet with:

```powershell
python .\scripts\make_aruco_marker.py --dictionary DICT_4X4_50 --id 0 --marker-size-mm 40 --quiet-border-mm 10 --output .\out\calibration\aruco_marker.pdf
```

### Ball caster

Keep the existing Approach-A convention: use two visually distinct random
speckle populations so the top and bottom hemispheres can be separated. The
example config expects high-saturation cyan/blue on the top (`H=85..115`) and
magenta on the bottom (`H=135..175`), plus a dark-yoke exclusion mask. Those
ranges are only a starting point and will not necessarily match your paint or
lighting. Sample the real colors with `inspect_ball_hsv.py`. The standalone
synthetic renderer deliberately uses its own internally matched red/cyan pair.

If only one marker color is possible, use an image-equator split instead (two
colors are usually more robust):

```yaml
ball:
  segment:
    mode: equator
    equator:
      angle_deg: 0
      offset_px: 0
      deadband_px: 4
```

Keep the marks matte, random, non-repeating, and distributed over the visible
ball surface. Do not paint a repeating latitude/longitude grid, the bearing
contact patches, or any surface the yoke rubs. Size the colored marks for about
3-8 camera pixels at the recording distance and aim for at least a few dozen
clean, separated tracks on each visible hemisphere; eight inliers is only the
estimator's rejection floor.

### Camera and lighting

- Bolt the camera rigidly to the robot chassis. A room-fixed camera violates
  both device models.
- For the swivel caster, use an oblique downward view that sees the top tag and
  a substantial wheel face. Both sidewalls are marked because the visible face
  changes during swivel. A brief roll dropout near an edge-on view is expected
  and is masked; reposition the camera if that band lasts too long.
- For the ball caster, the projected center/circle must remain fixed in the
  chassis camera. Make the ball large in frame without cropping its silhouette.
- Use bright diffuse light, short exposure, locked focus/zoom/resolution, no
  electronic stabilization, and preferably a global shutter. Avoid glare and
  preserve a constant frame rate with no dropped/duplicated frames.

## Windows setup

Open a normal PowerShell window. Python 3.10 must be available through the
Windows `py` launcher.

```powershell
Set-Location 'C:\Users\kyu\Documents\Upenn\Caster_Vision\caster_wheel_rot'
py -3.10 --version
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
.\.venv\Scripts\Activate.ps1
python --version
python -m pytest -q
```

`setup.ps1` creates `.venv`, installs the pinned dependencies, creates
`config.yaml` from `config.example.yaml` if it is absent, and runs the tests.
It never overwrites an existing `config.yaml`. `Set-ExecutionPolicy -Scope
Process` affects only the current PowerShell process.

To repair dependencies without rerunning tests:

```powershell
.\setup.ps1 -SkipTests
```

## Run the ground-truth validation first

There are two independent release gates: the complete Approach-A ball vision
gate, and the combined swivel/common-metrics/comparison gate. Run both before
interpreting real data.

The shorter ball run still exercises every A-E gate and sweep with fewer frames
(it can take a few minutes on a CPU):

```powershell
python .\scripts\validate_synthetic_ball.py --config .\config.example.yaml --quick --output .\out\synthetic_ball_quick
```

The full ball release gate is:

```powershell
python .\scripts\validate_synthetic_ball.py --config .\config.example.yaml --output .\out\synthetic_ball
```

The shorter combined run touches S-A through S-E plus the S5/S6 integration
gates and is useful while configuring:

```powershell
python .\scripts\validate_synthetic_swivel.py --config .\config.example.yaml --quick --output .\out\synthetic_quick
```

The full combined release gate is:

```powershell
python .\scripts\validate_synthetic_swivel.py --config .\config.example.yaml --output .\out\synthetic
```

Do not interpret real data unless **both** full commands end in `OVERALL: PASS`.
The combined command checks:

- S-A: frames, rotation signs, ray-plane round trip, and exact roll extraction;
- S-B: tag yaw below 1 degree RMSE clean and 2 degrees at 1-pixel noise;
- S-C: wheel roll below the same thresholds, small off-axis residual, and
  reference-dot phase recovery;
- S-D: straight, spin-in-place, and circle metrics against analytic truth;
- S5: rendered ball images through the production ball pipeline and adapter;
- S6: rendered ball and swivel clips through their production pipelines,
  the same metric function, and the comparison report; and
- S-E: pixel noise, blur, glare, speed, and edge-on/dropout behavior.

Artifacts include `validation_report.json`, robustness plots/data, generated
common-record/car-track fixtures, and `synthetic_comparison/report/`. The
separate renderer command below is what writes inspectable image frames and
`ground_truth.json`.

You can also render one inspectable swivel sequence:

```powershell
python .\synthetic\generate_swivel.py --case tracking --frames 91 --fps 60 --delta-phi-deg 3 --delta-psi-deg 1 --output .\out\synthetic\inspect
```

Or render an inspectable ball sequence:

```powershell
python .\synthetic\generate_ball.py --output .\out\synthetic\ball_demo --mode scripted --frames 121
```

The characterized limits are reported as degrees per frame. Size the camera
rate with:

```text
FPS >= max(peak |phi_dot| / phi_limit, peak |psi_dot| / psi_limit)
```

Use degrees/second with degrees/frame, then add a 1.5-2x safety factor. The
current full synthetic sweep remains below 2 degrees RMSE through 10 degrees/frame
of roll and 2 degrees/frame of swivel. For example, a 79.375 mm wheel rolling
at 1 m/s turns about 722 degrees/s, requiring at least 73 FPS from the roll
limit; approximately 120-150 FPS is a sensible safety range before considering
the peak swivel rate.

The standalone ball sweep currently reports a bounded 5 degrees/frame limit
(the 10 degrees/frame case is the first to exceed its 2-degree criterion). For
the ball, divide the largest expected `alpha`/`beta` angular speed by 5, then
apply the same 1.5-2x FPS safety factor.

## Coordinate and timing conventions

- Camera: OpenCV `+x` right, `+y` down, `+z` forward.
- Car: `+x` forward, `+y` left, `+z` up.
- `T_car_from_cam` transforms camera-frame points into car coordinates.
- `roll_axis_car` is horizontal and unit length. The rolling heading is
  `normalize(z cross roll_axis_car)` and is sign-calibrated so forward is `+x`.
- `omega_roll` is nonnegative. Direction is carried by `roll_axis_car`.
- Positive `omega_spin` is about car `+z`.
- Every `CasterFrame.t` must already be on the car-track clock. The stored
  `car_track.time_offset_s` is added to raw caster timestamps, not to the car
  CSV. A positive value moves caster samples later.

Video/image timing is currently constant-frame-rate: samples are assigned
`frame_index / selected_FPS`; container per-frame PTS and clock-drift fitting
are not used. `input.fps_override` takes precedence over video metadata, and a
trusted run stops when neither is available. Use CFR footage, verify that no
frames were dropped/duplicated, and check synchronization near both ends of a
long recording. Split the run or add a clock-rate calibration if drift exceeds
about one frame.

For a conventional zero pose, `axle0_car: [0, -1, 0]` gives a `+x` rolling
heading. Distances are metres, angles radians, and time seconds.

## Swivel-caster calibration

Keep the exact camera resolution, crop, focus, zoom, exposure mode, and mount
unchanged after calibration. All calibration writes are atomic and preserve
unrelated YAML keys.

### 1. Camera intrinsics

Capture at least 10 sharp checkerboard views spanning the image with varied
tilts. `--board-cols` and `--board-rows` are inner-corner counts. For a 9x6
board with 25 mm squares:

```powershell
python .\scripts\calibrate_swivel.py --config .\config.yaml intrinsics --images '.\data\camera_calibration\*.png' --board-cols 9 --board-rows 6 --square-size 0.025 --min-detections 8 --preview-dir .\out\calibration\intrinsics_previews --report .\out\calibration\intrinsics.json
```

Retake the set if coverage is poor or RMS reprojection error is around or above
1 pixel. Distortion and `K` must correspond to the real recording resolution.

### 2. Camera-to-car extrinsics

Measure at least four non-collinear 3D points in the car frame and their pixel
locations in the original camera image. The solver applies the configured
`camera.dist`, so do not pre-undistort these pixel coordinates. More points
with depth/height variation are better. A CSV uses:

```csv
x,y,z,u,v
0.10,-0.12,0.00,183.2,401.8
0.45,-0.12,0.00,511.7,395.1
0.45,0.12,0.00,534.9,258.4
0.10,0.12,0.00,201.5,263.2
```

Run:

```powershell
python .\scripts\calibrate_swivel.py --config .\config.yaml extrinsics --correspondences .\data\camera_calibration\car_image_points.csv --max-rms-px 2 --report .\out\calibration\extrinsics.json
```

If CAD/measurement already provides the exact transform, write it directly:

```powershell
python .\scripts\calibrate_swivel.py --config .\config.yaml extrinsics --T-car-from-cam '@.\data\camera_calibration\T_car_from_cam.json' --report .\out\calibration\extrinsics.json
```

The JSON file is a numeric 4x4 homogeneous matrix. Do not use a car-to-camera
matrix in this slot.

### 3. Measured caster geometry

Measure these in the car frame at the straight-ahead zero pose:

- `S`: a reference point on the vertical swivel axis;
- the vector from `S` to the wheel hub center (this includes trail and height);
- the signed axle direction;
- unloaded geometric wheel radius and wheel width.

Example for the values shipped only as placeholders in the example config:

```powershell
python .\scripts\calibrate_swivel.py --config .\config.yaml geometry --swivel-axis-car 0.250 0.000 0.160 --hub-offset0-car -0.050 0.000 -0.081 --axle0-car 0 -1 0 --wheel-radius-m 0.079375 --wheel-width-m 0.035 --report .\out\calibration\geometry.json
```

The command normalizes the axle and reports signed trail. These geometric
values define the changing hub/sidewall plane. `wheel_radius_m` is used for
vision geometry; the separately loaded `r_eff_m` below is used for rolling
speed and accounts for rubber compression.

### 4. Swivel zero

Park the fork straight ahead and stationary, with the tag fully visible, then
record a short clip:

```powershell
python .\scripts\calibrate_swivel.py --config .\config.yaml psi-zero --clip .\data\swivel\psi_zero.mp4 --start-s 0.5 --end-s 2.5 --report .\out\calibration\psi_zero.json
```

This writes `swivel.marker.psi0_rad`. A normal measurement run refuses to use
an unset zero.

### 5. Roll direction and loaded effective radius

Record a straight, known-forward run. A tracking-only pass writes
`out/swivel_sign/swivel_track.csv` even before `r_eff_m` is known:

```powershell
python .\scripts\run_swivel.py --config .\config.yaml --clip .\data\swivel\known_forward.mp4 --tracking-only --output .\out\swivel_sign
```

Read the first and last reliable `phi_rad` values and supply them in radians:

```powershell
python .\scripts\calibrate_swivel.py --config .\config.yaml roll-sign --phi-start 0.12 --phi-end 24.91 --report .\out\calibration\roll_sign.json
```

Then roll a loaded robot through a measured distance without steering, count
full wheel revolutions (or use reliable `phi` endpoints), and calibrate:

```powershell
python .\scripts\calibrate_swivel.py --config .\config.yaml radius --distance-m 2.000 --revolutions 4.02 --report .\out\calibration\swivel_radius.json
```

or:

```powershell
python .\scripts\calibrate_swivel.py --config .\config.yaml radius --distance-m 2.000 --phi-start 0.12 --phi-end 25.38 --report .\out\calibration\swivel_radius.json
```

The formula is `r_eff = distance / abs(delta_phi)`. Repeat at the intended
payload and tire pressure/temperature; a radius error appears as constant slip.

### 6. Clock synchronization

Best: record a shared hardware timestamp. Otherwise use a visible common event,
such as an LED flash or a sharp jerk independently timestamped by both streams:

```powershell
python .\scripts\calibrate_swivel.py --config .\config.yaml sync-event --caster-event-s 3.417 --car-event-s 3.451 --report .\out\calibration\sync.json
```

For cross-correlation, prepare two CSVs containing the **same physical signal**:

```csv
t,signal
0.000,0.02
0.010,0.03
```

Then run:

```powershell
python .\scripts\calibrate_swivel.py --config .\config.yaml sync-signals --caster-csv .\data\sync\caster_signal.csv --car-csv .\data\sync\car_signal.csv --max-offset-s 1.0 --report .\out\calibration\sync.json
```

Do not correlate a demanded car heading with the caster's mechanical response:
that would mix real swivel lag into the clock offset. Verify the remaining
error is below roughly one video frame. The sync commands set
`assumptions.car_and_caster_clocks_synchronized: true` and store the offset
added to caster timestamps.

That offset is acquisition-specific unless both systems share a persistent
hardware clock. If logs/cameras are restarted for each maneuver, synchronize
each clip and use a per-run config (or a manifest `caster_time_offset_s`) rather
than reusing one event offset for every ball/swivel run.

## Ball-caster calibration

The ball and swivel use the same camera intrinsics and clock convention when
the camera rig is unchanged.

### 1. Fixed projected circle

Automatic silhouette fit:

```powershell
python .\scripts\calibrate_ball_circle.py --config .\config.yaml --clip .\data\ball\circle_calibration.mp4 --preview .\out\calibration\ball_circle.png
```

The tool uses the undistorted first frame, so that frame must show the entire,
unobstructed ball silhouette.

For a noninteractive three-point fit, repeat `--point` exactly three times with
widely separated silhouette pixels:

```powershell
python .\scripts\calibrate_ball_circle.py --config .\config.yaml --clip .\data\ball\circle_calibration.mp4 --point 315 118 --point 203 302 --point 432 304 --preview .\out\calibration\ball_circle.png
```

Use `--manual` instead from a desktop session if you want to click the points.
Supplied/manual pixel points are measured in that displayed undistorted frame,
not the raw distorted image. The command writes `ball.circle`. The ball center
must remain fixed in the image.

### 2. Real HSV ranges

From a normal Windows desktop session:

```powershell
python .\scripts\inspect_ball_hsv.py --config .\config.yaml --clip .\data\ball\clip.mp4 --write
```

Press `T`, then sample several top marks; press `B` and sample bottom marks;
press `Y` and sample the dark yoke; press Enter to finish. This tool also uses
the undistorted first frame, which must clearly expose all three sample types.
The command writes `ball.segment`. Inspect the later tracking overlay for yoke
leakage or swapped hemisphere colors.

### 3. Ball axes

Record two clips starting from the same physically marked home pose: one pure,
monotonic roll and one pure, monotonic swivel/spin. Do not combine motions.

```powershell
python .\scripts\calibrate_ball_axes.py --config .\config.yaml --roll .\data\ball\pure_roll.mp4 --swivel .\data\ball\pure_swivel.mp4 --report .\out\calibration\ball_axes.json
```

The command writes `ball.R_bc` (ball axes to camera axes) and derives/writes
`ball.R_ball_to_car = R_car_from_camera @ R_bc` from the calibrated camera
extrinsics. This second matrix is required to express ball rolling in the same
car frame as the swivel caster; identity is correct only when those bases truly
coincide. If either clip moved in the negative right-hand-rule direction, add
`--roll-sign -1` or `--swivel-sign -1`.

### 4. Ball rolling-direction sign

Record a straight, known-forward ball run after the two axis transforms above
are calibrated. The first command intentionally uses tracking-only mode because
the radius and sign are not trusted yet:

```powershell
python .\scripts\run_ball.py --config .\config.yaml --clip .\data\ball\known_forward.mp4 --tracking-only --no-overlay --output .\out\ball_direction
python .\scripts\calibrate_ball_direction.py --config .\config.yaml --caster-frames .\out\ball_direction\caster_frames.json --report .\out\calibration\ball_direction.json
```

The calibration flips `ball.roll_direction_sign` if necessary and sets
`direction_sign_calibrated: true`, so `normalize(z cross roll_axis_car)` points
toward car `+x` throughout the known-forward interval. Do not assume the example
sign is correct for your camera/axis convention.

### 5. Loaded ball effective radius

Use a known straight distance under the intended payload and count effective
ball revolutions/angle from the calibrated ball output. The continuous rolling
angle for a straight pure-roll trial is the Approach-A `alpha_rad` change; do
not use either `beta` spin angle. Run:

```powershell
python .\scripts\calibrate_ball_radius.py --config .\config.yaml --distance-m 2.000 --revolutions 4.15 --report .\out\calibration\ball_radius.json
```

This writes `ball.r_eff_m`. Do not substitute the unloaded nominal radius if
you want meaningful slip. If you read the unwrapped alpha change directly,
use `--angle-rad <ABSOLUTE_ALPHA_CHANGE>` instead of `--revolutions`.

## Car-track CSV

The minimum columns are:

```csv
t,x,y,theta
0.000,0.000,0.000,0.000
0.010,0.001,0.000,0.000
```

Aliases such as `timestamp`, `x_world`, `y_world`, and `theta_car` are accepted.
Angles default to radians; set `car_track.theta_unit: deg` for degrees.
Optional `vx_world`, `vy_world`, and `omega` columns are used directly when all
rows contain them. Otherwise the loader unwraps/smooths pose and differentiates
it with the configured Savitzky-Golay window.
Optional velocities are SI regardless of `theta_unit`: `vx_world`/`vy_world`
are metres/second and `omega` is always radians/second, even if `theta` is in
degrees.

Set `car_track.r_arm_car_m` to the car-frame location of the caster mount
reference used for the swap: the swivel-axis reference for the standard caster,
or the ball/contact mount reference for the ball caster. For the swivel caster,
the adapter additionally supplies the dynamic trail/hub offset and its relative
velocity, so simultaneous chassis yaw and fork swivel are transferred to the
actual wheel point.

## Run real clips

The reviewed printed-wheel setup has an additional `config.improved.yaml`
profile. It uses calibrated tags 0 and 1, excludes their carriers, restricts
features to the white printed face, predicts KLT motion, and checks radial
consistency, pixel reprojection and feature spread. The original configuration
is preserved with the benchmark in `out/detection_upgrade/baseline_source/`.
The marker transforms and carrier footprints belong to this physical setup;
recalibrate after moving either marker. Generic example masks are disabled
because other wheels need different appearance settings.

```powershell
python .\scripts\run_swivel.py --config .\config.improved.yaml --clip .\data\swivel\clip.mp4 --tracking-only --output .\out\swivel_improved
```

Calibrate multiple tags with a stationary zero clip and a slow sweep where the
tags overlap in at least eight frames. The command fits full relative rotations,
rejects inconsistent observations, and writes a separate configuration/report.

```powershell
python .\scripts\calibrate_swivel_tags.py --config .\config.yaml --clip .\data\swivel\sweep.mp4 --zero-clip .\data\swivel\psi_zero.mp4 --ids 0 1 --output-config .\config.multitag.yaml --report .\out\calibration\multitag.json
```

For manual wheel initialization, first preview the moving projected face:

```powershell
python .\scripts\calibrate_wheel_face.py --config .\config.improved.yaml preview --clip .\data\swivel\clip.mp4 --output .\out\calibration\wheel_preview.mp4
```

Annotate several frames with visible faces at different swivel angles. Frame
numbers below are examples; choose sharp frames with accepted tag poses in your
clip. In the desktop window press `C` and click the face center, then `R` and
click at least five well-spaced points along the flat marked-face perimeter.
Press Enter to accept or Escape to cancel without replacing the annotation
file. Coordinates are stored in the undistorted image. Do not fit the rounded
tread silhouette or assume the bolt's front face is coplanar with the markings.

```powershell
python .\scripts\calibrate_wheel_face.py --config .\config.improved.yaml annotate --clip .\data\swivel\clip.mp4 --frames 62 124 249 374 --output .\out\calibration\wheel_points.json
python .\scripts\calibrate_wheel_face.py --config .\config.improved.yaml fit --annotations .\out\calibration\wheel_points.json --output-config .\config.refined.yaml --report .\out\calibration\wheel_fit.json
```

The fit refines the shared hub offset while preserving the measured radius,
width, camera and axle direction. `--fit-radius`/`--fit-width` require at least
three poses spanning 30 degrees; width additionally needs both faces. Fits
with excessive residuals, insufficient rank, or bound-limited adjustments are
rejected. Inspect a full-sweep preview with the refined config before using it.
This tool updates the moving 3D geometry; it does not freeze one image circle.

To review a carrier exclusion, select a frame where that tag is clear and click
around the tag, its white border, and the carrier. The polygon is converted to
that marker's coordinates and follows it through swivel. Repeat for each tag;
their printed corner orientations and carrier directions can differ. A planar
carrier polygon cannot exactly represent a raised or curved fork, so retain the
material mask or add measured `fork_polygons_m` for those parts.

```powershell
python .\scripts\calibrate_wheel_face.py --config .\config.improved.yaml mask --clip .\data\swivel\clip.mp4 --frame 62 --marker-id 0 --output-config .\config.masked.yaml
```

The tracking overlay now shows the exact usable mask, green inlier rings, red
rejected matches, tag IDs, failure reasons, and whether accumulated roll remains
complete. `interval_quality[i].roll_valid` means the individual increment was
measured; `phi_valid[i]` means the accumulated angle remains connected to frame
zero. After an unrepaired gap, later increments may be valid while absolute
roll stays incomplete. `phi_segment_id` marks separated tracking segments.
The numeric incomplete angle is the sum of observed motion, not a reconstruction
of unseen turns. Common velocities use measured increments directly so a
reference correction cannot create a spurious speed spike.

Short reference-dot repairs propagate through subsequent integration and are
limited by both gap duration and an explicit angular-speed bound. Long unseen
turn counts remain unknown. The benchmark keeps reference correction disabled
so the green mark can be used as a separate visual cross-check. It is not
external ground truth and shares calibration with the tracker.

Compare saved runs on the same frame intervals with:

```powershell
python .\scripts\compare_swivel_runs.py --before .\out\detection_upgrade\baseline --after .\out\detection_upgrade\improved_final --config .\config.improved.yaml --output .\out\detection_upgrade\comparison
```

The comparison writes coverage, paired green-reference errors, raw increment
series, and a plot. The sweep used for tag calibration is not a held-out clip;
the separate roll and combined clips provide additional checks. Accepted
coverage alone cannot establish angle accuracy. A run is only marked trusted
when calibration prerequisites and realized coverage/completeness checks pass.

Tracking-only commands are useful for setup and diagnostics. Their common
velocities are not calibrated and must not be used in the comparison.

```powershell
python .\scripts\run_swivel.py --config .\config.yaml --clip .\data\swivel\clip.mp4 --tracking-only --output .\out\swivel_tracking
python .\scripts\run_ball.py --config .\config.yaml --clip .\data\ball\clip.mp4 --tracking-only --output .\out\ball_tracking
```

After all calibration and synchronization steps, run trusted measurements:

```powershell
python .\scripts\run_swivel.py --config .\config.yaml --clip .\data\swivel\clip.mp4 --car-track .\data\car_tracks\swivel_track.csv --output .\out\swivel --loop-closure
python .\scripts\run_ball.py --config .\config.yaml --clip .\data\ball\clip.mp4 --car-track .\data\car_tracks\ball_track.csv --output .\out\ball
```

The swivel runner refuses normal measurement unless `K`, `T_car_from_cam`,
measured geometry, direction sign, `psi0`, and `r_eff` are calibrated, and it
refuses shared metrics when clock synchronization is not confirmed.
`run_swivel --tracking-only` may use approximate `K` and `psi0=0`, but it still
needs camera extrinsics and usable caster geometry for ray-plane lifting. If
`r_eff` is absent it writes the raw `phi`/`psi` track but cannot emit calibrated
common velocities. The ball runner's tracking-only mode can use identity axis
transforms and an `r_eff=1` placeholder. A trusted ball run requires `R_bc`,
`R_ball_to_car`, calibrated direction sign, and `r_eff`. Both retain quality and
dropout masks in `caster_frames.json` when common frames can be emitted.

Use a different output directory for every maneuver and repeat. The runners use
fixed artifact names inside that directory and will overwrite a prior run there.

Important swivel outputs include:

- `swivel_track.csv`: `t`, `phi`, `psi`, and validity flags;
- `caster_frames.json`: the device-independent seam;
- `tracking_overlay.mp4`: tag, mask, and quality diagnostics;
- `metrics/`: common plots/series/summary when a car track is supplied; and
- `results.json`: calibration metadata, raw tracks, quality, reference phase,
  self-consistency, loop closure, and metrics.

The loop-closure report gives modulo-360 closure error separately from net
unwrapped motion/turn count. A valid 360-degree wheel turn is therefore not
mistaken for 360 degrees of drift.

Important ball outputs are analogous:

- `results.csv`: integrated Approach-A angles, velocities, and quality; plus
  `increments.csv` for interval rotation matrices and explicit validity;
- `caster_frames.json`: the same device-independent seam;
- `tracking_overlay.mp4`: circle, masks, tracks, and quality diagnostics;
- `metrics/`: shared plots/series/summary when a car track is supplied; and
- `results.json`: calibration/trust metadata, raw rotations, diagnostics, and
  metrics.

For the ball adapter, `omega_spin` is the mean car-vertical angular component
over the valid hemispheres. The separate top/bottom vertical components remain
in `raw` as `beta1_dot` and `beta2_dot`; primary comparison metrics do not use
`omega_spin`.

## Paired comparison

Copy the manifest and edit each row to point at a ball/swivel pair captured on
the same floor with the same payload, camera mount, maneuver, and matched speed
profile:

```powershell
Copy-Item .\compare_manifest.example.yaml .\compare_manifest.yaml
notepad .\compare_manifest.yaml
python .\compare\run_comparison.py --manifest .\compare_manifest.yaml --output .\out\comparison
```

The example sets `require_pairs: true`, so the harness rejects a maneuver that
lacks either a ball or swivel row. Set it false only for deliberate single-run
diagnostics, not for a claimed head-to-head result.

Relative paths in the manifest, including each row's `config`, are resolved
relative to the manifest file rather than the shell's current directory.

The preferred reproducible row supplies precomputed common records:

```yaml
- id: straight-ball
  maneuver: straight
  device: ball
  caster_frames: out/ball/straight/caster_frames.json
  car_track: data/car_tracks/straight_ball.csv
  config: config.yaml
```

The config (or an explicit `r_arm_car_m: [x, y]` on the row/top level) is still
needed for the caster mount location. Without it, a precomputed ball row would
default to `[0, 0]` and compute the wrong contact velocity during chassis yaw.

A raw row instead supplies `clip` and a calibrated config:

```yaml
- id: straight-swivel
  maneuver: straight
  device: swivel
  clip: data/swivel/straight.mp4
  car_track: data/car_tracks/straight_swivel.csv
  config: config.yaml
```

Raw rows require explicitly confirmed clock synchronization and complete
device/camera calibration. Each row may add `caster_time_offset_s` only for a
small manifest-specific correction; the calibrated project offset is already
applied by the raw pipeline.

For precomputed rows, the harness rejects metadata explicitly marked
tracking-only, uncalibrated, approximate-camera, or unsynchronized. A stream
from an older exporter with no trust metadata is accepted under the
`CasterFrame` contract, so manually verify that its times already share the
car-track clock. If you aligned such a stream externally, record the assertion
as `clocks_synchronized: true` on that manifest row.

The report writes:

- `summary.csv`, `summary.json`, and `summary.md`;
- one shared-axis overlay plot per maneuver in `plots/`; and
- full series and summaries per run in `runs/<run-id>/`.

Use this standardized pair set:

| Maneuver | Primary observations |
| --- | --- |
| Straight line | slip and longitudinal efficiency baseline |
| Step turn | positive response lag, settling time, and initial-flip scrub |
| Increasing-speed slalom | alignment and shimmy onset |
| Spin in place | pure/near-pure lateral scrub |
| Figure 8 | sustained alignment and cumulative scrub |

Repeat runs and report uncertainty; do not attribute every difference to caster
geometry if floor, payload, tire condition, battery state, command profile, or
camera placement also changed.

## Metric interpretation

- Alignment is the signed angle from measured rolling heading to car contact
  velocity; summaries report mean absolute error.
- Scrub speed is lateral contact speed. Total scrub is its gap-safe time
  integral, and scrub fraction divides by contact-path length.
- Slip compares longitudinal contact speed with `omega_roll * r_eff`; roll
  dropout and near-zero longitudinal speed are masked.
- Positive lag means caster response occurs after demanded heading motion.
- Settling requires alignment to stay within the configured threshold/dwell.
- Shimmy is the dominant Welch-PSD peak only inside the configured frequency
  band and steady segment.
- Rolling efficiency is `integral(abs(v_long)) / integral(abs(v_contact))`.
  It is reported independently from scrub fraction. It is generally **not**
  `1 - scrub_fraction` because longitudinal and lateral magnitudes do not add
  linearly.

All integrals are gap-safe: a masked endpoint or acquisition gap prevents the
interval from silently contributing. Metrics consume only `CasterFrame` plus
the car track; `alpha/beta` and `phi/psi` stay in `raw` for auditability.

## Real-data acceptance checks

There is no encoder ground truth, so inspect self-consistency before believing
a comparison:

- tag detection is continuous and reprojection error is preferably below 2 px;
- roll inlier ratio is above about 0.7 and off-axis residual below about 2 deg;
- when the sidewall is well presented, the independent ellipse-corrected
  image-plane circular estimate agrees with primary ray-plane/Kabsch roll
  (`image_plane_circular_disagreement_deg` stays within a few degrees);
- `phi` is monotonic during a steady one-direction roll;
- the axis-derived rolling heading agrees with the tag-derived wheel heading;
- calibrated straight-run slip is near zero without an unexplained constant
  offset;
- roll invalidity is concentrated in the predicted edge-on band;
- a physically returned marked pose has only a few degrees modulo-360 closure
  error; and
- residual clock error is below approximately one frame.

Passing synthetic tests proves the implementation against its stated model. It
does not compensate for wrong measurements, rolling shutter, lens settings that
changed after calibration, glare, an unstable camera/ball center, or long hidden
wheel rotations.

## Angular velocity and measured wheel replay

`run_swivel.py` also writes `angular_motion.csv` and `results.json.angular_intervals`
in tracking-only mode. Each row describes frame `i` to `i+1`: timestamps, signed
roll/swivel changes (`delta_phi`, `delta_psi`), interval-average angular velocities
(`phi_dot`, `psi_dot`) in rad/s and deg/s, and separate validity flags. Missing
measurements are blank in CSV and null in JSON. An effective rolling radius is
needed for linear speed, not angular speed. Signs follow the geometry's phi/psi
conventions; forward/reverse interpretation still requires sign calibration.

Rates use the actual fitted roll increment, not differences across corrections
to accumulated roll. No smoothing or gap interpolation is applied. Current
timestamps use frame index / reported FPS, so variable frame timing and dropped
camera frames remain timing limitations. Differentiation amplifies angle noise.

To export existing results and make a video-aligned kinematic replay:

```powershell
.\.venv\Scripts\python.exe scripts/simulate_measured_swivel.py `
  --results out/detection_upgrade/improved_final/clip/results.json `
  --output out/measured_replay/clip
```

Outputs are `angular_motion.csv`, `angular_motion.png`, `measured_replay.mp4`,
`replay_preview.jpg`, and `replay_report.json`. Use `--export-only` for angular
outputs without rendering. Calibration comes from the results snapshot, so
editing config later does not silently change the reconstructed geometry.

The left panel overlays the projected wheel on undistorted footage; the right
panel renders a schematic wheel and fork driven by the measured angles. Hub
trail, axle direction, wheel width, and visible face all change with swivel;
a 180-degree swivel therefore moves the wheel circle to its opposite location.
Magenta probes are anchored to detected wheel features and propagated by the
measured rigid motion without subsequent feature redetection within a segment.
Sliding probes expose visual inconsistency. Occlusion masks hide probes where
the configured visible-wheel mask rejects them.

Roll spokes have an arbitrary zero within each continuous segment. Missing
heading hides the model; roll loss, edge-on views, and face changes reset the
phase and probes with a RE-ANCHOR label. This replay never recovers hidden turn
counts. The fork is a schematic centerline, not a fitted CAD silhouette. The
tool checks visual consistency; it does not establish measurement accuracy or
simulate contact forces, friction, slip, or dynamics.

To improve the projected circle, use `calibrate_wheel_face.py annotate` and
`fit` on several visible poses, including opposite swivel directions, followed
by `preview` (see the real-footage calibration workflow above). A single circle
at the starting pose constrains that pose; multi-pose fitting is needed to check
trail, axle, width, and camera/geometry consistency throughout a swivel. Rerun
tracking with the fitted config before replaying those new results.

## Repository layout

```text
common/       camera, tracking, rotations, car kinematics, CasterFrame
ball/         Approach-A sphere pipeline and common adapter
swivel/       tag, dynamic geometry, roll, reference dot, adapter, pipeline
metrics/      device-independent metrics and plots
synthetic/    deterministic renderers and S-A..S-E validation
compare/      paired manifest runner and report generation
scripts/      calibration, real-run, marker, and validation entry points
tests/        numeric, adapter, CLI, calibration, and report regression tests
data/         prepared input locations and format notes
```

## Troubleshooting

| Symptom | Action |
| --- | --- |
| ArUco is intermittent | Measure print scale, restore quiet border, remove glare/occlusion, increase marker pixels, and recalibrate `psi0`. |
| Too few sidewall tracks | Add irregular matte texture, improve face visibility/light, shorten exposure, or tune mask/corner settings. |
| Roll fails only near one swivel angle | That face is edge-on/occluded; mark both faces and change the oblique viewpoint. |
| Reference dot is absent | Tune its HSV/area gates from real images; do not duplicate the bold dot around a face. |
| High off-axis residual | Recheck camera extrinsics, hub/trail/axle geometry, tag zero, mask leakage, and mixed fork/wheel motion handling. |
| Alignment/scrub look shifted in time | Recalibrate clocks with a genuinely common event; do not tune mechanical lag away as timestamp offset. |
| Constant straight-run slip | Recalibrate loaded effective radius and roll sign. |
| Ball top/bottom disagree | Recheck `ball.R_bc`, circle stability, HSV separation, yoke mask, and pure-motion calibration clips. |
| Overlay looks good but real metrics are noisy | Inspect car-track differentiation/smoothing, timestamp residual, rolling shutter, and physical mount compliance. |
