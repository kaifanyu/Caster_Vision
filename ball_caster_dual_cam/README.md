# Dual-camera ball-caster calibration

This project records a Logitech C920 and Brio 101 with repeatable Linux V4L2
settings, calibrates their relative pose, and fits **one mechanical axis frame,
one shared roll, and two independent shell spins** to both camera images.
It uses the separated-hemisphere construction: two shell curvature centers are
separated by the physical rim gap and move with roll. Camera 1 is the C920.

The original `../ball_caster_rot` project is unchanged. Reused segmentation,
tracking and rotation initialization modules are documented in `VENDORED.md`.

**The supplied Kalibr intrinsics and stereo transform are imported.** Axis
calibration and measured camera timing are still pending. The supplied 100 mm
shell radius and 20 mm gap come from the original project's configuration;
verify these physical dimensions on your rig. See the
[current calibration results and next steps](docs/KALIBR_NEXT_STEPS.md).
For the shared-light timing measurement, new 12-second capture sequence and
stationary-home solver option, follow [timing and home calibration](docs/TIMING_AND_HOME.md).

## Files you edit

| File | Values |
|---|---|
| `config/rig.yaml` | Device paths, resolution/FPS, manual camera settings, per-camera color masks, circles, timing offset, measured radius/gap |
| `calibration/c920.yaml` | C920 `K`, `dist`, `image_size`, and recorded `capture_profile` |
| `calibration/brio101.yaml` | Brio 101 `K`, `dist`, `image_size`, and recorded `capture_profile` |
| `calibration/stereo.yaml` | Generated camera-to-camera `R_21`, `t_21_m`; translation in **meters** |
| `calibration/axes.yaml` | Generated `R_bc` and assembly pivot in C920 coordinates; created only when axis calibration passes |

All paths **inside `rig.yaml` are relative to that YAML file's directory**.
For example `../calibration/c920.yaml` resolves correctly when scripts are called
from another directory. Explicit CLI output paths are relative to the shell's
working directory. Examples below run from this project:

```bash
cd /home/hamr/Caster_Vision/ball_caster_dual_cam
python3 -m pytest -q
```

Python packages are listed in `requirements.txt`; `v4l2-ctl` comes from Linux
`v4l-utils`. The implementation is tested with the available NumPy 1.26, SciPy
1.11, OpenCV 4.6, PyYAML 6, and pytest 7 environment. To isolate dependencies,
create a virtual environment and install `requirements.txt`. No ROS node is
required. GUI preview commands require a graphical desktop.

## Recording during ROS hardware trajectories

`hamr_bringup/hamr_HW.launch.xml` now starts both cameras by default using this
checkout's `config/rig.yaml`. Wait for `Recording NOW`, then run the separate
`ros2 run reference_trajectory waypoint_traj_simple` command as usual. Ctrl+C on
the hardware launch finalizes both recordings. Videos, receive-timestamp CSVs and
`session.json` are saved in a new `~/hamster_ws/recordings/dual_cam_*` directory.
Long recordings use numbered AVI parts without restarting camera capture; keep
all parts with the manifest. The pipeline's video reader follows the parts.

See [the ROS recording guide](../../hamster_ws/src/hamr_control/hamr_bringup/CAMERA_RECORDING.md)
for build instructions, camera-only checks, output details and launch overrides.
The integration records motion for offline analysis; it does not start the
trajectory publisher or apply a live caster estimate to robot control.

## 1. Fix the cameras' capture settings

Read [the complete camera guide](docs/CAMERA_SETUP.md) for lighting, focus trials,
device controls, and synchronization limitations.

The saved profiles use **1920×1080, MJPG, 30 fps, fixed gain 180 and 4000 K white
balance**. Read the current `exposure_us` values from `config/rig.yaml`; the
historical readiness report used 7700 µs on C920 and 8000 µs on Brio 101, and
the C920 setting has since been adjusted. C920 focus is locked at **50**, zoom at **100**.
See [the readiness report](docs/CAMERA_READINESS.md) for measurements and remaining
lighting/calibration requirements. Increase illumination before shortening exposure
for faster motion. The circles and color thresholds are specific to this mounting.

* **C920:** set `cameras.c920.capture.focus` to a tested manual position. The setup
  tool disables autofocus before applying it and holds `zoom: 100`.
* **Brio 101:** fixed-focus lens. There is no autofocus to disable. Change mounting
  distance until stationary markings are sharp at native resolution. Do not add
  a `focus` entry. [Logitech specification](https://support.logi.com/hc/en-us/articles/16131499767191-Specification-Brio-101)

Inspect first (no device settings change):

```bash
python3 scripts/camera_setup.py --inspect
```

An explicit C920 trial, using a value within the advertised focus range:

```bash
python3 scripts/camera_setup.py --camera c920 --apply --focus 50 --preview
```

`50` is the retained lens position for this mounted rig, not a universal focus
value. Save any newly chosen value in `rig.yaml` and recheck intrinsics.
Change `exposure_us`, gain, and white balance in that file while checking previews:

```bash
python3 scripts/camera_setup.py --camera brio101 --apply --preview
python3 scripts/camera_setup.py --apply --output out/camera_profile.json
```

Both recording and setup verify supported modes, set manual exposure and white
balance, disable dynamic frame rate where available, and read controls back.
Unsupported or ignored requested settings produce an error instead of a claim of
success. The recorder reapplies the profile **after streaming starts**, drains
warmup frames and checks controls before either recording clock starts. It saves
the cameras' original MJPEG frames without JPEG re-encoding and uses four buffers
to avoid the observed Brio frame-rate loss with a single buffer.
Offline AVI reading prefers OpenCV's native MJPEG decoder to tolerate the Brio's
short APP0 metadata block, which older FFmpeg versions report as `unable to decode
APP fields`. Original recordings, frame order, and timestamp CSVs are preserved;
other codecs or builds without that reader fall back to the default backend.
Keep resolution, image orientation, crop, focus, zoom, illumination, and mounts
consistent. Keep physical red/green identities even when a view is upside down.

## 2. Enter or measure intrinsics

The supplied C920 and Brio intrinsics are already in their calibration files.
Their original capture settings are unknown (`capture_profile: null`), so matching
those intrinsics to the mounted rig still requires a checkerboard check. For a new
calibration, replace the values in each camera's calibration file. `K` is a 3×3 OpenCV matrix; `dist` is a measured
4/5/8/12/14-element distortion vector; `image_size` is `[width,height]`:

```yaml
image_size: [1920, 1080]
K: null     # replace with [[fx,0,cx],[0,fy,cy],[0,0,1]] using YOUR numbers
dist: null  # replace with YOUR distortion coefficients
capture_profile: null
```

For imported calibration, also copy the exact corresponding `capture` mapping
from the rig into `capture_profile` **only if those settings were actually used**.
Absent profile metadata is allowed but reported as unverified. Partial/empty or
mismatched supplied profiles are rejected. Do not claim zero distortion unless
the supplied calibration/image coordinate model supports it.

To measure intrinsics and stereo, record a checkerboard held **stationary for each
view**, with at least 10 sharp, varied positions and tilts seen by both cameras.
Keep the board near the caster's working volume, fill different image regions,
and include tilted views rather than only translating a front-facing board.
The example uses **9×6 INNER corners** and **measured 25 mm squares**; substitute
your actual printed board dimensions. The printed square size sets metric scale.

```bash
python3 scripts/record.py --mode checkerboard --duration 50 --output data/board_session
python3 scripts/extract_pairs.py --session data/board_session --every-seconds 2 --output data/board_images
```

Review both folders and remove blurred/moving-board pairs from **both** folders.
The extractor stores raw unrotated PNGs, matching filenames, and the recorded
capture profiles. Intrinsic tools copy those profiles into their output.

```bash
python3 scripts/calibrate_intrinsics.py --images data/board_images/c920 --cols 9 --rows 6 --square-m 0.025 --output calibration/c920.yaml
python3 scripts/calibrate_intrinsics.py --images data/board_images/brio101 --cols 9 --rows 6 --square-m 0.025 --output calibration/brio101.yaml
```

The tools require at least 10 usable images and RMS ≤1 pixel by default, check
image dimensions, and write diagnostic reports. Those gates assess fit quality;
they do not by themselves establish angular or metric accuracy.

## 3. Calibrate camera-to-camera geometry and timing

```bash
python3 scripts/calibrate_stereo.py --left-images data/board_images/c920 --right-images data/board_images/brio101 --cols 9 --rows 6 --square-m 0.025
```

Intrinsics are fixed during stereo calibration. The output convention is:

```text
X_brio101 = R_21 @ X_c920 + t_21_m
```

The tool resolves 180° checkerboard corner-order differences by consistency of
the relative camera pose across varied board tilts. It rejects ambiguous or
inconsistent data. Do not rotate one camera's images to visually match the other.
Stereo outputs include hashes of both intrinsic calibration files, preventing
accidental reuse after intrinsics change. Provenance checks for stereo, axes and
rendered results tolerate LF/CRLF line-ending conversion when moving between
Linux and Windows (including Git checkouts); other file changes still invalidate
the saved hashes. The underlying fixed-intrinsics method
is OpenCV `stereoCalibrate`. [OpenCV reference](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)

The supplied Kalibr calibration is now imported in `calibration/c920.yaml`
(`cam0`), `calibration/brio101.yaml` (`cam1`), and `calibration/stereo.yaml`.
The original numerical result is preserved in `calibration/kalibr_camchain.yaml`.
Kalibr's `cam1.T_cn_cnm1` maps cam0 coordinates to cam1 coordinates, matching
`R_21`/`t_21_m` directly. [Kalibr convention](https://github.com/ethz-asl/kalibr/wiki/yaml-formats)
With the mounts unchanged, proceed to axis calibration; there is no need to run
the checkerboard stereo command again. Keep the same resolution, focus and zoom.

**Timing is a separate measurement.** Both webcams use independent capture
threads and one host monotonic clock, but timestamps record frame-read completion,
not guaranteed exposure time. Record a repeated common flash or another sharp
event to measure relative latency and check drift. If Brio receives an event
30 ms later, set `timing.brio_offset_s: -0.030`. Set `timing.verified: true` only
after measuring it. The default is unverified and reported that way.

The tracker pairs each C920 frame to at most one Brio frame within
`max_pair_skew_ms`; it never duplicates a frame to fill a gap. This tolerance
bounds the **corrected host timestamps**, not unknown sensor/USB latency. These
webcams cannot provide hardware-triggered synchronization through this recorder.
Use moderate calibration speeds; fast motion/rolling shutter remains a limitation.
Feature tracking follows every native image between the first and last selected
pair, including unpaired images, to preserve track identities across pairing gaps.
Only observations at the selected pairs enter the joint geometric fit. A track
lost in an intermediate image gets a new identity; gaps are not interpolated.

## 4. Record pure roll and pure swivel; inspect tracking masks

Both clips must start at the **same known mechanical home roll**. Keep the caster
at home during startup. Wait for `Recording NOW`, hold home for three seconds,
then perform one isolated, monotonic
motion through a useful range, preferably 20–60 degrees or more with clear marks:

* `roll`: change the shared roll, holding each shell's spin fixed.
* `swivel`: hold roll at home; spin one or both shells about the common swivel
  axis. If both spin, use the same signed direction for the PCA initializer;
  their speeds can differ.

```bash
python3 scripts/record.py --mode roll --duration 10 --output data/roll_01
python3 scripts/record.py --mode swivel --duration 10 --output data/swivel_01
python3 scripts/preview.py --session data/roll_01 --camera c920 --frame 0 --output out/c920_masks.png
python3 scripts/preview.py --session data/roll_01 --camera brio101 --frame 0 --output out/brio_masks.png
```

Inspect masks on both colors at several poses. Tune each camera's `segment`
section; exclude the yoke, inner discs, rims and background. An enclosing circle
is used for the mask and **coarse initialization only**, not the final separated
shell model. Set `cameras.<name>.circle: [u0,v0,r_px]` in undistorted pixels if the
automatic seed is wrong. `preview.py --pick-circle` lets you choose three outer
silhouette points. Press Enter to accept: after saving the diagnostic preview,
the script automatically updates that camera's circle in `config/rig.yaml` (or
the file supplied with `--config`). `--circle U V R` also saves the supplied
circle. Use `--no-save` for a preview-only trial; Escape cancels the picker without
changing the config. Comments, relative paths, and the other camera's settings
are preserved. The PNG and companion YAML are still saved beside the preview.

```bash
python3 scripts/preview.py --session data/roll_01 --camera c920 --frame 0 --pick-circle --output out/c920_circle.png
python3 scripts/preview.py --session data/roll_01 --camera brio101 --frame 0 --pick-circle --output out/brio_circle.png
```

Use one saved circle per camera for a fixed mounting, then check it on several
frames from both roll and swivel clips without `--pick-circle`. Normal previews
do not change the config. Each camera also has its own red/green HSV thresholds
in `cameras.<name>.segment`. They are reused across both clip types; edit them
manually in `rig.yaml` only if the masks need adjustment, then rerun the previews.
Circle selection does not tune or overwrite the HSV thresholds.

## 5. Jointly calibrate axes, then measure motion

```bash
python3 scripts/calibrate_axes.py --roll data/roll_01 --swivel data/swivel_01 --max-frames 300 --output out/axes_01
```

`--roll-sign -1` / `--swivel-sign -1` describe a calibration recording made in the
negative mechanical direction. `geometry.red_shell_sign` defines whether the red
shell lies in local +z or -z. Confirm these physical conventions; image order does
not determine them. Optional `R_bc_seed` / `pivot_c920_m` can provide measured
initial guesses, but a successful image fit is still required.

The tool saves `report.json`, `roll_tracks.npz`, and `swivel_tracks.npz`. It writes
`calibration/axes.yaml` only if the fit converges and passes motion-excitation,
axis-rank, visibility, surface-spread, residual, and support checks. Rejected fits
return exit code 2 and preserve an earlier accepted axes file.

For an anchored fit that stalls, optional `--surface-refinement-passes 2` fits
each surface track independently before each of two joint solves. Use
`--max-nfev 250` to set the budget explicitly for each joint solve. The report
records stage costs, convergence and total joint evaluations. This keeps the
original quality gates and observations; it does not guarantee acceptance.
See [the staged-refinement and HSV guide](docs/TIMING_AND_HOME.md#4-refine-surface-points-before-joint-calibration)
for a complete command and mask-inspection instructions.

After a successful axis fit, replay the pure-roll recording itself with all three
angles free to check for unwanted estimated shell spin:

```bash
python3 scripts/run.py --session data/roll_01 --allow-calibration-clip --initial-roll-deg 0 --max-frames 300 --output out/roll_check_01
python3 scripts/render.py --results out/roll_check_01/results.json --output out/roll_check_01/overlays
```

Use a new output directory each run. This opt-in leaves the recorded session mode
unchanged and applies the general-motion model: it does **not** force the two spins
to zero. In a pure-roll clip, supported `alpha_deg` should change while supported
`beta_red_deg` and `beta_green_deg` stay near their starting zero. Review validity
flags, residuals and overlays as well as the curves. There is no measured accuracy
tolerance yet; reusing the calibration clip checks consistency, not accuracy on
new data. A separate roll check and mixed-motion recording provide a stronger
test. A pure-roll clip alone does not calibrate both axes; the swivel clip is
still required by `calibrate_axes.py`.

Record a general-motion session and supply its **known initial roll relative to
calibrated home**; use zero only when physically at home:

```bash
python3 scripts/record.py --mode motion --duration 6 --output data/motion_01
python3 scripts/run.py --session data/motion_01 --initial-roll-deg 0 --output out/motion_01
python3 scripts/render.py --results out/motion_01/results.json --output out/motion_01/overlays
```

Outputs are `results.csv`, `results.json`, and `tracks.npz`. CSV contains radians,
degrees, timestamps, and **separate validity flags for roll/red spin/green spin**.
Alpha is relative to mechanical home; each beta is relative to its own phase in
the first paired image. Missing estimates are blank/null, never held measurements.
Overlays draw the calibrated roll and moving swivel axes on both views; missing
roll is labeled unresolved. Playback uses median FPS; CSV times remain authoritative.

For arbitrary-motion tracking with native frames, keyframe recovery and one
shared two-camera Kalman trajectory, see [FUSED_MOTION.md](docs/FUSED_MOTION.md).
The new `scripts/track_motion.py` supports recorded sessions or two MP4/AVI files
with their original timestamp CSVs; its approximate rotation source is reported
separately from acceptance by the metric batch solver.

Export that saved shared trajectory as a portable interactive 3D replay with
`scripts/visualize_motion.py --results out/run2_fused_03/results.json --output out/run2_fused_03/orientation_3d.html`.
Open the HTML to play, scrub, orbit, and zoom one caster frame and its independent
shell spins. No tracking rerun or extra packages are needed. Unresolved motion
remains hidden; the replay does not fill gaps or improve accuracy. See
[3D viewer usage](docs/FUSED_MOTION.md#interactive-3d-orientation).

The original `scripts/run.py` is an **offline batch estimator**, defaulting to the
first **180 paired frames per clip**. `--max-frames N` increases this explicitly;
computation and memory grow with tracks and frames. It does not implement the old
project's long-clip keyframe relocalization, overlapping-window recovery, or a ROS
live stream. A completely disconnected shell-track segment retains an unknown
spin phase. It is not silently joined to the previous segment.

## How the algorithm works

1. Undistort each camera with its own K/dist; retain its original pixel coordinate
   system. Independently segment the physical red and green shells.
2. Run persistent pyramidal Lucas–Kanade tracking with a forward/backward check.
   A feature identity is scoped to its **camera, shell and clip**. No cross-camera
   triangle identification or dense stereo disparity is necessary.
3. For initialization, lift image points to an approximate enclosing sphere and
   estimate incremental rotations. Convert Brio increments to C920 coordinates
   with `R_21.T @ delta_R_brio @ R_21`. Pool both cameras' pure-motion evidence to
   initialize roll/swivel axes. These sphere estimates are seeds, not final output.
4. Fit the actual separated-shell image model to both cameras' pixels. With
   `F=R_bc`, pivot `C`, shell radius `r`, rim gap `g`, local unit landmark `p`,
   and shell sign `s`, a surface point in C920 coordinates is

   ```text
   X = C + F @ Rx(alpha) @ (s*g/2*[0,0,1] + r*Rz(beta_shell) @ p)
   pixel_c920 = project(K_c920, X)
   pixel_brio = project(K_brio, R_21 @ X + t_21_m)
   ```

   Each landmark remains on its signed hemisphere. During calibration, one shared
   SO(3) axis-frame correction, one pivot, clip angles and landmark coordinates
   minimize robust reprojection errors in both views. Known home poses and pure
   motions fix the reference ambiguities. Intrinsics, stereo transform, physical
   radius and gap stay fixed. During normal processing the calibrated axes and
   pivot are fixed, and only shared roll, separate spins and landmarks are fitted.
5. Reject unsupported output using robust pixel residuals, front-surface visibility,
   surface coverage, and temporal track connectivity to the known reference.
   One camera can preserve a shell's reference when the other loses it. Shared
   roll alone cannot invent an unseen shell's independent spin.

## Validation and limits

```bash
python3 -m pytest -q
```

Tests exercise oblique-camera synthetic geometry, joint axes/pivot recovery from
biased initialization, independent spins, outliers, disconnected/unseen shells,
degenerate support, timestamp matching, calibration provenance, and fake-device
recording/control failures. Synthetic accuracy is not a claim about the physical
cameras. Validate repeatability, both views' residuals, accepted frame coverage,
and an independent known-angle reference on the actual rig.

The models assume rigid camera mounts relative to the caster pivot, fixed metric
shell geometry, and orthogonal roll/swivel axes. They do not model yoke geometry,
shell holes, flex, rolling-shutter timing, slip, or ground contact. Wrong masks,
dimensions, intrinsics, timing, or impure calibration motions can bias results
even when an optimizer converges. The second camera helps visibility and geometric
conditioning; it does not automatically remove these systematic errors.
