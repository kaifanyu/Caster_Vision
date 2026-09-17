# C920 and Brio 101 setup

Use one saved profile for checkerboard calibration, axis calibration, and motion
recording. Profiles live under `cameras.<name>.capture` in `config/rig.yaml`. All
paths inside the rig file are relative to the rig file. The scripts work from any
current directory; examples below run from the repository root.

## Physical setup and initial settings

The Brio 101 has **fixed focus**, with no autofocus to disable and no motorized
focus adjustment. Logitech advertises 1920×1080 and 1280×720 at 30 fps. Move it
farther from the caster if stationary triangle edges are soft, and check the
nearest and farthest visible shell regions. There is no reliable universal
minimum mounting distance for this particular rig; judge native-resolution
sharpness. [Logitech specifications](https://support.logi.com/hc/en-us/articles/16131499767191-Specification-Brio-101).

The C920 has adjustable focus. Choose one manual lens position at the mounted
distance and retain it for both intrinsics and subsequent recordings. Changing
focus can change the camera intrinsics. The current mounted rig retains manual
position `50`; the script rejects `focus: null` rather than guessing a position.
See [the measured readiness report](CAMERA_READINESS.md).

Start with bright, diffuse illumination, 1920×1080, MJPG and requested 30 fps,
4000 µs exposure (1/250 s), low gain, fixed white balance, and C920 zoom 100.
Use 2000 µs (1/500 s) if motion blur persists and lighting permits. These are
starting values, not measured optima. Tune the two cameras independently; equal
white-balance numbers do not guarantee equal red/green colors across models.
Tune segmentation per camera after locking the profiles. `power_line_frequency`
is 2 for 60 Hz and 1 for 50 Hz if exposed by the driver. Short manual exposures
can still show lighting flicker; steady illumination is preferable.

Use an angle where the Brio sees markings hidden by the C920's yoke. The cameras
must both see the calibration board for stereo calibration. Keep the caster,
mounts and cameras rigid after calibration. Do not rotate, mirror, crop, digitally
zoom or resize saved images between calibration and processing. An upside-down
view is handled by its camera geometry; shell identities remain red/green.

## Inspect without changing devices

```bash
python3 scripts/camera_setup.py --inspect --output camera_capabilities.json
```

This is read-only and attempts both cameras, reporting a missing camera without
hiding the working one's capabilities. Check `device` entries against
`/dev/v4l/by-id/`; use the `video-index0` capture node. Avoid `/dev/video0` and
`/dev/video2`, whose assignments can change. Also inspect using:

```bash
v4l2-ctl --list-devices
ls -l /dev/v4l/by-id/
```

Install `v4l-utils` if `v4l2-ctl` is unavailable. Close browser, ROS and other
camera programs before recording. Prefer separate USB host controllers if a
shared controller cannot sustain both streams. MJPG reduces input USB bandwidth;
the MJPG recorder saves original camera JPEG buffers directly into indexed AVI
without decoding/re-encoding. Four V4L2 buffers avoid the observed Brio half-rate
behavior with a single buffer. Its observed frame-rate report is the evidence of
achieved throughput. The classic AVI writer has a roughly 4 GiB limit per camera
per session; use shorter sessions. Exceeding it fails the session while retaining
the earlier frames in a finalized file. Other input formats use the decoded path.

## Choose C920 focus, then save it

Read the permitted `focus_absolute` range and step from inspection. The connected
C920 was observed to support 0–250 in steps of 5; recheck on the actual device.
On a graphical desktop, try explicitly selected values, for example:

```bash
python3 scripts/camera_setup.py --camera c920 --apply --focus 50 --preview
python3 scripts/camera_setup.py --camera c920 --apply --focus 55 --preview
```

Those numbers are trial positions, not recommended final calibration values.
Compare stationary triangle corners over both shells. Preview is bounded to 20 s
by default, `q` exits, and `--preview-seconds 60` changes the duration. The preview
uses a resizable window with original pixels; zoom your display sufficiently to
judge sharpness. The script turns off autofocus before writing the requested
manual position. Headless systems should choose focus through their normal image
viewer/camera tool, then enter the measured setting in the rig.

Write the chosen number into `cameras.c920.capture.focus`; command-line trials do
not edit the rig. Keep C920 `zoom: 100`. Do not put a `focus` setting under Brio.

```bash
python3 scripts/camera_setup.py --camera brio101 --apply --preview
python3 scripts/camera_setup.py --apply --output camera_profile.json
```

`--apply` preflights all selected devices before writing. It checks supported
resolution/format/frame rate, allowed control ranges and menu values. It selects
manual exposure, disables dynamic frame rate if that control is exposed, disables
automatic white balance, sets temperature and gain, and locks C920 autofocus and
focus/zoom. It reads settings back and fails if a required setting is unsupported,
rounded, ignored or inconsistent. Brio focus controls are never written, even if
a driver unexpectedly advertises them.

The code accepts both older V4L2 names such as `exposure_auto`, `exposure_absolute`,
`focus_auto`, and the connected device's names `auto_exposure`,
`exposure_time_absolute`, `focus_automatic_continuous`. Exposure values in the rig
are **microseconds**; the code converts to the standard V4L2 100 µs units.
[Linux camera controls](https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html).

Only configure optional image controls (`brightness`, `contrast`, `saturation`,
`sharpness`, `backlight_compensation`) if needed and supported; the script will
validate and apply them when present. All exposed actual control values are saved
in the profile. If an unconfigured control matters to your image consistency,
copy its chosen value into the capture profile so it is reapplied each time.

## Intrinsics and calibration consistency

Place your measured matrix and distortion coefficients in
`calibration/c920.yaml` and `calibration/brio101.yaml`:

```yaml
image_size: [1920, 1080]
K: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]  # replace symbols with measured numbers
dist: [k1, k2, p1, p2, k3]               # standard OpenCV coefficients
capture_profile:                         # copy this camera's calibrated capture dict
  width: 1920
  height: 1080
  fps: 30
  fourcc: MJPG
  exposure_us: 4000
  gain: 0
  white_balance_kelvin: 4000
  power_line_frequency: 2
  # C920 additionally: focus: <chosen number>, zoom: 100
```

These symbols are explanatory and cannot be used as numeric calibration data.
`image_size` must match the saved raw images. `capture_profile` records how the
calibration images were made. The recorder requires the saved and requested profile dictionaries to have exactly
the same keys and values; missing, newly added or removed controls are rejected,
as are changed focus, exposure or frame size. An empty dictionary is invalid. Missing
intrinsics or a missing profile allow recording calibration data, but are reported
as unverified; they do not manufacture a valid calibration. The intrinsics files'
SHA-256 hashes and the complete rig snapshot are stored in every session.

Changing focus, zoom, resolution or image transforms requires checking/repeating
intrinsics calibration; moving a camera requires repeating stereo calibration.
If only illumination/exposure/gain changes, keep the historical calibration file
and explicitly document and verify the replacement profile. The strict profile
check is intended to catch accidental drift, including photometric differences
that can affect segmentation.

## Record every calibration and measurement through the same command

```bash
python3 scripts/record.py --mode checkerboard --duration 45 --output recordings/board_01
python3 scripts/record.py --mode roll --duration 12 --output recordings/roll_01
python3 scripts/record.py --mode swivel --duration 12 --output recordings/swivel_01
python3 scripts/record.py --mode motion --duration 20 --output recordings/motion_01
```

Start the axis sweeps at the instructed home pose and use moderate, smooth motion.
Pause the checkerboard at each pose before moving it to the next. Use varied board
orientations and image locations, with the full pattern sharp in both views. A
stopped board makes frame pairing much less sensitive to unsynchronized webcams.
See the repository README for extracting paired calibration images and running
intrinsics/stereo/axis calibration commands.

The recorder opens both cameras, requests the configured format, starts streaming
and **reapplies controls after the first streaming frame**. The C920 was observed
to change exposure on STREAMON even after a successful pre-stream readback. Five
subsequent warmup frames and another control check must complete on both cameras
before the common recording clock starts. Preview uses these checks too; a
standalone `--apply` without preview only verifies settings before streaming.
The recorder checks OpenCV and V4L2 format readback, validates each MJPEG frame's
encoded dimensions, and records each camera on a separate worker thread without
resizing. Controls are checked again at the end. Existing session directories
are refused. A `complete` session means capture/control checks passed; inspect
observed FPS and interval statistics separately to assess throughput.

Each session contains:

| File | Purpose |
|---|---|
| `c920.avi`, `brio101.avi` | Independent MJPG videos; stored AVI playback rate is the requested FPS |
| `c920_timestamps.csv`, `brio101_timestamps.csv` | Actual per-frame host receive timestamps, indexed from 0 |
| `session.json` | Requested and actual camera state, configuration/calibration hashes, timing assumptions, observed rates and status |

The CSV columns are `frame_index,timestamp_s,read_start_s,read_end_s`.
`timestamp_s` is the same as `read_end_s`; all times are seconds from one shared
monotonic origin. The cameras keep separate frame counts. **Use CSV times, not
AVI frame number / 30, for fitting.** The manifest's gap/dropped-frame estimates
are derived from receive intervals; the webcams do not expose authoritative
sensor frame counters through this recorder. `Ctrl+C` finishes files and marks
the session `interrupted`; capture/readback failure marks it `failed`.

## Timing is separate from calibration

These cameras do not have a hardware trigger. Matching host timestamps does not
guarantee matching exposure instants: USB transfer, frame buffering, exposure
duration, scheduling and rolling shutter can differ. Independent worker threads
avoid deliberately reading camera 2 only after camera 1, but do not remove these
effects.

Record a flashing light or another sharp common event visible to both cameras to
estimate a relative latency. Repeat the event through a clip to check offset and
drift. If Brio's event is timestamped later by 30 ms, use
`timing.brio_offset_s: -0.030`, because the processing convention is
`aligned_brio_time = recorded_brio_time + brio_offset_s`. Set `verified: true`
only after measuring timing on the actual recording setup. Repeat when the
capture configuration or USB arrangement changes. `max_pair_skew_ms` limits
pair matching but is not a bound on unknown exposure latency. Fast motion may
require a hardware-synchronized camera pair for the desired accuracy.

At 100 degrees/s, 10 ms of remaining mismatch corresponds to one degree of
motion. For initial setup, static checkerboard pauses and slower axis sweeps
make this limitation easier to control.
