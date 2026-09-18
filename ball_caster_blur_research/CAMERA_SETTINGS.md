# Recording with less motion blur

Yes: the capture exposure is a YAML setting in this repository. For the existing
Linux dual-camera recorder it is `cameras.<name>.capture.exposure_us` in
`ball_caster_dual_cam/config/rig.yaml`. Editing an analysis configuration after
recording cannot remove the blur already present in an AVI.

The original configuration and accepted reconstruction remain unchanged. Three
complete trial presets are supplied here:

| Preset | C920 exposure | Brio 101 exposure | Expected V4L2 exposure readback |
|---|---:|---:|---:|
| [capture_2ms.yaml](config/capture_2ms.yaml) | 2000 us | 2000 us | 20 on each camera |
| [capture_1ms.yaml](config/capture_1ms.yaml) | 1000 us | 1000 us | 10 on each camera |
| [capture_0p5ms.yaml](config/capture_0p5ms.yaml) | 500 us | 500 us | 5 on each camera |

These are **requested trials, not hardware-verified settings**. The software
configuration and control mapping have been tested without opening cameras. The
camera firmware may quantize or reject a value even when its advertised range
includes it. In this rig's earlier readiness check, C920 requested 4000 us settled
to 3800 us while streaming. If a trial reports a mismatch, the reported actual
value is only a candidate: set that camera's `exposure_us` to 100 times its
V4L2 readback and repeat the full preview/record/readback check to establish
stability. Do not disable the strict check or assume 1000 us was achieved. A
failed trial is not a completed measurement.

Start with 2 ms, then try 1 ms. Use 0.5 ms if the fastest visible marks still smear
and the images have adequate signal. The cameras may use different accepted
exposures; record both actual values. No hardware settings were changed during
this implementation.

## What changes and what remains fixed

The current run's `session.json` records C920 exposure **15600 us** and Brio
exposure **8000 us**, matching post-start, post-warmup and final control readback.
At the same image velocity, 1 ms gives approximately 15.6 times less blur on the
C920 and 8 times less on the Brio. It also collects that much less light at fixed
gain and illumination. The image does not automatically become useful just
because it becomes sharper.

The presets change exposure only, plus the relative paths needed from their new
folder. They retain:

| Setting | C920 | Brio 101 |
|---|---|---|
| Image size / codec / requested rate | 1920 x 1080, MJPG, 30 fps | 1920 x 1080, MJPG, 30 fps |
| Gain | 180, fixed during the first exposure sweep | 180, fixed during the first exposure sweep |
| White balance | 4000 K, manual | 4000 K, manual |
| Focus / zoom | Focus 50, autofocus off, zoom 100 | Fixed-focus lens; no focus entry |
| Brightness / contrast / saturation / sharpness | 128 each | 128 each |
| Backlight compensation | 0 | 0 |
| Anti-flicker menu | 2 (60 Hz) | 2 (60 Hz) |

Use strong **continuous, diffuse illumination on both shells**, especially the
C920 view of the green shell. Preserve the paint's contrast without clipping
bright marks to white. Use a light source that does not show bands or brightness
oscillation at the chosen short exposure; the 60 Hz menu does not guarantee that
an arbitrary LED lamp becomes flicker-free. Keep gain fixed for the first sweep
so its effect is not confused with exposure; then lower gain if stronger lighting
allows it. Increasing digital brightness or sharpness does not replace missing
photons or undo integration blur.

Keep camera mounts, focus, zoom, resolution and raw image orientation unchanged.
Changing these can invalidate geometric calibration. Exposure alone does not
change the ideal pinhole geometry, but darker/noisier images or changed lighting
can alter segmentation and feature localization. Check a stationary checkerboard
and shell masks again before using the new recordings for measurements.

## Why the YAML works

The checked path through the current source is:

1. `ball_caster_dual_cam/scripts/record.py --config <file>` calls
   `dualcam.capture.load_capture_config`.
2. `V4L2Camera.plan_controls` converts `exposure_us // 100` to the device's
   `exposure_time_absolute` / `exposure_absolute` control.
3. The recorder selects the driver's advertised **Manual Mode**, disables
   dynamic frame rate if available, fixes gain and white balance, and disables
   C920 autofocus before setting focus 50.
4. It reapplies settings **after streaming begins**, discards five warmup frames,
   and requires readback to match before starting the recording clock. It checks
   controls again at the end and writes the results into `session.json`.

Do not add `auto_exposure: false` or `autofocus: false` under `capture`:
those YAML keys are unsupported and capture validation rejects them. The existing
recorder already enforces those choices. No exposure-application code fix was
needed. A missing device control, unsupported format, rounding or ignored setter
is reported as an error instead of silently accepted.

YAML uses **microseconds**. Linux V4L2 exposure uses **100-microsecond units**;
`exposure_us: 1000` means V4L2 value `10`, not `1000`. The accepted configuration
requires multiples of 100 us. This conversion follows the [Linux camera control
reference](https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html).

## Exact commands on the Linux capture machine

These commands use the actual standalone recorder and support the new presets.
They run on the Linux computer connected to the cameras, **not native Windows
PowerShell**. The existing recorder depends on `/dev/v4l/...`, `v4l2-ctl`, and
OpenCV's V4L2 backend. Copy the new presets with the checkout to that machine;
the relative calibration paths assume these two project folders remain siblings.

From the `Caster_Vision` directory on that machine:

```bash
# Read capabilities only; nothing is changed by --inspect.
python3 ball_caster_dual_cam/scripts/camera_setup.py \
  --config ball_caster_blur_research/config/capture_2ms.yaml \
  --camera all --inspect \
  --output ball_caster_blur_research/experiments/capture_capabilities.json
```

Close other camera programs. Confirm each by-ID device path identifies the
correct camera; different serial numbers require updating that camera's `device`
entry in the trial preset. If `v4l2-ctl` is unavailable, install the Linux
`v4l-utils` package using the capture machine's package manager. The preview
commands below need a graphical desktop; on a headless system, use a short
recording and inspect its frames and saved control readbacks instead.

Preview **while streaming**, once per camera. This step applies the trial:

```bash
python3 ball_caster_dual_cam/scripts/camera_setup.py \
  --config ball_caster_blur_research/config/capture_2ms.yaml \
  --camera c920 --apply --preview --preview-seconds 20 \
  --output ball_caster_blur_research/experiments/c920_2ms_preview.json

python3 ball_caster_dual_cam/scripts/camera_setup.py \
  --config ball_caster_blur_research/config/capture_2ms.yaml \
  --camera brio101 --apply --preview --preview-seconds 20 \
  --output ball_caster_blur_research/experiments/brio101_2ms_preview.json
```

Then record a stationary lead-in followed by the same representative fast swivel
on both cameras. Start the motion after the terminal says `Recording NOW`.

```bash
python3 ball_caster_dual_cam/scripts/record.py \
  --config ball_caster_blur_research/config/capture_2ms.yaml \
  --mode motion --duration 12 \
  --output ball_caster_blur_research/experiments/capture_2ms_trial01
```

Repeat using `capture_1ms.yaml` and a new output directory such as
`capture_1ms_trial01`; then do the same for `capture_0p5ms.yaml` if needed.
Existing recording output directories are deliberately refused. A preview alone
does not test simultaneous USB bandwidth or a full recording's frame integrity.

All rig/calibration paths are relative to the selected YAML, independent of the
working directory. The supplied presets already resolve to the original
intrinsics, stereo and axes files. **Changing a preset does nothing unless the
recorder is started with that preset's `--config` path.**

## Verify what was actually saved

For the 1 ms trial, inspect these fields in the new `session.json`:

| Field (under each camera unless stated) | Required interpretation |
|---|---|
| Top-level `status` | `complete` for a full requested trial; investigate `failed` |
| `requested.exposure_us` | 1000 for the unmodified 1 ms preset |
| `controls`, `warmup_controls`, `final_controls` -> `exposure_time_absolute` | 10 on the current driver; older driver may call it `exposure_absolute` |
| Those same controls -> `auto_exposure` | Current driver uses 1 for Manual Mode; confirm menu labels on other drivers |
| Those same controls -> `exposure_dynamic_framerate` | 0 |
| Those same controls -> `white_balance_automatic` | 0 |
| C920 controls -> `focus_automatic_continuous`, `focus_absolute`, `zoom_absolute` | 0, 50, 100 |
| `actual_format` and `opencv_format` | 1920 x 1080, MJPG, 30 fps |
| Top-level `stats` | Actual frame count/rate and long intervals; approximately 30 fps is expected, not guaranteed by the AVI header |

Compare the recorded moving paint edges with stationary ones and the old
15.6/8 ms run. Look for reduced streak width, usable contrast on both red and
green shells, no clipping, no frame loss and stable color masks. Select the
shortest **verified** exposure that retains enough usable texture in both views.
At a projected feature speed of 3000 pixels/s, 2 ms still causes about 6 pixels
of motion blur; 1 ms causes 3 pixels and 0.5 ms causes 1.5 pixels. This is an
illustration, not a measured maximum speed for this run. Short exposure does not
resolve all 30 fps aliasing or guarantee recovery of whole turns.

Control readback is not an independent measurement of sensor integration time.
If precise blur-derived angular velocity is required, measure exposure/timing
with a controlled flashing source or equivalent calibration. Repeat the shared
timing event after changing exposure because USB receive timestamps are not
exposure midpoints. Do not mark `timing.verified: true` based on matching fps.

## Calibration metadata and ROS caveat

The current imported intrinsics have `capture_profile: null`, because the supplied
Kalibr calibration did not record its camera settings. The recorder accepts the
presets with an explicit metadata warning; it does not certify that the historical
focus matches the new settings. If a future intrinsics YAML contains a complete
non-null `capture_profile`, changing exposure will fail the strict comparison.
Preserve the historical file and verify/document the new capture profile in a
new calibration artifact; do not erase known metadata to suppress that check.

The local sibling checkout `../hamr_control/hamr_bringup/launch/recording.launch.py`
currently launches a **single-camera ROS `v4l2_camera` recorder** at defaults
640 x 480 YUYV and does not load this dual-camera YAML. Its
`hamr_HW.launch.xml` includes that launch file. Therefore editing `rig.yaml`
does **not** change that local ROS capture path. The dual-camera README describes
a newer integration on the Linux machine that is not present in this local
sibling source, so its launch argument names cannot be confirmed here.

For these trials, use the explicit standalone commands above. On the robot's
deployed checkout, inspect the camera recorder launch command/config argument and
confirm the resulting `session.json.config_snapshot` contains the trial exposure
values. Avoid launching a second recorder against cameras already opened by ROS.
The standalone recorder does not move the robot or publish trajectory commands.

## If recording directly on Windows

Editing this YAML is enough for the supplied **Linux V4L2** backend; it does not
create a native Windows capture backend. No Windows camera settings were touched.
For a Windows app exposing the DirectShow exposure property, values are log base
2 seconds: approximately `-9` = 1.953 ms, `-10` = 0.977 ms, `-11` = 0.488 ms.
Use manual exposure and check the particular driver's supported range. These
values are not microseconds and must **not** be pasted into `exposure_us`.
Different Windows camera APIs/apps can expose other representations; these
numbers specifically describe the [Microsoft DirectShow CameraControl exposure
property](https://learn.microsoft.com/en-us/windows/win32/api/strmif/ne-strmif-cameracontrolproperty).

Brio 101 is fixed focus and is specified for 1080p/30 fps; do not assume a YAML
`fps: 60` or a focus setting adds hardware capability. See the [Logitech Brio 101
specification](https://support.logi.com/hc/en-us/articles/16131499767191-Specification-Brio-101).

Reviewed against local capture source and recorded metadata on 2026-09-18.
