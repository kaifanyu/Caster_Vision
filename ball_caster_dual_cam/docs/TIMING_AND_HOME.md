# Measure timing, then record calibration

The recorder saves original receive timestamps from independent camera threads.
It does not shift timestamps or synchronize exposures. Offline pairing uses:

`aligned_brio_time = recorded_brio_time + timing.brio_offset_s`

A constant offset corrects relative latency. It does not lock frame phases,
remove USB jitter, or eliminate rolling-shutter and exposure-integration effects.
At 30 fps, individual frames are about 33 ms apart. Repeated events help estimate
the offset, but the result is not a guarantee of millisecond exposure accuracy.

## 1. Record a common light signal

Keep camera mounts, USB connections, resolution, FPS, focus and exposure settings
the same as for the axis recordings. Do not move or refocus cameras to see the
signal. Place a diffuse light source or illuminated patch visible to BOTH cameras,
preferably near the caster's image region and at similar image height. The same
physical light event must reach both views. Two independently controlled lights
or displays are unsuitable. Keep illumination moderate enough to avoid clipping.

```bash
python3 scripts/record.py --mode timing --duration 25 --output data/timing_01
```

Wait for `Recording NOW`. Leave the light OFF for about 2 seconds. Make 8-12
complete ON/OFF pulses, changing each ON and OFF duration irregularly between
about 0.4 and 1.2 seconds. Spread the sequence over at least 8 seconds. Finish
with the light OFF for at least 2 seconds. The caster remains stationary.
If you finish early, simply keep the light off until capture ends.

```bash
python3 scripts/calibrate_timing.py --session data/timing_01 \
  --pick-rois --output out/timing_01
```

Select a rectangle enclosing the same illuminated patch in each view, then press
Enter. Selection is on the first RAW frame, where the light should still be off;
the chosen patch must be identifiable. For a headless run, replace `--pick-rois`
with `--c920-roi X Y W H --brio101-roi X Y W H` using measured raw-pixel rectangles.

The command reads every camera frame at its own CSV timestamp, extracts patch
brightness, and matches the complete ON/OFF sequence. It does not pair the videos
before estimating the offset. It rejects incomplete/different pulse sequences,
low contrast, regular blinking, and excessive delay variation or drift. It saves:

- `report.json`: measured offset, per-event brackets, residual spread and drift.
- `c920_brightness.csv`, `brio101_brightness.csv`: inspectable signals.
- `timing_suggestion.yaml`: proposed timing values, only when checks pass.

Repeat in a fresh `data/timing_02` recording and compare the offsets and scatter.
Copy the measured `brio_offset_s` into `config/rig.yaml` only after reviewing both
results. For example, if Brio sees the same event at a receive timestamp 30 ms
later, the correction is **-0.030 seconds**. This number is an example, not your
measured offset. Record the report paths and uncertainty beside the setting.
`verified: true` is an operator declaration after repeatability checks; the tool
leaves it false. Treat a spread of tens of milliseconds as material for motion
measurement, even if the broad consistency checks pass.

No command here rewrites recording timestamps or changes the rig config.
Existing accepted axes become stale when the timing offset changes and must be
recalibrated. Repeat timing measurement if capture settings or USB layout change.

## 2. Record each axis separately

Use a repeatable mechanical home. Start each recording at exactly the same roll
pose. Paint phase may differ between recordings because surface landmarks are
local to each clip. Maintain a stable, rigid camera-to-caster mounting.

| Time after `Recording NOW` | Action |
|---|---|
| 0-3 seconds | Hold the complete assembly and both shell phases still at home. |
| 3-9 seconds | Smooth, one-direction sweep of about 40 degrees. |
| 9-12 seconds | Hold the final position; do not return to home during capture. |

The 3+4+3 second sequence is reasonable if the markings stay sharp and tracking
remains continuous; 3+6+3 gives more margin on this rig. The angle need not be
exactly 40 degrees: it excites the fit and is not imposed as ground truth.

For **roll**, change mechanical roll only and keep both shell spins fixed.
For **swivel**, keep mechanical roll locked at home and spin ONE shell about its
swivel axis. The other shell can stay still. Do not perform a roll motion in the
swivel recording. Keep hands off the tracked markings and out of both views.

```bash
python3 scripts/record.py --mode roll --duration 12 --output data/roll_home_04
# Physically restore the same mechanical home before the next command.
python3 scripts/record.py --mode swivel --duration 12 --output data/swivel_home_04
```

The recorder now prints retained pairs, largest paired interval, and gaps over
0.1 seconds. You can also inspect an existing recording without fitting:

```bash
python3 scripts/calibrate_timing.py --session data/swivel_home_04 \
  --pairing-only --output out/swivel_home_04_pairing
```

Aim for intervals near 33-67 ms. Inspect gaps over 0.1 seconds; multi-second gaps
during motion remain a problem. A measured constant offset may NOT remove them.
Do not widen `max_pair_skew_ms` or choose a false offset just to maximize pairs.
The camera clocks are free-running; persistent gaps may require lower motion
speed, improved capture stability, asynchronous geometric fitting, or synchronized
capture hardware. The last two are not implemented by this change.

## 3. Fit with a checked home interval

```bash
python3 scripts/calibrate_axes.py \
  --roll data/roll_home_04 --swivel data/swivel_home_04 \
  --home-hold-s 2.5 --max-frames 400 --output out/axes_home_04
```

The default 180-frame limit can cut off much of a 12-second clip. Use 400 to retain
the full sweep and final hold. `--home-hold-s 2.5` intentionally leaves a half-second
margin before the planned movement begins at 3 seconds. This duration is measured
from the first paired frame, so inspect timing if the beginning has missing pairs.

The new home option checks persistent tracks in BOTH shells and cameras against
the first paired image. Each tested image needs at least the configured minimum
reference tracks; the median displacement must stay within 2 pixels. If it fails,
inspect motion/blur/occlusion and shorten the interval only when a shorter interval
was actually stationary. Do not force a moving interval to be home.

The fit fixes all angles in this checked interval at zero. Tracks may anchor to
any observed, supported frame in that interval, rather than relying on one image.
This adds repeated physical observations of the known pose without an arbitrary
100x weight. Standalone solver callers declaring `home_hold_s` are responsible for
verifying that physical assumption; the CLI performs the pixel check.

Rotation initialization now estimates consecutive native-image rotations and
composes them between paired samples. This uses intermediate images even when
feature identities change over a long interval. A missing native rotation marks
the composed interval unavailable. If BOTH cameras lack a shell's interval, the
tracker warns and records that interval in the report; the angle initializer
holds its previous provisional guess there, not a measured zero rotation.
These approximate rotations only initialize the pixel fit. They do not fabricate
geometric track connections through a gap.

Reports retain per-component supported home counts, supported excursion and last
supported timestamp even when failed estimates are invalidated. Acceptance still
requires convergence, >=70% accepted observations per camera, <=2 px inlier RMSE,
>=15 degrees of supported axis excursion, and sufficient axis rank. The changes
do not weaken these gates or guarantee that an old recording will pass.

## Validation on the previous swivel clip

With the original circle settings, composing native increments restored valid
rotation initialization across the 3.80-5.18 s gap for both cameras and shells.
The earlier endpoint-only C920 red estimate was unavailable. This is an
initializer improvement, not an accepted axis fit. The same recording fails the
new 2.5-second stationary-home check: C920 red features move about 9.8 pixels.
Do not apply a 2.5-second zero-pose constraint to that old clip. The corresponding
diagnostic is `out/native_rotation_check_20260918/report.json`.

Reference: [Linux V4L2 timestamp semantics](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/buffer.html)
distinguish driver timestamp sources; this recorder uses host read-return time,
not a verified hardware exposure timestamp.
