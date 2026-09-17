# Kalibr import and axis calibration — 2026-09-17

Your updated Kalibr result is installed: `cam0` is the C920, `cam1` is the Brio 101.
Both calibrations use 1920×1080 pinhole/radtan images. The stereo baseline is
0.254111 m. This is the supplied conversion of
`stereo_checkerboard_rot180_20260917` into original, unrotated image coordinates;
do not rotate the input images. `cam1.T_cn_cnm1` maps C920 coordinates into Brio coordinates and
was copied directly into `R_21` and `t_21_m`, without inversion.

The files used by `config/rig.yaml` are:

| File | Content |
|---|---|
| `calibration/c920.yaml` | C920 intrinsic matrix and distortion |
| `calibration/brio101.yaml` | Brio intrinsic matrix and distortion |
| `calibration/stereo.yaml` | Camera-to-camera rotation/translation and intrinsic hashes |
| `calibration/kalibr_camchain.yaml` | Your complete supplied Kalibr result |

The files replaced by this update are preserved in
`calibration/archive/before_raw_stereo_20260917_202733_893040/`.
The earlier import's backup remains in
`calibration/archive/before_kalibr_20260917_001530/`.
Camera mounts were confirmed unchanged. Capture settings in `rig.yaml` were
preserved. The exact Kalibr capture-profile manifest and quality report were not
supplied, so those verifications are not claimed. Keep resolution, lens focus and
zoom consistent with the images used by Kalibr.

## What happened with the existing recordings

These are historical results from the earlier calibration; they have not been
rerun with the updated camera values. No accepted axes file exists yet.

`data/roll_01` was treated as the requested pure-roll clip, alongside
`data/swivel_01`. Current timing settings retain 127 and 130 pairs, respectively.
Both cameras see features on both colored shells. Preliminary rotation evidence
gave an 85.93° separation between roll and swivel directions; this is only an
initializer, not an accepted axis calibration.

The first full attempt failed the home-track support check. In the roll recording,
pairing jumps from source frame 0 to source frame 39, a 1.315 s interval. Only five
tracks at home survived the minimum-length filter; eight are required. Details:
`out/kalibr_axis_check_20260917_001530/axes/report.json` and
`out/kalibr_axis_check_20260917_001530/support_diagnosis.json`.

The tracker was corrected to follow every native frame between selected pairs.
Unpaired images now preserve feature identities; only synchronized paired pixels
enter the geometric fit. Pairing tolerance and acceptance thresholds were not
relaxed. Regression coverage includes a rendered sequence with missing timestamp
pairs and verifies the rotation over the entire paired interval.
On your actual roll video, a short tracking-only check retained 47 track IDs
across all first three selected pairs; see
`out/kalibr_axis_check_20260917_001530/native_tracking_short_check.json`.
This verifies improved tracking through the gap, not valid calibrated axes.

The retry was stopped after you clarified that the roll clip started roughly 8°
away from the swivel clip's initial roll pose. The current calibration model fixes
both clip starts to the same home roll. That assumption is false for these clips.
No accepted `calibration/axes.yaml` was produced. An approximate 8° estimate is
not an adequate replacement for a known starting angle.

## Next recordings

Keep both cameras mounted. There is no need to redo Kalibr merely because the
caster moves or the vehicle changes position, provided the camera-to-camera and
camera-to-caster mounting geometry stays rigid.

Use a repeatable mechanical home pose for both recordings. If you can reproduce
the existing swivel clip's starting roll exactly, only the roll clip needs
replacement. Otherwise record both again as below. Paint phase may differ;
mechanical roll must match. Red appearing on top in an image is not a universal
home definition because the cameras view the caster from different orientations.

```bash
cd /home/hamr/Caster_Vision/ball_caster_dual_cam
python3 scripts/record.py --mode roll --duration 10 --output data/roll_home_01
python3 scripts/record.py --mode swivel --duration 10 --output data/swivel_home_01
```

For each clip, wait for `Recording NOW`, hold home for three seconds, make a slow
20–60° monotonic sweep over about four seconds, then hold still. In roll, keep both
shell spins fixed. In swivel, hold mechanical roll at home and turn one or both
shells about the swivel axis; if turning both, use the same signed direction.
Keep hands off the visible painted surfaces. Use new directory names on retries.

Camera timing is still `verified: false`. Record repeated shared light events
visible to both cameras, measure relative delay, and set `timing.brio_offset_s`
before the final axis fit. For example, a Brio event received 30 ms later needs
offset -0.030 s. Do not tune the offset simply to make a fit pass. A zero offset
and host timestamp pairing alone do not verify exposure synchronization.

## Fit and inspect the pure-roll clip itself

First fit **both axes from both calibration clips**:

```bash
python3 scripts/calibrate_axes.py \
  --roll data/roll_home_01 --swivel data/swivel_home_01 \
  --max-frames 300 --output out/axes_home_01
```

Proceed only when this reports `Accepted axes` and the saved `report.json` has
`success: true`. The solver checks convergence, sufficient motion, axis rank,
distributed visible surface tracks, home connectivity, and image residuals in
both cameras. The configured inlier RMSE limit is 2 pixels and minimum inlier
fraction is 0.7; these are acceptance gates, not a measured angular accuracy.

Then replay the same pure-roll clip using the general-motion estimator:

```bash
python3 scripts/run.py --session data/roll_home_01 \
  --allow-calibration-clip --initial-roll-deg 0 --max-frames 300 \
  --output out/roll_home_check_01
python3 scripts/render.py --results out/roll_home_check_01/results.json \
  --output out/roll_home_check_01/overlays
```

The new `--allow-calibration-clip` flag does not relabel the source recording.
It lets all three angles vary with the calibrated axes/pivot fixed, rather than
forcing the two shell spins to zero. In `results.csv`, supported `alpha_deg`
should follow the roll sweep, while `beta_red_deg` and `beta_green_deg` should stay
near zero. Check their separate validity flags; blank values are unresolved.
The two AVI overlays show a blue fixed roll axis and a yellow moving swivel axis.

Reusing a calibration clip tests consistency. Record a separate roll check and
a mixed-motion clip for independent evaluation, ideally with known angles.
To track the old `roll_01` after obtaining valid axes, its initial roll relative
to the newly chosen home must be measured and supplied to `--initial-roll-deg`.
That tracking argument does not correct unequal starts in `calibrate_axes.py`.

The estimator uses KLT tracking and robust geometric fitting, not a Kalman filter.
All 89 repository tests passed after the tracking and replay changes. Physical
axis accuracy remains unvalidated until suitable recordings pass these checks.
