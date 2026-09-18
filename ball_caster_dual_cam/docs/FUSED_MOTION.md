# One trajectory from two native video streams

`scripts/track_motion.py` is a motion tracker using the accepted axes calibration.
It does not change `calibration/axes.yaml`, the original batch motion estimator,
or any calibration quality gate.

```bash
python3 scripts/track_motion.py \
  --session data/run2 --initial-roll-deg 0 \
  --output out/run2_fused --render
```

The output directory must be new or empty. All recorded frames are processed by
default; `--max-frames N` limits the number **per camera**, not the number of paired
frames. Processing is offline and is not advertised as real time.

## Inputs and reference

Use arbitrary roll and independent shell motion. No scripted roll-only or
swivel-only action is required during normal tracking. The initial roll relative
to calibrated home must be known; use zero only at home. Both first images must
show that same starting pose. Red/green spins are relative to this recording's
initial painted-feature phases. Camera mounts, focus/zoom/resolution and caster
pivot must match calibration. Translation of the whole caster is not modeled.

External MP4/AVI files are supported with their original timestamp CSVs:

```bash
python3 scripts/track_motion.py \
  --c920-video recordings/c920.mp4 --brio-video recordings/brio101.mp4 \
  --c920-timestamps recordings/c920_timestamps.csv \
  --brio-timestamps recordings/brio101_timestamps.csv \
  --initial-roll-deg 0 --output out/external_fused --render
```

CSV columns are `frame_index,timestamp_s`; indices start at zero and are
consecutive. Times must increase and use a common clock. Original decoded frame
order and calibrated image dimensions must be preserved. Do not invent equal
timestamps from an AVI's nominal FPS, or reuse sidecars after dropping frames.
An arbitrary MP4 alone does not identify its camera, optical calibration,
starting mechanical pose, or its timing relationship to a second recording.
External files have no capture-profile verification; current rig settings are
assumed and that assumption is recorded.

## Tracking and fusion

Every native frame is processed at its own recorded timestamp, with the configured
Brio offset applied. There is no 12 ms pair-selection gate in this workflow. The
camera offset must still be measured: accepting every frame does not make host
receive time equal to exposure time.

The default `--measurement-source rotation` ports the earlier project's temporal
rotation tracking: native KLT tracks, RANSAC, forward/backward validation,
distributed support, blur checks, direct keyframe matches, appearance verification,
and a bounded bank of accepted views for recovery. Motion predictions only seed
image searches. Failed/predicted views cannot become visual anchors. Existing
`temporal` settings can be supplied in a copied rig YAML; defaults retain the
original temporal quality gates and enable motion recovery.

Camera rotations are converted into the common C920/calibrated mechanism frame.
Unobserved shells are omitted, excessive off-mechanism rotation or disagreement
between shell roll estimates rejects the update. One six-state angle/rate Kalman
filter receives the remaining native-time measurements from **both** cameras.
It estimates common roll plus separate red and green spins, rather than averaging
two independent finished trajectories. Innovations are gated, covariance grows
without observations, and stale/high-uncertainty predictions are hidden.

The rotation source uses the approximate enclosing-sphere image model and is
**not a pass of the stricter separated-shell metric pixel fit**. Its 1.5 degree
measurement-noise assumption and reported covariance are tuning choices, not an
independently measured accuracy. Calibration bias and correlated landmark drift
are not included in this covariance. The result explicitly records its source.

`--measurement-source metric` instead requires a robust fixed-material-point fit
on the calibrated separated hemispheres for each update. It uses a bounded camera-
local keyframe bank, with forward/backward and appearance-verified image matches.
Only a known initial pose or accepted metric visual update can create new points.
One shell can update roll and its own spin without fabricating the other's spin.
This stricter path can reject real recordings when calibration/model error,
surface-point initialization or blur prevent sufficient pixel agreement. It does
not silently fall back to the approximate rotation source.

The joint batch surface-refinement settings in `solver:` apply to the existing
batch solver, not this native-time filter. New filter and metric-map settings use
the `fusion:` section. Examples of tuning values (defaults) are:

```yaml
fusion:
  accel_noise_deg_s2: 240.0
  innovation_gate_sigma: 5.0
  max_prediction_s: 0.35
  max_angle_std_deg: 10.0
  rotation_measurement_std_deg: 1.5
```

Larger motion noise allows faster acceleration but follows noisier measurements.
Increasing the prediction horizon does not restore missing image evidence.

You can retune a saved rotation-source result without decoding the videos again:

```bash
python3 scripts/refilter_motion.py --results out/run2_fused/results.json \
  --accel-noise-deg-s2 240 --measurement-std-deg 1.5 --max-prediction-s 0.35 \
  --output out/run2_refiltered
```

This changes only the filter. It preserves the saved visual acceptance decisions;
it cannot create new keyframe matches or restore observations rejected by tracking.

## Output interpretation

`results.csv` contains one event per native image, the shared roll and shell spin
angles/rates, per-component status/uncertainty/turn-count flags, caster-frame
quaternion (`xyzw`) and swivel-axis direction in C920 coordinates. `results.json`
additionally contains the visual source, rejection and keyframe diagnostics.

The caster-frame quaternion is `F @ Rx(roll)`. The roll axis is `F[:,0]`; the
swivel axis is `(F @ Rx(roll))[:,2]`. Each shell's material orientation additionally
uses its own `Rz(spin)`. The two independently spinning shells therefore cannot
be described by a single rigid-body quaternion.

- `home`: the supplied starting reference.
- `vision`: an accepted visual update from the configured measurement source.
- `predicted`: short continuation without a fresh update for that component.
- `unresolved`: no supported finite output within the prediction limits.
- `turn_count_valid=false`: a gap exceeded the prediction horizon; recovering an
  orientation does not determine whole revolutions that could have been missed.

Completion means processing finished, not that every frame was accepted or that
physical accuracy is proven. Inspect coverage throughout the movement, compare
known angles/return-to-home repeatability, and validate against an independent
reference before relying on absolute angular accuracy.

`--render` produces `overlays/c920_axes.avi` and `overlays/brio101_axes.avi`, using
the **same shared state** at each view's native timestamp. Orange means prediction;
unresolved moving axes are hidden. The original `scripts/render.py` also accepts
these result files. Playback uses median FPS; CSV timestamps are authoritative.

## Interactive 3D orientation

Export one combined 3D replay from an existing fused result:

```bash
python3 scripts/visualize_motion.py \
  --results out/run2_fused_03/results.json \
  --output out/run2_fused_03/orientation_3d.html
```

Choose a new or empty HTML output file, then open it in a browser. The export
reads the saved geometry and shared state; it does not rerun tracking or need the
original videos or current calibration files. It uses the existing Python
environment with no additional packages. The generated HTML contains all data
and viewer code and works offline without a server or downloaded assets.

Use **Play**, scrub the timeline, or step through native image events. Drag the
3D view to orbit and scroll to zoom. The view shows one caster frame combining
both cameras, with a fixed roll axis and moving swivel axis, plus independent
red/green shell spins. Switch between calibrated home and C920 coordinates;
both views are centered on the fixed pivot, so this is an orientation replay,
not a recovered position trajectory. Shell markings are illustrative rather
than reconstructed paint.

The viewer preserves each component's `home`, `vision`, `predicted`, and
`unresolved` labels. Predicted moving axes are dashed orange. Unresolved roll
hides moving axes and shells; an unresolved shell spin hides its material
markings while retaining its rotationally symmetric shape when roll is known.
There is no interpolation or gap filling, and source gaps longer than 100 ms
hide moving geometry. Whole-turn uncertainty remains visible after recovery.
The 3D display does not improve tracking or establish physical accuracy.

Better intrinsics, stereo extrinsics, axes, pivot geometry and timing can improve
agreement and reduce bias. They do not undo motion blur, recover entirely unseen
spin, or substitute for a starting reference. Automatic unknown-start operation
would additionally require a saved identifiable material map or an absolute pose
cue; this implementation does not claim to provide that capability.
