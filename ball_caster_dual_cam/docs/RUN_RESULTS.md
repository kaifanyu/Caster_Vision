# Supplied dual-camera recording: implementation and results

The completed deliverable is `out/orientation_final`. Open
`replay/combined_tracking.mp4` for both calibrated camera overlays and the simulated
separated hemispheres, or `orientation_3d.html` for interactive orientation replay.
`results.csv` contains the continuous best estimate; `strict_results.csv` omits
prior-dependent shell phases. Both use one shared corrected event timeline.

## Why this implementation

The old single-camera project already models separate hemisphere centers and
has useful offline tracking/refinement. The dual-camera project had two different
paths: a metric paired-frame solver and a shared Kalman tracker whose default
rotation observation approximated the assembly as an enclosing sphere. Pairing
at 12 ms dropped approximately 29% of cam0 frames in this recording and could
create artificial paired-data gaps of 1.55 seconds. The normal paired solver also
limited its input to the first 180 pairs.

The new offline path keeps every image at its native time. Camera-local material
tracks feed one calibrated projection model with a 100 mm shell radius, 20 mm
rim-plane gap, shared roll and independent shell spins. It fits trajectory knots,
material points, initial roll and relative camera timing against both views.
The causal filter, red/green separator and conditional azimuth fit supply useful
initializations; the final numeric trajectory comes from the full pixel bundle.
See [the model and implementation details](JOINT_OFFLINE.md).

## Measured processing results

| Quantity | Result |
|---|---:|
| Native cam0 / C920 images | 695 |
| Native cam1 / Brio 101 images | 728 |
| Total native images | 1,423 |
| Corrected recording span | 24.310 s |
| Collected feature observations | 180,625 |
| Persistent observations retained by final bundle | 163,279 |
| Accepted pixel observations | 83,516 |
| Accepted pixel RMSE, C920 / Brio | 1.324 / 1.361 px |
| Fitted initial roll, approximate 8-degree prior | 6.644 degrees |
| Additional Brio clock correction | +6.139 ms |
| Roll with connected visual support | 100% of native events |
| Red spin with strict starting-phase support | 78.57% |
| Green spin with strict starting-phase support | 46.31% |
| Local relative-spin evidence, red / green | 99.30% / 99.93% |
| Events with a finite best estimate | 100% |

The second camera contributes directly to the same state. In the strict bundle
diagnostics, Brio supplies nearby support when C920 has none at 48 roll events,
58 red-spin events and 15 green-spin events. These are support counts, not an
ablation experiment or proof of a specific accuracy improvement.

The initial two-view rim check was rejected because its edge residuals were too
large. The supplied positive 8-degree estimate was therefore retained as a finite
prior, and the bundle fit moved it to 6.644 degrees. `relative_angles` removes that
fitted starting roll; the camera transforms themselves were not rotated or changed.

## What the amber phases mean

Strict pixel support first becomes insufficient at approximately 11.168 s for
green and 18.998 s for red. Angular evidence confirms later loss of connection to
the starting phase. Actual local support gaps are short: green has a 58.7 ms
bracket, and red has four brackets of approximately 53.4-67.5 ms. The motion prior
bridges those intervals in the continuous best estimate. Subsequent local spin
changes remain observed, but their accumulated phase depends on those bridges.

The replay and main CSV explicitly mark affected phases `phase_estimated`; the
strict export keeps them blank. Whole-turn validity is false for both shell spins
at the end. The roll remains supported. A continuous simulation is therefore
available without silently treating interpolated phase as a measured quantity.

## Remaining physical accuracy limits

Low pixel residual is image/model consistency, not a bound on angular accuracy.
The nominal zero-home roll calibration clip fitted an initial +6.91 degrees when
allowed to adjust freely. That reveals remaining absolute-reference/model bias.
In the same nominal pure-roll clip, the full bundle showed about 0.63 degrees of
spurious red spin and 0.76 degrees of green spin. The conditional azimuth method
alone increased red cross-talk to 2.58 degrees, so it was not substituted for the
final pixel-bundle spin trajectory.

Intrinsic capture profiles and actual exposure synchronization are unverified.
The fitted time offset is conditional on geometry and image localization, not
an independent latency measurement. Existing calibration files were preserved.
The model assumes the supplied radius, gap, axes and fixed pivot. It estimates
orientation and the implied shell centers relative to the mounted camera rig;
it cannot recover robot/world translation from these recordings alone.
See [the independent estimator review](ESTIMATOR_LIMITATIONS.md).

## Reproduce

From `ball_caster_dual_cam`, use a new output directory:

```powershell
python scripts/analyze_joint.py --session data/run --initial-roll-deg 8 --cache out/joint_native_cache --output out/my_joint_run --render
```

To regenerate exports from the already fitted bundle without repeating the
nonlinear solve:

```powershell
python scripts/analyze_joint.py --session data/run --cache out/joint_native_cache --finalize-from out/joint_run_final/results.json --output out/my_replay --render
```

The second command checks video/configuration provenance, calibration hashes,
axes/pivot and saved parameter interpretation before reusing the fit. All caches,
trajectory knots, diagnostics and strict estimates remain available for review.

## Verification

The repository suite passed 222 tests and 89 subtests; its Windows symlink-privilege
test was excluded. Four additional final-export tests passed, covering short and
long gaps, phase validity, corrected timestamp alignment and input preservation.
The analytic sparse projection Jacobian was checked against finite differences;
synthetic stereo tests cover missing-camera observations, initial tilt and a known
23 ms timing shift. The recovered synthetic shift was approximately 22.87 ms.

Final export checks confirmed 1,423 distinct source-image events, ordered corrected
times, unit quaternions, null strict estimates for every prior-dependent phase,
and consistent validity flags. The 1600 by 900 H.264 replay contains 730 frames at
30 fps (24.333 seconds, 19.02 MB). A full decode completed without errors, and the
start, middle and ending overlays were visually inspected. The external yoke is
not part of the rendered occlusion model; diagnostic wires may cross it.
