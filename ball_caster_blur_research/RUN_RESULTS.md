# CoTracker3 offline: implementation and measured results

Follow-on completed: the [physical blur-fit report](BLUR_FIT_RESULTS.md) adds
12 real-video GPU trials, independent synthetic validation, measured texture
atlases and explicit ambiguity checks. It did not justify replacing this
trajectory. [Camera settings](CAMERA_SETTINGS.md) provides the separate exposure
presets and exact capture commands.

**Decision: use both cameras in the physical fit, retain conventional KLT as the
primary observation source, and keep CoTracker as an experimental supplement.**
The implementation works on this computer's GPU, but these tests do not show a
material improvement in real angular-velocity accuracy. CoTracker alone is less
reliable on this run than the accepted pipeline. The original pipeline and its
accepted outputs remain unchanged.

## Open the delivered outputs

- [Hybrid combined replay: both cameras and one simulated caster](experiments/hybrid_dual/replay/combined_tracking.mp4)
- [Hybrid interactive 3D orientation](experiments/hybrid_dual/orientation_3d.html)
- [Hybrid pose CSV](experiments/hybrid_dual/results.csv)
- [Angular rates, separate rate/phase/turn validity](experiments/hybrid_dual/angular_rates.csv)
- [CoTracker-only 3D result](experiments/cotracker_dual/orientation_3d.html)
- [Comparison tables](reports/comparison.md) and [comparison plots](reports/comparison.png)
- [Known-motion GPU tests](experiments/gpu_validation_rig_gap/REPORT.md)
- [Roll/swivel calibration-motion controls](experiments/controls/COMPARISON.md)
- [Implementation, validation plan, and reproduction commands](IMPLEMENTATION.md)

The combined replay was decoded completely: 730 frames, 1600x900, 30 fps,
24.333 seconds. Its underlying corrected recording interval is 24.310 seconds.
The display marks estimated phase explicitly. The drawn meridians illustrate
the model; they are not newly detected paint features.

## Actual GPU execution

Official CoTracker3 **scaled offline**, commit
`82e02e8029753ad4ef13cf06be7f4fc5facdda4d`, ran under PyTorch 2.5.1+cu121 on the
**NVIDIA GeForce RTX 4060**. Checkpoint SHA-256 and exact installed versions are
saved in [model_manifest.json](model_manifest.json) and
[environment.json](reports/environment.json).

The full run contains 695 C920 and 728 Brio native frames. Thirty-five overlapping
offline windows produced **127,158 spatial-deduplicated candidate observations**.
Model inference totaled **115.34 seconds**; this excludes decoding, undistortion,
mask generation, physical optimization and export. A standard 60-frame,
192-query window used **3.00 GB allocated / 4.28 GB reserved CUDA memory**.
Camera/window jobs run serially. The sparse calibrated pose fit runs on the CPU.

Both calibration recordings were also processed on the GPU: **77,424** candidate
observations for roll and **74,920** for swivel. All six control fits converged.
No pretrained weights were updated, and no global packages were installed.

## Integration with the gap and independent shells

Both cameras constrain the same roll angle and two independent spin angles.
The model retains the **100 mm radius, 20 mm gap, displaced centers, calibrated
intrinsics/extrinsics, native frame times and initial tilt prior**. Learned
features are camera-local material landmarks; they do not need to be matched
between views. A robust physical fit rejects many neural correspondences.

The hybrid retains nearby KLT observations instead of double-counting the same
pixel with CoTracker. It fitted an initial roll of **6.1925 degrees** from the
supplied approximate 8-degree prior. This is a calibrated-model estimate,
not independent evidence that the physical tilt was exactly 6.1925 degrees.
The accepted additional Brio timing correction, **+6.1387 ms**, is held fixed.

The hybrid retained 231,306 observations before final physical gates and accepted
103,357 after the 3-pixel/persistence/visibility checks. Its accepted-pixel RMSE is
1.366 px C920 and 1.374 px Brio. Those are **training residuals on retained pixels**,
so they are not the principal accuracy comparison.

Phase support improves only slightly over the accepted result: red has 1,119
home/vision events versus 1,118 previously; green has 661 versus 659. The remaining
304 red and 762 green events depend on phase estimates across gaps. Whole-turn
validity remains false after the corresponding ambiguity. A continuous animated
trajectory does not mean every displayed orientation is measured.

## Does CoTracker improve this recording?

The same reserved conventional tracks provide a conditional later-pixel check.
Whole reference tracks are excluded from those fits; a learned identity is
excluded completely if any of its observations overlaps a reference neighborhood.

| Dual-camera observations | Median error | p95 error | Within 3 px, including failures | Scored coverage |
|---|---:|---:|---:|---:|
| KLT | 4.21 px | 24.58 px | 36.6% | 85.4% |
| CoTracker | 4.32 px | 26.96 px | 35.4% | 83.7% |
| KLT + CoTracker | 4.19 px | 25.08 px | 36.6% | 85.3% |

The hybrid's 0.02 px median reduction accompanies a 0.50 px increase in the p95
tail. That is not a persuasive accuracy gain. All branches share the accepted
full-data trajectory as a nonlinear initializer, and all reference/learned
pixels come from the same recordings. These are conditional consistency checks,
not an independent physical accuracy test or statistical-significance result.

The home-motion controls agree with this conclusion. For nominal pure roll,
maximum unintended component motion is 1.042 degrees KLT, 1.011 CoTracker,
0.974 hybrid. For nominal red swivel it is 0.101, 0.086, 0.096 degrees respectively.
CoTracker's own residuals are much lower on these sharper clips, but the common
reference medians change only modestly and some error tails worsen. The manually
performed controls have no encoder reference.

## One camera or two?

**Use both**, with image-quality and physical-consistency gates per shell/view.
The benefit comes from complementary observations, not from treating two model
predictions as independent ground truth. At 21.2 seconds, the inspected Brio
window retains 56 green candidates versus 3 in C920. At 19.2 seconds, both views
lose the blurred red shell: C920 retains 1 red candidate and Brio none. Counts
are before metric gates and do not establish material identity by themselves.
See the [raw image audit](reports/track_preview/VISUAL_AUDIT.md).

The camera-ablation comparison uses each active camera's actual corrected start
time and its first observed image as the spin phase gauge. Inactive-camera events
are excluded from its reported denominator. Earlier common-grid mono trials
remain in `experiments/cotracker_c920` and `cotracker_brio101` for traceability;
their phase support was affected by an unobserved initial gauge and is superseded
by the `*_native` trials in the comparison report. Mono and dual counts are not
directly comparable; inspect fractions and their stated event/time support.

| CoTracker observation source | Red local-rate support | Green local-rate support |
|---|---:|---:|
| C920 only | 21.6% | 18.3% |
| Brio only | 54.3% | 45.3% |
| Both fused | 86.9% | 91.6% |

These are conditional observability fractions for this 30 Hz knot model and
these gates, not camera accuracy scores or fundamental limits of monocular
vision. Native sampling gaps and temporal interpolation rank affect them.
The final optimization stage converged for each comparison. C920-only reached
the first-stage evaluation limit before converging after robust reweighting;
none of these nonlinear fits is a guarantee of a global optimum.

## Angular velocity and the remaining blur limit

The new `angular_rates.csv` separates **local rate support**, **accumulated phase
status**, and **whole-turn validity**. In the hybrid, measurement-supported rates
cover 1,422/1,423 roll events, 1,277 red events and 1,375 green events. Other
intervals are marked predicted or unresolved. Rates can be supported after an
unknown phase offset; this does not repair the missing accumulated rotation.

The derivative is a local cubic diagnostic of the 30 Hz physical pose knots.
It does not model exposure integration, add temporal bandwidth, or validate
instantaneous peak speed. Unsupported predicted derivatives can make large
excursions; use the strict measured-rate columns for analyses that require
measurement support. Even those columns remain conditional on correct tracks,
calibration, carrier roll and timing, and are not confidence intervals.

Known-motion GPU tests use normalized separated hemispheres with the actual
gap/radius ratio, fixed known 8-degree carrier tilt, unique texture, foreground
occlusion, and sharp/8 ms/15.6 ms exposure. At slow sharp motion, point tracking
has 0.61 px median / 2.33 px p95 error. Conditional red/green interval-rate RMSE is
1.96/1.11 deg/s over every eligible interval. At fast sharp motion, rate RMSE is
17.34/8.12 deg/s on only 19/46 and 24/46 eligible intervals. Some high-confidence
tracks are severely wrong after occlusion. These tests use known geometry/home
points and oracle visibility, so their angular errors are not real-run accuracy.

The targeted follow-on experiment fitting exposure-integrated rendered texture
to measured images is now implemented separately; see [its results](BLUR_FIT_RESULTS.md).
It does not change the CoTracker trajectory reported here. Shorter exposure and an independent encoder
or short-exposure reference remain the practical way to test real fast-spin
accuracy rather than infer it from a smooth replay.

At completion of the CoTracker stage, **55 automated tests passed**, including independent synthetic physical pose fits,
monocular timing/phase-gauge recovery, coordinate mappings, rate observability,
and export handling. See [test summary](reports/unit_tests.json).
The baseline preservation check verifies all 97 saved source files and all six
accepted artifacts unchanged. New source, neural caches, controls, comparisons,
rate exports and replay outputs are isolated in this research folder.
