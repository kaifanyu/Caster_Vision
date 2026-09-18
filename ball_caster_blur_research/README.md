# Recovering fast swivel motion under blur

Research date: **2026-09-18**. The accepted `ball_caster_dual_cam` pipeline and its
completed `out/orientation_final` results are preserved. This new folder contains
research, reproducible diagnostics and a GPU-tested CoTracker3 implementation.
The implementation and validation plan is in [IMPLEMENTATION.md](IMPLEMENTATION.md);
measured results and outputs are in [RUN_RESULTS.md](RUN_RESULTS.md).

The physical exposure-integrated follow-on is implemented and tested on the
existing recordings: [results and limitations](BLUR_FIT_RESULTS.md),
[image comparisons](experiments/blur_physics/index.html). Separate short-exposure
YAML presets and exact recording commands are in [CAMERA_SETTINGS.md](CAMERA_SETTINGS.md).

**Experimental implementation:** CoTracker3 offline now supplies observations
inside the existing physical model, with one-camera, two-camera and hybrid
comparisons. See the measured acceptance decision in `RUN_RESULTS.md`. For the most
targeted software experiment, the new fitter compares **exposure-integrated images
of the known two hemispheres**. For future recordings, shorter exposure with stronger lighting is
the first acquisition change to test. None of these choices should silently
convert uncertain phase into measured angle or certify missing turns.

## What the recordings reveal

The run's actual saved camera-control readbacks agree with the requested exposure:
**C920 15.6 ms; Brio 101 8 ms**. These are integration settings, not independently
recorded exposure start/midpoint timestamps. Both streams operate around 30 Hz.
The original frames are 1920 by 1080; the approximate ball image radius is several
hundred pixels.

Using native same-track pixel displacement divided by timestamp separation and
multiplied by the corresponding exposure gives this diagnostic around **11.0-11.6 s**:

| Surface / camera | Approximate blur, median | Approximate blur, 95th percentile |
|---|---:|---:|
| Green / C920 | 20.5 px | 34.6 px |
| Green / Brio 101 | 11.9 px | 17.9 px |
| Red / C920 | 11.0 px | 22.5 px |
| Red / Brio 101 | 6.4 px | 13.8 px |

These are **motion-based approximations from surviving tracks**, not direct point
spread-function measurements. They combine projected roll and shell spin, assume
locally representative motion, and omit tracks already lost to blur. They support
testing exposure-aware estimation and shorter exposures; they do not prove blur
caused every failure. Yoke occlusion, grazing-angle geometry, repeated paint and
calibration/timing errors can coexist.

Inspection of the [11-second frame crops](diagnostics/output/contact_11s.jpg)
shows green paint streaking in C920 while much of that shell is hidden in Brio.
Thus the second camera may have a sharp image of the other shell without supplying
the missing green-spin evidence. Camera placement/coverage is worth checking
alongside exposure, particularly at these poses. Additional contact sheets cover
[19 seconds](diagnostics/output/contact_19s.jpg) and
[21 seconds](diagnostics/output/contact_21s.jpg).

The existing best estimate reaches approximately 250 deg/s shared roll, 316 deg/s
red spin and 417 deg/s green spin. The two spin peaks occur in `phase_estimated`
segments and are **not independently measured peak speeds**. Do not tune a new
tracker to reproduce those peaks as ground truth. Full numeric diagnostics are in
[core_diagnostics.json](diagnostics/output/core_diagnostics.json) and
[native_motion_by_frame.csv](diagnostics/output/native_motion_by_frame.csv).
The complete [recording audit](diagnostics/output/blur_diagnostics.json) also records
frame pacing and image sharpness proxies. C920 has 32 receive-time intervals over
50 ms, with a maximum near 100 ms; Brio has none over 50 ms. Host timing alone does
not prove exactly how many frames the sensor dropped.

## Should you use CoTracker?

**Yes, as a controlled experiment.** A temporally learned point tracker can use
other surface points and sharper frames before/after a difficult interval to
maintain or reacquire a material identity. Offline processing is well suited to
this saved-video task. It supplies 2D tracks; the calibrated metric backend must
still infer common roll and independent shell spin. This recommendation is an
inference about the task, not a published result on these recordings.
[Official CoTracker repository](https://github.com/facebookresearch/co-tracker),
[CoTracker3 paper](https://arxiv.org/abs/2410.11831).

Two practical traps matter here. CoTracker's default model resolution is only
384 by 512, so resizing the whole room can discard useful paint detail. Crop the
caster before inference, preserve the coordinate transform, and refine locations
at native resolution where possible. Also, visibility/confidence outputs are
learned scores, not calibrated angular uncertainty; the usual offline wrapper
does not return every underlying confidence value. Hidden-point predictions
cannot be counted as visible measurements. See the
[source-level tracker review](research/POINT_TRACKERS.md).

The workstation has an **RTX 4060 with about 8 GiB VRAM**. A full native-resolution
float32 video tensor alone requires roughly 16-17 GiB per camera. Start with short
overlapping cropped clips, one camera at a time. Actual model memory and runtime
must be measured; this hardware check does not guarantee a given configuration
fits. Do not merge the cameras into one artificial temporal video.

## Which other methods are worth considering?

| Approach | Best role in this project | Why it is not a complete solution by itself |
|---|---|---|
| **CoTracker3 offline** | First learned point-tracking experiment for material identity through difficult intervals | Predictions through invisible texture need verification; no direct physical angle/rate output |
| **TAPIR / BootsTAPIR**, with TAPNext as a current comparison | Independent reacquisition baseline using a different point-tracking architecture | Repeated paint and occluding boundaries remain difficult; checkpoints and protocols must be compared consistently |
| **SEA-RAFT / WAFT** | Distributed short-interval image motion to constrain local angular velocity | Dense flow may be coherently wrong under blur and does not establish persistent material phase |
| **Motion-from-Blur / MBA-VO principles** | Fit motion during the exposure directly using the known geometry | Requires a specialized renderer/texture model; their released pipelines are not drop-in caster solvers |
| **RVRT / NAFNet** | Optional deblurring ablation before tracking | Sharper reconstructed texture may be displaced or inferred; restoration quality is not angular accuracy |
| **Event-camera methods** | Future high-speed acquisition when conventional exposures lose useful texture | Requires actual event recordings and a shell-motion model; existing AVI cannot supply unrecorded event timing |

Primary implementations: [TAP family](https://github.com/google-deepmind/tapnet),
[SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT),
[WAFT](https://github.com/princeton-vl/WAFT),
[Motion-from-Blur](https://github.com/rozumden/MotionFromBlur),
[MBA-VO](https://github.com/ethliup/MBA-VO),
[RVRT](https://github.com/JingyunLiang/RVRT),
[NAFNet](https://github.com/megvii-research/NAFNet),
[event alignment](https://github.com/Haram-kim/Globally_Aligned_Events).
The detailed [method review](research/BLUR_METHODS.md) separates published evidence,
model mismatches and proposed adaptations, including relevant 2025-2026 work.

## The targeted change: let the blur constrain motion

The current pixel bundle treats a frame approximately as an instantaneous
projection. During a fast exposure that assumption is imperfect. A more physical
model averages rendered appearance along the shell trajectory during the measured
exposure, then compares that average with the original image. Motion-from-Blur
demonstrates this principle for moving objects; MBA-VO provides another direct
alignment reference for camera motion.
[Motion-from-Blur paper](https://openaccess.thecvf.com/content/CVPR2022/html/Rozumnyi_Motion-From-Blur_3D_Shape_and_Motion_Estimation_of_Motion-Blurred_Objects_in_CVPR_2022_paper.html).

Here the known radius, gap and calibrated cameras simplify the unknown motion to
three angle curves. Use learned tracks for initialization, an observed paint
texture atlas from sharper images, robust masks for the yoke and unreliable
surfaces, and exposure integration in each camera. Optimize the continuous angle
curves and obtain rate from their analytic derivatives. Keep exposure timing and
possible rolling-shutter effects distinct. This is the proposed specialized
implementation, not a capability already tested in this folder.

It can exploit information in the blur instead of requiring a sharp corner in
every frame. It still cannot guarantee a unique solution when texture is averaged
away, repeated, hidden, or consistent with different turn/direction hypotheses.

## Angular velocity and accumulated phase need different flags

The accepted run has local angular evidence at about 99.3% of red and 99.9% of
green native events, despite losing the initial phase connection after brief
gaps. A later segment can constrain `beta(t) + unknown_constant`; its derivative
does not depend on that constant. Therefore amber phase estimates do **not** mean
all subsequent rates are unobserved. Rates inside the gap, peak rates suppressed
by smoothing and sub-exposure accelerations require separate assessment.

The next experiment should export **rate evidence, phase validity and whole-turn
validity separately**, with the time window/bandwidth behind each rate estimate.
Use an independently synchronized encoder or high-speed reference to validate
real peak speed and timing. Neither the earlier reconstructed curve nor an AI
interpolated video supplies that ground truth.

## Future capture: the most direct reduction of blur

With a representative 500-pixel lever arm at 360 deg/s, the simple tangent-motion
calculation predicts about **49 px blur at 15.6 ms**, **25 px at 8 ms**, and
**3.1 px at 1 ms**. A two-pixel budget gives about **0.64 ms**. These are planning
examples, not measured motion during the uncertain interval. The exact value
depends on the calibrated projection and both angular rates.

Try a controlled **2, 1 and 0.5 ms exposure sweep**, if supported, with brighter
diffuse lighting and actual control readback. Target roughly one or two pixels
of motion without clipping the shells or losing paint contrast. More FPS reduces
inter-frame displacement; shorter exposure reduces intra-frame blur. For hardware
replacement, assess synchronized global-shutter capture at 120-240 Hz against the
actual speed and marker pattern. Global shutter alone does not cure long-exposure
blur. [Basler exposure guidance](https://docs.baslerweb.com/optimizing-image-quality),
[shutter explanation](https://docs.baslerweb.com/electronic-shutter-types).

The difficult interval's surviving-track 95th-percentile image speeds are around
2,200 px/s in each view. A two-pixel budget at that sampled speed implies about
0.9 ms exposure (one pixel about 0.45 ms). This gives a recording-specific starting
point for the sweep, with the same survivor-bias and constant-speed limitations.

More unique, nonrepeating surface marks and measured camera timing would help
reconnect phase. An encoder is especially valuable if accurate swivel rate is the
primary measurement. Detailed formulas, sampling ambiguity and capture steps are
in [capture and observability](research/CAPTURE_AND_OBSERVABILITY.md).

## Work prepared in this folder

- [Concrete experiment plan](EXPERIMENT_PLAN.md): controlled tracker comparisons,
  continuous-time velocity, exposure-integrated fitting and acceptance metrics.
- [Point-tracker research](research/POINT_TRACKERS.md): APIs, confidence semantics,
  coordinate transforms, checkpoints and licenses.
- [Blur-method research](research/BLUR_METHODS.md): relevant papers/repositories and
  their limits for this mechanism.
- [Exposure calculations](reports/exposure_budget.csv), regenerated by
  `python scripts/exposure_budget.py`.
- [Baseline snapshot manifest](baseline/manifest.json), verified by
  `python scripts/preserve_baseline.py --verify`.
- [Read-only recording diagnostic script](diagnostics/analyze_recordings.py), which
  writes new diagnostics here without changing the old pipeline.

The immediate implementation choice is **CoTracker3 offline plus the existing
metric backend on short failure windows**, with raw dense flow as a comparison.
The main accuracy investment is the exposure-integrated physical fit. Improvements
must be measured in angular/rate error and correctly retained uncertainty, not
just a smoother overlay or more tracks.
