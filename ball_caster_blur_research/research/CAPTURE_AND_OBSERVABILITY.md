# Exposure, sampling and what remains observable

Research checked 2026-09-18. Recommendations below are engineering deductions
for this known mechanism; they are not measured accuracy claims for an untested
tracker. The accepted reconstruction stays in `../ball_caster_dual_cam`.

## Three different problems

1. **Exposure blur:** each pixel integrates radiance while the shell moves. A
   sharper detector may improve localization, but a severely smeared feature
   does not have a single physically correct instantaneous pixel location.
2. **Temporal correspondence and aliasing:** even short exposures can leave a
   large jump between images. Repeated triangles may admit several plausible
   correspondences or missed rotations. More frames help this problem.
3. **Geometry, timing and visibility:** the two cameras can share blur while
   seeing different parts of the surface. Yoke occlusion, grazing-angle points,
   imperfect sphere/axis calibration and unknown exposure timing are additional
   error sources. A low sharpness score alone cannot identify the cause.

Higher frame rate and shorter exposure are distinct controls. Basler recommends
choosing exposure so image motion stays near one pixel; increasing gain also
increases noise. Global shutter removes row-to-row exposure timing differences
but does not freeze motion during a long exposure. Sources:
[image-quality guidance](https://docs.baslerweb.com/optimizing-image-quality),
[shutter types](https://docs.baslerweb.com/electronic-shutter-types).

## Quantitative planning example

For a point with effective image-plane lever arm `r_px`, approximately tangent to
the observed rotation, the small-motion approximation is

```text
image_speed_px_s ~ r_px * abs(omega_rad_s)
blur_length_px   ~ image_speed_px_s * exposure_s
angular_sweep    ~ abs(omega) * exposure_s
exposure_limit   ~ allowed_blur_px / image_speed_px_s
```

This ignores perspective, both joint rates acting together, changing visibility
and acceleration. The correct per-point prediction uses the calibrated projection
Jacobian: `pixel_velocity = J_alpha*alpha_dot + J_beta*beta_dot`. The simple formula
is useful for planning, not a rigorous upper bound or an observed blur measurement.

At a representative 500-pixel lever arm:

| Spin rate | Blur at 15.6 ms | Blur at 8 ms | Blur at 1 ms | Exposure for 2 px |
|---|---:|---:|---:|---:|
| 180 deg/s | 24.5 px | 12.6 px | 1.57 px | 1.27 ms |
| 360 deg/s | 49.0 px | 25.1 px | 3.14 px | 0.637 ms |
| 720 deg/s | 98.0 px | 50.3 px | 6.28 px | 0.318 ms |

These are hypothetical rates, not ground truth from the uncertain fast interval.
The script `scripts/exposure_budget.py` writes the complete planning table.
At 360 deg/s, frame-to-frame rotation is 12 degrees at 30 Hz and 3 degrees at
120 Hz; shortening the frame interval alone does not remove exposure blur.

## Practical next capture

Start a controlled exposure sweep at 2, 1 and 0.5 ms, subject to the cameras'
advertised/actually accepted control range. Add strong diffuse continuous light,
check clipped white shells and paint contrast, and measure the achieved exposure.
Aim for roughly 1-2 pixels of motion at the fastest useful surface points rather
than selecting a shutter speed by appearance alone. The image-speed diagnostics
in this folder help choose a first setting, but failed tracks are absent from
their speed distribution and can understate the fastest motion.

Keep camera pose, resolution, focus and zoom consistent with calibration. Disable
automatic exposure/focus changes for repeatability; retain actual control readback
and timestamps. With the existing Linux recorder, `exposure_us` is in microseconds;
the V4L2 absolute-exposure control uses 100-microsecond units. Thus a requested
1 ms corresponds to 10 control units where that mode is supported. This is not
a claim that every webcam accepts that setting.
[Linux camera control reference](https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html).

The original C920's official specifications describe 1080p at 30 fps. Do not
assume the Brio 101 has the capabilities of a different Brio model, or that
changing a software FPS field creates a high-speed stream. Query device formats
and verify unique captured frames. For a future camera change, evaluate global
shutter, hardware triggering, shorter exposure and 120-240 Hz as requirements;
the necessary rate depends on actual speed, marker distinctiveness and accuracy.
[C920 specifications](https://support.logi.com/hc/en-us/articles/360023307294-C920-Technical-Specifications).

Measure a common timing event in both views, ideally with exposure-level hardware
synchronization. A fitted constant camera offset cannot establish exposure start,
midpoint or rolling-shutter line delay independently. Row timing should be measured
before adding a rolling-shutter parameter to a metrology fit.

Use nonrepeating, well-spaced high-contrast markers with some uniquely identifiable
landmarks across each shell. Markers should be large enough to survive expected
blur but numerous enough to constrain pose. Identifiable landmarks can reconnect
phase after gaps; generic repeated triangles make that harder. An encoder on the
mechanical swivel is the strongest direct validation of swivel angle/rate if the
mechanism permits it. It does not measure two independently spinning shells unless
those rotations are instrumented separately.

## Limits that AI cannot remove without extra assumptions

If the visible texture repeats exactly `m` times per revolution, phase is ambiguous
modulo `2*pi/m`. For a nearest-phase unwrap with uniform sampling `f`, a sufficient
small-step condition is `abs(omega) < pi*f/m`; it is not a universal speed limit
for all patterned spheres. Unique markers, nonuniform timestamps and additional
motion/appearance constraints can change the ambiguity. A gap can still leave
multiple whole-turn hypotheses consistent with the images.

Exposure integration is a temporal low-pass operation. For constant-speed motion,
a texture harmonic of order `m` has an idealized attenuation proportional to
`sinc(m*omega*exposure/2)`. At its zeros that harmonic carries no measured phase.
A single symmetric exposure may also admit reversed motion with the same blur;
neighboring frames or another measurement is needed to choose direction. Stereo
helps when one view retains useful evidence; it does not guarantee observability
when both views lose the relevant texture information.

Consequently, learned occluded-point predictions and interpolated video frames
are hypotheses. They can be useful initializations but must not count as new
independent camera measurements or certify missing whole turns. Confidence must
refer to the physical angle/rate inference, not only a network visibility score.
