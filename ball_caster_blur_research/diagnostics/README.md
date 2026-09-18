# Recording blur diagnostics

Run from the new research folder:

```powershell
python diagnostics/analyze_recordings.py
```

The script reads the neighboring `ball_caster_dual_cam` recordings, native track cache, calibration and final results. It uses existing NumPy, OpenCV and PyYAML dependencies. Its default writes are confined to `diagnostics/output`; it does not edit the baseline. `blur_diagnostics.json` records input SHA256 hashes and a reproduction command.

## Recorded acquisition

Both streams are 1920 x 1080. Camera-control readbacks before streaming, after warmup and at completion agree: C920 exposure is **15.6 ms**, Brio 101 exposure is **8 ms**. These values come from the recorded controls, not merely current configuration. Exposure start/midpoint times were not measured.

The run contains 695 C920 frames at an observed mean 28.56 Hz and 728 Brio frames at 29.93 Hz. C920 has 32 receive intervals above 50 ms, reaching 100.06 ms; Brio's longest is 39.44 ms. The configured undistorted enclosing radii are 445.6 px and 513.9 px. Original receive timestamps, rather than AVI nominal frame rate, are used for displacement velocities.

## Approximate blur from native tracked displacements

These values are native undistorted KLT displacement divided by its recorded time interval, multiplied by the actual exposure setting. They describe surviving tracks, not measured blur kernels or upper bounds on missed motion.

| Interval | Shell | C920 median / p95 blur | Brio median / p95 blur |
|---|---|---:|---:|
| 11.0–11.6 s | Red | 11.0 / 22.5 px | 6.4 / 13.8 px |
| 11.0–11.6 s | Green | 20.5 / 34.6 px | 11.9 / 17.9 px |
| 18.7–19.5 s | Red | 12.2 / 20.2 px | 9.6 / 14.9 px |
| 18.7–19.5 s | Green | 2.2 / 17.0 px | 0.5 / 7.9 px |
| 21.0–21.4 s | Red | 20.9 / 30.9 px | 10.1 / 16.3 px |
| 21.0–21.4 s | Green | 9.8 / 20.5 px | 6.1 / 16.2 px |

The worst targeted p95 tracked speed is roughly 2,200 pixels/s in both cameras. At that speed, approximately 0.9 ms exposure corresponds to two pixels of motion. This is a planning estimate under constant velocity, not a guarantee: the fastest blurred/occluded tracks may already be absent.

`contact_11s.jpg`, `contact_19s.jpg` and `contact_21s.jpg` show each camera at a stationary reference and four target times. Full-ball panels are resized, while the 240 x 240 texture crops retain native scale. C920 green markings visibly streak around 11.2–11.6 s while the other view has limited green-shell visibility. Near 19.2 s, Brio red markings blur while its green shell remains relatively sharp. These examples show why visibility and blur should be analyzed together.

Shell-mask Laplacian variance falls to approximately 3–17% of its stationary reference in the selected C920 windows, versus 38–83% for Brio. These are sharpness proxies only: shell pose, paint density, mask selection, noise and illumination also change them. Absolute sharpness values are not equivalent between cameras.

## Rates and phase ambiguity

The existing fitted trajectory estimates peak magnitudes of about 250 degrees/s roll, 316 degrees/s red spin and 417 degrees/s green spin. The latter two peaks have `phase_estimated` status. These rates are **model estimates, not ground truth**. Peaks within the earlier strictly phase-supported subsets are approximately 298 degrees/s red and 220 degrees/s green.

`phase_estimated` means accumulated phase/turn ambiguity, not that every later local velocity is unobserved. Unknown constant phase cancels under differentiation. Existing conditional local angular support covers 99.30% of red and 99.93% of green events, whereas phase anchored to the start covers much less. The short local-evidence gaps near 11.45, 19.21–19.35 and 21.23 s can create persistent phase uncertainty even after clear images return. Subsequent research should export separate `rate_status`, `phase_valid` and `turn_count_valid` fields and test derivative observability within each local component.

Use `native_motion_by_frame.csv` for native displacement/blur summaries, `targeted_sharpness.csv` for sampled sharpness proxies, and `blur_diagnostics.json` for the complete numerical audit, conditional rate peaks, evidence gaps and limitations. `core_diagnostics.json` is the quick pre-image pass; the complete JSON is authoritative.
