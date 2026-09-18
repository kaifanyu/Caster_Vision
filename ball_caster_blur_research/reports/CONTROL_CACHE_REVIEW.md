# Read-only review of available control runs

The accepted pipeline contains complete native conventional-track caches for the two control clips:

- `../ball_caster_dual_cam/out/roll_native_cache/native_tracks.npz`: 344 C920 frames, 359 Brio frames, about 11.97 seconds, 111,333 observations. Source videos are `data/roll/{c920,brio101}.avi`.
- `../ball_caster_dual_cam/out/swivel_native_cache/native_tracks.npz`: 344 C920 frames, 359 Brio frames, about 11.97 seconds, 111,520 observations. Source videos are `data/swivel/{c920,brio101}.avi`.

The fitted geometry is two 0.100 m curvature-radius hemispheres separated by a 0.020 m rim-plane gap, red on local +z. The configuration explicitly uses original, unrotated calibrated 1920x1080 source images. Both control fits report all three motion components free, which makes unintended cross-component motion a useful consistency diagnostic. Neither control provides independent angle ground truth.

The latest roll control is `out/roll_validation_final/` in the accepted pipeline. `report.json` identifies the current NativeBundle with free initial roll, fitted Brio offset, and image-cue reseeding. It reports 41.746 degrees roll excursion; red/green spin ranges 0.628/1.419 degrees; maximum absolute spin 0.628/0.761 degrees; median pixel residual 1.127 px; 75.1% observations visible and within 3 px. The first 2.5 seconds span 0.172/0.252/0.363 degrees for roll/red/green. The fitted initial roll is 6.905 degrees and Brio offset is +27.570 ms, so absolute home and exposure synchronization are not independently validated by this control.

`roll_validation_final/bundle.npz` is an older export schema: `knots` has 361 samples but `angles` has 703 native events. It is **not** directly compatible with a consumer expecting one angle row per knot. It retains full optimized vector `x`, landmark keys, source-row mapping, residuals, visibility, offset, and a matching `seed.npz`; rebuild the matching NativeBundle and call `unpack(x)` to recover knot-angle arrays, checking parameter/key identity first. Avoid silently pairing its 703 event angles with 361 knots. The `roll_validation_bundle/` and `roll_validation_seed/bundle.npz` artifacts predate the current global-tilt/timing layout and should not be loaded as current parameter vectors.

`out/swivel_validation_seed/` contains `seed.npz`, `angles.csv`, and `report.json`, but no final batch bundle. Its free seed trajectory reports roll/red/green ranges of 0.154/58.431/0.118 degrees and median successful visual-update RMSE 1.417 px. These are seed-level results, not a like-for-like final batch control. Use this cache/seed to run a fresh unconstrained batch fit before comparing it with current learned-tracker branches.

The control files are suitable for stationary and cross-component consistency checks and for rebuilding new experimental branches under the same calibration. Their fitted trajectories must not be called measured ground truth or used as independent accuracy targets. Matching cache/calibration provenance and native time conventions is required before reuse.
