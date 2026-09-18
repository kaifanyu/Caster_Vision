# What the recovered orientation establishes

The two-camera solution provides a shared mechanical roll and separate red/green shell spins. The shell spins are relative to the recording's first painted-feature phases. The calibrated pivot is fixed; these results do not recover arbitrary translation of the caster.

## Conditional spin evidence

Projecting each observed feature ray onto its shell, removing the estimated common roll, and differencing azimuth along a material track is a useful way to estimate shell spin. Differencing eliminates the unknown painted-feature phase. The incidence, polar-distance, hemisphere, forward/backward tracking, residual and spatial-support gates reject important degeneracies. Both cameras can contribute without matching the same paint mark across views.

This estimator uses a weaker image model than the complete fixed-material-point reprojection fit. It tolerates latitude inconsistency while fitting azimuth. The current adjacent-observation latitude gate is 0.12 radians, approximately 6.9 degrees. Consequently an excellent azimuth fit can coexist with poor complete pixel reprojection. Report the component's source as **conditional azimuth evidence**, not as an accepted full metric pixel fit.

A residual below two degrees is an adjacent-edge consistency test. It is not a two-degree bound on absolute or accumulated spin error. Neighboring edges share observations and calibration errors; distinct camera-local track IDs are not statistically independent measurements. The imposed acceleration prior also affects estimates in weakly constrained regions.

## Evidence of remaining bias

The freely fitted `data/roll` calibration check in `out/roll_validation_final` obtained a 1.13-pixel median reprojection error and 75.1% visible observations within three pixels. However, its supplied zero-degree home moved to +6.91 degrees. This is direct evidence that low image residual does not establish absolute home accuracy under the fixed supplied geometry/calibration.

Using that roll trajectory as input to the conditional spin estimator produced a 2.58-degree maximum red-spin excursion and a 2.52-degree final red-spin offset during the nominally pure-roll clip. The complete bundle's red-spin maximum was 0.63 degrees. The conditional estimator's green-spin maximum was 0.91 degrees. Both conditional phases were supported throughout the clip. Thus improved coverage does not automatically improve angular accuracy. These are consistency checks against the intended calibration motion, not measurements from an independent encoder.

The corresponding numerical review is saved in `out/estimator_review/roll_conditional_review.json`. Radius 100 mm, gap 20 mm, fixed pivot, imported intrinsics, stereo transform, and the estimated roll/timing remain conditional inputs to the spin estimator. Neither camera's original intrinsic capture profile nor an independent exposure-timing validation is available.

## Phase observability and missing intervals

An edge connecting interpolated knots supplies one scalar equation. Joining every nonzero knot in a graph is necessary but does not establish that every connected phase is determined. Measurement-only rank or nullspace tests must exclude the acceleration and ridge priors. An individual reported phase is identifiable only when its interpolation row annihilates all remaining measurement null modes after fixing the initial phase.

An independent review of the run found the anchored red component full rank, but the initially graph-connected green component had one exact null mode localized to four knots at 11.286–11.386 seconds. Earlier green phases remained identifiable. Reject affected phase evaluations, rather than rejecting all earlier data or counting smoothing as measurement support. Details are in `out/estimator_review/anchored_rank.json` and `green_null_mode.json`.

After a disconnected interval, a smooth numerical curve does not establish recovered phase or whole revolutions. The final pipeline retains null values in its strict exports. Its continuous best-estimate replay explicitly uses `phase_estimated` after short gaps, with invalid turn-count flags, because local motion can resume while accumulated phase remains dependent on a motion prior. Longer gaps remain unresolved. The final numeric spins use the complete pixel bundle, with the conditional azimuth estimator supplying initialization and independent support diagnostics.

## Interpretation and next validation

Treat current fitted camera offsets as image-model nuisance parameters. Different centroid/separator timing probes gave inconsistent offsets; even calibration and run bundle offsets differ. Motion blur is another source of pixel-model error: at 180 degrees per second the recorded 15.6 ms C920 exposure spans about 2.8 degrees, and the 8 ms Brio exposure about 1.4 degrees.

The most informative next checks are camera-by-camera spin agreement on overlapping supported intervals, sensitivity to plausible roll/pivot/radius/timing perturbations, repeatable returns to known home, and independent angular ground truth. A shared flash or other common timing event would separately constrain exposure alignment. Until those checks are available, describe this output as a conditional reconstruction with explicitly marked gaps, rather than a verified absolute-accuracy trajectory.
