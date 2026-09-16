# Why the 557-frame run has no moving axes

Reviewed September 16, 2026, using `out/real/results.json`,
`mechanical_report.json`, the saved observation archive, and the source video.

## Immediate cause: mechanical solver did not converge

The run recorded `solver_did_not_converge`, solver status 0, and
`The maximum number of function evaluations is exceeded.` The configured
limit was 40. This solver exits before per-frame acceptance checks when it
does not converge, leaving both shells with **0/557 valid constrained poses**.
The renderer therefore hides their moving grids. Feature detection did run.

A separate refit with **200 evaluations** and unchanged quality gates also
failed to converge. Its outputs are under
`out/run_failure_audit/refit_more_iterations/`. Increasing the limit alone is
not a verified solution. Do not relax reprojection/inlier thresholds to make
the overlay appear, or treat unaccepted optimizer iterates as measurements.

## Tracking also lost its absolute reference

Before mechanical fitting, temporal tracking retained 131/557 top poses and
426/557 bottom poses. The independent offline pass refined neither shell;
both connected fits reported `invalid_frame_initialization_failed`.

| Shell | Start of persistent loss | Failed adjacent estimate |
|---|---|---|
| Top | Frame 133, 4.688 s | 57.3% inlier agreement; 0.574-degree mean residual |
| Bottom | Frame 438, 15.312 s | 0.632-degree mean residual |

Targets were at least 70% agreement and at most 0.5-degree residual.
Subsequent frame pairs often tracked well, but they could not establish the
missing rotation relative to the last trusted image. Direct recovery matches
failed; the last top reference at frame 132 expired after frame 177, and the
last bottom reference at frame 437 expired after frame 482. This explains why
later visible markings do not automatically restore absolute angles.

Longer keyframe age alone is insufficient: nearby recovery already failed
before those references expired. Offline recovery needs reliable image
connections across the gap, or separately reported local motion whose unknown
global offset remains explicit.

## The masks were present, with some remaining contamination

At frames 132–135 the processed masks still contained 698–920 detectable top
corners and 759–846 bottom corners. Frame 438 had 666 top and 526 bottom
corners. The earlier large false mask on the upper shell opening was largely
removed by the new HSV bounds.

Some bottom tracks still lie on the exposed inner disc instead of the modeled
outer sphere. See:

- [Source, actual masks, and observations](../out/run_failure_audit/mask_contact_sheet_selected.jpg)
- [Incorrect inner-disc observations](../out/run_failure_audit/inner_disc_false_tracks_frame0.jpg)

A mask-only experiment raising the bottom saturation minimum from 8 to 30
removed fresh corners in that disc region at frames 0 and 3, retaining 583/603
and 488/503 bottom corners respectively. Across 13 sampled frames, at least
488 bottom corners remained. This is a candidate for preview and retracking,
not measured improvement in angle accuracy. The main HSV config was preserved.

## Corrections and next steps

The observation selector was also discarding useful reverse/keyframe evidence
between every third frame. In bottom frame 1, for example, the archive had
151 reverse observations, but the old selection retained none: their matching
reference-image observations had lost the per-frame budget competition.
The new selector admits observations of a landmark together, keeps the
adjacent-track reservation, balances spatial coverage, and refills available
capacity without dropping the minimum track-length requirement.

With the same 100-observation cap, 2-pixel threshold, and 70% support gate, a
bounded fit of frames 0–119 improved from **34 top / 7 bottom** accepted poses
to **44 top / 38 bottom**. Bottom frame 1 improved from 8/23 inliers to 80/100.
The new fit converged in 32 evaluations. This compares acceptance coverage on
the same saved images; the selected observations changed, and this is not an
absolute angular-accuracy validation. The full 557-frame fit still exceeded
40 evaluations after this correction. All **284 automated tests passed**.

The final full-clip test combined the selector fix with a 200-evaluation budget.
It converged after **91 evaluations**, accepting **40/557 top poses and 36/557
bottom poses**. Results and diagnostics are under
`out/run_failure_audit/refit_coherent_more_iterations/`. The main config now
sets `offline.max_nfev: 200`; the HSV thresholds and physical gap were preserved.
The original failed `out/real` outputs were preserved. This restores some
supported poses but leaves 517 top and 521 bottom poses unresolved, so it is
still not a complete research measurement trajectory.

The HSV sampler now previews the actual pipeline masks, including growth,
shell separation, yoke exclusion, and the temporal boundary margin. Before
this fix it showed raw color thresholds. It also now honors `--hue-margin`
and `--sv-margin` in the preview, matching the saved ranges. Unsampled classes
show their configured ranges. Review via:

```powershell
.\.venv\Scripts\python.exe .\scripts\inspect_hsv.py --config .\config.yaml --frames 12 --write
```

Sample colored markings across different poses and lighting, excluding white
interior rims, holes, and the yoke. Keep enough spatially distributed marks;
corner count alone is not an accuracy metric.

The run and saved-refit commands now print the mechanical solver status and
accepted pose counts explicitly, distinguishing empty final results from
empty masks. The 20 mm gap / 200 mm diameter setting remains `0.10`.

The gap model currently describes two spherical caps with one common sphere
center. If the hardware instead consists of two complete hemispheres spaced
apart, their sphere centers are different and move with shared roll. That
requires a different image projection model; the gap ratio alone does not
specify this distinction. Confirm the physical construction before treating
the model as calibrated geometry.

The remaining algorithm work concerns isolating difficult intervals in the
long joint fit and recovering
the reference across tracking gaps. Sharper footage with smaller inter-frame
motion also helps image matching. Validation needs both supported coverage and
an independent motion reference before quantitative research conclusions.
