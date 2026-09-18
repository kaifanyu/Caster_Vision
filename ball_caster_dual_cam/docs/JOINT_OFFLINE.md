# One metric orientation trajectory from both cameras

Use the joint offline workflow for the supplied `data/run` recording:

```powershell
python scripts/analyze_joint.py --session data/run --initial-roll-deg 8 --cache out/joint_native_cache --output out/my_joint_run --render
```

Run this from `ball_caster_dual_cam`. C920 is **cam0** and Brio 101 is **cam1**.
The supplied intrinsics and stereo transform stay in their original, unrotated
1920 by 1080 pixel coordinates. Both views contribute observations to the same
roll and two independent shell-spin angles. A second camera is useful when it
can still observe a shell whose marks are blurred or hidden in the first view.

## What was already present, and what changes

| Path | Existing approach and its limits | Joint offline approach |
|---|---|---|
| `../ball_caster_rot` | Temporal tracking, keyframe recovery, offline material-point refinement, mechanical constraints and separate motion filtering for one camera. Its separated-shell geometry is already appropriate. | Reuses the color/KLT observation machinery while placing the two views in one calibrated metric model. |
| `dualcam/solver.py`, `scripts/run.py` | A correct separated-hemisphere bundle fit using selected paired images; normal default is the first 180 pairs. | `dualcam/native_observations.py` retains every native frame and each observation's own timestamp. No pairing gate discards observations. |
| `dualcam/fusion.py`, `scripts/track_motion.py` | One shared Kalman state, but the default `rotation` measurements come from an approximate enclosing sphere. Optional `metric` updates use fixed, camera-local material points. | `dualcam/offline.py` jointly adjusts the trajectory and material points against both views' original pixel observations. The final fit uses the separated centers throughout. |
| Starting pose | Existing workflows require a supplied, exactly fixed initial roll. | `dualcam/initial_pose.py` checks absolute rim evidence. The batch fit uses an approximate initial roll as a finite prior, so image evidence can adjust it. |
| Replay | Earlier native fusion overlays draw axes and the HTML replay shows a separate simulated assembly. | `dualcam/offline_render.py` combines both camera views and one separated-shell simulation, using the common trajectory at the relevant timestamps. |

The Kalman filter supplies a stable forward initialization; the final trajectory
comes from the offline image-space optimization. This is not a Rauch-Tung-Striebel
Kalman smoother. Future and past observations influence shared trajectory knots
through the batch solve and its acceleration prior.

## Geometry and orientation conventions

Let `F = R_bc` map the calibrated home frame into cam0, `C` be the fixed assembly
pivot, `r` the shell curvature radius, and `g` the distance between the hemisphere
rim planes. A material point `p` is a unit vector on its signed hemisphere.
For shell sign `s` and independent spin `beta_s`, the model is:

```text
B(t)       = F @ Rx(alpha(t))
center_s   = C + B(t) @ [0, 0, s*g/2]
X_cam0     = center_s + r*B(t) @ Rz(beta_s(t)) @ p
X_cam1     = R_21 @ X_cam0 + t_21_m
pixel_i    = project(K_i, X_cami)
```

The current configuration uses `r = 0.100 m` and `g = 0.020 m`. In the old single
camera project, `gap_fraction = 0.1` means a half-gap of `r*gap_fraction`; hence
the same physical gap is `2*r*gap_fraction = 0.020 m`. It does not describe
removing caps from a common sphere. The two whole hemispheres have displaced
curvature centers, and those centers move as the shared roll changes.

Roll is common to both shells. Red and green spins remain independent; neither
equal-spin nor no-slip constraints are imposed. `F @ Rx(alpha)` is the caster
frame orientation; shell material orientations additionally use their respective
`Rz(beta)`. A single quaternion cannot encode two independently spinning shells.
The initial painted phase of each shell defines its own zero spin.

Undistortion retains each camera's original intrinsic matrix. Pixel observations
and projections must use that same matrix; normalized coordinates returned by
`undistortPoints` without a projection matrix must not be mistaken for pixels.
See [OpenCV's camera and undistortion reference](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html).

## Observation collection, fusion and refinement

The collector tracks physical red and green markings independently in each camera
with forward/backward Lucas-Kanade checks. Landmark identities are scoped to
camera and shell. Matching an identical triangle between cameras is unnecessary:
the calibrated geometry and common angle trajectory couple the views. A cached
observation file records input/configuration provenance and can be reused when
rerunning the offline solver with unchanged tracking inputs.

The causal seed fits existing metric material points and updates the shared
angle/rate filter. If cam0 loses its track identities, fresh cam1 observations of
roll and the **same shell's spin** can initialize new cam0 points. This transfer
does not count as an additional visual measurement. In the causal mode a recent
prediction from the same camera alone cannot promote new points, and seeing red
does not establish green spin.

The offline workflow additionally permits **optimizer hypotheses**. Adjacent
image correspondences are reprojected using the metric hemispheres at the previous
image pose, giving a local motion guess that can follow acceleration and reversal
without depending entirely on the filter's velocity extrapolation. Its broader
eight-pixel gate and larger acceleration process noise apply to initialization
only; the final batch inlier gate remains three pixels. A hypothesis from an
unanchored previous pose is not an accepted final orientation measurement.

Tracks after a failed seed receive nuisance-point initial guesses so valid future
pixels are not deleted merely because the causal initializer lost its pose.
This does not establish connectivity to the starting reference or turn count.
The batch pixel residuals, geometry, temporal connectivity and component-support
tests must independently accept the final trajectory. Preserving observations
for optimization must never turn a prediction into claimed image evidence.

The batch problem uses an angle-knot trajectory, interpolated at every native
image timestamp, and two surface coordinates per material landmark. Its analytic
sparse Jacobian differentiates the full calibrated perspective projection.
Robust pixel residuals, a finite initial-roll prior and an acceleration prior are
optimized together. Gross mistracks and geometrically hidden surface points are
downweighted between refinement stages. SciPy supplies bounded robust nonlinear
least squares; see the [official `least_squares` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

Two global nuisance parameters refine the initial roll and relative camera time.
Initial roll has a three-degree prior and a bounded 25-degree correction. An
additional Brio time correction has an 80 ms bound and a 50 ms prior; cam0 defines
the clock reference. The exported events are reordered by corrected time, retain
their original receive time and frame index, and share one normalized origin.
The imported timing/calibration files are not overwritten. This motion-based
alignment is conditional on the geometry and feature localization; it does not
make the hardware timing independently verified.

Before the bundle fit, a maximum-margin separator between the physical red and
green feature sets provides a coarse absolute roll hypothesis. Projection uses
both calibrated camera frames, with sign fixed by physical shell identity. A
stable initial hold aligns its bias to the supplied starting estimate. Weak or
contradictory cues are rejected, and interpolation never bridges more than
150 ms. This corrects large accumulated roll drift in the initializer; separator
angles do not enter the final measurement residuals or create validity flags.

Acceleration regularization helps noise and initialization but is a modeling
assumption. A smooth curve is not proof that unobserved motion was measured.
Inspect component support, residuals, turn-count flags and unresolved intervals
in the saved results as well as the replay. No method can determine an independent
shell's missed whole revolutions if neither camera observes sufficient evidence.

After the first supported roll fit, `spin_initialization.py` intersects native
feature rays with each displaced sphere and removes common roll. Same-track
azimuth differences provide a new independent-spin initialization. A second
unrestricted pixel bundle refines all three angles, timing and material points.
The final numeric trajectory remains this full pixel solution: the reduced
azimuth estimator showed greater roll-to-spin cross-talk on the calibration clip.

The azimuth edges also provide a separate measurement-only phase check. Six
distinct camera-local tracks, normal-plane spread, accepted angular residuals,
connection to the starting phase, and QR/nullspace identifiability are required.
Acceleration priors contribute no equations to that support test. Local relative
spin evidence can resume after a brief gap without establishing accumulated phase.

The default final replay retains a continuous best estimate across local support
gaps bracketed by at most 150 ms. Such phases are labeled `phase_estimated`, drawn
in amber and keep invalid whole-turn flags. They remain prior-dependent after
local tracking resumes. Longer unsupported gaps remain null. `strict_results.csv`
and the JSON `strict_angles`/`strict_status` fields exclude these phase estimates.
Thus a completed reconstruction does not imply that every angle is measured or
that absolute physical accuracy has been verified.

## The approximately eight-degree initial tilt

`--initial-roll-deg 8` means **positive eight degrees relative to calibrated
mechanical home**, about the fixed roll axis. It is not an image-plane rotation
or a change to the stereo extrinsics. Reorientation uses matrix composition,
preserving the calibrated roll axis and moving the initial swivel axis correctly.

The optional image diagnostic searches a bounded interval around this seed using
the two projected front rim curves. It removes paint-marker borders from grayscale
edge evidence, compares edge orientations, trims yoke occlusion and checks for a
well-defined minimum and cross-camera agreement. The returned profile width is
a conditioning diagnostic, not calibrated angular uncertainty. Thick unpainted
rims and shadows can prevent a reliable absolute match; a rejected diagnostic
retains the supplied seed and records why. The batch initial-roll prior remains
finite rather than treating the user's approximate value as exact ground truth.

Both recordings should begin during the same stationary pose. Their first receive
timestamps need not coincide, and a moving start would make independently assigned
initial shell phases inconsistent. A rim cue estimates roll but cannot determine
absolute painted spin phase without a previously identified material map.

## Calibration and accuracy limits

The existing `calibration/axes.yaml` contains accepted axes and a metric pivot.
Its saved Linux source paths mention `roll32` and `swivel3`; the current local
input directories are named `roll` and `swivel`. A renamed directory alone does
not prove or disprove identical recordings. Reprocessing those clips provides
a useful consistency check, and should not silently replace the accepted axes.

The intrinsic files have no verified capture-profile metadata. Camera timing is
also marked unverified, with zero configured relative offset. Receive timestamps
are not exposure timestamps: an unmeasured camera latency biases a joint fit
during fast motion. Camera mounts, focus, zoom, metric radius/gap and the pivot
must remain consistent with calibration. These recordings also contain a large
yoke occlusion, thick rims, shadows and sparse paint, which limit usable surface
observations.

Low reprojection error demonstrates image/model agreement conditional on those
assumptions. It does not establish absolute angular accuracy, and neither the
filter's tuning covariance nor a successful optimizer accounts for all calibration
or timing bias. Validate known-angle motions and return-to-home repeatability,
preferably with an independent angular reference, before interpreting a numerical
fit residual as a physical accuracy claim.

The recovered trajectory is **orientation relative to the camera rig**, with
the two shell-center positions implied by the fixed pivot and roll. It does not
recover the robot's translation or a world position trajectory. That would need
an independently observed moving pivot/world reference and a corresponding
translation state. Illustrative markings in the simulated ball visualize phase;
they are not a reconstruction of the real painted triangles.

## Outputs

- `results.json`: one shared trajectory at every corrected native camera event,
  per-shell quaternions, roll relative to home and start, shell centers in cam0,
  support sources, residual summaries, clock correction, and provenance.
- `results.csv`: timestamped roll/red/green angles and rates, validity labels,
  whole-turn flags, caster quaternion and moving swivel axis. Null estimates are
  blank; uncertainty is left null because marginal physical covariance is not
  established.
- `bundle.npz`: optimized time knots, angle trajectories, material landmarks and
  accepted/rejected native pixel observations for reproducible analysis.
- `strict_results.csv`: supported angles only; prior-dependent phases are blank.
- `phase_evidence.npz`, `conditional_spin_check.json`: independent angular-edge
  support, phase connectivity, measurement rank and local gap diagnostics.
- `orientation_3d.html`: portable interactive replay; no server or network needed.
- `replay/combined_tracking.mp4`: synchronized camera overlays, a separate
  simulated two-shell assembly, and angle history; generated with `--render`.
- `initial_pose.json`, `roll_initialization.json`, `seed_diagnostics.json`: retain
  initialization hypotheses and rejections separately from the final fit.

The two hemispheres do not form one rigid body, so both shell quaternions are
needed alongside the shared caster frame. The simulation's white meridians show
relative material phase. They are illustrative marks, not matched triangles.
