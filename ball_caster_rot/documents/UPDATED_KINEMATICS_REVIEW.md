# Updated spherical-caster kinematics review

Reviewed 2026-09-10 against [spherical_caster_kinematics.pdf](spherical_caster_kinematics.pdf), all eight pages, and the current Python implementation. This is a review and implementation guide; it does not change the tracking equations or claim that the PDF's additional outputs are already implemented.

The current tracker measures three angular coordinates and their rates. **Its axis names and spin sign differ from the updated PDF.** Its existing synthetic tests validate the current convention, not the PDF's complete contact/slip model. The current 29 unit tests pass.

## 1. Current implementation and exact convention conversion

The code uses active, right-handed rotations on column vectors:

\[
S_h^B=R_x(\alpha_B)R_z(\beta_{h,B}),\qquad h\in\{\mathrm{top},\mathrm{bottom}\}.
\]

The rod is old ball **+x**; both shells share alpha. The shell spin is positive about its rotated **+z**. `rotation.py` explicitly documents this convention and decomposes with intrinsic `XYZ`, returning the middle angle as the off-model residual gamma: [ballrot/rotation.py](ballrot/rotation.py), lines 1-7 and 81-102. Calibration constructs `R_bc=[x_roll, y_derived, z_spin]`: [ballrot/integrate.py](ballrot/integrate.py), lines 340-378.

Camera increments satisfy `b = Delta_R @ a`; they accumulate by left multiplication and convert by `R_ball = R_bc.T @ R_camera @ R_bc`: [ballrot/estimate.py](ballrot/estimate.py), line 35; [ballrot/integrate.py](ballrot/integrate.py), lines 27-71. Each shell is decomposed separately; the reported common alpha averages valid shell alpha values, without uncertainty weighting (lines 95-160).

The PDF defines the rod as caster **+y**, with

\[
a(\alpha)=(\sin\alpha,0,\cos\alpha)^T,
\qquad
\omega_h=\dot\alpha\,\hat y_C-\dot\beta_h\,a(\alpha).
\]

These are PDF equations (1)-(5) and (10). In the same active rotation convention, the corresponding shell pose is

\[
S_h^C=R_y(\alpha_C)R_z(-\beta_{h,C}).
\]

That pose is inferred from the PDF's angular-velocity definition; the PDF does not explicitly write this product. Positive PDF beta was deliberately chosen to produce positive lateral translation.

A proper coordinate relabeling is possible without changing the geometric tracker. Let

\[
Q=R_z(\pi/2)=
\begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix},\qquad
v_C=Qv_B.
\]

Then old `+x` is PDF `+y`, old `+y` is PDF `-x`, and old `+z` is PDF `+z`. Consequently,

\[
QS_h^BQ^T=R_y(\alpha_B)R_z(\beta_{h,B}),\quad
\alpha_C=\alpha_B,\quad\beta_{h,C}=-\beta_{h,B},\quad
R_{C\to\mathrm{camera}}=R_{B\to\mathrm{camera}}Q^T.
\]

This identity assumes matching reference poses and the stated axis directions. It **does not** determine the mechanical home angle, vertical direction, or the positive direction actually used while recording calibration. Changing `Rx` to `Ry` alone would break calibration, simulation, existing configurations, and the sign conventions. A future implementation can either retain the current tracker with an explicit, tested PDF-output adapter, or migrate calibration, decomposition, renderer, tests, and configuration metadata together.

## 2. Mechanical home angle is a separate calibration

`accumulate_increments` starts every clip at identity ([ballrot/integrate.py](ballrot/integrate.py), lines 36-44). Thus current angles are **relative to the first video frame**, not automatically the PDF's absolute alpha. PDF alpha zero means the shell spin axis is vertical. `README.md`, lines 453-456, also identifies the first frame as the zero reference.

Before computing `sin(alpha)` or an effective contact radius, measure a mechanical initial tilt `alpha_0` relative to vertical and verify its direction. With a decomposition frame whose spin axis matches the initial pose, the appropriate mechanical angle is `alpha_mechanical = alpha_0 + delta_alpha`. A scalar offset is not sufficient to repair decomposition performed in a frame with an incorrectly oriented initial spin axis: the initial reference frame must first be corrected, or the footage must start at the calibrated home pose.

The reviewed contact sheet indicates a substantial, approximately quarter-turn difference between the measurement clip's starting assembly pose and the calibration pose. Treat that as a reference-pose mismatch requiring correction; it is not a calibrated 90-degree measurement. Do not obtain `alpha_0` from the software's automatic zero. A ball held or rolled above the floor can validate rotational tracking, but it cannot directly validate ground-contact or no-slip predictions.

## 3. Additional quantities from the PDF

Use physical radius `R` in metres, angles in radians, timestamps in seconds, and PDF beta signs. The tracker currently uses a unit-radius sphere solely for scale-free angular geometry ([ballrot/sphere.py](ballrot/sphere.py), lines 98-104); the pixel circle radius is not the physical radius.

| Quantity | PDF equation | Required information and interpretation |
|---|---|---|
| Spin-axis direction | `a=(sin(alpha),0,cos(alpha))` (5) | Absolute mechanical alpha and caster frame |
| Effective radius magnitude | `rho=R*abs(sin(alpha))` (7) | Physical radius and mechanical alpha |
| Shell angular-velocity vector | `omega_h=(-beta_dot_h*sin(alpha), alpha_dot, -beta_dot_h*cos(alpha))` (10) | PDF coordinate rates; a model-based vector, distinct from three Euler rates |
| Ideal caster velocity | `vx=R*alpha_dot`, `vy=R*sin(alpha)*beta_dot_contact` (14)-(15) | Correct contacting shell; ideal no-slip assumption |
| Jacobian and singularity | `J=diag(R,R*sin(alpha))`, `det(J)=R^2*sin(alpha)` (18),(21) | Singular whenever `sin(alpha)=0` |
| Ideal world velocity | `[X_dot,Y_dot]=Rz(psi)*[vx,vy]` (23)-(27) | Independent caster/chassis yaw `psi`; current footage does not infer it |
| Local accelerations | `ax=R*alpha_ddot`, `ay=R*cos(alpha)*alpha_dot*beta_dot+R*sin(alpha)*beta_ddot` (38)-(40) | Reliable differentiation and contact labels; not automatically world acceleration when yaw changes |
| Transition speed requirement | `beta_dot_n_plus=(rho_c/rho_n)*beta_dot_c_minus` (43) | Before/after contact labels, radii, rates; local positive-radius convention |
| Transition acceleration/jerk | `delta_ay=ay_plus-ay_minus`, `jerk_y~delta_ay/delta_t` (45)-(48) | Resolved transition timing and sufficiently accurate derivatives |
| Slip components | `sx=vx_actual-R*alpha_dot`, `sy=vy_actual-rho*beta_dot_contact` (51)-(52) | Independent actual chassis/center velocity; ideal velocity cannot also serve as its own ground truth |
| Friction-limited spin acceleration | `abs(beta_ddot)_max=mu_s*N*rho/I_beta` (61) | Friction coefficient, normal load, hemisphere inertia |
| Local force and impulse | `Ft=I_beta*beta_ddot/rho`, `J_impulse=I_beta*delta_beta_dot/rho` (76),(80) | Inertia and contact radius; impulse formula assumes radius effectively constant during the transition |

The PDF switches to **local `sin(alpha)>0`** from equation (8). Its positive `rho` formulas for transitions, slip, force, and friction must not be extended through other angular branches without maintaining consistent signs and contact conventions. In a fixed global signed convention, the velocity lever arm is `R*sin(alpha)`; its magnitude is `R*abs(sin(alpha))`. Near zero effective radius, report singular/ill-conditioned results instead of dividing by zero or presenting huge inverse estimates as precise measurements.

## 4. Two tracked hemispheres do not imply two ground contacts

Tracking top and bottom separately provides `beta_top` and `beta_bottom`; it does not identify which shell touches the ground. The PDF's `c` and `n` refer to the **current contact** and the **incoming contact**, rather than fixed image-top/image-bottom regions. Color identities should remain attached to the physical shell even as the assembly rotates or the video is turned upright.

For an ideal hemispherical split, a known signed shell axis, known ground normal, and verified color-to-shell assignment can predict the shell containing the lowest sphere point. At the seam, finite contact patches, loading, occlusion, or geometry deviations can make contact ambiguous. The current pipeline does not implement that inference. Use a synchronized contact annotation/sensor, or validate a geometric rule and preserve an explicit unknown/transition state. Do not simultaneously set both `R*sin(alpha)*beta_dot_top` and `R*sin(alpha)*beta_dot_bottom` equal to the measured chassis velocity. They may be exported as separate hypothetical no-slip velocities, with the contact selection recorded separately.

## 5. What current output files contain and omit

[ballrot/diagnostics.py](ballrot/diagnostics.py), lines 122-249, writes:

- `results.csv`: frame/time, shared alpha, both beta angles, radians/degrees, their finite-difference rates, separate shell alpha/gamma, inlier ratios, mean residuals, and tracked counts.
- `results.json`: metadata, angular time series, per-frame shell quality, and self-consistency summaries, including the independent differential-shell axis check supplied by `scripts/run.py`.
- `angles.png`, `angular_velocities.png`, and `diagnostics.png`; the pipeline optionally writes `tracking_overlay.mp4`.

Despite the broad names, these files do not serialize every tracking detail. `PipelineResult` retains camera-frame increments and accumulated rotations, step validity, accepted 2-D feature correspondences, viewing angles, per-feature residuals, and inlier/geometry-valid masks in memory ([ballrot/pipeline.py](ballrot/pipeline.py), lines 32-52). Lifted sphere vectors are computed during estimation but would need to be retained or recomputed for an export. The standard writer does not save these detailed arrays. A complete audit export would add them, frame timestamps/indices, calibration and convention identifiers, units, and contact/initial-pose metadata. The KLT tracker redetects correspondences for each frame pair, so pair-local feature indices must not be mislabeled as persistent feature trajectories across the whole clip.

No current standard export implements physical radius, contact state, metric velocity/path, absolute mechanical alpha, acceleration, jerk, slip, force, impulse, or friction parameters. Those require an additional model/output layer and the inputs listed above.

## 6. Accuracy limitations that matter for these outputs

- Angular rates are finite differences, not directly measured angular velocity. `angular_velocity` uses `numpy.gradient` on finite samples ([ballrot/integrate.py](ballrot/integrate.py), lines 164-180). It can bridge missing samples; acceleration and jerk would amplify noise further. Preserve native timestamps and use a documented smoothing/differentiation method with uncertainty estimates before reporting transition derivatives.
- On a failed frame pair, integration holds the previous orientation and marks that step invalid. Later steps can become valid again even though motion during the missing interval was lost (lines 44-49). Export cumulative validity or segment the reconstruction; do not interpret an apparently smooth continuation as recovered lost motion.
- Unit/synthetic tests and small residuals establish convention consistency and internal tracking quality. They do not establish ground truth for mechanical home angle, real camera calibration, true ground contact, slip, or the PDF's friction approximations.

An updated implementation should independently validate the coordinate conversion and beta sign, nonzero initial tilt, correct contact selection, singular configurations, missing-step propagation, constant-rate analytic motion, and metrics derived from independently supplied chassis velocity. Keep the existing angular outputs explicitly identified by convention so older results remain interpretable.
