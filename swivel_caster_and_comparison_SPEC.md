# Build Spec — Swivel-Caster Tracking + Cross-Device Comparison Layer

**Audience:** an autonomous coding agent.
**Companion to:** `ball_caster_rotation_pipeline_SPEC.md` (Approach A — the ball-caster speckle pipeline). Read that first; this document reuses its modules and conventions.

Read this whole document before writing code. Build the synthetic validation and the shared interface **before** touching real footage.

---

## 0. Context — what I'm building and why

I have a differential-drive robot car (two driven rear wheels; a passive caster up front). I want to **quantify how much better my custom ball caster is than a standard off-the-shelf swivel caster**, using vision.

Two hardware configurations get swapped onto the *same* car and run through the *same* maneuvers:
- **Config A — ball caster:** my custom two-hemisphere ball caster. Already instrumented by the Approach A pipeline, which outputs per-frame rotations of each hemisphere.
- **Config B — swivel caster:** a POWERTEC 6.25″ heavy-duty industrial swivel caster (solid rubber wheel, 330 lb rating, standard trailing-swivel design), mounted in place of the ball caster.

What I already have:
- **Car-level pose tracking:** the car's position and heading over time, in a world/ground frame. Treat this as an available input stream (see §5).
- **The Approach A ball pipeline** producing per-frame hemisphere rotations.

What I do **not** have and cannot get: any encoder on the caster. Both casters are passive/free-spinning, so — as in Approach A — **synthetic data with known motion is the only ground truth**. Build it first.

The goal is a fair, apples-to-apples comparison. The two devices have **different degrees of freedom**, so their raw joint angles are *not* directly comparable:
- Ball caster: `α` (roll about horizontal rod axis x), `β₁`, `β₂` (two independent hemisphere spins about vertical z).
- Swivel caster: `φ` (wheel roll about the axle), `ψ` (fork swivel about the vertical steering axis, with a trailing offset).

**Do not compare raw DOF.** Both pipelines instead feed a shared, device-agnostic metric layer built on contact-point kinematics. That layer is the heart of this task.

---

## 1. Your task

1. Build a **swivel-caster vision pipeline** that recovers `φ` and `ψ` from a chassis-mounted camera, validated synthetic-first.
2. Build a **shared metrics module** (`metrics.py`) that consumes a common per-frame record (`CasterFrame`, §5) plus the car-pose stream, and computes device-agnostic comparison metrics (alignment error, scrub, slip, swivel lag, shimmy, rolling efficiency).
3. Write **adapters** so both the swivel pipeline and the existing ball pipeline emit `CasterFrame`, so the same metrics code scores both.
4. Build a **comparison harness** that runs paired clips (same maneuver, each device) through the metrics and produces a report (plots + table).
5. Validate everything on synthetic data where the metrics have known analytic values, then run real footage judged by self-consistency.

---

## 2. Operating principles (follow these)

Same as Approach A: surface assumptions explicitly; simplest code that works; every milestone has a numeric pass gate and you loop until green; **synthetic before real**; pin every rotation/frame convention with a unit test. Reuse Approach A modules rather than reimplementing — do not fork `kabsch`, `pixel_to_ray`, KLT tracking, or the rotation utilities.

---

## 3. How this relates to the existing ball pipeline (reuse map)

Assume the Approach A repo exists. Restructure into one unified repo (§11) with a `common/` package the ball code already effectively uses. Reuse directly:

| Need here | Reuse from Approach A |
|---|---|
| Rotation utils (`Rx`, `Rz`, geodesic angle, euler) | `rotation.py` |
| Camera model, `pixel_to_ray`, undistort | `camera.py` |
| KLT detect + track + forward-backward cull | `track.py` |
| `kabsch`, `ransac_kabsch` | `estimate.py` |
| Frame loader (video or image sequence) | `io_frames.py` |
| Synthetic-first philosophy + staged gates | `synthetic/` structure |

New here: swivel geometry (ray–**plane** instead of ray–sphere), ArUco tag reading, the `CasterFrame` interface, `kinematics.py` (angular-velocity split + car-frame transfer), `metrics.py`, and the comparison harness.

If the Approach A repo is **not** present, stub the reused modules with the reference implementations from that spec and proceed — but flag that the ball adapter (§7.4) can only be tested once real ball outputs exist.

---

## 4. Coordinate frames & conventions (pin these)

- **Car frame:** origin at the car's tracked reference point; **x** forward, **y** left, **z** up. The caster camera is rigidly fixed to the chassis, so quantities it measures are naturally in the car frame.
- **World/ground frame:** the frame the car-pose stream is expressed in. 2D pose `(x, y, θ_car)` on the ground plane.
- **Camera frame:** OpenCV convention (x right, y down, z forward). Camera→car extrinsics are fixed and calibrated once (§6.2).
- **Caster mount offset** `r_arm` (2D, car frame): fixed vector from the car's tracked point to the caster's contact point. Config.
- **Rotation convention:** matrices act on column vectors; a rotation `R_incr` maps frame `k-1 → k` (i.e. `b = R_incr a`), consistent with Approach A. Angular velocity `ω = rotvec(R_incr) / dt`, expressed in the frame `R_incr` lives in (car frame).

Write a `tests/test_conventions.py` that asserts these and fails loudly if any flips.

---

## 5. The shared interface: `CasterFrame` + car track

**This is the seam that makes the comparison possible.** Both pipelines must produce a list of these; `metrics.py` consumes only this + the car track. Nothing device-specific leaks past this boundary.

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class CasterFrame:
    t: float                      # seconds, on the SAME clock as the car track (see §6.2 time sync)
    roll_axis_car: np.ndarray     # (3,) unit; horizontal axis of rolling, in the car frame
    omega_roll: float             # rad/s about roll_axis_car (rolling rate, ≥ 0)
    omega_spin: float             # rad/s about +z (swivel/spin rate, signed)
    r_eff: float                  # effective rolling radius (m); constant per device (config, calibrated §6.2)
    raw: dict = field(default_factory=dict)   # device-specific, for reference only:
                                              #   ball:   {"alpha","beta1","beta2","beta1_dot","beta2_dot"}
                                              #   swivel: {"phi","psi","phi_dot","psi_dot"}
```

Derived inside `metrics.py`, never stored device-specifically:
- `rolling_heading_car = normalize(z × roll_axis_car)[xy]` — the ground-plane direction the contact point moves under pure rolling. Sign calibrated so a straight forward run gives +x (§6.2).
- `v_roll = omega_roll * r_eff` — rolling speed the wheel/ball is producing.

**Car track input.** A time series on the same clock: `t, x, y, theta_car` (world frame). Provide a loader that reads CSV and computes `v_car_world(t)` and `omega_car(t) = θ̇_car` by finite difference (smooth with a light Savitzky-Golay filter; expose the window in config). If the source already provides velocities, accept them.

---

## 6. Swivel-caster geometry & tracking pipeline

### 6.1 DOF model

Two rigid parts:
- **Fork/yoke** — swivels about the (near-vertical) steering axis by `ψ`. Rigid; carries the wheel.
- **Wheel** — a disk that rolls about the axle by `φ`. The axle is fixed *in the fork frame* (horizontal in the fork's zero pose) and is carried around by `ψ`. The axle is offset behind the swivel axis by the **trail** — a fixed geometric parameter you note but never track per frame.

So the wheel's full rotation is `swivel ∘ roll`: `ψ` is shared with the fork (like `α` was shared between your hemispheres), and `φ` is the wheel's motion relative to the fork. Measure `ψ` from the fork, then `φ` relative to the fork.

### 6.2 Calibration (once per rig)

Implement `scripts/calibrate_swivel.py` covering:

1. **Camera intrinsics** `K`, distortion — checkerboard, or approximate + warn (as Approach A).
2. **Camera→car extrinsics** — the fixed pose of the caster camera in the car frame, and the fixed 3D location of the **wheel hub center** and **swivel axis** in the car frame (from the CAD/mount geometry, or by fitting the wheel-silhouette ellipse across a slow swivel sweep and triangulating the hub). Needed for the ray–plane lift (§6.4).
3. **Swivel zero `ψ0`** — align the tag's zero with car-forward: park the caster pointing straight ahead, record the tag yaw, store as offset.
4. **Roll-direction sign** — drive straight forward a short distance; fix the sign of `rolling_heading` so it comes out +x.
5. **Effective rolling radius `r_eff`** — do **not** use the 6.25″/2 nominal; the solid rubber compresses under load. Roll the car a known straight distance, count wheel revolutions from the `φ` track, `r_eff = distance / (2π · revs)`. A wrong `r_eff` shows up as a constant slip bias.
6. **Time synchronization (critical).** The caster-cam stream and the car-pose stream must share a clock. The metrics combine both; a fixed offset silently corrupts swivel-lag and scrub. Provide a sync method: a shared hardware timestamp if available, else a common event (an LED flash or a sharp deliberate jerk visible to both) that you align on, else cross-correlate a common signal (e.g. yaw rate vs. rolling-heading rate) during a spin and solve for the lag. Store the offset; apply it when loading. Verify residual sync error < ~1 frame.

### 6.3 Swivel `ψ` from the fork tag

Put an ArUco/AprilTag flat on the top of the fork/swivel housing. Detect with `cv2.aruco`; solve its pose (`estimatePoseSingleMarkers` / `solvePnP`) and extract the yaw about the car-vertical axis → `ψ` (subtract `ψ0`). This is **absolute, drift-free, and available at essentially all swivel angles** because the tag stays roughly horizontal and top-visible. It is the most robust signal in the system — anchor everything to it. Also unwrap `ψ` across ±180° and differentiate for `ψ_dot`.

### 6.4 Roll `φ` from wheel-sidewall speckle

Speckle the wheel's side face plus **one bold reference dot** (for absolute phase and full-turn counting). Recover roll as rotation about the known axle axis:

1. Each frame, the wheel's side plane is fully known in camera coords: hub center `Q` (from extrinsics) and plane normal `m` = axle direction, where `m` rotates with `ψ` (from the tag). No fitting per frame — it's determined.
2. Track speckle features with the reused KLT + forward-backward cull, masked to the visible sidewall.
3. **Lift tracked pixels to the disk plane by ray–plane intersection:**

```python
def unproject_to_plane(uv, K, Q, m):
    """Q: plane point (hub center, cam coords). m: unit plane normal (axle dir, cam coords)."""
    d = pixel_to_ray(uv, K)                 # reuse from camera.py
    denom = d @ m
    valid = np.abs(denom) > 1e-6
    t = (Q @ m) / np.where(valid, denom, 1.0)
    P = d * t[:, None]                      # points on the wheel plane, cam coords
    return P, valid
```

4. **Solve the roll increment as rotation about `m`**, reusing `kabsch`. Center points at the hub, fit, then take the component along the axle:

```python
from scipy.spatial.transform import Rotation
a = P_prev - Q            # (N,3) vectors from hub, camera coords
b = P_curr - Q
R = kabsch(a, b)          # reuse; b ≈ R a  (RANSAC-wrap it, as in estimate.py)
phi_incr = Rotation.from_matrix(R).as_rotvec() @ m_hat   # signed roll increment about the axle
```

The recovered `R` should be almost pure rotation about `m`; the off-axis part is a residual diagnostic. Accumulate and unwrap `φ`, using the reference dot to resolve full turns. `φ_dot` by differentiation.

**Sanity fallback:** when the sidewall is near face-on, features rotate about the projected hub center; `φ_incr` = circular mean of the per-feature angle change (corrected for elliptical foreshortening). Use this as an independent check on the primary method, not as the main path.

### 6.5 Emit `CasterFrame`

- `omega_spin = ψ_dot` (about +z).
- Roll: axle direction in the car frame is `m_car = Rz(ψ) · axle0_car`; `roll_axis_car = horizontal(m_car)` (project out any small z tilt); `omega_roll = |φ_dot|` (sign carried in `roll_axis_car`).
- `rolling_heading` follows from `roll_axis_car` in `metrics.py`; equivalently it's the wheel-pointing direction `[cos(ψ+heading0), sin(ψ+heading0)]` — cross-check the two agree.
- `raw = {"phi","psi","phi_dot","psi_dot"}`, `r_eff` from calibration.

### 6.6 Viewpoint handling

A swivel caster's two DOF are viewing-orthogonal: near ±90° swivel the wheel is edge-on to a side camera and `φ` is unreadable. Handle, cheapest first:
- **Speckle both sidewalls**; use `ψ` (always known from the tag) to select whichever face is presented, and transform accordingly.
- **Second camera** (or reuse an overhead one — if the car tracker is a ceiling camera, it already gives a clean swivel view).
- Accept brief `φ` dropouts near the poles and interpolate; mark those frames low-confidence so metrics can mask them.

The tag keeps `ψ` continuous regardless, so swivel-based metrics never drop out — only roll-based ones (slip) do, briefly.

---

## 7. The common metric layer (`metrics.py`) — the comparison core

### 7.1 Angular-velocity split (the unifying idea)

Any rolling element's angular velocity, expressed in the car frame, splits into a **horizontal part (rolling)** and a **vertical part (spin/swivel)**. Both devices produce this; it's what `CasterFrame` carries. Provide the helper both adapters use:

```python
def split_omega(R_incr, dt, z=np.array([0.0, 0.0, 1.0])):
    w = Rotation.from_matrix(R_incr).as_rotvec() / dt   # rad/s, car frame
    w_vert = float(w @ z)
    w_horiz = w - w_vert * z
    mag = np.linalg.norm(w_horiz)
    roll_axis = w_horiz / mag if mag > 1e-9 else np.array([1.0, 0.0, 0.0])
    return roll_axis, mag, w_vert          # roll_axis(unit), omega_roll(≥0), omega_spin(signed)
```

### 7.2 Car-frame velocity transfer

Move the car's measured velocity to the caster contact point (2D rigid body), then express in the car frame so it can be compared with the vision-measured heading:

```python
def contact_velocity_car(v_car_world, omega_car, theta_car, r_arm_car):
    c, s = np.cos(theta_car), np.sin(theta_car)
    Rm = np.array([[c, -s], [s, c]])          # car -> world
    r_world = Rm @ r_arm_car
    perp = np.array([-r_world[1], r_world[0]]) # z x r
    v_contact_world = v_car_world + omega_car * perp
    return Rm.T @ v_contact_world              # in car frame
```

### 7.3 Metrics

All computed per frame from `CasterFrame` + the car track (interpolated to the caster-cam timestamps after sync). Primary metrics depend only on `rolling_heading`, `omega_roll·r_eff`, and `v_contact` — so they're unaffected by the ball's ambiguous spin and are the fair head-to-head. Spin/shimmy are secondary.

```python
def signed_angle(a, b):   # 2D, radians, from a to b
    return np.arctan2(a[0]*b[1] - a[1]*b[0], a[0]*b[0] + a[1]*b[1])
```

- **Alignment error** (primary — maneuverability/lag): `signed_angle(rolling_heading, v_contact_car)`. Report time series and mean |·|.
- **Scrub** (primary — wasted lateral sliding): `v_long = v_contact·rolling_heading`; `v_lat = v_contact − v_long·rolling_heading`; `scrub_speed = |v_lat|`. Report `total_scrub = ∫ scrub_speed dt` and `scrub_fraction = ∫scrub_speed / ∫|v_contact|`.
- **Longitudinal slip** (primary — rolling fidelity): with `v_roll = omega_roll·r_eff` (signed along heading), `slip = (v_long − v_roll)/max(|v_long|, ε)`. Report mean and series. Needs valid `φ`, so mask roll-dropout frames.
- **Swivel response / lag** (secondary): cross-correlate `d/dt(rolling_heading angle)` against the demanded heading rate (`d/dt` of `v_contact` direction, or `omega_car`); lag = argmax offset × dt. For step-turn maneuvers also report settling time for |alignment error| to fall under a threshold.
- **Shimmy** (secondary): `scipy.signal.welch` PSD of `ψ` (swivel) or `rolling_heading` over a straight/steady segment; report dominant peak frequency (Hz) and amplitude (deg).
- **Rolling efficiency** (summary): `∫|v_long| / ∫|v_contact|` = `1 − scrub_fraction`.

Return a dataclass of series + scalar summaries; `diagnostics.py` plots them.

### 7.4 Adapters → `CasterFrame`

- **Swivel adapter:** wraps §6 outputs into `CasterFrame` (mostly direct; `omega_roll`, `roll_axis_car` from `φ_dot` and `ψ`; `omega_spin = ψ_dot`).
- **Ball adapter:** consumes the Approach A per-frame hemisphere rotations `R_top(k)`, `R_bottom(k)` (already in the car frame after that pipeline's frame calibration). For each, `split_omega`. The two hemispheres share the roll → **average their horizontal parts** for `roll_axis_car` / `omega_roll`; their vertical parts are `β̇₁`, `β̇₂` → put in `raw`, and set `omega_spin` to the vertical component most relevant to redirection (document the choice; the primary metrics don't depend on it). This mirrors Approach A's α/β decomposition.

Both adapters output the identical `CasterFrame` stream. `metrics.py` cannot tell which device produced it — that's the design goal.

---

## 8. Synthetic harness — build this first

Extend `synthetic/` with a **swivel-caster renderer** and metric ground-truth checks.

**Renderer.** Model the two rigid parts with the trail geometry:
- Wheel sidewall speckle: sample random points on a disk (radius `[0.2R, R]`) in the wheel's side plane (both faces), plus one reference dot. At time `t`, apply roll `R_axle(φ(t))` in the wheel frame, orient by swivel `Rz(ψ(t))`, place at the hub (offset from the swivel axis by the trail), transform car→camera via the known extrinsics, project with `K`, cull back-facing points. Reuse the projection/visibility logic from the Approach A renderer.
- Fork tag: render a real ArUco marker (`cv2.aruco.generateImageMarker`) warped onto the plate via homography at yaw `ψ(t)`, so the actual detector runs end-to-end.
- Toggleable degradations (as Approach A): pixel noise, motion blur, glare, and **fast swivel/roll** (large per-frame Δ). Also render the ±90° edge-on swivel poses to exercise roll dropout.
- Save frames + `ground_truth.json` (`K`, extrinsics, hub/axle, per-frame `φ`, `ψ`, and the scripted **car trajectory** so the metrics can be checked).

**Metric ground truth.** Because you script both the car trajectory and the ideal caster response, you can compute the true alignment error, scrub, slip, and lag analytically and check `metrics.py` reproduces them. Include canonical cases with obvious answers:
- Pure straight run → alignment error ≈ 0, scrub ≈ 0, slip ≈ 0.
- Spin-in-place (zero net translation, caster must whip around) → large scrub, `omega_roll` small.
- Constant-radius circle → constant nonzero alignment offset for a trailing swivel, ≈ 0 for an ideal ball.

**Staged gates (each a hard stop):**
- **S-A — conventions** (`pytest`): frame/rotation asserts (§4); `unproject_to_plane` round-trip < 1e-9; roll-about-axis extraction recovers a known `φ` < 1e-9.
- **S-B — swivel `ψ`** from rendered tag: recover < 1° (noise-free), < 2° at 1 px noise.
- **S-C — roll `φ`** via ray–plane + axis-Kabsch: recover < 1° (noise-free), < 2° at 1 px noise; verify off-axis residual stays small.
- **S-D — metrics on synthetic**: on the canonical cases, computed metrics match analytic ground truth within tight tolerance (alignment/scrub within a few % ; slip within 2%).
- **S-E — robustness sweep**: noise / blur / speed → RMSE curves; state the **Δψ and Δφ per frame** at which recovery degrades past 2° (this sizes the required FPS for the real camera). Confirm roll dropout is confined to the ±90° band and correctly flagged low-confidence.

`scripts/validate_synthetic_swivel.py` runs S-A…S-E and prints a PASS/FAIL table with measured numbers next to thresholds.

---

## 9. Real footage + self-consistency

Only after §8 passes. On real clips there's no ground truth; judge by self-consistency (mask roll-dropout frames):
- Tag pose reprojection error small; `ψ` continuous (no unwrap glitches).
- Roll: Kabsch inlier ratio > 0.7; off-axis residual small; `φ` monotonic under steady rolling.
- **Cross-check:** `rolling_heading` derived from `roll_axis` (via `split_omega`) agrees with the wheel-pointing direction from `ψ` (should match within a few degrees). Disagreement means bad extrinsics or `ψ0`.
- Straight-run slip ≈ 0 after `r_eff` calibration (any constant offset → refit `r_eff`).
- **Loop closure:** film a clip returning the caster to a marked pose; integrated drift a few degrees at most.

---

## 10. Standardized comparison protocol

Same car, swap casters, identical camera rig and metrics code — only the geometry model changes. For each maneuver, capture a paired clip per device and run both through `metrics.py`. Maneuver → metric it loads:

- **Straight line** → longitudinal slip, rolling efficiency (baseline; expect similar).
- **Step turn** (sharp heading change) → swivel lag, initial-flip scrub. The trailing swivel pays here.
- **Slalom, increasing speed** → shimmy onset frequency, alignment error.
- **Spin-in-place** (differential drive, zero translation) → pure scrub; the swivel must whip its wheel around, the ball needn't.
- **Figure-8** → sustained alignment error, cumulative scrub.

The comparison harness (`compare/run_comparison.py`) takes a manifest of `(maneuver, device, clip, car_track)` rows, produces per-maneuver overlay plots (both devices on shared axes) and a summary table (mean alignment error, total scrub, slip, shimmy freq, efficiency, per device per maneuver). Frame findings as hypotheses the data tests — expect the ball to show lower lag/scrub/no shimmy, likely trading off load rating and rolling resistance. Report surprises honestly.

---

## 11. Project structure (unified repo)

```
caster_tracking/
├── README.md                         # context, assumptions, how to run, PASS/FAIL summary
├── requirements.txt                  # numpy, scipy, opencv-python (+ contrib for aruco), matplotlib, pyyaml, pytest
├── config.example.yaml
├── common/                           # shared by BOTH devices
│   ├── rotation.py  camera.py  track.py  estimate.py  io_frames.py   # reused from Approach A
│   ├── kinematics.py                 # split_omega, contact_velocity_car, car-track loader
│   └── caster_frame.py               # the CasterFrame dataclass
├── ball/                             # Approach A pipeline + adapter to CasterFrame
├── swivel/
│   ├── geometry.py                   # hub/axle/plane model, unproject_to_plane
│   ├── tag.py                        # ArUco -> psi
│   ├── roll.py                       # speckle KLT -> phi (ray-plane + axis-Kabsch)
│   ├── pipeline.py                   # clip -> CasterFrame stream
│   └── adapter.py                    # -> CasterFrame
├── metrics/
│   ├── metrics.py                    # device-agnostic metrics
│   └── diagnostics.py                # plots
├── synthetic/
│   ├── generate_swivel.py            # renderer + ground truth
│   └── validate_synthetic_swivel.py  # S-A..S-E
├── compare/
│   └── run_comparison.py             # paired clips -> comparison report
├── scripts/
│   ├── run_swivel.py                 # real swivel clip -> CasterFrame + diagnostics
│   └── calibrate_swivel.py           # intrinsics, extrinsics, hub/axle, psi0, r_eff, time sync
└── tests/
    ├── test_conventions.py
    └── test_metrics.py               # canonical-case metric checks
```

Keep modules small and single-purpose. `metrics.py` must import nothing device-specific.

---

## 12. Milestones (loop until each gate is green)

```
S1  common/caster_frame.py + kinematics.py + metrics/metrics.py + tests/test_metrics.py
    → verify: canonical hand-built CasterFrame series give expected metrics
              (straight: align≈0, scrub≈0, slip≈0; spin-in-place: scrub large, omega_roll small)

S2  swivel/tag.py + synthetic tag rendering
    → verify: S-B — psi recovered <1° noise-free, <2° at 1px noise

S3  swivel/geometry.py + roll.py
    → verify: S-A + S-C — plane round-trip <1e-9; phi recovered <1° noise-free, <2° noisy

S4  swivel/pipeline.py + adapter.py + generate_swivel.py
    → verify: S-D metrics match analytic ground truth on canonical cases;
              S-E robustness report gives Δψ,Δφ-per-frame limits and confirms flagged roll dropout

S5  ball/adapter.py
    → verify: ball synthetic outputs wrap to CasterFrame; straight/turn metrics sane and
              match the ball's known scripted motion

S6  compare/run_comparison.py
    → verify: both devices' synthetic clips through the SAME metrics on the SAME car maneuver;
              metrics reproduce analytic truth; comparison report (plots + table) produced

S7  real footage
    → verify: scripts/run_swivel.py on a real clip passes §9 self-consistency;
              a paired real ball+swivel maneuver produces a comparison table
```

Do not skip ahead. A green S6 with a red S3 is meaningless.

---

## 13. Failure modes → what they indicate

- **`rolling_heading` from `roll_axis` disagrees with `ψ`-derived heading** → wrong camera→car extrinsics or `ψ0`. Fix calibration before trusting any metric.
- **Constant nonzero straight-run slip** → wrong `r_eff`; refit from a known-distance roll.
- **Swivel-lag / scrub look wrong but per-frame tracking is clean** → caster-cam and car-track clocks are not synced; redo §6.2 step 6.
- **Roll drops out mid-run** → the wheel hit the ±90° edge-on band; expected — confirm those frames are flagged low-confidence and masked from slip, and consider both-sidewall speckle or a second camera.
- **`φ` off-axis residual grows** → hub center / axle direction wrong (extrinsics), or speckle contaminated by the tread/fork; tighten the sidewall mask.
- **Metrics differ between devices but so does something uncontrolled** → the comparison isn't fair; re-check same floor, matched speeds, identical camera rig and maneuver. The whole point is that only the geometry model differs.
- **Everything passes synthetic, real is noisy** → real-world issues the render didn't model: rolling shutter (use global shutter), glare (matte finish/speckle), blur (more light / shorter exposure / higher FPS per the S-E Δ limits).

---

### Summary for the agent

Build the `CasterFrame` seam and the device-agnostic `metrics.py` first, prove them on hand-built canonical cases, then build the swivel pipeline (tag → `ψ`, ray–plane + axis-Kabsch → `φ`) and validate it against a synthetic swivel renderer with known motion and known metric ground truth. Wrap the existing ball pipeline in an adapter to the same seam, then run both devices through one shared metrics module on identical maneuvers. Raw DOF differ between the devices and must never be compared directly — the comparison lives entirely in the shared contact-kinematics metrics. Synthetic is the oracle; real footage is judged by self-consistency and a loop-closure clip.
