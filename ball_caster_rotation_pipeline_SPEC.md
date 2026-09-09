# Build Spec — Speckle-Based Rotation Measurement for a Two-Hemisphere Ball Caster

**Audience:** an autonomous coding agent.
**Deliverable:** a Python pipeline that, given a video or image sequence of a random-speckled ball caster, tracks the speckle and measures the three rotational DOF — plus a **synthetic ground-truth harness** that proves the pipeline is correct *before* it ever sees real footage.

Read this whole document before writing code. Do not start with the real-video path. Build and pass the synthetic validation first (Section 8), then apply to real footage (Section 9).

---

## 0. What "done" looks like

The system estimates, per frame, the rotation of each hemisphere and decomposes it into `q = [α, β₁, β₂]`:

- `α` — roll of the whole assembly about the horizontal rod axis (**x**), shared by both hemispheres.
- `β₁` — spin of the **top** hemisphere about the vertical axis (**z**).
- `β₂` — spin of the **bottom** hemisphere about **z**, independent of the top.

"Correct" is defined operationally by the pass gates in Sections 8 and 9. The headline gate: on synthetic data with known rotation, recovered `α, β₁, β₂` match ground truth to within a stated tolerance (target: RMSE < 2°). If that passes, the maths and tracking are sound and the same code runs on real video.

---

## 1. Operating principles (follow these)

1. **Surface assumptions.** Section 3 lists them. If any is wrong for the actual footage, stop and flag it — do not silently paper over it.
2. **Simplicity first.** This is a measurement tool, not a framework. No plugin systems, no premature abstractions. If a module is 200 lines and could be 60, rewrite it.
3. **Verifiable goals, loop until green.** Every milestone in Section 11 has an explicit numeric check. Implement, run the check, iterate. Do not move to the next milestone until the current one passes.
4. **Synthetic before real.** You have no encoder ground truth on the real rig, so synthetic data *is* your oracle. It is the single most important part of this build. Treat a failing synthetic test as a hard stop.
5. **Convention discipline.** Rotation conventions (matrix multiply order, axis mapping, `apply` direction) are where this class of code silently breaks. Pin every convention with a unit test (Stage A) that fails loudly if a convention flips.

---

## 2. The method in brief (why this works)

The camera is rigidly mounted to the robot chassis, so the ball only **rotates** — its center never translates. It therefore projects to a **fixed circle** in the image (fixed center + pixel radius), calibrated once.

That gives the core trick: any pixel on the ball back-projects to a point on the sphere via **ray–sphere intersection**. A tracked speckle dot moving from pixel `p₁` to `p₂` corresponds to two 3D surface directions `a` and `b` on the unit sphere, related by the rotation we want: `b = R·a`. With several tracked dots, solve for `R` in closed form (**Kabsch / orthogonal Procrustes**), robustified with RANSAC. Run it **separately** on the top-hemisphere dots and the bottom-hemisphere dots to get `R_top` and `R_bottom` per frame. Integrate over time and decompose into `α, β₁, β₂`.

**Scale-free.** Only *directions* on the sphere matter, and directions are invariant to the sphere's metric size. You never need the ball's real-world radius — only the projected circle (center + pixel radius) and the camera intrinsics. Set the sphere radius to 1 internally.

Pipeline stages: `frame → undistort → mask (ball / top / bottom / yoke) → detect+track speckle (KLT) → unproject tracked pairs to unit sphere → Kabsch+RANSAC per hemisphere → integrate → decompose → diagnostics`.

---

## 3. Assumptions & decision points (confirm before coding)

State each of these in the README and re-check against the actual test footage. Flag any mismatch.

1. **Camera fixed to chassis** → ball center + projected circle are constant across the clip. If the footage is from a world-fixed camera with a moving robot, this pipeline does **not** apply as written — stop and report.
2. **Two speckle colors**, one per hemisphere (Approach A). This is what lets us segment top vs. bottom. **Fallback if the footage is single-color:** segment by the detected equator line (dots above the equator ellipse = top, below = bottom). Implement color-based as primary, equator-split as fallback, selectable in config.
3. **The yoke occludes a strip** and must be masked out (it is a rigid black part, not sphere surface).
4. **Inter-frame rotation is small enough for KLT** (typically < ~10° between consecutive frames, and no dot crosses the whole visible cap in one step). If the ball spins fast relative to the frame rate, tracking breaks — the robustness sweep (Stage D) quantifies exactly where, which tells the user the FPS they need.
5. **Camera intrinsics `K` (+ distortion)** are available or calibratable. If not provided, allow an approximate `K` from image size and an assumed FOV, and warn that accuracy degrades.
6. **Axis frame for real data.** The Kabsch rotation comes out in *camera* coordinates. Decomposing into `α` (about ball-x) and `β` (about ball-z) requires the fixed ball→camera orientation `R_bc`. For synthetic data this is known exactly. For real data it must be **calibrated** (Section 7.7). Confirm which case you're in.

---

## 4. Project structure

```
ball_caster_rot/
├── README.md                 # assumptions, how to run, pass/fail summary
├── requirements.txt
├── config.example.yaml
├── ballrot/
│   ├── __init__.py
│   ├── rotation.py           # Rx, Rz, geodesic angle, euler decomposition
│   ├── camera.py             # intrinsics, undistort, pixel→ray
│   ├── sphere.py             # circle fit, sphere pose from circle, ray–sphere unproject
│   ├── segment.py            # ball/top/bottom/yoke masks (color + equator fallback)
│   ├── track.py              # KLT detect+track, limb/yoke culling, fwd-bwd check
│   ├── estimate.py           # kabsch, ransac_kabsch, per-hemisphere solve
│   ├── integrate.py          # accumulate increments, frame calibration, decompose α,β₁,β₂
│   ├── diagnostics.py        # residuals, inlier ratios, loop closure, plots
│   ├── io_frames.py          # video OR image-sequence loader (auto-detect)
│   └── pipeline.py           # wire it all together for one clip
├── synthetic/
│   ├── generate.py           # rotating speckle-sphere renderer with known GT
│   └── validate.py           # Stages A–D, prints PASS/FAIL vs thresholds
├── scripts/
│   ├── run.py                # real footage entry point
│   ├── validate_synthetic.py # runs the whole synthetic gate
│   └── calibrate_circle.py   # interactive/auto circle + axis calibration helper
└── tests/
    └── test_units.py         # Stage A numeric unit tests (pytest)
```

Keep modules small and single-purpose. `pipeline.py` orchestrates; it should read top-to-bottom like the stage list in Section 2.

---

## 5. Dependencies & environment

- Python 3.10+
- `numpy`, `scipy` (for `Rotation` and SVD convenience), `opencv-python`, `matplotlib`, `pyyaml`
- `pytest` for unit tests

Pin versions in `requirements.txt`. No deep-learning dependencies — this is pure geometry + classical CV.

---

## 6. Configuration (`config.example.yaml`)

```yaml
input:
  path: "data/clip.mp4"        # video file OR a directory/glob of frames
  type: "auto"                 # auto | video | images
  max_frames: null

camera:
  # 3x3 intrinsics; if null, approximate from image size + fov_deg and warn
  K: [[1000, 0, 640], [0, 1000, 360], [0, 0, 1]]
  dist: [0, 0, 0, 0, 0]        # OpenCV distortion coeffs; zeros = pinhole
  fov_deg: 60                  # used only if K is null

circle:
  # fixed projected circle of the ball, in pixels. Auto-fit if null.
  u0: null
  v0: null
  r_px: null
  refit_each_frame: false      # true only if the mount is not perfectly rigid

segment:
  mode: "color"                # color | equator
  top_hsv:    { lo: [ ... ], hi: [ ... ] }   # fill after inspecting footage
  bottom_hsv: { lo: [ ... ], hi: [ ... ] }
  yoke_hsv:   { lo: [ ... ], hi: [ ... ] }   # dark strip to exclude

track:
  max_corners: 400
  quality: 0.01
  min_distance_px: 6
  klt_win: 21
  fwd_bwd_err_px: 1.0          # forward-backward consistency gate
  limb_cull_deg: 65            # drop dots whose viewing angle exceeds this

estimate:
  ransac_iters: 200
  ransac_inlier_deg: 1.0       # angular inlier threshold on the sphere
  min_inliers: 8

frame_calib:
  # ball->camera orientation for real data. For synthetic it is known.
  R_bc: null                   # 3x3, or calibrate via scripts/calibrate_circle.py

output:
  dir: "out/"
  save_overlay_video: true
```

---

## 7. Core modules — specs and reference implementations

The reference snippets below are **correct-by-convention starting points**. Copy them, then let Stage A unit tests confirm the conventions. If a test fails, fix the snippet — don't fork the convention.

### 7.1 `rotation.py`

```python
import numpy as np
from scipy.spatial.transform import Rotation

def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def geodesic_angle(R1, R2):
    """Smallest rotation angle (radians) between two rotation matrices."""
    cos = (np.trace(R1 @ R2.T) - 1.0) / 2.0
    return np.arccos(np.clip(cos, -1.0, 1.0))

def decompose_alpha_beta(R_ball):
    """
    Model: R_ball = Rx(alpha) @ Rz(beta), i.e. intrinsic X then Z with the
    middle (Y) rotation forced to zero. Returns (alpha, beta, gamma_residual).
    gamma_residual should be ~0; it is a validity check on the 2-DOF model.
    NOTE: R_ball must be expressed in BALL axes, not camera axes (see 7.7).
    """
    alpha, gamma, beta = Rotation.from_matrix(R_ball).as_euler('XYZ')
    return alpha, beta, gamma
```

### 7.2 `camera.py`

- Hold `K` and `dist`. Provide `undistort_points(uv)` via `cv2.undistortPoints` (return pixels, not normalized) when `dist` is nonzero.
- `pixel_to_ray(uv, K)` → unit ray in camera coords (camera at origin, looking +z):

```python
def pixel_to_ray(uv, K):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    d = np.stack([(uv[:,0]-cx)/fx, (uv[:,1]-cy)/fy, np.ones(len(uv))], axis=1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)
```

### 7.3 `sphere.py` — circle fit, sphere pose, unprojection

**Circle fit (real data).** Fit the ball silhouette on one clear frame: threshold the ball vs. background, take the largest contour, `cv2.minEnclosingCircle` or a least-squares circle fit. Store `(u0, v0, r_px)`. If `refit_each_frame` is false, reuse it for the whole clip.

**Sphere pose from the circle (scale-free).** Set radius = 1; place the center along the ray through the circle center at the distance that reproduces the observed pixel radius:

```python
def sphere_pose_from_circle(u0, v0, r_px, K):
    """Returns (C, radius) with radius fixed to 1 (rotation is scale-free)."""
    fx = 0.5 * (K[0,0] + K[1,1])
    center_ray = pixel_to_ray(np.array([[u0, v0]]), K)[0]
    theta = np.arctan(r_px / fx)          # angular radius of the silhouette (paraxial)
    dist = 1.0 / np.sin(theta)
    return center_ray * dist, 1.0
```

For wide-FOV or strongly off-center balls, refine with the exact silhouette-cone relation; the paraxial form above is fine for typical lenses near image center.

**Ray–sphere unprojection** (pixels → unit directions on the sphere):

```python
def unproject_to_sphere(uv, K, C, radius):
    """
    uv: (N,2) pixels. Returns (dirs (N,3) unit vectors from sphere center,
    valid mask). dirs are the quantities Kabsch consumes.
    """
    d = pixel_to_ray(uv, K)               # (N,3) unit rays
    b = d @ C                             # d·C
    c = C @ C - radius**2
    disc = b*b - c
    valid = disc > 0
    t = b - np.sqrt(np.clip(disc, 0.0, None))   # near intersection
    P = d * t[:, None]                    # surface points, camera coords
    dirs = (P - C) / radius              # unit surface directions
    return dirs, valid
```

Also expose the **viewing angle** of a surface direction (for limb culling): the angle between the outward normal `n = dirs` and the view direction from camera to point. Cull dots whose viewing angle exceeds `limb_cull_deg` (grazing → unreliable).

### 7.4 `segment.py` — masks

Produce four boolean masks per frame: `ball` (inside the fitted circle), `yoke` (dark strip, excluded), `top`, `bottom`.

- **Color mode:** HSV-threshold the two speckle colors and the yoke, then `AND` with the ball mask and `AND NOT` the yoke mask. Light morphological open/close to denoise.
- **Equator fallback:** fit the equator ellipse (or take the horizontal diameter of the circle, adjusted for the small perspective tilt) and split ball-interior pixels by side. Use when `mode: equator`.

Detected/tracked features are assigned to top or bottom by which mask their pixel lands in. Drop features in `yoke` or outside `ball`.

### 7.5 `track.py` — KLT with culling

- Detect features per hemisphere mask with `cv2.goodFeaturesToTrack` (respect `max_corners`, `quality`, `min_distance_px`).
- Track frame→frame with `cv2.calcOpticalFlowPyrLK` (`klt_win`, pyramids).
- **Forward-backward check:** track back and reject any point whose round-trip error exceeds `fwd_bwd_err_px`.
- **Re-seed** features each frame (or when a hemisphere's count drops below a floor) so dots vanishing at the limb are continuously replaced by fresh ones entering the well-conditioned center.
- Emit, per frame pair and per hemisphere, matched pixel arrays `(uv_prev, uv_curr)`.

### 7.6 `estimate.py` — Kabsch + RANSAC

```python
def kabsch(a, b):
    """Rotation R minimizing sum ||R a_i - b_i||^2, so b ≈ R a. a,b: (N,3)."""
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R
```

```python
def ransac_kabsch(a, b, iters, inlier_rad, min_inliers, rng):
    """a,b: (N,3) unit dirs (prev, curr). Returns (R, inlier_mask) or None."""
    N = len(a)
    if N < 3:
        return None
    best_R, best_inl = None, None
    for _ in range(iters):
        idx = rng.choice(N, 3, replace=False)
        R = kabsch(a[idx], b[idx])
        # angular residual per correspondence
        pred = a @ R.T
        ang = np.arccos(np.clip(np.sum(pred * b, axis=1), -1, 1))
        inl = ang < inlier_rad
        if best_inl is None or inl.sum() > best_inl.sum():
            best_R, best_inl = R, inl
    if best_inl is None or best_inl.sum() < min_inliers:
        return None
    R = kabsch(a[best_inl], b[best_inl])     # refit on inliers
    return R, best_inl
```

Per frame pair: unproject `uv_prev`, `uv_curr` for each hemisphere, apply limb/validity culling, then `ransac_kabsch` → incremental `R_top(k)`, `R_bottom(k)`. Record inlier ratio and mean inlier residual for diagnostics.

### 7.7 `integrate.py` — accumulate, frame-calibrate, decompose

**Accumulate** incremental rotations into absolute per-hemisphere orientation (mind the order — increments are camera-frame rotations `b = R a`):

```python
R_top_abs[k]    = R_top(k)    @ R_top_abs[k-1]
R_bottom_abs[k] = R_bottom(k) @ R_bottom_abs[k-1]
```

**Frame calibration (camera → ball axes).** The accumulated rotations are in camera coords. Convert to ball coords before decomposition:

```python
R_ball = R_bc.T @ R_cam @ R_bc     # R_bc = ball→camera orientation (3x3)
```

- **Synthetic:** `R_bc` is exactly the value the renderer used. Use it directly (this also validates the decomposition maths independently of any calibration error).
- **Real:** estimate `R_bc` from a calibration clip. Practical procedure, implemented in `scripts/calibrate_circle.py`:
  - Capture a segment where the caster performs **pure swivel** (only β changes): every incremental `R_top` shares the same rotation axis — that axis, averaged, is the ball **z** in camera coords.
  - Capture a segment of **pure roll** (only α changes): the shared incremental axis is ball **x** in camera coords.
  - Orthonormalize `[x, y=z×x, z]` into `R_bc`. Store in config.
  - If capturing clean isolated motions is impractical, fall back to solving `R_bc` that best makes the observed `R_top`, `R_bottom` fit the `Rx(α)Rz(β)` model across the clip (minimize the `gamma_residual`).

**Decompose** each hemisphere's ball-frame rotation with `decompose_alpha_beta`:
- `α` = the shared X angle (average the two hemispheres' α, or take the top's; they should agree — their disagreement is a diagnostic).
- `β₁` = top's Z angle; `β₂` = bottom's Z angle.
- `gamma_residual` per hemisphere should stay near 0; a growing residual signals tracking error, bad `R_bc`, or that the physical motion left the assumed 2-DOF manifold.

Also expose **angular velocities** (finite-difference the absolute angles, or read them straight from the per-frame increments — the increment axis·angle *is* the instantaneous rotation, no integration needed). If the user only wants velocities, this path avoids drift entirely.

### 7.8 `diagnostics.py`

Compute and save:
- Per-frame **Kabsch mean inlier residual** (deg) and **RANSAC inlier ratio**, per hemisphere.
- **Tracked-point count** per hemisphere over time (coverage).
- **Forward-backward** tracking error distribution.
- **Loop closure:** if the clip returns to a marked start pose, report `geodesic_angle(R_abs[end], R_abs[start])` as accumulated drift.
- **Reversibility:** optionally run the increment sequence backward; forward∘backward should be ≈ identity.
- **`gamma_residual`** time series.
- Plots: `α, β₁, β₂` vs. time; angular velocities; all diagnostics above. Optional annotated overlay video showing tracked dots colored by hemisphere and the estimated axes projected onto the ball.

---

## 8. Synthetic ground-truth harness — **build this first**

This is the oracle. It renders a rotating speckle sphere with a **known** `q(t)`, runs the exact same pipeline, and compares.

### 8.1 `synthetic/generate.py`

```python
import numpy as np

def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2*i/n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.stack([np.sin(phi)*np.cos(theta),
                     np.sin(phi)*np.sin(theta),
                     np.cos(phi)], axis=1)   # (n,3) unit, ball frame

def project(points_ball_t0, R_motion, R_bc, C, radius, K):
    """
    points_ball_t0: (N,3) unit dirs at t=0. R_motion: applied in ball frame.
    Returns (uv (N,2), facing mask). Ball->camera orientation R_bc is fixed.
    """
    p = points_ball_t0 @ R_motion.T            # rotated material points (ball frame)
    n_cam = p @ R_bc.T                         # outward normals in camera frame
    x_cam = C + radius * n_cam                 # surface points in camera frame
    facing = np.sum(x_cam * n_cam, axis=1) < 0 # visible if normal faces camera
    proj = x_cam @ K.T
    uv = proj[:, :2] / proj[:, 2:3]
    return uv, facing
```

Generator responsibilities:
- Split `points_ball_t0` into **top** (`z_ball > 0`) and **bottom** (`z_ball < 0`); these get different colors.
- Drive a scripted, known trajectory: `α(t)`, `β₁(t)`, `β₂(t)` — include ramps, sinusoids, sign changes, and independent top/bottom motion. Store as ground truth.
- Per hemisphere per frame, build the motion `R_motion = Rx(α(t)) @ Rz(β(t))`, project, cull to `facing`, and draw filled dots (hemisphere color) on an image sized to `K`.
- Add a **yoke strip** occluder (draw a dark band; also remove any dots it covers).
- Toggleable degradations for Stage D: Gaussian pixel noise, motion blur (average a few sub-steps), a specular glare blob, dot-density reduction, and faster spin (larger per-frame Δangle).
- Save frames + a `ground_truth.json` (`R_bc`, `C`, `K`, per-frame `R_top`, `R_bottom`, `α, β₁, β₂`).

### 8.2 `synthetic/validate.py` — staged gates

Run these in order. **Each is a hard gate.**

**Stage A — numeric unit tests (no images).** In `tests/test_units.py`:
- `unproject_to_sphere` round-trips: take known sphere directions, project to pixels analytically, unproject, recover the same unit vectors. Error < 1e-9.
- `kabsch` recovers a known random `R` from clean corresponding directions. `geodesic_angle < 1e-6`.
- `decompose_alpha_beta(Rx(a) @ Rz(b))` returns `(a, b, ~0)` for random `a, b`. Error < 1e-9, residual < 1e-9.
- Convention guard: assert `b ≈ R a` (not `a ≈ R b`) for the Kabsch output. **If this flips, everything downstream is wrong — fail loudly.**

**Stage B — tracking on rendered frames (noise-free, moderate speed).** Render a clip, run the full pipeline with the *known* `R_bc`, `C`, `K` (skip circle fit and axis calibration for now — isolate tracking + geometry). Compare recovered per-frame `R_top`, `R_bottom` to ground truth.
- **Pass:** median per-frame `geodesic_angle < 0.5°`, 95th percentile `< 1.5°`.

**Stage C — decomposition end-to-end (noise-free).** From the known `α, β₁, β₂` sequences, compare recovered sequences after `decompose_alpha_beta`.
- **Pass:** RMSE of each of `α, β₁, β₂ < 1°`; median `gamma_residual < 0.5°`; recovered `α` from top and bottom agree within `1°`.

**Stage D — robustness sweeps (this sizes the real camera).** Re-run Stage C while sweeping one degradation at a time, and produce a short report:
- Pixel noise σ ∈ {0.25, 0.5, 1.0, 2.0} px → RMSE curve. Expect `< 2°` up to ~1 px.
- Inter-frame rotation Δ ∈ {2, 5, 10, 15, 20}° → find the Δ where median error first exceeds 2°. **This Δ_max, combined with the expected spin rate, dictates the minimum frame rate the user needs.** Report it explicitly.
- Motion blur on/off; glare on/off; speckle density halved → RMSE deltas.
- **Pass:** the report is generated and Δ_max is stated. (These sweeps characterize limits; they are not themselves a hard correctness gate beyond the noise-free stages.)

**Stage E gate for calibration (still synthetic).** Now exercise the *real-data* front end against synthetic truth: run circle-fit and the axis-calibration procedure (Section 7.7) on rendered clips instead of using the known values, then repeat Stage C.
- **Pass:** RMSE of `α, β₁, β₂ < 2°` using *estimated* circle and *estimated* `R_bc`. This proves the real-footage front end works when the answer is known.

`scripts/validate_synthetic.py` runs A–E and prints a single **PASS/FAIL summary table** with the measured numbers next to the thresholds.

---

## 9. Real-footage pipeline & self-consistency tests

Only run this after Section 8 passes. On real video there is no ground truth, so correctness is argued from **self-consistency**. `scripts/run.py`:

1. Load the clip (`io_frames.py` auto-detects video vs. image sequence).
2. Fit the projected circle (or read it from config); compute sphere pose.
3. Load or calibrate `R_bc` (Section 7.7).
4. Run the pipeline → `α, β₁, β₂` time series + diagnostics.
5. Emit plots, a results CSV/JSON, and (optionally) the overlay video.

**Self-consistency acceptance targets** (tune on the first good clip; treat large violations as failures to investigate, not numbers to force):
- Kabsch mean inlier residual per frame **< ~0.3–0.5°** per hemisphere.
- RANSAC inlier ratio **> 0.7**.
- Forward-backward tracking error median **< 1 px**.
- `gamma_residual` stays small and does not trend upward.
- Tracked-point count per hemisphere stays above `min_inliers` throughout (if it dips, increase speckle density, improve lighting, or raise FPS).
- **Loop-closure drift:** film a clip that returns the caster to a physically marked start pose; accumulated `geodesic_angle` at the end should be a few degrees at most over a modest run. (This is the closest thing to ground truth you can get on the real rig — recommend the user shoot one such clip.)

If residuals are low, inliers high, and loop-closure drift small, the measurement is trustworthy. If not, the diagnostics point at the cause (see Section 12).

---

## 10. CLI & entry points

- `python scripts/validate_synthetic.py --config config.example.yaml` → runs Stages A–E, prints PASS/FAIL table, writes plots to `out/synthetic/`. **Run this first.**
- `python scripts/run.py --config config.yaml` → real clip → results + diagnostics in `out/`.
- `python scripts/calibrate_circle.py --clip <file>` → helper to fit the circle and (from labeled pure-swivel / pure-roll segments) estimate `R_bc`, writing values back into a config stub.
- `pytest` → Stage A unit tests.

Every entry point prints its resolved assumptions (camera fixed? segmentation mode? circle source? `R_bc` source?) at startup so mismatches are caught immediately.

---

## 11. Milestones (loop until each check is green)

```
M1  rotation.py + camera.py + sphere.py + tests/test_units.py
    → verify: pytest Stage A all pass (round-trip <1e-9, Kabsch <1e-6, convention guard holds)

M2  synthetic/generate.py + minimal pipeline (known R_bc/C/K, no circle fit)
    → verify: Stage B — median per-frame rotation error <0.5°, p95 <1.5°

M3  integrate.py decomposition
    → verify: Stage C — α,β₁,β₂ RMSE <1°, gamma_residual median <0.5°, top/bottom α agree <1°

M4  robustness sweeps
    → verify: Stage D report generated; Δ_max (deg/frame before >2° error) stated

M5  real-data front end (circle fit + axis calibration) against synthetic truth
    → verify: Stage E — α,β₁,β₂ RMSE <2° using ESTIMATED circle and R_bc

M6  real-footage run + diagnostics
    → verify: on the provided clip, residuals/inlier/fwd-bwd/loop-closure targets in Section 9 met;
              plots + CSV/JSON + overlay video produced
```

Do not skip ahead. A green M6 with a red M3 is meaningless.

---

## 12. Common failure modes → what they indicate

- **High Kabsch residual, low inliers** → tracking is bad (blur, glare, too-fast spin, sparse speckle) or the yoke mask is leaking non-surface pixels. Check the overlay video first.
- **`α` from top and bottom disagree** → `R_bc` is wrong, or one hemisphere's segmentation is contaminated.
- **`gamma_residual` grows over time** → drift accumulation, or the physical motion violated the `Rx(α)Rz(β)` model (e.g., wobble in the mount). If the user only needs velocities, switch to the per-increment (non-integrated) readout and drift disappears.
- **Point count collapses near the limb** → re-seeding isn't aggressive enough, or `limb_cull_deg` is too strict; also a sign the camera sees too small a cap (raise the viewpoint / reduce grazing angle).
- **Stage B passes but Stage E fails** → the failure is in circle fit or axis calibration, not the core maths. Debug the front end in isolation against synthetic truth.
- **Everything passes on synthetic, real clip is noisy** → real-world issues the synthetic didn't model: rolling-shutter warping (needs a global-shutter camera), specular glare (matte finish), or motion blur (shorter exposure / more light / higher FPS). Feed the observed Δ per frame back into the Stage D curve to confirm the FPS is adequate.

---

### Summary for the agent

Build the geometry + Kabsch core, prove it with numeric unit tests, then with a synthetic rotating speckle sphere whose rotation you control, then exercise the real-data front end (circle fit + axis calibration) still against synthetic truth — and only then run real footage, judging it by self-consistency and a loop-closure clip. The synthetic harness is not optional scaffolding; it is how "can it correctly track and measure the setup?" gets answered.
