# Learned point trackers for the caster recordings

Initial literature snapshot: **2026-09-18**. The comparisons below motivated the experiment. CoTracker3 has subsequently been installed and GPU-tested; see [implementation](../IMPLEMENTATION.md) and [measured results](../RUN_RESULTS.md). Other listed trackers remain research candidates. The accepted `ball_caster_dual_cam` pipeline is unchanged. The executed CoTracker commit, checkpoint hash and preprocessing are pinned in `model_manifest.json`.

The strongest first comparison is **CoTracker3 offline versus BootsTAPIR offline**, using identical caster crops and material-point queries. Their useful contribution would be recovering the identity of visible surface marks across difficult intervals. The existing calibrated two-camera geometry should still decide whether those observations support a common physical trajectory. This recommendation is an engineering inference, not a demonstrated improvement on these videos.

## What the models offer

| Candidate | Relevant mechanism | Why test it here | Principal constraint |
| --- | --- | --- | --- |
| CoTracker3 offline | Joint point tracking with context across the supplied clip | Other visible marks and later sharp frames may help preserve a mark's identity across blur or occlusion | Full-clip processing consumes memory; plausible hidden trajectories are predictions |
| CoTracker3 online | Overlapping windows with persistent state | Useful runtime/memory baseline or eventual live operation | Forward processing has less future context |
| TAPIR / BootsTAPIR offline | Per-frame query matching followed by temporal refinement; BootsTAPIR adds training on real videos | Global matching provides a complementary way to reacquire a mark after local tracking fails | Repeated texture and occluding edges can still produce wrong identities |
| Causal BootsTAPIR | Causal refinement and cached state | Alternative live baseline | The causal model is a separate configuration/checkpoint |

The CoTracker3 authors report improved occlusion and long-term tracking with offline processing, and benchmark improvements over earlier trackers. Offline processes the full supplied sequence; online advances overlapping windows. Its confidence target is whether a prediction falls within **12 pixels** of ground truth, and its real-video fine-tuning freezes the visibility/confidence head. These are learned classification scores, not a guaranteed native-image error bound or a pixel covariance. The authors also remove the explicit global matching stage used by TAPIR/BootsTAPIR. None of these results establishes accuracy on this caster. [CoTracker3 paper, sections 3–4](https://arxiv.org/pdf/2410.11831).

TAPIR separates per-frame matching from iterative temporal refinement. This motivates testing it for reacquisition independently of CoTracker's joint context; it does not establish that TAPIR will reacquire this sphere more accurately. [TAPIR paper](https://arxiv.org/abs/2306.08637).

BootsTAP trains with unlabeled real videos and student–teacher consistency, including resizing and JPEG corruption. The paper discusses a background/occluding-boundary bias and a training adjustment called “snap to occluder.” Thus a sharp yoke edge beside a blurred shell is a particularly important failure case to inspect. Its released model adds higher-resolution and longer-clip training. These are not experiments on exposure-integrated rotational blur, and the paper supplies no caster-specific blur tolerance. [BootsTAP paper, sections 3 and 5, appendix B.4](https://arxiv.org/html/2402.00847v2).

The current official TAP repository also contains **TAPNext and TAPNext++**. BootsTAPIR is the requested close comparator, not a claim about the newest available tracker in 2026. Its README recommends 512×512 inference for BootsTAPIR. Benchmark tables distinguish first-frame and strided query protocols; scores from those protocols should not be mixed. [Official TAP repository](https://github.com/google-deepmind/tapnet).

## API details that affect correctness

**CoTracker3:** The standard predictor uses queries `(frame_index, x, y)`, rescales images and query coordinates to its model resolution, then rescales output coordinates. Explicit `backward_tracking=True` invokes a reverse pass for frames before a later query; the default is false. The wrapper also overwrites the query-frame prediction with the supplied query and marks it visible, so query-frame agreement is not independent validation. Offline returns a binary visibility test at `>0.9` and discards the separate raw confidence output. Online combines raw visibility and confidence and thresholds their product at `>0.6`. A research adapter should preserve the underlying raw outputs as well as the wrapper flag. [Official predictor implementation](https://github.com/facebookresearch/co-tracker/blob/main/cotracker/predictor.py).

The default model resolution is **384×512, height×width**. Returning coordinates in a 1920×1080 input frame does not restore detail discarded by resizing. The online predictor's usual window is 16 frames with an 8-frame step. [Model defaults](https://github.com/facebookresearch/co-tracker/blob/main/cotracker/models/core/cotracker/cotracker3_online.py), [factory configuration](https://github.com/facebookresearch/co-tracker/blob/main/hubconf.py).

Offline CNN extraction has a default 200-frame feature chunk, but concatenates those features for full-sequence tracking. This is **not** an automatic bounded-memory 200-frame tracking strategy. Likewise, the factory's 60-frame offline window parameter is not a documented hard maximum input length. [Offline implementation](https://github.com/facebookresearch/co-tracker/blob/main/cotracker/models/core/cotracker/cotracker3_offline.py).

**TAPIR/BootsTAPIR:** Queries use `(frame_index, y, x)`; output tracks use `(x, y)`. The official PyTorch demonstration normalizes image values to `[-1, 1]` and defines visibility as:

```text
(1 - sigmoid(occlusion)) * (1 - sigmoid(expected_dist)) > 0.5
```

Preserve both raw logits and the combined score. Despite its name, `expected_dist` is an error-related logit, not a distance in millimetres or a calibrated covariance. [Official PyTorch demo](https://github.com/google-deepmind/tapnet/blob/main/colabs/torch_tapir_demo.ipynb), [model output definitions](https://github.com/google-deepmind/tapnet/blob/main/tapnet/torch/tapir_model.py).

The model starts at 256×256 and supports higher-resolution refinement; query chunking helps memory. The official coordinate conversion uses grid-size ratios, whereas CoTracker uses an align-corners resize convention. A common adapter must test both transforms explicitly rather than silently applying one library's scaling formula to the other. [TAPIR PyTorch utilities](https://github.com/google-deepmind/tapnet/blob/main/tapnet/torch/utils.py), [model implementation](https://github.com/google-deepmind/tapnet/blob/main/tapnet/torch/tapir_model.py).

## Compute, checkpoints, and licenses

The CoTracker README strongly recommends GPU inference and describes CUDA PyTorch setup; small tasks can run on CPU. It does not establish a universal VRAM minimum for these videos. [CoTracker installation and usage](https://github.com/facebookresearch/co-tracker).

The BootsTAP authors report an informal **A100/JAX**, post-compilation result of 5.6 seconds for 10,000 points in a 50-frame 256×256 clip, and 30.1 fps for 400 causal tracks at that resolution. This is a hardware/configuration-specific result, not expected Windows CPU or full-HD performance. [BootsTAP paper, section 5](https://arxiv.org/html/2402.00847v2).

| Official asset | Download URL | License evidence |
| --- | --- | --- |
| CoTracker3 scaled offline, about 102 MB | [scaled_offline.pth](https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth) | Official model repository lists CC-BY-NC-4.0 |
| CoTracker3 scaled online, about 102 MB | [scaled_online.pth](https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth) | Same model repository |
| BootsTAPIR offline, PyTorch | [bootstapir_checkpoint_v2.pt](https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt) | Current TAP README explicitly licenses linked checkpoints Apache-2.0 |
| BootsTAPIR causal, PyTorch | [causal_bootstapir_checkpoint.pt](https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt) | Same checkpoint statement |
| TAPIR, PyTorch | [tapir_checkpoint_panning.pt](https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt) | Same checkpoint statement |

Sources: [official CoTracker model files and sizes](https://huggingface.co/facebook/cotracker3/tree/main), [CoTracker license](https://github.com/facebookresearch/co-tracker/blob/main/LICENSE.md), [TAP checkpoint list and licensing](https://github.com/google-deepmind/tapnet#license-and-disclaimer). Most CoTracker repository code is CC-BY-NC, with exceptions for some borrowed components; TAP repository software is Apache-2.0 except separately licensed TAPVid-3D material. These distinctions should be retained in any future experiment's dependency record.

A practical compute probe should measure 100–300 queried points on one cropped camera clip, then increase duration. Record device, peak VRAM, wall time, resolution, point count, precision, and temporal length. Do not infer inference memory from checkpoint size. The CoTracker probe is now complete: 60 frames and 192 queries used 3.00 GB allocated / 4.28 GB reserved CUDA memory. Exact execution details and full-recording measurements are in [RUN_RESULTS.md](../RUN_RESULTS.md).

## Proposed caster experiment

1. **Track each camera separately.** Preserve original source-frame IDs, timestamp sidecars, calibration, and the fitted inter-camera time offset. These are temporal monocular trackers, not stereo-correspondence solvers. Never concatenate cam0 and cam1 as consecutive temporal frames.
2. **Crop the caster with a fixed documented mapping.** Start from the same distortion convention as the geometric pipeline. Record crop origin, padding, resize, pixel-centre convention, and the inverse map to full-image coordinates. Allocate useful pixels to the ball instead of the room. Keep an uncropped comparison to detect crop-context effects.
3. **Query material marks in sharp frames on both sides of a failure interval.** Distribute queries across each shell. Exclude rim, silhouette, gap, yoke, highlights, and shadows: these are not fixed material locations. Test later queries with backward tracking. Keep camera-local IDs and shell labels; equal query indices in two cameras do not imply a common physical point.
4. **Retain observations and predictions separately.** Store native coordinates, raw model scores, visibility, camera/frame/time, query origin, crop transform, and model identity. An occluded model trajectory can initialize a fit but must not become an independent measured point. Reacquired visible points can create a useful long-range identity constraint only after verification.
5. **Use the existing physical checks.** Validate shell membership, ray–surface intersection, incidence/conditioning, material latitude consistency, and reprojection residuals. Compare forward and backward identities and inspect sharp-frame patches. Refine candidate locations at native resolution where the image supports it. Do not let both cameras sharing the same learned failure manufacture confidence.
6. **Keep temporal chunk connections explicit.** If memory requires overlapping clips, verify shared material identities in the overlap. Newly querying a predicted endpoint does not prove continuity of phase across an unseen interval. Retain unresolved turn count when no anchored correspondence closes that gap.

A useful adapter record is `(camera, source_frame, timestamp, track_id, shell, x_native, y_native, visibility_score, confidence_score, query_frame, is_model_prediction)`, plus a separate run manifest with transforms and model hashes. An explicit measurement-acceptance field should remain downstream of these raw observations.

## How to decide whether it actually helps

Evaluate difficult run windows around **10.8–11.6 s** and **19.1–19.4 s**, then the full run if the small experiment succeeds. Include the supplied roll and swivel calibration clips as controls. Use identical sharp-frame query points for the classical baseline, CoTracker3 offline, and BootsTAPIR offline; independently annotate a subset of visible held-out frames. Hide those annotations from tuning and fitting.

Report native-pixel error on visible marks, identity switches after reappearance, accepted tracks per shell and camera, longest verified identity span, false-visible predictions, and true cross-gap phase constraints. Also report runtime/VRAM and the fraction of supported versus predicted orientation. Pure-roll/swivel cross-talk is a useful diagnostic; it is not absolute ground truth for the run. Low fitted reprojection error alone is not proof of angular accuracy.

The criterion for success is **more correct, independently supported material correspondence across the current failures at comparable false-acceptance rates**. A smoother animation, more output points, or a finite angle at every frame is insufficient.

Engineering limit: these models may use sharp frames before and after a blurred interval to infer identity and motion. They do not establish that a completely smeared or repeatedly patterned shell has a unique observable phase or turn count. Neither cited paper provides a quantitative rotational-motion-blur guarantee for this apparatus. Deblurred or learned hidden positions should therefore remain candidate evidence until checked against actual visible marks and calibrated geometry.
