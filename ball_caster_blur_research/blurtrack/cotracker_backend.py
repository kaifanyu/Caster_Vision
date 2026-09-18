"""Local CoTracker3 offline inference with unthresholded model evidence.

The adapter follows the official align-corners coordinate mapping, but does not
overwrite query-frame predictions or discard the model's confidence channel.
All returned coordinates are in the supplied RGB image grid (crop mapping and
camera calibration belong to the caller). No network or torch.hub is used.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_f


def _load_local_model(repo: Path, checkpoint: Path):
    """Import the explicitly selected local checkout, then load only tensors."""
    if not (repo / "cotracker/models/core/cotracker/cotracker3_offline.py").is_file():
        raise FileNotFoundError(f"CoTracker3 checkout not found: {repo}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Local CoTracker3 checkpoint not found: {checkpoint}")
    previous = sys.modules.get("cotracker")
    if previous is not None:
        origin = getattr(previous, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(repo):
            raise RuntimeError("A different cotracker package is already imported; use a fresh process")
    sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module("cotracker.models.core.cotracker.cotracker3_offline")
        utilities = importlib.import_module("cotracker.models.core.model_utils")
    finally:
        sys.path.remove(str(repo))
    if not Path(module.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("Imported CoTracker did not originate in the requested checkout")
    model = module.CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    return model, utilities.get_points_on_a_grid


@contextmanager
def _seeded_inference(seed: int, device: torch.device):
    """Restore RNG/backend flags on exit; fixed hardware remains a requirement."""
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    old_benchmark = torch.backends.cudnn.benchmark
    old_deterministic = torch.backends.cudnn.deterministic
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(seed)
            yield
    finally:
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.deterministic = old_deterministic
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32


class OfflineCoTracker:
    """Locally loaded CoTracker3; use one instance serially on its GPU.

    ``support_grid_size=6`` reproduces the official sparse-query support grid;
    zero disables it. Support tracks never appear in returned observations.
    Raw visibility/confidence are sigmoid probabilities, not calibrated errors.
    A query's own image sample is flagged because it is not an independent test
    of that supplied query, even though this adapter does not force its output.
    """

    def __init__(self, repo, checkpoint, device="cuda", *, support_grid_size=6, seed=0, iters=6, feature_chunk_size=16):
        self.repo = Path(repo).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; select device='cpu' explicitly")
        if not isinstance(support_grid_size, int) or support_grid_size < 0:
            raise ValueError("support_grid_size must be a nonnegative integer")
        if not isinstance(iters, int) or iters < 1:
            raise ValueError("iters must be a positive integer")
        if not isinstance(feature_chunk_size, int) or feature_chunk_size < 1:
            raise ValueError("feature_chunk_size must be a positive integer")
        self.support_grid_size = support_grid_size
        self.seed = int(seed)
        self.iters = iters
        self.feature_chunk_size = feature_chunk_size
        # Construction can initialize random parameters before loading weights.
        with _seeded_inference(self.seed, self.device):
            self.model, self._grid = _load_local_model(self.repo, self.checkpoint)
            self.model = self.model.eval().to(self.device)
        self.model_resolution = tuple(int(v) for v in self.model.model_resolution)
        if len(self.model_resolution) != 2 or min(self.model_resolution) < 2:
            raise ValueError("Invalid model_resolution; expected (height, width)")

    @staticmethod
    def _validate(frames, queries):
        frames = np.asarray(frames)
        queries = np.asarray(queries)
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError("frames must be uint8 RGB with shape [T,H,W,3]")
        count, height, width, _ = frames.shape
        if count < 2 or min(height, width) < 2:
            raise ValueError("frames need T>=2, H>=2, W>=2")
        if queries.ndim != 2 or queries.shape[1] != 3 or len(queries) == 0:
            raise ValueError("queries must have nonempty shape [N,3], ordered (t,x,y)")
        if not (np.issubdtype(queries.dtype, np.integer) or np.issubdtype(queries.dtype, np.floating)) or not np.isfinite(queries).all():
            raise ValueError("queries must contain finite numeric coordinates")
        t, x, y = queries.T
        if np.any(t != np.floor(t)) or np.any(t < 0) or np.any(t >= count):
            raise ValueError("query times must be integer source-frame indices within this clip")
        if np.any(x < 0) or np.any(x > width - 1) or np.any(y < 0) or np.any(y > height - 1):
            raise ValueError("query (x,y) lies outside the input image grid")
        return frames, queries.astype(np.float32, copy=True)

    def _run_model(self, video, queries):
        output = self.model(video=video, queries=queries, iters=self.iters, fmaps_chunk_size=self.feature_chunk_size)
        if not isinstance(output, (tuple, list)) or len(output) < 3:
            raise RuntimeError("Expected raw CoTracker3 tracks, visibility, and confidence")
        tracks, visibility, confidence = output[:3]
        shape = (1, video.shape[1], queries.shape[1])
        if tuple(tracks.shape) != shape + (2,) or tuple(visibility.shape) != shape or tuple(confidence.shape) != shape:
            raise RuntimeError("Unexpected CoTracker3 output shape")
        if not all(torch.isfinite(value).all().item() for value in (tracks, visibility, confidence)):
            raise RuntimeError("CoTracker3 produced nonfinite predictions")
        if any(torch.any((value < 0) | (value > 1)).item() for value in (visibility, confidence)):
            raise RuntimeError("Expected raw sigmoid probabilities in [0,1]")
        return tracks, visibility, confidence

    def predict(self, frames_rgb_uint8, queries, backward=True, *, verify_reverse=False) -> dict[str, Any]:
        """Predict [T,N,2] coordinates and [T,N] raw visibility/confidence.

        The reverse pass fills only samples before each query, including the raw
        confidence channel. With ``verify_reverse``, also return the disagreement
        between opposite temporal orderings. That quantity is a model consistency
        diagnostic, not an independent measurement or endpoint roundtrip test.
        Query-frame predictions remain raw and are marked ``is_query_frame``.
        Runtime includes preprocessing, transfer, inference and output transfer;
        CUDA peak allocation includes the model, with an incremental peak also
        reported. Calls must not run concurrently with other GPU experiments.
        """
        frames, queries = self._validate(frames_rgb_uint8, queries)
        count, height, width, _ = frames.shape
        n_queries = len(queries)
        query_frames = queries[:, 0].astype(np.int64)
        before_query = np.arange(count)[:, None] < query_frames[None]
        is_query = np.arange(count)[:, None] == query_frames[None]
        use_reverse = bool(verify_reverse or (backward and before_query.any()))
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(self.device)
            allocated_start = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            allocated_start = 0
        started = time.perf_counter()
        with _seeded_inference(self.seed, self.device):
            # Resize on CPU before transfer: do not retain full native-resolution
            # float video on the GPU alongside its model-resolution counterpart.
            video = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
            video = torch_f.interpolate(video, size=self.model_resolution, mode="bilinear", align_corners=True)
            video = video.unsqueeze(0).to(self.device)
            scale = torch.tensor([(self.model_resolution[1] - 1) / (width - 1),
                                  (self.model_resolution[0] - 1) / (height - 1)], device=self.device)
            model_queries = torch.tensor(queries, device=self.device).unsqueeze(0)
            model_queries[..., 1:] *= scale
            if self.support_grid_size:
                points = self._grid(self.support_grid_size, self.model_resolution, device=self.device)
                support = torch.cat((torch.zeros_like(points[..., :1]), points), dim=-1)
                model_queries = torch.cat((model_queries, support), dim=1)
            tracks, visibility, confidence = self._run_model(video, model_queries)
            # Remove support outputs immediately; preserve original model values.
            tracks = tracks[:, :, :n_queries]
            visibility = visibility[:, :, :n_queries]
            confidence = confidence[:, :, :n_queries]
            reverse_error = None
            if use_reverse:
                reverse_queries = model_queries.clone()
                reverse_queries[..., 0] = count - 1 - reverse_queries[..., 0]
                rev_tracks, rev_visibility, rev_confidence = self._run_model(video.flip(1), reverse_queries)
                rev_tracks = rev_tracks[:, :, :n_queries].flip(1)
                rev_visibility = rev_visibility[:, :, :n_queries].flip(1)
                rev_confidence = rev_confidence[:, :, :n_queries].flip(1)
                if verify_reverse:
                    reverse_error = torch.linalg.vector_norm((tracks - rev_tracks) / scale, dim=-1)[0].cpu().numpy().copy()
                if backward:
                    mask = torch.tensor(before_query, device=self.device).unsqueeze(0)
                    tracks = torch.where(mask[..., None], rev_tracks, tracks)
                    visibility = torch.where(mask, rev_visibility, visibility)
                    confidence = torch.where(mask, rev_confidence, confidence)
            result = {
                "tracks": (tracks[0] / scale).cpu().numpy().copy(),
                "visibility": visibility[0].cpu().numpy().copy(),
                "confidence": confidence[0].cpu().numpy().copy(),
                "is_query_frame": is_query,
                "independent_of_query": ~is_query,
                "query_frame_forced": False,
                "direction": np.where(before_query & backward, -1, 1).astype(np.int8),
                "queries": queries.copy(),
            }
            if reverse_error is not None:
                result["reverse_disagreement_px"] = reverse_error
        if cuda:
            torch.cuda.synchronize(self.device)
        runtime = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(self.device) if cuda else 0
        reserved = torch.cuda.max_memory_reserved(self.device) if cuda else 0
        metrics = {
            "runtime_s": runtime,
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name(self.device) if cuda else "cpu",
            "cuda_peak_allocated_bytes": int(peak),
            "cuda_peak_reserved_bytes": int(reserved),
            "cuda_allocated_start_bytes": int(allocated_start),
            "cuda_peak_incremental_bytes": int(max(0, peak - allocated_start)),
            "input_shape": list(frames.shape),
            "model_resolution": list(self.model_resolution),
            "n_queries": n_queries,
            "support_grid_size": self.support_grid_size,
            "feature_chunk_size": self.feature_chunk_size,
            "model_passes": 2 if use_reverse else 1,
            "seed": self.seed,
            "precision": "float32",
            "query_frame_forced": False,
        }
        result["score"] = result["visibility"] * result["confidence"]
        result["metrics"] = metrics
        result["runtime_s"] = runtime
        result["cuda_peak_allocated_bytes"] = int(peak)
        return result
