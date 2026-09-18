"""CPU model-double tests; no weights, network, or GPU required."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from blurtrack.cotracker_backend import OfflineCoTracker


class FakeModel(torch.nn.Module):
    model_resolution = (5, 9)

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, *, video, queries, iters, fmaps_chunk_size):
        assert not torch.is_grad_enabled()
        assert not self.training
        assert fmaps_chunk_size == 16
        self.calls.append((video.clone(), queries.clone(), iters))
        count, n_queries = video.shape[1], queries.shape[1]
        reverse = video[0, 0, 0, 0, 0] > video[0, -1, 0, 0, 0]
        delta = 2.0 if reverse else 1.0
        coordinates = queries[..., 1:].unsqueeze(1).expand(1, count, n_queries, 2).clone() + delta
        visibility = torch.full((1, count, n_queries), .2 if reverse else .8)
        confidence = torch.full((1, count, n_queries), .3 if reverse else .7)
        return coordinates, visibility, confidence, None


def fake_grid(size, extent, device):
    assert extent == (5, 9)
    return torch.ones((1, size * size, 2), device=device)


class CoTrackerBackendTests(unittest.TestCase):
    def make_backend(self, support_grid_size=0):
        model = FakeModel()
        with patch("blurtrack.cotracker_backend._load_local_model", return_value=(model, fake_grid)):
            backend = OfflineCoTracker("unused_repo", "unused_checkpoint", "cpu", support_grid_size=support_grid_size)
        return backend, model

    @staticmethod
    def frames():
        return np.stack([np.full((9, 17, 3), i * 20, np.uint8) for i in range(4)])

    def test_coordinates_rgb_order_and_raw_query_prediction(self):
        backend, model = self.make_backend()
        frames = self.frames()
        frames[:, :, :, 1] += 3
        frames[:, :, :, 2] += 7
        query = np.array([[0, 4, 2], [0, 16, 8]], np.float32)
        query_copy = query.copy()
        output = backend.predict(frames, query)
        video, model_query, iters = model.calls[0]
        self.assertEqual(tuple(video.shape), (1, 4, 3, 5, 9))
        np.testing.assert_array_equal(video[0, 0, :, 0, 0].numpy(), [0, 3, 7])
        np.testing.assert_array_equal(model_query[0].numpy(), [[0, 2, 1], [0, 8, 4]])
        self.assertEqual(iters, 6)
        np.testing.assert_allclose(output["tracks"], np.broadcast_to(query[None, :, 1:] + 2, (4, 2, 2)))
        np.testing.assert_array_equal(query, query_copy)
        self.assertFalse(output["query_frame_forced"])
        self.assertTrue(output["is_query_frame"][0].all())
        self.assertFalse(output["independent_of_query"][0].any())
        self.assertEqual(output["metrics"]["model_passes"], 1)
        np.testing.assert_allclose(output["visibility"], .8)
        np.testing.assert_allclose(output["confidence"], .7)
        np.testing.assert_allclose(output["score"], .56)

    def test_reverse_time_query_order_and_confidence_merge(self):
        backend, model = self.make_backend(support_grid_size=2)
        queries = np.array([[2, 4, 2], [0, 6, 4]], np.float32)
        output = backend.predict(self.frames(), queries, verify_reverse=True)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0][1].shape[1], 6)
        np.testing.assert_array_equal(model.calls[1][1][0, :, 0].numpy(), [1, 3, 3, 3, 3, 3])
        self.assertEqual(output["tracks"].shape, (4, 2, 2))
        np.testing.assert_allclose(output["tracks"][:2, 0], [[8, 6], [8, 6]])
        np.testing.assert_allclose(output["tracks"][2:, 0], [[6, 4], [6, 4]])
        np.testing.assert_allclose(output["visibility"][:, 0], [.2, .2, .8, .8])
        np.testing.assert_allclose(output["confidence"][:, 0], [.3, .3, .7, .7])
        np.testing.assert_allclose(output["confidence"][:, 1], .7)
        np.testing.assert_allclose(output["reverse_disagreement_px"], np.sqrt(8), rtol=1e-6)
        np.testing.assert_array_equal(output["direction"][:, 0], [-1, -1, 1, 1])
        self.assertTrue(output["is_query_frame"][2, 0])

    def test_reverse_check_does_not_merge_when_backward_disabled(self):
        backend, _ = self.make_backend()
        output = backend.predict(self.frames(), [[3, 4, 2]], backward=False, verify_reverse=True)
        np.testing.assert_allclose(output["confidence"], .7)
        np.testing.assert_array_equal(output["direction"], 1)

    def test_align_corners_uses_actual_image_endpoints(self):
        backend, model = self.make_backend()
        frames = np.zeros((2, 11, 21, 3), np.uint8)
        frames[:, -1, -1] = [50, 100, 200]
        result = backend.predict(frames, [[0, 20, 10]])
        np.testing.assert_allclose(model.calls[0][1][0, 0].numpy(), [0, 8, 4])
        np.testing.assert_allclose(model.calls[0][0][0, 0, :, -1, -1].numpy(), [50, 100, 200])
        np.testing.assert_allclose(result["tracks"][0, 0], [22.5, 12.5])

    def test_validation_rejects_wrong_shapes_dtypes_and_query_order(self):
        backend, model = self.make_backend()
        bad = [
            (self.frames().astype(np.float32), [[0, 2, 3]]),
            (self.frames()[0], [[0, 2, 3]]),
            (self.frames()[:1], [[0, 2, 3]]),
            (self.frames(), np.zeros((0, 3))),
            (self.frames(), [[.5, 2, 3]]),
            (self.frames(), [[.999999999, 2, 3]]),
            (self.frames(), [[0, 2+1j, 3]]),
            (self.frames(), [[4, 2, 3]]),
            (self.frames(), [[0, 2, 10]]),
            (self.frames(), [[0, -1, 3]]),
            (self.frames(), [[0, float("nan"), 3]]),
        ]
        for frames, queries in bad:
            with self.subTest(shape=frames.shape, queries=queries), self.assertRaises(ValueError):
                backend.predict(frames, queries)
        self.assertEqual(len(model.calls), 0)

    def test_backend_preserves_rng_and_flags(self):
        backend, _ = self.make_backend()
        before = torch.random.get_rng_state().clone()
        benchmark = torch.backends.cudnn.benchmark
        deterministic = torch.backends.cudnn.deterministic
        output = backend.predict(self.frames(), [[0, 2, 3]])
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        self.assertEqual(torch.backends.cudnn.benchmark, benchmark)
        self.assertEqual(torch.backends.cudnn.deterministic, deterministic)
        self.assertEqual(output["metrics"]["cuda_peak_allocated_bytes"], 0)
        self.assertGreater(output["runtime_s"], 0)

    def test_model_output_contract_rejects_missing_scores_shapes_and_nonfinite(self):
        backend, model = self.make_backend()
        tracks = torch.zeros((1, 4, 1, 2))
        scores = torch.full((1, 4, 1), .5)
        invalid = [
            (tracks, scores),
            (tracks, scores[..., None], scores, None),
            (tracks * float("nan"), scores, scores, None),
            (tracks, scores, scores + 1, None),
        ]
        for output in invalid:
            with self.subTest(length=len(output)), patch.object(model, "forward", return_value=output), self.assertRaises(RuntimeError):
                backend.predict(self.frames(), [[0, 2, 3]])

    def test_local_assets_are_required_without_network_fallback(self):
        with self.assertRaises(FileNotFoundError):
            OfflineCoTracker("deliberately_missing_checkout", "missing_checkpoint", "cpu")


if __name__ == "__main__":
    unittest.main()
