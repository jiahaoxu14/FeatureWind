from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import generate_tangent_map as genmod
from featurewind.core.dim_reader import ProjectionRunner
from featurewind.core.tsne import tsne


def _sample_points() -> list[list[float]]:
    return [
        [0.05, 0.10, 0.20],
        [0.10, 0.15, 0.30],
        [0.15, 0.18, 0.28],
        [0.80, 0.60, 0.72],
        [0.78, 0.58, 0.70],
        [0.82, 0.66, 0.76],
    ]


class GenerateTMapTest(unittest.TestCase):
    def _cpu_only(self):
        patches = [mock.patch("torch.cuda.is_available", return_value=False)]
        try:
            import torch

            if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "is_available"):
                patches.append(mock.patch("torch.backends.mps.is_available", return_value=False))
        except Exception:
            pass
        return patches

    def test_cache_key_stability(self) -> None:
        points = np.asarray(_sample_points(), dtype=np.float32)
        key_a = genmod._build_cache_key(points, "tsne", 3.0, "balanced", 0)
        key_b = genmod._build_cache_key(points, "tsne", 3.0, "balanced", 0)
        key_c = genmod._build_cache_key(points, "tsne", 3.0, "balanced", 1)
        self.assertEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_c)

    def test_generate_tmap_is_deterministic_for_fixed_seed(self) -> None:
        df = pd.DataFrame(_sample_points(), columns=["a", "b", "c"])
        with tempfile.TemporaryDirectory() as tmpdir:
            patchers = self._cpu_only()
            for patcher in patchers:
                patcher.start()
            try:
                result_a = genmod.generate_tmap(
                    df,
                    "tsne",
                    perplexity=2.0,
                    quality="draft",
                    cache="off",
                    seed=17,
                    labels=["x"] * len(df),
                    cache_dir=tmpdir,
                )
                result_b = genmod.generate_tmap(
                    df,
                    "tsne",
                    perplexity=2.0,
                    quality="draft",
                    cache="off",
                    seed=17,
                    labels=["x"] * len(df),
                    cache_dir=tmpdir,
                )
            finally:
                for patcher in reversed(patchers):
                    patcher.stop()

        range_a = np.asarray([entry["range"] for entry in result_a["tmap"]], dtype=float)
        range_b = np.asarray([entry["range"] for entry in result_b["tmap"]], dtype=float)
        tangent_a = np.asarray([entry["tangent"] for entry in result_a["tmap"]], dtype=float)
        tangent_b = np.asarray([entry["tangent"] for entry in result_b["tmap"]], dtype=float)
        np.testing.assert_allclose(range_a, range_b, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(tangent_a, tangent_b, atol=1e-6, rtol=1e-6)

    def test_cache_policy_auto_off_and_refresh(self) -> None:
        points = _sample_points()
        feature_columns = ["a", "b", "c"]
        fake_positions = np.asarray([[idx * 0.1, idx * 0.2] for idx in range(len(points))], dtype=float)
        fake_grads = np.ones((len(points), 2, len(feature_columns)), dtype=float) * 0.25
        call_counter = {"count": 0}

        def fake_compute(points_arg, projection_arg, params=None, normalize=True):
            del projection_arg, params, normalize
            call_counter["count"] += 1
            return {
                "points": np.asarray(points_arg, dtype=float),
                "positions": fake_positions.copy(),
                "grads": fake_grads.copy(),
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(genmod, "compute_tangent_map_data", side_effect=fake_compute):
                genmod.generate_tmap(
                    points,
                    "tsne",
                    quality="draft",
                    cache="auto",
                    seed=0,
                    labels=["label"] * len(points),
                    feature_columns=feature_columns,
                    cache_dir=tmpdir,
                )
                self.assertEqual(call_counter["count"], 1)

                genmod.generate_tmap(
                    points,
                    "tsne",
                    quality="draft",
                    cache="auto",
                    seed=0,
                    labels=["label"] * len(points),
                    feature_columns=feature_columns,
                    cache_dir=tmpdir,
                )
                self.assertEqual(call_counter["count"], 1)

                genmod.generate_tmap(
                    points,
                    "tsne",
                    quality="draft",
                    cache="refresh",
                    seed=0,
                    labels=["label"] * len(points),
                    feature_columns=feature_columns,
                    cache_dir=tmpdir,
                )
                self.assertEqual(call_counter["count"], 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(genmod, "compute_tangent_map_data", side_effect=fake_compute):
                genmod.generate_tmap(
                    points,
                    "tsne",
                    quality="draft",
                    cache="off",
                    seed=0,
                    labels=["label"] * len(points),
                    feature_columns=feature_columns,
                    cache_dir=tmpdir,
                )
                self.assertEqual(call_counter["count"], 3)
                self.assertEqual(list(Path(tmpdir).glob("*.npz")), [])

    def test_schema_parity(self) -> None:
        patchers = self._cpu_only()
        for patcher in patchers:
            patcher.start()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                payload = genmod.generate_tmap(
                    pd.DataFrame(_sample_points(), columns=["a", "b", "c"]),
                    "mds",
                    quality="balanced",
                    cache="off",
                    seed=0,
                    labels=["one", "two", "three", "four", "five", "six"],
                    cache_dir=tmpdir,
                )
        finally:
            for patcher in reversed(patchers):
                patcher.stop()

        self.assertEqual(set(payload.keys()), {"tmap", "Col_labels"})
        self.assertEqual(payload["Col_labels"], ["a", "b", "c"])
        self.assertEqual(len(payload["tmap"]), 6)
        entry = payload["tmap"][0]
        self.assertEqual(set(entry.keys()), {"domain", "range", "tangent", "class", "label"})
        self.assertEqual(len(entry["domain"]), 3)
        self.assertEqual(len(entry["range"]), 2)
        self.assertEqual(np.asarray(entry["tangent"], dtype=float).shape, (2, 3))

    def test_full_and_chunked_jacobian_match(self) -> None:
        patchers = self._cpu_only()
        for patcher in patchers:
            patcher.start()
        try:
            params_common = {
                "perplexity": 2.0,
                "quality": "draft",
                "cache": "off",
                "seed": 5,
            }
            runner_full = ProjectionRunner(tsne, {**params_common, "jacobian_strategy": "full"})
            runner_full.calculateValues(_sample_points())

            runner_chunked = ProjectionRunner(tsne, {**params_common, "jacobian_strategy": "chunked"})
            runner_chunked.calculateValues(_sample_points())
        finally:
            for patcher in reversed(patchers):
                patcher.stop()

        np.testing.assert_allclose(runner_full.grads, runner_chunked.grads, atol=1e-5, rtol=1e-4)
        self.assertEqual(runner_full.run_stats["jacobian_strategy_used"], "full")
        self.assertEqual(runner_chunked.run_stats["jacobian_strategy_used"], "chunked")

    def test_chunked_jacobian_falls_back_to_serial(self) -> None:
        patchers = self._cpu_only()
        for patcher in patchers:
            patcher.start()
        try:
            runner = ProjectionRunner(
                tsne,
                {
                    "perplexity": 2.0,
                    "quality": "draft",
                    "cache": "off",
                    "seed": 9,
                    "jacobian_strategy": "chunked",
                },
            )
            with mock.patch.object(
                ProjectionRunner,
                "_compute_jacobian_tsne_chunked",
                side_effect=RuntimeError("batched grad unavailable"),
            ):
                runner.calculateValues(_sample_points())
        finally:
            for patcher in reversed(patchers):
                patcher.stop()

        self.assertEqual(runner.run_stats["jacobian_strategy_used"], "serial-fallback")
        self.assertEqual(np.asarray(runner.grads).shape, (6, 2, 3))


if __name__ == "__main__":
    unittest.main()
