from __future__ import annotations

import unittest

import numpy as np

from featurewind.eval.synthetic_global_cases import (
    build_curved_global_umap_case,
    build_near_linear_global_umap_case,
    fit_global_umap_case,
)


class SyntheticGlobalCasesTest(unittest.TestCase):
    def test_case_generation_is_deterministic(self) -> None:
        case_a = build_curved_global_umap_case(seed=0)
        case_b = build_curved_global_umap_case(seed=0)
        np.testing.assert_allclose(case_a.X, case_b.X)
        self.assertEqual(case_a.eval_feature_indices, case_b.eval_feature_indices)

    def test_fitted_umap_pipeline_is_deterministic_for_fixed_seed(self) -> None:
        spec = build_near_linear_global_umap_case(seed=0)
        fitted_a = fit_global_umap_case(spec, seed=3, dr_backend="umap", fd_epsilon_frac=0.02)
        fitted_b = fit_global_umap_case(spec, seed=3, dr_backend="umap", fd_epsilon_frac=0.02)
        np.testing.assert_allclose(fitted_a.Y, fitted_b.Y, atol=1e-6)
        np.testing.assert_allclose(fitted_a.grad_vectors, fitted_b.grad_vectors, atol=1e-6)

    def test_oracle_trail_anchors_step_zero_to_fitted_embedding(self) -> None:
        spec = build_curved_global_umap_case(seed=0)
        fitted = fit_global_umap_case(spec, seed=0, dr_backend="umap", fd_epsilon_frac=0.02)
        delta = fitted.step_delta(0, 0.1)
        trail = fitted.oracle_trail(10, 0, delta=delta, steps=4)
        np.testing.assert_allclose(trail[0], fitted.Y[10], atol=1e-8)
        self.assertEqual(trail.shape, (5, 2))

    def test_finite_difference_local_vectors_are_nonzero_for_eval_feature(self) -> None:
        spec = build_curved_global_umap_case(seed=0)
        fitted = fit_global_umap_case(spec, seed=0, dr_backend="umap", fd_epsilon_frac=0.02)
        magnitudes = np.linalg.norm(fitted.grad_vectors[:, 0, :], axis=1)
        self.assertGreater(float(np.median(magnitudes)), 1e-4)


if __name__ == "__main__":
    unittest.main()
