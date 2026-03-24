from __future__ import annotations

import unittest

import numpy as np

from featurewind.eval.advection import build_field_context, build_point_cloud_field_context


class AdvectionTest(unittest.TestCase):
    def test_integrate_trail_keeps_expected_sample_count(self) -> None:
        u_grid = np.ones((8, 8), dtype=float) * 0.5
        v_grid = np.zeros((8, 8), dtype=float)
        context = build_field_context(u_grid, v_grid, bbox=(0.0, 1.0, 0.0, 1.0))
        result = context.integrate_trail(0.1, 0.5, delta_feature=0.1, steps=5)
        self.assertTrue(result["valid"])
        self.assertEqual(result["points"].shape, (6, 2))
        self.assertAlmostEqual(result["points"][-1, 0], 0.35, places=6)

    def test_point_cloud_sampler_tracks_constant_field(self) -> None:
        positions = np.array(
            [
                [0.1, 0.1],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.9, 0.9],
                [0.5, 0.5],
            ],
            dtype=float,
        )
        vectors = np.tile(np.array([[0.4, 0.0]], dtype=float), (positions.shape[0], 1))
        context = build_point_cloud_field_context(
            positions,
            vectors,
            bbox=(0.0, 1.0, 0.0, 1.0),
            k_neighbors=4,
            support_k=3,
            support_percentile=1.0,
            query_support_factor=2.0,
            bbox_margin_scale=0.0,
        )
        eligibility = context.check_start_eligibility(0.5, 0.5, point_idx=4)
        self.assertTrue(eligibility["valid"])
        result = context.integrate_trail(0.5, 0.5, delta_feature=0.1, steps=5)
        self.assertTrue(result["valid"])
        self.assertEqual(result["points"].shape, (6, 2))
        self.assertAlmostEqual(result["points"][-1, 0], 0.7, places=4)


if __name__ == "__main__":
    unittest.main()
