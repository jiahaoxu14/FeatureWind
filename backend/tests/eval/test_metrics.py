from __future__ import annotations

import unittest

import numpy as np

from featurewind.eval.metrics import (
    compute_case_metrics,
    direction_agreement,
    endpoint_error,
    mean_path_deviation,
    total_turning_angle,
)


class MetricsTest(unittest.TestCase):
    def test_identical_paths_have_zero_error(self) -> None:
        path = np.array([[0.0, 0.0], [0.4, 0.1], [0.8, 0.2]], dtype=float)
        self.assertAlmostEqual(endpoint_error(path, path), 0.0)
        self.assertAlmostEqual(mean_path_deviation(path, path), 0.0)
        self.assertAlmostEqual(direction_agreement(path, path), 1.0)

    def test_turning_angle_detects_curvature(self) -> None:
        path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=float)
        self.assertGreater(total_turning_angle(path), 80.0)

    def test_compute_case_metrics_prefers_better_prediction(self) -> None:
        reference = np.array([[0.0, 0.0], [0.5, 0.2], [1.0, 0.5]], dtype=float)
        trail = reference.copy()
        baseline = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=float)
        metrics = compute_case_metrics(reference, trail, baseline)
        self.assertTrue(metrics["trail_wins_endpoint"])
        self.assertTrue(metrics["trail_wins_mean_path"])


if __name__ == "__main__":
    unittest.main()
