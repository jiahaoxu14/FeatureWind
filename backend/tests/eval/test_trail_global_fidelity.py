from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from featurewind.eval.metrics import compute_stepwise_case_metrics
from featurewind.eval.trail_global_fidelity import (
    AnchoredGlobalAxisCurve,
    GlobalTrailFidelityConfig,
    _choose_representatives,
    build_anchored_global_axis_path,
    build_linear_baseline_path,
    _normalize_path_length_for_display,
    _polyline_length,
    run_trail_global_fidelity,
)


class TrailGlobalFidelityTest(unittest.TestCase):
    def test_linear_baseline_path_stays_straight(self) -> None:
        start = np.array([1.0, -0.5], dtype=float)
        vector = np.array([0.2, 0.4], dtype=float)
        path = build_linear_baseline_path(start, vector, delta=0.1, steps=4)
        self.assertEqual(path.shape, (5, 2))
        np.testing.assert_allclose(path[-1], np.array([1.08, -0.34], dtype=float))

    def test_stepwise_metrics_align_with_fixed_step_index(self) -> None:
        oracle = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=float,
        )
        trail = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.1],
                [2.0, 0.1],
                [3.0, 0.1],
            ],
            dtype=float,
        )
        linear = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.2],
                [2.0, 0.2],
                [3.0, 0.2],
            ],
            dtype=float,
        )
        metrics = compute_stepwise_case_metrics(oracle, trail, linear)
        np.testing.assert_allclose(metrics["endpoint_error_trail_by_step"], np.array([0.1, 0.1, 0.1]))
        np.testing.assert_allclose(metrics["endpoint_error_baseline_by_step"], np.array([0.2, 0.2, 0.2]))
        self.assertTrue(np.all(metrics["trail_wins_endpoint_by_step"]))
        self.assertTrue(np.all(metrics["trail_wins_path_by_step"]))

    def test_anchored_global_axis_path_starts_at_actual_point(self) -> None:
        axis_curve = AnchoredGlobalAxisCurve(
            feature_idx=0,
            z_samples=np.array([-1.0, 0.0, 1.0], dtype=float),
            xy_samples=np.array(
                [
                    [-1.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                ],
                dtype=float,
            ),
            bin_median_z=np.array([-1.0, 0.0, 1.0], dtype=float),
            bin_median_xy=np.array(
                [
                    [-1.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                ],
                dtype=float,
            ),
        )
        start = np.array([0.25, 0.4], dtype=float)
        result = build_anchored_global_axis_path(start, axis_curve, steps=4, delta_t=0.25)
        path = np.asarray(result["path"], dtype=float)
        np.testing.assert_allclose(path[0], start)
        np.testing.assert_allclose(path[1], np.array([0.5, 0.4], dtype=float))

    def test_anchored_global_axis_path_can_be_shortened_to_target_step_length(self) -> None:
        axis_curve = AnchoredGlobalAxisCurve(
            feature_idx=0,
            z_samples=np.array([0.0, 1.0], dtype=float),
            xy_samples=np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                ],
                dtype=float,
            ),
            bin_median_z=np.array([0.0, 1.0], dtype=float),
            bin_median_xy=np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                ],
                dtype=float,
            ),
        )
        start = np.array([0.2, -0.3], dtype=float)
        result = build_anchored_global_axis_path(
            start,
            axis_curve,
            steps=2,
            delta_t=0.5,
            target_first_step_length=0.1,
        )
        path = np.asarray(result["path"], dtype=float)
        np.testing.assert_allclose(path[0], start)
        np.testing.assert_allclose(path[1], np.array([0.3, -0.3], dtype=float))
        self.assertAlmostEqual(float(result["scale"]), 0.2)

    def test_display_path_length_normalization_matches_target_length(self) -> None:
        path = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
            ],
            dtype=float,
        )
        normalized = _normalize_path_length_for_display(path, target_length=2.0)
        np.testing.assert_allclose(normalized[0], path[0])
        np.testing.assert_allclose(normalized[-1], np.array([1.0, 1.0], dtype=float))
        self.assertAlmostEqual(_polyline_length(normalized), 2.0)

    def test_display_path_length_normalization_keeps_zero_length_path(self) -> None:
        path = np.array(
            [
                [1.5, -0.5],
                [1.5, -0.5],
                [1.5, -0.5],
            ],
            dtype=float,
        )
        normalized = _normalize_path_length_for_display(path, target_length=3.0)
        np.testing.assert_allclose(normalized, path)

    def test_representatives_are_ranked_by_best_static_oracle_match(self) -> None:
        records = [
            {
                "dataset": "demo",
                "feature_name": "feature_0",
                "point_idx": 7,
                "path_deviation_trail_by_step": [0.20, 0.20, 0.20],
                "endpoint_error_trail_by_step": [0.20, 0.20, 0.20],
            },
            {
                "dataset": "demo",
                "feature_name": "feature_0",
                "point_idx": 3,
                "path_deviation_trail_by_step": [0.08, 0.09, 0.10],
                "endpoint_error_trail_by_step": [0.11, 0.12, 0.13],
            },
            {
                "dataset": "demo",
                "feature_name": "feature_0",
                "point_idx": 5,
                "path_deviation_trail_by_step": [0.08, 0.09, 0.12],
                "endpoint_error_trail_by_step": [0.15, 0.16, 0.17],
            },
        ]
        selected = _choose_representatives(records)
        self.assertEqual([record["point_idx"] for record in selected], [3, 5, 7])

    def test_smoke_run_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="featurewind_global_eval_") as tmpdir:
            result = run_trail_global_fidelity(
                GlobalTrailFidelityConfig(
                    output_dir=Path(tmpdir),
                    steps=4,
                    delta_frac=0.1,
                    fd_epsilon_frac=0.02,
                    datasets=("near_linear_global_umap",),
                    dr_backend="umap",
                )
            )
            self.assertGreater(result["attempted_case_count"], 0)
            self.assertGreater(result["valid_case_count"], 0)
            self.assertTrue(Path(result["summary_metrics_csv"]).exists())
            self.assertTrue(Path(result["per_case_json"]).exists())
            self.assertTrue(Path(result["step_endpoint_error_png"]).exists())
            self.assertTrue(Path(result["step_path_deviation_png"]).exists())
            self.assertTrue(Path(result["step_winrate_png"]).exists())
            self.assertTrue(Path(result["representative_paths_png"]).exists())
            self.assertTrue(Path(result["paper_summary_md"]).exists())
            with open(result["summary_metrics_csv"], "r", encoding="utf-8", newline="") as handle:
                methods = {row["method"] for row in csv.DictReader(handle)}
            self.assertEqual(methods, {"static_trail", "linear_trail", "anchored_global_nonlinear_axis"})


if __name__ == "__main__":
    unittest.main()
