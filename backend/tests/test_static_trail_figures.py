from __future__ import annotations

import base64
import io
import tempfile
import unittest
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import numpy as np

from featurewind.visualization.static_trail_figures import (
    export_static_trail_figures,
    resolve_export_feature_colors,
    sample_trail_points,
    select_target_and_controls,
)

matplotlib.use("Agg")


class StaticTrailFigureExportTest(unittest.TestCase):
    def test_sample_trail_points_caps_dense_path_to_twelve_samples(self) -> None:
        points = np.column_stack([
            np.linspace(0.0, 3.0, 40),
            np.linspace(0.0, 1.5, 40),
        ])
        sampled, progress = sample_trail_points(points)
        self.assertEqual(sampled.shape, (12, 2))
        self.assertEqual(progress.shape, (12,))
        self.assertAlmostEqual(float(progress[0]), 0.0)
        self.assertAlmostEqual(float(progress[-1]), 1.0)

    def test_sample_trail_points_accepts_point_objects(self) -> None:
        points = [
            {"x": 0.0, "y": 0.0},
            {"x": 0.5, "y": 0.25},
            {"x": 1.0, "y": 0.5},
            {"x": 1.5, "y": 0.75},
            {"x": 2.0, "y": 1.0},
            {"x": 2.5, "y": 1.25},
            {"x": 3.0, "y": 1.5},
            {"x": 3.5, "y": 1.75},
        ]
        sampled, progress = sample_trail_points(points)
        self.assertEqual(sampled.shape, (8, 2))
        self.assertEqual(progress.shape, (8,))
        self.assertAlmostEqual(float(sampled[0, 0]), 0.0)
        self.assertAlmostEqual(float(sampled[-1, 0]), 3.5)

    def test_select_target_and_controls_prefers_monotonic_active_feature(self) -> None:
        sampled_values = np.asarray(
            [
                [0.10, 0.80, 0.45, 0.50],
                [0.18, 0.79, 0.47, 0.51],
                [0.27, 0.78, 0.46, 0.49],
                [0.39, 0.81, 0.44, 0.50],
                [0.52, 0.80, 0.43, 0.48],
                [0.63, 0.79, 0.44, 0.49],
                [0.74, 0.82, 0.46, 0.50],
                [0.88, 0.81, 0.45, 0.51],
            ],
            dtype=float,
        )
        feature_names = ["target", "control_a", "control_b", "same_color"]
        feature_colors = ["#4477AA", "#EE6677", "#228833", "#4477AA"]

        target_idx, control_indices, scores = select_target_and_controls(
            sampled_values,
            feature_names,
            feature_colors,
            active_feature_indices=[0, 1],
            preferred_feature_index=None,
            max_controls=2,
        )

        self.assertEqual(target_idx, 0)
        self.assertEqual(control_indices, [2, 1])
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[0], scores[2])

    def test_resolve_export_feature_colors_overrides_kernel_length(self) -> None:
        colors = resolve_export_feature_colors(
            ["kernel_width", "kernel_length", "area"],
            ["#111111", "#222222", "#333333"],
        )

        self.assertEqual(colors, ["#111111", "#96360e", "#333333"])

    def test_export_static_trail_figures_writes_png_and_pdf(self) -> None:
        positions = np.asarray(
            [
                [0.0, 0.0],
                [0.2, 0.1],
                [0.4, 0.2],
                [0.6, 0.3],
                [0.8, 0.4],
                [1.0, 0.5],
            ],
            dtype=float,
        )
        feature_values = np.asarray(
            [
                [0.10, 0.70, 0.55],
                [0.22, 0.68, 0.54],
                [0.34, 0.67, 0.56],
                [0.51, 0.69, 0.55],
                [0.72, 0.68, 0.54],
                [0.90, 0.67, 0.56],
            ],
            dtype=float,
        )
        trails = [
            {
                "pointIndex": 2,
                "featureIndex": 0,
                "points": [{"x": float(x), "y": float(y)} for x, y in positions.tolist()],
            }
        ]
        buffer = io.BytesIO()
        canvas = np.ones((12, 18, 3), dtype=float)
        canvas[..., 0] = 0.95
        canvas[..., 1] = 0.97
        canvas[..., 2] = 1.0
        mpimg.imsave(buffer, canvas, format="png")
        canvas_snapshot = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_static_trail_figures(
                dataset_name="demo_dataset",
                positions=positions,
                feature_values=feature_values,
                feature_names=["target", "control_a", "control_b"],
                feature_colors=["#4477AA", "#EE6677", "#228833"],
                trails=trails,
                active_feature_indices=[0],
                canvas_snapshot_data_url=canvas_snapshot,
                canvas_view={"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 0.5},
                output_root=Path(tmpdir),
            )

            self.assertEqual(len(result["figures"]), 1)
            png_path = Path(result["figures"][0]["png"])
            pdf_path = Path(result["figures"][0]["pdf"])
            self.assertTrue(png_path.exists())
            self.assertTrue(pdf_path.exists())
            self.assertEqual(result["figures"][0]["targetFeature"], "target")


if __name__ == "__main__":
    unittest.main()
