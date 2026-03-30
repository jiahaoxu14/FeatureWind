from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from featurewind.preprocessing.csv_label_utils import humanize_point_labels, load_csv_features_and_labels
from featurewind.preprocessing.dataset_layout_utils import (
    compute_horizontal_alignment_rotation,
    orient_dataset_for_display,
)
from featurewind.visualization.color_system import apply_dataset_feature_color_overrides


class RoutesCsvLabelsTest(unittest.TestCase):
    def test_load_csv_features_and_labels_preserves_string_labels(self) -> None:
        df = pd.DataFrame(
            {
                "label": ["Kama", "Rosa", "Canadian"],
                "area": [0.1, 0.2, 0.3],
                "perimeter": [0.4, 0.5, 0.6],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "seeds.csv"
            df.to_csv(csv_path, index=False)
            points, col_labels, point_labels = load_csv_features_and_labels(str(csv_path))

        self.assertEqual(col_labels, ["area", "perimeter"])
        self.assertEqual(point_labels, ["Kama", "Rosa", "Canadian"])
        self.assertEqual(points, [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])

    def test_maybe_humanize_point_labels_maps_seeds_codes(self) -> None:
        col_labels = [
            "area",
            "perimeter",
            "compactness",
            "kernel_length",
            "kernel_width",
            "asymmetry_coefficient",
            "groove_length",
        ]
        point_labels = [1, 2, 3, "2", "Canadian"]
        self.assertEqual(
            humanize_point_labels(col_labels, point_labels),
            ["Kama", "Rosa", "Canadian", "Rosa", "Canadian"],
        )

    def test_compute_horizontal_alignment_rotation_prefers_horizontal_layout(self) -> None:
        positions = [
            [-3.0, -0.4],
            [-2.0, -0.2],
            [-1.0, -0.1],
            [1.0, 0.1],
            [2.0, 0.2],
            [3.0, 0.4],
        ]
        rotation = compute_horizontal_alignment_rotation(positions)
        self.assertGreater(abs(rotation), 0.0)
        self.assertLess(abs(rotation), 0.3)

    def test_orient_dataset_for_display_rotates_seeds_positions_and_gradients(self) -> None:
        col_labels = [
            "area",
            "perimeter",
            "compactness",
            "kernel_length",
            "kernel_width",
            "asymmetry_coefficient",
            "groove_length",
        ]
        positions = [
            [-3.0, -0.4],
            [-2.0, -0.2],
            [-1.0, -0.1],
            [1.0, 0.1],
            [2.0, 0.2],
            [3.0, 0.4],
        ]
        grads = [[[1.0, 0.0]] for _ in positions]
        rotated_positions, rotated_grads, meta = orient_dataset_for_display(col_labels, positions, grads)

        width = float(rotated_positions[:, 0].max() - rotated_positions[:, 0].min())
        height = float(rotated_positions[:, 1].max() - rotated_positions[:, 1].min())
        self.assertGreater(width, height)
        self.assertEqual(rotated_grads.shape, (6, 1, 2))
        self.assertEqual(meta["dataset"], "seeds")
        self.assertEqual(meta["target_axis"], "horizontal")

    def test_apply_dataset_feature_color_overrides_uses_non_blue_simple2d_palette(self) -> None:
        colors = apply_dataset_feature_color_overrides(
            ["horizontal_signal", "vertical_signal"],
            ["#4477AA", "#EE6677"],
        )

        self.assertEqual(colors, ["#0f766e", "#c2410c"])

    def test_apply_dataset_feature_color_overrides_uses_dark_breast_cancer_family_palette(self) -> None:
        col_labels = [
            "radius1",
            "texture1",
            "perimeter1",
            "area1",
            "smoothness1",
            "compactness1",
            "concavity1",
            "concave_points1",
            "symmetry1",
            "fractal_dimension1",
            "radius2",
            "texture2",
            "perimeter2",
            "area2",
            "smoothness2",
            "compactness2",
            "concavity2",
            "concave_points2",
            "symmetry2",
            "fractal_dimension2",
            "radius3",
            "texture3",
            "perimeter3",
            "area3",
            "smoothness3",
            "compactness3",
            "concavity3",
            "concave_points3",
            "symmetry3",
            "fractal_dimension3",
        ]
        family_assignments = [idx % 4 for idx in range(len(col_labels))]
        colors = apply_dataset_feature_color_overrides(
            col_labels,
            ["#4477AA"] * len(col_labels),
            family_assignments=family_assignments,
        )

        self.assertEqual(colors[:4], ["#0f766e", "#7c3aed", "#b91c1c", "#166534"])
        self.assertEqual(colors[0], colors[4])
        self.assertEqual(colors[1], colors[5])


if __name__ == "__main__":
    unittest.main()
