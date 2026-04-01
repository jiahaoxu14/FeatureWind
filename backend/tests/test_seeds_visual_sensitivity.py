from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from featurewind.analysis.seeds_visual_sensitivity import (
    GridConditionContext,
    SensitivityCondition,
    build_oat_conditions,
    build_static_trail,
    compute_field_cosine_mean,
    compute_support_iou,
    compute_trail_distance_metrics,
    run_seeds_visual_sensitivity,
)


def _has_scipy() -> bool:
    try:
        import scipy  # noqa: F401
    except Exception:
        return False
    return True


class SeedsVisualSensitivityTest(unittest.TestCase):
    def test_build_oat_conditions_returns_fixed_reference_and_oat_order(self) -> None:
        conditions = build_oat_conditions(
            reference_grid_res=25,
            reference_interpolation="linear",
            reference_mask_radius=1,
        )

        self.assertEqual([condition.condition_id for condition in conditions], [
            "reference",
            "grid_15",
            "grid_20",
            "grid_30",
            "linear_nearest",
            "nearest",
            "radius_0",
            "radius_2",
            "radius_3",
        ])
        self.assertTrue(conditions[0].is_reference)
        self.assertEqual(conditions[0].grid_res, 25)
        self.assertEqual(conditions[0].interpolation_method, "linear")
        self.assertEqual(conditions[0].mask_radius, 1)
        self.assertEqual(conditions[4].interpolation_method, "linear-nearest")
        self.assertEqual(conditions[5].interpolation_method, "nearest")
        self.assertEqual(conditions[6].mask_radius, 0)
        self.assertEqual(conditions[7].mask_radius, 2)
        self.assertEqual(conditions[8].mask_radius, 3)

    def test_reference_metrics_are_identity_and_shifted_trail_increases_distance(self) -> None:
        support = np.asarray([True, False, True, True], dtype=bool)
        vectors = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=float,
        )

        self.assertEqual(compute_support_iou(support, support), 1.0)
        self.assertAlmostEqual(
            compute_field_cosine_mean(vectors, vectors, support, support),
            1.0,
            places=7,
        )

        reference = np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=float)
        identical = reference.copy()
        shifted = reference + np.asarray([0.0, 0.2], dtype=float)
        bbox = (0.0, 1.0, 0.0, 1.0)

        mean_dev_same, endpoint_same = compute_trail_distance_metrics(reference, identical, bbox=bbox, sample_count=10)
        mean_dev_shifted, endpoint_shifted = compute_trail_distance_metrics(reference, shifted, bbox=bbox, sample_count=10)

        self.assertEqual(mean_dev_same, 0.0)
        self.assertEqual(endpoint_same, 0.0)
        self.assertGreater(mean_dev_shifted, 0.0)
        self.assertGreater(endpoint_shifted, 0.0)

    def test_static_trail_rollout_is_deterministic_for_constant_field(self) -> None:
        grid_res = 10
        xs = np.linspace(0.05, 0.95, grid_res)
        ys = np.linspace(0.05, 0.95, grid_res)
        grid_x, grid_y = np.meshgrid(xs, ys)
        condition = SensitivityCondition(
            condition_id="reference",
            label="reference",
            grid_res=grid_res,
            interpolation_method="linear",
            mask_radius=1,
            is_reference=True,
        )
        context = GridConditionContext(
            condition=condition,
            bbox=(0.0, 1.0, 0.0, 1.0),
            grid_x=grid_x,
            grid_y=grid_y,
            u_grid=np.ones((grid_res, grid_res), dtype=float),
            v_grid=np.zeros((grid_res, grid_res), dtype=float),
            unmasked=np.ones((grid_res, grid_res), dtype=bool),
            weak_threshold=0.01,
            p99=1.0,
            canvas_px=600,
            speed_scale=1.0,
        )

        trail_a = build_static_trail(context, start_x=0.25, start_y=0.5, point_index=0, max_steps=5)
        trail_b = build_static_trail(context, start_x=0.25, start_y=0.5, point_index=0, max_steps=5)

        self.assertEqual(trail_a.stop_reason, "max-steps")
        self.assertEqual(trail_a.point_count, 6)
        np.testing.assert_allclose(trail_a.points, trail_b.points)
        self.assertGreater(float(trail_a.points[-1, 0]), float(trail_a.points[0, 0]))

    @unittest.skipUnless(_has_scipy(), "scipy is required for the Seeds sensitivity smoke test")
    def test_default_run_writes_figure_and_metrics_csv(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        tmap_path = repo_root / "backend/datasets/examples/seeds/seeds_tsne.tmap"

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ.setdefault("MPLCONFIGDIR", os.path.join(tmpdir, "mpl"))
            result = run_seeds_visual_sensitivity(
                tmap_path=tmap_path,
                output_dir=tmpdir,
            )

            self.assertIsNone(result.figure_png)
            self.assertIsNone(result.figure_pdf)
            self.assertTrue(result.metrics_csv.exists())
            self.assertTrue(result.summary_md.exists())
            self.assertEqual(len(result.panel_pngs), 9)
            self.assertEqual(len(result.panel_pdfs), 9)
            self.assertTrue(all(path.exists() for path in result.panel_pngs))
            self.assertTrue(all(path.exists() for path in result.panel_pdfs))

            with result.metrics_csv.open() as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 9)
            self.assertEqual([row["condition_id"] for row in rows], [
                "reference",
                "grid_15",
                "grid_20",
                "grid_30",
                "linear_nearest",
                "nearest",
                "radius_0",
                "radius_2",
                "radius_3",
            ])


if __name__ == "__main__":
    unittest.main()
