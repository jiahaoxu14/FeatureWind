from __future__ import annotations

import unittest

import numpy as np

from featurewind import config as fw_config
from featurewind.physics import grid_computation


class GridComputationTest(unittest.TestCase):
    def test_supported_interpolation_methods_produce_finite_grids(self) -> None:
        positions = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=float,
        )
        vectors = np.tile(np.asarray([[1.0, 0.0]], dtype=float), (len(positions), 1))

        fw_config.initialize_global_state()
        fw_config.set_bounding_box(positions)
        grid_x, grid_y, _cell_centers_x, _cell_centers_y = grid_computation.create_grid_coordinates(12)

        for method in grid_computation.SUPPORTED_INTERPOLATION_METHODS:
            with self.subTest(method=method):
                grid_u, grid_v = grid_computation.interpolate_feature_onto_grid(
                    positions,
                    vectors,
                    grid_x,
                    grid_y,
                    interpolation_method=method,
                )
                self.assertEqual(grid_u.shape, (12, 12))
                self.assertEqual(grid_v.shape, (12, 12))
                self.assertTrue(np.isfinite(grid_u).all())
                self.assertTrue(np.isfinite(grid_v).all())
                self.assertGreater(float(np.max(grid_u)), 0.5)

    def test_hybrid_interpolation_falls_back_to_nearest_on_degenerate_points(self) -> None:
        positions = np.asarray(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
            ],
            dtype=float,
        )
        vectors = np.tile(np.asarray([[0.75, 0.0]], dtype=float), (len(positions), 1))
        grid_x, grid_y = np.meshgrid(np.linspace(0.0, 1.0, 9), np.linspace(-0.2, 0.2, 9))

        for method in ("linear-nearest", "cubic-nearest"):
            with self.subTest(method=method):
                grid_u, grid_v = grid_computation.interpolate_feature_onto_grid(
                    positions,
                    vectors,
                    grid_x,
                    grid_y,
                    interpolation_method=method,
                )
                self.assertTrue(np.isfinite(grid_u).all())
                self.assertTrue(np.isfinite(grid_v).all())
                self.assertGreater(float(np.max(grid_u)), 0.5)

    def test_invalid_interpolation_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            grid_computation.normalize_interpolation_method("unsupported")

    def test_interpolation_hull_preserves_bridge_between_separated_clusters(self) -> None:
        positions = np.asarray(
            [
                [-1.0, -0.2],
                [-1.0, 0.2],
                [-0.75, 0.0],
                [1.0, -0.2],
                [1.0, 0.2],
                [0.75, 0.0],
            ],
            dtype=float,
        )
        all_grad_vectors = np.zeros((len(positions), 1, 2), dtype=float)
        all_grad_vectors[:, 0, 0] = 1.0

        fw_config.initialize_global_state()
        fw_config.set_bounding_box(positions)

        orig_radius = getattr(fw_config, "MASK_DILATE_RADIUS_CELLS", 1)
        orig_include_hull = getattr(fw_config, "MASK_INCLUDE_INTERPOLATION_HULL", True)
        try:
            fw_config.MASK_DILATE_RADIUS_CELLS = 0
            fw_config.MASK_INCLUDE_INTERPOLATION_HULL = False
            _, unmasked_local_only, _ = grid_computation.build_dilated_support_mask(
                positions,
                25,
                bbox=fw_config.bounding_box,
            )

            fw_config.MASK_INCLUDE_INTERPOLATION_HULL = True
            _, unmasked_with_hull, final_mask = grid_computation.build_dilated_support_mask(
                positions,
                25,
                bbox=fw_config.bounding_box,
            )

            (
                interp_u_sum,
                _interp_v_sum,
                _interp_argmax,
                _grid_x,
                _grid_y,
                _grid_u_feats,
                _grid_v_feats,
                _cell_dominant_features,
                _grid_u_all_feats,
                _grid_v_all_feats,
                cell_centers_x,
                cell_centers_y,
                _final_mask_from_build,
            ) = grid_computation.build_grids(
                positions,
                25,
                [0],
                all_grad_vectors,
                ["feature_0"],
            )
        finally:
            fw_config.MASK_DILATE_RADIUS_CELLS = orig_radius
            fw_config.MASK_INCLUDE_INTERPOLATION_HULL = orig_include_hull

        center_i = int(np.argmin(np.abs(cell_centers_y)))
        center_j = int(np.argmin(np.abs(cell_centers_x)))

        self.assertFalse(bool(unmasked_local_only[center_i, center_j]))
        self.assertTrue(bool(unmasked_with_hull[center_i, center_j]))
        self.assertFalse(bool(final_mask[center_i, center_j]))

        sample = np.asarray([[cell_centers_y[center_i], cell_centers_x[center_j]]], dtype=float)
        center_u = float(interp_u_sum(sample)[0])
        self.assertGreater(center_u, 0.5)

    def test_apply_support_mask_to_visualization_fields_masks_all_feature_layers(self) -> None:
        grid_u = np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=float,
        )
        grid_v = np.asarray(
            [
                [[0.5, 1.5], [2.5, 3.5]],
                [[4.5, 5.5], [6.5, 7.5]],
            ],
            dtype=float,
        )
        dominant = np.asarray([[0, 1], [1, 0]], dtype=int)
        final_mask = np.asarray([[False, True], [True, False]], dtype=bool)

        masked_u, masked_v, masked_dominant = grid_computation.apply_support_mask_to_visualization_fields(
            grid_u,
            grid_v,
            dominant,
            final_mask,
        )

        np.testing.assert_array_equal(masked_u[:, final_mask], 0.0)
        np.testing.assert_array_equal(masked_v[:, final_mask], 0.0)
        np.testing.assert_array_equal(masked_dominant[final_mask], -1)
        np.testing.assert_array_equal(masked_u[:, ~final_mask], grid_u[:, ~final_mask])
        np.testing.assert_array_equal(masked_v[:, ~final_mask], grid_v[:, ~final_mask])
        np.testing.assert_array_equal(masked_dominant[~final_mask], dominant[~final_mask])


if __name__ == "__main__":
    unittest.main()
