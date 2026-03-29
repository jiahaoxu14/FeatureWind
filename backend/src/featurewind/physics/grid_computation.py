"""
Grid computation module for FeatureWind.

This module handles grid building, velocity field interpolation, and spatial discretization
of gradient vector fields for visualization and particle simulation.
"""

import os
import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.spatial import Delaunay, QhullError
from .. import config

SUPPORTED_INTERPOLATION_METHODS = (
    "linear-nearest",
    "linear",
    "nearest",
    "cubic-nearest",
)


def normalize_interpolation_method(method):
    """
    Normalize and validate the requested point-to-grid interpolation method.
    """
    value = str(method or "linear-nearest").strip().lower()
    aliases = {
        "default": "linear-nearest",
        "linear_nearest": "linear-nearest",
        "linear+nearest": "linear-nearest",
        "linear+nearest-fallback": "linear-nearest",
        "cubic_nearest": "cubic-nearest",
        "cubic+nearest": "cubic-nearest",
        "cubic+nearest-fallback": "cubic-nearest",
    }
    normalized = aliases.get(value, value)
    if normalized not in SUPPORTED_INTERPOLATION_METHODS:
        raise ValueError(
            "Unsupported interpolation method "
            f"{method!r}. Supported methods: {', '.join(SUPPORTED_INTERPOLATION_METHODS)}"
        )
    return normalized




def create_grid_coordinates(grid_res):
    """
    Create cell-center grid coordinates.
    
    Args:
        grid_res (int): Grid resolution (number of cells per dimension)
        
    Returns:
        tuple: (grid_x, grid_y, cell_centers_x, cell_centers_y)
    """
    xmin, xmax, ymin, ymax = config.bounding_box
    
    # Create cell center coordinates
    cell_centers_x = np.linspace(xmin + (xmax-xmin)/(2*grid_res), 
                                 xmax - (xmax-xmin)/(2*grid_res), grid_res)
    cell_centers_y = np.linspace(ymin + (ymax-ymin)/(2*grid_res), 
                                 ymax - (ymax-ymin)/(2*grid_res), grid_res)
    grid_x, grid_y = np.meshgrid(cell_centers_x, cell_centers_y)
    
    return grid_x, grid_y, cell_centers_x, cell_centers_y


def build_occupied_cell_mask(positions, grid_res, bbox=None):
    """
    Build a boolean occupied-cell grid from point positions.

    A cell is occupied if it contains at least one data point.
    """
    xmin, xmax, ymin, ymax = bbox or config.bounding_box
    occupied = np.zeros((grid_res, grid_res), dtype=bool)
    if positions is None:
        return occupied
    cell_width = (xmax - xmin) / grid_res
    cell_height = (ymax - ymin) / grid_res
    try:
        iterable = np.asarray(positions)
    except Exception:
        iterable = positions
    for pos in iterable:
        px, py = float(pos[0]), float(pos[1])
        i = int((py - ymin) / cell_height)
        j = int((px - xmin) / cell_width)
        i = max(0, min(grid_res - 1, i))
        j = max(0, min(grid_res - 1, j))
        occupied[i, j] = True
    return occupied


def create_cell_center_mesh(grid_res, bbox=None):
    """
    Create cell-center coordinates for an arbitrary bounding box.
    """
    xmin, xmax, ymin, ymax = bbox or config.bounding_box
    cell_centers_x = np.linspace(
        xmin + (xmax - xmin) / (2 * grid_res),
        xmax - (xmax - xmin) / (2 * grid_res),
        grid_res,
    )
    cell_centers_y = np.linspace(
        ymin + (ymax - ymin) / (2 * grid_res),
        ymax - (ymax - ymin) / (2 * grid_res),
        grid_res,
    )
    return np.meshgrid(cell_centers_x, cell_centers_y)


def dilate_cell_mask(mask, radius_cells):
    """
    Dilate a cell mask by an integer Chebyshev radius using 8-neighborhood growth.
    """
    out = np.asarray(mask, dtype=bool).copy()
    radius = max(0, int(radius_cells))
    if radius == 0 or out.size == 0:
        return out
    h, w = out.shape
    for _ in range(radius):
        padded = np.pad(out, 1, mode='constant', constant_values=False)
        grown = np.zeros_like(out, dtype=bool)
        for di in range(3):
            for dj in range(3):
                grown |= padded[di:di + h, dj:dj + w]
        out = grown
    return out


def build_interpolation_hull_mask(positions, grid_res, bbox=None):
    """
    Rasterize the convex interpolation hull onto grid-cell centers.

    This keeps cells that lie inside the domain where linear interpolation between
    distant clusters is still meaningful, even when no raw points occupy the gap.
    """
    if positions is None:
        return None
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return None

    try:
        triangulation = Delaunay(points)
    except (QhullError, ValueError):
        return None

    grid_x, grid_y = create_cell_center_mesh(grid_res, bbox=bbox)
    query = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    inside = triangulation.find_simplex(query) >= 0
    return inside.reshape(grid_x.shape)


def build_dilated_support_mask(positions, grid_res, radius_cells=None, bbox=None):
    """
    Build occupied, unmasked, and final masked grids from local support plus
    the global interpolation hull.
    """
    radius = getattr(config, "MASK_DILATE_RADIUS_CELLS", 1) if radius_cells is None else radius_cells
    occupied_cells = build_occupied_cell_mask(positions, grid_res, bbox=bbox)
    unmasked_cells = dilate_cell_mask(occupied_cells, radius)
    if bool(getattr(config, "MASK_INCLUDE_INTERPOLATION_HULL", True)):
        hull_cells = build_interpolation_hull_mask(positions, grid_res, bbox=bbox)
        if hull_cells is not None:
            unmasked_cells |= hull_cells
    final_mask = np.logical_not(unmasked_cells)
    return occupied_cells, unmasked_cells, final_mask


def interpolate_feature_onto_grid(positions, vectors, grid_x, grid_y, interpolation_method="linear-nearest"):
    """
    Interpolate a single feature's vectors onto the grid.

    Supported methods:
        - linear-nearest: linear interpolation with nearest fallback outside the hull
        - linear: pure linear interpolation (zeros outside the hull after NaN cleanup)
        - nearest: pure nearest-neighbor interpolation
        - cubic-nearest: cubic interpolation with nearest fallback outside the hull
    
    Args:
        positions (np.ndarray): Data point positions, shape (N, 2)
        vectors (np.ndarray): Feature vectors at data points, shape (N, 2)
        grid_x, grid_y (np.ndarray): Grid coordinate meshes
        interpolation_method (str): Point-to-grid interpolation method
        
    Returns:
        tuple: (grid_u, grid_v) - interpolated velocity components
    """
    interpolation_method = normalize_interpolation_method(interpolation_method)

    if interpolation_method == "nearest":
        grid_u = griddata(positions, vectors[:, 0], (grid_x, grid_y), method="nearest", fill_value=0.0)
        grid_v = griddata(positions, vectors[:, 1], (grid_x, grid_y), method="nearest", fill_value=0.0)
    else:
        primary_method = "cubic" if interpolation_method == "cubic-nearest" else "linear"
        use_nearest_fallback = interpolation_method.endswith("-nearest")
        try:
            grid_u_primary = griddata(positions, vectors[:, 0], (grid_x, grid_y), method=primary_method, fill_value=np.nan)
            grid_v_primary = griddata(positions, vectors[:, 1], (grid_x, grid_y), method=primary_method, fill_value=np.nan)
        except Exception:
            if not use_nearest_fallback:
                raise
            grid_u_primary = None
            grid_v_primary = None

        if grid_u_primary is None or grid_v_primary is None:
            grid_u = griddata(positions, vectors[:, 0], (grid_x, grid_y), method="nearest", fill_value=0.0)
            grid_v = griddata(positions, vectors[:, 1], (grid_x, grid_y), method="nearest", fill_value=0.0)
        else:
            nan_mask = np.isnan(grid_u_primary) | np.isnan(grid_v_primary)
            if np.any(nan_mask) and use_nearest_fallback:
                grid_u_nn = griddata(positions, vectors[:, 0], (grid_x, grid_y), method="nearest", fill_value=0.0)
                grid_v_nn = griddata(positions, vectors[:, 1], (grid_x, grid_y), method="nearest", fill_value=0.0)
                grid_u = np.where(nan_mask, grid_u_nn, grid_u_primary)
                grid_v = np.where(nan_mask, grid_v_nn, grid_v_primary)
            else:
                grid_u = grid_u_primary
                grid_v = grid_v_primary

    # Replace any residual NaNs (extreme edge cases) with zeros
    grid_u = np.nan_to_num(grid_u, nan=0.0)
    grid_v = np.nan_to_num(grid_v, nan=0.0)

    return grid_u, grid_v


## Removed soft dominance computation (unused in visuals)


def build_grids(
    positions,
    grid_res,
    top_k_indices,
    all_grad_vectors,
    col_labels,
    output_dir=".",
    interpolation_method="linear-nearest",
):
    """
    Build velocity grids for top-k features and compute dominance information.
    
    Args:
        positions (np.ndarray): Data point positions, shape (N, 2)
        grid_res (int): Grid resolution
        top_k_indices (np.ndarray): Indices of top features
        all_grad_vectors (np.ndarray): All gradient vectors, shape (N, M, 2)
        col_labels (list): Feature column labels
        output_dir (str): Directory for output files
        interpolation_method (str): Point-to-grid interpolation method
        
    Returns:
        tuple: Grid interpolators and computed grids
    """
    interpolation_method = normalize_interpolation_method(interpolation_method)

    # Setup interpolation grid using cell-center convention
    grid_x, grid_y, cell_centers_x, cell_centers_y = create_grid_coordinates(grid_res)
    
    # Interpolate velocity fields for the top-k features
    grid_u_feats, grid_v_feats = [], []
    for feat_idx in top_k_indices:
        # Extract the vectors for the given feature
        vectors = all_grad_vectors[:, feat_idx, :]  # shape: (#points, 2)
        grid_u, grid_v = interpolate_feature_onto_grid(
            positions,
            vectors,
            grid_x,
            grid_y,
            interpolation_method=interpolation_method,
        )
        grid_u_feats.append(grid_u)
        grid_v_feats.append(grid_v)
    
    grid_u_feats = np.array(grid_u_feats)  # shape: (k, grid_res, grid_res)
    grid_v_feats = np.array(grid_v_feats)  # shape: (k, grid_res, grid_res)
    
    # Create the combined (summed) velocity field for the top-k features
    grid_u_sum = np.sum(grid_u_feats, axis=0)  # shape: (grid_res, grid_res)
    grid_v_sum = np.sum(grid_v_feats, axis=0)  # shape: (grid_res, grid_res)
    
    # Apply occupied-cell dilation masking on the final summed field only.
    # Any cell outside the dilated occupied-cell support is zeroed out before
    # building continuous interpolators.
    final_mask = None
    try:
        _, _, final_mask = build_dilated_support_mask(positions, grid_u_sum.shape[0], bbox=config.bounding_box)
        # Apply mask to the final summed field
        grid_u_sum[final_mask] = 0.0
        grid_v_sum[final_mask] = 0.0
    except Exception as e:
        # If masking fails for any reason, proceed without final masking
        print(f"Final support masking skipped due to error: {e}")
    
    # Determine the dominant feature at each grid cell from ALL features
    # First, compute grids for ALL features to find true dominant feature
    num_features = all_grad_vectors.shape[1]
    grid_u_all_feats, grid_v_all_feats = [], []
    
    for feat_idx in range(num_features):
        # Extract vectors for this feature
        vectors = all_grad_vectors[:, feat_idx, :]  # shape: (#points, 2)
        grid_u, grid_v = interpolate_feature_onto_grid(
            positions,
            vectors,
            grid_x,
            grid_y,
            interpolation_method=interpolation_method,
        )
        grid_u_all_feats.append(grid_u)
        grid_v_all_feats.append(grid_v)
    
    grid_u_all_feats = np.array(grid_u_all_feats)  # shape: (num_features, grid_res, grid_res)
    grid_v_all_feats = np.array(grid_v_all_feats)  # shape: (num_features, grid_res, grid_res)
    
    # Compute magnitudes for ALL features
    grid_mag_all_feats = np.sqrt(grid_u_all_feats**2 + grid_v_all_feats**2)
    
    # Create dominant features for each grid cell using ALL features
    cell_dominant_features = np.zeros((grid_res, grid_res), dtype=int)
    
    for i in range(grid_res):
        for j in range(grid_res):
            # Get magnitudes directly from cell centers (no averaging needed)
            cell_mags = grid_mag_all_feats[:, i, j]
            # Hard argmax (dominant feature index)
            dominant_idx = int(np.argmax(cell_mags))
            cell_dominant_features[i, j] = dominant_idx
    
    
    # Save the dominant feature grids
    np.savetxt(os.path.join(output_dir, "cell_dominant_features.csv"), 
               cell_dominant_features, delimiter=",", fmt="%d")
    
    # Build grid interpolators from the computed fields
    # Use cell-center coordinates for all interpolation (unified convention)
    interp_u_sum = RegularGridInterpolator((cell_centers_y, cell_centers_x),
                                         grid_u_sum, bounds_error=False, fill_value=0.0)
    interp_v_sum = RegularGridInterpolator((cell_centers_y, cell_centers_x),
                                         grid_v_sum, bounds_error=False, fill_value=0.0)
    # Use same cell center coordinates for dominant feature interpolation
    interp_argmax = RegularGridInterpolator((cell_centers_y, cell_centers_x),
                                          cell_dominant_features, method='nearest',
                                          bounds_error=False, fill_value=-1)
    
    return (interp_u_sum, interp_v_sum, interp_argmax, grid_x, grid_y, 
            grid_u_feats, grid_v_feats, cell_dominant_features, 
            grid_u_all_feats, grid_v_all_feats, cell_centers_x, cell_centers_y,
            final_mask)


    # Removed: build_grids_alternative (unused)


    # Removed: create_zero_grids (unused)
