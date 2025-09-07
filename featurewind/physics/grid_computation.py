"""
Grid computation module for FeatureWind.

This module handles grid building, velocity field interpolation, and spatial discretization
of gradient vector fields for visualization and particle simulation.
"""

import os
import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, binary_closing
from .. import config




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


def interpolate_feature_onto_grid(positions, vectors, grid_x, grid_y):
    """
    Interpolate a single feature's vectors onto the grid with robust outside-hull handling.
    
    Strategy:
        1) Linear interpolation inside the Delaunay triangulation (best quality)
        2) Nearest-neighbor fallback outside the hull to avoid zeros
    
    Args:
        positions (np.ndarray): Data point positions, shape (N, 2)
        vectors (np.ndarray): Feature vectors at data points, shape (N, 2)
        grid_x, grid_y (np.ndarray): Grid coordinate meshes
        
    Returns:
        tuple: (grid_u, grid_v) - interpolated velocity components
    """
    # First pass: linear interpolation with NaN outside the hull
    grid_u_lin = griddata(positions, vectors[:, 0], (grid_x, grid_y), method='linear', fill_value=np.nan)
    grid_v_lin = griddata(positions, vectors[:, 1], (grid_x, grid_y), method='linear', fill_value=np.nan)

    # Identify cells outside the triangulation (NaNs from linear interpolation)
    nan_mask = np.isnan(grid_u_lin) | np.isnan(grid_v_lin)

    if np.any(nan_mask):
        # Second pass: nearest-neighbor interpolation everywhere
        grid_u_nn = griddata(positions, vectors[:, 0], (grid_x, grid_y), method='nearest', fill_value=0.0)
        grid_v_nn = griddata(positions, vectors[:, 1], (grid_x, grid_y), method='nearest', fill_value=0.0)
        # Combine: use linear where available, nearest elsewhere
        grid_u = np.where(nan_mask, grid_u_nn, grid_u_lin)
        grid_v = np.where(nan_mask, grid_v_nn, grid_v_lin)
    else:
        grid_u, grid_v = grid_u_lin, grid_v_lin

    # Replace any residual NaNs (extreme edge cases) with zeros
    grid_u = np.nan_to_num(grid_u, nan=0.0)
    grid_v = np.nan_to_num(grid_v, nan=0.0)

    return grid_u, grid_v


## Removed soft dominance computation (unused in visuals)


def build_grids(positions, grid_res, top_k_indices, all_grad_vectors, col_labels, output_dir="."):
    """
    Build velocity grids for top-k features and compute dominance information.
    
    Args:
        positions (np.ndarray): Data point positions, shape (N, 2)
        grid_res (int): Grid resolution
        top_k_indices (np.ndarray): Indices of top features
        all_grad_vectors (np.ndarray): All gradient vectors, shape (N, M, 2)
        col_labels (list): Feature column labels
        output_dir (str): Directory for output files
        
    Returns:
        tuple: Grid interpolators and computed grids
    """
    # Setup interpolation grid using cell-center convention
    grid_x, grid_y, cell_centers_x, cell_centers_y = create_grid_coordinates(grid_res)
    
    # Interpolate velocity fields for the top-k features
    grid_u_feats, grid_v_feats = [], []
    for feat_idx in top_k_indices:
        # Extract the vectors for the given feature
        vectors = all_grad_vectors[:, feat_idx, :]  # shape: (#points, 2)
        grid_u, grid_v = interpolate_feature_onto_grid(positions, vectors, grid_x, grid_y)
        grid_u_feats.append(grid_u)
        grid_v_feats.append(grid_v)
    
    grid_u_feats = np.array(grid_u_feats)  # shape: (k, grid_res, grid_res)
    grid_v_feats = np.array(grid_v_feats)  # shape: (k, grid_res, grid_res)
    
    # Create the combined (summed) velocity field for the top-k features
    grid_u_sum = np.sum(grid_u_feats, axis=0)  # shape: (grid_res, grid_res)
    grid_v_sum = np.sum(grid_v_feats, axis=0)  # shape: (grid_res, grid_res)
    
    # Apply buffer-based masking on the final summed field only.
    # Any cell outside data points and their buffer (defined by MASK_BUFFER_FACTOR)
    # is zeroed out before building continuous interpolators.
    try:
        xmin, xmax, ymin, ymax = config.bounding_box
        grid_res_local = grid_u_sum.shape[0]
        # Cell sizes for cell-center grid
        cell_width = (xmax - xmin) / grid_res_local
        cell_height = (ymax - ymin) / grid_res_local
        
        # Initialize all cells as masked (True); we will unmask buffered regions
        final_mask = np.ones_like(grid_u_sum, dtype=bool)
        
        buffer_x = cell_width * config.MASK_BUFFER_FACTOR
        buffer_y = cell_height * config.MASK_BUFFER_FACTOR
        
        # Unmask cells within buffer of any data point
        for pos in positions:
            px, py = pos[0], pos[1]
            i_start = max(0, int((py - buffer_y - ymin) / cell_height))
            i_end = min(grid_res_local, int((py + buffer_y - ymin) / cell_height) + 1)
            j_start = max(0, int((px - buffer_x - xmin) / cell_width))
            j_end = min(grid_res_local, int((px + buffer_x - xmin) / cell_width) + 1)
            final_mask[i_start:i_end, j_start:j_end] = False
        
        # Ensure the exact cell containing each point is unmasked
        for pos in positions:
            i = int((pos[1] - ymin) / cell_height)
            j = int((pos[0] - xmin) / cell_width)
            i = max(0, min(grid_res_local - 1, i))
            j = max(0, min(grid_res_local - 1, j))
            final_mask[i, j] = False
        
        # Apply mask to the final summed field
        grid_u_sum[final_mask] = 0.0
        grid_v_sum[final_mask] = 0.0
    except Exception as e:
        # If masking fails for any reason, proceed without final masking
        print(f"Final buffer masking skipped due to error: {e}")
    
    # Determine the dominant feature at each grid cell from ALL features
    # First, compute grids for ALL features to find true dominant feature
    num_features = all_grad_vectors.shape[1]
    grid_u_all_feats, grid_v_all_feats = [], []
    
    for feat_idx in range(num_features):
        # Extract vectors for this feature
        vectors = all_grad_vectors[:, feat_idx, :]  # shape: (#points, 2)
        grid_u, grid_v = interpolate_feature_onto_grid(positions, vectors, grid_x, grid_y)
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
            grid_u_all_feats, grid_v_all_feats, cell_centers_x, cell_centers_y)


    # Removed: build_grids_alternative (unused)


    # Removed: create_zero_grids (unused)
