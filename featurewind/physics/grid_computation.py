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
    Interpolate a single feature's vectors onto the grid with grid cell boundary-aware masking.
    
    Args:
        positions (np.ndarray): Data point positions, shape (N, 2)
        vectors (np.ndarray): Feature vectors at data points, shape (N, 2)
        grid_x, grid_y (np.ndarray): Grid coordinate meshes
        
    Returns:
        tuple: (grid_u, grid_v) - interpolated velocity components
    """
    # Interpolate each component onto the grid
    grid_u = griddata(positions, vectors[:, 0], (grid_x, grid_y), method='linear', fill_value=0.0)
    grid_v = griddata(positions, vectors[:, 1], (grid_x, grid_y), method='linear', fill_value=0.0)
    
    # Two-stage masking: exact cell identification + minimal buffer
    if len(positions) >= 2:
        try:
            xmin, xmax, ymin, ymax = config.bounding_box
            grid_res = grid_x.shape[0]
            
            # Calculate cell size
            cell_width = (xmax - xmin) / grid_res
            cell_height = (ymax - ymin) / grid_res
            
            # Stage 1: Find cells containing data points (no buffer) - these are protected
            protected_cells = np.zeros(grid_x.shape, dtype=bool)
            for pos in positions:
                # Find exact cell containing this point
                i = int((pos[1] - ymin) / cell_height)
                j = int((pos[0] - xmin) / cell_width)
                i = max(0, min(grid_res - 1, i))
                j = max(0, min(grid_res - 1, j))
                protected_cells[i, j] = True
            
            # Stage 2: Add minimal buffer for interpolation quality
            mask = np.ones(grid_x.shape, dtype=bool)  # True = mask out, False = keep
            
            for pos in positions:
                px, py = pos[0], pos[1]
                
                # Use configurable buffer factor for tighter control
                buffer_x = cell_width * config.MASK_BUFFER_FACTOR
                buffer_y = cell_height * config.MASK_BUFFER_FACTOR
                
                # Find range of cells to unmask
                i_start = max(0, int((py - buffer_y - ymin) / cell_height))
                i_end = min(grid_res, int((py + buffer_y - ymin) / cell_height) + 1)
                j_start = max(0, int((px - buffer_x - xmin) / cell_width))
                j_end = min(grid_res, int((px + buffer_x - xmin) / cell_width) + 1)
                
                # Unmask buffer region
                mask[i_start:i_end, j_start:j_end] = False
            
            # Guarantee: Protected cells are NEVER masked (even if buffer is 0)
            mask[protected_cells] = False
            
            # Apply mask
            grid_u[mask] = 0.0
            grid_v[mask] = 0.0
            
        except Exception as e:
            print(f"Two-stage masking failed: {e}, skipping masking")
            # If masking fails, don't mask anything
            pass
    
    return grid_u, grid_v


def compute_soft_dominance(cell_mags, temperature=None):
    """
    Compute soft dominance probabilities using temperature-based softmax.
    
    Args:
        cell_mags (np.ndarray): Feature magnitudes at a grid cell
        temperature (float, optional): Softmax temperature parameter
        
    Returns:
        tuple: (softmax_probs, dominant_idx)
    """
    if temperature is None:
        temperature = config.TEMPERATURE_SOFTMAX
    
    # Add small epsilon to avoid division by zero
    cell_mags_safe = cell_mags + 1e-8
    
    # Compute softmax probabilities
    softmax_scores = np.exp(cell_mags_safe / temperature)
    softmax_probs = softmax_scores / np.sum(softmax_scores)
    
    # Get dominant feature index
    dominant_idx = np.argmax(cell_mags)
    
    return softmax_probs, dominant_idx


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
    # Store soft dominance probabilities for better visualization
    cell_soft_dominance = np.zeros((grid_res, grid_res, num_features))
    
    for i in range(grid_res):
        for j in range(grid_res):
            # Get magnitudes directly from cell centers (no averaging needed)
            cell_mags = grid_mag_all_feats[:, i, j]
            
            
            # Compute soft dominance using temperature-based softmax
            softmax_probs, dominant_idx = compute_soft_dominance(cell_mags)
            
            # Store probabilities for this cell
            cell_soft_dominance[i, j, :] = softmax_probs
            
            # Store the dominant feature (still need one for compatibility)
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
            cell_soft_dominance)


def build_grids_alternative(positions, grid_res, all_grad_vectors, k_local, output_dir="."):
    """
    Alternative grid builder using local feature selection.
    
    For each grid cell, interpolates the velocity component of every feature,
    then locally selects the top k_local features (by magnitude) and sums their contributions.
    
    Args:
        positions (np.ndarray): Data point positions
        grid_res (int): Grid resolution  
        all_grad_vectors (np.ndarray): All gradient vectors, shape (N, M, 2)
        k_local (int): Number of local top features to select per cell
        output_dir (str): Output directory
        
    Returns:
        tuple: (interp_u_sum_local, interp_v_sum_local, interp_argmax_local, 
                grid_argmax_local, grid_x, grid_y)
    """
    # Setup interpolation grid using cell-center convention
    grid_x, grid_y, cell_centers_x, cell_centers_y = create_grid_coordinates(grid_res)
    
    num_features = all_grad_vectors.shape[1]

    # For each feature, interpolate the velocity fields onto the grid
    grid_u_all, grid_v_all = [], []
    for m in range(num_features):
        vectors = all_grad_vectors[:, m, :]  # shape: (#points, 2)
        grid_u, grid_v = interpolate_feature_onto_grid(positions, vectors, grid_x, grid_y)
        grid_u_all.append(grid_u)
        grid_v_all.append(grid_v)
    
    grid_u_all = np.array(grid_u_all)  # shape: (M, grid_res, grid_res)
    grid_v_all = np.array(grid_v_all)  # shape: (M, grid_res, grid_res)
    
    # For each grid cell, select the top k_local features based on magnitude and sum their vectors
    grid_u_sum_local = np.zeros((grid_res, grid_res))
    grid_v_sum_local = np.zeros((grid_res, grid_res))
    # Also store the dominant (highest magnitude) feature index per cell for color-coding
    grid_argmax_local = np.zeros((grid_res, grid_res), dtype=int)
    
    for i in range(grid_res):
        for j in range(grid_res):
            # Compute magnitude of each feature at this grid cell
            mags = np.sqrt(grid_u_all[:, i, j]**2 + grid_v_all[:, i, j]**2)
            # Get indices of the top k_local features (highest magnitude first)
            top_indices = np.argsort(-mags)[:k_local]
            # Sum velocity components for these selected features
            grid_u_sum_local[i, j] = np.sum(grid_u_all[top_indices, i, j])
            grid_v_sum_local[i, j] = np.sum(grid_v_all[top_indices, i, j])
            grid_argmax_local[i, j] = top_indices[0]  # the most dominant one
    
    # Build RegularGridInterpolators to be used in the animation
    interp_u_sum_local = RegularGridInterpolator((cell_centers_y, cell_centers_x),
                                               grid_u_sum_local, bounds_error=False, fill_value=0.0)
    interp_v_sum_local = RegularGridInterpolator((cell_centers_y, cell_centers_x),
                                               grid_v_sum_local, bounds_error=False, fill_value=0.0)
    interp_argmax_local = RegularGridInterpolator((cell_centers_y, cell_centers_x),
                                                grid_argmax_local, method='nearest', 
                                                bounds_error=False, fill_value=-1)
    
    # Optionally save local dominant feature grid to CSV
    np.savetxt(os.path.join(output_dir, "grid_argmax_local_alternative.csv"), 
               grid_argmax_local, delimiter=",", fmt="%d")
    
    return (interp_u_sum_local, interp_v_sum_local, interp_argmax_local, 
            grid_argmax_local, grid_x, grid_y)


def create_zero_grids(grid_res):
    """
    Create zero velocity grids and interpolators.
    
    Args:
        grid_res (int): Grid resolution
        
    Returns:
        tuple: Zero interpolators for u and v components
    """
    grid_x, grid_y, cell_centers_x, cell_centers_y = create_grid_coordinates(grid_res)
    
    # Set velocity grids to zero
    grid_u_zero = np.zeros((grid_res, grid_res))
    grid_v_zero = np.zeros((grid_res, grid_res))
    
    # Create zero interpolators for smooth motion
    interp_u_zero = RegularGridInterpolator((cell_centers_y, cell_centers_x),
                                          grid_u_zero, bounds_error=False, fill_value=0.0)
    interp_v_zero = RegularGridInterpolator((cell_centers_y, cell_centers_x),
                                          grid_v_zero, bounds_error=False, fill_value=0.0)
    
    return interp_u_zero, interp_v_zero