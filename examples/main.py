import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.spatial import cKDTree, ConvexHull
from scipy.ndimage import maximum_filter, gaussian_filter

try:
    import colorcet as cc
    COLORCET_AVAILABLE = True
except ImportError:
    COLORCET_AVAILABLE = False
    print("Warning: colorcet not available. Using fallback colors.")

from featurewind.TangentPoint import TangentPoint


def PreProcessing(tangentmaps):
    with open(tangentmaps, "r") as f:
        data_import = json.loads(f.read())

    tmap = data_import['tmap']
    Col_labels = data_import['Col_labels']
    points = []
    for tmap_entry in tmap:
        point = TangentPoint(tmap_entry, 1.0, Col_labels)
        points.append(point)
    valid_points = [p for p in points if p.valid]
    all_positions = np.array([p.position for p in valid_points])  # shape: (#points, 2)
    all_grad_vectors = [p.gradient_vectors for p in valid_points]  # list of (#features, 2)
    all_grad_vectors = np.array(all_grad_vectors)                  # shape = (#points, M, 2)

    return valid_points, all_grad_vectors, all_positions, Col_labels


def pick_top_k_features(all_grad_vectors):
    # Compute average magnitude of each feature across all points
    feature_magnitudes = np.linalg.norm(all_grad_vectors, axis=2)  # shape (#points, M)
    avg_magnitudes = feature_magnitudes.mean(axis=0)               # shape (M,)

    # Sort descending, get top k indices
    top_k_indices = np.argsort(-avg_magnitudes)[:k]
    return top_k_indices, avg_magnitudes

def build_grids(positions, grid_res, top_k_indices, all_grad_vectors, Col_labels, output_dir="."):
    # (1) Setup interpolation grid using cell-center convention
    # grid_res represents number of grid cells, create cell centers
    xmin, xmax, ymin, ymax = bounding_box
    # Create cell center coordinates
    cell_centers_x = np.linspace(xmin + (xmax-xmin)/(2*grid_res), xmax - (xmax-xmin)/(2*grid_res), grid_res)
    cell_centers_y = np.linspace(ymin + (ymax-ymin)/(2*grid_res), ymax - (ymax-ymin)/(2*grid_res), grid_res)
    grid_x, grid_y = np.meshgrid(cell_centers_x, cell_centers_y)
    
    # Build a KD-tree and determine distance to the nearest data point at each grid cell
    kdtree = cKDTree(positions)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    distances, _ = kdtree.query(grid_points, k=1)
    dist_grid = distances.reshape(grid_x.shape)
    
    # Compute adaptive threshold based on local point density
    def compute_adaptive_threshold(positions, kdtree, percentile=75):
        """Compute threshold based on local point density"""
        # Get k-nearest neighbor distances for each point
        k = min(5, len(positions))  # Use 5-NN or fewer if dataset is small
        if k <= 1:
            # Fallback for very small datasets
            return max(abs(xmax - xmin), abs(ymax - ymin)) * 0.1
        
        distances, _ = kdtree.query(positions, k=k+1)  # k+1 because first is self
        local_densities = distances[:, 1:].mean(axis=1)  # Average k-NN distance per point
        
        # Use a percentile of local densities as threshold
        adaptive_threshold = np.percentile(local_densities, percentile)
        return adaptive_threshold
    
    threshold = compute_adaptive_threshold(positions, kdtree, percentile=75)
    print(f"Adaptive distance threshold: {threshold:.4f} (based on local density)")
    print(f"  Data points: {len(positions)}, using {min(5, len(positions))}-NN distances")
    
    # (2) Interpolate velocity fields for the top-k features
    grid_u_feats, grid_v_feats = [], []
    for feat_idx in top_k_indices:
        # Extract the vectors for the given feature.
        vectors = all_grad_vectors[:, feat_idx, :]  # shape: (#points, 2)
        # Interpolate each component onto the grid with smooth transitions
        grid_u = griddata(positions, vectors[:, 0], (grid_x, grid_y), method='linear', fill_value=0.0)
        grid_v = griddata(positions, vectors[:, 1], (grid_x, grid_y), method='linear', fill_value=0.0)
        # Mask out grid cells too far from any data.
        mask = dist_grid > threshold
        grid_u[mask] = 0.0
        grid_v[mask] = 0.0
        grid_u_feats.append(grid_u)
        grid_v_feats.append(grid_v)
    grid_u_feats = np.array(grid_u_feats)  # shape: (k, num_vertices, num_vertices)
    grid_v_feats = np.array(grid_v_feats)  # shape: (k, num_vertices, num_vertices)
    
    # Create the combined (summed) velocity field for the top-k features.
    grid_u_sum = np.sum(grid_u_feats, axis=0)  # shape: (num_vertices, num_vertices)
    grid_v_sum = np.sum(grid_v_feats, axis=0)  # shape: (num_vertices, num_vertices)
    
    # (3) Determine the dominant feature at each grid cell from ALL features
    # First, compute grids for ALL features to find true dominant feature
    num_features = all_grad_vectors.shape[1]
    grid_u_all_feats, grid_v_all_feats = [], []
    
    for feat_idx in range(num_features):
        # Extract vectors for this feature
        vectors = all_grad_vectors[:, feat_idx, :]  # shape: (#points, 2)
        # Interpolate each component onto the grid with smooth transitions
        grid_u = griddata(positions, vectors[:, 0], (grid_x, grid_y), method='linear', fill_value=0.0)
        grid_v = griddata(positions, vectors[:, 1], (grid_x, grid_y), method='linear', fill_value=0.0)
        # Apply same masking as top-k features
        mask = dist_grid > threshold
        grid_u[mask] = 0.0
        grid_v[mask] = 0.0
        grid_u_all_feats.append(grid_u)
        grid_v_all_feats.append(grid_v)
    
    grid_u_all_feats = np.array(grid_u_all_feats)  # shape: (num_features, num_vertices, num_vertices)
    grid_v_all_feats = np.array(grid_v_all_feats)  # shape: (num_features, num_vertices, num_vertices)
    
    # Compute magnitudes for ALL features
    grid_mag_all_feats = np.sqrt(grid_u_all_feats**2 + grid_v_all_feats**2)
    
    # Create dominant features for each grid cell using ALL features
    cell_dominant_features = np.zeros((grid_res, grid_res), dtype=int)
    # Store soft dominance probabilities for better visualization
    cell_soft_dominance = np.zeros((grid_res, grid_res, num_features))
    
    for i in range(grid_res):
        for j in range(grid_res):
            # Get magnitudes directly from cell centers (no averaging needed)
            cell_mags = np.zeros(num_features)
            
            for feat_idx in range(num_features):
                # Get magnitude directly from cell center
                cell_mags[feat_idx] = grid_mag_all_feats[feat_idx, i, j]
            
            # Debug specific problematic cell
            if i == 27 and j == 11:
                print(f"\nDebugging grid cell (27,11):")
                print("Cell magnitudes for all features:")
                sorted_indices = np.argsort(-cell_mags)  # Sort by magnitude descending
                for rank, feat_idx in enumerate(sorted_indices[:10]):  # Show top 10
                    if feat_idx < len(Col_labels):
                        print(f"  Rank {rank+1}: Feature {feat_idx} ({Col_labels[feat_idx]}): {cell_mags[feat_idx]:.6f}")
                    else:
                        print(f"  Rank {rank+1}: Feature {feat_idx} (out of range): {cell_mags[feat_idx]:.6f}")
            
            # Compute soft dominance using temperature-based softmax
            # Add small epsilon to avoid division by zero
            cell_mags_safe = cell_mags + 1e-8
            
            # Temperature parameter: lower = more decisive, higher = more uncertainty
            temperature = 0.5  # Adjust this to control softness (0.1=hard, 1.0=soft, 2.0=very soft)
            
            # Compute softmax probabilities
            softmax_scores = np.exp(cell_mags_safe / temperature)
            softmax_probs = softmax_scores / np.sum(softmax_scores)
            
            # Store probabilities for this cell
            cell_soft_dominance[i, j, :] = softmax_probs
            
            # Store the dominant feature (still need one for compatibility)
            dominant_feat_idx = np.argmax(cell_mags)
            cell_dominant_features[i, j] = dominant_feat_idx
    
    print("Cell dominant features shape:", cell_dominant_features.shape)
    
    # Debug: Print statistics about dominant features
    unique_features, counts = np.unique(cell_dominant_features, return_counts=True)
    print("Dominant features found in grid cells:")
    for feat_idx, count in zip(unique_features, counts):
        if feat_idx < len(Col_labels):
            print(f"  Feature {feat_idx} ({Col_labels[feat_idx]}): {count} cells")
        else:
            print(f"  Feature {feat_idx} (index out of range): {count} cells")
    
    # Cell center coordinates already created above
    print("Cell centers x range:", cell_centers_x[0], "to", cell_centers_x[-1])
    print("Cell centers y range:", cell_centers_y[0], "to", cell_centers_y[-1])
    
    # Save the dominant feature grids.
    np.savetxt(os.path.join(output_dir, "cell_dominant_features.csv"), cell_dominant_features, delimiter=",", fmt="%d")
    
    # (4) Build grid interpolators from the computed fields.
    # Use cell-center coordinates for all interpolation (unified convention)
    interp_u_sum = RegularGridInterpolator((cell_centers_y, cell_centers_x),
                                             grid_u_sum, bounds_error=False, fill_value=0.0)
    interp_v_sum = RegularGridInterpolator((cell_centers_y, cell_centers_x),
                                             grid_v_sum, bounds_error=False, fill_value=0.0)
    # Use same cell center coordinates for dominant feature interpolation
    interp_argmax = RegularGridInterpolator((cell_centers_y, cell_centers_x),
                                             cell_dominant_features, method='nearest',
                                             bounds_error=False, fill_value=-1)
    
    return interp_u_sum, interp_v_sum, interp_argmax, grid_x, grid_y, grid_u_feats, grid_v_feats, cell_dominant_features, grid_u_all_feats, grid_v_all_feats, cell_centers_x, cell_centers_y, cell_soft_dominance

def build_grids_alternative(positions, grid_res, all_grad_vectors, k_local, output_dir="."):
    """
    Alternative grid builder:
      - Divides space with grid_res.
      - For each grid cell, interpolates the velocity component of every feature.
      - Locally selects the top k_local features (by magnitude) and sums their contributions.
    Returns:
      - RegularGridInterpolators for the aggregated u‐ and v‑fields.
      - An integer grid of the locally dominant feature (the one with highest magnitude in that cell).
    """
    # (1) Setup interpolation grid using cell-center convention
    xmin, xmax, ymin, ymax = bounding_box  # assuming bounding_box is defined globally
    cell_centers_x = np.linspace(xmin + (xmax-xmin)/(2*grid_res), xmax - (xmax-xmin)/(2*grid_res), grid_res)
    cell_centers_y = np.linspace(ymin + (ymax-ymin)/(2*grid_res), ymax - (ymax-ymin)/(2*grid_res), grid_res)
    grid_x, grid_y = np.meshgrid(cell_centers_x, cell_centers_y)
    
    num_features = all_grad_vectors.shape[1]

     # Build a KD-tree and determine distance to the nearest data point at each grid cell
    kdtree = cKDTree(positions)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    distances, _ = kdtree.query(grid_points, k=1)
    dist_grid = distances.reshape(grid_x.shape)
    
    # Compute adaptive threshold based on local point density
    def compute_adaptive_threshold_alt(positions, kdtree, percentile=75):
        """Compute threshold based on local point density"""
        k = min(5, len(positions))  # Use 5-NN or fewer if dataset is small
        if k <= 1:
            # Fallback for very small datasets
            return max(abs(xmax - xmin), abs(ymax - ymin)) * 0.1
        
        distances, _ = kdtree.query(positions, k=k+1)  # k+1 because first is self
        local_densities = distances[:, 1:].mean(axis=1)  # Average k-NN distance per point
        
        # Use a percentile of local densities as threshold
        adaptive_threshold = np.percentile(local_densities, percentile)
        return adaptive_threshold
    
    threshold = compute_adaptive_threshold_alt(positions, kdtree, percentile=75)
    print(f"Adaptive distance threshold (alt): {threshold:.4f} (based on local density)")
    
    # (2) For each feature, interpolate the velocity fields onto the grid.
    grid_u_all, grid_v_all = [], []
    for m in range(num_features):
        vectors = all_grad_vectors[:, m, :]  # shape: (#points, 2)
        grid_u = griddata(positions, vectors[:, 0], (grid_x, grid_y), method='linear', fill_value=0.0)
        grid_v = griddata(positions, vectors[:, 1], (grid_x, grid_y), method='linear', fill_value=0.0)
        mask = dist_grid > threshold
        grid_u[mask] = 0.0
        grid_v[mask] = 0.0
        grid_u_all.append(grid_u)
        grid_v_all.append(grid_v)
    grid_u_all = np.array(grid_u_all)  # shape: (M, grid_res, grid_res)
    grid_v_all = np.array(grid_v_all)  # shape: (M, grid_res, grid_res)
    
    # (3) For each grid cell, select the top k_local features based on magnitude and sum their vectors.
    grid_u_sum_local = np.zeros((grid_res, grid_res))
    grid_v_sum_local = np.zeros((grid_res, grid_res))
    # also store the dominant (highest magnitude) feature index per cell for color-coding etc.
    grid_argmax_local = np.zeros((grid_res, grid_res), dtype=int)
    
    for i in range(grid_res):
        for j in range(grid_res):
            # Compute magnitude of each feature at this grid cell.
            mags = np.sqrt(grid_u_all[:, i, j]**2 + grid_v_all[:, i, j]**2)
            # Get indices of the top k_local features (highest magnitude first)
            top_indices = np.argsort(-mags)[:k_local]
            # Sum velocity components for these selected features.
            grid_u_sum_local[i, j] = np.sum(grid_u_all[top_indices, i, j])
            grid_v_sum_local[i, j] = np.sum(grid_v_all[top_indices, i, j])
            grid_argmax_local[i, j] = top_indices[0]  # the most dominant one
    
    # (4) Build RegularGridInterpolators to be used in the animation
    interp_u_sum_local = RegularGridInterpolator((cell_centers_y, cell_centers_x),
                                                 grid_u_sum_local, bounds_error=False, fill_value=0.0)
    interp_v_sum_local = RegularGridInterpolator((cell_centers_y, cell_centers_x),
                                                 grid_v_sum_local, bounds_error=False, fill_value=0.0)
    interp_argmax_local = RegularGridInterpolator((cell_centers_y, cell_centers_x),
                                                  grid_argmax_local, method='nearest', 
                                                  bounds_error=False, fill_value=-1)
    
    # Optionally save local dominant feature grid to CSV.
    np.savetxt(os.path.join(output_dir, "grid_argmax_local_alternative.csv"), grid_argmax_local, delimiter=",", fmt="%d")
    
    return interp_u_sum_local, interp_v_sum_local, interp_argmax_local, grid_argmax_local, grid_x, grid_y

def create_particles(num_particles, cell_dominant_features=None, grid_res=None):
    xmin, xmax, ymin, ymax = bounding_box
    particle_positions = np.column_stack((
        np.random.uniform(xmin, xmax, size=num_particles),
        np.random.uniform(ymin, ymax, size=num_particles)
    ))

    max_lifetime = 400
    tail_gap = 10
    particle_lifetimes = np.zeros(num_particles, dtype=int)
    histories = np.full((num_particles, tail_gap + 1, 2), np.nan)
    histories[:, :] = particle_positions[:, None, :]

    # A single LineCollection for all particles
    lc = LineCollection([], linewidths=1.5, zorder=2)

    # We'll store everything in a dict just for cleanliness
    system = {
        'particle_positions': particle_positions,
        'particle_lifetimes': particle_lifetimes,
        'histories': histories,
        'tail_gap': tail_gap,
        'max_lifetime': max_lifetime,
        'linecoll': lc,
    }

    return system

def prepare_figure(ax, valid_points, Col_labels, k, grad_indices, feature_colors, lc, all_positions=None, all_grad_vectors=None):
    xmin, xmax, ymin, ymax = bounding_box
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # ax.set_title(f"Top {k} Features Combined - Single Particle System")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    feature_idx = 2
    domain_values = np.array([p.domain[feature_idx] for p in valid_points])
    domain_min, domain_max = domain_values.min(), domain_values.max()

    # Plot underlying data points once
    # ax.scatter(all_positions[:,0], all_positions[:,1], c='black', s=5, label='Data Points')
    # Collect all labels from valid_points
    unique_labels = sorted(set(p.tmap_label for p in valid_points))
    print("Unique labels:", unique_labels)

    # Define multiple distinct markers (expand this list if you need more)
    markers = ["o", "s", "^", "D", "v", "<", ">"]

    for i, lab in enumerate(unique_labels):
        indices = [j for j, p in enumerate(valid_points) if p.tmap_label == lab]
        positions_lab = np.array([valid_points[j].position for j in indices])
        # alphas = (np.array([valid_points[j].domain[0] for j in indices]) - domain_min) / (domain_max - domain_min + 1e-9)
        normalized = (np.array([valid_points[j].domain[feature_idx] for j in indices]) - domain_min) / (domain_max - domain_min + 1e-9)
        alphas = 0.2 + normalized * 0.8
        # positions_lab = np.array([p.position for p in valid_points if p.tmap_label == lab])
        marker_style = markers[i % len(markers)]
        # colors_lab = [[78/255, 121/255, 167/255, a] for a in alphas]
        # colors_lab = [[225/255, 87/255, 89/255, a] for a in alphas]
        # colors_lab = [[242/255, 142/255, 43/255, a] for a in alphas]
        # "#4E79A7", "#F28E2B", "#E15759"

        ax.scatter(
            positions_lab[:, 0],
            positions_lab[:, 1],
            marker=marker_style,
            color="gray",
            # color = colors_lab,
            # color="white",
            s=10,
            label=f"Label {lab}",
            zorder=4
        )
        

    # Add single line collection
    ax.add_collection(lc)

    # Note: Legend moved to separate location - see main() function
    ax.grid(False)

    # Compute the aggregated arrow for each data point by summing its top-k feature vectors.
    # Note: all_grad_vectors has shape (#points, M, 2) and top_k_indices holds the selected feature indices.
    aggregated_vectors = np.sum(all_grad_vectors[:, grad_indices, :], axis=1)  # shape: (#points, 2)

    # for i, feat_idx in enumerate(grad_indices):
    #     # Extract the color for this feature
    #     color_i = feature_colors[i]
    #     # Draw the individual arrows using quiver.
    #     ax.quiver(
    #         all_positions[:, 0], all_positions[:, 1],
    #         all_grad_vectors[:, feat_idx, 0], all_grad_vectors[:, feat_idx, 1],
    #         color=color_i,  # choose a color that stands out
    #         angles='xy', scale_units='xy', scale=3, width=0.002, headwidth=2, headlength=3, alpha=1,
    #     )

    # Draw the aggregated arrows using quiver.
    # You can adjust the scale parameter to get the desired arrow length.
    # ax.quiver(
    #     all_positions[:, 0], all_positions[:, 1],
    #     aggregated_vectors[:, 0], aggregated_vectors[:, 1],
    #     color='black',  # choose a color that stands out
    #     angles='xy', scale_units='xy', scale=2, width=0.003, headwidth=2, headlength=3, alpha=1.0,
    # )

    return 0

def update(frame, system, interp_u_sum, interp_v_sum, interp_argmax, k, velocity_scale=0.1, grid_u_sum=None, grid_v_sum=None, grid_res=None):
    xmin, xmax, ymin, ymax = bounding_box
    pp = system['particle_positions']
    lt = system['particle_lifetimes']
    his = system['histories']
    lc_ = system['linecoll']
    max_lifetime = system['max_lifetime']

    # Increase lifetime
    lt += 1

    # Helper function to get velocity at any position
    def get_velocity(positions):
        # Always prefer smooth bilinear interpolation for consistent motion
        if 'interp_u_sum' in system and 'interp_v_sum' in system:
            # Direction-conditioned mode: use updated interpolators from system
            current_interp_u = system['interp_u_sum']
            current_interp_v = system['interp_v_sum']
            U = current_interp_u(positions)
            V = current_interp_v(positions)
        elif interp_u_sum is not None and interp_v_sum is not None:
            # Top-K mode: use original interpolators
            U = interp_u_sum(positions)
            V = interp_v_sum(positions)
        else:
            # Fallback: direct grid indexing (should rarely be used)
            if 'grid_u_sum' in system and 'grid_v_sum' in system:
                current_grid_u = system['grid_u_sum']
                current_grid_v = system['grid_v_sum']
                # Convert positions to grid cell indices
                cell_i_indices = np.clip(((positions[:, 1] - ymin) / (ymax - ymin) * grid_res).astype(int), 0, grid_res - 1)
                cell_j_indices = np.clip(((positions[:, 0] - xmin) / (xmax - xmin) * grid_res).astype(int), 0, grid_res - 1)
                # Sample velocity directly from grid cells
                U = current_grid_u[cell_i_indices, cell_j_indices]
                V = current_grid_v[cell_i_indices, cell_j_indices]
            else:
                # No velocity field available
                U = np.zeros(len(positions))
                V = np.zeros(len(positions))
        
        return np.column_stack((U, V)) * velocity_scale

    # Adaptive time stepping with CFL-like condition
    def adaptive_rk4_step(pos, target_dt, get_vel_func, max_error=1e-3):
        """
        Adaptive RK4 with embedded Heun method for error estimation.
        Returns: (new_position, actual_dt_used, error_estimate)
        """
        # Calculate velocity at current position
        vel = get_vel_func(pos)
        speed = np.linalg.norm(vel, axis=1)
        
        # CFL-like condition: dt should be proportional to cell_size / |v|
        if grid_res is not None and len(bounding_box) >= 4:
            cell_size = min(
                (bounding_box[1] - bounding_box[0]) / grid_res,  # dx
                (bounding_box[3] - bounding_box[2]) / grid_res   # dy
            )
            # CFL number around 0.5 for stability
            cfl_number = 0.5
            max_speed = np.maximum(speed, 1e-6)  # Avoid division by zero
            cfl_dt = cfl_number * cell_size / max_speed
            
            # Use minimum of target dt and CFL-limited dt for each particle
            dt_per_particle = np.minimum(target_dt, cfl_dt)
            # Use the most restrictive dt for this step
            dt = np.min(dt_per_particle)
        else:
            dt = target_dt
        
        # Ensure minimum time step
        dt = max(dt, 1e-4)
        
        # RK4 step
        k1 = get_vel_func(pos)
        k2 = get_vel_func(pos + 0.5 * dt * k1)
        k3 = get_vel_func(pos + 0.5 * dt * k2)
        k4 = get_vel_func(pos + dt * k3)
        
        rk4_result = pos + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        # Embedded Heun method for error estimation (simpler RK2)
        heun_k1 = k1
        heun_k2 = get_vel_func(pos + dt * heun_k1)
        heun_result = pos + dt * (heun_k1 + heun_k2) / 2.0
        
        # Estimate local truncation error
        error = np.linalg.norm(rk4_result - heun_result, axis=1)
        max_error_this_step = np.max(error)
        
        return rk4_result, dt, max_error_this_step
    
    # Get initial velocity for speed calculation (used for particle coloring)
    velocity = get_velocity(pp)
    
    # Adaptive integration with error control
    total_time = 0.0
    target_total_time = 1.0
    current_pos = pp.copy()
    
    max_steps = 10  # Prevent infinite loops
    step_count = 0
    
    while total_time < target_total_time and step_count < max_steps:
        remaining_time = target_total_time - total_time
        target_dt = min(0.25, remaining_time)  # Start with quarter steps
        
        new_pos, dt_used, error_est = adaptive_rk4_step(current_pos, target_dt, get_velocity)
        
        # Simple error control: accept step if error is reasonable
        if error_est < 0.01 or dt_used < 1e-3:
            # Accept the step
            current_pos = new_pos
            total_time += dt_used
            step_count += 1
        else:
            # Reduce time step and try again
            target_dt *= 0.5
            if target_dt < 1e-4:
                # Force acceptance with very small step
                current_pos = new_pos
                total_time += dt_used
                step_count += 1
    
    pp[:] = current_pos

    # Shift history
    his[:, :-1, :] = his[:, 1:, :]
    his[:, -1, :] = pp

    # Reinitialize out-of-bounds or over-age particles
    for i in range(len(pp)):
        x, y = pp[i]
        if (x < xmin or x > xmax or y < ymin or y > ymax
            or lt[i] > max_lifetime):
            pp[i] = [
                np.random.uniform(xmin, xmax),
                np.random.uniform(ymin, ymax)
            ]
            his[i] = pp[i]
            lt[i] = 0

    # Randomly reinitialize some fraction
    num_to_reinit = int(0.05 * len(pp))
    if num_to_reinit > 0:
        idxs = np.random.choice(len(pp), num_to_reinit, replace=False)
        for idx in idxs:
            pp[idx] = [
                np.random.uniform(xmin, xmax),
                np.random.uniform(ymin, ymax)
            ]
            his[idx] = pp[idx]
            lt[idx] = 0

    # Build line segments
    n_active = len(pp)
    tail_gap = system['tail_gap']
    segments = np.zeros((n_active * tail_gap, 2, 2))
    colors_rgba = np.zeros((n_active * tail_gap, 4))

    # Compute speeds for alpha fade
    speeds = np.linalg.norm(velocity, axis=1)
    max_speed = speeds.max() + 1e-9  # avoid division by zero

    # Use black color for all particles
    for i in range(n_active):
        # All particles are black regardless of grid cell or feature
        r, g, b = 0, 0, 0  # Black color
        alpha_part = speeds[i] / max_speed

        for t in range(tail_gap):
            seg_idx = i * tail_gap + t
            segments[seg_idx, 0, :] = his[i, t, :]
            segments[seg_idx, 1, :] = his[i, t + 1, :]

            # Combine alpha with any additional fade for the tail
            # age_factor = 1.0 - (t / tail_gap)
            age_factor = (t+1) / tail_gap
            # age_factor = 1.0

            # Multiply them
            alpha_min = 0.15
            alpha_final = max(alpha_min, alpha_part * age_factor)

            # Assign the final RGBA
            colors_rgba[seg_idx] = [r, g, b, alpha_final]

    lc_.set_segments(segments)
    lc_.set_colors(colors_rgba)

    return (lc_,)


def main():
    # Setup paths relative to repository root
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    tangent_map_path = os.path.join(repo_root, 'tangentmaps', 'breast_cancer.tmap')
    output_dir = os.path.join(repo_root, 'output')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the tangent map data
    valid_points, all_grad_vectors, all_positions, Col_labels = PreProcessing(tangent_map_path)

    # Set the number of top features to visualize
    global k 
    k = len(Col_labels)
    # k = 5

    # Compute the bounding box and make it square
    global bounding_box
    xmin, xmax = all_positions[:,0].min(), all_positions[:,0].max()
    ymin, ymax = all_positions[:,1].min(), all_positions[:,1].max()
    
    # Add some padding
    x_padding = (xmax - xmin) * 0.05
    y_padding = (ymax - ymin) * 0.05
    xmin -= x_padding
    xmax += x_padding
    ymin -= y_padding
    ymax += y_padding
    
    # Make the bounding box square by expanding the smaller dimension
    x_range = xmax - xmin
    y_range = ymax - ymin
    
    if x_range > y_range:
        # Expand y range to match x range
        y_center = (ymin + ymax) / 2
        ymin = y_center - x_range / 2
        ymax = y_center + x_range / 2
    else:
        # Expand x range to match y range
        x_center = (xmin + xmax) / 2
        xmin = x_center - y_range / 2
        xmax = x_center + y_range / 2
    
    bounding_box = [xmin, xmax, ymin, ymax]

    # Set the velocity scale
    global velocity_scale
    velocity_scale = 0.04

    # Set the grid resolution scale
    grid_res_scale = 0.15

    # feature colors
    global real_feature_rgba

    # Pick top k features based on average magnitude
    top_k_indices, avg_magnitudes = pick_top_k_features(all_grad_vectors)
    print("Top k feature indices:", top_k_indices)
    print("Their average magnitudes:", avg_magnitudes[top_k_indices])
    print("Their labels:", [Col_labels[i] for i in top_k_indices])
    
    # Debug: Find "mean symmetry" feature index
    mean_symmetry_idx = None
    for i, label in enumerate(Col_labels):
        if "mean symmetry" in label.lower():
            mean_symmetry_idx = i
            print(f"Found 'mean symmetry' at feature index {i}: {label}")
            break
    if mean_symmetry_idx is None:
        print("Warning: 'mean symmetry' feature not found in labels")

    # grad_indices = [2]
    grad_indices = top_k_indices

    # Create a grid for interpolation
    grid_res = (min(abs(xmax - xmin), abs(ymax - ymin)) * grid_res_scale).astype(int)
    grid_res = 40
    print("Grid resolution:", grid_res)

    # KD-tree scale no longer needed - using adaptive thresholding

    # Declare variables that will be used in nested functions
    grid_u_feats = None
    grid_v_feats = None 
    cell_dominant_features = None
    
    interp_u_sum, interp_v_sum, interp_argmax, grid_x, grid_y, grid_u_feats, grid_v_feats, cell_dominant_features, grid_u_all_feats, grid_v_all_feats, cell_centers_x, cell_centers_y, cell_soft_dominance = build_grids(
        all_positions, grid_res, grad_indices, all_grad_vectors, Col_labels, output_dir=output_dir
    )

    # interp_u_sum, interp_v_sum, interp_argmax, grid_argmax_local, grid_x, grid_y = build_grids_alternative(
    #     all_positions, grid_res, all_grad_vectors, k_local=len(grad_indices), kdtree_scale=kdtree_scale
    # )
    # grad_indices = np.unique(grid_argmax_local)

    # Create the combined (summed) velocity field for the top-k features.
    grid_u_sum = np.sum(grid_u_feats, axis=0)  # shape: (grid_res, grid_res)
    grid_v_sum = np.sum(grid_v_feats, axis=0)  # shape: (grid_res, grid_res)

    # Generate distinct colors using ColorCET Glasbey or fallback to Tableau
    def generate_distinct_colors(n_colors):
        if COLORCET_AVAILABLE and hasattr(cc, 'glasbey'):
            # Use ColorCET Glasbey for maximum distinctness
            if n_colors <= len(cc.glasbey):
                return [cc.glasbey[i] for i in range(n_colors)]
            else:
                # If we need more colors than available, cycle through
                return [cc.glasbey[i % len(cc.glasbey)] for i in range(n_colors)]
        else:
            # Fallback to expanded Tableau-like palette
            base_colors = [
                "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", 
                "#EDC949", "#AF7AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
            ]
            return [base_colors[i % len(base_colors)] for i in range(n_colors)]
    
    # Generate colors only for top 6 features
    top_6_indices = top_k_indices[:6]  # Only first 6 features get colors
    top_6_colors = generate_distinct_colors(6)
    
    # Debug: Print color assignments
    print("\nColor assignments for top 6 features:")
    for i, feat_idx in enumerate(top_6_indices):
        if feat_idx < len(Col_labels):
            print(f"  Feature {feat_idx} ({Col_labels[feat_idx]}): {top_6_colors[i]}")
        else:
            print(f"  Feature {feat_idx} (index out of range): {top_6_colors[i]}")
    
    # Extract colors for the selected top features (only if they're in top 6)
    feature_colors = []
    for feat_idx in grad_indices:
        if feat_idx in top_6_indices:
            color_idx = list(top_6_indices).index(feat_idx)
            feature_colors.append(top_6_colors[color_idx])
        else:
            feature_colors.append('#808080')  # Gray for non-top-6 features

    # Build a mapping only for top 6 features to their colors
    all_feature_rgba = {}
    for i, feat_idx in enumerate(top_6_indices):
        all_feature_rgba[feat_idx] = to_rgba(top_6_colors[i])
    
    # Build initial mapping for selected features to RGBA for particle coloring
    # Only top 6 features get colors, others are excluded from coloring
    real_feature_rgba = {}
    for feat_idx in grad_indices:
        if feat_idx in all_feature_rgba:
            real_feature_rgba[feat_idx] = all_feature_rgba[feat_idx]

    # Create the particle system
    num_particles = 2500
    system = create_particles(num_particles, cell_dominant_features, grid_res)
    system['cell_dominant_features'] = cell_dominant_features  # Store for reinitialization
    system['grid_u_all_feats'] = grid_u_all_feats  # Store for wind vane
    system['grid_v_all_feats'] = grid_v_all_feats  # Store for wind vane
    system['grid_u_sum'] = grid_u_sum  # Store velocity grids for particle animation
    system['grid_v_sum'] = grid_v_sum
    # Store initial interpolators for smooth motion
    system['interp_u_sum'] = interp_u_sum
    system['interp_v_sum'] = interp_v_sum
    lc = system['linecoll']

    # prepare the figure with 2 subplots and space for controls
    fig = plt.figure(figsize=(12, 8))
    
    # Create main subplots
    ax1 = plt.subplot2grid((20, 2), (0, 0), rowspan=18)
    ax2 = plt.subplot2grid((20, 2), (0, 1), rowspan=18)
    # --- UI Mode Controls ---
    # Mode selection radio buttons
    ax_mode = fig.add_axes([0.05, 0.02, 0.25, 0.06])
    mode_radio = RadioButtons(ax_mode, ('Top-K Mode', 'Direction-Conditioned Mode'))
    mode_radio.set_active(0)  # Start with Top-K mode
    
    # Current mode state
    current_mode = {'mode': 'top_k'}  # 'top_k' or 'direction_conditioned'
    
    # --- Top-K Mode Controls ---
    # Slider for selecting k in Top k mode
    ax_k = fig.add_axes([0.35, 0.02, 0.30, 0.03])
    k_slider = Slider(ax_k, 'Top k Features', 1, len(Col_labels), valinit=len(grad_indices), 
                      valfmt='%d', facecolor='lightgreen', alpha=0.7)
    
    # --- Direction-Conditioned Mode Controls ---
    # Angle slider for direction specification
    ax_angle = fig.add_axes([0.70, 0.06, 0.25, 0.03])
    angle_slider = Slider(ax_angle, 'Direction (°)', 0, 360, valinit=0, 
                         valfmt='%.0f°', facecolor='lightblue', alpha=0.7)
    
    # Magnitude slider for desired flow strength
    ax_magnitude = fig.add_axes([0.70, 0.02, 0.25, 0.03])
    magnitude_slider = Slider(ax_magnitude, 'Magnitude', 0.1, 2.0, valinit=1.0, 
                             valfmt='%.1f', facecolor='lightcoral', alpha=0.7)
    
    # Status text for direction-conditioned mode
    ax_status = fig.add_axes([0.35, 0.06, 0.30, 0.03])
    ax_status.text(0.5, 0.5, 'Select grid cells and specify direction', 
                   ha='center', va='center', fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    ax_status.set_xlim(0, 1)
    ax_status.set_ylim(0, 1)
    ax_status.axis('off')
    
    # Initially hide direction-conditioned controls
    ax_angle.set_visible(False)
    ax_magnitude.set_visible(False)
    ax_status.set_visible(False)
    
    # Data structures for direction-conditioned mode
    selected_cells = set()  # Set of (i, j) grid cell indices
    user_constraints = {}   # {(i, j): {"direction": (dx, dy), "weight": float}}
    constraint_arrows = []  # Visual arrows for constraints
    cell_highlight_patches = []  # Visual highlighting for selected cells
    
    # Mode switching callback
    def switch_mode(label):
        nonlocal current_mode
        if label == 'Top-K Mode':
            current_mode['mode'] = 'top_k'
            # Show top-k controls, hide direction controls
            ax_k.set_visible(True)
            ax_angle.set_visible(False)
            ax_magnitude.set_visible(False)
            ax_status.set_visible(False)
            # Clear any selected cells and constraints
            selected_cells.clear()
            user_constraints.clear()
            clear_constraint_visuals()
        else:  # Direction-Conditioned Mode
            current_mode['mode'] = 'direction_conditioned'
            # Hide top-k controls, show direction controls
            ax_k.set_visible(False)
            ax_angle.set_visible(True)
            ax_magnitude.set_visible(True)
            ax_status.set_visible(True)
            # Reset to no features selected (no wind)
            reset_to_no_features()
            update_status_text("Select grid cells and specify direction")
        
        # Redraw the interface
        fig.canvas.draw_idle()
    
    mode_radio.on_clicked(switch_mode)
    
    # Helper functions for mode switching
    def clear_constraint_visuals():
        """Remove all constraint arrows and highlights from the visualization"""
        for arrow in constraint_arrows:
            try:
                arrow.remove()
            except (ValueError, AttributeError):
                pass
        constraint_arrows.clear()
        
        # Clear cell highlighting patches
        for patch in cell_highlight_patches:
            try:
                patch.remove()
            except (ValueError, AttributeError):
                pass
        cell_highlight_patches.clear()
    
    def update_cell_highlighting():
        """Update visual highlighting for selected grid cells"""
        # Clear existing highlights
        for patch in cell_highlight_patches:
            try:
                patch.remove()
            except (ValueError, AttributeError):
                pass
        cell_highlight_patches.clear()
        
        # Add highlights for all selected cells
        for cell_i, cell_j in selected_cells:
            # Calculate cell boundaries
            cell_xmin = xmin + cell_j * (xmax - xmin) / grid_res
            cell_xmax = xmin + (cell_j + 1) * (xmax - xmin) / grid_res
            cell_ymin = ymin + cell_i * (ymax - ymin) / grid_res
            cell_ymax = ymin + (cell_i + 1) * (ymax - ymin) / grid_res
            
            # Create highlighting rectangle
            from matplotlib.patches import Rectangle
            highlight = Rectangle((cell_xmin, cell_ymin), 
                                cell_xmax - cell_xmin, cell_ymax - cell_ymin,
                                facecolor='yellow', alpha=0.3, 
                                edgecolor='orange', linewidth=2, zorder=5)
            ax1.add_patch(highlight)
            cell_highlight_patches.append(highlight)
    
    def reset_to_no_features():
        """Reset system to have no features selected (no particle flow)"""
        nonlocal grad_indices, grid_u_sum, grid_v_sum
        grad_indices = []  # No features selected
        # Set velocity grids to zero
        grid_u_sum = np.zeros_like(grid_u_sum)
        grid_v_sum = np.zeros_like(grid_v_sum)
        # Update system grids
        system['grid_u_sum'] = grid_u_sum
        system['grid_v_sum'] = grid_v_sum
        # Create zero interpolators for smooth motion
        system['interp_u_sum'] = RegularGridInterpolator((cell_centers_y, cell_centers_x),
                                                         grid_u_sum, bounds_error=False, fill_value=0.0)
        system['interp_v_sum'] = RegularGridInterpolator((cell_centers_y, cell_centers_x),
                                                         grid_v_sum, bounds_error=False, fill_value=0.0)
    
    def update_status_text(message):
        """Update the status text in direction-conditioned mode"""
        ax_status.clear()
        ax_status.text(0.5, 0.5, message, ha='center', va='center', fontsize=9,
                      bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        ax_status.set_xlim(0, 1)
        ax_status.set_ylim(0, 1)
        ax_status.axis('off')
    
    # Direction specification callbacks
    def update_direction_constraints(val=None):
        """Update direction constraints for selected cells based on slider values"""
        if current_mode['mode'] != 'direction_conditioned' or not selected_cells:
            return
            
        # Get current angle and magnitude from sliders
        angle_deg = angle_slider.val
        magnitude = magnitude_slider.val
        
        # Convert angle to direction vector
        angle_rad = np.radians(angle_deg)
        direction_x = np.cos(angle_rad) * magnitude
        direction_y = np.sin(angle_rad) * magnitude
        
        # Update constraints for all selected cells
        for cell in selected_cells:
            user_constraints[cell] = {
                "direction": (direction_x, direction_y),
                "weight": 1.0,
                "enabled": True
            }
        
        # Update constraint visualization
        update_constraint_arrows()
        
        # Trigger feature optimization
        optimize_features_for_constraints()
        
        # Update status text with current constraint info
        num_cells = len(selected_cells)
        update_status_text(f"{num_cells} cell(s) constrained: {angle_deg:.0f}°, mag={magnitude:.1f}")
        
        print(f"Updated constraints: angle={angle_deg:.0f}°, magnitude={magnitude:.1f}")
        fig.canvas.draw_idle()
    
    # Connect direction sliders to callback
    angle_slider.on_changed(update_direction_constraints)
    magnitude_slider.on_changed(update_direction_constraints)
    
    def update_constraint_arrows():
        """Update visual arrows showing user constraints on selected cells"""
        # Clear existing constraint arrows
        for arrow in constraint_arrows:
            try:
                arrow.remove()
            except (ValueError, AttributeError):
                pass
        constraint_arrows.clear()
        
        # Draw arrows for each constraint
        for (cell_i, cell_j), constraint in user_constraints.items():
            if not constraint.get("enabled", True):
                continue
                
            # Calculate cell center
            cell_center_x = xmin + (cell_j + 0.5) * (xmax - xmin) / grid_res
            cell_center_y = ymin + (cell_i + 0.5) * (ymax - ymin) / grid_res
            
            # Get direction vector
            dx, dy = constraint["direction"]
            
            # Scale arrow for visibility
            arrow_scale = 0.8 * min((xmax - xmin) / grid_res, (ymax - ymin) / grid_res)
            arrow_dx = dx * arrow_scale
            arrow_dy = dy * arrow_scale
            
            # Create arrow
            arrow = ax1.arrow(cell_center_x, cell_center_y, arrow_dx, arrow_dy,
                            head_width=arrow_scale*0.2, head_length=arrow_scale*0.2,
                            fc='red', ec='red', linewidth=2, alpha=0.8, zorder=6)
            constraint_arrows.append(arrow)
    
    def optimize_features_for_constraints():
        """Find the best combination of features that match user constraints"""
        if not user_constraints:
            # No constraints, reset to no features
            reset_to_no_features()
            return
        
        print(f"Optimizing features for {len(user_constraints)} constraints...")
        
        # Show constraints being optimized for
        for (i, j), constraint in user_constraints.items():
            dx, dy = constraint["direction"]
            angle = np.degrees(np.arctan2(dy, dx))
            mag = np.sqrt(dx**2 + dy**2)
            print(f"  Cell ({i},{j}): angle={angle:.0f}°, magnitude={mag:.2f}")
        
        # Parameters for optimization
        max_features = min(10, len(Col_labels))  # Limit search space
        best_score = -float('inf')
        best_features = []
        
        # Try different feature combinations using greedy forward selection
        current_features = []
        remaining_features = list(range(len(Col_labels)))
        
        for step in range(max_features):
            best_candidate = None
            best_candidate_score = -float('inf')
            
            # Try adding each remaining feature
            for feat_idx in remaining_features:
                candidate_features = current_features + [feat_idx]
                score = evaluate_feature_combination(candidate_features)
                
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = feat_idx
            
            # If adding this feature improves the score, add it
            if best_candidate is not None and best_candidate_score > best_score:
                current_features.append(best_candidate)
                remaining_features.remove(best_candidate)
                best_score = best_candidate_score
                best_features = current_features.copy()
                print(f"Added feature {best_candidate} ({Col_labels[best_candidate]}), score: {best_score:.3f}")
            else:
                # No improvement, stop search
                break
        
        # Apply the best feature combination
        apply_feature_combination(best_features)
        
        # Update status with optimization results
        if best_features:
            feature_names = [Col_labels[i] for i in best_features[:3]]  # Show first 3 feature names
            if len(best_features) > 3:
                status_msg = f"Using {len(best_features)} features: {', '.join(feature_names)}..."
            else:
                status_msg = f"Using {len(best_features)} features: {', '.join(feature_names)}"
            update_status_text(status_msg)
        else:
            update_status_text("No suitable features found")
        
        print(f"Final optimization: {len(best_features)} features, score: {best_score:.3f}")
    
    def evaluate_feature_combination(feature_indices):
        """Evaluate how well a feature combination matches user constraints"""
        if not feature_indices or not user_constraints:
            return -1.0
        
        # Create combined velocity field for these features
        # Use the same shape as the all-features grids
        if len(grid_u_all_feats) > 0:
            combined_u = np.zeros_like(grid_u_all_feats[0])
            combined_v = np.zeros_like(grid_v_all_feats[0])
        else:
            combined_u = np.zeros((grid_res, grid_res))
            combined_v = np.zeros((grid_res, grid_res))
        
        for feat_idx in feature_indices:
            if feat_idx < len(grid_u_all_feats):
                combined_u += grid_u_all_feats[feat_idx]
                combined_v += grid_v_all_feats[feat_idx]
        
        total_score = 0.0
        total_weight = 0.0
        
        # Evaluate alignment at each constrained cell
        for (cell_i, cell_j), constraint in user_constraints.items():
            if not constraint.get("enabled", True):
                continue
            
            # Get desired direction
            desired_dx, desired_dy = constraint["direction"]
            desired_magnitude = np.sqrt(desired_dx**2 + desired_dy**2)
            
            if desired_magnitude < 1e-6:
                continue  # Skip zero constraints
            
            # Get actual velocity directly from cell center (no averaging needed with cell-centered grids)
            actual_u = combined_u[cell_i, cell_j]
            actual_v = combined_v[cell_i, cell_j]
            
            actual_magnitude = np.sqrt(actual_u**2 + actual_v**2)
            
            if actual_magnitude < 1e-6:
                # No flow at this location
                alignment = 0.0
            else:
                # Compute cosine similarity (dot product of normalized vectors)
                alignment = (actual_u * desired_dx + actual_v * desired_dy) / (actual_magnitude * desired_magnitude)
                # Bonus for magnitude matching
                magnitude_ratio = min(actual_magnitude / desired_magnitude, desired_magnitude / actual_magnitude)
                alignment = alignment * (0.7 + 0.3 * magnitude_ratio)  # Weight direction more than magnitude
            
            weight = constraint.get("weight", 1.0)
            total_score += alignment * weight
            total_weight += weight
        
        return total_score / max(total_weight, 1e-6)
    
    def apply_feature_combination(feature_indices):
        """Apply the optimized feature combination to the system"""
        nonlocal grad_indices, grid_u_sum, grid_v_sum
        
        grad_indices = feature_indices
        
        if not feature_indices:
            # No features selected
            grid_u_sum = np.zeros_like(grid_u_sum)
            grid_v_sum = np.zeros_like(grid_v_sum)
        else:
            # Sum the selected features - use consistent shape
            if len(grid_u_all_feats) > 0:
                grid_u_sum = np.zeros_like(grid_u_all_feats[0])
                grid_v_sum = np.zeros_like(grid_v_all_feats[0])
            else:
                grid_u_sum = np.zeros((grid_res, grid_res))
                grid_v_sum = np.zeros((grid_res, grid_res))
            
            for feat_idx in feature_indices:
                if feat_idx < len(grid_u_all_feats):
                    grid_u_sum += grid_u_all_feats[feat_idx]
                    grid_v_sum += grid_v_all_feats[feat_idx]
        
        # Update system grids for particle animation
        system['grid_u_sum'] = grid_u_sum
        system['grid_v_sum'] = grid_v_sum
        
        # Create RegularGridInterpolators for smooth bilinear interpolation
        # Use cell-center coordinates for consistent indexing
        system['interp_u_sum'] = RegularGridInterpolator((cell_centers_y, cell_centers_x),
                                                         grid_u_sum, bounds_error=False, fill_value=0.0)
        system['interp_v_sum'] = RegularGridInterpolator((cell_centers_y, cell_centers_x),
                                                         grid_v_sum, bounds_error=False, fill_value=0.0)
    
    # callback to update selected features based on Top k only
    def update_features(val):
        nonlocal grad_indices, interp_u_sum, interp_v_sum, interp_argmax, grid_u_sum, grid_v_sum
        k_val = int(val)
        grad_indices = list(np.argsort(-avg_magnitudes)[:k_val])
        # Rebuild grids for new top-k selection
        interp_u_sum, interp_v_sum, interp_argmax, _, _, grid_u_feats, grid_v_feats, cell_dominant_features, grid_u_all_feats_new, grid_v_all_feats_new, _, _, cell_soft_dominance_new = build_grids(
            all_positions, grid_res, grad_indices, all_grad_vectors, Col_labels, output_dir=output_dir)
        
        # Update the combined velocity fields used by particle animation
        grid_u_sum = np.sum(grid_u_feats, axis=0)  # shape: (grid_res, grid_res)
        grid_v_sum = np.sum(grid_v_feats, axis=0)  # shape: (grid_res, grid_res)
        
        # Update system with new grids that animation will use
        system['grid_u_sum'] = grid_u_sum
        system['grid_v_sum'] = grid_v_sum
        system['grid_u_all_feats'] = grid_u_all_feats_new
        system['grid_v_all_feats'] = grid_v_all_feats_new
        system['cell_dominant_features'] = cell_dominant_features  # Update dominant features too
        
        # Update interpolators for consistent smooth motion
        system['interp_u_sum'] = interp_u_sum
        system['interp_v_sum'] = interp_v_sum
        # Update color mappings for particles using top-6 logic
        new_feature_colors = []
        for feat_idx in grad_indices:
            if feat_idx in top_6_indices:
                color_idx = list(top_6_indices).index(feat_idx)
                new_feature_colors.append(top_6_colors[color_idx])
            else:
                new_feature_colors.append('#808080')  # Gray for non-top-6 features
        
        # Mutate the real_feature_rgba mapping in place for the particle update
        real_feature_rgba.clear()
        for feat_idx in grad_indices:
            if feat_idx in all_feature_rgba:
                real_feature_rgba[feat_idx] = all_feature_rgba[feat_idx]
        # Redraw main feature wind map
        ax1.clear()
        prepare_figure(ax1, valid_points, Col_labels, k, grad_indices,
                       new_feature_colors, lc, all_positions, all_grad_vectors)
        # Redraw grid visualization with updated colors
        draw_grid_visualization()
        # Refresh grid cell visualization
        update_grid_cell_visualization()
        fig.canvas.draw_idle()
    
    k_slider.on_changed(update_features)
    
    # Make both subplots square
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    
    # Prepare the first subplot with the main visualization
    prepare_figure(ax1, valid_points, Col_labels, k, grad_indices, feature_colors, lc, all_positions, all_grad_vectors)
    ax1.set_title('Feature Wind Map', fontsize=12, pad=10)
    
    # Prepare the second subplot (additional canvas)
    xmin, xmax, ymin, ymax = bounding_box
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    # Prepare the second subplot (Wind Vane) - no background
    ax2.set_title('Wind Vane', fontsize=12, pad=10)
    
    # Mouse position tracking for grid cell analysis
    current_mouse_pos = [(xmin + xmax) / 2, (ymin + ymax) / 2]  # Initialize at center
    
    # Initialize aggregate point visualization elements
    aggregate_point_marker = None
    aggregate_arrows = []
    
    # Mouse interaction variables for grid cell tracking
    mouse_data = {
        'position': current_mouse_pos,
        'grid_cell': [grid_res//2, grid_res//2]  # Initialize at center grid cell
    }
    
    # Remove brush size update function - no longer needed
    
    def update_grid_cell_visualization():
        nonlocal aggregate_point_marker, aggregate_arrows
        
        # Clear previous aggregate visualization
        if aggregate_point_marker:
            try:
                aggregate_point_marker.remove()
            except (ValueError, AttributeError):
                pass  # Already removed or doesn't exist
            aggregate_point_marker = None
        
        for arrow in aggregate_arrows:
            try:
                arrow.remove()
            except (ValueError, AttributeError):
                pass  # Already removed or doesn't exist
        aggregate_arrows.clear()
        
        current_cell = mouse_data['grid_cell']
        cell_i, cell_j = current_cell
        
        # Validate grid cell indices
        if cell_i < 0 or cell_i >= grid_res or cell_j < 0 or cell_j >= grid_res:
            fig.canvas.draw_idle()
            return
        
        # Get the averaged vectors for the current grid cell from the 4 corner vertices
        # The grid cell (cell_i, cell_j) has corner vertices at:
        # (cell_i, cell_j), (cell_i+1, cell_j), (cell_i, cell_j+1), (cell_i+1, cell_j+1)
        
        cell_vectors = np.zeros((len(Col_labels), 2))  # Initialize for all features
        
        # Get vectors for ALL features from the all-features grid
        for feat_idx in range(len(Col_labels)):
            if 'grid_u_all_feats' in system and 'grid_v_all_feats' in system:
                # Use the all-features grid data
                corner_u = (system['grid_u_all_feats'][feat_idx, cell_i, cell_j] + 
                           system['grid_u_all_feats'][feat_idx, cell_i+1, cell_j] +
                           system['grid_u_all_feats'][feat_idx, cell_i, cell_j+1] + 
                           system['grid_u_all_feats'][feat_idx, cell_i+1, cell_j+1]) / 4.0
                
                corner_v = (system['grid_v_all_feats'][feat_idx, cell_i, cell_j] + 
                           system['grid_v_all_feats'][feat_idx, cell_i+1, cell_j] +
                           system['grid_v_all_feats'][feat_idx, cell_i, cell_j+1] + 
                           system['grid_v_all_feats'][feat_idx, cell_i+1, cell_j+1]) / 4.0
                
                cell_vectors[feat_idx] = [corner_u, corner_v]
                
                # Debug: Print the corrected magnitude for problematic cell
                if cell_i == 27 and cell_j == 11 and feat_idx in [21, 8]:  # Check our two key features
                    corrected_mag = np.linalg.norm([corner_u, corner_v])
                    feat_name = Col_labels[feat_idx] if feat_idx < len(Col_labels) else f"Feature {feat_idx}"
                    print(f"  CORRECTED Feature {feat_idx} ({feat_name}): {corrected_mag:.6f}")
                    
                    # Debug: Compare with the grid calculation method
                    grid_mag_from_build = np.sqrt(system['grid_u_all_feats'][feat_idx]**2 + system['grid_v_all_feats'][feat_idx]**2)
                    grid_corner_mag = (grid_mag_from_build[cell_i, cell_j] + 
                                      grid_mag_from_build[cell_i+1, cell_j] +
                                      grid_mag_from_build[cell_i, cell_j+1] + 
                                      grid_mag_from_build[cell_i+1, cell_j+1]) / 4.0
                    print(f"    Grid build method magnitude: {grid_corner_mag:.6f}")
                    print(f"    Difference: {abs(corrected_mag - grid_corner_mag):.6f}")
            else:
                # Fallback: only calculate for selected features using grid_u_feats
                if feat_idx in grad_indices:
                    k_idx = list(grad_indices).index(feat_idx)
                    corner_u = (grid_u_feats[k_idx, cell_i, cell_j] + 
                               grid_u_feats[k_idx, cell_i+1, cell_j] +
                               grid_u_feats[k_idx, cell_i, cell_j+1] + 
                               grid_u_feats[k_idx, cell_i+1, cell_j+1]) / 4.0
                    
                    corner_v = (grid_v_feats[k_idx, cell_i, cell_j] + 
                               grid_v_feats[k_idx, cell_i+1, cell_j] +
                               grid_v_feats[k_idx, cell_i, cell_j+1] + 
                               grid_v_feats[k_idx, cell_i+1, cell_j+1]) / 4.0
                    
                    cell_vectors[feat_idx] = [corner_u, corner_v]
        
        # Get the dominant feature for this cell
        dominant_feature = cell_dominant_features[cell_i, cell_j]
        
        # Debug: Print wind vane cell information
        if cell_i < len(Col_labels):
            dominant_feature_name = Col_labels[dominant_feature] if dominant_feature < len(Col_labels) else f"Feature {dominant_feature}"
            print(f"Wind Vane showing grid cell ({cell_i},{cell_j}), dominant feature: {dominant_feature} ({dominant_feature_name})")
        
        # Place grid cell point at center of Wind Vane
        center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
        
        # Draw grid cell marker in Wind Vane (always at center) 
        # Use color of dominant feature
        dominant_color = all_feature_rgba.get(dominant_feature, (0.5, 0.5, 0.5, 1.0))
        aggregate_point_marker = ax2.scatter(center_x, center_y, 
                                           s=120, c=[dominant_color], marker='s', 
                                           zorder=10, label=f'Grid Cell ({cell_i},{cell_j})')
        
        # Draw gradient vector arrows for each feature in the grid cell
        # Calculate dynamic scaling to use 90% of canvas
        non_zero_vectors = []
        vector_magnitudes = []
        
        # Debug: Show which features are being considered in wind vane
        if cell_i == 27 and cell_j == 11:
            print(f"Wind vane considering selected features: {grad_indices}")
            print("Vector magnitudes in wind vane:")
            
        # Only include selected top-k features in wind vane
        for feat_idx in grad_indices:
            vec = cell_vectors[feat_idx]
            
            # Use the same magnitude calculation method as grid cell dominant feature
            if 'grid_u_all_feats' in system and 'grid_v_all_feats' in system:
                grid_mag_from_build = np.sqrt(system['grid_u_all_feats'][feat_idx]**2 + system['grid_v_all_feats'][feat_idx]**2)
                mag = (grid_mag_from_build[cell_i, cell_j] + 
                      grid_mag_from_build[cell_i+1, cell_j] +
                      grid_mag_from_build[cell_i, cell_j+1] + 
                      grid_mag_from_build[cell_i+1, cell_j+1]) / 4.0
            else:
                # Fallback to the old method
                mag = np.linalg.norm(vec)
            
            # Debug output for problematic cell
            if cell_i == 27 and cell_j == 11:
                feat_name = Col_labels[feat_idx] if feat_idx < len(Col_labels) else f"Feature {feat_idx}"
                print(f"  Feature {feat_idx} ({feat_name}): {mag:.6f} (CONSISTENT)")
            
            if mag > 0:
                non_zero_vectors.append((feat_idx, vec))
                vector_magnitudes.append(mag)
        
        if vector_magnitudes:
            # Find the longest vector
            max_magnitude = max(vector_magnitudes)
            # Scale so longest vector takes up 45% of canvas radius (90% diameter coverage)
            canvas_size = min(xmax - xmin, ymax - ymin)
            target_length = canvas_size * 0.45
            dynamic_scale = max_magnitude / target_length
            
            # Calculate all endpoint positions for convex hull
            endpoints = []
            vector_info = []  # Store (feat_idx, gradient_vector, pos_endpoint, neg_endpoint)
            
            for i, (feat_idx, gradient_vector) in enumerate(non_zero_vectors):
                # Scale the gradient vector to match the consistent magnitude
                consistent_magnitude = vector_magnitudes[i]
                original_magnitude = np.linalg.norm(gradient_vector)
                if original_magnitude > 0:
                    scale_factor = consistent_magnitude / original_magnitude
                    scaled_gradient_vector = gradient_vector * scale_factor
                else:
                    scaled_gradient_vector = gradient_vector
                
                pos_endpoint = np.array([center_x + scaled_gradient_vector[0] / dynamic_scale,
                                       center_y + scaled_gradient_vector[1] / dynamic_scale])
                neg_endpoint = np.array([center_x - scaled_gradient_vector[0] / dynamic_scale,
                                       center_y - scaled_gradient_vector[1] / dynamic_scale])
                
                endpoints.append(pos_endpoint)
                endpoints.append(neg_endpoint)
                vector_info.append((feat_idx, scaled_gradient_vector, pos_endpoint, neg_endpoint))
            
            # Calculate convex hull
            endpoints_array = np.array(endpoints)
            hull = ConvexHull(endpoints_array)
            hull_points = endpoints_array[hull.vertices]
            
            # Draw convex hull
            from matplotlib.patches import Polygon
            hull_polygon = Polygon(hull_points, fill=False, edgecolor='black',
                               linewidth=1, alpha=0.7, zorder=7, linestyle='--')
            ax2.add_patch(hull_polygon)
            aggregate_arrows.append(hull_polygon)
            
            # Calculate and draw covariance ellipse
            # Collect all vector endpoints (scaled the same way as vectors)
            all_vector_endpoints = []
            for i, (feat_idx, gradient_vector) in enumerate(non_zero_vectors):
                # Scale the gradient vector to match the consistent magnitude
                consistent_magnitude = vector_magnitudes[i]
                original_magnitude = np.linalg.norm(gradient_vector)
                if original_magnitude > 0:
                    scale_factor = consistent_magnitude / original_magnitude
                    scaled_gradient_vector = gradient_vector * scale_factor
                else:
                    scaled_gradient_vector = gradient_vector
                
                # Positive endpoint
                pos_endpoint = np.array([scaled_gradient_vector[0] / dynamic_scale,
                                       scaled_gradient_vector[1] / dynamic_scale])
                # Negative endpoint  
                neg_endpoint = np.array([-scaled_gradient_vector[0] / dynamic_scale,
                                       -scaled_gradient_vector[1] / dynamic_scale])
                all_vector_endpoints.append(pos_endpoint)
                all_vector_endpoints.append(neg_endpoint)
            
            if len(all_vector_endpoints) >= 2:
                endpoints_matrix = np.array(all_vector_endpoints)  # shape: (n_endpoints, 2)
                
                # Calculate covariance matrix
                cov_matrix = np.cov(endpoints_matrix.T)  # 2x2 covariance matrix
                
                # Eigendecomposition for ellipse parameters
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                
                # Sort eigenvalues and eigenvectors in descending order
                # (eigh returns them in ascending order)
                order = eigenvals.argsort()[::-1]
                eigenvals = eigenvals[order]
                eigenvecs = eigenvecs[:, order]
                
                # Calculate ellipse parameters (scaled smaller to fit inside convex hull)
                confidence_scale = 1.5  # Reduced from 2.0 to make ellipse smaller
                width = 2 * confidence_scale * np.sqrt(eigenvals[0])   # Major axis (largest eigenvalue)
                height = 2 * confidence_scale * np.sqrt(eigenvals[1])  # Minor axis (smallest eigenvalue)
                
                # Calculate angle of rotation (in degrees) using the major axis eigenvector
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                
                # Create ellipse centered at Wind Vane center
                cov_ellipse = Ellipse((center_x, center_y), width, height,
                                    angle=angle, fill=False,
                                    edgecolor='black', linewidth=2,
                                    alpha=0.6, zorder=11, linestyle='-')
                ax2.add_patch(cov_ellipse)
                aggregate_arrows.append(cov_ellipse)
            
            # Determine which endpoints are on the convex hull boundary
            hull_vertices_set = set(hull.vertices)
            
            for i, (feat_idx, gradient_vector, pos_endpoint, neg_endpoint) in enumerate(vector_info):
                # Check if either endpoint is on the convex hull
                pos_on_hull = (2 * i) in hull_vertices_set if hull_vertices_set else False
                neg_on_hull = (2 * i + 1) in hull_vertices_set if hull_vertices_set else False
                
                # Get the consistent magnitude for this feature
                consistent_magnitude = vector_magnitudes[i]  # This now uses the grid build method
                
                # Scale the gradient vector to match the consistent magnitude
                original_magnitude = np.linalg.norm(gradient_vector)
                if original_magnitude > 0:
                    # Scale the vector to have the consistent magnitude
                    scale_factor = consistent_magnitude / original_magnitude
                    scaled_gradient_vector = gradient_vector * scale_factor
                else:
                    scaled_gradient_vector = gradient_vector
                
                # Use feature colors with intensity based on vector magnitude and dominance
                if feat_idx in all_feature_rgba:
                    base_color = all_feature_rgba[feat_idx]
                    base_r, base_g, base_b = base_color[:3]
                    
                    # Calculate intensity based on consistent vector magnitude
                    vector_magnitude = consistent_magnitude
                    max_mag = max(vector_magnitudes) if vector_magnitudes else 1.0
                    mag_intensity = 0.3 + 0.7 * (vector_magnitude / max_mag)
                    
                    # Use soft dominance instead of hard dominance boost
                    # Get the probability for this feature from soft dominance
                    if 'cell_soft_dominance' in globals() and cell_soft_dominance is not None:
                        # cell_i and cell_j are already available from mouse_data['grid_cell'] above
                        # Get soft dominance probability for this feature
                        if feat_idx < cell_soft_dominance.shape[2]:
                            soft_prob = cell_soft_dominance[cell_i, cell_j, feat_idx]
                            dominance_boost = 0.8 + 0.8 * soft_prob  # Scale from 0.8 to 1.6 based on probability
                        else:
                            dominance_boost = 1.0
                    else:
                        # Fallback to hard dominance if soft dominance not available
                        dominance_boost = 1.3 if feat_idx == dominant_feature else 1.0
                    
                    intensity = min(1.0, mag_intensity * dominance_boost)
                    
                    # Add uncertainty visualization through desaturation
                    uncertainty_factor = 1.0  # Default: no desaturation
                    if 'cell_soft_dominance' in globals() and cell_soft_dominance is not None:
                        # Calculate uncertainty as entropy of the probability distribution
                        if feat_idx < cell_soft_dominance.shape[2]:
                            probs = cell_soft_dominance[cell_i, cell_j, :]
                            # Compute entropy: higher entropy = more uncertainty
                            entropy = -np.sum(probs * np.log(probs + 1e-8))
                            max_entropy = np.log(len(probs))  # Maximum possible entropy
                            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                            
                            # Convert entropy to desaturation factor: high uncertainty = more desaturation
                            uncertainty_factor = 0.5 + 0.5 * (1.0 - normalized_entropy)  # Range [0.5, 1.0]
                    
                    # Apply desaturation by blending with gray
                    gray_value = 0.5  # Mid-gray
                    desaturated_r = base_r * uncertainty_factor + gray_value * (1 - uncertainty_factor)
                    desaturated_g = base_g * uncertainty_factor + gray_value * (1 - uncertainty_factor)  
                    desaturated_b = base_b * uncertainty_factor + gray_value * (1 - uncertainty_factor)
                    
                    # Adjust alpha based on hull membership
                    pos_alpha = intensity * (0.9 if pos_on_hull else 0.6)
                    neg_alpha = intensity * (0.7 if neg_on_hull else 0.4)
                    
                    pos_color = (desaturated_r, desaturated_g, desaturated_b, pos_alpha)
                    neg_color = (desaturated_r, desaturated_g, desaturated_b, neg_alpha)
                else:
                    # Fallback to black/gray scheme
                    pos_alpha = 0.8 if pos_on_hull else 0.4
                    neg_alpha = 0.6 if neg_on_hull else 0.3
                    
                    pos_color = (0.0, 0.0, 0.0, pos_alpha) if pos_on_hull else (0.5, 0.5, 0.5, pos_alpha)
                    neg_color = (0.0, 0.0, 0.0, neg_alpha) if neg_on_hull else (0.5, 0.5, 0.5, neg_alpha)
                
                # Draw positive direction arrow (solid) using scaled vector
                arrow_pos = ax2.arrow(center_x, center_y,
                                    scaled_gradient_vector[0] / dynamic_scale,
                                    scaled_gradient_vector[1] / dynamic_scale,
                                    head_width=canvas_size * 0.02,
                                    head_length=canvas_size * 0.03,
                                    fc=pos_color, ec=pos_color,
                                    length_includes_head=True, zorder=9)
                aggregate_arrows.append(arrow_pos)
                
                # Draw negative direction arrow (solid arrow + dashed overlay) using scaled vector
                arrow_neg = ax2.arrow(center_x, center_y,
                                    -scaled_gradient_vector[0] / dynamic_scale,
                                    -scaled_gradient_vector[1] / dynamic_scale,
                                    head_width=canvas_size * 0.02,
                                    head_length=canvas_size * 0.03,
                                    fc=neg_color, ec=neg_color,
                                    length_includes_head=True, zorder=8)
                aggregate_arrows.append(arrow_neg)
                
                # Overlay dashed line to create dashed effect using scaled vector
                neg_end_x = center_x - scaled_gradient_vector[0] / dynamic_scale
                neg_end_y = center_y - scaled_gradient_vector[1] / dynamic_scale
                
                # Draw white dashed line over the arrow to create gaps
                dash_line = ax2.plot([center_x, neg_end_x], [center_y, neg_end_y], 
                                   color='white', linestyle='-', linewidth=3,
                                   alpha=0.8, zorder=8.5,
                                   dashes=[5, 5])[0]  # 5 points on, 5 points off
                aggregate_arrows.append(dash_line)
                
                # Add feature name labels for boundary vectors
                if pos_on_hull or neg_on_hull:
                    feature_name = Col_labels[feat_idx]
                    
                    # Add label for positive vector if it's on hull
                    if pos_on_hull:
                        pos_end_x = center_x + scaled_gradient_vector[0] / dynamic_scale
                        pos_end_y = center_y + scaled_gradient_vector[1] / dynamic_scale
                        
                        # Calculate label offset (slightly beyond arrow tip)
                        offset_factor = 1.2
                        label_x = center_x + (scaled_gradient_vector[0] / dynamic_scale) * offset_factor
                        label_y = center_y + (scaled_gradient_vector[1] / dynamic_scale) * offset_factor
                        
                        pos_label = ax2.text(label_x, label_y, feature_name,
                                           fontsize=8, ha='center', va='center',
                                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                                           zorder=10)
                        aggregate_arrows.append(pos_label)
                    
                    # Add label for negative vector if it's on hull
                    if neg_on_hull:
                        # Calculate label offset (slightly beyond arrow tip)
                        offset_factor = 1.2
                        label_x = center_x - (scaled_gradient_vector[0] / dynamic_scale) * offset_factor
                        label_y = center_y - (scaled_gradient_vector[1] / dynamic_scale) * offset_factor
                        
                        neg_label = ax2.text(label_x, label_y, f"-{feature_name}",
                                           fontsize=8, ha='center', va='center',
                                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8),
                                           zorder=10)
                        aggregate_arrows.append(neg_label)
        
        fig.canvas.draw_idle()
    
    def update_mouse_grid_cell(mouse_x, mouse_y):
        """Convert mouse position to grid cell indices and update visualization"""
        if mouse_x is None or mouse_y is None:
            return
            
        # Convert mouse position to grid cell indices
        # Grid cells range from 0 to grid_res-1
        cell_i = int((mouse_y - ymin) / (ymax - ymin) * grid_res)
        cell_j = int((mouse_x - xmin) / (xmax - xmin) * grid_res)
        
        # Clamp to valid range
        cell_i = max(0, min(grid_res - 1, cell_i))
        cell_j = max(0, min(grid_res - 1, cell_j))
        
        # Update mouse data if cell changed
        if mouse_data['grid_cell'] != [cell_i, cell_j]:
            mouse_data['position'] = [mouse_x, mouse_y]
            mouse_data['grid_cell'] = [cell_i, cell_j]
            print(f"Grid cell: ({cell_i}, {cell_j})")
            update_grid_cell_visualization()
    
    def on_motion(event):
        """Handle mouse motion over the feature wind map"""
        if event.inaxes == ax1 and event.xdata is not None and event.ydata is not None:
            # Only update wind vane in top-k mode, reduce debug printing
            if current_mode['mode'] == 'top_k':
                update_mouse_grid_cell(event.xdata, event.ydata)
                fig.canvas.draw_idle()
    
    def on_click(event):
        """Handle mouse clicks for grid cell selection in direction-conditioned mode"""
        if (event.inaxes == ax1 and event.xdata is not None and event.ydata is not None 
            and current_mode['mode'] == 'direction_conditioned'):
            
            # Convert click position to grid cell indices
            cell_i = int((event.ydata - ymin) / (ymax - ymin) * grid_res)
            cell_j = int((event.xdata - xmin) / (xmax - xmin) * grid_res)
            
            # Clamp to valid range
            cell_i = max(0, min(grid_res - 1, cell_i))
            cell_j = max(0, min(grid_res - 1, cell_j))
            
            # Toggle cell selection (Ctrl+click for multi-select, regular click for single select)
            if event.key == 'ctrl':
                # Multi-select mode: toggle this cell
                if (cell_i, cell_j) in selected_cells:
                    selected_cells.remove((cell_i, cell_j))
                    if (cell_i, cell_j) in user_constraints:
                        del user_constraints[(cell_i, cell_j)]
                    print(f"Deselected cell ({cell_i}, {cell_j})")
                else:
                    selected_cells.add((cell_i, cell_j))
                    print(f"Selected cell ({cell_i}, {cell_j})")
            else:
                # Single select mode: clear previous and select this cell
                selected_cells.clear()
                user_constraints.clear()
                selected_cells.add((cell_i, cell_j))
                print(f"Selected cell ({cell_i}, {cell_j}) (cleared previous)")
            
            # Update visual highlighting
            update_cell_highlighting()
            
            # Update status text
            if selected_cells:
                update_status_text(f"{len(selected_cells)} cell(s) selected. Adjust angle/magnitude sliders.")
            else:
                update_status_text("Select grid cells and specify direction")
            
            fig.canvas.draw_idle()
    
    # Connect events
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Initial grid cell visualization
    update_grid_cell_visualization()
    
    # Draw grid lines with colors based on dominant features
    def draw_grid_visualization():
        # Draw grid lines from the grid arrays
        n_rows, n_cols = grid_x.shape
        print("Grid shape:", grid_x.shape)
        
        # Draw vertical grid lines
        for col in range(n_cols):
            ax1.plot(grid_x[:, col], grid_y[:, col], color='gray', linestyle='--', linewidth=0.3, alpha=0.5, zorder=1)
        
        # Draw horizontal grid lines  
        for row in range(n_rows):
            ax1.plot(grid_x[row, :], grid_y[row, :], color='gray', linestyle='--', linewidth=0.3, alpha=0.5, zorder=1)
        
        # Color-code grid cells based on dominant features
        for i in range(grid_res):
            for j in range(grid_res):
                # Get cell boundaries
                cell_xmin = xmin + j * (xmax - xmin) / grid_res
                cell_xmax = xmin + (j + 1) * (xmax - xmin) / grid_res
                cell_ymin = ymin + i * (ymax - ymin) / grid_res
                cell_ymax = ymin + (i + 1) * (ymax - ymin) / grid_res
                
                # No grid cell coloring - keep cells transparent
                # (Grid coloring removed for monochromatic visualization)
    
    # Draw the grid visualization
    draw_grid_visualization()
    
    # Feature colors legend removed per user request
    
    # # Draw grid lines from the grid arrays
    # n_rows, n_cols = grid_x.shape
    # print("Grid shape:", grid_x.shape)
    # for col in range(n_cols):
    #     ax.plot(grid_x[:, col], grid_y[:, col], color='gray', linestyle='--', linewidth=0.5)
    # for row in range(n_rows):
    #     ax.plot(grid_x[row, :], grid_y[row, :], color='gray', linestyle='--', linewidth=0.5)

    # cell_centers = np.empty((n_rows-1, n_cols-1, 2))
    # for i in range(n_rows - 1):
    #     for j in range(n_cols - 1):
    #         # For each cell, you can choose the center as the mean of the 4 corner coordinates.
    #         cx = (grid_x[i, j] + grid_x[i+1, j+1]) / 2.0
    #         cy = (grid_y[i, j] + grid_y[i+1, j+1]) / 2.0
    #         cell_centers[i, j, :] = [cx, cy]

    # grid_argmax = interp_argmax(cell_centers.reshape(-1, 2)).reshape(n_rows-1, n_cols-1).astype(int)

    # # grid_argmax is of shape (n_rows, n_cols) corresponding to the grid points.
    # # Create a fine mesh grid (increase res_fine for higher resolution Voronoi)
    # res_fine = 100  # for example
    # xx, yy = np.meshgrid(np.linspace(xmin, xmax, res_fine),
    #                     np.linspace(ymin, ymax, res_fine))

    # # Use grid points directly as seeds for the Voronoi diagram.
    # # points_seeds has shape (n_rows*n_cols, 2) from grid_x and grid_y.
    # points_seeds = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    # # Here, interp_argmax returns the dominant feature for any queried coordinate.
    # # Use it on the grid points (the seeds) to get a seed_feature per grid point.
    # seed_features = interp_argmax(points_seeds).astype(int)  # shape: (n_rows*n_cols,)
    # print("Seed features shape:", seed_features.shape)

    # # Create a fine mesh grid (increase res_fine for higher resolution Voronoi)
    # xx, yy = np.meshgrid(np.linspace(xmin, xmax, res_fine),
    #                     np.linspace(ymin, ymax, res_fine))

    # # Build a KD-tree from the grid points and query for the nearest seed for every point in the fine mesh.
    # tree = cKDTree(points_seeds)
    # _, seed_idx = tree.query(np.c_[xx.ravel(), yy.ravel()])
    # seed_idx = seed_idx.reshape(xx.shape)
    # print("Seed indices shape:", seed_idx.shape)

    # # Construct the Voronoi color image: for each fine-grid pixel assign the color of the nearest seed.
    # voronoi_color = np.zeros((res_fine, res_fine, 4))
    # for i in range(res_fine):
    #     for j in range(res_fine):
    #         feat = seed_features[seed_idx[i, j]]
    #         # Lookup the color for this feature from the real_feature_rgba mapping.
    #         voronoi_color[i, j, :] = real_feature_rgba.get(feat, (0, 0, 0, 1))

    # # Overlay the Voronoi background.
    # ax.imshow(voronoi_color, extent=(xmin, xmax, ymin, ymax), origin='lower',
    #         interpolation='none', alpha=0.5)
    
    # # Overlay grid vectors on the figure.
    # for i in grad_indices:
    #     # Extract the color for this feature
    #     color_i = feature_colors[i]
    #     # Draw the individual arrows using quiver.
    #     ax.quiver(
    #         grid_x, grid_y,
    #         grid_u_feats[i], grid_v_feats[i],
    #         color=color_i,  # choose a color that stands out
    #         angles='xy', scale_units='xy', scale=1, width=0.003, headwidth=2, headlength=3, alpha=0.8,
    #     )
    # ax.quiver(
    #     grid_x, grid_y,
    #     grid_u_sum, grid_v_sum,
    #     color="black", angles='xy', scale_units='xy', scale=0.5, width=0.003, headwidth=2, headlength=3, alpha=0.8
    # )

    for spine in ax1.spines.values():
        spine.set_visible(False)

    for frame in range(5):
        # Update the system state for this frame.
        update(frame, system, interp_u_sum, interp_v_sum, interp_argmax, k, velocity_scale, None, None, grid_res)
        # Save the current state of the figure.
        fig.savefig(os.path.join(output_dir, f"frame_{frame}.png"), dpi=300)

    # Create the animation
    anim = FuncAnimation(fig, 
                         lambda frame: update(frame, system, interp_u_sum, interp_v_sum, interp_argmax, k, velocity_scale, None, None, grid_res), 
                         frames=1000, interval=30, blit=False)

    # Save the figure as a PNG file with 300 dpi.
    fig.savefig(os.path.join(output_dir, "featurewind_figure.png"), dpi=300)
    plt.show()


if  __name__ == "__main__":
    main()