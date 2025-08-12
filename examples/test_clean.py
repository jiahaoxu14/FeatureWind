"""
FeatureWind Visualization - Clean Implementation

This script creates animated particle flow visualizations from high-dimensional data,
showing how features flow across a 2D projection space using grid-based vector fields.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.widgets import Slider, CheckButtons
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.spatial import cKDTree, ConvexHull
from matplotlib.patches import Polygon

# Add source path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import colorcet for better colors
try:
    import colorcet as cc
    COLORCET_AVAILABLE = True
except ImportError:
    COLORCET_AVAILABLE = False
    print("Warning: colorcet not available. Using fallback colors.")

from featurewind.TangentPoint import TangentPoint

# Global variables
bounding_box = None
k = None
real_feature_rgba = None


def load_data(tangent_map_path):
    """Load and preprocess tangent map data."""
    with open(tangent_map_path, "r") as f:
        data_import = json.loads(f.read())

    tmap = data_import['tmap']
    col_labels = data_import['Col_labels']
    
    # Create TangentPoint objects
    points = [TangentPoint(entry, 1.0, col_labels) for entry in tmap]
    valid_points = [p for p in points if p.valid]
    
    # Extract positions and gradient vectors
    all_positions = np.array([p.position for p in valid_points])
    all_grad_vectors = np.array([p.gradient_vectors for p in valid_points])
    
    return valid_points, all_grad_vectors, all_positions, col_labels


def select_top_features(all_grad_vectors, k):
    """Select top k features based on average gradient magnitude."""
    feature_magnitudes = np.linalg.norm(all_grad_vectors, axis=2)
    avg_magnitudes = feature_magnitudes.mean(axis=0)
    top_k_indices = np.argsort(-avg_magnitudes)[:k]
    return top_k_indices, avg_magnitudes


def create_grids(positions, all_grad_vectors, top_k_indices, grid_res, kdtree_scale):
    """Create velocity grids and determine dominant features for each grid cell."""
    global bounding_box
    xmin, xmax, ymin, ymax = bounding_box
    num_vertices = grid_res + 1
    
    # Create grid
    grid_x, grid_y = np.mgrid[xmin:xmax:complex(num_vertices), 
                              ymin:ymax:complex(num_vertices)]
    
    # Build distance mask
    kdtree = cKDTree(positions)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    distances, _ = kdtree.query(grid_points, k=1)
    dist_grid = distances.reshape(grid_x.shape)
    threshold = max(abs(xmax - xmin), abs(ymax - ymin)) * kdtree_scale
    
    # Interpolate velocity fields for top-k features
    grid_u_feats, grid_v_feats = [], []
    for feat_idx in top_k_indices:
        vectors = all_grad_vectors[:, feat_idx, :]
        grid_u = griddata(positions, vectors[:, 0], (grid_x, grid_y), method='nearest')
        grid_v = griddata(positions, vectors[:, 1], (grid_x, grid_y), method='nearest')
        
        # Apply distance mask
        mask = dist_grid > threshold
        grid_u[mask] = 0.0
        grid_v[mask] = 0.0
        
        grid_u_feats.append(grid_u)
        grid_v_feats.append(grid_v)
    
    grid_u_feats = np.array(grid_u_feats)
    grid_v_feats = np.array(grid_v_feats)
    grid_u_sum = np.sum(grid_u_feats, axis=0)
    grid_v_sum = np.sum(grid_v_feats, axis=0)
    
    # Create grids for ALL features to determine true dominance
    num_features = all_grad_vectors.shape[1]
    grid_u_all, grid_v_all = [], []
    
    for feat_idx in range(num_features):
        vectors = all_grad_vectors[:, feat_idx, :]
        grid_u = griddata(positions, vectors[:, 0], (grid_x, grid_y), method='nearest')
        grid_v = griddata(positions, vectors[:, 1], (grid_x, grid_y), method='nearest')
        
        mask = dist_grid > threshold
        grid_u[mask] = 0.0
        grid_v[mask] = 0.0
        
        grid_u_all.append(grid_u)
        grid_v_all.append(grid_v)
    
    grid_u_all = np.array(grid_u_all)
    grid_v_all = np.array(grid_v_all)
    
    # Calculate dominant feature for each grid cell
    grid_mag_all = np.sqrt(grid_u_all**2 + grid_v_all**2)
    cell_dominant_features = np.zeros((grid_res, grid_res), dtype=int)
    
    for i in range(grid_res):
        for j in range(grid_res):
            # Average magnitudes from 4 corner vertices for each feature
            corner_mags = np.zeros(num_features)
            for feat_idx in range(num_features):
                corner_sum = (grid_mag_all[feat_idx, i, j] + 
                             grid_mag_all[feat_idx, i+1, j] +
                             grid_mag_all[feat_idx, i, j+1] + 
                             grid_mag_all[feat_idx, i+1, j+1])
                corner_mags[feat_idx] = corner_sum / 4.0
            
            cell_dominant_features[i, j] = np.argmax(corner_mags)
    
    # Create interpolators
    interp_u_sum = RegularGridInterpolator(
        (grid_x[:, 0], grid_y[0, :]), grid_u_sum, 
        bounds_error=False, fill_value=0.0)
    interp_v_sum = RegularGridInterpolator(
        (grid_x[:, 0], grid_y[0, :]), grid_v_sum, 
        bounds_error=False, fill_value=0.0)
    
    # Cell center interpolator for dominant features
    cell_centers_x = np.linspace(xmin + (xmax-xmin)/(2*grid_res), 
                                xmax - (xmax-xmin)/(2*grid_res), grid_res)
    cell_centers_y = np.linspace(ymin + (ymax-ymin)/(2*grid_res), 
                                ymax - (ymax-ymin)/(2*grid_res), grid_res)
    interp_argmax = RegularGridInterpolator(
        (cell_centers_x, cell_centers_y), cell_dominant_features, 
        method='nearest', bounds_error=False, fill_value=-1)
    
    return (interp_u_sum, interp_v_sum, interp_argmax, grid_x, grid_y, 
            grid_u_feats, grid_v_feats, cell_dominant_features, 
            grid_u_sum, grid_v_sum, grid_u_all, grid_v_all)


def create_particles(num_particles):
    """Initialize particle system."""
    global bounding_box
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
    
    lc = LineCollection([], linewidths=1.5, zorder=2)
    
    return {
        'particle_positions': particle_positions,
        'particle_lifetimes': particle_lifetimes,
        'histories': histories,
        'tail_gap': tail_gap,
        'max_lifetime': max_lifetime,
        'linecoll': lc
    }


def update_particles(frame, system, interp_u_sum, interp_v_sum, interp_argmax, 
                    velocity_scale, grid_u_sum, grid_v_sum, grid_res, 
                    cell_dominant_features):
    """Update particle positions and colors using RK4 integration with sub-stepping."""
    global bounding_box, real_feature_rgba
    xmin, xmax, ymin, ymax = bounding_box
    
    pp = system['particle_positions']
    lt = system['particle_lifetimes']
    his = system['histories']
    lc_ = system['linecoll']
    max_lifetime = system['max_lifetime']
    tail_gap = system['tail_gap']
    
    # Increase lifetime
    lt += 1
    
    # Helper function to get velocity at any position
    def get_velocity(positions):
        if grid_u_sum is not None and grid_v_sum is not None and grid_res is not None:
            cell_i = np.clip(((positions[:, 1] - ymin) / (ymax - ymin) * grid_res).astype(int), 0, grid_res - 1)
            cell_j = np.clip(((positions[:, 0] - xmin) / (xmax - xmin) * grid_res).astype(int), 0, grid_res - 1)
            U = grid_u_sum[cell_i, cell_j]
            V = grid_v_sum[cell_i, cell_j]
        else:
            U = interp_u_sum(positions)
            V = interp_v_sum(positions)
        return np.column_stack((U, V)) * velocity_scale
    
    # RK4 integration with sub-stepping
    n_substeps = 4
    dt_sub = 1.0 / n_substeps
    velocity = get_velocity(pp)  # For speed calculation
    
    for _ in range(n_substeps):
        k1 = get_velocity(pp)
        k2 = get_velocity(pp + 0.5 * dt_sub * k1)
        k3 = get_velocity(pp + 0.5 * dt_sub * k2)
        k4 = get_velocity(pp + dt_sub * k3)
        velocity_sub = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        pp += dt_sub * velocity_sub
    
    # Update history
    his[:, :-1, :] = his[:, 1:, :]
    his[:, -1, :] = pp
    
    # Reinitialize particles
    for i in range(len(pp)):
        x, y = pp[i]
        if (x < xmin or x > xmax or y < ymin or y > ymax or lt[i] > max_lifetime):
            pp[i] = [np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)]
            his[i] = pp[i]
            lt[i] = 0
    
    # Random reinitialization
    num_to_reinit = int(0.05 * len(pp))
    if num_to_reinit > 0:
        idxs = np.random.choice(len(pp), num_to_reinit, replace=False)
        for idx in idxs:
            pp[idx] = [np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)]
            his[idx] = pp[idx]
            lt[idx] = 0
    
    # Build colored line segments
    n_active = len(pp)
    segments = np.zeros((n_active * tail_gap, 2, 2))
    colors_rgba = np.zeros((n_active * tail_gap, 4))
    
    speeds = np.linalg.norm(velocity, axis=1)
    max_speed = speeds.max() + 1e-9
    
    for i in range(n_active):
        # Get current grid cell color
        x, y = pp[i]
        cell_i = int(np.clip((y - ymin) / (ymax - ymin) * grid_res, 0, grid_res - 1))
        cell_j = int(np.clip((x - xmin) / (xmax - xmin) * grid_res, 0, grid_res - 1))
        this_feat_id = cell_dominant_features[cell_i, cell_j]
        
        # Look up color
        if this_feat_id in real_feature_rgba:
            r, g, b, _ = real_feature_rgba[this_feat_id]
            alpha_part = speeds[i] / max_speed
        else:
            r, g, b, _ = (0, 0, 0, 1)
            alpha_part = 0.3
        
        # Create trail segments
        for t in range(tail_gap):
            seg_idx = i * tail_gap + t
            segments[seg_idx, 0, :] = his[i, t, :]
            segments[seg_idx, 1, :] = his[i, t + 1, :]
            
            age_factor = (t + 1) / tail_gap
            alpha_final = max(0.15, alpha_part * age_factor)
            colors_rgba[seg_idx] = [r, g, b, alpha_final]
    
    lc_.set_segments(segments)
    lc_.set_colors(colors_rgba)
    
    return (lc_,)


def update_wind_vane(ax2, grid_cell, selected_features, all_feature_rgba, 
                     cell_dominant_features, grid_u_all, grid_v_all, col_labels):
    """Update wind vane visualization for the selected grid cell."""
    global bounding_box
    xmin, xmax, ymin, ymax = bounding_box
    
    # Clear previous wind vane
    ax2.clear()
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('Wind Vane', fontsize=12)
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    cell_i, cell_j = grid_cell
    grid_res = cell_dominant_features.shape[0]
    
    # Validate grid cell
    if cell_i < 0 or cell_i >= grid_res or cell_j < 0 or cell_j >= grid_res:
        return
    
    # Get dominant feature and calculate vectors for all features
    dominant_feature = cell_dominant_features[cell_i, cell_j]
    center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    
    # Draw grid cell marker (colored by dominant feature)
    if dominant_feature in all_feature_rgba:
        dominant_color = all_feature_rgba[dominant_feature]
    else:
        dominant_color = (0.5, 0.5, 0.5, 1.0)
    
    ax2.scatter(center_x, center_y, s=120, c=[dominant_color], marker='s', 
               zorder=10, label=f'Grid Cell ({cell_i},{cell_j})')
    
    # Calculate vector magnitudes using grid build method (consistent with dominance)
    vector_magnitudes = []
    vector_info = []
    
    for feat_idx in selected_features:
        # Use same method as grid cell dominant feature calculation
        grid_mag = np.sqrt(grid_u_all[feat_idx]**2 + grid_v_all[feat_idx]**2)
        mag = (grid_mag[cell_i, cell_j] + grid_mag[cell_i+1, cell_j] +
               grid_mag[cell_i, cell_j+1] + grid_mag[cell_i+1, cell_j+1]) / 4.0
        
        if mag > 0:
            # Get direction vector (averaged from corners for direction)
            corner_u = (grid_u_all[feat_idx, cell_i, cell_j] + 
                       grid_u_all[feat_idx, cell_i+1, cell_j] +
                       grid_u_all[feat_idx, cell_i, cell_j+1] + 
                       grid_u_all[feat_idx, cell_i+1, cell_j+1]) / 4.0
            corner_v = (grid_v_all[feat_idx, cell_i, cell_j] + 
                       grid_v_all[feat_idx, cell_i+1, cell_j] +
                       grid_v_all[feat_idx, cell_i, cell_j+1] + 
                       grid_v_all[feat_idx, cell_i+1, cell_j+1]) / 4.0
            
            # Scale direction vector to have consistent magnitude
            original_mag = np.sqrt(corner_u**2 + corner_v**2)
            if original_mag > 0:
                scale_factor = mag / original_mag
                scaled_u, scaled_v = corner_u * scale_factor, corner_v * scale_factor
            else:
                scaled_u, scaled_v = corner_u, corner_v
            
            vector_magnitudes.append(mag)
            vector_info.append((feat_idx, scaled_u, scaled_v, mag))
    
    if not vector_info:
        return
    
    # Dynamic scaling for visualization
    max_magnitude = max(vector_magnitudes)
    canvas_size = min(xmax - xmin, ymax - ymin)
    target_length = canvas_size * 0.3
    dynamic_scale = max_magnitude / target_length if max_magnitude > 0 else 1.0
    
    # Calculate endpoints for convex hull and ellipse
    endpoints = []
    for feat_idx, u, v, mag in vector_info:
        pos_endpoint = np.array([center_x + u / dynamic_scale, center_y + v / dynamic_scale])
        neg_endpoint = np.array([center_x - u / dynamic_scale, center_y - v / dynamic_scale])
        endpoints.append(pos_endpoint)
        endpoints.append(neg_endpoint)
    
    # Draw convex hull
    if len(endpoints) >= 6:  # Need at least 3 unique points (6 endpoints)
        try:
            endpoints_array = np.array(endpoints)
            hull = ConvexHull(endpoints_array)
            hull_points = endpoints_array[hull.vertices]
            
            hull_polygon = Polygon(hull_points, fill=False, edgecolor='black',
                                 linewidth=1.5, alpha=0.7, zorder=7, linestyle='--')
            ax2.add_patch(hull_polygon)
        except Exception:
            pass  # Skip hull if computation fails
    
    # Draw covariance ellipse
    if len(vector_info) >= 2:
        try:
            # Collect scaled vector endpoints relative to center
            ellipse_points = []
            for feat_idx, u, v, mag in vector_info:
                ellipse_points.append([u / dynamic_scale, v / dynamic_scale])
                ellipse_points.append([-u / dynamic_scale, -v / dynamic_scale])
            
            endpoints_matrix = np.array(ellipse_points)
            
            # Calculate covariance matrix
            cov_matrix = np.cov(endpoints_matrix.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            
            # Sort eigenvalues and eigenvectors in descending order
            order = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[order]
            eigenvecs = eigenvecs[:, order]
            
            # Calculate ellipse parameters
            confidence_scale = 1.5
            width = 2 * confidence_scale * np.sqrt(eigenvals[0])
            height = 2 * confidence_scale * np.sqrt(eigenvals[1])
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            
            # Create ellipse
            ellipse = Ellipse((center_x, center_y), width, height, angle=angle,
                            fill=False, edgecolor='black', linewidth=2,
                            alpha=0.6, zorder=11, linestyle='-')
            ax2.add_patch(ellipse)
        except Exception:
            pass  # Skip ellipse if computation fails
    
    # Draw arrows
    for feat_idx, u, v, mag in vector_info:
        # Get feature color
        if feat_idx in all_feature_rgba:
            base_color = all_feature_rgba[feat_idx]
            r, g, b = base_color[:3]
            # Boost intensity if dominant
            if feat_idx == dominant_feature:
                alpha = 0.9
            else:
                alpha = 0.6 * (mag / max_magnitude)
        else:
            r, g, b = (0.5, 0.5, 0.5)
            alpha = 0.4
        
        # Draw positive direction arrow
        ax2.arrow(center_x, center_y, u / dynamic_scale, v / dynamic_scale,
                 head_width=canvas_size * 0.02, head_length=canvas_size * 0.025,
                 fc=(r, g, b, alpha), ec=(r, g, b, alpha),
                 length_includes_head=True, zorder=9)
        
        # Draw negative direction arrow (dashed style) using annotate
        ax2.annotate('', xy=(center_x - u / dynamic_scale, center_y - v / dynamic_scale),
                    xytext=(center_x, center_y),
                    arrowprops=dict(arrowstyle='->', 
                                  connectionstyle='arc3,rad=0',
                                  linestyle='--',
                                  linewidth=1.5,
                                  color=(r, g, b, alpha*0.7)),
                    zorder=8)
        
        # Add feature label for dominant feature
        if feat_idx == dominant_feature and feat_idx < len(col_labels):
            label_x = center_x + (u / dynamic_scale) * 1.2
            label_y = center_y + (v / dynamic_scale) * 1.2
            ax2.text(label_x, label_y, col_labels[feat_idx],
                    fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    zorder=10)


def draw_grid_visualization(ax1, grid_x, grid_y, grid_res, cell_dominant_features, 
                          all_feature_rgba, grid_u_sum, grid_v_sum):
    """Draw grid lines and colored grid cells."""
    global bounding_box
    xmin, xmax, ymin, ymax = bounding_box
    
    # Draw grid lines
    n_rows, n_cols = grid_x.shape
    for col in range(n_cols):
        ax1.plot(grid_x[:, col], grid_y[:, col], color='gray', linestyle='--', 
                linewidth=0.3, alpha=0.5, zorder=1)
    for row in range(n_rows):
        ax1.plot(grid_x[row, :], grid_y[row, :], color='gray', linestyle='--', 
                linewidth=0.3, alpha=0.5, zorder=1)
    
    # Color grid cells based on dominant features
    for i in range(grid_res):
        for j in range(grid_res):
            # Get cell boundaries
            cell_xmin = xmin + j * (xmax - xmin) / grid_res
            cell_xmax = xmin + (j + 1) * (xmax - xmin) / grid_res
            cell_ymin = ymin + i * (ymax - ymin) / grid_res
            cell_ymax = ymin + (i + 1) * (ymax - ymin) / grid_res
            
            # Check if cell has no values (is masked out)
            # Sample the 4 corner vertices of this cell from grid_u_sum and grid_v_sum
            corner_u_sum = (grid_u_sum[i, j] + grid_u_sum[i+1, j] + 
                           grid_u_sum[i, j+1] + grid_u_sum[i+1, j+1])
            corner_v_sum = (grid_v_sum[i, j] + grid_v_sum[i+1, j] + 
                           grid_v_sum[i, j+1] + grid_v_sum[i+1, j+1])
            
            # If all corner values are zero, the cell is masked out
            if abs(corner_u_sum) < 1e-10 and abs(corner_v_sum) < 1e-10:
                cell_color = (1.0, 1.0, 1.0, 0.3)  # White for empty cells
            else:
                # Get dominant feature for this cell
                dominant_feature = cell_dominant_features[i, j]
                
                # Get color for this feature
                if dominant_feature in all_feature_rgba:
                    cell_color = all_feature_rgba[dominant_feature]
                    cell_color = (*cell_color[:3], 0.15)  # Low alpha for background
                else:
                    cell_color = (0.5, 0.5, 0.5, 0.1)  # Gray fallback
            
            # Draw colored rectangle for cell
            rect = Rectangle((cell_xmin, cell_ymin), 
                           cell_xmax - cell_xmin, 
                           cell_ymax - cell_ymin,
                           facecolor=cell_color, 
                           edgecolor='none',
                           zorder=0)
            ax1.add_patch(rect)


def setup_figure(valid_points, col_labels, feature_colors, lc, all_positions):
    """Setup the main figure with data points and mouse interaction."""
    global bounding_box
    xmin, xmax, ymin, ymax = bounding_box
    
    fig = plt.figure(figsize=(14, 7))
    
    # Create subplots with space for controls
    ax1 = plt.subplot2grid((20, 2), (0, 0), rowspan=18)
    ax2 = plt.subplot2grid((20, 2), (0, 1), rowspan=18)
    
    # Configure main plot
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.set_aspect('equal')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Feature Wind Map', fontsize=12)
    
    # Plot data points with different markers for different classes
    unique_labels = sorted(set(p.tmap_label for p in valid_points))
    markers = ["o", "s", "^", "D", "v", "<", ">"]
    
    for i, lab in enumerate(unique_labels):
        indices = [j for j, p in enumerate(valid_points) if p.tmap_label == lab]
        positions_lab = np.array([valid_points[j].position for j in indices])
        marker_style = markers[i % len(markers)]
        
        ax1.scatter(positions_lab[:, 0], positions_lab[:, 1],
                   marker=marker_style, color="gray", s=10,
                   label=f"Label {lab}", zorder=4)
    
    ax1.add_collection(lc)
    ax1.grid(False)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    
    # Configure wind vane plot
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('Wind Vane', fontsize=12)
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    # Add grid toggle control
    ax_grid_toggle = fig.add_axes([0.02, 0.02, 0.15, 0.1])
    grid_toggle = CheckButtons(ax_grid_toggle, ['Show Grid'], [False])
    
    return fig, ax1, ax2, grid_toggle


def generate_colors(n_colors):
    """Generate distinct colors for features."""
    if COLORCET_AVAILABLE and hasattr(cc, 'glasbey'):
        if n_colors <= len(cc.glasbey):
            return [cc.glasbey[i] for i in range(n_colors)]
        else:
            return [cc.glasbey[i % len(cc.glasbey)] for i in range(n_colors)]
    else:
        base_colors = [
            "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC949"
        ]
        return [base_colors[i % len(base_colors)] for i in range(n_colors)]


def main():
    """Main function to run the FeatureWind visualization."""
    global bounding_box, k, real_feature_rgba
    
    # Setup paths
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    tangent_map_path = os.path.join(repo_root, 'tangentmaps', 'breast_cancer.tmap')
    output_dir = os.path.join(repo_root, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    valid_points, all_grad_vectors, all_positions, col_labels = load_data(tangent_map_path)
    k = len(col_labels)
    
    # Setup bounding box
    xmin, xmax = all_positions[:, 0].min(), all_positions[:, 0].max()
    ymin, ymax = all_positions[:, 1].min(), all_positions[:, 1].max()
    
    # Add padding and make square
    padding = 0.05
    x_padding = (xmax - xmin) * padding
    y_padding = (ymax - ymin) * padding
    xmin, xmax = xmin - x_padding, xmax + x_padding
    ymin, ymax = ymin - y_padding, ymax + y_padding
    
    x_range, y_range = xmax - xmin, ymax - ymin
    if x_range > y_range:
        y_center = (ymin + ymax) / 2
        ymin, ymax = y_center - x_range / 2, y_center + x_range / 2
    else:
        x_center = (xmin + xmax) / 2
        xmin, xmax = x_center - y_range / 2, x_center + y_range / 2
    
    bounding_box = [xmin, xmax, ymin, ymax]
    
    # Parameters
    velocity_scale = 0.04
    grid_res = 40
    kdtree_scale = 0.03
    
    # Select top features and create grids
    top_k_indices, avg_magnitudes = select_top_features(all_grad_vectors, k)
    grid_data = create_grids(all_positions, all_grad_vectors, top_k_indices, 
                           grid_res, kdtree_scale)
    
    (interp_u_sum, interp_v_sum, interp_argmax, grid_x, grid_y, 
     grid_u_feats, grid_v_feats, cell_dominant_features, 
     grid_u_sum, grid_v_sum, grid_u_all, grid_v_all) = grid_data
    
    # Setup colors (only top 6 get distinct colors)
    top_6_indices = top_k_indices[:6]
    top_6_colors = generate_colors(6)
    
    all_feature_rgba = {}
    for i, feat_idx in enumerate(top_6_indices):
        all_feature_rgba[feat_idx] = to_rgba(top_6_colors[i])
    
    real_feature_rgba = {feat_idx: all_feature_rgba[feat_idx] 
                        for feat_idx in top_k_indices if feat_idx in all_feature_rgba}
    
    # Create particles and figure
    num_particles = 2500
    system = create_particles(num_particles)
    system['cell_dominant_features'] = cell_dominant_features
    
    feature_colors = [top_6_colors[list(top_6_indices).index(i)] if i in top_6_indices 
                     else '#808080' for i in top_k_indices]
    
    fig, ax1, ax2, grid_toggle = setup_figure(valid_points, col_labels, feature_colors, 
                                            system['linecoll'], all_positions)
    
    # Grid visualization state
    grid_visible = False
    grid_artists = []  # Store grid drawing elements for removal
    
    def toggle_grid_visualization(label):
        nonlocal grid_visible, grid_artists
        
        # Clear existing grid elements
        for artist in grid_artists:
            artist.remove()
        grid_artists.clear()
        
        # Toggle state
        grid_visible = not grid_visible
        
        if grid_visible:
            # Store current axes elements to track what we add
            current_patches = list(ax1.patches)
            current_lines = list(ax1.lines)
            
            # Draw grid
            draw_grid_visualization(ax1, grid_x, grid_y, grid_res, 
                                  cell_dominant_features, all_feature_rgba, 
                                  grid_u_sum, grid_v_sum)
            
            # Store new elements for later removal
            grid_artists.extend([p for p in ax1.patches if p not in current_patches])
            grid_artists.extend([l for l in ax1.lines if l not in current_lines])
        
        fig.canvas.draw_idle()
    
    # Connect grid toggle
    grid_toggle.on_clicked(toggle_grid_visualization)
    
    # Mouse interaction for wind vane
    current_grid_cell = [grid_res//2, grid_res//2]  # Start at center
    
    def on_mouse_move(event):
        if event.inaxes == ax1 and event.xdata is not None and event.ydata is not None:
            # Convert mouse position to grid cell
            cell_i = int(np.clip((event.ydata - ymin) / (ymax - ymin) * grid_res, 0, grid_res - 1))
            cell_j = int(np.clip((event.xdata - xmin) / (xmax - xmin) * grid_res, 0, grid_res - 1))
            
            if current_grid_cell != [cell_i, cell_j]:
                current_grid_cell[0], current_grid_cell[1] = cell_i, cell_j
                update_wind_vane(ax2, current_grid_cell, top_k_indices, all_feature_rgba,
                               cell_dominant_features, grid_u_all, grid_v_all, col_labels)
                fig.canvas.draw_idle()
    
    def on_key_press(event):
        if event.key == 'g':  # Press 'g' to toggle grid
            toggle_grid_visualization('Show Grid')
    
    # Connect events
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Initialize wind vane
    update_wind_vane(ax2, current_grid_cell, top_k_indices, all_feature_rgba,
                    cell_dominant_features, grid_u_all, grid_v_all, col_labels)
    
    # Create animation
    anim = FuncAnimation(
        fig, 
        lambda frame: update_particles(
            frame, system, interp_u_sum, interp_v_sum, interp_argmax, 
            velocity_scale, grid_u_sum, grid_v_sum, grid_res, cell_dominant_features
        ),
        frames=1000, interval=30, blit=False
    )
    
    # Save and show
    fig.savefig(os.path.join(output_dir, "featurewind_clean.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    main()