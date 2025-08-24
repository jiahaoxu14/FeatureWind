"""
Visualization core module for FeatureWind.

This module handles plot setup, figure preparation, and core visualization
components for feature flow visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from scipy.spatial import ConvexHull
import config


def prepare_figure(ax, valid_points, col_labels, k, grad_indices, feature_colors, lc, 
                  all_positions=None, all_grad_vectors=None):
    """
    Prepare the main visualization figure with data points and particle system.
    
    Args:
        ax: Matplotlib axes object
        valid_points: List of valid TangentPoint objects
        col_labels: Feature column labels
        k: Number of top features
        grad_indices: Gradient indices
        feature_colors: Colors for features
        lc: Line collection for particles
        all_positions: All position data
        all_grad_vectors: All gradient vectors
        
    Returns:
        int: Status code (0 for success)
    """
    xmin, xmax, ymin, ymax = config.bounding_box
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot underlying data points with different markers for each label
    feature_idx = 2  # Use feature index 2 for alpha computation
    domain_values = np.array([p.domain[feature_idx] for p in valid_points])
    domain_min, domain_max = domain_values.min(), domain_values.max()

    # Collect all labels from valid_points
    unique_labels = sorted(set(p.tmap_label for p in valid_points))
    print("Unique labels:", unique_labels)

    # Define multiple distinct markers
    markers = config.MARKER_STYLES

    for i, lab in enumerate(unique_labels):
        indices = [j for j, p in enumerate(valid_points) if p.tmap_label == lab]
        positions_lab = np.array([valid_points[j].position for j in indices])
        normalized = (np.array([valid_points[j].domain[feature_idx] for j in indices]) - domain_min) / (domain_max - domain_min + 1e-9)
        alphas = 0.2 + normalized * 0.8
        marker_style = markers[i % len(markers)]

        ax.scatter(
            positions_lab[:, 0],
            positions_lab[:, 1],
            marker=marker_style,
            color="gray",
            s=10,
            label=f"Label {lab}",
            zorder=4
        )

    # Add particle line collection
    ax.add_collection(lc)
    ax.grid(False)

    return 0


def prepare_wind_vane_subplot(ax2):
    """
    Prepare the wind vane subplot for feature vector visualization.
    
    Args:
        ax2: Matplotlib axes object for wind vane
    """
    ax2.set_xlim(-0.6, 0.6)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Wind Vane", fontsize=12, pad=10)
    
    # Draw unit circle
    circle = plt.Circle((0, 0), 0.5, fill=False, color='lightgray', linewidth=1)
    ax2.add_patch(circle)
    
    # Remove spines for cleaner look
    for spine in ax2.spines.values():
        spine.set_visible(False)


def update_wind_vane(ax2, mouse_data, system, col_labels, selected_features, feature_colors):
    """
    Update the wind vane visualization showing feature vectors at a grid cell.
    
    Args:
        ax2: Matplotlib axes object for wind vane
        mouse_data: Dictionary containing mouse interaction data
        system: Particle system dictionary
        col_labels: Feature column labels
        selected_features: Selected feature indices
        feature_colors: Colors for features
    """
    # Clear previous aggregate visualization
    for artist in ax2.collections + ax2.patches:
        if hasattr(artist, 'remove'):
            artist.remove()
    
    # Clear previous text and annotations
    for text in ax2.texts:
        text.remove()
    
    if 'grid_cell' not in mouse_data or mouse_data['grid_cell'] is None:
        return
    
    cell_i, cell_j = mouse_data['grid_cell']
    grid_res = mouse_data.get('grid_res', config.DEFAULT_GRID_RES)
    
    # Validate grid cell indices
    if cell_i < 0 or cell_i >= grid_res or cell_j < 0 or cell_j >= grid_res:
        return
    
    # Get vectors for ALL features from the all-features grid
    vectors_all = []
    mags_all = []
    
    if 'grid_u_all_feats' in system and 'grid_v_all_feats' in system:
        grid_u_all_feats = system['grid_u_all_feats']
        grid_v_all_feats = system['grid_v_all_feats']
        
        for feat_idx in range(len(col_labels)):
            # Use cell-center values directly (consistent with optimization and no bounds errors)
            u_val = grid_u_all_feats[feat_idx, cell_i, cell_j]
            v_val = grid_v_all_feats[feat_idx, cell_i, cell_j]
            vectors_all.append([u_val, v_val])
            mags_all.append(np.sqrt(u_val**2 + v_val**2))
    else:
        return  # Can't visualize without grid data
    
    # Get the dominant feature for this cell
    dominant_feature = -1
    if 'cell_dominant_features' in system:
        cell_dominant_features = system['cell_dominant_features']
        dominant_feature = cell_dominant_features[cell_i, cell_j]
    
    # Debug: Print wind vane cell information
    print(f"\nWind vane showing grid cell ({cell_i}, {cell_j})")
    print(f"Dominant feature: {dominant_feature}")
    print(f"Number of vectors: {len(vectors_all)}")
    
    # Place grid cell point at center of Wind Vane
    # Draw grid cell marker in Wind Vane (always at center)
    # Use color of dominant feature
    if (dominant_feature >= 0 and dominant_feature < len(selected_features) and
        dominant_feature < len(feature_colors)):
        dominant_color = feature_colors[dominant_feature]
    else:
        dominant_color = 'black'
    
    ax2.scatter(0, 0, c=dominant_color, s=80, marker='s', edgecolor='black', 
               linewidth=2, zorder=10, label=f'Cell ({cell_i},{cell_j})')
    
    # Draw gradient vector arrows for each feature in the grid cell
    # Calculate dynamic scaling to use 90% of canvas
    max_canvas_radius = 0.45  # 90% of 0.5 radius
    
    # Debug: Show which features are being considered in wind vane
    print(f"Selected features for wind vane: {selected_features}")
    
    # Only include selected top-k features in wind vane
    vectors_selected = []
    mags_selected = []
    features_selected = []
    
    for i, feat_idx in enumerate(selected_features):
        if feat_idx < len(vectors_all):
            # Use consistent cell-center magnitude (same as optimization)
            mag = mags_all[feat_idx]
            if mag > 1e-8:  # Only show non-zero vectors
                vectors_selected.append(vectors_all[feat_idx])
                mags_selected.append(mag)
                features_selected.append(feat_idx)
    
    if len(vectors_selected) > 0:
        # Find the longest vector
        max_mag = max(mags_selected)
        
        # Scale so longest vector takes up 45% of canvas radius (90% diameter coverage)
        if max_mag > 0:
            scale_factor = max_canvas_radius / max_mag
        else:
            scale_factor = 1.0
        
        # Calculate all endpoint positions for convex hull
        all_endpoints = []
        vector_info = []
        
        for i, (vector, mag, feat_idx) in enumerate(zip(vectors_selected, mags_selected, features_selected)):
            # Scale the gradient vector to match the consistent magnitude
            scaled_vector = np.array(vector) * scale_factor
            
            # Positive endpoint
            pos_end = scaled_vector
            # Negative endpoint
            neg_end = -scaled_vector
            
            all_endpoints.extend([pos_end, neg_end])
            vector_info.append({
                'vector': scaled_vector,
                'feat_idx': feat_idx,
                'mag': mag,
                'pos_end': pos_end,
                'neg_end': neg_end
            })
        
        if len(all_endpoints) >= 3:
            try:
                # Calculate convex hull
                hull = ConvexHull(all_endpoints)
                
                # Draw convex hull
                hull_points = np.array(all_endpoints)[hull.vertices]
                hull_polygon = plt.Polygon(hull_points, fill=True, alpha=0.1, 
                                         color='lightblue', edgecolor='blue', linewidth=1)
                ax2.add_patch(hull_polygon)
                
                # Calculate and draw covariance ellipse
                # Collect all vector endpoints (scaled the same way as vectors)
                ellipse_points = []
                for info in vector_info:
                    # Scale the gradient vector to match the consistent magnitude
                    scaled_vector = info['vector']
                    
                    # Positive endpoint
                    ellipse_points.append(info['pos_end'])
                    # Negative endpoint  
                    ellipse_points.append(info['neg_end'])
                
                if len(ellipse_points) >= 2:
                    ellipse_points = np.array(ellipse_points)
                    
                    try:
                        # Calculate covariance matrix
                        cov_matrix = np.cov(ellipse_points.T)
                        
                        # Eigendecomposition for ellipse parameters
                        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                        
                        # Sort eigenvalues and eigenvectors in descending order
                        # (eigh returns them in ascending order)
                        order = eigenvalues.argsort()[::-1]
                        eigenvalues = eigenvalues[order]
                        eigenvectors = eigenvectors[:, order]
                        
                        # Calculate ellipse parameters (scaled smaller to fit inside convex hull)
                        ellipse_scale = 0.8  # Make ellipse 80% of the span
                        width = 2 * np.sqrt(eigenvalues[0]) * ellipse_scale
                        height = 2 * np.sqrt(eigenvalues[1]) * ellipse_scale
                        
                        # Calculate angle of rotation (in degrees) using the major axis eigenvector
                        angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
                        
                        # Create ellipse centered at Wind Vane center
                        ellipse = Ellipse((0, 0), width, height, angle=angle,
                                        fill=False, color='red', linewidth=1.5, linestyle='--',
                                        alpha=0.7)
                        ax2.add_patch(ellipse)
                    except Exception as e:
                        print(f"Error calculating covariance ellipse: {e}")
                        
            except Exception as e:
                print(f"Error calculating convex hull: {e}")
        
        # Determine which endpoints are on the convex hull boundary
        if len(all_endpoints) >= 3:
            hull_vertex_indices = set(hull.vertices) if 'hull' in locals() else set()
        else:
            hull_vertex_indices = set()
        
        for info in vector_info:
            feat_idx = info['feat_idx']
            scaled_vector = info['vector']
            
            # Check if either endpoint is on the convex hull
            # Find indices by comparing arrays
            pos_on_hull = False
            neg_on_hull = False
            
            for idx, endpoint in enumerate(all_endpoints):
                if np.allclose(endpoint, info['pos_end']):
                    pos_on_hull = idx in hull_vertex_indices
                    break
                    
            for idx, endpoint in enumerate(all_endpoints):
                if np.allclose(endpoint, info['neg_end']):
                    neg_on_hull = idx in hull_vertex_indices
                    break
            
            # Get the consistent magnitude for this feature
            vector_magnitude = info['mag']
            
            # Scale the gradient vector to match the consistent magnitude
            if vector_magnitude > 1e-8:
                # Scale the vector to have the consistent magnitude
                pass  # Already scaled above
            else:
                continue  # Skip zero-magnitude vectors
            
            # Use feature colors with intensity based on vector magnitude and dominance
            if feat_idx < len(feature_colors):
                base_color = feature_colors[feat_idx]
                # Parse hex color to RGB
                if isinstance(base_color, str) and base_color.startswith('#'):
                    base_r = int(base_color[1:3], 16) / 255.0
                    base_g = int(base_color[3:5], 16) / 255.0
                    base_b = int(base_color[5:7], 16) / 255.0
                    
                    # Calculate intensity based on consistent vector magnitude
                    max_mag = max(mags_selected) if mags_selected else 1.0
                    mag_intensity = 0.3 + 0.7 * (vector_magnitude / max_mag)
                    
                    # Use soft dominance instead of hard dominance boost
                    # Get the probability for this feature from soft dominance
                    if 'cell_soft_dominance' in system and system['cell_soft_dominance'] is not None:
                        cell_soft_dom = system['cell_soft_dominance']
                        if feat_idx < cell_soft_dom.shape[2]:
                            soft_prob = cell_soft_dom[cell_i, cell_j, feat_idx]
                            dominance_boost = 0.8 + 0.8 * soft_prob  # Scale from 0.8 to 1.6 based on probability
                        else:
                            dominance_boost = 1.0
                    else:
                        # Fallback to hard dominance if soft dominance not available
                        dominance_boost = 1.3 if feat_idx == dominant_feature else 1.0
                    
                    intensity = min(1.0, mag_intensity * dominance_boost)
                    
                    # Add uncertainty visualization through desaturation
                    uncertainty_factor = 1.0  # Default: no desaturation
                    if 'cell_soft_dominance' in system and system['cell_soft_dominance'] is not None:
                        # Calculate uncertainty as entropy of the probability distribution
                        cell_soft_dom = system['cell_soft_dominance']
                        if feat_idx < cell_soft_dom.shape[2]:
                            probs = cell_soft_dom[cell_i, cell_j, :]
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
                    pos_color = (0, 0, 0, pos_alpha)
                    neg_color = (0.3, 0.3, 0.3, neg_alpha)
            else:
                # Fallback colors
                pos_color = (0, 0, 0, 0.6)
                neg_color = (0.3, 0.3, 0.3, 0.4)
            
            # Draw vector arrows
            ax2.annotate('', xy=info['pos_end'], xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=pos_color, lw=2,
                                      shrinkA=0, shrinkB=0, alpha=pos_color[3]))
            ax2.annotate('', xy=info['neg_end'], xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=neg_color, lw=1.5,
                                      shrinkA=0, shrinkB=0, alpha=neg_color[3]))
            
            # Add feature labels at endpoints (only for significant vectors)
            if vector_magnitude > max_mag * 0.3:  # Only label vectors > 30% of max magnitude
                if feat_idx < len(col_labels):
                    label = col_labels[feat_idx][:8]  # Truncate long labels
                    ax2.text(info['pos_end'][0] * 1.1, info['pos_end'][1] * 1.1, label,
                            fontsize=8, ha='center', va='center', alpha=0.7)


def setup_figure_layout():
    """
    Setup the main figure layout with subplots and controls.
    
    Returns:
        tuple: (fig, ax, ax2, control_axes) - Figure and axes objects
    """
    # Create figure with subplots and space for controls
    fig = plt.figure(figsize=(16, 8))
    
    # Create main subplots
    ax = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    
    # Control areas will be added by ui_controls module
    control_axes = {
        'mode_ax': plt.subplot2grid((2, 3), (1, 2)),
    }
    
    # Make both subplots square
    ax.set_aspect('equal')
    ax2.set_aspect('equal')
    
    return fig, ax, ax2, control_axes


def save_figure_frames(fig, output_dir, num_frames=5):
    """
    Save individual figure frames as PNG files.
    
    Args:
        fig: Matplotlib figure object
        output_dir (str): Output directory path
        num_frames (int): Number of frames to save
    """
    import os
    
    for frame in range(num_frames):
        fig.savefig(os.path.join(output_dir, f"frame_{frame}.png"), dpi=config.DPI)


def save_final_figure(fig, output_dir, filename="featurewind_figure.png"):
    """
    Save the final figure as a PNG file.
    
    Args:
        fig: Matplotlib figure object  
        output_dir (str): Output directory path
        filename (str): Filename for the saved figure
    """
    import os
    
    fig.savefig(os.path.join(output_dir, filename), dpi=config.DPI)