"""
Visualization core module for FeatureWind.

This module handles plot setup, figure preparation, and core visualization
components for feature flow visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull
import config


def prepare_figure(ax, valid_points, col_labels, k, grad_indices, feature_colors, lc, 
                  all_positions=None, all_grad_vectors=None, grid_res=None):
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
    
    # Set up grid ticks to match interpolation grid cell boundaries
    if grid_res is None:
        grid_res = config.DEFAULT_GRID_RES
    
    # Create grid cell boundary coordinates (not cell centers)
    # For grid_res cells, we need grid_res+1 boundary lines
    x_boundaries = np.linspace(xmin, xmax, grid_res + 1)
    y_boundaries = np.linspace(ymin, ymax, grid_res + 1)
    
    # Remove all ticks and tick labels
    ax.set_xticks([])  # Remove tick marks
    ax.set_yticks([])  # Remove tick marks
    
    # Manually draw grid lines to preserve them
    for x in x_boundaries:
        ax.axvline(x, alpha=0.3, linewidth=0.5, color='lightgray', zorder=1)
    for y in y_boundaries:
        ax.axhline(y, alpha=0.3, linewidth=0.5, color='lightgray', zorder=1)

    # Plot underlying data points with different markers for each label
    feature_idx = 2  # Use feature index 2 for alpha computation
    domain_values = np.array([p.domain[feature_idx] for p in valid_points])
    domain_min, domain_max = domain_values.min(), domain_values.max()

    # Collect all labels from valid_points
    unique_labels = sorted(set(p.tmap_label for p in valid_points))

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
    
    # Grid lines are manually drawn above, so no need for ax.grid()
    ax.set_axisbelow(True)  # Put grid behind other elements

    return 0


def prepare_wind_vane_subplot(ax2):
    """
    Prepare the wind vane subplot for feature vector visualization.
    
    Args:
        ax2: Matplotlib axes object for wind vane
    """
    ax2.set_xlim(-0.8, 0.8)  # Larger limits for bigger wind vane
    ax2.set_ylim(-0.8, 0.8)
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Wind Vane", fontsize=16, pad=15)  # Larger title
    
    # Draw reference circle - larger for bigger wind vane
    circle = plt.Circle((0, 0), 0.7, fill=False, color='lightgray', linewidth=1.5)
    ax2.add_patch(circle)
    
    # Remove spines for cleaner look
    for spine in ax2.spines.values():
        spine.set_visible(False)


def update_wind_vane(ax2, mouse_data, system, col_labels, selected_features, feature_colors, 
                     family_assignments=None):
    """
    Update the wind vane visualization with family-based colors.
    
    Args:
        ax2: Matplotlib axes object for wind vane
        mouse_data: Dictionary containing mouse interaction data
        system: Particle system dictionary
        col_labels: Feature column labels
        selected_features: Selected feature indices
        feature_colors: Colors for features (family-based)
        family_assignments: Optional array of family assignments for features
    """
    # Clear previous visualization completely
    for artist in ax2.collections + ax2.patches + ax2.lines:
        if hasattr(artist, 'remove'):
            artist.remove()
    
    # Clear previous text and annotations (including annotation arrows)
    for text in ax2.texts:
        text.remove()
    
    # Clear annotation arrows specifically
    for child in ax2.get_children():
        if hasattr(child, 'arrow_patch'):
            child.remove()
    
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
    max_canvas_radius = 0.63  # 90% of 0.7 radius (enlarged wind vane)
    
    
    # Only include selected top-k features in wind vane
    vectors_selected = []
    mags_selected = []
    features_selected = []
    
    # Collect vectors for selected features only
    # Include even small vectors to show complete picture
    for i, feat_idx in enumerate(selected_features):
        if feat_idx < len(vectors_all):
            vectors_selected.append(vectors_all[feat_idx])
            mags_selected.append(mags_all[feat_idx])
            features_selected.append(feat_idx)
    
    # Calculate the sum vector (what actually drives particles)
    sum_vector = np.sum(vectors_selected, axis=0) if vectors_selected else np.array([0, 0])
    sum_magnitude = np.linalg.norm(sum_vector)
    
    # Only show masked if the SUM has no flow (matching particle behavior)
    is_masked_cell = sum_magnitude < 1e-6
    
    # Show masked cell indication if the sum vector has no flow
    if is_masked_cell:
        # Draw a visual indication that this cell is masked/has no flow
        ax2.text(0, 0, 'MASKED\nCELL', ha='center', va='center', 
                fontsize=12, color='red', weight='bold', alpha=0.7)
        ax2.scatter(0, 0, c='lightgray', s=120, marker='X', edgecolor='red', 
                   linewidth=2, zorder=10, alpha=0.7)
        return
    
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
            # Skip truly zero vectors for cleaner visualization
            if mag < 1e-12:  # More lenient threshold
                continue
                
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
                                         facecolor='lightblue', edgecolor='blue', linewidth=1)
                ax2.add_patch(hull_polygon)
                
            except Exception as e:
                pass  # Silently skip convex hull on error
        
        # Draw wind vane arrow showing the sum vector (actual flow direction)
        # This is drawn regardless of convex hull success
        if sum_magnitude > 1e-6 and len(vectors_selected) > 0:
            # Create traditional wind vane with TRULY fixed geometry and alpha-based magnitude encoding
            vane_length = 0.35  # Fixed radius (0.7 diameter fits well in unit circle)
            
            # Get ONLY the direction (unit vector) from sum vector - ignore magnitude for positioning
            if sum_magnitude > 1e-8:
                flow_direction = sum_vector / sum_magnitude  # Pure direction, no magnitude influence
            else:
                flow_direction = np.array([1, 0])  # Default direction for zero flow
            
            # Alpha encoding for magnitude (0.3 to 0.9 range)  
            # Clamp to valid alpha range [0, 1]
            alpha_ratio = min(1.0, sum_magnitude / max_mag) if max_mag > 0 else 0.0
            magnitude_alpha = max(0.3, min(0.9, 0.3 + 0.6 * alpha_ratio))
            
            # Wind vane components with FIXED geometry - clearly distinct from thin feature vectors
            perp_direction = np.array([-flow_direction[1], flow_direction[0]])  # Perpendicular
            
            # 1. Main shaft (thick central line) - much thicker than feature vectors
            shaft_length = vane_length * 0.8  # 80% of radius
            shaft_start = -flow_direction * shaft_length * 0.5
            shaft_end = flow_direction * shaft_length * 0.5
            ax2.plot([shaft_start[0], shaft_end[0]], [shaft_start[1], shaft_end[1]], 
                    color=dominant_color, linewidth=6, alpha=magnitude_alpha, 
                    solid_capstyle='round', zorder=20)  # Much thicker than feature vectors
            
            # 2. Large arrow head (pointing in flow direction) - fixed position at radius
            arrow_head_pos = flow_direction * vane_length  # Fixed at vane_length radius
            arrow_size = vane_length * 0.4  # Larger arrow head
            
            arrow_vertices = np.array([
                arrow_head_pos,  # Tip at fixed radius
                arrow_head_pos - flow_direction * arrow_size + perp_direction * arrow_size * 0.5,  # Left base
                arrow_head_pos - flow_direction * arrow_size - perp_direction * arrow_size * 0.5   # Right base
            ])
            
            arrow_triangle = plt.Polygon(arrow_vertices, color=dominant_color, 
                                       alpha=magnitude_alpha, zorder=21)
            ax2.add_patch(arrow_triangle)
            
            # 3. Large flag tail (showing flow origin) - fixed position opposite to arrow
            flag_center = -flow_direction * vane_length  # Fixed at opposite end
            flag_size = vane_length * 0.35  # Large flag
            
            # Create rectangular flag
            flag_vertices = np.array([
                flag_center + perp_direction * flag_size * 0.5,  # Top
                flag_center + perp_direction * flag_size * 0.5 + flow_direction * flag_size * 0.6,  # Top-right
                flag_center - perp_direction * flag_size * 0.5 + flow_direction * flag_size * 0.6,  # Bottom-right
                flag_center - perp_direction * flag_size * 0.5,  # Bottom
            ])
            
            flag_rectangle = plt.Polygon(flag_vertices, color=dominant_color, 
                                       alpha=magnitude_alpha * 0.8, zorder=21)
            ax2.add_patch(flag_rectangle)
        
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
            
            # Only use color for vectors on the convex hull boundary
            if (pos_on_hull or neg_on_hull) and feat_idx < len(feature_colors):
                # Vector is on hull boundary - use feature color
                base_color = feature_colors[feat_idx]
                
                # Import color utilities for family-based coloring
                try:
                    from color_system import hex_to_rgb
                    
                    # Use base color directly without lightness ramp to match particle colors
                    base_r, base_g, base_b = hex_to_rgb(base_color)
                    
                    # Higher alpha for hull vectors to make them prominent
                    pos_alpha = 0.9 if pos_on_hull else 0.3
                    neg_alpha = 0.9 if neg_on_hull else 0.3
                    
                    pos_color = (base_r, base_g, base_b, pos_alpha)
                    neg_color = (base_r, base_g, base_b, neg_alpha)
                    
                except ImportError:
                    # Fallback if color_system is not available
                    if isinstance(base_color, str) and base_color.startswith('#'):
                        base_r = int(base_color[1:3], 16) / 255.0
                        base_g = int(base_color[3:5], 16) / 255.0
                        base_b = int(base_color[5:7], 16) / 255.0
                        
                        pos_alpha = 0.9 if pos_on_hull else 0.3
                        neg_alpha = 0.9 if neg_on_hull else 0.3
                        
                        pos_color = (base_r, base_g, base_b, pos_alpha)
                        neg_color = (base_r, base_g, base_b, neg_alpha)
                    else:
                        # Use gray for non-hull vectors
                        pos_color = (0.6, 0.6, 0.6, 0.9 if pos_on_hull else 0.3)
                        neg_color = (0.6, 0.6, 0.6, 0.9 if neg_on_hull else 0.3)
            else:
                # Vector is NOT on hull boundary - use gray color
                gray_value = 0.6
                # Lower alpha for non-hull vectors
                pos_alpha = 0.2
                neg_alpha = 0.2
                pos_color = (gray_value, gray_value, gray_value, pos_alpha)
                neg_color = (gray_value, gray_value, gray_value, neg_alpha)
            
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
    Setup the main figure layout with professional styling.
    
    Returns:
        tuple: (fig, ax, ax2) - Figure and axes objects
    """
    # Create figure with enlarged wind vane and space for controls and legends
    fig = plt.figure(figsize=(18, 10))
    
    # Apply professional background
    from color_system import BACKGROUND_COLOR
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Create main subplots - leave space for legends on the left
    ax = plt.subplot2grid((2, 5), (0, 1), rowspan=2, colspan=2)  # Main plot takes 2/5 width  
    ax2 = plt.subplot2grid((2, 5), (0, 3), rowspan=2, colspan=2)  # Wind vane takes 2/5 width
    
    # Make both subplots square
    ax.set_aspect('equal')
    ax2.set_aspect('equal')
    
    return fig, ax, ax2


def apply_professional_styling(fig, ax1, ax2):
    """
    Apply professional color styling to the entire visualization.
    
    Args:
        fig: matplotlib figure object
        ax1: main plot axes
        ax2: wind vane axes
    """
    from color_system import BACKGROUND_COLOR, GRID_COLOR, TEXT_COLOR
    
    # Set figure background
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Style main plot
    ax1.set_facecolor(BACKGROUND_COLOR)
    ax1.grid(True, alpha=0.3, linewidth=0.5, color=GRID_COLOR, zorder=0)
    
    # Remove tick marks but keep grid
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Style spines
    for spine in ax1.spines.values():
        spine.set_color(TEXT_COLOR)
        spine.set_linewidth(0.8)
    
    # Style wind vane (keep white background for contrast)
    ax2.set_facecolor('white')
    
    # Style wind vane spines
    for spine in ax2.spines.values():
        spine.set_color(TEXT_COLOR)
        spine.set_linewidth(0.5)
        
    # Wind vane title styling
    ax2.set_title("Wind Vane", fontsize=14, color=TEXT_COLOR, pad=15, weight='bold')


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