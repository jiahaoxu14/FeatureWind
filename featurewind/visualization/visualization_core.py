"""
Visualization core module for FeatureWind.

This module handles plot setup, figure preparation, and core visualization
components for feature flow visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull
from .. import config


def highlight_unmasked_cells(ax, system, grid_res=None, valid_points=None):
    """
    Highlight unmasked cells using the SAME buffer-based mask as the final field.
    This avoids mismatches between the visual overlay and runtime masking.
    """
    if grid_res is None:
        grid_res = config.DEFAULT_GRID_RES

    xmin, xmax, ymin, ymax = config.bounding_box
    dx = (xmax - xmin) / grid_res
    dy = (ymax - ymin) / grid_res

    # Build buffer-based mask directly from data positions
    positions = None
    if valid_points:
        try:
            positions = np.array([p.position for p in valid_points])
        except Exception:
            positions = None

    # Default: if no positions available, fall back to all masked (no overlay)
    if positions is None or len(positions) == 0:
        return

    # Initialize all cells as masked; unmask buffered regions around each point
    unmasked_cells = np.zeros((grid_res, grid_res), dtype=bool)

    buffer_x = dx * config.MASK_BUFFER_FACTOR
    buffer_y = dy * config.MASK_BUFFER_FACTOR

    for px, py in positions:
        # Buffer rectangle in index space
        i_start = max(0, int((py - buffer_y - ymin) / dy))
        i_end   = min(grid_res, int((py + buffer_y - ymin) / dy) + 1)
        j_start = max(0, int((px - buffer_x - xmin) / dx))
        j_end   = min(grid_res, int((px + buffer_x - xmin) / dx) + 1)
        unmasked_cells[i_start:i_end, j_start:j_end] = True

        # Ensure exact cell is unmasked
        i = int((py - ymin) / dy)
        j = int((px - xmin) / dx)
        i = max(0, min(grid_res - 1, i))
        j = max(0, min(grid_res - 1, j))
        unmasked_cells[i, j] = True

    # Persist mask for wind vane and other consumers
    try:
        system['unmasked_cells'] = unmasked_cells
        system['cells_with_data'] = unmasked_cells.copy()
        if positions is not None:
            system['positions'] = positions
    except Exception:
        pass

    # Optionally draw semi-transparent gray rectangles over unmasked cells
    if not getattr(config, 'SHOW_UNMASKED_OVERLAY', True):
        return
    for i in range(grid_res):
        for j in range(grid_res):
            if unmasked_cells[i, j]:
                x_left = xmin + j * dx
                y_bottom = ymin + i * dy
                rect = Rectangle((x_left, y_bottom), dx, dy,
                                 facecolor='gray', alpha=0.1,
                                 edgecolor='gray', linewidth=0.5, zorder=2)
                ax.add_patch(rect)


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
    
    # Manually draw grid lines to preserve them (optional)
    if getattr(config, 'SHOW_GRID_LINES', True):
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

        # Hollow or solid data points based on config
        if getattr(config, 'HOLLOW_DATA_POINTS', True):
            edge_alpha = float(getattr(config, 'DATA_POINT_ALPHA', 0.35))
            edge_width = float(getattr(config, 'DATA_POINT_EDGEWIDTH', 0.6))
            # Stroke color #555 with configured alpha
            stroke_val = 0x55 / 255.0
            edge_color = (stroke_val, stroke_val, stroke_val, edge_alpha)
            ax.scatter(
                positions_lab[:, 0],
                positions_lab[:, 1],
                marker=marker_style,
                facecolors='none',
                edgecolors=edge_color,
                linewidths=edge_width,
                s=getattr(config, 'DATA_POINT_SIZE', 48),
                label=f"Label {lab}",
                zorder=getattr(config, 'DATA_POINT_ZORDER', 20)
            )
        else:
            fill_alpha = float(getattr(config, 'DATA_POINT_ALPHA', 0.35))
            fill_color = (0.5, 0.5, 0.5, fill_alpha)
            ax.scatter(
                positions_lab[:, 0],
                positions_lab[:, 1],
                marker=marker_style,
                color=fill_color,
                edgecolors='#555555',
                linewidths=float(getattr(config, 'DATA_POINT_EDGEWIDTH', 0.6)),
                s=getattr(config, 'DATA_POINT_SIZE', 48),
                label=f"Label {lab}",
                zorder=getattr(config, 'DATA_POINT_ZORDER', 20)
            )

    # Add particle line collection only in feature wind map mode
    try:
        if (getattr(config, 'VIS_MODE', 'feature_wind_map') == 'feature_wind_map'
            and lc is not None
            and not getattr(config, 'FEATURE_CLOCK_ENABLED', False)):
            ax.add_collection(lc)
        else:
            # Ensure trails are hidden when Feature Clock is enabled
            if lc is not None:
                try:
                    lc.set_visible(False)
                except Exception:
                    pass
    except Exception:
        pass
    
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
    # Title will be set in apply_professional_styling for consistent styling
    
    # Draw reference circle - larger for bigger wind vane
    circle = plt.Circle((0, 0), 0.7, fill=False, color='lightgray', linewidth=1.5)
    ax2.add_patch(circle)
    
    # Remove spines for cleaner look
    for spine in ax2.spines.values():
        spine.set_visible(False)


def update_wind_vane(ax2, mouse_data, system, col_labels, selected_features, feature_colors, 
                     family_assignments=None, feature_clock_override=False):
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
    
    # Validate grid cell indices against actual grid
    if cell_i < 0 or cell_j < 0:
        return
    # If system provides grid shape, prefer it for bounds
    try:
        if 'grid_u_sum' in system:
            rs, cs = system['grid_u_sum'].shape
            if cell_i >= rs or cell_j >= cs:
                return
    except Exception:
        if cell_i >= grid_res or cell_j >= grid_res:
            return
    
    # Consistent masking check using unified mask from main view
    masked = False
    unmasked = system.get('unmasked_cells', None)
    if unmasked is not None and 0 <= cell_i < unmasked.shape[0] and 0 <= cell_j < unmasked.shape[1]:
        masked = not bool(unmasked[cell_i, cell_j])

    if masked:
        ax2.text(0.5, 0.5, "Masked cell", transform=ax2.transAxes,
                 ha='center', va='center', fontsize=10, color='gray')
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

    # Diagnostics: report whether the hovered cell center lies inside the data convex hull
    try:
        from scipy.spatial import Delaunay
        xmin, xmax, ymin, ymax = config.bounding_box
        # Derive grid_res from system grids to avoid mismatch
        rs, cs = system['grid_u_all_feats'].shape[1:]
        dx = (xmax - xmin) / cs
        dy = (ymax - ymin) / rs
        cell_center_x = xmin + (cell_j + 0.5) * dx
        cell_center_y = ymin + (cell_i + 0.5) * dy
        # Build/cached triangulation of positions
        if 'positions' in system and system['positions'] is not None:
            if 'positions_tris' not in system or system.get('positions_tris_n', 0) != len(system['positions']):
                try:
                    system['positions_tris'] = Delaunay(system['positions'])
                    system['positions_tris_n'] = len(system['positions'])
                except Exception:
                    system['positions_tris'] = None
            tri = system.get('positions_tris', None)
            if tri is not None:
                inside = tri.find_simplex(np.array([[cell_center_x, cell_center_y]])) >= 0
                print(f"  ↳ Cell center ({cell_center_x:.5f}, {cell_center_y:.5f}) inside data hull: {bool(inside)}")
    except Exception:
        pass

    # Print all feature vectors for this unmasked cell to the terminal (on cell change)
    try:
        last_printed = system.get('vane_last_printed_cell', None)
        current_cell = (int(cell_i), int(cell_j))
        if last_printed != current_cell:
            system['vane_last_printed_cell'] = current_cell
            print(f"\nWind-Vane cell ({cell_i},{cell_j}) feature vectors:")
            for feat_idx in range(len(col_labels)):
                u_val, v_val = vectors_all[feat_idx]
                mag_val = mags_all[feat_idx]
                name = col_labels[feat_idx] if feat_idx < len(col_labels) else f"feat_{feat_idx}"
                print(f"  {feat_idx:3d} | {name:>20s} : (u={u_val:+.5f}, v={v_val:+.5f}) | |v|={mag_val:.5f}")
    except Exception:
        pass
    
    # Get the dominant feature for this cell
    dominant_feature = -1
    if 'cell_dominant_features' in system:
        cell_dominant_features = system['cell_dominant_features']
        dominant_feature = cell_dominant_features[cell_i, cell_j]
    
    
    # Place grid cell point at center of Wind Vane
    # Draw grid cell marker in Wind Vane (always at center)
    # Use color of magnitude-dominant feature for center marker
    # dominant_feature is an absolute index; ensure it is among selected features
    if (dominant_feature is not None and dominant_feature >= 0 and
        (dominant_feature in selected_features) and
        dominant_feature < len(feature_colors)):
        center_marker_color = feature_colors[dominant_feature]
    else:
        center_marker_color = 'black'
    
    ax2.scatter(0, 0, c=center_marker_color, s=80, marker='s', edgecolor='black', 
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
    
    # Compute summed vector for direction/magnitude cues in the vane
    sum_vector = np.sum(vectors_selected, axis=0) if vectors_selected else np.array([0.0, 0.0])
    sum_magnitude = np.linalg.norm(sum_vector)
    
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

        # Ensure a deterministic shared color map exists for this cell when Feature Clock is enabled
        if getattr(config, 'FEATURE_CLOCK_ENABLED', False):
            try:
                cell_key = (int(cell_i), int(cell_j))
                existing_key = system.get('feature_clock_color_cell')
                shared_map = system.get('feature_clock_color_map') if existing_key == cell_key else None
            except Exception:
                shared_map = None
            if shared_map is None:
                try:
                    try:
                        top_n = int(getattr(config, 'FEATURE_CLOCK_TOP_N', 4))
                    except Exception:
                        top_n = 4
                    # Rank by magnitude (desc)
                    rank_sorted = sorted([(mags_selected[i], features_selected[i]) for i in range(len(features_selected))],
                                         key=lambda x: x[0], reverse=True)
                    order_index = {f: r for r, (_, f) in enumerate(rank_sorted)}
                    top_feats = [f for (_, f) in rank_sorted[:max(1, top_n)]]
                    # Hull features
                    hull_feats = set()
                    try:
                        hull = ConvexHull(all_endpoints)
                        hull_vertex_indices = set(hull.vertices)
                        for info in vector_info:
                            pos_on_hull = any(np.allclose(info['pos_end'], all_endpoints[idx]) for idx in hull_vertex_indices)
                            neg_on_hull = any(np.allclose(info['neg_end'], all_endpoints[idx]) for idx in hull_vertex_indices)
                            if pos_on_hull or neg_on_hull:
                                hull_feats.add(info['feat_idx'])
                    except Exception:
                        hull_feats = set()
                    # Deterministic union ordering
                    hull_minus_sorted = sorted([f for f in hull_feats if f not in set(top_feats)], key=lambda f: order_index.get(f, 1e9))
                    union_feats = top_feats + hull_minus_sorted
                    palette = getattr(config, 'GLASBEY_COLORS', ['#1f77b4'])
                    shared_map = {f: palette[i % len(palette)] for i, f in enumerate(union_feats)}
                    system['feature_clock_color_cell'] = cell_key
                    system['feature_clock_color_map'] = shared_map
                except Exception:
                    pass

        # Feature Clock mode: highlight top-N by magnitude and remove vane arrow/convex hull logic
        if feature_clock_override:
            try:
                top_n = int(getattr(config, 'FEATURE_CLOCK_TOP_N', 4))
            except Exception:
                top_n = 4
            # Rank features by magnitude (desc) within current selection
            rank_sorted = sorted([(mags_selected[i], features_selected[i]) for i in range(len(features_selected))],
                                 key=lambda x: x[0], reverse=True)
            top_feats = [f for (_, f) in rank_sorted[:max(1, top_n)]]

            # Prefer a shared cell color map if one exists
            cmap = None
            try:
                if system.get('feature_clock_color_cell') == (int(cell_i), int(cell_j)):
                    cmap = system.get('feature_clock_color_map', None)
            except Exception:
                cmap = None
            # If none, build from top_feats only (deterministic order by rank)
            if cmap is None:
                palette = getattr(config, 'GLASBEY_COLORS', ['#1f77b4'])
                cmap = {f: palette[r % len(palette)] for r, f in enumerate(top_feats)}
                try:
                    system['feature_clock_color_cell'] = (int(cell_i), int(cell_j))
                    system['feature_clock_color_map'] = cmap
                except Exception:
                    pass

            # Draw ONLY top-N vectors in the Feature Clock pane (positive direction only)
            top_set = set(top_feats)
            for info in vector_info:
                feat_idx = info['feat_idx']
                if feat_idx not in top_set:
                    continue
                color = cmap.get(feat_idx, getattr(config, 'GLASBEY_COLORS', ['#1f77b4'])[0])
                ax2.arrow(0, 0, info['pos_end'][0], info['pos_end'][1],
                          head_width=0.04, head_length=0.04, fc=color, ec=color,
                          linewidth=1.8, alpha=0.9, zorder=8)
                # Label highlighted vectors
                if feat_idx < len(col_labels):
                    ax2.text(info['pos_end'][0] * 1.1, info['pos_end'][1] * 1.1,
                             col_labels[feat_idx][:8], fontsize=8, ha='center', va='center', alpha=0.9)

            # Skip classic hull/vane arrow flow in Feature Clock mode
            return
        
        # Handle degenerate cases for convex hull
        if len(vectors_selected) == 1:
            # Single feature case - draw special visualization
            vector = vectors_selected[0]
            scaled_vector = np.array(vector) * scale_factor
            # Use shared Feature Clock color if available for coherence
            color = feature_colors[features_selected[0]] if features_selected[0] < len(feature_colors) else 'black'
            if getattr(config, 'FEATURE_CLOCK_ENABLED', False):
                try:
                    if system.get('feature_clock_color_cell') == (int(cell_i), int(cell_j)):
                        cmap = system.get('feature_clock_color_map', None)
                        if cmap and features_selected[0] in cmap:
                            color = cmap[features_selected[0]]
                except Exception:
                    pass
            
            # Draw bidirectional arrow
            ax2.arrow(0, 0, scaled_vector[0], scaled_vector[1], 
                     head_width=0.05, head_length=0.05, 
                     fc=color, ec=color, linewidth=2, alpha=0.8, zorder=8)
            ax2.arrow(0, 0, -scaled_vector[0], -scaled_vector[1], 
                     head_width=0.05, head_length=0.05, 
                     fc=color, ec=color, linewidth=2, alpha=0.8, zorder=8)
            
            # Draw magnitude circle
            from matplotlib.patches import Circle
            magnitude_circle = Circle((0, 0), radius=np.linalg.norm(scaled_vector),
                                     fill=False, color=color, linewidth=1.5, 
                                     linestyle='--', alpha=0.4, zorder=7)
            ax2.add_patch(magnitude_circle)
            
            # Add feature label
            ax2.text(scaled_vector[0]*1.2, scaled_vector[1]*1.2, 
                    col_labels[features_selected[0]][:20], 
                    ha='center', va='center', fontsize=9, color=color)
            
        elif len(all_endpoints) >= 3:
            try:
                # Calculate convex hull
                hull = ConvexHull(all_endpoints)
                
                # Calculate convex hull for boundary detection (but don't draw it)
                hull_points = np.array(all_endpoints)[hull.vertices]
                # hull_polygon = plt.Polygon(hull_points, fill=True, alpha=0.1, 
                #                          facecolor='lightblue', edgecolor='blue', linewidth=1)
                # ax2.add_patch(hull_polygon)  # Hidden per user request
                
            except Exception as e:
                # Degenerate hull (e.g., collinear or duplicate points) — draw all vectors as arrows
                for i, (vector, feat_idx) in enumerate(zip(vectors_selected, features_selected)):
                    scaled_vector = np.array(vector) * scale_factor
                    color = feature_colors[feat_idx] if feat_idx < len(feature_colors) else 'black'
                    ax2.arrow(0, 0, scaled_vector[0], scaled_vector[1],
                              head_width=0.04, head_length=0.04,
                              fc=color, ec=color, linewidth=1.5, alpha=0.7, zorder=8-i)
                    ax2.arrow(0, 0, -scaled_vector[0], -scaled_vector[1],
                              head_width=0.04, head_length=0.04,
                              fc=color, ec=color, linewidth=1.5, alpha=0.7, zorder=8-i)
        
        # Wind meter removed per user request

        # Draw wind vane arrow showing the sum vector (actual flow direction)
        # Always draw for unmasked cells when there are selected vectors
        if len(vectors_selected) > 0:
            # Create traditional wind vane with TRULY fixed geometry and alpha-based magnitude encoding
            # Slightly smaller geometry for a subtler appearance
            vane_length = 0.30  # reduced from 0.35
            
            # Get ONLY the direction (unit vector) from sum vector - ignore magnitude for positioning
            if sum_magnitude > 1e-8:
                flow_direction = sum_vector / sum_magnitude  # Pure direction, no magnitude influence
            else:
                flow_direction = np.array([1, 0])  # Default direction for zero flow
            
            # Find the feature with highest contribution in the flow direction
            directional_contributions = []
            for i, (vector, feat_idx) in enumerate(zip(vectors_selected, features_selected)):
                # Calculate dot product with flow direction (projection)
                contribution = np.dot(vector, flow_direction)
                directional_contributions.append((contribution, feat_idx))
            
            # Find feature with highest directional contribution
            if directional_contributions:
                best_contribution, best_feature_idx = max(directional_contributions, key=lambda x: x[0])
                # Use color of the feature that contributes most in flow direction
                if best_feature_idx < len(feature_colors):
                    dominant_color = feature_colors[best_feature_idx]
                else:
                    dominant_color = 'black'
            else:
                dominant_color = 'black'
            
            # Alpha aligned with particle trail alpha: use current frame max particle speed
            # Compute a comparable "cell speed" as ||sum_vector|| scaled like particle velocities
            try:
                last_v = system.get('last_velocity', None)
                if last_v is not None and len(last_v) > 0:
                    max_speed = float(np.linalg.norm(last_v, axis=1).max() + 1e-9)
                else:
                    max_speed = 1.0
            except Exception:
                max_speed = 1.0
            cell_speed = float(np.linalg.norm(sum_vector)) * float(getattr(config, 'velocity_scale', 1.0))
            speed_ratio = cell_speed / max_speed if max_speed > 0 else 0.0
            speed_ratio = max(0.0, min(1.0, speed_ratio))
            magnitude_alpha = max(0.3, min(1.0, 0.3 + 0.7 * speed_ratio))
            
            
            # Wind vane components with FIXED geometry - clearly distinct from thin feature vectors
            perp_direction = np.array([-flow_direction[1], flow_direction[0]])  # Perpendicular
            
            # 1. Main shaft (thick central line) - much thicker than feature vectors
            shaft_length = vane_length * 0.75  # reduced from 0.8
            shaft_start = -flow_direction * shaft_length * 0.5
            shaft_end = flow_direction * shaft_length * 0.5
            ax2.plot([shaft_start[0], shaft_end[0]], [shaft_start[1], shaft_end[1]], 
                    color=dominant_color, linewidth=6, alpha=magnitude_alpha, 
                    solid_capstyle='round', zorder=20)  # Much thicker than feature vectors
            
            # 2. Large arrow head (pointing in flow direction) - fixed position at radius
            arrow_head_pos = flow_direction * vane_length  # Fixed at vane_length radius
            arrow_size = vane_length * 0.32  # reduced from 0.4
            
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
            flag_size = vane_length * 0.28  # reduced from 0.35
            
            # Create rectangular flag
            flag_vertices = np.array([
                flag_center + perp_direction * flag_size * 0.5,  # Top
                flag_center + perp_direction * flag_size * 0.5 + flow_direction * flag_size * 0.6,  # Top-right
                flag_center - perp_direction * flag_size * 0.5 + flow_direction * flag_size * 0.6,  # Bottom-right
                flag_center - perp_direction * flag_size * 0.5,  # Bottom
            ])
            
            flag_rectangle = plt.Polygon(flag_vertices, color=dominant_color, 
                                       alpha=magnitude_alpha, zorder=21)
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
            
            # Color assignment with Feature Clock coherence
            use_shared = False
            shared_color = None
            if getattr(config, 'FEATURE_CLOCK_ENABLED', False):
                try:
                    if system.get('feature_clock_color_cell') == (int(cell_i), int(cell_j)):
                        cmap = system.get('feature_clock_color_map', None)
                        if cmap and feat_idx in cmap:
                            shared_color = cmap[feat_idx]
                            use_shared = True
                except Exception:
                    use_shared = False

            # Only use color for vectors on the convex hull boundary
            if use_shared:
                base_color = shared_color
                try:
                    from .color_system import hex_to_rgb
                    base_r, base_g, base_b = hex_to_rgb(base_color)
                    pos_alpha = 0.9
                    pos_color = (base_r, base_g, base_b, pos_alpha)
                except Exception:
                    pos_color = (0.6, 0.6, 0.6, 0.9)
            elif (pos_on_hull or neg_on_hull) and feat_idx < len(feature_colors):
                # Vector is on hull boundary - use feature color
                base_color = feature_colors[feat_idx]
                
                # Import color utilities for family-based coloring
                try:
                    from .color_system import hex_to_rgb
                    
                    # Use base color directly without lightness ramp to match particle colors
                    base_r, base_g, base_b = hex_to_rgb(base_color)
                    
                    # Higher alpha for hull vectors to make them prominent (either endpoint on hull)
                    pos_alpha = 0.9
                    pos_color = (base_r, base_g, base_b, pos_alpha)
                    
                except ImportError:
                    # Fallback if color_system is not available
                    if isinstance(base_color, str) and base_color.startswith('#'):
                        base_r = int(base_color[1:3], 16) / 255.0
                        base_g = int(base_color[3:5], 16) / 255.0
                        base_b = int(base_color[5:7], 16) / 255.0
                        pos_color = (base_r, base_g, base_b, 0.9)
                    else:
                        # Use gray for non-hull vectors
                        pos_color = (0.6, 0.6, 0.6, 0.9)
            else:
                # Vector is NOT on hull boundary
                if use_shared and shared_color is not None:
                    try:
                        from .color_system import hex_to_rgb
                        base_r, base_g, base_b = hex_to_rgb(shared_color)
                        pos_alpha = 0.9
                        pos_color = (base_r, base_g, base_b, pos_alpha)
                    except Exception:
                        pos_color = (0.6, 0.6, 0.6, 0.9)
                else:
                    gray_value = 0.6
                    pos_alpha = 0.2
                    pos_color = (gray_value, gray_value, gray_value, pos_alpha)
            
            # Draw vector arrows
            ax2.annotate('', xy=info['pos_end'], xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=pos_color, lw=2,
                                      shrinkA=0, shrinkB=0, alpha=pos_color[3]))
            
            # Add feature labels ONLY for vectors on convex hull boundary
            if (pos_on_hull or neg_on_hull) and feat_idx < len(col_labels):
                label = col_labels[feat_idx][:8]  # Truncate long labels
                # Label near positive endpoint for consistency
                ax2.text(info['pos_end'][0] * 1.1, info['pos_end'][1] * 1.1, label,
                        fontsize=8, ha='center', va='center', alpha=0.8)


def setup_figure_layout():
    """
    Setup the main figure layout with professional styling.
    
    Returns:
        tuple: (fig, ax, ax2) - Figure and axes objects
    """
    # Create figure with enlarged wind vane and space for controls and legends
    fig = plt.figure(figsize=(18, 10))
    
    # Apply professional background
    from .color_system import BACKGROUND_COLOR
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
    from .color_system import BACKGROUND_COLOR, GRID_COLOR, TEXT_COLOR
    
    # Set figure background
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Style main plot
    ax1.set_facecolor(BACKGROUND_COLOR)
    ax1.grid(bool(getattr(config, 'SHOW_GRID_LINES', True)), alpha=0.3, linewidth=0.5, color=GRID_COLOR, zorder=0)
    
    # Remove tick marks but keep grid
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Remove spines (borders)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    
    # Apply unified title styling to both axes
    title_kwargs = dict(fontsize=14, color=TEXT_COLOR, pad=14, weight='bold')
    ax1.set_title("Wind Map", **title_kwargs)

    # Style wind vane - align with main plot background
    ax2.set_facecolor(BACKGROUND_COLOR)
    
    # Style wind vane spines
    for spine in ax2.spines.values():
        spine.set_color(TEXT_COLOR)
        spine.set_linewidth(0.5)
        
    # Wind vane title styling (identical to map)
    ax2.set_title("Wind Vane", **title_kwargs)


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
