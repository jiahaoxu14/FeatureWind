#!/usr/bin/env python3
"""
Modular FeatureWind Visualization Script

This is the main entry point for the modular FeatureWind visualization system.
It orchestrates all the components: data processing, grid computation, particle system,
visualization, and UI controls.

Usage:
    python main_modular.py

This script provides the same functionality as the original main.py but with
improved modularity, maintainability, and extensibility.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import all our modular components
import config
import data_processing
import grid_computation
import particle_system
import visualization_core
import ui_controls


def main():
    """Main function orchestrating the FeatureWind visualization."""
    print("Starting modular FeatureWind visualization...")
    
    # Initialize configuration
    config.initialize_global_state()
    
    # Setup paths relative to repository root
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    tangent_map_path = os.path.join(repo_root, config.DEFAULT_TANGENT_MAP)
    print(f"Trying to load tangent map from: {tangent_map_path}")
    print(f"File exists: {os.path.exists(tangent_map_path)}")
    if not os.path.exists(tangent_map_path):
        # Fallback to breast_cancer.tmap if the configured file doesn't exist
        tangent_map_path = os.path.join(repo_root, 'tangentmaps', 'breast_cancer.tmap')
        print(f"Falling back to: {tangent_map_path}")
    
    print(f"Final tangent map path: {tangent_map_path}")
    
    output_dir = os.path.join(repo_root, 'output')
    if not os.path.exists(output_dir):
        output_dir = config.DEFAULT_OUTPUT_DIR
        
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load and preprocess the tangent map data
    print("Loading tangent map data...")
    valid_points, all_grad_vectors, all_positions, col_labels = data_processing.preprocess_tangent_map(tangent_map_path)
    
    # Validate the loaded data
    data_processing.validate_data(valid_points, all_grad_vectors, all_positions, col_labels)
    
    # Set global configuration values
    config.k = len(col_labels)  # Use all features initially
    # config.k = 5  # Uncomment to limit to top 5 features
    
    # Compute and set the bounding box
    config.set_bounding_box(all_positions)
    
    print(f"Loaded {len(valid_points)} valid points with {len(col_labels)} features")
    print(f"Bounding box: {config.bounding_box}")
    
    # Step 2: Feature selection
    print("Selecting top features...")
    top_k_indices, avg_magnitudes = data_processing.pick_top_k_features(all_grad_vectors, config.k)
    
    print("Top k feature indices:", top_k_indices)
    print("Their average magnitudes:", avg_magnitudes[top_k_indices])
    print("Their labels:", [col_labels[i] for i in top_k_indices])
    
    # Find specific features for debugging (e.g., "mean symmetry")
    mean_symmetry_idx = data_processing.find_feature_by_name(col_labels, "mean symmetry")
    if mean_symmetry_idx is not None:
        print(f"Found 'mean symmetry' at feature index {mean_symmetry_idx}: {col_labels[mean_symmetry_idx]}")
    else:
        print("Warning: 'mean symmetry' feature not found in labels")
    
    # Step 3: Grid computation
    print("Building velocity grids...")
    grad_indices = top_k_indices
    
    # Set grid resolution
    grid_res = int((min(abs(config.bounding_box[1] - config.bounding_box[0]), 
                       abs(config.bounding_box[3] - config.bounding_box[2])) * config.grid_res_scale))
    grid_res = config.DEFAULT_GRID_RES  # Override with default
    print("Grid resolution:", grid_res)
    
    # Build the main grids
    grid_data = grid_computation.build_grids(
        all_positions, grid_res, grad_indices, all_grad_vectors, col_labels, output_dir=output_dir
    )
    
    # Unpack grid data
    (interp_u_sum, interp_v_sum, interp_argmax, grid_x, grid_y, 
     grid_u_feats, grid_v_feats, cell_dominant_features, 
     grid_u_all_feats, grid_v_all_feats, cell_centers_x, cell_centers_y, 
     cell_soft_dominance) = grid_data
    
    # Create the combined (summed) velocity field for the top-k features
    grid_u_sum = np.sum(grid_u_feats, axis=0)  # shape: (grid_res, grid_res)
    grid_v_sum = np.sum(grid_v_feats, axis=0)  # shape: (grid_res, grid_res)
    
    # Step 4: Setup colors and visualization elements
    print("Setting up colors and visualization...")
    
    # Generate colors for features using config
    feature_colors = [config.get_feature_color(i, len(col_labels)) for i in range(min(config.MAX_FEATURES_WITH_COLORS, len(col_labels)))]
    print("Feature colors:", feature_colors[:6])  # Show first 6
    
    # Build initial mapping for selected features to RGBA for particle coloring
    # Only top 6 features get colors, others are excluded from coloring
    config.real_feature_rgba = {}
    for i, feat_idx in enumerate(grad_indices):
        if i < config.MAX_FEATURES_WITH_COLORS and feat_idx < len(feature_colors):
            color_hex = feature_colors[i]
            # Convert hex to RGBA (assuming hex format)
            if color_hex.startswith('#'):
                r = int(color_hex[1:3], 16) / 255.0
                g = int(color_hex[3:5], 16) / 255.0
                b = int(color_hex[5:7], 16) / 255.0
                config.real_feature_rgba[feat_idx] = (r, g, b, 1.0)
    
    # Step 5: Create particle system
    print("Creating particle system...")
    system = particle_system.create_particles(
        config.DEFAULT_NUM_PARTICLES, cell_dominant_features, grid_res)
    
    # Store grid data in system for access by UI and particles
    system.update({
        'grid_u_sum': grid_u_sum,
        'grid_v_sum': grid_v_sum,
        'grid_u_all_feats': grid_u_all_feats,
        'grid_v_all_feats': grid_v_all_feats,
        'cell_dominant_features': cell_dominant_features,
        'cell_soft_dominance': cell_soft_dominance,
        'interp_u_sum': interp_u_sum,
        'interp_v_sum': interp_v_sum,
        'interp_argmax': interp_argmax
    })
    
    # Step 6: Setup figure and visualization
    print("Setting up figure and visualization...")
    fig, ax1, ax2 = visualization_core.setup_figure_layout()
    
    # Prepare the main subplot
    lc = system['linecoll']
    visualization_core.prepare_figure(ax1, valid_points, col_labels, config.k, grad_indices, 
                                    feature_colors, lc, all_positions, all_grad_vectors, grid_res)
    
    # Prepare the wind vane subplot
    visualization_core.prepare_wind_vane_subplot(ax2)
    
    # Step 7: Setup UI controls
    print("Setting up UI controls...")
    ui_controller = ui_controls.UIController(fig, ax1, ax2, system, grid_data, col_labels)
    
    # Pass the all_grad_vectors for proper feature selection in UI
    ui_controller.all_grad_vectors = all_grad_vectors
    
    # Setup mouse interactions
    ui_controls.setup_mouse_interactions(ui_controller)
    
    # Store mouse interaction data for wind vane updates
    mouse_data = {'grid_cell': None, 'grid_res': grid_res}
    
    # Mouse position tracking for grid cell analysis
    def on_mouse_move(event):
        """Handle mouse movement for wind vane updates."""
        if event.inaxes == ax1 and event.xdata is not None and event.ydata is not None:
            xmin, xmax, ymin, ymax = config.bounding_box
            
            if xmin <= event.xdata <= xmax and ymin <= event.ydata <= ymax:
                cell_j = int((event.xdata - xmin) / (xmax - xmin) * grid_res)
                cell_i = int((event.ydata - ymin) / (ymax - ymin) * grid_res)
                
                # Clamp to grid bounds
                cell_i = max(0, min(grid_res - 1, cell_i))
                cell_j = max(0, min(grid_res - 1, cell_j))
                
                mouse_data['grid_cell'] = (cell_i, cell_j)
                
                # Update wind vane
                visualization_core.update_wind_vane(ax2, mouse_data, system, col_labels, 
                                                  grad_indices, feature_colors)
                fig.canvas.draw_idle()
    
    # Connect mouse movement handler
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    
    # Step 8: Animation setup
    print("Setting up animation...")
    
    def update_frame(frame):
        """Animation update function."""
        return particle_system.update_particles(
            system, interp_u_sum, interp_v_sum, grid_u_sum, grid_v_sum, grid_res)
    
    # Create the animation
    anim = FuncAnimation(fig, update_frame, frames=config.ANIMATION_FRAMES, 
                        interval=config.ANIMATION_INTERVAL, blit=False)
    
    # Save the figure as a PNG file
    visualization_core.save_final_figure(fig, output_dir, "featurewind_modular_figure.png")
    print(f"Saved final figure to {output_dir}")
    
    # Show the interactive visualization
    print("Starting interactive visualization...")
    print("- Use radio buttons to switch between Top-K and Direction-Conditioned modes")
    print("- In Top-K mode: Use the slider to select number of features")
    print("- In Direction-Conditioned mode: Click on grid cells and use angle/magnitude sliders")
    print("- Move mouse over the main plot to see feature vectors in the wind vane")
    
    plt.show()


if __name__ == "__main__":
    main()