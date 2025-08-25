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
    
    # Initialize configuration
    config.initialize_global_state()
    
    # Setup paths relative to repository root
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    tangent_map_path = os.path.join(repo_root, config.DEFAULT_TANGENT_MAP)
    if not os.path.exists(tangent_map_path):
        # Fallback to breast_cancer.tmap if the configured file doesn't exist
        tangent_map_path = os.path.join(repo_root, 'tangentmaps', 'breast_cancer.tmap')
    
    output_dir = os.path.join(repo_root, 'output')
    if not os.path.exists(output_dir):
        output_dir = config.DEFAULT_OUTPUT_DIR
        
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load and preprocess the tangent map data
    valid_points, all_grad_vectors, all_positions, col_labels = data_processing.preprocess_tangent_map(tangent_map_path)
    
    # Validate the loaded data
    data_processing.validate_data(valid_points, all_grad_vectors, all_positions, col_labels)
    
    # Analyze gradient magnitudes and auto-scale velocity if needed
    magnitudes = np.linalg.norm(all_grad_vectors, axis=2)
    avg_magnitude = magnitudes.mean()
    
    # Auto-scale velocity based on gradient magnitudes
    # Target average magnitude should be around 1.0 for stability
    target_avg_magnitude = 1.0
    if avg_magnitude > target_avg_magnitude * 2:  # If magnitudes are too large
        scale_factor = target_avg_magnitude / avg_magnitude
        config.velocity_scale = config.velocity_scale * scale_factor
    elif avg_magnitude < target_avg_magnitude * 0.1:  # If magnitudes are too small
        scale_factor = target_avg_magnitude / avg_magnitude
        config.velocity_scale = config.velocity_scale * scale_factor
    
    # Set global configuration values
    config.k = len(col_labels)  # Use all features initially
    # config.k = 5  # Uncomment to limit to top 5 features
    
    # Compute and set the bounding box
    config.set_bounding_box(all_positions)
    
    
    # Debug: Verify coordinate system alignment
    
    # Step 2: Feature selection
    top_k_indices, avg_magnitudes = data_processing.pick_top_k_features(all_grad_vectors, config.k)
    
    
    # Find specific features for debugging (e.g., "mean symmetry")
    mean_symmetry_idx = data_processing.find_feature_by_name(col_labels, "mean symmetry")
    # Feature lookup complete
    
    # Step 3: Grid computation
    grad_indices = top_k_indices
    
    # Set grid resolution
    grid_res = int((min(abs(config.bounding_box[1] - config.bounding_box[0]), 
                       abs(config.bounding_box[3] - config.bounding_box[2])) * config.grid_res_scale))
    grid_res = config.DEFAULT_GRID_RES  # Override with default
    
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
    
    # Step 4: Feature clustering and family-based color assignment
    
    # Import the new clustering and color modules
    import feature_clustering
    import color_system
    
    # Determine number of families: use actual count if ≤6, otherwise fix to 6
    n_features = len(col_labels)
    n_families = min(n_features, config.MAX_FEATURE_FAMILIES)
    
    # Perform feature clustering based on vector field directional similarity  
    family_assignments, similarity_matrix, clustering_metrics = feature_clustering.cluster_features_by_direction(
        grid_u_all_feats, grid_v_all_feats, n_families=n_families
    )
    
    # Assign Paul Tol colors to families
    feature_colors = color_system.assign_family_colors(family_assignments)
    
    
    # Analyze the feature families
    family_analysis = feature_clustering.analyze_feature_families(
        family_assignments, col_labels, similarity_matrix
    )
    
    # Build feature color mapping for backward compatibility
    config.real_feature_rgba = {}
    for i, feat_idx in enumerate(range(len(col_labels))):
        if feat_idx < len(feature_colors):
            color_hex = feature_colors[feat_idx]
            if color_hex.startswith('#'):
                r, g, b = color_system.hex_to_rgb(color_hex)
                config.real_feature_rgba[feat_idx] = (r, g, b, 1.0)
    
    # Step 5: Create particle system
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
    
    # Step 6: Setup figure with professional styling and legends
    fig, ax1, ax2 = visualization_core.setup_figure_layout()
    
    # Apply professional styling
    visualization_core.apply_professional_styling(fig, ax1, ax2)
    
    # Create comprehensive legend system
    import legend_manager
    legend_axes = legend_manager.create_comprehensive_legend(
        fig, family_assignments, col_labels, feature_colors, 
        legend_position='upper_left', show_instructions=True
    )
    
    
    # Prepare the main subplot
    lc = system['linecoll']
    visualization_core.prepare_figure(ax1, valid_points, col_labels, config.k, grad_indices, 
                                    feature_colors, lc, all_positions, all_grad_vectors, grid_res)
    
    # Prepare the wind vane subplot
    visualization_core.prepare_wind_vane_subplot(ax2)
    
    # Store family info in system for particle coloring and UI
    system['family_assignments'] = family_assignments
    system['feature_colors'] = feature_colors
    
    # Step 7: Setup UI controls
    ui_controller = ui_controls.UIController(fig, ax1, ax2, system, grid_data, col_labels)
    
    # Pass the all_grad_vectors for proper feature selection in UI
    ui_controller.all_grad_vectors = all_grad_vectors
    
    # Setup reliable mouse interactions using the enhanced event system
    import event_manager
    
    # Create reliable event handling system with family colors
    event_mgr = event_manager.create_reliable_event_system(
        fig, ax1, ax2, ui_controller, system, col_labels, grad_indices, feature_colors, grid_res
    )
    
    # Store additional family info in event manager for wind vane updates
    if hasattr(event_mgr, 'set_family_info'):
        event_mgr.set_family_info(family_assignments, feature_colors)
    
    # Step 8: Animation setup with enhanced performance monitoring
    
    animation_frame_count = 0
    
    def update_frame(frame):
        """Enhanced animation update function with family-based coloring."""
        nonlocal animation_frame_count
        animation_frame_count += 1
        
        try:
            # Update particles with family-based coloring
            if hasattr(particle_system, 'update_particles_with_families'):
                # Use enhanced particle update with family coloring
                result = particle_system.update_particles_with_families(
                    system, interp_u_sum, interp_v_sum, grid_u_sum, grid_v_sum, 
                    grid_res, family_assignments, feature_colors)
            else:
                # Fallback to standard particle update
                result = particle_system.update_particles(
                    system, interp_u_sum, interp_v_sum, grid_u_sum, grid_v_sum, grid_res)
            
            # Periodic performance check (every 150 frames ~ 5 seconds at 30 FPS)
            if animation_frame_count % 150 == 0:
                event_mgr.get_performance_stats()
                
                # Check for interaction failures and suggest solutions
                if hasattr(event_mgr, 'failed_updates') and event_mgr.failed_updates > 10:
                    pass  # Silently handle high failure rate
                    
                # Print family clustering info occasionally
                if animation_frame_count % 450 == 0:  # Every ~15 seconds
                    pass  # Silently track feature families
            
            return result
            
        except Exception as e:
            pass  # Silently handle animation frame error
            return []
    
    # Create the animation
    anim = FuncAnimation(fig, update_frame, frames=config.ANIMATION_FRAMES, 
                        interval=config.ANIMATION_INTERVAL, blit=False)
    
    # Save the figure as a PNG file
    visualization_core.save_final_figure(fig, output_dir, "featurewind_modular_figure.png")
    
    # Add keyboard shortcuts for troubleshooting
    def on_key_press(event):
        """Handle keyboard shortcuts for troubleshooting."""
        if event.key == 'f5' or event.key == 'r':
            pass  # Refresh visualization
            event_mgr.force_refresh()
        elif event.key == 'p':
            pass  # Show performance stats
        elif event.key == 'h':
            pass  # Show help
    
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Show the interactive visualization
    
    plt.show()


if __name__ == "__main__":
    main()