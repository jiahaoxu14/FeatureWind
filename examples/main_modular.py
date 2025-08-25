#!/usr/bin/env python3
"""
Modular FeatureWind Visualization Script

This is the main entry point for the modular FeatureWind visualization system.
It orchestrates all the components: data processing, grid computation, particle system,
visualization, and UI controls.

Usage:
    python main_modular.py
    python main_modular.py --feature "mean radius"
    python main_modular.py --feature "worst perimeter"

This script provides the same functionality as the original main.py but with
improved modularity, maintainability, and extensibility.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
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


def parse_arguments():
    """Parse command-line arguments for visualization modes."""
    parser = argparse.ArgumentParser(
        description='FeatureWind - Visualize feature flow fields',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_modular.py                      # Default: top 5 features
  python main_modular.py --top-k 10           # Top 10 features
  python main_modular.py --top-k all          # All features
  python main_modular.py --feature "mean radius"  # Single feature mode
  python main_modular.py --list-features      # List available features
        """
    )
    
    # Create mutually exclusive group for feature vs top-k mode
    mode_group = parser.add_mutually_exclusive_group()
    
    mode_group.add_argument(
        '--feature', '-f',
        type=str,
        default=None,
        help='Visualize a single feature by name (partial match supported)'
    )
    
    mode_group.add_argument(
        '--top-k', '-k',
        type=str,
        default='5',
        help='Number of top features to visualize (integer or "all", default: 5)'
    )
    
    parser.add_argument(
        '--list-features', '-l',
        action='store_true',
        help='List all available features and exit'
    )
    
    parser.add_argument(
        '--tangent-map', '-t',
        type=str,
        default=None,
        help='Path to tangent map file (default: breast_cancer.tmap)'
    )
    
    return parser.parse_args()


def main():
    """Main function orchestrating the FeatureWind visualization."""
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Initialize configuration
    config.initialize_global_state()
    
    # Setup paths relative to repository root
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    
    # Determine tangent map path
    if args.tangent_map:
        tangent_map_path = args.tangent_map
    else:
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
    
    # Handle --list-features option
    if args.list_features:
        print("\nAvailable features:")
        print("-" * 40)
        for i, feature in enumerate(col_labels):
            print(f"{i:3d}: {feature}")
        print("-" * 40)
        print(f"Total: {len(col_labels)} features")
        sys.exit(0)
    
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
    
    # Parse top-k value from arguments
    if args.feature is None:  # Top-K mode
        if args.top_k.lower() == 'all':
            config.k = len(col_labels)
            print(f"Visualizing all {len(col_labels)} features")
        else:
            try:
                config.k = int(args.top_k)
                if config.k < 1:
                    config.k = 1
                elif config.k > len(col_labels):
                    config.k = len(col_labels)
                    print(f"Note: Only {len(col_labels)} features available, using all")
                else:
                    print(f"Visualizing top {config.k} features")
            except ValueError:
                print(f"Warning: Invalid top-k value '{args.top_k}', using default of 5")
                config.k = 5
    else:
        config.k = 1  # Single feature mode
    
    # Compute and set the bounding box
    config.set_bounding_box(all_positions)
    
    
    # Debug: Verify coordinate system alignment
    
    # Step 2: Feature selection
    single_feature_mode = args.feature is not None
    single_feature_name = args.feature
    
    if single_feature_mode:
        # Single feature mode
        feature_idx = data_processing.find_feature_by_name(col_labels, single_feature_name)
        if feature_idx == -1:
            print(f"\nError: Feature '{single_feature_name}' not found!")
            print("\nDid you mean one of these?")
            # Show partial matches
            matches = [f for f in col_labels if single_feature_name.lower() in f.lower()]
            for match in matches[:5]:
                print(f"  - {match}")
            print("\nUse --list-features to see all available features.")
            sys.exit(1)
        
        grad_indices = [feature_idx]
        avg_magnitudes = np.linalg.norm(all_grad_vectors[:, feature_idx, :], axis=1).mean()
        print(f"\nSingle Feature Mode: {col_labels[feature_idx]}")
        
        # Update window title
        config.WINDOW_TITLE = f"FeatureWind - Single Feature: {col_labels[feature_idx]}"
    else:
        # Top-K mode
        top_k_indices, avg_magnitudes = data_processing.pick_top_k_features(all_grad_vectors, config.k)
        grad_indices = top_k_indices
        if config.k == len(col_labels):
            config.WINDOW_TITLE = f"FeatureWind - All {len(col_labels)} Features"
        else:
            config.WINDOW_TITLE = f"FeatureWind - Top {len(grad_indices)} Features"
    
    # Step 3: Grid computation
    
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
    
    # Set window title
    fig.canvas.manager.set_window_title(config.WINDOW_TITLE)
    
    # Apply professional styling
    visualization_core.apply_professional_styling(fig, ax1, ax2)
    
    # Create comprehensive legend system
    import legend_manager
    legend_axes = legend_manager.create_comprehensive_legend(
        fig, family_assignments, col_labels, feature_colors, 
        legend_position='upper_left', show_instructions=False
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
    
    # Set single feature mode in UI controller
    ui_controller.single_feature_mode = single_feature_mode
    if single_feature_mode:
        ui_controller.single_feature_name = col_labels[grad_indices[0]]
        ui_controller.single_feature_idx = grad_indices[0]
    
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
    
    # Keyboard shortcuts handled by event manager
    # fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Show the interactive visualization
    
    plt.show()


if __name__ == "__main__":
    main()