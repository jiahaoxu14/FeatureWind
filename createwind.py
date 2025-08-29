#!/usr/bin/env python3
"""
Modular FeatureWind Visualization Script

This is the main entry point for the modular FeatureWind visualization system.
It orchestrates all the components: data processing, grid computation, particle system,
visualization, and UI controls.

Usage:
    python main_modular.py --tangent-map data/tangentmaps/breast_cancer.tmap --top-k 5
    python main_modular.py --tangent-map data/tangentmaps/breast_cancer.tmap --feature "mean radius"

This script provides the same functionality as the original main.py but with
improved modularity, maintainability, and extensibility.
"""

import sys
import os
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Suppress all runtime warnings from scipy and numpy that don't affect functionality
warnings.simplefilter('ignore', RuntimeWarning)

# Import all our modular components from the new structure
from featurewind import config
from featurewind.preprocessing import data_processing
from featurewind.physics import grid_computation, particle_system
from featurewind.visualization import visualization_core
from featurewind.ui import ui_controls


def parse_arguments():
    """Parse command-line arguments for visualization modes (CLI only)."""
    parser = argparse.ArgumentParser(
        description='FeatureWind - Visualize feature flow fields',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_modular.py --tangent-map data/tangentmaps/breast_cancer.tmap --top-k 10
  python main_modular.py --tangent-map data/tangentmaps/breast_cancer.tmap --top-k all
  python main_modular.py --tangent-map data/tangentmaps/breast_cancer.tmap --feature "mean radius"
  python main_modular.py --tangent-map data/tangentmaps/breast_cancer.tmap --list-features
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
        default=None,
        help='Number of top features to visualize (integer or "all"). Required if --feature not provided.'
    )
    
    parser.add_argument(
        '--list-features', '-l',
        action='store_true',
        help='List all available features and exit'
    )
    
    parser.add_argument(
        '--tangent-map', '-t',
        type=str,
        required=True,
        help='Path to tangent map file (required)'
    )
    
    return parser.parse_args()


def main():
    """Main function orchestrating the FeatureWind visualization."""
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Initialize configuration
    config.initialize_global_state()
    
    # Setup paths relative to repository root (main_modular.py is now at project root)
    repo_root = os.path.dirname(__file__)
    
    # Determine tangent map path (CLI required; no legacy fallback)
    tangent_map_path = args.tangent_map
    if not os.path.isabs(tangent_map_path):
        tangent_map_path = os.path.join(repo_root, tangent_map_path)
    if not os.path.exists(tangent_map_path):
        print(f"Error: Tangent map not found at '{tangent_map_path}'. Provide a valid path with --tangent-map.")
        sys.exit(2)
    
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
    
    # Enforce CLI-only mode selection
    if not args.feature and not args.top_k and not args.list_features:
        print("Error: specify either --feature NAME or --top-k N|all (or use --list-features).")
        sys.exit(2)

    # Parse top-k value from arguments
    if args.feature is None:  # Top-K mode
        if args.top_k is None:
            print("Error: --top-k is required when --feature is not provided.")
            sys.exit(2)
        if isinstance(args.top_k, str) and args.top_k.lower() == 'all':
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
                print(f"Error: Invalid --top-k value '{args.top_k}'. Use an integer or 'all'.")
                sys.exit(2)
    else:
        config.k = 1  # Single feature mode
    
    # Compute and set the bounding box
    config.set_bounding_box(all_positions)
    
    
    # Debug: Verify coordinate system alignment
    
    # Helper function for adaptive velocity scaling
    def compute_adaptive_velocity_scale(grad_indices, all_grad_vectors):
        """Scale velocity based on ACTUAL gradient strength."""
        
        # Get gradients for selected features
        selected_grads = all_grad_vectors[:, grad_indices, :]
        
        # Compute the SUMMED gradient magnitude (what actually drives particles)
        summed_grads = np.sum(selected_grads, axis=1)  # Sum across features
        avg_flow_magnitude = np.linalg.norm(summed_grads, axis=1).mean()
        
        # Define minimum flow magnitude for good visibility
        min_visible_magnitude = 4.0  # Minimum for visible trails
        
        # Compute scaling factor to ensure visibility
        if avg_flow_magnitude < 1e-6:
            # Nearly zero flow - need maximum boost
            scale_factor = 10.0
        elif avg_flow_magnitude < min_visible_magnitude:
            # Boost weak flow to minimum visible level
            scale_factor = min_visible_magnitude / avg_flow_magnitude
            # Clamp to reasonable range
            scale_factor = np.clip(scale_factor, 1.0, 10.0)
        else:
            # Flow is already strong enough - no reduction needed
            scale_factor = 1.0
        
        # Apply to base velocity scale
        adaptive_scale = config.velocity_scale * scale_factor
        
        return adaptive_scale, avg_flow_magnitude, scale_factor
    
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
    
    # Apply adaptive velocity scaling
    adaptive_scale, actual_flow_magnitude, scale_factor = compute_adaptive_velocity_scale(grad_indices, all_grad_vectors)
    original_velocity_scale = config.velocity_scale
    config.velocity_scale = adaptive_scale
    
    # Store for UI display
    config.actual_flow_magnitude = actual_flow_magnitude
    config.velocity_scale_factor = scale_factor
    config.adaptive_velocity_enabled = True
    
    print(f"Actual flow magnitude: {actual_flow_magnitude:.4f}")
    print(f"Applying {scale_factor:.1f}× velocity boost for visibility")
    
    # Step 3: Grid computation
    
    # Set grid resolution
    grid_res = config.DEFAULT_GRID_RES
    
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
    from featurewind.analysis import feature_clustering
    from featurewind.visualization import color_system
    
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
    
    # Step 5: Create temporary system dict with velocity fields for smart particle initialization
    temp_system = {
        'grid_u_sum': grid_u_sum,
        'grid_v_sum': grid_v_sum,
        'grid_u_all_feats': grid_u_all_feats,
        'grid_v_all_feats': grid_v_all_feats,
        'cell_dominant_features': cell_dominant_features,
        'cell_soft_dominance': cell_soft_dominance,
        'interp_u_sum': interp_u_sum,
        'interp_v_sum': interp_v_sum,
        'interp_argmax': interp_argmax
    }
    
    # Create particle system with smart initialization in unmasked cells
    system = particle_system.create_particles(
        config.DEFAULT_NUM_PARTICLES, cell_dominant_features, grid_res, 
        temp_system, valid_points)
    
    # Step 6: Setup figure with professional styling and legends
    fig, ax1, ax2 = visualization_core.setup_figure_layout()
    
    # Set window title
    fig.canvas.manager.set_window_title(config.WINDOW_TITLE)
    
    # Apply professional styling
    visualization_core.apply_professional_styling(fig, ax1, ax2)
    
    # Create comprehensive legend system
    from featurewind.visualization import legend_manager
    legend_axes = legend_manager.create_comprehensive_legend(
        fig, family_assignments, col_labels, feature_colors, 
        legend_position='upper_left', show_instructions=False
    )
    
    
    # Prepare the main subplot
    lc = system['linecoll']
    visualization_core.prepare_figure(ax1, valid_points, col_labels, config.k, grad_indices, 
                                    feature_colors, lc, all_positions, all_grad_vectors, grid_res)
    
    # Highlight unmasked cells in the main plot (gray overlay)
    visualization_core.highlight_unmasked_cells(ax1, system, grid_res, valid_points)
    
    # Add wind strength indicator (hidden per user request)
    # import wind_strength_indicator
    # wind_indicator = wind_strength_indicator.WindStrengthIndicator(ax1)
    # if hasattr(config, 'adaptive_velocity_enabled') and config.adaptive_velocity_enabled:
    #     wind_indicator.update(config.actual_flow_magnitude, config.velocity_scale_factor)
    #     
    #     # Add static flow indicators for extremely weak wind
    #     if config.actual_flow_magnitude < 0.005:  # Very weak threshold
    #         wind_strength_indicator.add_static_flow_indicators(
    #             ax1, grid_u_sum, grid_v_sum, grid_x, grid_y, threshold=0.005
    #         )
    
    # Prepare the wind vane subplot
    visualization_core.prepare_wind_vane_subplot(ax2)
    
    # Store family info in system for particle coloring and UI
    system['family_assignments'] = family_assignments
    system['feature_colors'] = feature_colors
    system['grad_indices'] = grad_indices  # Store selected feature indices for color alignment
    
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
    from featurewind.ui import event_manager
    
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
                    grid_res, family_assignments, feature_colors, grad_indices)
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
