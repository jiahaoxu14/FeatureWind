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
    if not os.path.exists(tangent_map_path):
        # Fallback to breast_cancer.tmap if the configured file doesn't exist
        tangent_map_path = os.path.join(repo_root, 'tangentmaps', 'breast_cancer.tmap')
    
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
    
    print(f"Loaded {len(valid_points)} valid points with {len(col_labels)} features")
    print(f"Bounding box: {config.bounding_box}")
    
    # Debug: Verify coordinate system alignment
    print(f"Grid resolution: {config.DEFAULT_GRID_RES}x{config.DEFAULT_GRID_RES}")
    print(f"Data point range: X[{all_positions[:,0].min():.2f}, {all_positions[:,0].max():.2f}], Y[{all_positions[:,1].min():.2f}, {all_positions[:,1].max():.2f}]")
    
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
    
    # Setup reliable mouse interactions using the enhanced event system
    import event_manager
    
    # Create reliable event handling system (replaces conflicting event handlers)
    event_mgr = event_manager.create_reliable_event_system(
        fig, ax1, ax2, ui_controller, system, col_labels, grad_indices, feature_colors, grid_res
    )
    
    # Step 8: Animation setup with enhanced performance monitoring
    print("Setting up animation with performance monitoring...")
    
    animation_frame_count = 0
    
    def update_frame(frame):
        """Enhanced animation update function with performance monitoring."""
        nonlocal animation_frame_count
        animation_frame_count += 1
        
        try:
            # Update particles
            result = particle_system.update_particles(
                system, interp_u_sum, interp_v_sum, grid_u_sum, grid_v_sum, grid_res)
            
            # Periodic performance check (every 150 frames ~ 5 seconds at 30 FPS)
            if animation_frame_count % 150 == 0:
                event_mgr.get_performance_stats()
                
                # Check for interaction failures and suggest solutions
                if hasattr(event_mgr, 'failed_updates') and event_mgr.failed_updates > 10:
                    print("⚠ High interaction failure rate detected!")
                    print("Tips: 1) Move mouse slower, 2) Check system performance, 3) Try F5 to refresh")
            
            return result
            
        except Exception as e:
            print(f"Animation frame error: {e}")
            return []
    
    # Create the animation
    anim = FuncAnimation(fig, update_frame, frames=config.ANIMATION_FRAMES, 
                        interval=config.ANIMATION_INTERVAL, blit=False)
    
    # Save the figure as a PNG file
    visualization_core.save_final_figure(fig, output_dir, "featurewind_modular_figure.png")
    print(f"Saved final figure to {output_dir}")
    
    # Add keyboard shortcuts for troubleshooting
    def on_key_press(event):
        """Handle keyboard shortcuts for troubleshooting."""
        if event.key == 'f5' or event.key == 'r':
            print("🔄 Refreshing visualization...")
            event_mgr.force_refresh()
        elif event.key == 'p':
            print("📊 Performance stats:")
            stats = event_mgr.get_performance_stats()
            if stats:
                print(f"  Events/sec: {stats['events_per_second']:.1f}")
                print(f"  Failure rate: {stats['failure_rate_percent']:.1f}%")
            else:
                print("  No recent activity")
        elif event.key == 'h':
            print("\n🔧 FeatureWind Keyboard Shortcuts:")
            print("  F5 or R: Force refresh visualization")
            print("  P: Show performance statistics")
            print("  H: Show this help message")
    
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Show the interactive visualization
    print("Starting enhanced interactive visualization...")
    print("\n📊 Interface Controls:")
    print("- Use radio buttons to switch between Top-K and Direction-Conditioned modes")
    print("- In Top-K mode: Use the slider to select number of features")
    print("- In Direction-Conditioned mode: Click on grid cells and use angle/magnitude sliders")
    print("- Move mouse over the main plot to see feature vectors in the wind vane")
    print("\n🔧 Troubleshooting:")
    print("- If interactions stop working, press F5 or R to refresh")
    print("- Press P to check performance stats")
    print("- Press H for keyboard shortcuts help")
    print("- If problems persist, move mouse slower or check system performance")
    
    plt.show()


if __name__ == "__main__":
    main()