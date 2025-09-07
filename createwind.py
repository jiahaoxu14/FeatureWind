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
import re
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

    # Optional: filter features by name using a regex (e.g., '3$' to keep columns ending with 3)
    parser.add_argument(
        '--name-filter', '-F',
        type=str,
        default=None,
        help="Regex to filter feature names before selection (e.g., '3$' for 'worst' metrics)"
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
    
    # Analyze gradient magnitudes (no adaptive velocity scaling applied)
    magnitudes = np.linalg.norm(all_grad_vectors, axis=2)
    avg_magnitude = magnitudes.mean()
    
    # Enforce CLI-only mode selection
    if not args.feature and not args.top_k and not args.list_features:
        print("Error: specify either --feature NAME or --top-k N|all (or use --list-features).")
        sys.exit(2)

    # Optional: filter feature set by name regex
    filtered_indices = list(range(len(col_labels)))
    if args.name_filter:
        try:
            pattern = re.compile(args.name_filter)
            filtered_indices = [i for i, name in enumerate(col_labels) if pattern.search(name)]
            if len(filtered_indices) == 0:
                print(f"Error: name-filter '{args.name_filter}' matched 0 features.")
                sys.exit(2)
            print(f"Applied name-filter '{args.name_filter}': kept {len(filtered_indices)}/{len(col_labels)} features.")
        except re.error as e:
            print(f"Error: invalid regex for --name-filter: {e}")
            sys.exit(2)

    # Parse top-k value from arguments
    if args.feature is None:  # Top-K mode
        if args.top_k is None:
            print("Error: --top-k is required when --feature is not provided.")
            sys.exit(2)
        if isinstance(args.top_k, str) and args.top_k.lower() == 'all':
            # Use all filtered features
            config.k = len(filtered_indices)
            grad_indices = filtered_indices
            print(f"Visualizing all {len(grad_indices)} filtered features")
        else:
            try:
                config.k = int(args.top_k)
                if config.k < 1:
                    config.k = 1
                # Compute top-k within the filtered set
                feature_magnitudes = np.linalg.norm(all_grad_vectors, axis=2).mean(axis=0)
                # Sort within filtered indices by average magnitude (descending)
                filtered_with_scores = [(i, feature_magnitudes[i]) for i in filtered_indices]
                filtered_with_scores.sort(key=lambda x: -x[1])
                grad_indices = [i for i, _ in filtered_with_scores[:config.k]]
                # Adjust k in case fewer filtered features exist
                config.k = len(grad_indices)
                if config.k < int(args.top_k):
                    print(f"Note: Only {config.k} filtered features available, using all of them")
                print(f"Visualizing top {config.k} filtered features")
            except ValueError:
                print(f"Error: Invalid --top-k value '{args.top_k}'. Use an integer or 'all'.")
                sys.exit(2)
    else:
        config.k = 1  # Single feature mode
    
    # Compute and set the bounding box
    config.set_bounding_box(all_positions)
    
    
    # Debug: Verify coordinate system alignment
    
    # Adaptive velocity scaling removed
    
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
        # Top-K mode already computed grad_indices above
        if args.top_k and isinstance(args.top_k, str) and args.top_k.lower() == 'all':
            config.WINDOW_TITLE = f"FeatureWind - All {len(grad_indices)} Filtered Features"
        else:
            config.WINDOW_TITLE = f"FeatureWind - Top {len(grad_indices)} Filtered Features"
    
    # Adaptive velocity scaling removed: keep config.velocity_scale fixed
    
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
     grid_u_all_feats, grid_v_all_feats, cell_centers_x, cell_centers_y) = grid_data

    # Create the combined (summed) velocity field for the top-k features
    grid_u_sum = np.sum(grid_u_feats, axis=0)  # shape: (grid_res, grid_res)
    grid_v_sum = np.sum(grid_v_feats, axis=0)  # shape: (grid_res, grid_res)
    
    # Report dominant cell counts per feature, ignoring masked cells (unified strategy)
    # Keep counts available for later family-level aggregation diagnostics
    counts = None
    total_unmasked = None
    try:
        xmin, xmax, ymin, ymax = config.bounding_box
        dx = (xmax - xmin) / grid_res
        dy = (ymax - ymin) / grid_res
        # Threshold-based unmasking using summed magnitude
        sum_magnitudes = np.sqrt(grid_u_sum**2 + grid_v_sum**2)
        threshold = 1e-6
        unmasked = sum_magnitudes > threshold
        # Never mask cells that contain data points
        if valid_points:
            cells_with_data = np.zeros((grid_res, grid_res), dtype=bool)
            for point in valid_points:
                x, y = point.position
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    j = int((x - xmin) / dx)
                    i = int((y - ymin) / dy)
                    i = max(0, min(i, grid_res - 1))
                    j = max(0, min(j, grid_res - 1))
                    cells_with_data[i, j] = True
            unmasked |= cells_with_data

        # In single-feature mode, restrict dominance and particle coloring to the selected feature only
        if single_feature_mode and isinstance(grad_indices, (list, tuple, np.ndarray)) and len(grad_indices) == 1:
            only_idx = int(grad_indices[0])
            cdf_override = np.full_like(cell_dominant_features, -1)
            cdf_override[unmasked] = only_idx
            cell_dominant_features = cdf_override
        # Count only unmasked cells
        masked_filter = unmasked.ravel()
        filtered = cell_dominant_features.ravel()[masked_filter]
        counts = np.bincount(filtered, minlength=len(col_labels))
        total_unmasked = int(np.count_nonzero(unmasked))
        print("\nDominant cell counts per feature (out of", total_unmasked, "unmasked cells):")
        for idx, count in enumerate(counts):
            print(f"  {idx:3d} | {col_labels[idx]}: {int(count)}")
    except Exception:
        pass
    
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

    # Color assignment
    if bool(getattr(config, 'USE_PER_FEATURE_COLORS', False)):
        # Ignore families; assign distinct colors per feature using the Glasbey palette
        palette = list(getattr(config, 'GLASBEY_COLORS', []))
        if not palette:
            palette = ["#1f77b4"]
        feature_colors = [palette[i % len(palette)] for i in range(len(col_labels))]
        # For legend consistency, treat each feature as its own family
        try:
            import numpy as _np
            family_assignments = _np.arange(len(col_labels))
        except Exception:
            pass
        print(f"\nUsing per-feature distinct colors for all {len(col_labels)} features (no families).")
    else:
        # Assign Paul Tol colors to families
        feature_colors = color_system.assign_family_colors(family_assignments)

    # Diagnostics: show selected features -> family -> color mapping
    try:
        print("\nSelected features (index | name -> family, color):")
        for sel_idx in (grad_indices if isinstance(grad_indices, (list, tuple, np.ndarray)) else [grad_indices]):
            fam = int(family_assignments[sel_idx]) if family_assignments is not None else -1
            col = feature_colors[sel_idx] if feature_colors and sel_idx < len(feature_colors) else "#000000"
            print(f"  {sel_idx:3d} | {col_labels[sel_idx]} -> family {fam}, color {col}")
    except Exception:
        pass

    # If only a few features are selected, switch to distinct per-feature colors to avoid
    # multiple features sharing the same family hue (which can be visually misleading).
    if not bool(getattr(config, 'USE_PER_FEATURE_COLORS', False)):
        try:
            n_selected = len(grad_indices)
            if (config.COLOR_BY_FEATURE_WHEN_FEW and 
                isinstance(grad_indices, (list, tuple, np.ndarray)) and 
                n_selected <= config.FEATURE_COLOR_DISTINCT_THRESHOLD):
                palette = config.GLASBEY_COLORS
                for rank, feat_idx in enumerate(grad_indices):
                    if rank < len(palette) and feat_idx < len(feature_colors):
                        feature_colors[feat_idx] = palette[rank]
                print(f"\nApplied distinct per-feature colors for {n_selected} selected features (threshold {config.FEATURE_COLOR_DISTINCT_THRESHOLD}).")
                for rank, feat_idx in enumerate(grad_indices):
                    print(f"  {feat_idx:3d} | {col_labels[feat_idx]} -> color {feature_colors[feat_idx]}")
        except Exception:
            pass

    # In single-feature mode, enforce a consistent, user-configurable color
    try:
        if single_feature_mode and isinstance(grad_indices, (list, tuple, np.ndarray)) and len(grad_indices) == 1:
            sel_idx = int(grad_indices[0])
            if sel_idx < len(feature_colors):
                feature_colors[sel_idx] = getattr(config, 'SINGLE_FEATURE_COLOR', '#CCBB44')
    except Exception:
        pass

    # If we computed per-feature dominance counts earlier, also aggregate by family to compare with hues
    try:
        if counts is not None and family_assignments is not None and len(family_assignments) == len(col_labels):
            fam_counts = {}
            for idx, cnt in enumerate(counts):
                fam_id = int(family_assignments[idx])
                fam_counts[fam_id] = fam_counts.get(fam_id, 0) + int(cnt)
            print("\nDominant cell counts by family (for comparison with family hues):")
            for fam_id in sorted(fam_counts.keys()):
                # representative color: first feature in this family
                rep_idx = next((i for i, f in enumerate(family_assignments) if int(f) == fam_id), None)
                fam_color = feature_colors[rep_idx] if rep_idx is not None and rep_idx < len(feature_colors) else "#000000"
                print(f"  family {fam_id}: {fam_counts[fam_id]} cells, color {fam_color}")
    except Exception:
        pass
    
    
    # Analyze the feature families
    # Analysis info is not used in the current flow
    
    # Build feature color mapping for backward compatibility
    config.real_feature_rgba = {}
    for i, feat_idx in enumerate(range(len(col_labels))):
        if feat_idx < len(feature_colors):
            color_hex = feature_colors[feat_idx]
            if color_hex.startswith('#'):
                r, g, b = color_system.hex_to_rgb(color_hex)
                config.real_feature_rgba[feat_idx] = (r, g, b, 1.0)
    
    # Step 5: Create system dictionary; if in feature wind mode, add particles
    system = {
        'grid_u_sum': grid_u_sum,
        'grid_v_sum': grid_v_sum,
        'grid_u_all_feats': grid_u_all_feats,
        'grid_v_all_feats': grid_v_all_feats,
        'cell_dominant_features': cell_dominant_features,
        'interp_u_sum': interp_u_sum,
        'interp_v_sum': interp_v_sum,
        'interp_argmax': interp_argmax
    }
    if getattr(config, 'VIS_MODE', 'feature_wind_map') == 'feature_wind_map':
        system = particle_system.create_particles(
            config.DEFAULT_NUM_PARTICLES, cell_dominant_features, grid_res, 
            system)
    
    # Step 6: Setup figure with professional styling and legends
    fig, ax1, ax2 = visualization_core.setup_figure_layout()
    
    # Set window title
    fig.canvas.manager.set_window_title(config.WINDOW_TITLE)
    
    # Apply professional styling
    visualization_core.apply_professional_styling(fig, ax1, ax2)
    
    # Create legend: show only selected feature in single-feature mode
    from featurewind.visualization import legend_manager
    if single_feature_mode and isinstance(grad_indices, (list, tuple, np.ndarray)) and len(grad_indices) == 1:
        sel_idx = int(grad_indices[0])
        # Build filtered arrays so the legend lists only the selected feature
        import numpy as _np
        legend_axes = {}
        legend_axes['families'] = legend_manager.create_family_legend(
            fig,
            _np.array([family_assignments[sel_idx]]),
            [col_labels[sel_idx]],
            [feature_colors[sel_idx]],
            legend_position='upper_left'
        )
    else:
        legend_axes = legend_manager.create_comprehensive_legend(
            fig, family_assignments, col_labels, feature_colors,
            legend_position='upper_left', show_instructions=False
        )
    
    
    # Prepare the main subplot
    lc = system['linecoll'] if 'linecoll' in system else None
    visualization_core.prepare_figure(ax1, valid_points, col_labels, config.k, grad_indices, 
                                    feature_colors, lc, all_positions, all_grad_vectors, grid_res)

    # If in DimReader mode, draw filled isocontours
    # In single-feature mode, use only that feature's field for contours
    if getattr(config, 'VIS_MODE', 'feature_wind_map') == 'dimreader':
        if single_feature_mode and grid_u_feats is not None and len(grid_u_feats) == 1:
            magnitude = np.sqrt(grid_u_feats[0]**2 + grid_v_feats[0]**2)
        else:
            magnitude = np.sqrt(grid_u_sum**2 + grid_v_sum**2)
        try:
            cf = ax1.contourf(grid_x, grid_y, magnitude, levels=16, cmap='Greys', zorder=1)
            ax1.contour(grid_x, grid_y, magnitude, levels=16, colors='k', alpha=0.2, linewidths=0.4, zorder=2)
            # Optionally add a colorbar in this mode
            fig.colorbar(cf, ax=ax1, shrink=0.9, pad=0.02)
        except Exception:
            pass
    
    # Initial spawn markers removed
    
    # Highlight unmasked cells in the main plot (gray overlay)
    visualization_core.highlight_unmasked_cells(ax1, system, grid_res, valid_points)
    
    # Wind strength indicator omitted
    
    # Prepare the wind vane subplot(s); optionally add a second pane as "Feature Clock"
    ax2_for_events = ax2
    if getattr(config, 'FEATURE_CLOCK_ENABLED', False):
        try:
            # Split the existing ax2 area into two side-by-side panes
            pos = ax2.get_position()
            fig.delaxes(ax2)
            gap = 0.01 * pos.width
            left_w = (pos.width - gap) / 2.0
            right_w = left_w
            ax2_left = fig.add_axes([pos.x0, pos.y0, left_w, pos.height])
            ax2_right = fig.add_axes([pos.x0 + left_w + gap, pos.y0, right_w, pos.height])
            visualization_core.prepare_wind_vane_subplot(ax2_left)
            visualization_core.prepare_wind_vane_subplot(ax2_right)
            # Titles (optional)
            try:
                from featurewind import config as _cfg
                if bool(getattr(_cfg, 'SHOW_TITLES', True)):
                    ax2_left.set_title("Wind Vane", fontsize=12, weight='bold')
                    ax2_right.set_title("Feature Clock", fontsize=12, weight='bold')
                else:
                    ax2_left.set_title("")
                    ax2_right.set_title("")
            except Exception:
                pass
            ax2_for_events = (ax2_left, ax2_right)
        except Exception:
            visualization_core.prepare_wind_vane_subplot(ax2)
            ax2_for_events = ax2
    else:
        visualization_core.prepare_wind_vane_subplot(ax2)
        ax2_for_events = ax2
    
    # Store family info in system for particle coloring and UI
    system['family_assignments'] = family_assignments
    system['feature_colors'] = feature_colors
    system['grad_indices'] = grad_indices  # Store selected feature indices for color alignment
    
    # Step 7: Setup UI controls
    # Pass the effective vane axes (single or tuple) so UI can snapshot correctly
    ui_controller = ui_controls.UIController(fig, ax1, ax2_for_events, system, grid_data, col_labels)
    
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
        fig, ax1, ax2_for_events, ui_controller, system, col_labels, grad_indices, feature_colors, grid_res
    )
    # Allow UI to communicate back to event manager (for selection updates)
    try:
        ui_controller.event_manager = event_mgr
    except Exception:
        pass
    
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

        finally:
            # Auto snapshots: save every K frames when enabled
            try:
                if bool(getattr(config, 'AUTO_SNAPSHOT_ENABLED', False)):
                    kframes = int(getattr(config, 'AUTO_SNAPSHOT_EVERY_K_FRAMES', 0) or 0)
                    if kframes > 0 and (animation_frame_count % kframes == 0):
                        if ui_controller and hasattr(ui_controller, 'save_auto_snapshot'):
                            ui_controller.save_auto_snapshot(animation_frame_count)
            except Exception:
                pass
    
    # Create the animation
    anim = None
    if getattr(config, 'VIS_MODE', 'feature_wind_map') == 'feature_wind_map':
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
