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
                  all_positions=None, all_grad_vectors=None, grid_res=None, system=None):
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

    # Optional: color cells by dominant feature (uses precomputed argmax per cell)
    try:
        if bool(getattr(config, 'SHOW_CELL_DOMINANCE', False)) and isinstance(system, dict):
            cdf = system.get('cell_dominant_features', None)
            cx = system.get('cell_centers_x', None)
            cy = system.get('cell_centers_y', None)
            if cdf is not None and cx is not None and cy is not None:
                H, W = cdf.shape
                from .color_system import hex_to_rgb
                rgba = np.zeros((H, W, 4), dtype=float)
                alpha = float(getattr(config, 'CELL_DOM_ALPHA', 0.18))
                # Respect explicit unmasked cells if present
                unmasked = system.get('unmasked_cells', None)
                for i in range(H):
                    for j in range(W):
                        if unmasked is not None and (not bool(unmasked[i, j])):
                            rgba[i, j, :] = (0.0, 0.0, 0.0, 0.0)
                            continue
                        fid = int(cdf[i, j])
                        if 0 <= fid < len(feature_colors):
                            color_val = feature_colors[fid]
                            if isinstance(color_val, str):
                                r, g, b = hex_to_rgb(color_val)
                            else:
                                r, g, b = color_val[:3]
                        else:
                            r, g, b = (0.6, 0.6, 0.6)
                        rgba[i, j, :3] = (r, g, b)
                        rgba[i, j, 3] = alpha
                # Draw with imshow (cell centers → extent bounds)
                extent = (xmin, xmax, ymin, ymax)
                ax.imshow(rgba, origin='lower', extent=extent, interpolation='nearest', zorder=int(getattr(config, 'CELL_DOM_ZORDER', 3)))
    except Exception:
        pass

    # Plot underlying data points with different markers for each label (optional)
    if bool(getattr(config, 'SHOW_DATA_POINTS', True)):
        feature_idx = 2  # Use feature index 2 for alpha computation
        try:
            domain_values = np.array([p.domain[feature_idx] for p in valid_points])
            domain_min, domain_max = domain_values.min(), domain_values.max()
        except Exception:
            domain_min, domain_max = 0.0, 1.0

        # Collect all labels from valid_points
        unique_labels = sorted(set(p.tmap_label for p in valid_points))

        # Define multiple distinct markers
        markers = config.MARKER_STYLES

        for i, lab in enumerate(unique_labels):
            indices = [j for j, p in enumerate(valid_points) if p.tmap_label == lab]
            positions_lab = np.array([valid_points[j].position for j in indices])
            try:
                normalized = (np.array([valid_points[j].domain[feature_idx] for j in indices]) - domain_min) / (domain_max - domain_min + 1e-9)
            except Exception:
                normalized = np.zeros(len(indices))
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
        show_particles = bool(getattr(config, 'SHOW_PARTICLES', True)) and not bool(getattr(config, 'SHOW_INITIAL_SPAWNS', False))
        if (getattr(config, 'VIS_MODE', 'feature_wind_map') == 'feature_wind_map'
            and lc is not None
            and show_particles
            and not getattr(config, 'FEATURE_CLOCK_ENABLED', False)):
            ax.add_collection(lc)
        else:
            # Ensure trails are hidden when not showing particles or when Feature Clock is enabled
            if lc is not None:
                try:
                    lc.set_visible(False)
                except Exception:
                    pass
    except Exception:
        pass
    
    # Optionally draw initial particle spawn positions as solid small circles
    try:
        if isinstance(system, dict) and bool(getattr(config, 'SHOW_INITIAL_SPAWNS', False)):
            spawns = system.get('initial_spawn_positions', None)
            if spawns is not None and len(spawns) > 0:
                # Filter spawns to only unmasked cells when mask is available
                try:
                    unmasked_cells = system.get('unmasked_cells', None)
                    if unmasked_cells is not None:
                        grid_res_local = unmasked_cells.shape[0]
                        xmin, xmax, ymin, ymax = config.bounding_box
                        import numpy as _np
                        j_idx = _np.clip(((spawns[:, 0] - xmin) / (xmax - xmin) * grid_res_local).astype(int), 0, grid_res_local - 1)
                        i_idx = _np.clip(((spawns[:, 1] - ymin) / (ymax - ymin) * grid_res_local).astype(int), 0, grid_res_local - 1)
                        keep = _np.array([bool(unmasked_cells[i_idx[t], j_idx[t]]) for t in range(len(spawns))])
                        spawns_draw = spawns[keep]
                        i_idx = i_idx[keep]
                        j_idx = j_idx[keep]
                    else:
                        spawns_draw = spawns
                        i_idx = j_idx = None
                except Exception:
                    spawns_draw = spawns
                    i_idx = j_idx = None

                if len(spawns_draw) > 0:
                    ax.scatter(
                        spawns_draw[:, 0], spawns_draw[:, 1],
                        s=getattr(config, 'INITIAL_SPAWN_SIZE', 10),
                        c=getattr(config, 'INITIAL_SPAWN_COLOR', '#222222'),
                        alpha=float(getattr(config, 'INITIAL_SPAWN_ALPHA', 0.6)),
                        zorder=int(getattr(config, 'INITIAL_SPAWN_ZORDER', 26)),
                        marker='o', edgecolors='none'
                    )
                # Add short vectors indicating initial flow direction, colored by cell dominance
                if bool(getattr(config, 'SHOW_INITIAL_SPAWN_VECTORS', True)):
                    try:
                        U_sum = system.get('grid_u_sum', None)
                        V_sum = system.get('grid_v_sum', None)
                        cdf = system.get('cell_dominant_features', None)
                        xmin, xmax, ymin, ymax = config.bounding_box
                        grid_res_local = U_sum.shape[0] if U_sum is not None else int(getattr(config, 'DEFAULT_GRID_RES', 20))
                        # Compute raw vectors at spawn cells
                        import numpy as _np
                        if i_idx is None or j_idx is None:
                            j_idx = _np.clip(((spawns[:, 0] - xmin) / (xmax - xmin) * grid_res_local).astype(int), 0, grid_res_local - 1)
                            i_idx = _np.clip(((spawns[:, 1] - ymin) / (ymax - ymin) * grid_res_local).astype(int), 0, grid_res_local - 1)
                            spawns_draw = spawns
                        if U_sum is not None and V_sum is not None:
                            U0 = U_sum[i_idx, j_idx]
                            V0 = V_sum[i_idx, j_idx]
                        else:
                            U0 = _np.zeros(len(spawns))
                            V0 = _np.zeros(len(spawns))
                        # Scale vectors
                        rel = float(getattr(config, 'INITIAL_SPAWN_VEC_REL_SCALE', 0.03))
                        abs_scale = float(getattr(config, 'INITIAL_SPAWN_VEC_ABS_SCALE', 0.0))
                        plot_width = float(xmax - xmin)
                        mags = _np.sqrt(U0**2 + V0**2)
                        eps = 1e-12
                        if abs_scale > 0:
                            scale = abs_scale
                        else:
                            p95 = _np.percentile(mags, 95) if mags.size > 0 else 1.0
                            target = rel * plot_width
                            scale = target / (p95 + eps)
                        U = U0 * scale
                        V = V0 * scale
                        # Colors by dominant feature id
                        colors = []
                        for ii, jj in zip(i_idx, j_idx):
                            try:
                                fid = int(cdf[ii, jj]) if cdf is not None else -1
                                if 0 <= fid < len(feature_colors):
                                    colors.append(feature_colors[fid])
                                else:
                                    colors.append('#333333')
                            except Exception:
                                colors.append('#333333')
                        ax.quiver(spawns_draw[:, 0], spawns_draw[:, 1], U, V,
                                  angles='xy', scale_units='xy', scale=1.0,
                                  color=colors, alpha=float(getattr(config, 'INITIAL_SPAWN_VEC_ALPHA', 0.9)),
                                  zorder=int(getattr(config, 'INITIAL_SPAWN_VEC_ZORDER', 27)),
                                  width=float(getattr(config, 'INITIAL_SPAWN_VEC_WIDTH', 0.003)))
                    except Exception:
                        pass
    except Exception:
        pass
    
    # Grid lines are manually drawn above, so no need for ax.grid()
    ax.set_axisbelow(True)  # Put grid behind other elements

    # Optional: overlay per-feature gradient vectors at data points (paper figures)
    try:
        from .. import config as _cfg
        if bool(getattr(_cfg, 'SHOW_DATA_VECTORS_ON_MAP', False)) and \
           all_positions is not None and all_grad_vectors is not None and \
           isinstance(grad_indices, (list, tuple, np.ndarray)) and len(grad_indices) > 0:
            xmin, xmax, ymin, ymax = _cfg.bounding_box
            plot_width = float(xmax - xmin)
            # Build a robust scale
            rel = float(getattr(_cfg, 'DATA_VECTOR_REL_SCALE', 0.03))
            abs_scale = float(getattr(_cfg, 'DATA_VECTOR_ABS_SCALE', 0.0))
            alpha = float(getattr(_cfg, 'DATA_VECTOR_ALPHA', 0.55))
            z = int(getattr(_cfg, 'DATA_VECTOR_ZORDER', 25))
            # For each selected feature, draw its vectors
            import numpy as _np
            P = _np.asarray(all_positions)
            # Optional mask: prefer explicit unmasked_cells from system
            respect_mask = bool(getattr(_cfg, 'SHOW_VECTOR_OVERLAY_RESPECT_MASK', False))
            cell_mask = None
            if respect_mask and isinstance(system, dict):
                try:
                    if system.get('unmasked_cells') is not None:
                        cell_mask = _np.asarray(system.get('unmasked_cells'))
                    else:
                        U_sum = system.get('grid_u_sum', None)
                        V_sum = system.get('grid_v_sum', None)
                        if U_sum is not None and V_sum is not None:
                            mags = _np.sqrt(U_sum**2 + V_sum**2)
                            thr = float(getattr(_cfg, 'MASK_THRESHOLD', 1e-6))
                            cell_mask = (mags > thr)
                except Exception:
                    cell_mask = None
            # Feature subset resolution: allow a single configured feature override
            cfg_feat = getattr(_cfg, 'DATA_VECTOR_FEATURE', None)
            feat_indices = list(grad_indices)
            try:
                if isinstance(cfg_feat, int):
                    feat_indices = [int(cfg_feat)]
                elif isinstance(cfg_feat, (list, tuple, np.ndarray)):
                    feat_indices = [int(i) for i in cfg_feat]
                elif isinstance(cfg_feat, str) and len(cfg_feat) > 0:
                    # If comma-separated ints, parse; else do substring match
                    if ',' in cfg_feat:
                        parts = [p.strip() for p in cfg_feat.split(',') if p.strip()]
                        parsed = []
                        for p in parts:
                            try:
                                parsed.append(int(p))
                            except Exception:
                                pass
                        if parsed:
                            feat_indices = parsed
                        else:
                            low = cfg_feat.lower()
                            matches = [i for i, name in enumerate(col_labels) if low in str(name).lower()]
                            if matches:
                                feat_indices = [matches[0]]
                    else:
                        low = cfg_feat.lower()
                        matches = [i for i, name in enumerate(col_labels) if low in str(name).lower()]
                        if matches:
                            feat_indices = [matches[0]]
            except Exception:
                pass

            for feat_idx in feat_indices:
                if feat_idx < 0 or feat_idx >= all_grad_vectors.shape[1]:
                    continue
                V = _np.asarray(all_grad_vectors)[:, int(feat_idx), :]  # shape (N,2)
                # Auto scale: target 95th pct length to rel*plot_width
                norms = _np.linalg.norm(V, axis=1)
                eps = 1e-12
                if abs_scale > 0:
                    scale = abs_scale
                else:
                    p95 = _np.percentile(norms, 95) if norms.size > 0 else 1.0
                    target = rel * plot_width
                    scale = target / (p95 + eps)
                U = V[:, 0] * scale
                W = V[:, 1] * scale
                color = feature_colors[feat_idx] if feat_idx < len(feature_colors) else '#333333'
                # Optional masking by cell
                if respect_mask and cell_mask is not None and grid_res is not None:
                    try:
                        xmin, xmax, ymin, ymax = _cfg.bounding_box
                        j_idx = _np.clip(((P[:, 0] - xmin) / (xmax - xmin) * grid_res).astype(int), 0, grid_res - 1)
                        i_idx = _np.clip(((P[:, 1] - ymin) / (ymax - ymin) * grid_res).astype(int), 0, grid_res - 1)
                        keep = _np.array([bool(cell_mask[i_idx[t], j_idx[t]]) for t in range(len(P))])
                        P_draw = P[keep]
                        U_draw = U[keep]
                        W_draw = W[keep]
                    except Exception:
                        P_draw, U_draw, W_draw = P, U, W
                else:
                    P_draw, U_draw, W_draw = P, U, W
                ax.quiver(P_draw[:, 0], P_draw[:, 1], U_draw, W_draw,
                          angles='xy', scale_units='xy', scale=1.0,
                          color=color, alpha=alpha, zorder=z, width=0.002)
    except Exception:
        pass

    # Optional: overlay grid (interpolated) vectors per cell
    try:
        from .. import config as _cfg
        if bool(getattr(_cfg, 'SHOW_GRID_VECTORS_ON_MAP', False)) and isinstance(system, dict):
            cx = system.get('cell_centers_x', None)
            cy = system.get('cell_centers_y', None)
            if cx is not None and cy is not None:
                GX, GY = np.meshgrid(cx, cy)
                # Resolve feature choice
                choice = getattr(_cfg, 'GRID_VECTOR_FEATURE', None)
                alpha = float(getattr(_cfg, 'GRID_VECTOR_ALPHA', 0.50))
                z = int(getattr(_cfg, 'GRID_VECTOR_ZORDER', 24))
                rel = float(getattr(_cfg, 'GRID_VECTOR_REL_SCALE', 0.05))
                abs_scale = float(getattr(_cfg, 'GRID_VECTOR_ABS_SCALE', 0.0))
                xmin, xmax, ymin, ymax = _cfg.bounding_box
                plot_width = float(xmax - xmin)
                # Build a single base mask: prefer explicit unmasked_cells
                base_mask = None
                if bool(getattr(_cfg, 'SHOW_VECTOR_OVERLAY_RESPECT_MASK', False)):
                    try:
                        if system.get('unmasked_cells') is not None:
                            base_mask = np.asarray(system.get('unmasked_cells'))
                        else:
                            U_sum = system.get('grid_u_sum', None)
                            V_sum = system.get('grid_v_sum', None)
                            if U_sum is not None and V_sum is not None:
                                sum_mags = np.sqrt(U_sum**2 + V_sum**2)
                                thr = float(getattr(_cfg, 'MASK_THRESHOLD', 1e-6))
                                base_mask = (sum_mags > thr)
                    except Exception:
                        base_mask = None
                # Determine which grid to draw
                U_grid = None
                V_grid = None
                color = '#333333'
                if isinstance(choice, str) and choice.strip().lower() == 'sum':
                    U_grid = system.get('grid_u_sum', None)
                    V_grid = system.get('grid_v_sum', None)
                    color = '#000000'
                else:
                    # Choose a feature index
                    feat_indices = list(grad_indices) if isinstance(grad_indices, (list, tuple, np.ndarray)) else []
                    try:
                        if isinstance(choice, int):
                            feat_indices = [int(choice)]
                        elif isinstance(choice, str) and len(choice) > 0:
                            low = choice.lower()
                            matches = [i for i, name in enumerate(col_labels) if low in str(name).lower()]
                            if matches:
                                feat_indices = [matches[0]]
                    except Exception:
                        pass
                    if feat_indices:
                        f = int(feat_indices[0])
                        U_all = system.get('grid_u_all_feats', None)
                        V_all = system.get('grid_v_all_feats', None)
                        if U_all is not None and V_all is not None and 0 <= f < U_all.shape[0]:
                            U_grid = U_all[f]
                            V_grid = V_all[f]
                            color = feature_colors[f] if f < len(feature_colors) else '#333333'
                # Draw if available; support multiple features
                if isinstance(choice, str) and choice.strip().lower() == 'sum':
                    if U_grid is not None and V_grid is not None:
                        mags = np.sqrt(U_grid**2 + V_grid**2)
                        eps = 1e-12
                        if abs_scale > 0:
                            scale = abs_scale
                        else:
                            p95 = np.percentile(mags, 95) if mags.size > 0 else 1.0
                            target = rel * plot_width
                            scale = target / (p95 + eps)
                        # Apply base mask from summed field if available
                        U_plot = U_grid * scale
                        V_plot = V_grid * scale
                        if base_mask is not None and base_mask.shape == U_plot.shape:
                            M = base_mask
                            ax.quiver(GX[M], GY[M], U_plot[M], V_plot[M],
                                  angles='xy', scale_units='xy', scale=1.0,
                                  color=color, alpha=alpha, zorder=z, width=0.002)
                        else:
                            ax.quiver(GX, GY, U_plot, V_plot,
                                      angles='xy', scale_units='xy', scale=1.0,
                                      color=color, alpha=alpha, zorder=z, width=0.002)
                else:
                    # Determine list of feature indices as above
                    feat_indices = list(grad_indices) if isinstance(grad_indices, (list, tuple, np.ndarray)) else []
                    try:
                        if isinstance(choice, int):
                            feat_indices = [int(choice)]
                        elif isinstance(choice, (list, tuple, np.ndarray)):
                            feat_indices = [int(i) for i in choice]
                        elif isinstance(choice, str) and len(choice) > 0:
                            if ',' in choice:
                                parts = [p.strip() for p in choice.split(',') if p.strip()]
                                parsed = []
                                for p in parts:
                                    try:
                                        parsed.append(int(p))
                                    except Exception:
                                        pass
                                if parsed:
                                    feat_indices = parsed
                                else:
                                    low = choice.lower()
                                    matches = [i for i, name in enumerate(col_labels) if low in str(name).lower()]
                                    if matches:
                                        feat_indices = [matches[0]]
                            else:
                                low = choice.lower()
                                matches = [i for i, name in enumerate(col_labels) if low in str(name).lower()]
                                if matches:
                                    feat_indices = [matches[0]]
                    except Exception:
                        pass
                    U_all = system.get('grid_u_all_feats', None)
                    V_all = system.get('grid_v_all_feats', None)
                    if U_all is not None and V_all is not None:
                        for f in feat_indices:
                            if 0 <= f < U_all.shape[0]:
                                U_grid = U_all[int(f)]
                                V_grid = V_all[int(f)]
                                c = feature_colors[f] if f < len(feature_colors) else '#333333'
                                mags = np.sqrt(U_grid**2 + V_grid**2)
                                eps = 1e-12
                                if abs_scale > 0:
                                    scale = abs_scale
                                else:
                                    p95 = np.percentile(mags, 95) if mags.size > 0 else 1.0
                                    target = rel * plot_width
                                    scale = target / (p95 + eps)
                                U_plot = U_grid * scale
                                V_plot = V_grid * scale
                                if base_mask is not None and base_mask.shape == U_plot.shape:
                                    M = base_mask
                                    ax.quiver(GX[M], GY[M], U_plot[M], V_plot[M],
                                              angles='xy', scale_units='xy', scale=1.0,
                                              color=c, alpha=alpha, zorder=z, width=0.002)
                                else:
                                    ax.quiver(GX, GY, U_plot, V_plot,
                                              angles='xy', scale_units='xy', scale=1.0,
                                              color=c, alpha=alpha, zorder=z, width=0.002)
    except Exception:
        pass

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
    # Match background to main wind-map styling
    try:
        from .color_system import BACKGROUND_COLOR
        ax2.set_facecolor(BACKGROUND_COLOR)
    except Exception:
        pass
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
    
    # Support multi-cell selections: use aggregated vectors over selected cells when provided
    selected_cells = mouse_data.get('selected_cells', []) or []
    use_selection = isinstance(selected_cells, (list, tuple)) and len(selected_cells) > 0

    # Fallback to single hovered cell when no selection
    if not use_selection:
        if 'grid_cell' not in mouse_data or mouse_data['grid_cell'] is None:
            return
        cell_i, cell_j = mouse_data['grid_cell']
    else:
        # If selection exists but no hovered cell, set a dummy cell index for downstream checks
        # We will avoid using (cell_i, cell_j) in selection mode for data access
        cell_i, cell_j = (-1, -1)

    grid_res = mouse_data.get('grid_res', config.DEFAULT_GRID_RES)
    
    # Validate grid cell indices against actual grid (only when not aggregating a selection)
    if not use_selection:
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
    unmasked = system.get('unmasked_cells', None)
    if use_selection:
        # Filter selection to unmasked cells
        valid_cells = []
        if unmasked is not None and hasattr(unmasked, 'shape'):
            for (ii, jj) in selected_cells:
                if 0 <= ii < unmasked.shape[0] and 0 <= jj < unmasked.shape[1] and bool(unmasked[ii, jj]):
                    valid_cells.append((int(ii), int(jj)))
        else:
            valid_cells = [(int(ii), int(jj)) for (ii, jj) in selected_cells]
        if len(valid_cells) == 0:
            ax2.text(0.5, 0.5, "Masked selection", transform=ax2.transAxes,
                     ha='center', va='center', fontsize=10, color='gray')
            return
    else:
        masked = False
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
            if use_selection:
                # Sum vectors across selected cells
                u_sum = 0.0
                v_sum = 0.0
                for (ii, jj) in valid_cells:
                    u_sum += float(grid_u_all_feats[feat_idx, ii, jj])
                    v_sum += float(grid_v_all_feats[feat_idx, ii, jj])
                u_val, v_val = u_sum, v_sum
            else:
                # Use cell-center values directly (consistent with optimization and no bounds errors)
                u_val = float(grid_u_all_feats[feat_idx, cell_i, cell_j])
                v_val = float(grid_v_all_feats[feat_idx, cell_i, cell_j])
            vectors_all.append([u_val, v_val])
            mags_all.append(np.sqrt(u_val**2 + v_val**2))
    else:
        return  # Can't visualize without grid data

    # Diagnostics: report whether the hovered cell center lies inside the data convex hull
    if not use_selection:
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
        if use_selection:
            last_printed = system.get('vane_last_printed_sel', None)
            sel_key = tuple(sorted(valid_cells))
            if last_printed != sel_key:
                system['vane_last_printed_sel'] = sel_key
                print(f"\nWind-Vane selection {list(sel_key)} aggregated feature vectors:")
                for feat_idx in range(len(col_labels)):
                    u_val, v_val = vectors_all[feat_idx]
                    mag_val = mags_all[feat_idx]
                    name = col_labels[feat_idx] if feat_idx < len(col_labels) else f"feat_{feat_idx}"
                    print(f"  {feat_idx:3d} | {name:>20s} : (u={u_val:+.5f}, v={v_val:+.5f}) | |v|={mag_val:.5f}")
        else:
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
    if use_selection and 'grid_u_all_feats' in system and 'grid_v_all_feats' in system:
        try:
            grid_u_all_feats = system['grid_u_all_feats']
            grid_v_all_feats = system['grid_v_all_feats']
            # Sum magnitudes across selected cells for each feature and take argmax
            feat_mags = []
            for feat_idx in range(len(col_labels)):
                s = 0.0
                for (ii, jj) in valid_cells:
                    u = float(grid_u_all_feats[feat_idx, ii, jj])
                    v = float(grid_v_all_feats[feat_idx, ii, jj])
                    s += np.hypot(u, v)
                feat_mags.append(s)
            if len(feat_mags) > 0:
                dominant_feature = int(np.argmax(feat_mags))
        except Exception:
            dominant_feature = -1
    elif 'cell_dominant_features' in system and not use_selection:
        cell_dominant_features = system['cell_dominant_features']
        dominant_feature = int(cell_dominant_features[cell_i, cell_j])
    
    
    # Place aggregated point at center as a simple black dot (consistent across modes)
    center_label = (f'Selection ({len(valid_cells)} cells)' if use_selection
                    else f'Cell ({cell_i},{cell_j})')
    ax2.scatter(0, 0, c='black', s=30, marker='o', edgecolor='none',
               zorder=10, label=center_label)
    
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
        
        # Calculate positive endpoint positions for convex hull (exclude negative vectors)
        all_endpoints = []
        vector_info = []
        
        for i, (vector, mag, feat_idx) in enumerate(zip(vectors_selected, mags_selected, features_selected)):
            # Scale the gradient vector to match the consistent magnitude
            scaled_vector = np.array(vector) * scale_factor
            
            # Positive endpoint only
            pos_end = scaled_vector
            # Negative endpoint intentionally excluded from convex hull and drawing
            neg_end = -scaled_vector
            
            all_endpoints.append(pos_end)
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
                            if pos_on_hull:
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
                          linewidth=1.8, alpha=0.9, zorder=22)
                # Label highlighted vectors (optional)
                if bool(getattr(config, 'SHOW_VECTOR_LABELS', True)) and feat_idx < len(col_labels):
                    ax2.text(info['pos_end'][0] * 1.1, info['pos_end'][1] * 1.1,
                             col_labels[feat_idx], fontsize=8, ha='center', va='center', alpha=0.9, zorder=23)

            # Also draw an enclosing ring (no direction dot) for the Feature Clock
            try:
                max_vec_radius = max(np.linalg.norm(info['vector']) for info in vector_info) if vector_info else 0.0
            except Exception:
                max_vec_radius = 0.0
            scale = float(getattr(config, 'WIND_VANE_RING_SCALE', 1.04))
            max_r = float(getattr(config, 'WIND_VANE_RING_MAX_R', 0.66))
            ring_r = max_vec_radius * scale
            if not np.isfinite(ring_r) or ring_r <= 1e-9:
                ring_r = min(0.2, max_r)
            ring_r = min(ring_r, max_r)

            ring_color = getattr(config, 'WIND_VANE_RING_COLOR', '#999999')
            ring = plt.Circle((0.0, 0.0), radius=ring_r, fill=False,
                              edgecolor=ring_color, linewidth=1.5,
                              alpha=0.6, zorder=19)
            ax2.add_patch(ring)

            # Skip classic hull/needle flow in Feature Clock mode
            return
        
        # Handle degenerate cases for convex hull
        if len(vectors_selected) == 1:
            # Single feature case - draw positive direction only (no negative vector)
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
            
            # Draw arrow in positive direction only
            ax2.arrow(0, 0, scaled_vector[0], scaled_vector[1], 
                     head_width=0.05, head_length=0.05, 
                     fc=color, ec=color, linewidth=2, alpha=0.8, zorder=22)
            
            # Draw magnitude circle
            from matplotlib.patches import Circle
            magnitude_circle = Circle((0, 0), radius=np.linalg.norm(scaled_vector),
                                     fill=False, color=color, linewidth=1.5, 
                                     linestyle='--', alpha=0.4, zorder=7)
            ax2.add_patch(magnitude_circle)
            
            # Add feature label (optional)
            if bool(getattr(config, 'SHOW_VECTOR_LABELS', True)):
                ax2.text(scaled_vector[0]*1.2, scaled_vector[1]*1.2, 
                        col_labels[features_selected[0]], 
                        ha='center', va='center', fontsize=9, color=color, zorder=23)
            
        elif len(all_endpoints) >= 3 and bool(getattr(config, 'WIND_VANE_USE_CONVEX_HULL', True)):
            try:
                # Calculate convex hull
                hull = ConvexHull(all_endpoints)
                
                # Calculate convex hull for boundary detection; optionally draw it
                hull_points = np.array(all_endpoints)[hull.vertices]
                if bool(getattr(config, 'WIND_VANE_SHOW_HULL', False)):
                    edge_c = getattr(config, 'WIND_VANE_HULL_EDGE_COLOR', '#1f77b4')
                    edge_w = float(getattr(config, 'WIND_VANE_HULL_EDGE_WIDTH', 1.0))
                    z_hull = int(getattr(config, 'WIND_VANE_HULL_ZORDER', 18))
                    # Draw boundary only (no fill inside the hull)
                    hull_polygon = plt.Polygon(
                        hull_points,
                        closed=True,
                        fill=False,
                        edgecolor=edge_c,
                        linewidth=edge_w,
                        zorder=z_hull,
                    )
                    ax2.add_patch(hull_polygon)
                    # If user wants ONLY the filled hull, return early
                    if bool(getattr(config, 'WIND_VANE_ONLY_HULL', False)):
                        return
                
            except Exception as e:
                # Degenerate hull (e.g., collinear or duplicate points) — draw positive vectors as arrows only
                for i, (vector, feat_idx) in enumerate(zip(vectors_selected, features_selected)):
                    scaled_vector = np.array(vector) * scale_factor
                    color = feature_colors[feat_idx] if feat_idx < len(feature_colors) else 'black'
                ax2.arrow(0, 0, scaled_vector[0], scaled_vector[1],
                          head_width=0.04, head_length=0.04,
                          fc=color, ec=color, linewidth=1.5, alpha=0.7, zorder=22)
        
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
            # Wind vane representation: choose between ring+dot or traditional needle
            vane_style = str(getattr(config, 'WIND_VANE_STYLE', 'needle')).lower()
            perp_direction = np.array([-flow_direction[1], flow_direction[0]])  # Perpendicular

            if vane_style in ('circle', 'ring_dot', 'wedge'):
                # Ring that encloses all feature vectors + a dot at flow direction
                try:
                    max_vec_radius = max(np.linalg.norm(info['vector']) for info in vector_info) if vector_info else 0.0
                except Exception:
                    max_vec_radius = 0.0
                scale = float(getattr(config, 'WIND_VANE_RING_SCALE', 1.04))
                max_r = float(getattr(config, 'WIND_VANE_RING_MAX_R', 0.66))
                ring_r = max_vec_radius * scale
                if not np.isfinite(ring_r) or ring_r <= 1e-9:
                    ring_r = min(0.2, max_r)  # fallback small ring
                ring_r = min(ring_r, max_r)

                # Draw enclosing ring
                ring_color = getattr(config, 'WIND_VANE_RING_COLOR', '#999999')
                ring = plt.Circle((0.0, 0.0), radius=ring_r, fill=False,
                                  edgecolor=ring_color, linewidth=1.5,
                                  alpha=0.6, zorder=19)
                ax2.add_patch(ring)

                # Direction dot: always placed ON the ring (position encodes direction only)
                dot_r = float(getattr(config, 'WIND_VANE_CIRCLE_RADIUS', 0.055))
                # Keep dot on the ring circumference
                dot_center = flow_direction * ring_r if sum_magnitude > 1e-12 else np.array([ring_r, 0.0])
                # Choose color from the selected feature with the highest projection
                # onto the aggregated flow direction (directional contribution)
                ring_dot_color = 'black'
                try:
                    if 'best_feature_idx' in locals() and best_feature_idx is not None:
                        if 0 <= best_feature_idx < len(feature_colors):
                            ring_dot_color = feature_colors[best_feature_idx]
                    elif (dominant_feature is not None and dominant_feature >= 0 and
                          (dominant_feature in selected_features) and
                          dominant_feature < len(feature_colors)):
                        ring_dot_color = feature_colors[dominant_feature]
                    elif 'dominant_color' in locals():
                        ring_dot_color = dominant_color
                except Exception:
                    pass
                # Alpha mode: speed vs field strength
                dot_alpha_mode = str(getattr(config, 'RING_DOT_ALPHA_MODE', 'speed')).lower()
                if dot_alpha_mode == 'field':
                    # Normalize summed magnitude against the longest individual vector magnitude
                    # Use smooth tanh to avoid hard saturation when vectors align strongly
                    denom = float(max_mag) if 'max_mag' in locals() and max_mag > 1e-12 else (sum_magnitude + 1e-9)
                    base = float(sum_magnitude) / (denom + 1e-9)
                    ratio = float(np.tanh(base))  # 0..~1
                    dot_alpha = max(0.3, min(1.0, 0.3 + 0.7 * ratio))
                else:
                    dot_alpha = magnitude_alpha
                dir_dot = plt.Circle((float(dot_center[0]), float(dot_center[1])), radius=dot_r,
                                     color=ring_dot_color, alpha=dot_alpha, zorder=21)
                ax2.add_patch(dir_dot)

                # Optional faint guide from center for readability
                if getattr(config, 'WIND_VANE_CIRCLE_GUIDE', False):
                    ax2.plot([0, dot_center[0]], [0, dot_center[1]], color=ring_dot_color,
                             linewidth=1.0, alpha=magnitude_alpha*0.5, zorder=20)
            else:
                # 1. Main shaft (thick central line) - much thicker than feature vectors
                shaft_length = vane_length * 0.75  # reduced from 0.8
                shaft_start = -flow_direction * shaft_length * 0.5
                shaft_end = flow_direction * shaft_length * 0.5
                ax2.plot([shaft_start[0], shaft_end[0]], [shaft_start[1], shaft_end[1]], 
                        color=dominant_color, linewidth=6, alpha=magnitude_alpha, 
                        solid_capstyle='round', zorder=20)
                
                # 2. Large arrow head (pointing in flow direction) - fixed position at radius
                arrow_head_pos = flow_direction * vane_length  # Fixed at vane_length radius
                arrow_size = vane_length * 0.32
                
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
                flag_size = vane_length * 0.28
                
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
        if not bool(getattr(config, 'WIND_VANE_USE_CONVEX_HULL', True)):
            hull_vertex_indices = set(range(len(all_endpoints)))  # treat all as on hull
        else:
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
            
            for idx, endpoint in enumerate(all_endpoints):
                if np.allclose(endpoint, info['pos_end']):
                    pos_on_hull = idx in hull_vertex_indices
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
            elif pos_on_hull and feat_idx < len(feature_colors):
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
                gray_value = 0.6
                pos_alpha = 0.2
                pos_color = (gray_value, gray_value, gray_value, pos_alpha)
            
            # Draw vector arrows
            ax2.annotate('', xy=info['pos_end'], xytext=(0, 0), zorder=22,
                        arrowprops=dict(arrowstyle='->', color=pos_color, lw=2,
                                      shrinkA=0, shrinkB=0, alpha=pos_color[3]))
            
            # Add feature labels ONLY for vectors on convex hull boundary (positive only, optional)
            if bool(getattr(config, 'SHOW_VECTOR_LABELS', True)) and pos_on_hull and feat_idx < len(col_labels):
                label = col_labels[feat_idx]
                # Label near positive endpoint for consistency
                ax2.text(info['pos_end'][0] * 1.1, info['pos_end'][1] * 1.1, label,
                        fontsize=8, ha='center', va='center', alpha=0.8, zorder=23)


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
    
    # Apply unified title styling to both axes (optional)
    if bool(getattr(config, 'SHOW_TITLES', True)):
        title_kwargs = dict(fontsize=14, color=TEXT_COLOR, pad=14, weight='bold')
        ax1.set_title("Wind Map", **title_kwargs)
    else:
        try:
            ax1.set_title("")
        except Exception:
            pass

    # Style wind vane - align with main plot background
    ax2.set_facecolor(BACKGROUND_COLOR)
    
    # Style wind vane spines
    for spine in ax2.spines.values():
        spine.set_color(TEXT_COLOR)
        spine.set_linewidth(0.5)
        
    # Wind vane title styling (identical to map; optional)
    if bool(getattr(config, 'SHOW_TITLES', True)):
        ax2.set_title("Wind Vane", **title_kwargs)
    else:
        try:
            ax2.set_title("")
        except Exception:
            pass


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
