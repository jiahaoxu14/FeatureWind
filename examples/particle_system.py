"""
Particle system module for FeatureWind.

This module handles particle physics, including adaptive RK4 integration,
particle reseeding, and trajectory management for flow field visualization.
Enhanced with family-based coloring system.
"""

import numpy as np
from matplotlib.collections import LineCollection
import config


def create_particles(num_particles=None, cell_dominant_features=None, grid_res=None):
    """
    Create a particle system for flow visualization.
    
    Args:
        num_particles (int, optional): Number of particles to create
        cell_dominant_features (np.ndarray, optional): Dominant features per grid cell
        grid_res (int, optional): Grid resolution
        
    Returns:
        dict: Particle system dictionary with positions, lifetimes, histories, etc.
    """
    if num_particles is None:
        num_particles = config.DEFAULT_NUM_PARTICLES
        
    xmin, xmax, ymin, ymax = config.bounding_box
    particle_positions = np.column_stack((
        np.random.uniform(xmin, xmax, size=num_particles),
        np.random.uniform(ymin, ymax, size=num_particles)
    ))

    max_lifetime = config.PARTICLE_LIFETIME
    tail_gap = config.TAIL_LENGTH
    particle_lifetimes = np.zeros(num_particles, dtype=int)
    histories = np.full((num_particles, tail_gap + 1, 2), np.nan)
    histories[:, :] = particle_positions[:, None, :]

    # A single LineCollection for all particles
    lc = LineCollection([], linewidths=1.5, zorder=2)

    # Store everything in a dict
    system = {
        'particle_positions': particle_positions,
        'particle_lifetimes': particle_lifetimes,
        'histories': histories,
        'tail_gap': tail_gap,
        'max_lifetime': max_lifetime,
        'linecoll': lc,
    }

    return system


def get_velocity_at_positions(positions, system, interp_u_sum=None, interp_v_sum=None, 
                            grid_u_sum=None, grid_v_sum=None, grid_res=None):
    """
    Get velocity at particle positions using interpolation or grid lookup.
    
    Args:
        positions (np.ndarray): Particle positions, shape (N, 2)
        system (dict): Particle system dictionary
        interp_u_sum, interp_v_sum: Velocity interpolators
        grid_u_sum, grid_v_sum: Velocity grids
        grid_res (int): Grid resolution
        
    Returns:
        np.ndarray: Velocity vectors at positions, shape (N, 2)
    """
    xmin, xmax, ymin, ymax = config.bounding_box
    
    # Always prefer smooth bilinear interpolation for consistent motion
    if 'interp_u_sum' in system and 'interp_v_sum' in system:
        # Direction-conditioned mode: use updated interpolators from system
        current_interp_u = system['interp_u_sum']
        current_interp_v = system['interp_v_sum']
        # Fix coordinate order: interpolators expect (y, x) but we have (x, y)
        U = current_interp_u(positions[:, [1, 0]])  # Swap to (y, x)
        V = current_interp_v(positions[:, [1, 0]])  # Swap to (y, x)
    elif interp_u_sum is not None and interp_v_sum is not None:
        # Top-K mode: use original interpolators
        # Fix coordinate order: interpolators expect (y, x) but we have (x, y)
        U = interp_u_sum(positions[:, [1, 0]])  # Swap to (y, x)
        V = interp_v_sum(positions[:, [1, 0]])  # Swap to (y, x)
    else:
        # Fallback: direct grid indexing (should rarely be used)
        if 'grid_u_sum' in system and 'grid_v_sum' in system:
            current_grid_u = system['grid_u_sum']
            current_grid_v = system['grid_v_sum']
            # Convert positions to grid cell indices
            cell_i_indices = np.clip(((positions[:, 1] - ymin) / (ymax - ymin) * grid_res).astype(int), 0, grid_res - 1)
            cell_j_indices = np.clip(((positions[:, 0] - xmin) / (xmax - xmin) * grid_res).astype(int), 0, grid_res - 1)
            # Sample velocity directly from grid cells
            U = current_grid_u[cell_i_indices, cell_j_indices]
            V = current_grid_v[cell_i_indices, cell_j_indices]
        else:
            # No velocity field available
            U = np.zeros(len(positions))
            V = np.zeros(len(positions))
    
    velocity = np.column_stack((U, V)) * config.velocity_scale
    
    # Add safety check to prevent runaway particles
    velocity_magnitudes = np.linalg.norm(velocity, axis=1)
    max_safe_velocity = config.MAX_SAFE_VELOCITY
    
    # Clip velocities that are too large
    mask = velocity_magnitudes > max_safe_velocity
    if np.any(mask):
        velocity[mask] = velocity[mask] / velocity_magnitudes[mask, np.newaxis] * max_safe_velocity
        
    return velocity


def adaptive_rk4_step(pos, target_dt, get_vel_func, grid_res=None, max_error=None):
    """
    Adaptive RK4 with embedded Heun method for error estimation.
    
    Args:
        pos (np.ndarray): Current positions
        target_dt (float): Target time step
        get_vel_func (callable): Function to get velocity at positions
        grid_res (int, optional): Grid resolution for CFL condition
        max_error (float, optional): Maximum allowed error
        
    Returns:
        tuple: (new_position, actual_dt_used, error_estimate, estimated_velocity)
    """
    if max_error is None:
        max_error = config.ERROR_TOLERANCE
        
    # Calculate velocity at current position
    vel = get_vel_func(pos)
    speed = np.linalg.norm(vel, axis=1)
    
    # CFL-like condition: dt should be proportional to cell_size / |v|
    if grid_res is not None and len(config.bounding_box) >= 4:
        cell_size = min(
            (config.bounding_box[1] - config.bounding_box[0]) / grid_res,  # dx
            (config.bounding_box[3] - config.bounding_box[2]) / grid_res   # dy
        )
        # CFL number around 0.5 for stability
        cfl_number = config.CFL_NUMBER
        max_speed = np.maximum(speed, 1e-6)  # Avoid division by zero
        cfl_dt = cfl_number * cell_size / max_speed
        
        # Use minimum of target dt and CFL-limited dt for each particle
        dt_per_particle = np.minimum(target_dt, cfl_dt)
        # Use the most restrictive dt for this step
        dt = np.min(dt_per_particle)
    else:
        dt = target_dt
    
    # Ensure minimum time step
    dt = max(dt, config.MIN_TIME_STEP)
    
    # RK4 step
    k1 = get_vel_func(pos)
    k2 = get_vel_func(pos + 0.5 * dt * k1)
    k3 = get_vel_func(pos + 0.5 * dt * k2)
    k4 = get_vel_func(pos + dt * k3)
    
    rk4_result = pos + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    # Estimate velocity at final position using RK4 intermediate values
    # This is more accurate than k4 alone and avoids recomputing velocity
    estimated_final_velocity = (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    # Embedded Heun method for error estimation (simpler RK2)
    heun_k1 = k1
    heun_k2 = get_vel_func(pos + dt * heun_k1)
    heun_result = pos + dt * (heun_k1 + heun_k2) / 2.0
    
    # Estimate local truncation error
    error = np.linalg.norm(rk4_result - heun_result, axis=1)
    max_error_this_step = np.max(error)
    
    return rk4_result, dt, max_error_this_step, estimated_final_velocity


def density_aware_reseed(system, grid_res):
    """
    Implement density and divergence-aware particle reseeding for temporal coherence.
    
    Args:
        system (dict): Particle system dictionary
        grid_res (int): Grid resolution
    """
    pp = system['particle_positions']
    lt = system['particle_lifetimes']
    his = system['histories']
    max_lifetime = system['max_lifetime']
    
    xmin, xmax, ymin, ymax = config.bounding_box
    
    if len(pp) == 0:
        return
        
    # Create density grid (coarser than flow field for efficiency)
    density_res = max(8, grid_res // 4)  # Adaptive resolution
    density_grid = np.zeros((density_res, density_res))
    divergence_grid = np.zeros((density_res, density_res))
    
    # Compute particle density
    for pos in pp:
        if xmin <= pos[0] <= xmax and ymin <= pos[1] <= ymax:
            grid_x = int((pos[0] - xmin) / (xmax - xmin) * (density_res - 1))
            grid_y = int((pos[1] - ymin) / (ymax - ymin) * (density_res - 1))
            grid_x = max(0, min(density_res - 1, grid_x))
            grid_y = max(0, min(density_res - 1, grid_y))
            density_grid[grid_y, grid_x] += 1
    
    # Compute flow divergence on density grid
    for i in range(1, density_res - 1):
        for j in range(1, density_res - 1):
            # Map density grid coordinates to flow field coordinates
            x = xmin + j / (density_res - 1) * (xmax - xmin)
            y = ymin + i / (density_res - 1) * (ymax - ymin)
            
            # Sample velocity at neighboring points using our interpolators
            h = min((xmax - xmin) / density_res, (ymax - ymin) / density_res)
            try:
                u_right = system['interp_u_sum']([[x + h, y]])[0]
                u_left = system['interp_u_sum']([[x - h, y]])[0]
                v_up = system['interp_v_sum']([[x, y + h]])[0]
                v_down = system['interp_v_sum']([[x, y - h]])[0]
                
                # Compute divergence: div = du/dx + dv/dy
                du_dx = (u_right - u_left) / (2 * h)
                dv_dy = (v_up - v_down) / (2 * h)
                divergence_grid[i, j] = du_dx + dv_dy
            except:
                divergence_grid[i, j] = 0.0
    
    # Target density (particles per cell)
    total_particles = len(pp)
    target_density = total_particles / (density_res * density_res)
    
    # Find cells that need reseeding (low density, especially in divergent regions)
    reseed_candidates = []
    remove_candidates = []
    
    for i in range(density_res):
        for j in range(density_res):
            current_density = density_grid[i, j]
            div_factor = max(0, divergence_grid[i, j])  # Only consider divergent regions
            
            # Cells with low density and positive divergence need particles
            if current_density < target_density * 0.7:
                reseed_weight = (target_density - current_density) * (1 + div_factor * 0.5)
                reseed_candidates.append((i, j, reseed_weight))
            
            # Cells with high density could lose particles
            elif current_density > target_density * 1.5:
                remove_candidates.append((i, j, current_density - target_density))
    
    # Limit reseeding to maintain temporal coherence (max 2% per frame)
    max_reseed = int(config.MAX_RESEED_RATE * len(pp))
    if max_reseed > 0 and reseed_candidates:
        # Sort by reseeding weight
        reseed_candidates.sort(key=lambda x: x[2], reverse=True)
        
        num_reseeded = 0
        for i, j, weight in reseed_candidates:
            if num_reseeded >= max_reseed:
                break
            
            # Find a particle to respawn (prefer older particles in high-density regions)
            candidate_particles = []
            for idx, pos in enumerate(pp):
                # Check if particle is in a high-density region or very old
                px_grid = int((pos[0] - xmin) / (xmax - xmin) * (density_res - 1))
                py_grid = int((pos[1] - ymin) / (ymax - ymin) * (density_res - 1))
                px_grid = max(0, min(density_res - 1, px_grid))
                py_grid = max(0, min(density_res - 1, py_grid))
                
                # Prefer particles from high-density regions or old particles
                if (density_grid[py_grid, px_grid] > target_density * 1.3 or 
                    lt[idx] > max_lifetime * 0.8):
                    candidate_particles.append(idx)
            
            if candidate_particles:
                # Choose particle to respawn (prefer oldest)
                idx = max(candidate_particles, key=lambda x: lt[x])
            else:
                # Fallback: choose randomly from all particles
                idx = np.random.randint(len(pp))
            
            # Respawn in the target cell with some jitter
            cell_x = xmin + (j + np.random.uniform(-0.4, 0.4)) / (density_res - 1) * (xmax - xmin)
            cell_y = ymin + (i + np.random.uniform(-0.4, 0.4)) / (density_res - 1) * (ymax - ymin)
            
            # Ensure within bounds
            cell_x = max(xmin, min(xmax, cell_x))
            cell_y = max(ymin, min(ymax, cell_y))
            
            pp[idx] = [cell_x, cell_y]
            his[idx] = pp[idx]
            lt[idx] = 0
            num_reseeded += 1


def reinitialize_particles(system):
    """
    Reinitialize out-of-bounds or over-age particles.
    
    Args:
        system (dict): Particle system dictionary
    """
    pp = system['particle_positions']
    lt = system['particle_lifetimes']
    his = system['histories']
    max_lifetime = system['max_lifetime']
    
    xmin, xmax, ymin, ymax = config.bounding_box
    
    for i in range(len(pp)):
        x, y = pp[i]
        if (x < xmin or x > xmax or y < ymin or y > ymax
            or lt[i] > max_lifetime):
            pp[i] = [
                np.random.uniform(xmin, xmax),
                np.random.uniform(ymin, ymax)
            ]
            his[i] = pp[i]
            lt[i] = 0


def get_dominant_feature_at_position(position, system):
    """
    Get the dominant feature at a given position.
    
    Args:
        position (np.ndarray): Position coordinates [x, y]
        system (dict): Particle system dictionary
        
    Returns:
        int: Index of dominant feature, or -1 if none
    """
    if 'cell_dominant_features' not in system:
        return -1
        
    xmin, xmax, ymin, ymax = config.bounding_box
    cell_dominant_features = system['cell_dominant_features']
    grid_res = cell_dominant_features.shape[0]
    
    # Convert position to grid cell indices
    if xmin <= position[0] <= xmax and ymin <= position[1] <= ymax:
        cell_j = int((position[0] - xmin) / (xmax - xmin) * grid_res)
        cell_i = int((position[1] - ymin) / (ymax - ymin) * grid_res)
        
        # Clamp to valid range
        cell_i = max(0, min(grid_res - 1, cell_i))
        cell_j = max(0, min(grid_res - 1, cell_j))
        
        return cell_dominant_features[cell_i, cell_j]
    
    return -1


def get_directional_dominant_feature_at_position(position, velocity_direction, system):
    """
    Get the feature that contributes most to the particle's movement direction.
    
    Args:
        position (np.ndarray): Position coordinates [x, y]
        velocity_direction (np.ndarray): Normalized velocity direction [dx, dy]
        system (dict): Particle system dictionary
        
    Returns:
        int: Index of directionally dominant feature, or -1 if none
    """
    if ('grid_u_all_feats' not in system or 'grid_v_all_feats' not in system or 
        np.linalg.norm(velocity_direction) < 1e-8):
        return -1
        
    xmin, xmax, ymin, ymax = config.bounding_box
    grid_u_all_feats = system['grid_u_all_feats']
    grid_v_all_feats = system['grid_v_all_feats']
    
    num_features = grid_u_all_feats.shape[0]
    grid_res = grid_u_all_feats.shape[1]
    
    # Convert position to grid cell indices
    if xmin <= position[0] <= xmax and ymin <= position[1] <= ymax:
        cell_j = int((position[0] - xmin) / (xmax - xmin) * grid_res)
        cell_i = int((position[1] - ymin) / (ymax - ymin) * grid_res)
        
        # Clamp to valid range
        cell_i = max(0, min(grid_res - 1, cell_i))
        cell_j = max(0, min(grid_res - 1, cell_j))
        
        # Get all feature vectors at this position using vectorized operations
        u_vals = grid_u_all_feats[:, cell_i, cell_j]  # Shape: (num_features,)
        v_vals = grid_v_all_feats[:, cell_i, cell_j]  # Shape: (num_features,)
        feature_vectors = np.column_stack([u_vals, v_vals])  # Shape: (num_features, 2)
        
        # Calculate magnitudes for all features at once
        magnitudes = np.linalg.norm(feature_vectors, axis=1)
        
        # Filter out zero vectors
        valid_mask = magnitudes > 1e-8
        if not np.any(valid_mask):
            return -1
        
        # Calculate directional contributions for all valid features at once
        contributions = np.dot(feature_vectors[valid_mask], velocity_direction)
        
        # Find the best contribution among valid features
        if len(contributions) > 0:
            best_idx_among_valid = np.argmax(contributions)
            # Map back to original feature index
            valid_indices = np.where(valid_mask)[0]
            best_feature_idx = valid_indices[best_idx_among_valid]
        else:
            best_feature_idx = -1
        
        return best_feature_idx
    
    return -1


def get_magnitude_at_position(position, feature_idx, system):
    """
    Get gradient magnitude for a specific feature at a position.
    
    Args:
        position (np.ndarray): Position coordinates [x, y]
        feature_idx (int): Feature index
        system (dict): Particle system dictionary
        
    Returns:
        float: Gradient magnitude at position
    """
    if ('grid_u_all_feats' not in system or 'grid_v_all_feats' not in system or 
        feature_idx < 0):
        return 0.0
        
    xmin, xmax, ymin, ymax = config.bounding_box
    grid_u_all_feats = system['grid_u_all_feats']
    grid_v_all_feats = system['grid_v_all_feats']
    grid_res = grid_u_all_feats.shape[1]
    
    if feature_idx >= grid_u_all_feats.shape[0]:
        return 0.0
    
    # Convert position to grid cell indices
    if xmin <= position[0] <= xmax and ymin <= position[1] <= ymax:
        cell_j = int((position[0] - xmin) / (xmax - xmin) * grid_res)
        cell_i = int((position[1] - ymin) / (ymax - ymin) * grid_res)
        
        # Clamp to valid range
        cell_i = max(0, min(grid_res - 1, cell_i))
        cell_j = max(0, min(grid_res - 1, cell_j))
        
        u_val = grid_u_all_feats[feature_idx, cell_i, cell_j]
        v_val = grid_v_all_feats[feature_idx, cell_i, cell_j]
        
        return np.sqrt(u_val**2 + v_val**2)
    
    return 0.0


def get_dominance_at_position(position, feature_idx, system):
    """
    Get dominance/uncertainty measure for a feature at a position.
    
    Args:
        position (np.ndarray): Position coordinates [x, y] 
        feature_idx (int): Feature index
        system (dict): Particle system dictionary
        
    Returns:
        float: Dominance value (0-1, higher = more dominant)
    """
    if 'cell_soft_dominance' not in system or feature_idx < 0:
        # Fallback: use hard dominance
        dominant_feat = get_dominant_feature_at_position(position, system)
        return 1.0 if dominant_feat == feature_idx else 0.3
        
    xmin, xmax, ymin, ymax = config.bounding_box
    cell_soft_dominance = system['cell_soft_dominance']
    
    if cell_soft_dominance is None or feature_idx >= cell_soft_dominance.shape[2]:
        return 0.5  # Neutral dominance
    
    grid_res = cell_soft_dominance.shape[0]
    
    # Convert position to grid cell indices
    if xmin <= position[0] <= xmax and ymin <= position[1] <= ymax:
        cell_j = int((position[0] - xmin) / (xmax - xmin) * grid_res)
        cell_i = int((position[1] - ymin) / (ymax - ymin) * grid_res)
        
        # Clamp to valid range
        cell_i = max(0, min(grid_res - 1, cell_i))
        cell_j = max(0, min(grid_res - 1, cell_j))
        
        return cell_soft_dominance[cell_i, cell_j, feature_idx]
    
    return 0.5


def update_particle_colors_family_based(system, family_assignments=None, feature_colors=None):
    """
    Update particle colors based on family assignment, magnitude, and dominance.
    
    Args:
        system (dict): Particle system dictionary
        family_assignments (np.ndarray, optional): Family ID for each feature
        feature_colors (list, optional): Hex colors for each feature
        
    Returns:
        np.ndarray: RGBA colors for each particle
    """
    particle_positions = system['particle_positions']
    n_particles = len(particle_positions)
    
    # Initialize colors array
    particle_colors = np.zeros((n_particles, 4))  # RGBA
    
    if family_assignments is None or feature_colors is None:
        # Fallback: use speed-based grayscale coloring
        if 'last_velocity' in system:
            speeds = np.linalg.norm(system['last_velocity'], axis=1)
            max_speed = speeds.max() + 1e-9
            normalized_speeds = speeds / max_speed
            
            for i in range(n_particles):
                intensity = 0.3 + 0.7 * normalized_speeds[i]
                particle_colors[i] = [0, 0, 0, intensity]
        else:
            particle_colors[:] = [0, 0, 0, 0.5]  # Default black with 50% alpha
        
        return particle_colors
    
    # Import color utilities
    from color_system import hex_to_rgb
    
    # Get maximum magnitudes for each feature (for normalization)
    max_magnitudes = {}
    if 'grid_u_all_feats' in system and 'grid_v_all_feats' in system:
        grid_u_all_feats = system['grid_u_all_feats']
        grid_v_all_feats = system['grid_v_all_feats']
        
        for feat_idx in range(len(family_assignments)):
            if feat_idx < grid_u_all_feats.shape[0]:
                magnitude_grid = np.sqrt(grid_u_all_feats[feat_idx]**2 + grid_v_all_feats[feat_idx]**2)
                max_magnitudes[feat_idx] = magnitude_grid.max()
            else:
                max_magnitudes[feat_idx] = 1.0
    
    # Color each particle based on its directionally dominant feature
    # Get particle velocities for directional analysis
    velocities = system.get('last_velocity', None)
    
    for i, position in enumerate(particle_positions):
        # Use directional contribution if velocity is available
        if velocities is not None and i < len(velocities):
            velocity = velocities[i]
            speed = np.linalg.norm(velocity)
            
            if speed > 1e-8:
                # Use directional contribution coloring
                velocity_direction = velocity / speed  # Normalize to unit vector
                dominant_feature = get_directional_dominant_feature_at_position(position, velocity_direction, system)
            else:
                # Fallback to magnitude-based for stationary particles
                dominant_feature = get_dominant_feature_at_position(position, system)
        else:
            # Fallback to magnitude-based coloring if no velocity data
            dominant_feature = get_dominant_feature_at_position(position, system)
        
        if dominant_feature >= 0 and dominant_feature < len(feature_colors):
            # Get base family color - use directly without lightness modulation for consistency
            base_color = feature_colors[dominant_feature]
            rgb = hex_to_rgb(base_color)
            
            # Get local gradient magnitude for alpha modulation instead of lightness
            magnitude = get_magnitude_at_position(position, dominant_feature, system)
            max_magnitude = max_magnitudes.get(dominant_feature, 1.0)
            magnitude_factor = magnitude / max_magnitude if max_magnitude > 0 else 1.0
            
            # Get dominance for additional alpha modulation
            dominance = get_dominance_at_position(position, dominant_feature, system)
            
            # Combine magnitude and dominance for alpha (not lightness)
            alpha = 0.3 + 0.6 * magnitude_factor * dominance
            alpha = min(0.9, max(0.3, alpha))
            
            particle_colors[i] = [*rgb, alpha]
            
        else:
            # No dominant feature or invalid feature - use neutral gray
            particle_colors[i] = [0.5, 0.5, 0.5, 0.3]
    
    return particle_colors


def update_particle_visualization(system, velocity, family_assignments=None, feature_colors=None):
    """
    Update particle visualization with family-based coloring.
    
    Args:
        system (dict): Particle system dictionary
        velocity (np.ndarray): Current particle velocities
        family_assignments (np.ndarray, optional): Family assignments for features
        feature_colors (list, optional): Hex colors for features
        
    Returns:
        tuple: Updated line collection
    """
    pp = system['particle_positions']
    his = system['histories']
    lc_ = system['linecoll']
    tail_gap = system['tail_gap']
    
    # Build line segments
    n_active = len(pp)
    segments = np.zeros((n_active * tail_gap, 2, 2))
    colors_rgba = np.zeros((n_active * tail_gap, 4))

    # Compute speeds for alpha fade
    speeds = np.linalg.norm(velocity, axis=1)
    max_speed = speeds.max() + 1e-9  # avoid division by zero
    
    # Store velocity in system for color computation
    system['last_velocity'] = velocity

    # Get family-based colors for particles
    if family_assignments is not None and feature_colors is not None:
        particle_colors = update_particle_colors_family_based(system, family_assignments, feature_colors)
    else:
        # Fallback: speed-based grayscale coloring  
        particle_colors = np.zeros((n_active, 4))
        for i in range(n_active):
            intensity = 0.3 + 0.7 * (speeds[i] / max_speed)
            particle_colors[i] = [0, 0, 0, intensity]  # Black with speed-based alpha

    # Build trail segments with family-based colors
    for i in range(n_active):
        # Use particle's family-based color
        r, g, b, base_alpha = particle_colors[i]
        
        # Modulate alpha by speed for additional visual feedback
        speed_alpha = speeds[i] / max_speed

        for t in range(tail_gap):
            seg_idx = i * tail_gap + t
            segments[seg_idx, 0, :] = his[i, t, :]
            segments[seg_idx, 1, :] = his[i, t + 1, :]

            # Age-based fade for trail effect
            age_factor = (t+1) / tail_gap
            
            # Dynamic alpha with better range: combines speed and age multiplicatively
            # But with higher base to maintain visibility
            alpha_min = 0.25
            # Use power function for more dramatic speed differences
            speed_factor = speed_alpha ** 0.7  # Power < 1 compresses low values less
            alpha_final = max(alpha_min, min(1.0, 0.4 + 0.6 * speed_factor * age_factor))

            # Assign the final RGBA
            colors_rgba[seg_idx] = [r, g, b, alpha_final]

    lc_.set_segments(segments)
    lc_.set_colors(colors_rgba)
    
    return (lc_,)


def update_particles_with_families(system, interp_u_sum=None, interp_v_sum=None, 
                                  grid_u_sum=None, grid_v_sum=None, grid_res=None,
                                  family_assignments=None, feature_colors=None):
    """
    Enhanced particle update function with family-based coloring.
    
    Args:
        system (dict): Particle system dictionary
        interp_u_sum, interp_v_sum: Velocity interpolators
        grid_u_sum, grid_v_sum: Velocity grids
        grid_res (int): Grid resolution
        family_assignments (np.ndarray): Family assignments for features
        feature_colors (list): Hex colors for features
        
    Returns:
        tuple: Updated line collection with family colors
    """
    # First do standard physics update
    result = update_particles(system, interp_u_sum, interp_v_sum, grid_u_sum, grid_v_sum, grid_res)
    
    # Then update visualization with family colors
    if 'last_velocity' in system:
        velocity = system['last_velocity']
        return update_particle_visualization(system, velocity, family_assignments, feature_colors)
    
    return result


def update_particles(system, interp_u_sum=None, interp_v_sum=None, 
                    grid_u_sum=None, grid_v_sum=None, grid_res=None):
    """
    Main particle update function that handles physics and visualization.
    
    Args:
        system (dict): Particle system dictionary
        interp_u_sum, interp_v_sum: Velocity interpolators
        grid_u_sum, grid_v_sum: Velocity grids
        grid_res (int): Grid resolution
        
    Returns:
        tuple: Updated line collection
    """
    pp = system['particle_positions']
    lt = system['particle_lifetimes']
    his = system['histories']
    
    # Increase lifetime
    lt += 1

    # Helper function to get velocity at any position
    def get_velocity(positions):
        return get_velocity_at_positions(positions, system, interp_u_sum, interp_v_sum, 
                                       grid_u_sum, grid_v_sum, grid_res)

    # Adaptive integration with error control
    total_time = 0.0
    target_total_time = 1.0
    current_pos = pp.copy()
    
    max_steps = 10  # Prevent infinite loops
    step_count = 0
    final_velocity = None  # Store final velocity for accurate coloring
    
    while total_time < target_total_time and step_count < max_steps:
        remaining_time = target_total_time - total_time
        target_dt = min(0.25, remaining_time)  # Start with quarter steps
        
        new_pos, dt_used, error_est, estimated_velocity = adaptive_rk4_step(
            current_pos, target_dt, get_velocity, grid_res)
        
        # Simple error control: accept step if error is reasonable
        if error_est < 0.01 or dt_used < 1e-3:
            # Accept the step
            current_pos = new_pos
            total_time += dt_used
            step_count += 1
            
            # Use the estimated velocity from RK4 intermediate steps
            final_velocity = estimated_velocity
        else:
            # Reduce time step and try again
            target_dt *= 0.5
            if target_dt < 1e-4:
                # Force acceptance with very small step
                current_pos = new_pos
                total_time += dt_used
                step_count += 1
                
                # Use the estimated velocity even for forced steps
                final_velocity = estimated_velocity
    
    pp[:] = current_pos
    
    # Use the current velocity after integration for accurate coloring
    velocity = final_velocity if final_velocity is not None else get_velocity(pp)

    # Shift history
    his[:, :-1, :] = his[:, 1:, :]
    his[:, -1, :] = pp

    # Reinitialize out-of-bounds or over-age particles
    reinitialize_particles(system)

    # Density/divergence-aware reseeding for temporal coherence
    if grid_res is not None:
        density_aware_reseed(system, grid_res)

    # Update visualization
    return update_particle_visualization(system, velocity)