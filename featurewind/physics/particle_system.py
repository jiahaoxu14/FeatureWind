"""
Particle system module for FeatureWind.

This module handles particle physics, including adaptive RK4 integration,
particle reseeding, and trajectory management for flow field visualization.
Enhanced with family-based coloring system.
"""

import numpy as np
from matplotlib.collections import LineCollection
from .. import config


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
    
    # Properly initialize history for all particles
    for i in range(num_particles):
        histories[i, :, :] = particle_positions[i]

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
    Get velocity at particle positions using optimized vectorized interpolation.
    
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
    n_particles = len(positions)
    
    # Pre-allocate velocity array for better memory performance
    velocity = np.zeros((n_particles, 2))
    
    # Vectorized coordinate swapping for interpolation (swap x,y to y,x)
    positions_yx = positions[:, [1, 0]]
    
    # Always prefer smooth bilinear interpolation for consistent motion
    if 'interp_u_sum' in system and 'interp_v_sum' in system:
        # Use updated interpolators from system
        current_interp_u = system['interp_u_sum']
        current_interp_v = system['interp_v_sum']
        # Single vectorized call per component
        velocity[:, 0] = current_interp_u(positions_yx)
        velocity[:, 1] = current_interp_v(positions_yx)
    elif interp_u_sum is not None and interp_v_sum is not None:
        # Top-K mode: use original interpolators  
        # Single vectorized call per component
        velocity[:, 0] = interp_u_sum(positions_yx)
        velocity[:, 1] = interp_v_sum(positions_yx)
    else:
        # Fallback: vectorized grid indexing
        if 'grid_u_sum' in system and 'grid_v_sum' in system:
            current_grid_u = system['grid_u_sum']
            current_grid_v = system['grid_v_sum']
            # Vectorized grid cell index computation
            cell_i_indices = np.clip(((positions[:, 1] - ymin) / (ymax - ymin) * grid_res).astype(int), 0, grid_res - 1)
            cell_j_indices = np.clip(((positions[:, 0] - xmin) / (xmax - xmin) * grid_res).astype(int), 0, grid_res - 1)
            # Vectorized velocity sampling
            velocity[:, 0] = current_grid_u[cell_i_indices, cell_j_indices]
            velocity[:, 1] = current_grid_v[cell_i_indices, cell_j_indices]
    
    # Apply velocity scaling
    velocity *= config.velocity_scale
    
    # Vectorized safety check to prevent runaway particles
    velocity_magnitudes = np.linalg.norm(velocity, axis=1)
    max_safe_velocity = config.MAX_SAFE_VELOCITY
    
    # Vectorized velocity clipping
    exceed_mask = velocity_magnitudes > max_safe_velocity
    if np.any(exceed_mask):
        # Normalize and scale in one operation
        velocity[exceed_mask] *= (max_safe_velocity / velocity_magnitudes[exceed_mask, np.newaxis])
        
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




def reinitialize_particles(system):
    """
    Reinitialize out-of-bounds, over-age, or particles in masked regions.
    Respawn particles in areas with valid flow to avoid immediate re-masking.
    
    Args:
        system (dict): Particle system dictionary
    """
    pp = system['particle_positions']
    lt = system['particle_lifetimes']
    his = system['histories']
    max_lifetime = system['max_lifetime']
    
    xmin, xmax, ymin, ymax = config.bounding_box
    
    # Get dominant features for all particles to detect masked regions
    dominant_features = get_dominant_features_vectorized(pp, system)
    
    # Create a list of valid respawn locations (cells with flow)
    valid_respawn_locations = []
    if 'cell_dominant_features' in system:
        cell_dominant_features = system['cell_dominant_features']
        grid_res = cell_dominant_features.shape[0]
        
        # Find all grid cells that have valid flow (not -1)
        for i in range(grid_res):
            for j in range(grid_res):
                if cell_dominant_features[i, j] != -1:
                    # Convert grid cell to world coordinates (cell center)
                    x_center = xmin + (j + 0.5) / grid_res * (xmax - xmin)
                    y_center = ymin + (i + 0.5) / grid_res * (ymax - ymin)
                    valid_respawn_locations.append([x_center, y_center])
    
    # Fallback: if no valid locations found, create a grid of positions across the domain
    if not valid_respawn_locations:
        # Create a coarse grid of respawn positions across the entire domain
        for i in range(5):
            for j in range(5):
                x_pos = xmin + (j + 0.5) / 5 * (xmax - xmin)
                y_pos = ymin + (i + 0.5) / 5 * (ymax - ymin)
                valid_respawn_locations.append([x_pos, y_pos])
    
    for i in range(len(pp)):
        x, y = pp[i]
        # Check for out-of-bounds or over-age particles
        # Be more lenient with masked regions - only reinitialize if particle is stuck for a while
        in_masked_region = dominant_features[i] == -1
        stuck_in_masked_region = in_masked_region and lt[i] > max_lifetime * 0.3  # Only after 30% of lifetime
        
        if (x < xmin or x > xmax or y < ymin or y > ymax
            or lt[i] > max_lifetime
            or stuck_in_masked_region):
            
            # Choose a random valid location with some jitter
            base_location = valid_respawn_locations[np.random.randint(len(valid_respawn_locations))]
            jitter_x = np.random.uniform(-0.1, 0.1) * (xmax - xmin) / 40  # Small jitter
            jitter_y = np.random.uniform(-0.1, 0.1) * (ymax - ymin) / 40
            
            new_x = np.clip(base_location[0] + jitter_x, xmin, xmax)
            new_y = np.clip(base_location[1] + jitter_y, ymin, ymax)
            
            pp[i] = [new_x, new_y]
            # Fill entire history with the new position
            his[i, :, :] = pp[i]
            lt[i] = 0


def get_dominant_features_vectorized(positions, system):
    """
    Get dominant features for multiple positions using vectorized operations.
    
    Args:
        positions (np.ndarray): Position coordinates, shape (N, 2)
        system (dict): Particle system dictionary
        
    Returns:
        np.ndarray: Dominant feature indices, shape (N,), -1 for invalid positions
    """
    if 'cell_dominant_features' not in system or len(positions) == 0:
        return np.full(len(positions), -1)
        
    xmin, xmax, ymin, ymax = config.bounding_box
    cell_dominant_features = system['cell_dominant_features']
    grid_res = cell_dominant_features.shape[0]
    
    # Vectorized position to grid conversion
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    
    # Check bounds
    in_bounds = ((x_coords >= xmin) & (x_coords <= xmax) & 
                 (y_coords >= ymin) & (y_coords <= ymax))
    
    # Vectorized grid cell computation
    cell_j_indices = np.clip(((x_coords - xmin) / (xmax - xmin) * grid_res).astype(int), 0, grid_res - 1)
    cell_i_indices = np.clip(((y_coords - ymin) / (ymax - ymin) * grid_res).astype(int), 0, grid_res - 1)
    
    # Initialize result with -1 for out-of-bounds positions
    result = np.full(len(positions), -1)
    
    # Sample dominant features for in-bounds positions
    result[in_bounds] = cell_dominant_features[cell_i_indices[in_bounds], cell_j_indices[in_bounds]]
    
    return result


def get_dominant_feature_at_position(position, system):
    """
    Get the dominant feature at a given position (wrapper for backwards compatibility).
    
    Args:
        position (np.ndarray): Position coordinates [x, y]
        system (dict): Particle system dictionary
        
    Returns:
        int: Index of dominant feature, or -1 if none
    """
    return get_dominant_features_vectorized(position.reshape(1, 2), system)[0]




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
    Update particle colors based on family assignment, magnitude, and dominance (optimized).
    
    Args:
        system (dict): Particle system dictionary
        family_assignments (np.ndarray, optional): Family ID for each feature
        feature_colors (list, optional): Hex colors for each feature
        
    Returns:
        np.ndarray: RGBA colors for each particle
    """
    particle_positions = system['particle_positions']
    n_particles = len(particle_positions)
    
    # Pre-allocate colors array
    particle_colors = np.zeros((n_particles, 4))  # RGBA
    
    if family_assignments is None or feature_colors is None:
        # Fallback: vectorized speed-based grayscale coloring
        if 'last_velocity' in system:
            speeds = np.linalg.norm(system['last_velocity'], axis=1)
            max_speed = speeds.max() + 1e-9
            normalized_speeds = speeds / max_speed
            
            # Vectorized intensity calculation
            particle_colors[:, 3] = 0.3 + 0.7 * normalized_speeds  # Alpha only
            # RGB stays 0 for black
        else:
            particle_colors[:] = [0, 0, 0, 0.5]  # Default black with 50% alpha
        
        return particle_colors
    
    # Import color utilities
    from ..visualization.color_system import hex_to_rgb
    
    # Pre-compute maximum magnitudes for each feature (cache for efficiency)
    max_magnitudes = {}
    if 'grid_u_all_feats' in system and 'grid_v_all_feats' in system:
        grid_u_all_feats = system['grid_u_all_feats']
        grid_v_all_feats = system['grid_v_all_feats']
        
        # Vectorized magnitude computation for all features at once
        magnitude_grids = np.sqrt(grid_u_all_feats**2 + grid_v_all_feats**2)
        max_magnitudes = {feat_idx: magnitude_grids[feat_idx].max() 
                         for feat_idx in range(min(len(family_assignments), magnitude_grids.shape[0]))}
    
    # Get dominant features for all particles at once (vectorized)
    velocities = system.get('last_velocity', None)
    
    if velocities is not None:
        # Use vectorized approach for particles with significant velocity
        speeds = np.linalg.norm(velocities, axis=1)
        significant_speed_mask = speeds > 1e-8
        
        # Use magnitude-based dominance for all particles
        dominant_features = get_dominant_features_vectorized(particle_positions, system)
    else:
        # All particles use magnitude-based dominance
        dominant_features = get_dominant_features_vectorized(particle_positions, system)
    
    # Vectorized color assignment
    valid_feature_mask = (dominant_features >= 0) & (dominant_features < len(feature_colors))
    valid_indices = np.where(valid_feature_mask)[0]
    
    if len(valid_indices) > 0:
        # Process valid particles in batches
        valid_features = dominant_features[valid_indices]
        
        for feat_idx in np.unique(valid_features):
            feat_mask = valid_features == feat_idx
            particle_indices = valid_indices[feat_mask]
            
            # Get base color for this feature
            base_color = feature_colors[feat_idx]
            rgb = hex_to_rgb(base_color)
            
            # Vectorized alpha computation
            max_magnitude = max_magnitudes.get(feat_idx, 1.0)
            if max_magnitude > 0:
                # Get magnitudes for these particles (could be optimized further)
                magnitudes = np.array([get_magnitude_at_position(particle_positions[i], feat_idx, system) 
                                     for i in particle_indices])
                magnitude_factors = magnitudes / max_magnitude
            else:
                magnitude_factors = np.ones(len(particle_indices))
            
            # Get dominance values (could be vectorized)
            dominances = np.array([get_dominance_at_position(particle_positions[i], feat_idx, system)
                                 for i in particle_indices])
            
            # Vectorized alpha calculation
            alphas = 0.3 + 0.6 * magnitude_factors * dominances
            alphas = np.clip(alphas, 0.3, 0.9)
            
            # Assign colors
            particle_colors[particle_indices, :3] = rgb
            particle_colors[particle_indices, 3] = alphas
    
    # Handle invalid features: don't immediately mark for respawn, just color them differently
    invalid_mask = ~valid_feature_mask
    if np.any(invalid_mask):
        # Don't immediately mark these particles for respawn - let them try to find valid regions
        # Only mark them for respawn if they've been invalid for a while
        invalid_indices = np.where(invalid_mask)[0]
        for idx in invalid_indices:
            # Only mark for respawn if already near end of lifetime
            if system['particle_lifetimes'][idx] > system['max_lifetime'] * 0.8:
                system['particle_lifetimes'][idx] = system['max_lifetime'] + 1
        
        # For this frame, assign them the color of the nearest valid feature to avoid gray
        if len(feature_colors) > 0:
            # Use the most common feature color as fallback
            if len(valid_indices) > 0:
                # Find the most frequent valid feature
                valid_features = dominant_features[valid_indices]
                most_common_feature = np.bincount(valid_features).argmax()
                fallback_color = feature_colors[most_common_feature]
            else:
                # Use first feature color as fallback
                fallback_color = feature_colors[0]
            
            fallback_rgb = hex_to_rgb(fallback_color)
            particle_colors[invalid_mask, :3] = fallback_rgb
            particle_colors[invalid_mask, 3] = 0.3  # Low alpha to indicate they're being respawned
    
    return particle_colors


def update_particle_visualization(system, velocity, family_assignments=None, feature_colors=None, grad_indices=None):
    """
    Update particle visualization with family-based coloring.
    
    Args:
        system (dict): Particle system dictionary
        velocity (np.ndarray): Current particle velocities
        family_assignments (np.ndarray, optional): Family assignments for features
        feature_colors (list, optional): Hex colors for features
        grad_indices (list, optional): Selected feature indices for single feature mode
        
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

    # Get colors for particles based on mode
    from .. import config
    single_feature_mode = hasattr(config, 'adaptive_velocity_enabled') and hasattr(config, 'k') and config.k == 1
    
    if single_feature_mode:
        # Single feature mode - use uniform color for the single feature
        if (family_assignments is not None and feature_colors is not None and 
            grad_indices is not None and len(grad_indices) > 0 and len(feature_colors) > 0):
            
            # Use the color of the actual selected feature (not just index 0)
            selected_feature_idx = grad_indices[0]  # The actual feature index selected
            if selected_feature_idx < len(feature_colors):
                single_feature_color = feature_colors[selected_feature_idx]
            else:
                # Fallback if index is out of range - use first color
                single_feature_color = feature_colors[0]
            
            # Convert hex to RGB if needed
            if isinstance(single_feature_color, str) and single_feature_color.startswith('#'):
                from ..visualization import color_system
                r, g, b = color_system.hex_to_rgb(single_feature_color)
            else:
                r, g, b = single_feature_color[:3] if len(single_feature_color) >= 3 else (0, 0, 0)
            
            # All particles get the same color with speed-based alpha
            particle_colors = np.zeros((n_active, 4))
            for i in range(n_active):
                intensity = 0.5 + 0.5 * (speeds[i] / max_speed)  # Alpha based on speed
                particle_colors[i] = [r, g, b, intensity]
        else:
            # Fallback to blue for single feature when color data isn't available
            particle_colors = np.zeros((n_active, 4))
            for i in range(n_active):
                intensity = 0.5 + 0.5 * (speeds[i] / max_speed)
                particle_colors[i] = [0.2, 0.4, 0.8, intensity]  # Blue with speed-based alpha
    elif family_assignments is not None and feature_colors is not None:
        # Multi-feature mode - use family-based coloring
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
    
    # Apply adaptive appearance scaling for weak winds
    if hasattr(config, 'velocity_scale_factor'):
        scale_particle_appearance(lc_, config.actual_flow_magnitude, config.velocity_scale_factor)
    
    return (lc_,)


def scale_particle_appearance(line_collection, actual_magnitude, scale_factor):
    """
    Adjust particle visual properties based on wind strength.
    
    Args:
        line_collection: Matplotlib LineCollection
        actual_magnitude: Actual flow magnitude
        scale_factor: Display scaling factor
    """
    if scale_factor > 3.0:
        # Very weak wind - make particles more visible
        line_widths = 2.0  # Thicker lines
        alpha_boost = 1.5  # More opaque
    elif scale_factor > 1.5:
        # Weak wind - slightly enhanced
        line_widths = 1.5
        alpha_boost = 1.2
    else:
        # Normal wind - standard appearance  
        line_widths = 1.0
        alpha_boost = 1.0
    
    line_collection.set_linewidths(line_widths)
    
    # Adjust alpha values
    current_colors = line_collection.get_colors()
    if len(current_colors) > 0:
        adjusted_colors = current_colors.copy()
        adjusted_colors[:, 3] = np.clip(current_colors[:, 3] * alpha_boost, 0, 1)
        line_collection.set_colors(adjusted_colors)


def update_particles_with_families(system, interp_u_sum=None, interp_v_sum=None, 
                                  grid_u_sum=None, grid_v_sum=None, grid_res=None,
                                  family_assignments=None, feature_colors=None, grad_indices=None):
    """
    Enhanced particle update function with family-based coloring.
    
    Args:
        system (dict): Particle system dictionary
        interp_u_sum, interp_v_sum: Velocity interpolators
        grid_u_sum, grid_v_sum: Velocity grids
        grid_res (int): Grid resolution
        family_assignments (np.ndarray): Family assignments for features
        feature_colors (list): Hex colors for features
        grad_indices (list): Selected feature indices for single feature mode
        
    Returns:
        tuple: Updated line collection with family colors
    """
    # First do standard physics update
    result = update_particles(system, interp_u_sum, interp_v_sum, grid_u_sum, grid_v_sum, grid_res)
    
    # Then update visualization with family colors
    if 'last_velocity' in system:
        velocity = system['last_velocity']
        return update_particle_visualization(system, velocity, family_assignments, feature_colors, grad_indices)
    
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
    
    max_steps = 5   # Reduced from 10 for better performance while maintaining RK4 accuracy
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

    # Recalculate velocity after particle repositioning to ensure correct coloring
    velocity = get_velocity(pp)

    # Update visualization with adaptive appearance
    # Extract grad_indices from system if available for single feature mode color alignment
    grad_indices = system.get('grad_indices', None)
    family_assignments = system.get('family_assignments', None) 
    feature_colors = system.get('feature_colors', None)
    
    return update_particle_visualization(system, velocity, family_assignments, feature_colors, grad_indices)