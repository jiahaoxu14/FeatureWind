"""
Particle system module for FeatureWind.

This module handles particle physics using simple Euler integration,
particle reseeding, and trajectory management for flow field visualization.
Enhanced with family-based coloring system.
"""

import numpy as np
from matplotlib.collections import LineCollection
from .. import config


def create_particles(num_particles=None, cell_dominant_features=None, grid_res=None, system=None):
    """
    Create a particle system for flow visualization with simple random initialization in unmasked cells.
    
    Args:
        num_particles (int, optional): Number of particles to create
        cell_dominant_features (np.ndarray, optional): Dominant features per grid cell
        grid_res (int, optional): Grid resolution
        system (dict, optional): Existing system with velocity fields for validation
        
    Returns:
        dict: Particle system dictionary with positions, lifetimes, histories, etc.
    """
    if num_particles is None:
        num_particles = config.DEFAULT_NUM_PARTICLES
        
    xmin, xmax, ymin, ymax = config.bounding_box
    
    # Helper: random valid position in unmasked cells only
    def get_random_valid_position():
        max_attempts = 50
        for _ in range(max_attempts):
            test_x = np.random.uniform(xmin, xmax)
            test_y = np.random.uniform(ymin, ymax)
            if cell_dominant_features is not None and grid_res is not None:
                cell_j = int((test_x - xmin) / (xmax - xmin) * grid_res)
                cell_i = int((test_y - ymin) / (ymax - ymin) * grid_res)
                cell_i = max(0, min(grid_res - 1, cell_i))
                cell_j = max(0, min(grid_res - 1, cell_j))
                if cell_dominant_features[cell_i, cell_j] != -1:
                    return test_x, test_y
            else:
                # If no mask provided, accept any random position
                return test_x, test_y
        # Fallback to center if no unmasked cell found quickly
        return (xmin + xmax) / 2, (ymin + ymax) / 2
    
    # Initialize particles in valid (unmasked) positions from the start
    particle_positions = np.zeros((num_particles, 2))
    for i in range(num_particles):
        particle_positions[i] = get_random_valid_position()

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
    system_dict = {
        'particle_positions': particle_positions,
        'particle_lifetimes': particle_lifetimes,
        'histories': histories,
        'tail_gap': tail_gap,
        'max_lifetime': max_lifetime,
        'linecoll': lc,
    }
    
    # Copy over existing system data if provided
    if system:
        for key, value in system.items():
            if key not in system_dict:
                system_dict[key] = value

    return system_dict


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


def euler_step(pos, dt, get_vel_func):
    """
    Simple Euler integration step.
    
    Args:
        pos (np.ndarray): Current positions
        dt (float): Time step
        get_vel_func (callable): Function to get velocity at positions
        
    Returns:
        tuple: (new_position, velocity)
    """
    # Get velocity at current position
    velocity = get_vel_func(pos)
    
    # Euler integration: new_pos = pos + velocity * dt
    new_pos = pos + velocity * dt
    
    return new_pos, velocity






def reinitialize_particles(system):
    """
    Immediately reinitialize out-of-bounds, over-age, or particles in masked regions.
    Uses random uniform distribution across all unmasked areas for even coverage.
    
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
    
    # Create a uniform random respawn function for unmasked areas (mask-only)
    def get_random_valid_position():
        """Respawn by sampling a random unmasked cell and a random point within it."""
        if 'cell_dominant_features' in system:
            cdf = system['cell_dominant_features']
            grid_res_local = cdf.shape[0]
            unmasked = np.argwhere(cdf != -1)
            if unmasked.size > 0:
                # Pick a random unmasked cell (i, j)
                i, j = unmasked[np.random.randint(0, len(unmasked))]
                dx = (xmax - xmin) / grid_res_local
                dy = (ymax - ymin) / grid_res_local
                # Sample uniformly inside that cell
                x = xmin + j * dx + np.random.rand() * dx
                y = ymin + i * dy + np.random.rand() * dy
                return x, y
        # Fallback: uniform anywhere in domain
        return np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)
    
    for i in range(len(pp)):
        x, y = pp[i]
        
        # Check if particle is out of bounds (immediate respawn)
        out_of_bounds = (x < xmin or x > xmax or y < ymin or y > ymax)
        
        # Check if particle is over age (immediate respawn)
        over_age = lt[i] > max_lifetime
        
        # Check if particle is in a masked region (immediate respawn)
        in_masked_region = False
        if 'interp_u_sum' in system and 'interp_v_sum' in system:
            try:
                interp_u_sum = system['interp_u_sum']
                interp_v_sum = system['interp_v_sum']
                test_pos = np.array([[x, y]])
                # RegularGridInterpolator expects (y, x)
                test_pos_yx = test_pos[:, [1, 0]]
                u_val = interp_u_sum(test_pos_yx)[0]
                v_val = interp_v_sum(test_pos_yx)[0]
                sum_magnitude = np.sqrt(u_val**2 + v_val**2)
                in_masked_region = sum_magnitude <= 1e-6
            except:
                # Fallback to dominant features
                in_masked_region = dominant_features[i] == -1
        else:
            # Fallback to dominant features
            in_masked_region = dominant_features[i] == -1
        
        # Immediately respawn particles that are out of bounds, over age, or in masked regions
        if out_of_bounds or over_age or in_masked_region:
            # Get a random position in an unmasked area
            new_x, new_y = get_random_valid_position()
            
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
    
    # Use magnitude-based dominance for all particles
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

    # Get colors for particles based on dominant feature of the cell they occupy
    if family_assignments is not None and feature_colors is not None:
        # Use family-based coloring regardless of single/multi-feature selection
        particle_colors = update_particle_colors_family_based(system, family_assignments, feature_colors)
    else:
        # Fallback: speed-based grayscale coloring
        particle_colors = np.zeros((n_active, 4))
        for i in range(n_active):
            intensity = 0.3 + 0.7 * (speeds[i] / max_speed)
            particle_colors[i] = [0, 0, 0, intensity]

    # Build trail segments with per-segment color based on the dominant feature
    # Use segment end-point positions (his[:, 1:, :]) to determine the cell color
    positions_for_color = his[:, 1:, :].reshape(-1, 2)
    dom_feats_for_segments = get_dominant_features_vectorized(positions_for_color, system)

    # Prepare feature index -> RGB map
    feature_rgb_map = {}
    if feature_colors is not None:
        from ..visualization.color_system import hex_to_rgb
        for feat_idx in range(min(len(feature_colors), (system.get('grid_u_all_feats', np.zeros((0,))).shape[0] if isinstance(system.get('grid_u_all_feats', None), np.ndarray) else len(feature_colors)))):
            base_color = feature_colors[feat_idx]
            feature_rgb_map[feat_idx] = hex_to_rgb(base_color) if isinstance(base_color, str) and base_color.startswith('#') else (base_color[0], base_color[1], base_color[2])

    # Fill segments and colors
    for i in range(n_active):
        # Alpha based only on normalized speed (no age factor)
        speed_alpha = speeds[i] / max_speed
        alpha_value = np.clip(0.3 + 0.7 * speed_alpha, 0.0, 1.0)

        for t in range(tail_gap):
            seg_idx = i * tail_gap + t
            segments[seg_idx, 0, :] = his[i, t, :]
            segments[seg_idx, 1, :] = his[i, t + 1, :]

            # Determine color from segment end-point dominant feature
            dom_feat = dom_feats_for_segments[seg_idx] if seg_idx < len(dom_feats_for_segments) else -1
            if dom_feat in feature_rgb_map:
                r, g, b = feature_rgb_map[dom_feat]
            else:
                r, g, b = 0.0, 0.0, 0.0
            colors_rgba[seg_idx] = [r, g, b, alpha_value]

    lc_.set_segments(segments)
    lc_.set_colors(colors_rgba)
    
    # Adaptive appearance scaling removed
    
    return (lc_,)


# Removed: scale_particle_appearance (adaptive visual scaling)


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

    # Simple Euler integration with fixed time step
    dt = 1.0  # Fixed time step for simplicity
    new_pos, velocity = euler_step(pp, dt, get_velocity)
    
    # Update particle positions
    pp[:] = new_pos
    

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
