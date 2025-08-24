"""
Particle system module for FeatureWind.

This module handles particle physics, including adaptive RK4 integration,
particle reseeding, and trajectory management for flow field visualization.
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


def update_particle_visualization(system, velocity):
    """
    Update particle visualization (line segments and colors).
    
    Args:
        system (dict): Particle system dictionary
        velocity (np.ndarray): Current particle velocities
        
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

    # Use black color for all particles
    for i in range(n_active):
        # All particles are black regardless of grid cell or feature
        r, g, b = 0, 0, 0  # Black color
        alpha_part = speeds[i] / max_speed

        for t in range(tail_gap):
            seg_idx = i * tail_gap + t
            segments[seg_idx, 0, :] = his[i, t, :]
            segments[seg_idx, 1, :] = his[i, t + 1, :]

            # Combine alpha with any additional fade for the tail
            age_factor = (t+1) / tail_gap

            # Multiply them
            alpha_min = 0.15
            alpha_final = max(alpha_min, alpha_part * age_factor * config.ALPHA_FADE_FACTOR)

            # Assign the final RGBA
            colors_rgba[seg_idx] = [r, g, b, alpha_final]

    lc_.set_segments(segments)
    lc_.set_colors(colors_rgba)
    
    return (lc_,)


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