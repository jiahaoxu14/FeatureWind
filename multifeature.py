import sys
sys.path.insert(1, 'funcs')

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba  # For converting color names -> RGBA
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.spatial import cKDTree

import ScalarField
import TangentPoint
import TangentPointSet

# -------------------------------------------------------------------------
# 1) Read data from JSON file and extract tmap & labels
# -------------------------------------------------------------------------
with open("tangentmaps/breast_cancer.tmap", "r") as f:
    data_import = json.loads(f.read())

tmap = data_import['tmap']
Col_labels = data_import['Col_labels']

# -------------------------------------------------------------------------
# 2) Create TangentPoint instances with an initial dummy scale factor
# -------------------------------------------------------------------------
points = []
for tmap_entry in tmap:
    point = TangentPoint.TangentPoint(tmap_entry, 1.0, Col_labels)
    points.append(point)

valid_points = [p for p in points if p.valid]

# Extract all positions and all gradient vectors (for computing scale factor)
all_positions = np.array([p.position for p in valid_points])
all_gradient_vectors = np.vstack([p.gradient_vectors for p in valid_points])

# -------------------------------------------------------------------------
# 3) Compute a global scale factor
# -------------------------------------------------------------------------
gradient_lengths = np.linalg.norm(all_gradient_vectors, axis=1)
max_gradient_length = np.max(gradient_lengths)

x_range = np.max(all_positions[:, 0]) - np.min(all_positions[:, 0])
y_range = np.max(all_positions[:, 1]) - np.min(all_positions[:, 1])
position_range = max(x_range, y_range)

desired_fraction = 0.1  # fraction of bounding box for max gradient
scale_factor = (position_range * desired_fraction) / max_gradient_length
print("Computed scale factor:", scale_factor)

# Update each Point instance with the computed scale factor
for p in valid_points:
    p.update_scale_factor(scale_factor)

# -------------------------------------------------------------------------
# 4) Choose which gradient indices you want to overlay
# -------------------------------------------------------------------------
grad_indices = [3, 10]  # e.g. if your TangentPoints have multiple gradients

# -------------------------------------------------------------------------
# 5) Build data structures for each grad index in a dictionary 'systems'
# -------------------------------------------------------------------------
systems = {}
colors = ['red', 'blue', 'green', 'purple']  # Add more if needed

for i, g_idx in enumerate(grad_indices):
    # Gather positions & vectors for this grad index
    positions_gi = []
    vectors_gi = []
    for p in valid_points:
        if len(p.gradient_vectors) > g_idx:
            positions_gi.append(p.position)
            vectors_gi.append(p.gradient_vectors[g_idx])
    positions_gi = np.array(positions_gi)
    vectors_gi = np.array(vectors_gi)

    # Create grid for interpolation
    xmin, xmax = positions_gi[:, 0].min(), positions_gi[:, 0].max()
    ymin, ymax = positions_gi[:, 1].min(), positions_gi[:, 1].max()
    grid_res = 20
    grid_x, grid_y = np.mgrid[xmin:xmax:complex(grid_res),
                              ymin:ymax:complex(grid_res)]

    # Interpolate onto the grid
    grid_u = griddata(positions_gi, vectors_gi[:, 0], (grid_x, grid_y), method='nearest')
    grid_v = griddata(positions_gi, vectors_gi[:, 1], (grid_x, grid_y), method='nearest')

    # # Replace NaNs -> 0 for safety
    # grid_u[np.isnan(grid_u)] = 0.0
    # grid_v[np.isnan(grid_v)] = 0.0

    # Build a KDTree of your data positions
    kdtree = cKDTree(positions_gi)

    # Flatten grid for a query
    grid_points_2d = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    distances, _ = kdtree.query(grid_points_2d, k=1)

    # Choose some threshold
    threshold = grid_res * 0.15

    # Reshape distances back to grid
    dist_grid = distances.reshape(grid_x.shape)

    # Set values outside the threshold to zero
    mask = (dist_grid > threshold)
    grid_u[mask] = 0.0
    grid_v[mask] = 0.0

    # Build interpolators
    interp_u = RegularGridInterpolator((grid_x[:, 0], grid_y[0, :]),
                                       grid_u, bounds_error=False, fill_value=0.0)
    interp_v = RegularGridInterpolator((grid_x[:, 0], grid_y[0, :]),
                                       grid_v, bounds_error=False, fill_value=0.0)

    # Initialize particles
    num_particles = 2000
    particle_positions = np.column_stack((
        np.random.uniform(xmin, xmax, size=num_particles),
        np.random.uniform(ymin, ymax, size=num_particles)
    ))

    # Keep track of trails
    max_lifetime = 400
    tail_gap = 10
    particle_lifetimes = np.zeros(num_particles, dtype=int)
    histories = np.full((num_particles, tail_gap + 1, 2), np.nan)
    histories[:, :] = particle_positions[:, None, :]

    # Create a Matplotlib LineCollection for the trails
    lc = LineCollection([], linewidths=1.5, zorder=2)

    # (A) Compute the reference max speed for this system, so we can fade each grad_index differently
    #     We'll approximate by the largest magnitude from the original gradient vectors.
    speeds_here = np.linalg.norm(vectors_gi, axis=1)
    speed_max = speeds_here.max() if len(speeds_here) else 1.0

    # Store everything in a dict
    color_name = colors[i % len(colors)]
    systems[g_idx] = {
        'positions': positions_gi,
        'vectors': vectors_gi,
        'interp_u': interp_u,
        'interp_v': interp_v,
        'particle_positions': particle_positions,
        'particle_lifetimes': particle_lifetimes,
        'histories': histories,
        'base_color': color_name,
        'linecoll': lc,
        'xmin': xmin, 'xmax': xmax,
        'ymin': ymin, 'ymax': ymax,
        'tail_gap': tail_gap,
        'max_lifetime': max_lifetime,
        'speed_max': speed_max,  # store max speed for this system
    }

# -------------------------------------------------------------------------
# 6) Prepare a figure and add each system's LineCollection
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

xmin_all = np.min(all_positions[:, 0])
xmax_all = np.max(all_positions[:, 0])
ymin_all = np.min(all_positions[:, 1])
ymax_all = np.max(all_positions[:, 1])

ax.set_xlim(xmin_all, xmax_all)
ax.set_ylim(ymin_all, ymax_all)
ax.set_title("Fading Trails Separately per grad_index")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.axis('equal')

# Plot underlying data points once
ax.scatter(all_positions[:, 0], all_positions[:, 1], c='black', s=5, label='Data Points')

# Convert each system's base color to RGBA
from matplotlib.colors import to_rgba
for g_idx, system in systems.items():
    base_rgba = to_rgba(system['base_color'])
    system['base_rgba'] = base_rgba
    system['linecoll'].set_color(base_rgba)

    # Give each LineCollection a unique label
    system['linecoll'].set_label(Col_labels[g_idx])

    ax.add_collection(system['linecoll'])

ax.legend()
ax.grid(False)

# -------------------------------------------------------------------------
# 7) Define the update function for animation
# -------------------------------------------------------------------------
def update(frame):
    all_lcs = []
    for g_idx, system in systems.items():
        interp_u = system['interp_u']
        interp_v = system['interp_v']
        particle_positions = system['particle_positions']
        particle_lifetimes = system['particle_lifetimes']
        histories = system['histories']
        tail_gap = system['tail_gap']
        max_lifetime = system['max_lifetime']
        xmin, xmax = system['xmin'], system['xmax']
        ymin, ymax = system['ymin'], system['ymax']
        lc = system['linecoll']
        speed_max = system['speed_max']
        base_rgba = system['base_rgba']

        # Increase lifetime
        particle_lifetimes += 1

        # Interpolate velocities
        U = interp_u(particle_positions)
        V = interp_v(particle_positions)
        velocity = np.column_stack((U, V))

        # Move particles
        particle_positions += velocity * 1  # dt=1

        # Shift history
        histories[:, :-1, :] = histories[:, 1:, :]
        histories[:, -1, :] = particle_positions

        # Reinitialize out-of-bounds or over-age particles
        for i in range(len(particle_positions)):
            x, y = particle_positions[i]
            if (x < xmin or x > xmax or y < ymin or y > ymax
                or particle_lifetimes[i] > max_lifetime):
                particle_positions[i] = [
                    np.random.uniform(xmin, xmax),
                    np.random.uniform(ymin, ymax)
                ]
                histories[i] = particle_positions[i]
                particle_lifetimes[i] = 0

        # Randomly reinitialize some fraction each frame
        num_to_reinit = int(0.05 * len(particle_positions))
        if num_to_reinit > 0:
            indices = np.random.choice(len(particle_positions), num_to_reinit, replace=False)
            for idx in indices:
                particle_positions[idx] = [
                    np.random.uniform(xmin, xmax),
                    np.random.uniform(ymin, ymax)
                ]
                histories[idx] = particle_positions[idx]
                particle_lifetimes[idx] = 0

        # Build line segments for faint trails
        n_active = len(particle_positions)
        segments = np.zeros((n_active * tail_gap, 2, 2))
        
        # (B) Create a color array for each segment
        colors_rgba = np.zeros((n_active * tail_gap, 4))

        # Compute speeds (for the fade logic)
        speeds = np.linalg.norm(velocity, axis=1)

        for i in range(n_active):
            # We'll do alpha ~ speed/speed_max for this system
            # so each grad_index can have its own fade curve.
            alpha_speed = speeds[i] / (speed_max + 1e-9)

            # If you want to invert it (slow => higher alpha),
            # do alpha_speed = 1 - alpha_speed, or something else entirely.

            for t in range(tail_gap):
                seg_idx = i * tail_gap + t
                segments[seg_idx, 0, :] = histories[i, t, :]
                segments[seg_idx, 1, :] = histories[i, t + 1, :]

                # (C) Combine alpha for older segments if you like
                # fade out older tail segments:
                # age_factor = (t + 1) / (tail_gap + 1)
                age_factor = 1

                # base_rgba: (r, g, b, 1)
                r, g, b, _ = base_rgba
                alpha_final = alpha_speed * age_factor

                colors_rgba[seg_idx] = [r, g, b, alpha_final]

        # Update the line collection with segment data
        lc.set_segments(segments)
        lc.set_colors(colors_rgba)

        all_lcs.append(lc)

    # Return all line collections so Matplotlib updates them
    return all_lcs

# -------------------------------------------------------------------------
# 8) Create and run the animation
# -------------------------------------------------------------------------
anim = FuncAnimation(fig, update, frames=1000, interval=30, blit=False)
plt.show()