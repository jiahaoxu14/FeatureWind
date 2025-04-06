import sys
sys.path.insert(1, 'funcs')  # Ensure local modules in 'funcs' folder can be imported

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata, RegularGridInterpolator

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

# -------------------------------------------------------------------------
# 3) Collect gradient vectors and positions from valid Point instances
# -------------------------------------------------------------------------
valid_points = [point for point in points if point.valid]
all_positions = np.array([point.position for point in valid_points])
all_gradient_vectors = np.vstack([point.gradient_vectors for point in valid_points])

# -------------------------------------------------------------------------
# 4) Compute a scale factor based on the maximum gradient length
#    and the overall position range
# -------------------------------------------------------------------------
gradient_lengths = np.linalg.norm(all_gradient_vectors, axis=1)
max_gradient_length = np.max(gradient_lengths)

x_range = np.max(all_positions[:, 0]) - np.min(all_positions[:, 0])
y_range = np.max(all_positions[:, 1]) - np.min(all_positions[:, 1])
position_range = max(x_range, y_range)

desired_fraction = 0.1  # fraction of the bounding box for max gradient
scale_factor = (position_range * desired_fraction) / max_gradient_length
print("Computed scale factor:", scale_factor)

# Update each Point instance with the computed scale factor
for point in valid_points:
    point.update_scale_factor(scale_factor)

# -------------------------------------------------------------------------
# 5) Extract one gradient index (e.g., 0) from all points
# -------------------------------------------------------------------------
grad_index = 0
positions = []
vectors = []

for point in valid_points:
    if len(point.gradient_vectors) > grad_index:
        positions.append(point.position)
        vectors.append(point.gradient_vectors[grad_index])

positions = np.array(positions)
vectors = np.array(vectors)

# -------------------------------------------------------------------------
# 6) Build a grid over which to interpolate U and V components
# -------------------------------------------------------------------------
xmin, xmax = positions[:, 0].min(), positions[:, 0].max()
ymin, ymax = positions[:, 1].min(), positions[:, 1].max()
grid_res = 20

grid_x, grid_y = np.mgrid[xmin:xmax:complex(grid_res), ymin:ymax:complex(grid_res)]
grid_u = griddata(positions, vectors[:, 0], (grid_x, grid_y), method='linear')
grid_v = griddata(positions, vectors[:, 1], (grid_x, grid_y), method='linear')

# Build interpolators for U and V, used for streaming particles
interp_u = RegularGridInterpolator((grid_x[:, 0], grid_y[0, :]), grid_u,
                                   bounds_error=False, fill_value=0.0)
interp_v = RegularGridInterpolator((grid_x[:, 0], grid_y[0, :]), grid_v,
                                   bounds_error=False, fill_value=0.0)

# -------------------------------------------------------------------------
# 7) Initialize particle positions randomly within bounding box
# -------------------------------------------------------------------------
num_particles = 4000
particle_positions = np.column_stack((
    np.random.uniform(xmin, xmax, size=num_particles),
    np.random.uniform(ymin, ymax, size=num_particles)
))
prev_positions = np.copy(particle_positions)  # store old positions if needed

# -------------------------------------------------------------------------
# 8) Setup matplotlib Figure
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Create a LineCollection for streaming line segments
lc_long = LineCollection([], linewidths=1.5, zorder=2)
ax.add_collection(lc_long)

# Scatter the underlying data points
ax.scatter(positions[:, 0], positions[:, 1], c='black', s=5, label='Data Points')

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_title("Particle Flow Visualization")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.axis('equal')
ax.legend()
ax.grid(False)

# Precompute max speed for optional coloring or normalization
all_speeds = np.linalg.norm(vectors, axis=1)
speed_max = all_speeds.max()

# -------------------------------------------------------------------------
# 9) Define a function to interpolate the vector field at arbitrary points
# -------------------------------------------------------------------------
def interpolate_vector_field(xy_points, known_positions, known_vectors):
    """ Interpolates the vector field (U, V) at the given xy_points. """
    U = griddata(known_positions, known_vectors[:, 0], xy_points,
                 method='linear', fill_value=0)
    V = griddata(known_positions, known_vectors[:, 1], xy_points,
                 method='linear', fill_value=0)
    return np.column_stack((U, V))

# -------------------------------------------------------------------------
# 10) Define animation update function
# -------------------------------------------------------------------------

# Define maximum lifetime for particles
max_lifetime = 500
particle_lifetimes = np.zeros(num_particles, dtype=int)

tail_gap = 10
histories = np.full((num_particles, tail_gap + 1, 2), np.nan)
histories[:, :] = particle_positions[:, None, :]

def reinitialize_particle(i):
    """Reinitialize particle i at a random position and reset history & lifetime."""
    particle_positions[i] = [
        np.random.uniform(xmin, xmax),
        np.random.uniform(ymin, ymax)
    ]
    histories[i] = particle_positions[i]  # reset the entire trail to this new position
    particle_lifetimes[i] = 0

def update(frame):
    global particle_positions, histories, segments, particle_lifetimes

    # Increase lifetime counter for all particles
    particle_lifetimes += 1

    # 1) Interpolate velocity:
    velocity = interpolate_vector_field(particle_positions, positions, vectors)
    speeds = np.linalg.norm(velocity, axis=1)

    # 2) Update positions:
    particle_positions += velocity * 1

    # 3) Shift older positions in history:
    histories[:, :-1, :] = histories[:, 1:, :]
    histories[:, -1, :] = particle_positions

    # 4) Reinitialize particles that go out of bounds or exceed max_lifetime:
    for i in range(len(particle_positions)):
        if (particle_positions[i, 0] < xmin or particle_positions[i, 0] > xmax or
            particle_positions[i, 1] < ymin or particle_positions[i, 1] > ymax or
            particle_lifetimes[i] > max_lifetime):
            reinitialize_particle(i)

    # Randomly reinitialize a small fraction each frame for progressive renewal
    num_to_reinit = int(0.1 * len(particle_positions))
    if num_to_reinit > 0:
        indices_to_reinit = np.random.choice(len(particle_positions), num_to_reinit, replace=False)
        for i in indices_to_reinit:
            reinitialize_particle(i)

    # 5) Build line segments for faint trails:
    n_active = len(particle_positions)
    trail_segments = np.zeros((n_active * tail_gap, 2, 2), dtype=float)

    for i in range(n_active):
        for t in range(tail_gap):
            seg_idx = i * tail_gap + t
            trail_segments[seg_idx, 0, :] = histories[i, t, :]
            trail_segments[seg_idx, 1, :] = histories[i, t + 1, :]

    # 6) Assign colors with a fade based on segment age + speed:
    trail_colors_rgba = np.zeros((n_active * tail_gap, 4))
    speed_alpha = speeds / (speeds.max() + 1e-9)

    for i in range(n_active):
        for t in range(tail_gap):
            seg_idx = i * tail_gap + t
            # Fade older segments more
            segment_age_factor = (t + 1) / (tail_gap + 1)
            trail_colors_rgba[seg_idx] = [1.0, 0.0, 0.0,
                                          speed_alpha[i] * segment_age_factor]

    lc_long.set_segments(trail_segments)
    lc_long.set_colors(trail_colors_rgba)

    return (lc_long,)

# -------------------------------------------------------------------------
# 11) Create and run the animation
# -------------------------------------------------------------------------
anim = FuncAnimation(fig, update, frames=2000, interval=10, blit=False)
plt.show()