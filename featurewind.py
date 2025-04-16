import sys
sys.path.insert(1, 'funcs')

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter, gaussian_filter

import TangentPoint


def PreProcessing(tangentmaps):
    with open(tangentmaps, "r") as f:
        data_import = json.loads(f.read())

    tmap = data_import['tmap']
    Col_labels = data_import['Col_labels']
    points = []
    for tmap_entry in tmap:
        point = TangentPoint.TangentPoint(tmap_entry, 1.0, Col_labels)
        points.append(point)
    valid_points = [p for p in points if p.valid]
    all_positions = np.array([p.position for p in valid_points])  # shape: (#points, 2)
    all_grad_vectors = [p.gradient_vectors for p in valid_points]  # list of (#features, 2)
    all_grad_vectors = np.array(all_grad_vectors)                  # shape = (#points, M, 2)

    # Flatten across features to find the global max gradient length
    gradient_lengths = np.linalg.norm(all_grad_vectors.reshape(-1, 2), axis=1)
    max_gradient_length = np.max(gradient_lengths)
    x_range = np.max(all_positions[:, 0]) - np.min(all_positions[:, 0])
    y_range = np.max(all_positions[:, 1]) - np.min(all_positions[:, 1])
    position_range = max(x_range, y_range)
    desired_fraction = 0.1
    scale_factor = (position_range * desired_fraction) / (max_gradient_length + 1e-9)
    print("Computed scale factor:", scale_factor)

    # Update each valid point with the computed scale factor
    for p in valid_points:
        p.update_scale_factor(scale_factor)
    return valid_points, all_grad_vectors, all_positions, Col_labels


def pick_top_k_features(all_grad_vectors):
    # Compute average magnitude of each feature across all points
    feature_magnitudes = np.linalg.norm(all_grad_vectors, axis=2)  # shape (#points, M)
    avg_magnitudes = feature_magnitudes.mean(axis=0)               # shape (M,)

    # Sort descending, get top k indices
    top_k_indices = np.argsort(-avg_magnitudes)[:k]
    return top_k_indices, avg_magnitudes

def build_grids(positions, grid_res, top_k_indices, avg_magnitudes, all_grad_vectors, kdtree_scale=0.1):

    xmin, xmax, ymin, ymax = bounding_box

    grid_x, grid_y = np.mgrid[xmin:xmax:complex(grid_res),
                            ymin:ymax:complex(grid_res)]
    
    # Build a KD-tree from your known positions
    kdtree = cKDTree(positions)

    # Flatten grid for querying distances
    grid_points_2d = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    distances, _ = kdtree.query(grid_points_2d, k=1)
    dist_grid = distances.reshape(grid_x.shape)

    # Choose a distance threshold beyond which we consider data "out of range"
    threshold = max(abs(xmax - xmin), abs(ymax - ymin)) * kdtree_scale
    print("Distance threshold:", threshold)

    grid_u_feats = []
    grid_v_feats = []
    for feat_idx in top_k_indices:
        # gather the (x,y) vectors for this feature
        vectors_j = all_grad_vectors[:, feat_idx, :]  # shape (#points, 2)

        # Interpolate onto the grid
        grid_u = griddata(positions, vectors_j[:,0], (grid_x, grid_y), method='nearest')
        grid_v = griddata(positions, vectors_j[:,1], (grid_x, grid_y), method='nearest')

        # Mask out the grid points that are too far from the data
        mask = dist_grid > threshold
        grid_u[mask] = 0.0
        grid_v[mask] = 0.0

        # Store
        grid_u_feats.append(grid_u)
        grid_v_feats.append(grid_v)

    # Store in arrays
    grid_u_feats_all = []
    grid_v_feats_all = []
    for feat_idx in np.argsort(-avg_magnitudes):
        vectors_j = all_grad_vectors[:, feat_idx, :]  # shape (#points, 2)

        # Interpolate onto the grid
        grid_u = griddata(positions, vectors_j[:, 0], (grid_x, grid_y), method='nearest')
        grid_v = griddata(positions, vectors_j[:, 1], (grid_x, grid_y), method='nearest')

        # Mask out cells too far from data
        mask = dist_grid > threshold
        grid_u[mask] = 0.0
        grid_v[mask] = 0.0

        # Store in arrays
        grid_u_feats_all.append(grid_u)
        grid_v_feats_all.append(grid_v)

    grid_u_feats_all = np.array(grid_u_feats_all)  # shape (M, grid_res, grid_res)
    grid_v_feats_all = np.array(grid_v_feats_all)  # shape (M, grid_res, grid_res)
    grid_u_sum_all = np.sum(grid_u_feats_all, axis=0)  # shape (grid_res, grid_res)
    grid_v_sum_all = np.sum(grid_v_feats_all, axis=0)  # shape (grid_res, grid_res)

    grid_u_feats = np.array(grid_u_feats)  # shape (k, grid_res, grid_res)
    grid_v_feats = np.array(grid_v_feats)  # shape (k, grid_res, grid_res)

    # 5a) Create the **combined** velocity field = sum of the top k features
    grid_u_sum = np.sum(grid_u_feats, axis=0)  # shape (grid_res, grid_res)
    grid_v_sum = np.sum(grid_v_feats, axis=0)  # shape (grid_res, grid_res)
    
    # 5b) For coloring: find the "dominant feature index" at each grid cell
    # We'll compute magnitude of each feature in each cell, pick argmax along axis=0
    grid_mag_feats = np.sqrt(grid_u_feats**2 + grid_v_feats**2)  # shape (k, grid_res, grid_res)
    # grid_mag_feats is shape (k, grid_res, grid_res)
    window_size = 3  # or any odd integer (3, 5, etc.) controlling neighborhood size
    sigma = 1.5  # standard deviation for Gaussian filter
    grid_mag_feats_local = np.zeros_like(grid_mag_feats)
    grid_mag_feats_gaussian = np.zeros_like(grid_mag_feats)

    # Apply a Gaussian filter to smooth the features
    for f in range(grid_mag_feats.shape[0]):  # f in [0..k-1]
        # Apply a maximum filter (local neighborhood of size x size)
        grid_mag_feats_local[f] = maximum_filter(grid_mag_feats[f], size=window_size)
        grid_mag_feats_gaussian[f] = gaussian_filter(grid_mag_feats[f], sigma=sigma)
        
    # Now pick the feature with the largest local maximum at each cell
    grid_argmax = np.argmax(grid_mag_feats_gaussian, axis=0)

    # Optionally save to CSV
    np.savetxt("grid_argmax.csv", np.argmax(grid_mag_feats, axis=0), delimiter=",", fmt="%d")
    np.savetxt("grid_argmax_local.csv", grid_argmax, delimiter=",", fmt="%d")
    # This integer grid tells us which feature is dominant at that (x,y).
    # 5c) Build interpolators
    interp_u_sum = RegularGridInterpolator((grid_x[:,0], grid_y[0,:]),
                                        grid_u_sum, bounds_error=False, fill_value=0.0)
    interp_v_sum = RegularGridInterpolator((grid_x[:,0], grid_y[0,:]),
                                        grid_v_sum, bounds_error=False, fill_value=0.0)
    interp_argmax = RegularGridInterpolator((grid_x[:,0], grid_y[0,:]),
                                            grid_argmax, method='nearest',
                                            bounds_error=False, fill_value=-1)
    # Assign a color to each top feature
    base_colors = ['red','blue','green','magenta','orange','cyan']  # pick as many as you need
    # Use Tableau 10 color palette
    tableau_colors = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
        "#59A14F", "#EDC949", "#AF7AA1", "#FF9DA7",
        "#9C755F", "#BAB0AC"
    ]
    feature_colors = [tableau_colors[i % len(tableau_colors)] for i in range(k)]
    feature_rgba = [to_rgba(c) for c in feature_colors]

    return feature_colors, interp_u_sum, interp_v_sum, interp_argmax

def create_particles(num_particles):
    xmin, xmax, ymin, ymax = bounding_box
    particle_positions = np.column_stack((
        np.random.uniform(xmin, xmax, size=num_particles),
        np.random.uniform(ymin, ymax, size=num_particles)
    ))

    max_lifetime = 400
    tail_gap = 10
    particle_lifetimes = np.zeros(num_particles, dtype=int)
    histories = np.full((num_particles, tail_gap + 1, 2), np.nan)
    histories[:, :] = particle_positions[:, None, :]

    # A single LineCollection for all particles
    lc = LineCollection([], linewidths=1.5, zorder=2)

    # We'll store everything in a dict just for cleanliness
    system = {
        'particle_positions': particle_positions,
        'particle_lifetimes': particle_lifetimes,
        'histories': histories,
        'tail_gap': tail_gap,
        'max_lifetime': max_lifetime,
        'linecoll': lc,
    }

    return system

def prepare_figure(ax, valid_points, Col_labels, k, top_k_indices, feature_colors, lc, all_positions=None, all_grad_vectors=None):
    xmin, xmax, ymin, ymax = bounding_box
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"Top {k} Features Combined - Single Particle System")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis('equal')

    # Plot underlying data points once
    # ax.scatter(all_positions[:,0], all_positions[:,1], c='black', s=5, label='Data Points')
    # Collect all labels from valid_points
    unique_labels = sorted(set(p.tmap_label for p in valid_points))
    print("Unique labels:", unique_labels)

    # Define multiple distinct markers (expand this list if you need more)
    markers = ["o", "s", "D", "^", "v", "<", ">"]

    for i, lab in enumerate(unique_labels):
        # Extract positions belonging to this label
        positions_lab = np.array([p.position for p in valid_points if p.tmap_label == lab])
        marker_style = markers[i % len(markers)]

        ax.scatter(
            positions_lab[:, 0],
            positions_lab[:, 1],
            marker=marker_style,
            color="gray",
            s=10,
            label=f"Label {lab}"
        )

    # Add single line collection
    ax.add_collection(lc)

    # Build a legend or color swatches
    proxy_lines = []
    for j, feat_idx in enumerate(top_k_indices):
        lbl = f"Feat {feat_idx} - {Col_labels[feat_idx]}"
        color_j = feature_colors[j]
        proxy = plt.Line2D([0],[0], linestyle="none", marker="o", color=color_j, label=lbl)
        proxy_lines.append(proxy)

    ax.legend(handles=proxy_lines, loc='upper right')
    ax.grid(False)

    # Compute the aggregated arrow for each data point by summing its top-k feature vectors.
    # Note: all_grad_vectors has shape (#points, M, 2) and top_k_indices holds the selected feature indices.
    aggregated_vectors = np.sum(all_grad_vectors[:, top_k_indices, :], axis=1)  # shape: (#points, 2)

    # Draw the aggregated arrows using quiver.
    # You can adjust the scale parameter to get the desired arrow length.
    ax.quiver(
        all_positions[:, 0], all_positions[:, 1],
        aggregated_vectors[:, 0], aggregated_vectors[:, 1],
        color='gray',  # choose a color that stands out
        angles='xy', scale_units='xy', scale=2.0, width=0.001, headwidth=2, headlength=5, alpha=0,
    )

    return 0

def update(frame, system, interp_u_sum, interp_v_sum, interp_argmax, feature_colors, k, velocity_scale=0.1):
    
    feature_rgba = [to_rgba(c) for c in feature_colors]
    xmin, xmax, ymin, ymax = bounding_box
    pp = system['particle_positions']
    lt = system['particle_lifetimes']
    his = system['histories']
    lc_ = system['linecoll']
    max_lifetime = system['max_lifetime']

    # Increase lifetime
    lt += 1

    # Interpolate velocity
    U = interp_u_sum(pp)
    V = interp_v_sum(pp)
    velocity = np.column_stack((U, V)) * velocity_scale

    # Move particles
    pp += velocity

    # Shift history
    his[:, :-1, :] = his[:, 1:, :]
    his[:, -1, :] = pp

    # Reinitialize out-of-bounds or over-age particles
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

    # Randomly reinitialize some fraction
    num_to_reinit = int(0.05 * len(pp))
    if num_to_reinit > 0:
        idxs = np.random.choice(len(pp), num_to_reinit, replace=False)
        for idx in idxs:
            pp[idx] = [
                np.random.uniform(xmin, xmax),
                np.random.uniform(ymin, ymax)
            ]
            his[idx] = pp[idx]
            lt[idx] = 0

    # Build line segments
    n_active = len(pp)
    tail_gap = system['tail_gap']
    segments = np.zeros((n_active * tail_gap, 2, 2))
    colors_rgba = np.zeros((n_active * tail_gap, 4))

    # Compute speeds for alpha fade
    speeds = np.linalg.norm(velocity, axis=1)
    max_speed = speeds.max() + 1e-9  # avoid division by zero

    # Find dominant feature index for each particle
    feat_ids = interp_argmax(pp).astype(int)  # each is in [0..k-1] or -1

    for i in range(n_active):
        this_feat_id = feat_ids[i]
        # If outside domain => pick a default color
        if this_feat_id < 0 or this_feat_id >= k:
            # e.g. black with a nominal alpha
            r, g, b, _ = (0, 0, 0, 1)
            alpha_part = 0.3
        else:
            r, g, b, _ = feature_rgba[this_feat_id]
            # Scale alpha by speed (change factor as desired)
            alpha_part = speeds[i] / max_speed

        for t in range(tail_gap):
            seg_idx = i * tail_gap + t
            segments[seg_idx, 0, :] = his[i, t, :]
            segments[seg_idx, 1, :] = his[i, t + 1, :]

            # Combine alpha with any additional fade for the tail
            # age_factor = (tail_gap - t) / (tail_gap + 1)
            age_factor = 1.0

            # Multiply them
            alpha_final = alpha_part * age_factor

            # Assign the final RGBA
            colors_rgba[seg_idx] = [r, g, b, alpha_final]

    lc_.set_segments(segments)
    lc_.set_colors(colors_rgba)

    return (lc_,)

def main():
    # Load the tangent map data
    valid_points, all_grad_vectors, all_positions, Col_labels = PreProcessing("tangentmaps/breast_cancer.tmap")

    # Set the number of top features to visualize
    global k 
    k = 5

    # Compute the bounding box
    global bounding_box
    xmin, xmax = all_positions[:,0].min(), all_positions[:,0].max()
    ymin, ymax = all_positions[:,1].min(), all_positions[:,1].max()
    bounding_box = [xmin, xmax, ymin, ymax]

    # Set the velocity scale
    global velocity_scale
    velocity_scale = 0.1

    # Set the grid resolution scale
    grid_res_scale = 0.15

    # Set the KD-tree scale
    kdtree_scale = 0.1

    # Pick top k features based on average magnitude
    top_k_indices, avg_magnitudes = pick_top_k_features(all_grad_vectors)
    print("Top k feature indices:", top_k_indices)
    print("Their average magnitudes:", avg_magnitudes[top_k_indices])
    print("Their labels:", [Col_labels[i] for i in top_k_indices])

    # grad_indices = [0]
    grad_indices = top_k_indices

    # Create a grid for interpolation
    grid_res = (min(abs(xmax - xmin), abs(ymax - ymin)) * grid_res_scale).astype(int)
    # grid_res = 10
    print("Grid resolution:", grid_res)
    feature_colors, interp_u_sum, interp_v_sum, interp_argmax = build_grids(
        all_positions, grid_res, grad_indices, avg_magnitudes, all_grad_vectors, kdtree_scale=kdtree_scale
    )

    # Create the particle system
    system = create_particles(2000)
    lc = system['linecoll']

    # prepare the figure
    fig, ax = plt.subplots(figsize=(8,6))
    prepare_figure(ax, valid_points, Col_labels, k, grad_indices, feature_colors, lc, all_positions, all_grad_vectors)

    # Create the animation
    anim = FuncAnimation(fig, 
                         lambda frame: update(frame, system, interp_u_sum, interp_v_sum, interp_argmax, feature_colors, k, velocity_scale), 
                         frames=1000, interval=30, blit=False)
    plt.show()


if  __name__ == "__main__":
    main()