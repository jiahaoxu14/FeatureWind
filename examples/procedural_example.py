import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter, gaussian_filter

from featurewind.TangentPoint import TangentPoint


def PreProcessing(tangentmaps):
    with open(tangentmaps, "r") as f:
        data_import = json.loads(f.read())

    tmap = data_import['tmap']
    Col_labels = data_import['Col_labels']
    points = []
    for tmap_entry in tmap:
        point = TangentPoint(tmap_entry, 1.0, Col_labels)
        points.append(point)
    valid_points = [p for p in points if p.valid]
    all_positions = np.array([p.position for p in valid_points])  # shape: (#points, 2)
    all_grad_vectors = [p.gradient_vectors for p in valid_points]  # list of (#features, 2)
    all_grad_vectors = np.array(all_grad_vectors)                  # shape = (#points, M, 2)

    return valid_points, all_grad_vectors, all_positions, Col_labels


def pick_top_k_features(all_grad_vectors):
    # Compute average magnitude of each feature across all points
    feature_magnitudes = np.linalg.norm(all_grad_vectors, axis=2)  # shape (#points, M)
    avg_magnitudes = feature_magnitudes.mean(axis=0)               # shape (M,)

    # Sort descending, get top k indices
    top_k_indices = np.argsort(-avg_magnitudes)[:k]
    return top_k_indices, avg_magnitudes

def build_grids(positions, grid_res, top_k_indices, all_grad_vectors, kdtree_scale=0.1, output_dir="."):
    # (1) Setup interpolation grid and distance mask
    xmin, xmax, ymin, ymax = bounding_box
    grid_x, grid_y = np.mgrid[xmin:xmax:complex(grid_res),
                                ymin:ymax:complex(grid_res)]
    
    # Build a KD-tree and determine distance to the nearest data point at each grid cell
    kdtree = cKDTree(positions)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    distances, _ = kdtree.query(grid_points, k=1)
    dist_grid = distances.reshape(grid_x.shape)
    threshold = max(abs(xmax - xmin), abs(ymax - ymin)) * kdtree_scale
    print("Distance threshold:", threshold)
    
    # (2) Interpolate velocity fields for the top-k features
    grid_u_feats, grid_v_feats = [], []
    for feat_idx in top_k_indices:
        # Extract the vectors for the given feature.
        vectors = all_grad_vectors[:, feat_idx, :]  # shape: (#points, 2)
        # Interpolate each component onto the grid.
        grid_u = griddata(positions, vectors[:, 0], (grid_x, grid_y), method='nearest')
        grid_v = griddata(positions, vectors[:, 1], (grid_x, grid_y), method='nearest')
        # Mask out grid cells too far from any data.
        mask = dist_grid > threshold
        grid_u[mask] = 0.0
        grid_v[mask] = 0.0
        grid_u_feats.append(grid_u)
        grid_v_feats.append(grid_v)
    grid_u_feats = np.array(grid_u_feats)  # shape: (k, grid_res, grid_res)
    grid_v_feats = np.array(grid_v_feats)  # shape: (k, grid_res, grid_res)
    
    # Create the combined (summed) velocity field for the top-k features.
    grid_u_sum = np.sum(grid_u_feats, axis=0)  # shape: (grid_res, grid_res)
    grid_v_sum = np.sum(grid_v_feats, axis=0)  # shape: (grid_res, grid_res)
    
    # (3) Determine the dominant feature at each grid cell (using Gaussian smoothing)
    # Compute the magnitude of each feature on the grid.
    grid_mag_feats = np.sqrt(grid_u_feats**2 + grid_v_feats**2)  # shape: (k, grid_res, grid_res)
    sigma = 1.0  # standard deviation for smoothing
    grid_mag_feats_gaussian = np.zeros_like(grid_mag_feats)
    for f in range(grid_mag_feats.shape[0]):
        grid_mag_feats_gaussian[f] = gaussian_filter(grid_mag_feats[f], sigma=sigma)

    rel_idx = np.argmax(grid_mag_feats_gaussian, axis=0)
    grid_argmax = np.take(top_k_indices, rel_idx)
    print("Grid argmax shape:", grid_argmax.shape)
    
    # Optionally save the dominant feature grid.
    np.savetxt(os.path.join(output_dir, "grid_argmax.csv"), np.argmax(grid_mag_feats, axis=0), delimiter=",", fmt="%d")
    np.savetxt(os.path.join(output_dir, "grid_argmax_local.csv"), grid_argmax, delimiter=",", fmt="%d")
    
    # (4) Build grid interpolators from the computed fields.
    interp_u_sum = RegularGridInterpolator((grid_x[:, 0], grid_y[0, :]),
                                             grid_u_sum, bounds_error=False, fill_value=0.0)
    interp_v_sum = RegularGridInterpolator((grid_x[:, 0], grid_y[0, :]),
                                             grid_v_sum, bounds_error=False, fill_value=0.0)
    interp_argmax = RegularGridInterpolator((grid_x[:, 0], grid_y[0, :]),
                                             grid_argmax, method='nearest',
                                             bounds_error=False, fill_value=-1)
    
    return interp_u_sum, interp_v_sum, interp_argmax, grid_x, grid_y, grid_u_feats, grid_v_feats

def build_grids_alternative(positions, grid_res, all_grad_vectors, k_local, kdtree_scale=0.1, output_dir="."):
    """
    Alternative grid builder:
      - Divides space with grid_res.
      - For each grid cell, interpolates the velocity component of every feature.
      - Locally selects the top k_local features (by magnitude) and sums their contributions.
    Returns:
      - RegularGridInterpolators for the aggregated u‐ and v‑fields.
      - An integer grid of the locally dominant feature (the one with highest magnitude in that cell).
    """
    # (1) Setup interpolation grid
    xmin, xmax, ymin, ymax = bounding_box  # assuming bounding_box is defined globally
    grid_x, grid_y = np.mgrid[xmin:xmax:complex(grid_res),
                              ymin:ymax:complex(grid_res)]
    
    num_features = all_grad_vectors.shape[1]

     # Build a KD-tree and determine distance to the nearest data point at each grid cell
    kdtree = cKDTree(positions)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    distances, _ = kdtree.query(grid_points, k=1)
    dist_grid = distances.reshape(grid_x.shape)
    threshold = max(abs(xmax - xmin), abs(ymax - ymin)) * kdtree_scale
    print("Distance threshold:", threshold)
    
    # (2) For each feature, interpolate the velocity fields onto the grid.
    grid_u_all, grid_v_all = [], []
    for m in range(num_features):
        vectors = all_grad_vectors[:, m, :]  # shape: (#points, 2)
        grid_u = griddata(positions, vectors[:, 0], (grid_x, grid_y), method='nearest')
        grid_v = griddata(positions, vectors[:, 1], (grid_x, grid_y), method='nearest')
        mask = dist_grid > threshold
        grid_u[mask] = 0.0
        grid_v[mask] = 0.0
        grid_u_all.append(grid_u)
        grid_v_all.append(grid_v)
    grid_u_all = np.array(grid_u_all)  # shape: (M, grid_res, grid_res)
    grid_v_all = np.array(grid_v_all)  # shape: (M, grid_res, grid_res)
    
    # (3) For each grid cell, select the top k_local features based on magnitude and sum their vectors.
    grid_u_sum_local = np.zeros((grid_res, grid_res))
    grid_v_sum_local = np.zeros((grid_res, grid_res))
    # also store the dominant (highest magnitude) feature index per cell for color-coding etc.
    grid_argmax_local = np.zeros((grid_res, grid_res), dtype=int)
    
    for i in range(grid_res):
        for j in range(grid_res):
            # Compute magnitude of each feature at this grid cell.
            mags = np.sqrt(grid_u_all[:, i, j]**2 + grid_v_all[:, i, j]**2)
            # Get indices of the top k_local features (highest magnitude first)
            top_indices = np.argsort(-mags)[:k_local]
            # Sum velocity components for these selected features.
            grid_u_sum_local[i, j] = np.sum(grid_u_all[top_indices, i, j])
            grid_v_sum_local[i, j] = np.sum(grid_v_all[top_indices, i, j])
            grid_argmax_local[i, j] = top_indices[0]  # the most dominant one
    
    # (4) Build RegularGridInterpolators to be used in the animation
    interp_u_sum_local = RegularGridInterpolator((grid_x[:,0], grid_y[0,:]),
                                                 grid_u_sum_local, bounds_error=False, fill_value=0.0)
    interp_v_sum_local = RegularGridInterpolator((grid_x[:,0], grid_y[0,:]),
                                                 grid_v_sum_local, bounds_error=False, fill_value=0.0)
    interp_argmax_local = RegularGridInterpolator((grid_x[:,0], grid_y[0,:]),
                                                  grid_argmax_local, method='nearest', 
                                                  bounds_error=False, fill_value=-1)
    
    # Optionally save local dominant feature grid to CSV.
    np.savetxt(os.path.join(output_dir, "grid_argmax_local_alternative.csv"), grid_argmax_local, delimiter=",", fmt="%d")
    
    return interp_u_sum_local, interp_v_sum_local, interp_argmax_local, grid_argmax_local, grid_x, grid_y

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

def prepare_figure(ax, valid_points, Col_labels, k, grad_indices, feature_colors, lc, all_positions=None, all_grad_vectors=None):
    xmin, xmax, ymin, ymax = bounding_box
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # ax.set_title(f"Top {k} Features Combined - Single Particle System")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    ax.axis('equal')
    plt.xticks([])
    plt.yticks([])

    feature_idx = 2
    domain_values = np.array([p.domain[feature_idx] for p in valid_points])
    domain_min, domain_max = domain_values.min(), domain_values.max()

    # Plot underlying data points once
    # ax.scatter(all_positions[:,0], all_positions[:,1], c='black', s=5, label='Data Points')
    # Collect all labels from valid_points
    unique_labels = sorted(set(p.tmap_label for p in valid_points))
    print("Unique labels:", unique_labels)

    # Define multiple distinct markers (expand this list if you need more)
    markers = ["o", "s", "^", "D", "v", "<", ">"]

    for i, lab in enumerate(unique_labels):
        indices = [j for j, p in enumerate(valid_points) if p.tmap_label == lab]
        positions_lab = np.array([valid_points[j].position for j in indices])
        # alphas = (np.array([valid_points[j].domain[0] for j in indices]) - domain_min) / (domain_max - domain_min + 1e-9)
        normalized = (np.array([valid_points[j].domain[feature_idx] for j in indices]) - domain_min) / (domain_max - domain_min + 1e-9)
        alphas = 0.2 + normalized * 0.8
        # positions_lab = np.array([p.position for p in valid_points if p.tmap_label == lab])
        marker_style = markers[i % len(markers)]
        # colors_lab = [[78/255, 121/255, 167/255, a] for a in alphas]
        # colors_lab = [[225/255, 87/255, 89/255, a] for a in alphas]
        # colors_lab = [[242/255, 142/255, 43/255, a] for a in alphas]
        # "#4E79A7", "#F28E2B", "#E15759"

        ax.scatter(
            positions_lab[:, 0],
            positions_lab[:, 1],
            marker=marker_style,
            color="gray",
            # color = colors_lab,
            # color="white",
            s=10,
            label=f"Label {lab}",
            zorder=4
        )
        

    # Add single line collection
    ax.add_collection(lc)

    # Build a legend or color swatches
    proxy_lines = []
    for j, feat_idx in enumerate(grad_indices):
        lbl = f"Feat {feat_idx} - {Col_labels[feat_idx]}"
        color_j = feature_colors[j]
        proxy = plt.Line2D([0],[0], linestyle="none", marker="o", color=color_j, label=lbl)
        proxy_lines.append(proxy)

    ax.legend(handles=proxy_lines, loc='upper right')
    ax.grid(False)

    # Compute the aggregated arrow for each data point by summing its top-k feature vectors.
    # Note: all_grad_vectors has shape (#points, M, 2) and top_k_indices holds the selected feature indices.
    aggregated_vectors = np.sum(all_grad_vectors[:, grad_indices, :], axis=1)  # shape: (#points, 2)

    # for i, feat_idx in enumerate(grad_indices):
    #     # Extract the color for this feature
    #     color_i = feature_colors[i]
    #     # Draw the individual arrows using quiver.
    #     ax.quiver(
    #         all_positions[:, 0], all_positions[:, 1],
    #         all_grad_vectors[:, feat_idx, 0], all_grad_vectors[:, feat_idx, 1],
    #         color=color_i,  # choose a color that stands out
    #         angles='xy', scale_units='xy', scale=3, width=0.002, headwidth=2, headlength=3, alpha=1,
    #     )

    # Draw the aggregated arrows using quiver.
    # You can adjust the scale parameter to get the desired arrow length.
    # ax.quiver(
    #     all_positions[:, 0], all_positions[:, 1],
    #     aggregated_vectors[:, 0], aggregated_vectors[:, 1],
    #     color='black',  # choose a color that stands out
    #     angles='xy', scale_units='xy', scale=2, width=0.003, headwidth=2, headlength=3, alpha=1.0,
    # )

    return 0

def update(frame, system, interp_u_sum, interp_v_sum, interp_argmax, k, velocity_scale=0.1):
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
    feat_ids = interp_argmax(pp)  # each is in [0..k-1] or -1

    for i in range(n_active):
        this_feat_id = feat_ids[i]
        # Look up the real feature index in our mapping. If not present, assign a default (black).
        if this_feat_id not in real_feature_rgba:
            r, g, b, _ = (0, 0, 0, 1)
            alpha_part = 0.3
        else:
            r, g, b, _ = real_feature_rgba[this_feat_id]
            alpha_part = speeds[i] / max_speed

        for t in range(tail_gap):
            seg_idx = i * tail_gap + t
            segments[seg_idx, 0, :] = his[i, t, :]
            segments[seg_idx, 1, :] = his[i, t + 1, :]

            # Combine alpha with any additional fade for the tail
            # age_factor = 1.0 - (t / tail_gap)
            age_factor = (t+1) / tail_gap
            # age_factor = 1.0

            # Multiply them
            alpha_min = 0.15
            alpha_final = max(alpha_min, alpha_part * age_factor)

            # Assign the final RGBA
            colors_rgba[seg_idx] = [r, g, b, alpha_final]

    lc_.set_segments(segments)
    lc_.set_colors(colors_rgba)

    return (lc_,)


def main():
    # Setup paths relative to repository root
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    tangent_map_path = os.path.join(repo_root, 'tangentmaps', 'onehelix.tmap')
    output_dir = os.path.join(repo_root, 'output')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the tangent map data
    valid_points, all_grad_vectors, all_positions, Col_labels = PreProcessing(tangent_map_path)

    # Set the number of top features to visualize
    global k 
    k = len(Col_labels)
    # k = 5

    # Compute the bounding box
    global bounding_box
    xmin, xmax = all_positions[:,0].min(), all_positions[:,0].max()
    ymin, ymax = all_positions[:,1].min(), all_positions[:,1].max()
    bounding_box = [xmin, xmax, ymin, ymax]

    # Set the velocity scale
    global velocity_scale
    velocity_scale = 0.04

    # Set the grid resolution scale
    grid_res_scale = 0.15

    # feature colors
    global real_feature_rgba

    # Pick top k features based on average magnitude
    top_k_indices, avg_magnitudes = pick_top_k_features(all_grad_vectors)
    print("Top k feature indices:", top_k_indices)
    print("Their average magnitudes:", avg_magnitudes[top_k_indices])
    print("Their labels:", [Col_labels[i] for i in top_k_indices])

    # grad_indices = [2]
    grad_indices = top_k_indices

    # Create a grid for interpolation
    grid_res = (min(abs(xmax - xmin), abs(ymax - ymin)) * grid_res_scale).astype(int)
    grid_res = 15
    print("Grid resolution:", grid_res)

    # Set the KD-tree scale
    # kdtree_scale = 0.01 * grid_res
    kdtree_scale = 0.03

    interp_u_sum, interp_v_sum, interp_argmax, grid_x, grid_y, grid_u_feats, grid_v_feats = build_grids(
        all_positions, grid_res, grad_indices, all_grad_vectors, kdtree_scale=kdtree_scale, output_dir=output_dir
    )

    # interp_u_sum, interp_v_sum, interp_argmax, grid_argmax_local, grid_x, grid_y = build_grids_alternative(
    #     all_positions, grid_res, all_grad_vectors, k_local=len(grad_indices), kdtree_scale=kdtree_scale
    # )
    # grad_indices = np.unique(grid_argmax_local)

    # Create the combined (summed) velocity field for the top-k features.
    grid_u_sum = np.sum(grid_u_feats, axis=0)  # shape: (grid_res, grid_res)
    grid_v_sum = np.sum(grid_v_feats, axis=0)  # shape: (grid_res, grid_res)

    # Define feature colors using a Tableau-like palette.
    tableau_colors = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
        "#59A14F", "#EDC949", "#AF7AA1", "#FF9DA7",
        "#9C755F", "#BAB0AC"
    ]
    # Use as many colors as there are top features.
    feature_colors = [tableau_colors[i % len(tableau_colors)] for i in range(len(grad_indices))]

    # In main() (or after you pick top_k_indices), build a mapping for real indices to RGBA.
    real_feature_rgba = {feat_idx: to_rgba(feature_colors[i])
                        for i, feat_idx in enumerate(grad_indices)}

    # Create the particle system
    num_particles = 2500
    system = create_particles(num_particles)
    lc = system['linecoll']

    # prepare the figure
    fig, ax = plt.subplots(figsize=(8,6))
    prepare_figure(ax, valid_points, Col_labels, k, grad_indices, feature_colors, lc, all_positions, all_grad_vectors)
    
    # # Draw grid lines from the grid arrays
    # n_rows, n_cols = grid_x.shape
    # print("Grid shape:", grid_x.shape)
    # for col in range(n_cols):
    #     ax.plot(grid_x[:, col], grid_y[:, col], color='gray', linestyle='--', linewidth=0.5)
    # for row in range(n_rows):
    #     ax.plot(grid_x[row, :], grid_y[row, :], color='gray', linestyle='--', linewidth=0.5)

    # cell_centers = np.empty((n_rows-1, n_cols-1, 2))
    # for i in range(n_rows - 1):
    #     for j in range(n_cols - 1):
    #         # For each cell, you can choose the center as the mean of the 4 corner coordinates.
    #         cx = (grid_x[i, j] + grid_x[i+1, j+1]) / 2.0
    #         cy = (grid_y[i, j] + grid_y[i+1, j+1]) / 2.0
    #         cell_centers[i, j, :] = [cx, cy]

    # grid_argmax = interp_argmax(cell_centers.reshape(-1, 2)).reshape(n_rows-1, n_cols-1).astype(int)

    # # grid_argmax is of shape (n_rows, n_cols) corresponding to the grid points.
    # # Create a fine mesh grid (increase res_fine for higher resolution Voronoi)
    # res_fine = 100  # for example
    # xx, yy = np.meshgrid(np.linspace(xmin, xmax, res_fine),
    #                     np.linspace(ymin, ymax, res_fine))

    # # Use grid points directly as seeds for the Voronoi diagram.
    # # points_seeds has shape (n_rows*n_cols, 2) from grid_x and grid_y.
    # points_seeds = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    # # Here, interp_argmax returns the dominant feature for any queried coordinate.
    # # Use it on the grid points (the seeds) to get a seed_feature per grid point.
    # seed_features = interp_argmax(points_seeds).astype(int)  # shape: (n_rows*n_cols,)
    # print("Seed features shape:", seed_features.shape)

    # # Create a fine mesh grid (increase res_fine for higher resolution Voronoi)
    # xx, yy = np.meshgrid(np.linspace(xmin, xmax, res_fine),
    #                     np.linspace(ymin, ymax, res_fine))

    # # Build a KD-tree from the grid points and query for the nearest seed for every point in the fine mesh.
    # tree = cKDTree(points_seeds)
    # _, seed_idx = tree.query(np.c_[xx.ravel(), yy.ravel()])
    # seed_idx = seed_idx.reshape(xx.shape)
    # print("Seed indices shape:", seed_idx.shape)

    # # Construct the Voronoi color image: for each fine-grid pixel assign the color of the nearest seed.
    # voronoi_color = np.zeros((res_fine, res_fine, 4))
    # for i in range(res_fine):
    #     for j in range(res_fine):
    #         feat = seed_features[seed_idx[i, j]]
    #         # Lookup the color for this feature from the real_feature_rgba mapping.
    #         voronoi_color[i, j, :] = real_feature_rgba.get(feat, (0, 0, 0, 1))

    # # Overlay the Voronoi background.
    # ax.imshow(voronoi_color, extent=(xmin, xmax, ymin, ymax), origin='lower',
    #         interpolation='none', alpha=0.5)
    
    # # Overlay grid vectors on the figure.
    # for i in grad_indices:
    #     # Extract the color for this feature
    #     color_i = feature_colors[i]
    #     # Draw the individual arrows using quiver.
    #     ax.quiver(
    #         grid_x, grid_y,
    #         grid_u_feats[i], grid_v_feats[i],
    #         color=color_i,  # choose a color that stands out
    #         angles='xy', scale_units='xy', scale=1, width=0.003, headwidth=2, headlength=3, alpha=0.8,
    #     )
    # ax.quiver(
    #     grid_x, grid_y,
    #     grid_u_sum, grid_v_sum,
    #     color="black", angles='xy', scale_units='xy', scale=0.5, width=0.003, headwidth=2, headlength=3, alpha=0.8
    # )

    for spine in ax.spines.values():
        spine.set_visible(False)

    for frame in range(5):
        # Update the system state for this frame.
        update(frame, system, interp_u_sum, interp_v_sum, interp_argmax, k, velocity_scale)
        # Save the current state of the figure.
        fig.savefig(os.path.join(output_dir, f"frame_{frame}.png"), dpi=300)

    # Create the animation
    anim = FuncAnimation(fig, 
                         lambda frame: update(frame, system, interp_u_sum, interp_v_sum, interp_argmax, k, velocity_scale), 
                         frames=1000, interval=30, blit=False)

    # Save the figure as a PNG file with 300 dpi.
    fig.savefig(os.path.join(output_dir, "featurewind_figure.png"), dpi=300)
    plt.show()


if  __name__ == "__main__":
    main()