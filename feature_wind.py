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
from scipy.ndimage import gaussian_filter
from typing import Optional

import TangentPoint

TABLEAU_COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC949", "#AF7AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]


class FeatureWind:
    def __init__(self,
                 tangentmap_path,
                 number_of_features,
                 grid_size=100,
                 velocity_scale=0.01,
                 kdtree_scale=0.1,
                 number_of_particles=2000,
                 figure_size=(10, 10)):
        
        self.tangentmap_path = tangentmap_path
        self.number_of_features = number_of_features
        self.grid_size = grid_size
        self.velocity_scale = velocity_scale
        self.kdtree_scale = kdtree_scale
        self.number_of_particles = number_of_particles
        self.figure_size = figure_size

        self.preprocess()
        self.bounding_box = [self.all_positions[:, 0].min(), self.all_positions[:, 0].max(),
                            self.all_positions[:, 1].min(), self.all_positions[:, 1].max()]
        
        self.grid_size = grid_size

        self.pick_features()
        self.build_grids()
        self.init_particles()
        self.prepare_figure()
        
    def preprocess(self):
        with open(self.tangentmap_path, 'r') as f:
            data = json.load(f)
        
        tmap = data['tmap']
        Col_labels = data['Col_labels']
        points = [TangentPoint.TangentPoint(entry, 1.0, Col_labels) for entry in tmap]
        self.valid_points = [p for p in points if p.valid]
        self.all_positions = np.array([p.position for p in self.valid_points])
        self.all_grad_vectors = np.array([p.gradient_vectors for p in self.valid_points])
        self.Col_labels = Col_labels

    def pick_features(self):
        magnitudes = np.linalg.norm(self.all_grad_vectors, axis=2)
        avg_magnitudes = magnitudes.mean(axis=0)

        self.top_indices = np.argsort(-avg_magnitudes)[:self.number_of_features]
        self.feature_colors = [TABLEAU_COLORS[i % len(TABLEAU_COLORS)] for i in range(self.number_of_features)]
        self.real_feature_rgba = {i: to_rgba(self.feature_colors[i], alpha=0.5) for i in range(self.number_of_features)}

    def build_grids(self):
        xmin, xmax, ymin, ymax = self.bounding_box
        grid_x, grid_y = np.mgrid[xmin:xmax:self.grid_size*1j, 
                                  ymin:ymax:self.grid_size*1j]
        
        tree = cKDTree(self.all_positions)
        distances, indices = tree.query(np.column_stack((grid_x.ravel(), grid_y.ravel())), 
                                         k=self.kdtree_scale)
        dist_grid = distances.reshape(grid_x.shape)
        threshold = max(abs(xmax - xmin), abs(ymax - ymin)) * self.kdtree_scale

        grid_u_feats, grid_v_feats = [], []
        for feat_idx in self.top_indices:
            vectors = self.all_grad_vectors[:, feat_idx, :]
            grid_u = griddata(self.all_positions, vectors[:, 0], (grid_x, grid_y), method='linear')
            grid_v = griddata(self.all_positions, vectors[:, 1], (grid_x, grid_y), method='linear')

            mask = dist_grid > threshold
            grid_u[mask] = 0.0
            grid_v[mask] = 0.0
            grid_u_feats.append(grid_u)
            grid_v_feats.append(grid_v)
        
        self.grid_u_feats = np.array(grid_u_feats)  # shape: (k, grid_res, grid_res)
        self.grid_v_feats = np.array(grid_v_feats)  # shape: (k, grid_res, grid_res)
        grid_u_sum = np.sum(self.grid_u_feats, axis=0)
        grid_v_sum = np.sum(self.grid_v_feats, axis=0)

        grid_mag = np.sqrt(self.grid_u_feats**2 + self.grid_v_feats**2)
        grid_mag_smooth = np.zeros_like(grid_mag)
        for f in range(grid_mag.shape[0]):
            grid_mag_smooth[f] = gaussian_filter(grid_mag[f], sigma=1.0)

        rel_idx = np.argmax(grid_mag_smooth, axis=0)
        grid_argmax = np.take(self.top_indices, rel_idx)

        self.interp_u_sum = RegularGridInterpolator((grid_x[:, 0], 
                                                     grid_y[0, :]), 
                                                     grid_u_sum,
                                                     bounds_error=False, fill_value=0.0)
        self.interp_v_sum = RegularGridInterpolator((grid_x[:, 0], 
                                                     grid_y[0, :]), 
                                                     grid_v_sum,
                                                     bounds_error=False, fill_value=0.0)
        self.interp_argmax = RegularGridInterpolator((grid_x[:, 0], 
                                                      grid_y[0, :]), 
                                                      grid_argmax,
                                                      method='nearest',
                                                      bounds_error=False, fill_value=0.0)
    
    def init_particles(self):
        xmin, xmax, ymin, ymax = self.bounding_box
        particle_positions = np.column_stack((
            np.random.uniform(xmin, xmax, size=self.number_of_particles),
            np.random.uniform(ymin, ymax, size=self.number_of_particles)
        ))

        tail_gap = 10
        self.system = dict(
            particle_positions=particle_positions,
            particle_lifetimes=np.zeros(self.number_of_particles, dtype=int),
            histories=np.broadcast_to(particle_positions[:,None,:], 
                                      (self.number_of_particles, tail_gap + 1, 2)).copy(),
                                      tail_gap=tail_gap,
                                      max_lifetime=400,
                                      linecoll = LineCollection([], linewidths=1.5, zorder=2),
                                      arrow_patches=[]
        )

    def prepare_figure(self):
        self.fig, self.ax = plt.subplots(figsize=self.figure_size)
        xmin, xmax, ymin, ymax = self.bounding_box
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.axis('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.grid(False)

        unique_labels = sorted(set(p.tmap_label for p in self.valid_points))

        markers = ['o', 's', '^', 'D', 'v']
        for i, lab in enumerate(unique_labels):
            pts = np.array([p.position for p in self.valid_points if p.tmap_label == lab])
            self.ax.scatter(pts[:, 0], pts[:, 1], s=5, color="gray", 
                            marker=markers[i % len(markers)], label=lab, alpha=0.5)
            
        self.ax.add_collection(self.system['linecoll'])

        proxies = [plt.Line2D([0], [0], color=TABLEAU_COLORS[i % len(TABLEAU_COLORS)],
                              marker=markers[i % len(markers)], linestyle='None', markersize=5) 
                   for i, lab in enumerate(unique_labels)]
        self.ax.legend(proxies, unique_labels, loc='upper right', fontsize=8, frameon=False)

    def update(self,frame):
        xmin, xmax, ymin, ymax = self.bounding_box
        pp = self.system["particle_positions"]
        lt = self.system["particle_lifetimes"]
        his = self.system["histories"]
        lc = self.system["linecoll"]
        max_lifetime = self.system["max_lifetime"]
        tail_gap = self.system["tail_gap"]

        # advance lifetime
        lt += 1

        # advect
        vel = np.column_stack((self.interp_u_sum(pp),
                               self.interp_v_sum(pp))) * self.velocity_scale
        pp += vel

        # shift history
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

        # Reinitialize a small fraction of particles randomly
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

        # build line segments + colours
        n = len(pp)
        segments = np.zeros((n * tail_gap, 2, 2))
        colours = np.zeros((n * tail_gap, 4))

        speeds = np.linalg.norm(vel, axis=1)
        max_speed = speeds.max() + 1e-9
        feat_ids = self.interp_argmax(pp)

        for i in range(n):
            this_feat_id = feat_ids[i]
            # Look up the real feature index in our mapping. If not present, assign a default (black).
            if this_feat_id not in self.real_feature_rgba:
                alpha_part = 0
            else:
                alpha_part = speeds[i] / max_speed
                alpha_part = 1.0

            rgba = self.real_feature_rgba.get(feat_ids[i], (0, 0, 0, 1))
            # alpha_part = 1.0
            for t in range(tail_gap):
                idx = i * tail_gap + t
                segments[idx, 0] = his[i, t]
                segments[idx, 1] = his[i, t + 1]
                colours[idx] = (*rgba[:3], alpha_part)

        lc.set_segments(segments)
        lc.set_colors(colours)

        # # --- arrow heads ---
        # for patch in self.system["arrow_patches"]:
        #     patch.remove()
        # self.system["arrow_patches"].clear()
        # for i in range(n):
        #     start = his[i, -2]
        #     end = his[i, -1]
        #     dx, dy = end - start
        #     arrow = self.ax.arrow(start[0], start[1], dx, dy,
        #                           head_width=0.1, head_length=0.15,
        #                           fc="k", ec="k", zorder=7)
        #     self.system["arrow_patches"].append(arrow)

        return (lc,)
    
    def animate(self, frames: int = 1000, interval: int = 30,
                blit: bool = False, save: bool = False,
                save_path: Optional[str] = None):
        """Run the animation. If *save* is True, an MP4 is written."""
        anim = FuncAnimation(self.fig, self.update,
                             frames=frames, interval=interval, blit=blit)
        if save:
            if save_path is None:
                save_path = "feature_wind.mp4"
            anim.save(save_path, dpi=150, fps=1000 // interval,
                      codec="libx264")
        return anim
    

if __name__ == "__main__":
    fw = FeatureWind("tangentmaps/tworings.tmap", 
                     number_of_features=3, 
                     kdtree_scale=0.03, 
                     velocity_scale=0.01,
                     grid_size=20,
                     number_of_particles=2000,
                     figure_size=(10, 8))
    anim = fw.animate(frames=1000, interval=30)
    plt.show()