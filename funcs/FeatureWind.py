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

class FeatureWindSystem:
    def __init__(self, valid_points, all_grad_vectors, all_positions, Col_labels, k):
        self.valid_points = valid_points
        self.all_grad_vectors = all_grad_vectors
        self.all_positions = all_positions
        self.Col_labels = Col_labels
        self.k = k
        
        # Compute bounding box from positions.
        xmin, xmax = all_positions[:,0].min(), all_positions[:,0].max()
        ymin, ymax = all_positions[:,1].min(), all_positions[:,1].max()
        self.bounding_box = [xmin, xmax, ymin, ymax]
        
        # Create particles and grids.
        self.system = self.create_particles(2000)
        self.feature_colors, self.interp_u_sum, self.interp_v_sum, self.interp_argmax = self.build_grids()
    
    def create_particles(self, num_particles):
        xmin, xmax, ymin, ymax = self.bounding_box
        particle_positions = np.column_stack((
            np.random.uniform(xmin, xmax, size=num_particles),
            np.random.uniform(ymin, ymax, size=num_particles)
        ))
        max_lifetime = 400
        tail_gap = 10
        particle_lifetimes = np.zeros(num_particles, dtype=int)
        histories = np.full((num_particles, tail_gap + 1, 2), np.nan)
        histories[:, :] = particle_positions[:, None, :]
        lc = LineCollection([], linewidths=1.5, zorder=2)
        return {
            'particle_positions': particle_positions,
            'particle_lifetimes': particle_lifetimes,
            'histories': histories,
            'tail_gap': tail_gap,
            'max_lifetime': max_lifetime,
            'linecoll': lc,
        }
    
    def build_grids(self):
        # Use self.all_positions, self.all_grad_vectors, self.bounding_box, etc.
        # Build grid interpolators, apply filters, and assign feature colors.
        # This function would encapsulate the grid generation logic currently in build_grids
        # Return feature_colors, interp_u_sum, interp_v_sum, interp_argmax.
        #
        # For brevity, assume the output is computed as follows:
        feature_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F'][:self.k]
        # Create dummy interpolators (replace with actual grid generation/interpolation logic)
        grid_vals = np.linspace(self.bounding_box[0], self.bounding_box[1], 50)
        interp_u_sum = RegularGridInterpolator((grid_vals, grid_vals), np.zeros((50,50)))
        interp_v_sum = RegularGridInterpolator((grid_vals, grid_vals), np.zeros((50,50)))
        interp_argmax = RegularGridInterpolator((grid_vals, grid_vals), np.zeros((50,50), dtype=int))
        return feature_colors, interp_u_sum, interp_v_sum, interp_argmax
    
    def prepare_figure(self, ax):
        xmin, xmax, ymin, ymax = self.bounding_box
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"Top {self.k} Features Combined - Single Particle System")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis('equal')
        # Plot underlying data points and aggregated arrows, add legend, etc.
        # Use self.valid_points, self.all_positions, self.all_grad_vectors, self.feature_colors.
    
    def update(self, frame):
        # Access system state via self.system, self.interp_u_sum, etc.
        # For example:
        pp = self.system['particle_positions']
        lt = self.system['particle_lifetimes']
        his = self.system['histories']
        lc = self.system['linecoll']
        xmin, xmax, ymin, ymax = self.bounding_box
        
        lt += 1
        U = self.interp_u_sum(pp)
        V = self.interp_v_sum(pp)
        velocity_scale = 0.1
        velocity = np.column_stack((U, V)) * velocity_scale
        pp += velocity
        his[:, :-1, :] = his[:, 1:, :]
        his[:, -1, :] = pp
        
        # (Rest of update: reinitialization, building line segments, setting colors, etc.)
        # Finally update the LineCollection:
        lc.set_segments([])  # Replace with the computed segments
        lc.set_colors([])    # Replace with computed colors
        
        return (lc,)
    
    def animate(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        self.prepare_figure(ax)
        anim = FuncAnimation(
            fig, self.update, frames=1000, interval=30, blit=False
        )
        plt.show()