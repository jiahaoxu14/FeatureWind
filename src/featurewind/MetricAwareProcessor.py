"""
Metric-Aware Feature Wind Processor
Extends the basic FeatureWindProcessor with metric-aware rendering capabilities
using Jacobian-based pushforward transformations for research-grade visualization.
"""

import numpy as np
import sys
import os
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.spatial import cKDTree

# Add path for DimReader
sys.path.insert(0, os.path.dirname(__file__))
from DimReader import ProjectionRunner


class MetricAwareProcessor:
    """
    Research-grade processor that computes metric-aware feature flows
    using t-SNE Jacobian matrices for accurate gradient pushforward.
    """
    
    def __init__(self):
        self.projection_runner = None
        self.valid_points = None
        self.feature_names = None
        self.bounding_box = None
        self.metric_aware_enabled = True
        
        # Grid and interpolation data
        self.grid_data = None
        self.top_k_indices = None
        
    def load_data_and_compute_tsne(self, points_data, feature_names):
        """
        Load high-dimensional data and compute t-SNE projection with Jacobian.
        
        Args:
            points_data: numpy array of shape (n_points, n_features)
            feature_names: list of feature names
        """
        self.feature_names = feature_names
        
        # Initialize projection runner with t-SNE
        self.projection_runner = ProjectionRunner("tsne")
        
        # Compute t-SNE projection and gradients
        print("Computing t-SNE projection with Jacobian...")
        self.projection_runner.calculateValues(points_data)
        
        # Extract projected positions
        self.projected_positions = self.projection_runner.outPoints.detach().numpy()
        
        # Setup bounding box
        self._setup_bounding_box()
        
        print(f"Computed projection for {len(points_data)} points with {len(feature_names)} features")
        return True
        
    def _setup_bounding_box(self):
        """Setup bounding box with padding and square aspect ratio."""
        positions = self.projected_positions
        xmin, xmax = positions[:, 0].min(), positions[:, 0].max()
        ymin, ymax = positions[:, 1].min(), positions[:, 1].max()
        
        # Add padding and make square
        padding = 0.05
        x_padding = (xmax - xmin) * padding
        y_padding = (ymax - ymin) * padding
        xmin, xmax = xmin - x_padding, xmax + x_padding
        ymin, ymax = ymin - y_padding, ymax + y_padding
        
        x_range, y_range = xmax - xmin, ymax - ymin
        if x_range > y_range:
            y_center = (ymin + ymax) / 2
            ymin, ymax = y_center - x_range / 2, y_center + x_range / 2
        else:
            x_center = (xmin + xmax) / 2
            xmin, xmax = x_center - y_range / 2, x_center + y_range / 2
        
        self.bounding_box = [xmin, xmax, ymin, ymax]
        
    def select_top_features(self, k):
        """
        Select top k features based on gradient magnitude analysis.
        
        Args:
            k: number of top features to select
            
        Returns:
            tuple: (top_k_indices, avg_magnitudes)
        """
        if self.projection_runner is None:
            raise ValueError("Data not loaded. Run load_data_and_compute_tsne first.")
            
        n_points = len(self.projected_positions)
        n_features = len(self.feature_names)
        
        # Compute average gradient magnitudes across all points
        feature_magnitudes = []
        
        for feat_idx in range(n_features):
            total_magnitude = 0.0
            
            for point_idx in range(n_points):
                # Get feature unit vector (basis vector e_k)
                unit_vector = np.zeros(n_features)
                unit_vector[feat_idx] = 1.0
                
                if self.metric_aware_enabled:
                    # Use metric-aware pushforward
                    grad_2d = self.projection_runner.metric_normalized_pushforward(point_idx, unit_vector)
                else:
                    # Use simple pushforward
                    grad_2d = self.projection_runner.pushforward_vector(point_idx, unit_vector)
                
                magnitude = np.linalg.norm(grad_2d)
                total_magnitude += magnitude
                
            avg_magnitude = total_magnitude / n_points
            feature_magnitudes.append(avg_magnitude)
        
        # Select top-k features
        feature_magnitudes = np.array(feature_magnitudes)
        top_k_indices = np.argsort(-feature_magnitudes)[:k]
        
        self.top_k_indices = top_k_indices
        
        print(f"Selected top {k} features based on gradient magnitude:")
        for i, feat_idx in enumerate(top_k_indices):
            print(f"  {i+1}. {self.feature_names[feat_idx]}: {feature_magnitudes[feat_idx]:.4f}")
        
        return top_k_indices, feature_magnitudes
        
    def create_metric_aware_grids(self, grid_res=40, kdtree_scale=0.03):
        """
        Create velocity grids using metric-aware gradient pushforward.
        
        Args:
            grid_res: grid resolution
            kdtree_scale: distance threshold for masking
            
        Returns:
            dict: grid data including interpolators and dominant features
        """
        if self.top_k_indices is None:
            raise ValueError("Features not selected. Run select_top_features first.")
            
        xmin, xmax, ymin, ymax = self.bounding_box
        num_vertices = grid_res + 1
        n_points = len(self.projected_positions)
        n_features = len(self.feature_names)
        
        # Create grid
        grid_x, grid_y = np.mgrid[xmin:xmax:complex(num_vertices), 
                                  ymin:ymax:complex(num_vertices)]
        
        # Build distance mask
        kdtree = cKDTree(self.projected_positions)
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        distances, _ = kdtree.query(grid_points, k=1)
        dist_grid = distances.reshape(grid_x.shape)
        threshold = max(abs(xmax - xmin), abs(ymax - ymin)) * kdtree_scale
        
        # Compute metric-aware gradients for each feature at each data point
        print("Computing metric-aware feature gradients...")
        
        all_gradients = np.zeros((n_points, n_features, 2))
        
        for point_idx in range(n_points):
            if point_idx % 50 == 0:
                print(f"  Processing point {point_idx}/{n_points}")
                
            for feat_idx in range(n_features):
                # Create feature unit vector
                unit_vector = np.zeros(n_features)
                unit_vector[feat_idx] = 1.0
                
                if self.metric_aware_enabled:
                    grad_2d = self.projection_runner.metric_normalized_pushforward(point_idx, unit_vector)
                else:
                    grad_2d = self.projection_runner.pushforward_vector(point_idx, unit_vector)
                
                all_gradients[point_idx, feat_idx, :] = grad_2d
        
        # Interpolate velocity fields for top-k features
        grid_u_feats, grid_v_feats = [], []
        grid_u_all, grid_v_all = [], []
        
        print("Interpolating velocity fields...")
        
        for feat_idx in range(n_features):
            vectors = all_gradients[:, feat_idx, :]
            grid_u = griddata(self.projected_positions, vectors[:, 0], (grid_x, grid_y), method='nearest')
            grid_v = griddata(self.projected_positions, vectors[:, 1], (grid_x, grid_y), method='nearest')
            
            # Apply distance mask
            mask = dist_grid > threshold
            grid_u[mask] = 0.0
            grid_v[mask] = 0.0
            
            grid_u_all.append(grid_u)
            grid_v_all.append(grid_v)
            
            if feat_idx in self.top_k_indices:
                grid_u_feats.append(grid_u)
                grid_v_feats.append(grid_v)
        
        grid_u_feats = np.array(grid_u_feats)
        grid_v_feats = np.array(grid_v_feats)
        grid_u_all = np.array(grid_u_all)
        grid_v_all = np.array(grid_v_all)
        
        # Sum top-k features
        grid_u_sum = np.sum(grid_u_feats, axis=0)
        grid_v_sum = np.sum(grid_v_feats, axis=0)
        
        # Calculate dominant feature for each grid cell
        grid_mag_all = np.sqrt(grid_u_all**2 + grid_v_all**2)
        cell_dominant_features = np.zeros((grid_res, grid_res), dtype=int)
        
        for i in range(grid_res):
            for j in range(grid_res):
                # Average magnitudes from 4 corner vertices for each feature
                corner_mags = np.zeros(n_features)
                for feat_idx in range(n_features):
                    corner_sum = (grid_mag_all[feat_idx, i, j] + 
                                 grid_mag_all[feat_idx, i+1, j] +
                                 grid_mag_all[feat_idx, i, j+1] + 
                                 grid_mag_all[feat_idx, i+1, j+1])
                    corner_mags[feat_idx] = corner_sum / 4.0
                
                cell_dominant_features[i, j] = np.argmax(corner_mags)
        
        # Create interpolators
        interp_u_sum = RegularGridInterpolator(
            (grid_x[:, 0], grid_y[0, :]), grid_u_sum, 
            bounds_error=False, fill_value=0.0)
        interp_v_sum = RegularGridInterpolator(
            (grid_x[:, 0], grid_y[0, :]), grid_v_sum, 
            bounds_error=False, fill_value=0.0)
        
        self.grid_data = {
            'grid_x': grid_x,
            'grid_y': grid_y,
            'grid_u_sum': grid_u_sum,
            'grid_v_sum': grid_v_sum,
            'grid_u_all': grid_u_all,
            'grid_v_all': grid_v_all,
            'cell_dominant_features': cell_dominant_features,
            'interp_u_sum': interp_u_sum,
            'interp_v_sum': interp_v_sum,
            'grid_res': grid_res,
            'all_gradients': all_gradients  # Store for wind vane analysis
        }
        
        print("Metric-aware grids created successfully")
        return self.grid_data
        
    def get_wind_vane_analysis(self, point_idx):
        """
        Get detailed wind vane analysis for a specific point.
        
        Args:
            point_idx: index of the point to analyze
            
        Returns:
            dict: wind vane data with metric-aware vectors
        """
        if self.grid_data is None:
            raise ValueError("Grids not created. Run create_metric_aware_grids first.")
            
        position = self.projected_positions[point_idx]
        all_gradients = self.grid_data['all_gradients']
        
        # Get metric tensor for this point
        metric_tensor = self.projection_runner.compute_metric_tensor(point_idx)
        metric_condition = np.linalg.cond(metric_tensor)
        
        # Analyze top-k features
        vector_info = []
        for feat_idx in self.top_k_indices:
            gradient = all_gradients[point_idx, feat_idx, :]
            magnitude = np.linalg.norm(gradient)
            
            if magnitude > 1e-10:
                vector_info.append({
                    'feature_idx': int(feat_idx),
                    'feature_name': self.feature_names[feat_idx],
                    'u': float(gradient[0]),
                    'v': float(gradient[1]),
                    'magnitude': float(magnitude),
                    'angle': float(np.arctan2(gradient[1], gradient[0]) * 180 / np.pi)
                })
        
        # Compute resultant (sum of top-k)
        resultant = np.sum([all_gradients[point_idx, feat_idx, :] for feat_idx in self.top_k_indices], axis=0)
        
        return {
            'point_idx': point_idx,
            'position': position.tolist(),
            'vectors': vector_info,
            'resultant': {
                'u': float(resultant[0]),
                'v': float(resultant[1]),
                'magnitude': float(np.linalg.norm(resultant)),
                'angle': float(np.arctan2(resultant[1], resultant[0]) * 180 / np.pi)
            },
            'metric_condition': float(metric_condition),
            'metric_aware': self.metric_aware_enabled
        }
        
    def toggle_metric_aware(self, enabled=None):
        """Toggle metric-aware rendering on/off."""
        if enabled is None:
            self.metric_aware_enabled = not self.metric_aware_enabled
        else:
            self.metric_aware_enabled = enabled
            
        return self.metric_aware_enabled