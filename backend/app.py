"""
Flask backend for FeatureWind visualization system.
Handles data processing, grid computation, and serves API endpoints for the React frontend.
"""

import sys
import os
import json
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.spatial import cKDTree

# Add source path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from featurewind.TangentPoint import TangentPoint

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

class FeatureWindProcessor:
    def __init__(self):
        self.valid_points = None
        self.all_grad_vectors = None
        self.all_positions = None
        self.col_labels = None
        self.bounding_box = None
        self.grid_data = None
        self.top_k_indices = None
        
    def load_data(self, tangent_map_path):
        """Load and preprocess tangent map data."""
        with open(tangent_map_path, "r") as f:
            data_import = json.loads(f.read())

        tmap = data_import['tmap']
        col_labels = data_import['Col_labels']
        
        # Create TangentPoint objects
        points = [TangentPoint(entry, 1.0, col_labels) for entry in tmap]
        valid_points = [p for p in points if p.valid]
        
        # Extract positions and gradient vectors
        all_positions = np.array([p.position for p in valid_points])
        all_grad_vectors = np.array([p.gradient_vectors for p in valid_points])
        
        self.valid_points = valid_points
        self.all_grad_vectors = all_grad_vectors
        self.all_positions = all_positions
        self.col_labels = col_labels
        
        self._setup_bounding_box()
        return True
        
    def _setup_bounding_box(self):
        """Setup bounding box with padding and square aspect ratio."""
        xmin, xmax = self.all_positions[:, 0].min(), self.all_positions[:, 0].max()
        ymin, ymax = self.all_positions[:, 1].min(), self.all_positions[:, 1].max()
        
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
        """Select top k features based on average gradient magnitude."""
        feature_magnitudes = np.linalg.norm(self.all_grad_vectors, axis=2)
        avg_magnitudes = feature_magnitudes.mean(axis=0)
        top_k_indices = np.argsort(-avg_magnitudes)[:k]
        self.top_k_indices = top_k_indices
        return top_k_indices, avg_magnitudes
        
    def create_grids(self, grid_res=40, kdtree_scale=0.03):
        """Create velocity grids and determine dominant features for each grid cell."""
        xmin, xmax, ymin, ymax = self.bounding_box
        num_vertices = grid_res + 1
        
        # Create grid
        grid_x, grid_y = np.mgrid[xmin:xmax:complex(num_vertices), 
                                  ymin:ymax:complex(num_vertices)]
        
        # Build distance mask
        kdtree = cKDTree(self.all_positions)
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        distances, _ = kdtree.query(grid_points, k=1)
        dist_grid = distances.reshape(grid_x.shape)
        threshold = max(abs(xmax - xmin), abs(ymax - ymin)) * kdtree_scale
        
        # Interpolate velocity fields for top-k features
        grid_u_feats, grid_v_feats = [], []
        for feat_idx in self.top_k_indices:
            vectors = self.all_grad_vectors[:, feat_idx, :]
            grid_u = griddata(self.all_positions, vectors[:, 0], (grid_x, grid_y), method='nearest')
            grid_v = griddata(self.all_positions, vectors[:, 1], (grid_x, grid_y), method='nearest')
            
            # Apply distance mask
            mask = dist_grid > threshold
            grid_u[mask] = 0.0
            grid_v[mask] = 0.0
            
            grid_u_feats.append(grid_u)
            grid_v_feats.append(grid_v)
        
        grid_u_feats = np.array(grid_u_feats)
        grid_v_feats = np.array(grid_v_feats)
        grid_u_sum = np.sum(grid_u_feats, axis=0)
        grid_v_sum = np.sum(grid_v_feats, axis=0)
        
        # Create grids for ALL features to determine true dominance
        num_features = self.all_grad_vectors.shape[1]
        grid_u_all, grid_v_all = [], []
        
        for feat_idx in range(num_features):
            vectors = self.all_grad_vectors[:, feat_idx, :]
            grid_u = griddata(self.all_positions, vectors[:, 0], (grid_x, grid_y), method='nearest')
            grid_v = griddata(self.all_positions, vectors[:, 1], (grid_x, grid_y), method='nearest')
            
            mask = dist_grid > threshold
            grid_u[mask] = 0.0
            grid_v[mask] = 0.0
            
            grid_u_all.append(grid_u)
            grid_v_all.append(grid_v)
        
        grid_u_all = np.array(grid_u_all)
        grid_v_all = np.array(grid_v_all)
        
        # Calculate dominant feature for each grid cell
        grid_mag_all = np.sqrt(grid_u_all**2 + grid_v_all**2)
        cell_dominant_features = np.zeros((grid_res, grid_res), dtype=int)
        
        for i in range(grid_res):
            for j in range(grid_res):
                # Average magnitudes from 4 corner vertices for each feature
                corner_mags = np.zeros(num_features)
                for feat_idx in range(num_features):
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
            'grid_res': grid_res
        }
        
        return self.grid_data

# Global processor instance
processor = FeatureWindProcessor()

@app.route('/api/load_data', methods=['POST'])
def load_data():
    """Load tangent map data from specified file."""
    data = request.get_json()
    filename = data.get('filename', 'breast_cancer.tmap')
    
    tangent_map_path = os.path.join(os.path.dirname(__file__), 'tangentmaps', filename)
    
    if not os.path.exists(tangent_map_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        processor.load_data(tangent_map_path)
        return jsonify({
            'success': True,
            'num_points': len(processor.valid_points),
            'num_features': len(processor.col_labels),
            'bounding_box': processor.bounding_box,
            'feature_labels': processor.col_labels
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/setup_visualization', methods=['POST'])
def setup_visualization():
    """Setup visualization with specified parameters."""
    data = request.get_json()
    k = data.get('k', len(processor.col_labels) if processor.col_labels else 10)
    grid_res = data.get('grid_res', 40)
    kdtree_scale = data.get('kdtree_scale', 0.03)
    
    if processor.all_grad_vectors is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    try:
        # Select top features
        top_k_indices, avg_magnitudes = processor.select_top_features(k)
        
        # Create grids
        grid_data = processor.create_grids(grid_res, kdtree_scale)
        
        return jsonify({
            'success': True,
            'top_k_indices': top_k_indices.tolist(),
            'avg_magnitudes': avg_magnitudes.tolist(),
            'grid_res': grid_res,
            'bounding_box': processor.bounding_box
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_data_points', methods=['GET'])
def get_data_points():
    """Get all data points for visualization."""
    if processor.valid_points is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    points_data = []
    for point in processor.valid_points:
        points_data.append({
            'position': point.position.tolist(),
            'label': point.tmap_label
        })
    
    return jsonify({
        'points': points_data,
        'bounding_box': processor.bounding_box
    })

@app.route('/api/get_velocity_field', methods=['POST'])
def get_velocity_field():
    """Get velocity field for particle animation."""
    data = request.get_json()
    positions = np.array(data.get('positions', []))
    
    if processor.grid_data is None:
        return jsonify({'error': 'Grid data not initialized'}), 400
    
    if len(positions) == 0:
        return jsonify({'velocities': []})
    
    try:
        interp_u = processor.grid_data['interp_u_sum']
        interp_v = processor.grid_data['interp_v_sum']
        
        U = interp_u(positions)
        V = interp_v(positions)
        
        velocities = np.column_stack((U, V))
        
        return jsonify({
            'velocities': velocities.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_wind_vane_data', methods=['POST'])
def get_wind_vane_data():
    """Get wind vane data for a specific grid cell."""
    data = request.get_json()
    cell_i = data.get('cell_i', 0)
    cell_j = data.get('cell_j', 0)
    
    if processor.grid_data is None:
        return jsonify({'error': 'Grid data not initialized'}), 400
    
    try:
        grid_res = processor.grid_data['grid_res']
        cell_dominant_features = processor.grid_data['cell_dominant_features']
        grid_u_all = processor.grid_data['grid_u_all']
        grid_v_all = processor.grid_data['grid_v_all']
        grid_x = processor.grid_data['grid_x']
        grid_y = processor.grid_data['grid_y']
        
        # Validate grid cell
        if cell_i < 0 or cell_i >= grid_res or cell_j < 0 or cell_j >= grid_res:
            return jsonify({'error': 'Invalid grid cell'}), 400
        
        dominant_feature = cell_dominant_features[cell_i, cell_j]
        
        # Calculate cell center for sampling
        xmin, xmax, ymin, ymax = processor.bounding_box
        sample_cx = xmin + (cell_j + 0.5) * (xmax - xmin) / grid_res
        sample_cy = ymin + (cell_i + 0.5) * (ymax - ymin) / grid_res
        
        # Get vectors for each feature at cell center
        vector_info = []
        for feat_idx in processor.top_k_indices:
            # Sample feature vector at cell center using interpolation
            feat_u_interp = RegularGridInterpolator(
                (grid_x[:, 0], grid_y[0, :]), grid_u_all[feat_idx], 
                bounds_error=False, fill_value=0.0)
            feat_v_interp = RegularGridInterpolator(
                (grid_x[:, 0], grid_y[0, :]), grid_v_all[feat_idx], 
                bounds_error=False, fill_value=0.0)
            
            center_u = feat_u_interp(np.array([[sample_cx, sample_cy]])).item()
            center_v = feat_v_interp(np.array([[sample_cx, sample_cy]])).item()
            center_mag = np.sqrt(center_u**2 + center_v**2)
            
            if center_mag > 0:
                vector_info.append({
                    'feature_idx': int(feat_idx),
                    'u': center_u,
                    'v': center_v,
                    'magnitude': center_mag,
                    'is_dominant': feat_idx == dominant_feature,
                    'label': processor.col_labels[feat_idx] if feat_idx < len(processor.col_labels) else f'Feature {feat_idx}'
                })
        
        # Calculate resultant vector
        interp_u = processor.grid_data['interp_u_sum']
        interp_v = processor.grid_data['interp_v_sum']
        resultant_u = interp_u(np.array([[sample_cx, sample_cy]])).item()
        resultant_v = interp_v(np.array([[sample_cx, sample_cy]])).item()
        
        return jsonify({
            'cell_position': [cell_i, cell_j],
            'sample_center': [sample_cx, sample_cy],
            'dominant_feature': int(dominant_feature),
            'vectors': vector_info,
            'resultant': {
                'u': resultant_u,
                'v': resultant_v,
                'magnitude': np.sqrt(resultant_u**2 + resultant_v**2)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_grid_data', methods=['GET'])
def get_grid_data():
    """Get grid visualization data."""
    if processor.grid_data is None:
        return jsonify({'error': 'Grid data not initialized'}), 400
    
    try:
        grid_x = processor.grid_data['grid_x']
        grid_y = processor.grid_data['grid_y']
        cell_dominant_features = processor.grid_data['cell_dominant_features']
        grid_u_sum = processor.grid_data['grid_u_sum']
        grid_v_sum = processor.grid_data['grid_v_sum']
        grid_res = processor.grid_data['grid_res']
        
        # Prepare grid lines
        grid_lines = []
        n_rows, n_cols = grid_x.shape
        
        # Vertical lines
        for col in range(n_cols):
            line = []
            for row in range(n_rows):
                line.append([grid_x[row, col], grid_y[row, col]])
            grid_lines.append(line)
        
        # Horizontal lines
        for row in range(n_rows):
            line = []
            for col in range(n_cols):
                line.append([grid_x[row, col], grid_y[row, col]])
            grid_lines.append(line)
        
        # Prepare cell data
        cells = []
        xmin, xmax, ymin, ymax = processor.bounding_box
        
        for i in range(grid_res):
            for j in range(grid_res):
                cell_xmin = xmin + j * (xmax - xmin) / grid_res
                cell_xmax = xmin + (j + 1) * (xmax - xmin) / grid_res
                cell_ymin = ymin + i * (ymax - ymin) / grid_res
                cell_ymax = ymin + (i + 1) * (ymax - ymin) / grid_res
                
                # Check if cell has values
                corner_u_sum = (grid_u_sum[i, j] + grid_u_sum[i+1, j] + 
                               grid_u_sum[i, j+1] + grid_u_sum[i+1, j+1])
                corner_v_sum = (grid_v_sum[i, j] + grid_v_sum[i+1, j] + 
                               grid_v_sum[i, j+1] + grid_v_sum[i+1, j+1])
                
                is_empty = abs(corner_u_sum) < 1e-10 and abs(corner_v_sum) < 1e-10
                
                cells.append({
                    'i': i,
                    'j': j,
                    'bounds': [cell_xmin, cell_ymin, cell_xmax, cell_ymax],
                    'dominant_feature': int(cell_dominant_features[i, j]),
                    'is_empty': is_empty
                })
        
        return jsonify({
            'grid_lines': grid_lines,
            'cells': cells,
            'grid_res': grid_res
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/available_files', methods=['GET'])
def available_files():
    """Get list of available tangent map files."""
    tangent_dir = os.path.join(os.path.dirname(__file__), 'tangentmaps')
    
    if not os.path.exists(tangent_dir):
        return jsonify({'files': []})
    
    files = [f for f in os.listdir(tangent_dir) if f.endswith('.tmap')]
    return jsonify({'files': files})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)