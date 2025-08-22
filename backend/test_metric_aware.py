#!/usr/bin/env python3
"""
Quick test script for metric-aware functionality
"""

import sys
import os
import numpy as np
import json

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from featurewind.MetricAwareProcessor import MetricAwareProcessor

def test_metric_aware():
    print("Testing MetricAwareProcessor...")
    
    # Create synthetic test data
    n_points = 50  # Small dataset for quick testing
    n_features = 4
    
    # Generate simple 2D manifold data
    angles = np.linspace(0, 2*np.pi, n_points)
    radius = 1.0 + 0.3 * np.sin(3 * angles)
    
    points_data = np.column_stack([
        radius * np.cos(angles),      # x coordinate
        radius * np.sin(angles),      # y coordinate 
        np.sin(2 * angles),          # z coordinate
        np.cos(3 * angles)           # w coordinate
    ])
    
    feature_names = ['x', 'y', 'z', 'w']
    
    print(f"Generated test data: {points_data.shape}")
    
    try:
        # Test processor
        processor = MetricAwareProcessor()
        
        print("Loading data and computing t-SNE...")
        processor.load_data_and_compute_tsne(points_data, feature_names)
        
        print("Selecting top features...")
        top_k_indices, avg_magnitudes = processor.select_top_features(3)
        
        print("Creating metric-aware grids...")
        grid_data = processor.create_metric_aware_grids(grid_res=10)
        
        print("Testing wind vane analysis...")
        wind_vane_data = processor.get_wind_vane_analysis(0)
        
        print("✅ All tests passed!")
        print(f"Top features: {top_k_indices}")
        print(f"Wind vane vectors: {len(wind_vane_data['vectors'])}")
        print(f"Resultant magnitude: {wind_vane_data['resultant']['magnitude']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_metric_aware()