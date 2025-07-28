#!/usr/bin/env python3
"""
Basic example using the FeatureWind class for visualization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
from featurewind import FeatureWind

def main():
    # Paths relative to repository root
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    tangent_map_path = os.path.join(repo_root, 'tangentmaps', 'tworings.tmap')
    output_dir = os.path.join(repo_root, 'output')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create FeatureWind visualization
    fw = FeatureWind(
        tangentmap_path=tangent_map_path,
        number_of_features=3,
        kdtree_scale=0.03,
        velocity_scale=0.01,
        grid_size=20,
        number_of_particles=2000,
        figure_size=(10, 8)
    )
    
    # Create animation and optionally save to output directory
    save_animation = False  # Set to True to save animation
    if save_animation:
        save_path = os.path.join(output_dir, "basic_example.mp4")
        anim = fw.animate(frames=100, interval=30, save=True, save_path=save_path)
    else:
        anim = fw.animate(frames=1000, interval=30)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()