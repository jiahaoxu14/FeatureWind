#!/usr/bin/env python3
"""
Generate a single helix with 500 points for FeatureWind case study.
"""

import numpy as np
import pandas as pd
import sys

def generate_single_helix(n_points=300):
    """Generate n_points forming a single helix structure with clear spatial progression."""
    
    # Parameters for the single helix
    
    # Generate parameter t for the helix (8 full turns for clear structure)
    t = np.linspace(0, 8 * np.pi, n_points)
    
    # Single helix coordinates
    x = np.cos(t)
    y = np.sin(t)  
    z = t / (3.5 * np.pi)  # Rising helix from 0 to 4
    
    # Create labels - hardcode all points to label 0
    labels = np.zeros(n_points, dtype=int)
    
    # # Add some noise to make the problem more interesting for gradient computation
    # noise_scale = 0.05
    # x += np.random.normal(0, noise_scale, n_points)
    # y += np.random.normal(0, noise_scale, n_points)
    # z += np.random.normal(0, noise_scale, n_points)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z,
        'label': labels
    })
    
    # Shuffle the dataset to mix the segments
    # df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def main():
    """Generate and save the single helix dataset."""
    
    # Parse command line arguments
    n_points = 300  # default
    if len(sys.argv) > 1:
        try:
            n_points = int(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid number of points '{sys.argv[1]}'. Using default {n_points}.")
    
    print(f"Generating single helix with {n_points} points...")
    df = generate_single_helix(n_points)
    
    # Save to CSV
    output_file = f"examples/singlehelix/single_helix_{n_points}.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Dataset saved to: {output_file}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Label distribution: {df['label'].value_counts().sort_index().to_dict()}")
    
    # Display basic statistics
    print("\nDataset preview:")
    print(df.head())
    
    print("\nFeature ranges:")
    print(df.describe())

if __name__ == "__main__":
    main()