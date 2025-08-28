#!/usr/bin/env python3
"""
Generate a single helix with 500 points for FeatureWind case study.
"""

import numpy as np
import pandas as pd

def generate_single_helix_500():
    """Generate 500 points forming a single helix structure with clear spatial progression."""
    
    # Parameters for the single helix
    n_points = 500
    
    # Generate parameter t for the helix (8 full turns for clear structure)
    t = np.linspace(0, 8 * np.pi, n_points)
    
    # Single helix coordinates
    x = np.cos(t)
    y = np.sin(t)  
    z = t / (2 * np.pi)  # Rising helix from 0 to 4
    
    # Create labels based on height progression for clear spatial structure
    # This creates 4 distinct segments along the z-axis
    labels = np.floor(z).astype(int)
    # Ensure labels are in range [0, 3]
    labels = np.clip(labels, 0, 3)
    
    # Add some noise to make the problem more interesting for gradient computation
    noise_scale = 0.05
    x += np.random.normal(0, noise_scale, n_points)
    y += np.random.normal(0, noise_scale, n_points)
    z += np.random.normal(0, noise_scale, n_points)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z,
        'label': labels
    })
    
    # Shuffle the dataset to mix the segments
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def main():
    """Generate and save the single helix dataset."""
    
    print("Generating single helix with 500 points...")
    df = generate_single_helix_500()
    
    # Save to CSV
    output_file = "examples/singlehelix/single_helix_500.csv"
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