#!/usr/bin/env python3
"""
Generate a double helix with 200 points for FeatureWind case study.
"""

import numpy as np
import pandas as pd

def generate_double_helix_500():
    """Generate 500 points forming a double helix structure with clear spatial structure."""
    
    # Parameters for the double helix
    n_points = 500
    n_per_helix = n_points // 2  # 250 points per helix
    
    # Generate parameter t for the helix (6 full turns for more structure)
    t1 = np.linspace(0, 6 * np.pi, n_per_helix)  # First helix
    t2 = np.linspace(0, 6 * np.pi, n_per_helix)  # Second helix
    
    # First helix (label = 0)
    x1 = np.cos(t1)
    y1 = np.sin(t1)
    z1 = t1 / (2 * np.pi)  # Rising helix from 0 to 3
    labels1 = np.zeros(n_per_helix)
    
    # Second helix (label = 1) - phase shifted by π
    x2 = np.cos(t2 + np.pi)  # Phase shift by π
    y2 = np.sin(t2 + np.pi)
    z2 = t2 / (2 * np.pi)   # Same height progression
    labels2 = np.ones(n_per_helix)
    
    # Combine both helices
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    z = np.concatenate([z1, z2])
    labels = np.concatenate([labels1, labels2]).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z,
        'label': labels
    })
    
    # Shuffle the dataset to mix the helices
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def main():
    """Generate and save the double helix dataset."""
    
    print("Generating double helix with 500 points...")
    df = generate_double_helix_500()
    
    # Save to CSV
    output_file = "double_helix_500.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Dataset saved to: {output_file}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Display basic statistics
    print("\nDataset preview:")
    print(df.head())
    
    print("\nFeature ranges:")
    print(df.describe())

if __name__ == "__main__":
    main()