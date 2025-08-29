#!/usr/bin/env python3
"""
Generate an S-curve dataset for FeatureWind case study.

The S-curve demonstrates clear gradient transitions between:
- "Along the curve" direction (tangent to the curve)  
- "Across the thickness" direction (normal to the curve)

This creates crisp wind-vane dominance changes without periodic ambiguity.
"""

import numpy as np
import pandas as pd
import sys
import os

def generate_scurve(n_points=400, noise_scale=0.03):
    """
    Generate n_points forming an S-curve manifold with varying curvature and thickness.
    
    Args:
        n_points: Number of points to generate
        noise_scale: Amount of noise to add (0 = perfect curve)
    
    Returns:
        DataFrame with x, y, z coordinates and curve_position parameter
    """
    
    # Generate parameter s along the curve length [0, 1]
    s = np.linspace(0, 1, n_points)
    
    # S-curve parametrization in 3D
    # x: S-shaped curve with varying curvature
    # y: thickness variation (wider in middle, thinner at ends)
    # z: slight vertical variation for 3D embedding
    
    # Main S-curve in x-direction (using tanh for smooth S-shape)
    curve_param = 4 * (s - 0.5)  # Scale to [-2, 2] for good S-shape
    x_center = 2.0 * np.tanh(curve_param)  # S-curve from -2 to +2
    
    # Thickness variation: wider in middle, thinner at ends
    thickness = 0.8 * np.exp(-8 * (s - 0.5)**2) + 0.2  # Gaussian-like thickness
    
    # Random thickness displacement (perpendicular to curve)
    thickness_offset = np.random.uniform(-thickness, thickness, n_points)
    
    # Compute tangent direction for proper perpendicular displacement
    # dx/ds for the S-curve
    dx_ds = 8 * (1 - np.tanh(curve_param)**2)  # derivative of tanh
    
    # Perpendicular direction (rotate tangent by 90 degrees in xy-plane)
    # Tangent = (dx_ds, 0), Perpendicular = (0, 1) normalized
    perp_x = np.zeros_like(dx_ds)  # perpendicular component in x
    perp_y = np.ones_like(dx_ds)   # perpendicular component in y
    
    # Apply thickness displacement
    x = x_center + perp_x * thickness_offset
    y = perp_y * thickness_offset
    
    # Slight z-variation for 3D embedding (follows curve curvature)
    z = 0.5 * np.sin(2 * np.pi * s) * (1 - abs(thickness_offset) / np.max(thickness))
    
    # Add controlled noise
    if noise_scale > 0:
        x += np.random.normal(0, noise_scale, n_points)
        y += np.random.normal(0, noise_scale, n_points)
        z += np.random.normal(0, noise_scale, n_points)
    
    # Create labels - hardcode all points to label 0
    labels = np.zeros(n_points, dtype=int)
    
    # Create DataFrame with only x, y, z coordinates
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
    """Generate and save the S-curve dataset."""
    
    # Parse command line arguments
    n_points = 400  # default
    noise_scale = 0.03  # default
    
    if len(sys.argv) > 1:
        try:
            n_points = int(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid number of points '{sys.argv[1]}'. Using default {n_points}.")
    
    if len(sys.argv) > 2:
        try:
            noise_scale = float(sys.argv[2])
        except ValueError:
            print(f"Error: Invalid noise scale '{sys.argv[2]}'. Using default {noise_scale}.")
    
    print(f"Generating S-curve with {n_points} points and noise scale {noise_scale}...")
    df = generate_scurve(n_points, noise_scale)
    
    # Ensure output directory exists
    output_dir = "examples/scurve"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_file = f"{output_dir}/scurve_{n_points}.csv"
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
    
    print("\nS-curve dataset characteristics:")
    print("- Features x, y, z: 3D coordinates of S-shaped manifold")
    print("- Manifold has varying curvature and thickness")
    print("- Labels: All points labeled as 0")
    
    print("\nExpected gradient behavior:")
    print("- Gradients should show 'along curve' vs 'across thickness' directions")
    print("- Wind vane should show clear dominance changes between tangent/normal directions")
    print("- Clean geodesics with varying curvature for crisp visualization")

if __name__ == "__main__":
    main()