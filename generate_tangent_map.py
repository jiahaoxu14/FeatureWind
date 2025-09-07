#!/usr/bin/env python3
"""
Generate tangent maps from CSV datasets with automatic label handling.

This script processes CSV files by:
1. Extracting labels from the dataset
2. Running tangent map generation on features only
3. Adding labels back to the final tangent map
4. Creating a complete .tmap file for FeatureWind visualization

Usage:
    python generate_tangent_map.py dataset.csv tsne [output_name]

Example:
    python generate_tangent_map.py examples/helix/double_helix_200.csv tsne helix
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def identify_label_column(df):
    """
    Identify the label column in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        str or None: Name of the label column
    """
    # Common label column names
    label_candidates = ['label', 'class', 'target', 'y']
    
    for col in label_candidates:
        if col.lower() in [c.lower() for c in df.columns]:
            # Find the actual column name with correct case
            for actual_col in df.columns:
                if actual_col.lower() == col.lower():
                    return actual_col
    
    # If no standard label column found, check the last column
    # If it's not numeric, it might be a label
    last_col = df.columns[-1]
    try:
        pd.to_numeric(df[last_col])
    except (ValueError, TypeError):
        # Last column is not numeric, likely a label
        return last_col
    
    return None


def extract_features_and_labels(csv_file):
    """
    Extract features and labels from the CSV file.
    
    Args:
        csv_file (str): Path to the CSV file
        
    Returns:
        tuple: (feature_df, labels, label_column, feature_columns)
    """
    print(f"Loading dataset: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Identify label column
    label_column = identify_label_column(df)
    
    if label_column:
        print(f"Found label column: '{label_column}'")
        labels = df[label_column].tolist()
        feature_columns = [col for col in df.columns if col != label_column]
        feature_df = df[feature_columns]
    else:
        print("No label column found, treating all columns as features")
        labels = [0] * len(df)  # Dummy labels
        feature_columns = df.columns.tolist()
        feature_df = df
        label_column = None
    
    print(f"Features: {feature_columns}")
    print(f"Dataset shape: {df.shape}, Features: {feature_df.shape}")
    
    return feature_df, labels, label_column, feature_columns


def run_tangent_map_generation(feature_df, projection):
    """
    Run tangent map generation on the feature DataFrame.
    
    Args:
        feature_df (pd.DataFrame): DataFrame with only feature columns
        projection (str): Projection method (e.g., 'tsne')
        
    Returns:
        str: Path to the generated .tmap file
    """
    # Create temporary file for features-only data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        temp_csv_path = temp_file.name
        feature_df.to_csv(temp_csv_path, index=False)
    
    try:
        # Run tangent map generation using module approach to handle imports
        print(f"Running tangent map generation with {projection}...")
        
        # Use python -m approach from the project root to handle relative imports
        result = subprocess.run([
            sys.executable, "-m", "featurewind.core.tangent_map", temp_csv_path, projection
        ], cwd=Path(__file__).parent, capture_output=False, text=True)
        
        if result.returncode != 0:
            print("Error running tangent map generation:")
            print(result.stderr)
            raise RuntimeError(f"Tangent map generation failed: {result.stderr}")
        
        print("Tangent map generation completed successfully")
        
        # Find the generated .tmap file (should be in the same directory as temp file)
        temp_name = Path(temp_csv_path).stem
        tmap_file = Path(temp_csv_path).parent / f"{temp_name}_TangentMap_{projection}.tmap"
        
        if not tmap_file.exists():
            raise FileNotFoundError(f"Expected tangent map file not found: {tmap_file}")
        
        return str(tmap_file)
        
    finally:
        # Clean up temporary CSV file
        os.unlink(temp_csv_path)


def add_labels_to_tangent_map(tmap_file, labels, feature_columns, output_name=None, input_csv_path=None):
    """
    Add labels and column information to the tangent map.
    
    Args:
        tmap_file (str): Path to the tangent map file
        labels (list): List of labels for each data point
        feature_columns (list): List of feature column names
        output_name (str, optional): Custom output filename
        input_csv_path (str, optional): Path to the original CSV file for directory reference
        
    Returns:
        str: Path to the enhanced tangent map file
    """
    print(f"Adding labels to tangent map: {tmap_file}")
    
    # Read the raw tangent map
    with open(tmap_file, 'r') as f:
        tmap = json.load(f)
    
    print(f"Loaded tangent map with {len(tmap)} entries")
    
    # Add labels to each tangent map entry
    for i, tangent_entry in enumerate(tmap):
        if i < len(labels):
            tangent_entry['class'] = labels[i]
        else:
            tangent_entry['class'] = 0  # Default label
        tangent_entry['label'] = False  # Standard field
    
    # Create the final data structure
    final_data = {
        'tmap': tmap,
        'Col_labels': feature_columns
    }
    
    # Determine output directory and filename
    if input_csv_path:
        # Save to the same directory as the input CSV
        input_dir = Path(input_csv_path).parent
        if output_name:
            output_file = input_dir / f"{output_name}.tmap"
        else:
            # Use the CSV filename but with .tmap extension
            csv_stem = Path(input_csv_path).stem
            output_file = input_dir / f"{csv_stem}.tmap"
    else:
        # Fallback to current directory
        if output_name:
            output_file = f"{output_name}.tmap"
        else:
            tmap_filename = Path(tmap_file).name
            output_file = tmap_filename
    
    # Save the enhanced tangent map
    with open(output_file, 'w') as f:
        json.dump(final_data, f)
    
    print(f"Enhanced tangent map saved to: {output_file}")
    print(f"Total entries: {len(tmap)}")
    print(f"Feature columns: {feature_columns}")
    if tmap:
        unique_labels = list(set(entry['class'] for entry in tmap))
        print(f"Unique labels: {unique_labels}")
    
    # Clean up the temporary tangent map file
    os.unlink(tmap_file)
    
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description='Generate tangent maps from CSV datasets with automatic label handling'
    )
    parser.add_argument('csv_file', help='Input CSV file')
    parser.add_argument('projection', choices=['tsne', 'mds'], help='Projection method')
    parser.add_argument('output_name', nargs='?', help='Output filename (without extension)')
    
    args = parser.parse_args()
    
    try:
        # Extract features and labels
        feature_df, labels, label_column, feature_columns = extract_features_and_labels(args.csv_file)
        
        # Generate tangent map on features only
        tmap_file = run_tangent_map_generation(feature_df, args.projection)
        
        # Add labels back and save final result
        output_file = add_labels_to_tangent_map(tmap_file, labels, feature_columns, args.output_name, args.csv_file)
        
        print(f"\n✓ Successfully generated tangent map: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
