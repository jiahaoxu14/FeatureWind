#!/usr/bin/env python3
"""
Generate Tangent Map Script

This script takes a dataset as input and outputs a tangent map file (.tmap) 
that can be used with FeatureWind for visualization.

Usage:
    python generate_tangent_map.py <input_csv> <method> [options]
    
Example:
    python generate_tangent_map.py breast_data.csv tsne --output breast_cancer.tmap --target target
"""

import sys
import os
import json
import argparse
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

# Add the source directory to the Python path to import featurewind modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def preprocess_data(input_file, target_column=None, normalize=True, output_prefix=None):
    """
    Preprocess the input dataset for tangent map generation.
    
    Args:
        input_file (str): Path to the input CSV file
        target_column (str): Name of the target/label column (optional)
        normalize (bool): Whether to normalize features to [0, 1] range
        output_prefix (str): Prefix for output files (optional)
    
    Returns:
        tuple: (preprocessed_file_path, target_values, feature_columns)
    """
    print(f"Loading dataset: {input_file}")
    data = pd.read_csv(input_file)
    
    # Extract target values if specified
    target_values = None
    feature_columns = None
    
    if target_column and target_column in data.columns:
        target_values = data[target_column].tolist()
        feature_data = data.drop(columns=[target_column])
        feature_columns = feature_data.columns.tolist()
        print(f"Found target column '{target_column}' with {len(set(target_values))} unique values")
    else:
        feature_data = data
        feature_columns = data.columns.tolist()
        # Create dummy target values (all zeros)
        target_values = [0.0] * len(data)
        print(f"No target column specified, using dummy labels")
    
    # Convert all columns to numeric, handling any non-numeric values
    for col in feature_data.columns:
        feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
    
    # Drop rows with any missing values
    before_dropna = len(feature_data)
    feature_data = feature_data.dropna()
    after_dropna = len(feature_data)
    
    if before_dropna != after_dropna:
        print(f"Dropped {before_dropna - after_dropna} rows with missing values")
        # Update target values to match
        target_values = [target_values[i] for i in feature_data.index]
    
    # Normalize features if requested
    if normalize:
        print("Normalizing features to [0, 1] range...")
        feature_data = (feature_data - feature_data.min()) / (feature_data.max() - feature_data.min())
        feature_data = feature_data.round(3)
    
    # Generate output filename
    if output_prefix is None:
        input_path = Path(input_file)
        output_prefix = input_path.stem + "_normalized"
    
    processed_file = f"{output_prefix}.csv"
    feature_data.to_csv(processed_file, index=False)
    print(f"Preprocessed data saved to: {processed_file}")
    
    return processed_file, target_values, feature_columns


def generate_tangent_map(input_file, method='tsne'):
    """
    Generate tangent map using the TangentMap.py script.
    
    Args:
        input_file (str): Path to the preprocessed CSV file
        method (str): Dimensionality reduction method ('tsne', 'umap', 'pca', etc.)
    
    Returns:
        str: Path to the generated .tmap file
    """
    print(f"Generating tangent map using {method.upper()}...")
    
    # Find the TangentMap.py script
    script_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'featurewind', 'TangentMap.py')
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"TangentMap.py not found at {script_path}")
    
    # Run the TangentMap.py script
    cmd = ['python', script_path, input_file, method]
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Tangent map generation completed successfully")
        if result.stdout:
            print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running TangentMap.py: {e}")
        if e.stderr:
            print("Error output:", e.stderr)
        raise
    
    # Determine the expected output filename
    input_path = Path(input_file)
    tmap_file = f"{input_path.stem}_TangentMap_{method}.tmap"
    
    if not os.path.exists(tmap_file):
        raise FileNotFoundError(f"Expected tangent map file not found: {tmap_file}")
    
    return tmap_file


def postprocess_tangent_map(tmap_file, target_values, feature_columns, output_file):
    """
    Post-process the tangent map file to add labels and metadata.
    
    Args:
        tmap_file (str): Path to the raw .tmap file
        target_values (list): List of target/label values
        feature_columns (list): List of feature column names
        output_file (str): Path for the final output file
    """
    print(f"Post-processing tangent map: {tmap_file}")
    
    # Load the raw tangent map data
    with open(tmap_file, 'r') as f:
        tmap_data = json.load(f)
    
    print(f"Loaded tangent map with {len(tmap_data)} points")
    
    # Add class labels and metadata to each point
    for i, tangent_point in enumerate(tmap_data):
        if i < len(target_values):
            tangent_point['class'] = target_values[i]
        else:
            tangent_point['class'] = 0.0  # Default value
        tangent_point['label'] = False  # Not pre-labeled
    
    # Create the final data structure
    final_data = {
        'tmap': tmap_data,
        'Col_labels': feature_columns
    }
    
    # Save the final tangent map file
    with open(output_file, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    print(f"Final tangent map saved to: {output_file}")
    print(f"Number of data points: {len(tmap_data)}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Feature names: {feature_columns[:5]}..." if len(feature_columns) > 5 else f"Feature names: {feature_columns}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate tangent map files from datasets for FeatureWind visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with automatic normalization
    python generate_tangent_map.py data.csv tsne
    
    # Specify target column and output file
    python generate_tangent_map.py data.csv tsne --target label --output my_data.tmap
    
    # Skip normalization
    python generate_tangent_map.py data.csv umap --no-normalize
    
    # Use different dimensionality reduction method
    python generate_tangent_map.py data.csv pca --target class --output pca_result.tmap
        """
    )
    
    parser.add_argument('input_file', 
                       help='Path to input CSV file')
    parser.add_argument('method', 
                       choices=['tsne', 'umap', 'pca', 'mds'], 
                       default='tsne',
                       help='Dimensionality reduction method (default: tsne)')
    parser.add_argument('--target', '-t', 
                       help='Name of the target/label column in the dataset')
    parser.add_argument('--output', '-o', 
                       help='Output .tmap filename (default: auto-generated)')
    parser.add_argument('--no-normalize', 
                       action='store_true',
                       help='Skip feature normalization')
    parser.add_argument('--keep-intermediate', 
                       action='store_true',
                       help='Keep intermediate files (normalized CSV and raw .tmap)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return 1
    
    # Generate output filename if not specified
    if args.output is None:
        input_path = Path(args.input_file)
        args.output = f"{input_path.stem}_{args.method}.tmap"
    
    try:
        # Step 1: Preprocess the data
        processed_file, target_values, feature_columns = preprocess_data(
            args.input_file, 
            args.target, 
            normalize=not args.no_normalize
        )
        
        # Step 2: Generate tangent map
        raw_tmap_file = generate_tangent_map(processed_file, args.method)
        
        # Step 3: Post-process and finalize
        postprocess_tangent_map(raw_tmap_file, target_values, feature_columns, args.output)
        
        # Clean up intermediate files unless requested to keep them
        if not args.keep_intermediate:
            if os.path.exists(processed_file):
                os.remove(processed_file)
                print(f"Removed intermediate file: {processed_file}")
            if os.path.exists(raw_tmap_file):
                os.remove(raw_tmap_file)
                print(f"Removed intermediate file: {raw_tmap_file}")
        
        print(f"\n✓ Successfully generated tangent map: {args.output}")
        print(f"  You can now use this file with FeatureWind:")
        print(f"  python examples/test.py")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())