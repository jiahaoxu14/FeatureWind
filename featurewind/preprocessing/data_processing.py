"""
Data processing module for FeatureWind.

This module handles loading tangent map data, preprocessing, and feature selection.
"""

import json
import numpy as np
from ..core.tangent_point import TangentPoint
from .. import config


def preprocess_tangent_map(tangent_map_path):
    """
    Load and preprocess tangent map data from a .tmap file.
    
    Args:
        tangent_map_path (str): Path to the tangent map file
        
    Returns:
        tuple: (valid_points, all_grad_vectors, all_positions, col_labels)
            - valid_points: List of valid TangentPoint objects
            - all_grad_vectors: numpy array of shape (#points, M, 2) with gradient vectors
            - all_positions: numpy array of shape (#points, 2) with 2D positions  
            - col_labels: List of feature column labels
    """
    with open(tangent_map_path, "r") as f:
        data_import = json.loads(f.read())

    tmap = data_import['tmap']
    col_labels = data_import['Col_labels']
    
    points = []
    for tmap_entry in tmap:
        point = TangentPoint(tmap_entry, 1.0, col_labels)
        points.append(point)
    
    valid_points = [p for p in points if p.valid]
    all_positions = np.array([p.position for p in valid_points])  # shape: (#points, 2)
    all_grad_vectors = [p.gradient_vectors for p in valid_points]  # list of (#features, 2)
    all_grad_vectors = np.array(all_grad_vectors)  # shape = (#points, M, 2)

    return valid_points, all_grad_vectors, all_positions, col_labels


def pick_top_k_features(all_grad_vectors, k=None):
    """
    Select the top k features based on average gradient magnitude across all points.
    
    Args:
        all_grad_vectors (np.ndarray): Gradient vectors, shape (#points, M, 2)
        k (int, optional): Number of top features to select. If None, uses config.k
        
    Returns:
        tuple: (top_k_indices, avg_magnitudes)
            - top_k_indices: Array of indices of top k features
            - avg_magnitudes: Array of average magnitudes for all features
    """
    if k is None:
        k = config.k
    
    # Compute average magnitude of each feature across all points
    feature_magnitudes = np.linalg.norm(all_grad_vectors, axis=2)  # shape (#points, M)
    avg_magnitudes = feature_magnitudes.mean(axis=0)  # shape (M,)

    # Sort descending, get top k indices
    top_k_indices = np.argsort(-avg_magnitudes)[:k]
    return top_k_indices, avg_magnitudes


def validate_data(valid_points, all_grad_vectors, all_positions, col_labels):
    """
    Validate the consistency of loaded data structures.
    
    Args:
        valid_points: List of TangentPoint objects
        all_grad_vectors: Gradient vectors array
        all_positions: Positions array  
        col_labels: Feature labels list
        
    Returns:
        bool: True if data is consistent, raises ValueError otherwise
    """
    n_points = len(valid_points)
    n_features = len(col_labels)
    
    if all_positions.shape != (n_points, 2):
        raise ValueError(f"Position array shape {all_positions.shape} doesn't match {n_points} points")
    
    if all_grad_vectors.shape != (n_points, n_features, 2):
        raise ValueError(f"Gradient array shape {all_grad_vectors.shape} doesn't match expected ({n_points}, {n_features}, 2)")
    
    return True


def get_feature_statistics(all_grad_vectors, col_labels, top_k_indices=None):
    """
    Compute statistics for features in the dataset.
    
    Args:
        all_grad_vectors (np.ndarray): Gradient vectors, shape (#points, M, 2)
        col_labels (list): Feature column labels
        top_k_indices (np.ndarray, optional): Indices of top features to analyze
        
    Returns:
        dict: Dictionary with feature statistics
    """
    feature_magnitudes = np.linalg.norm(all_grad_vectors, axis=2)  # shape (#points, M)
    avg_magnitudes = feature_magnitudes.mean(axis=0)  # shape (M,)
    std_magnitudes = feature_magnitudes.std(axis=0)
    max_magnitudes = feature_magnitudes.max(axis=0)
    min_magnitudes = feature_magnitudes.min(axis=0)
    
    stats = {
        'avg_magnitudes': avg_magnitudes,
        'std_magnitudes': std_magnitudes,
        'max_magnitudes': max_magnitudes,
        'min_magnitudes': min_magnitudes,
        'feature_labels': col_labels
    }
    
    if top_k_indices is not None:
        stats['top_k_indices'] = top_k_indices
        stats['top_k_labels'] = [col_labels[i] for i in top_k_indices]
        stats['top_k_avg_magnitudes'] = avg_magnitudes[top_k_indices]
    
    return stats


def find_feature_by_name(col_labels, feature_name, case_sensitive=False):
    """
    Find the index of a feature by name.
    
    Args:
        col_labels (list): List of feature column labels
        feature_name (str): Name to search for
        case_sensitive (bool): Whether the search should be case sensitive
        
    Returns:
        int or None: Index of the feature if found, None otherwise
    """
    for i, label in enumerate(col_labels):
        if case_sensitive:
            if feature_name in label:
                return i
        else:
            if feature_name.lower() in label.lower():
                return i
    return None


# Legacy function name for backwards compatibility
def PreProcessing(tangentmaps):
    """
    Legacy wrapper for preprocess_tangent_map function.
    
    Args:
        tangentmaps (str): Path to tangent map file
        
    Returns:
        tuple: Same as preprocess_tangent_map
    """
    return preprocess_tangent_map(tangentmaps)