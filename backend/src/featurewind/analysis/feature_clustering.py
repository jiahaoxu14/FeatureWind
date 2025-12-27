"""
Feature clustering module for FeatureWind visualization.

This module implements feature-agnostic clustering based on vector field
directional similarity, completely independent of feature names.
"""

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score


def cluster_features_by_direction(grid_u_all_feats, grid_v_all_feats, n_families=6, 
                                 min_magnitude_threshold=1e-6):
    """
    Cluster features based on vector field directional similarity using cosine similarity.
    
    Args:
        grid_u_all_feats: numpy array of shape (M_features, grid_res, grid_res) - U components
        grid_v_all_feats: numpy array of shape (M_features, grid_res, grid_res) - V components  
        n_families: int, target number of families (5-8 recommended for colors)
        min_magnitude_threshold: float, minimum magnitude to consider valid vectors
    
    Returns:
        tuple: (family_assignments, similarity_matrix, clustering_metrics)
            - family_assignments: numpy array of shape (M_features,) with family IDs (0 to n_families-1)
            - similarity_matrix: numpy array of shape (M_features, M_features) with pairwise similarities
            - clustering_metrics: dict with quality metrics
    """
    n_features = grid_u_all_feats.shape[0]
    grid_res = grid_u_all_feats.shape[1]
    
    
    # Combine U and V components into vector fields
    vector_fields = np.stack([grid_u_all_feats, grid_v_all_feats], axis=-1)  # (M, grid_res, grid_res, 2)
    
    # Initialize pairwise similarity matrix
    similarity_matrix = np.zeros((n_features, n_features))
    
    
    # Compute pairwise directional similarity
    for i in range(n_features):
        similarity_matrix[i, i] = 1.0  # Self-similarity
        
        for j in range(i + 1, n_features):
            field_i = vector_fields[i]  # (grid_res, grid_res, 2)
            field_j = vector_fields[j]  # (grid_res, grid_res, 2)
            
            # Compute magnitudes at each grid point
            magnitude_i = np.linalg.norm(field_i, axis=-1)  # (grid_res, grid_res)
            magnitude_j = np.linalg.norm(field_j, axis=-1)  # (grid_res, grid_res)
            
            # Create mask for valid (non-zero magnitude) regions
            valid_mask = (magnitude_i > min_magnitude_threshold) & (magnitude_j > min_magnitude_threshold)
            
            if not np.any(valid_mask):
                # No valid vectors for comparison
                similarity_matrix[i, j] = similarity_matrix[j, i] = 0.0
                continue
            
            # Unit normalize vectors at each grid point (only for valid regions)
            unit_i = np.zeros_like(field_i)
            unit_j = np.zeros_like(field_j)
            
            unit_i[valid_mask] = field_i[valid_mask] / magnitude_i[valid_mask, np.newaxis]
            unit_j[valid_mask] = field_j[valid_mask] / magnitude_j[valid_mask, np.newaxis]
            
            # Compute cosine similarity at each grid point
            cosine_sim = np.sum(unit_i * unit_j, axis=-1)  # (grid_res, grid_res)
            
            # Average cosine similarity over valid regions
            valid_cosine_values = cosine_sim[valid_mask]
            if len(valid_cosine_values) > 0:
                avg_similarity = np.mean(valid_cosine_values)
            else:
                avg_similarity = 0.0  # Default similarity for empty regions
            
            # Ensure similarity is non-negative (handle numerical issues)
            avg_similarity = max(0.0, avg_similarity)
            
            similarity_matrix[i, j] = similarity_matrix[j, i] = avg_similarity
    
    
    # Perform spectral clustering
    
    try:
        clustering = SpectralClustering(
            n_clusters=n_families,
            affinity='precomputed',
            random_state=42,
            n_init=10,
            assign_labels='kmeans'
        )
        
        family_assignments = clustering.fit_predict(similarity_matrix)
        
    except Exception as e:
        pass  # Use fallback k-means clustering
        # Fallback to distance-based clustering
        distance_matrix = 1 - similarity_matrix
        from sklearn.cluster import KMeans
        from sklearn.manifold import MDS
        
        # Use MDS to embed in Euclidean space, then k-means
        mds = MDS(n_components=min(10, n_features-1), dissimilarity='precomputed', random_state=42)
        embeddings = mds.fit_transform(distance_matrix)
        
        kmeans = KMeans(n_clusters=n_families, random_state=42, n_init=10)
        family_assignments = kmeans.fit_predict(embeddings)
    
    # Compute clustering quality metrics
    clustering_metrics = {}
    
    try:
        # Silhouette score (higher is better, range [-1, 1])
        distance_matrix = 1 - similarity_matrix
        silhouette = silhouette_score(distance_matrix, family_assignments, metric='precomputed')
        clustering_metrics['silhouette'] = silhouette
        
    except Exception as e:
        pass  # Could not compute silhouette score
        clustering_metrics['silhouette'] = 0.0
    
    # Compute within-cluster similarity (higher is better)
    within_cluster_similarities = []
    for family_id in range(n_families):
        family_mask = (family_assignments == family_id)
        if np.sum(family_mask) > 1:
            family_similarities = similarity_matrix[np.ix_(family_mask, family_mask)]
            # Average similarity within this family (excluding diagonal)
            mask = ~np.eye(family_similarities.shape[0], dtype=bool)
            within_values = family_similarities[mask]
            if len(within_values) > 0:
                avg_within_similarity = np.mean(within_values)
            else:
                avg_within_similarity = 0.0
            within_cluster_similarities.append(avg_within_similarity)
    
    if within_cluster_similarities:
        clustering_metrics['avg_within_cluster_similarity'] = np.mean(within_cluster_similarities)
    else:
        clustering_metrics['avg_within_cluster_similarity'] = 0.0
    
    # Compute between-cluster similarity (lower is better for separation)
    between_cluster_similarities = []
    for i in range(n_families):
        for j in range(i + 1, n_families):
            mask_i = (family_assignments == i)
            mask_j = (family_assignments == j)
            if np.sum(mask_i) > 0 and np.sum(mask_j) > 0:
                between_similarities = similarity_matrix[np.ix_(mask_i, mask_j)]
                if between_similarities.size > 0:
                    avg_between_similarity = np.mean(between_similarities)
                    between_cluster_similarities.append(avg_between_similarity)
    
    if between_cluster_similarities:
        clustering_metrics['avg_between_cluster_similarity'] = np.mean(between_cluster_similarities)
    else:
        clustering_metrics['avg_between_cluster_similarity'] = 0.0
    
    # Print results
    unique_families, family_counts = np.unique(family_assignments, return_counts=True)
    
    
    return family_assignments, similarity_matrix, clustering_metrics


def auto_select_n_families(grid_u_all_feats, grid_v_all_feats, k_range=(4, 8), 
                          min_magnitude_threshold=1e-6):
    """
    Automatically select the optimal number of families using silhouette analysis.
    
    Args:
        grid_u_all_feats: numpy array of shape (M_features, grid_res, grid_res)
        grid_v_all_feats: numpy array of shape (M_features, grid_res, grid_res)
        k_range: tuple, range of k values to test (min_k, max_k)
        min_magnitude_threshold: float, minimum magnitude threshold
    
    Returns:
        tuple: (optimal_k, all_scores)
            - optimal_k: int, best number of families
            - all_scores: dict, silhouette scores for each k tested
    """
    n_features = grid_u_all_feats.shape[0]
    min_k, max_k = k_range
    
    # Ensure we don't exceed the number of features
    max_k = min(max_k, n_features - 1)
    min_k = min(min_k, max_k)
    
    print(f"Auto-selecting optimal number of families from range [{min_k}, {max_k}]...")
    
    scores = {}
    
    for k in range(min_k, max_k + 1):
        print(f"  Testing k={k}...")
        
        try:
            family_assignments, similarity_matrix, metrics = cluster_features_by_direction(
                grid_u_all_feats, grid_v_all_feats, n_families=k, 
                min_magnitude_threshold=min_magnitude_threshold
            )
            
            scores[k] = metrics['silhouette']
            print(f"    k={k}: silhouette={metrics['silhouette']:.3f}")
            
        except Exception as e:
            print(f"    k={k}: failed ({e})")
            scores[k] = -1.0
    
    # Select k with highest silhouette score
    if scores:
        optimal_k = max(scores.keys(), key=lambda k: scores[k])
        print(f"✓ Optimal number of families: {optimal_k} (silhouette={scores[optimal_k]:.3f})")
        return optimal_k, scores
    else:
        print("Warning: Could not determine optimal k, using default k=6")
        return 6, {}


def analyze_feature_families(family_assignments, col_labels, similarity_matrix=None):
    """
    Analyze and report the characteristics of feature families.
    
    Args:
        family_assignments: numpy array of family IDs for each feature
        col_labels: list of feature names
        similarity_matrix: optional similarity matrix for detailed analysis
    
    Returns:
        dict: Analysis results with family characteristics
    """
    n_features = len(family_assignments)
    unique_families = np.unique(family_assignments)
    
    analysis = {
        'n_families': len(unique_families),
        'family_sizes': {},
        'family_members': {},
        'family_representatives': {}
    }
    
    for family_id in unique_families:
        family_mask = (family_assignments == family_id)
        family_indices = np.where(family_mask)[0]
        family_size = np.sum(family_mask)
        
        family_features = [col_labels[i] for i in family_indices]
        
        analysis['family_sizes'][family_id] = family_size
        analysis['family_members'][family_id] = family_features
        
        # Find representative feature (most similar to others in family)
        if similarity_matrix is not None and family_size > 1:
            family_similarities = similarity_matrix[np.ix_(family_indices, family_indices)]
            if family_similarities.size > 0:
                avg_similarities = np.mean(family_similarities, axis=1)
            else:
                avg_similarities = np.zeros(len(family_indices))
            representative_idx = family_indices[np.argmax(avg_similarities)]
            representative_name = col_labels[representative_idx]
            analysis['family_representatives'][family_id] = representative_name
    
    return analysis


def get_family_summary_names(family_assignments, col_labels, max_name_length=15):
    """
    Generate short, descriptive names for each family based on feature names.
    
    Args:
        family_assignments: numpy array of family IDs
        col_labels: list of feature names
        max_name_length: maximum length for family names
    
    Returns:
        dict: {family_id: descriptive_name}
    """
    unique_families = np.unique(family_assignments)
    family_names = {}
    
    for family_id in unique_families:
        family_indices = np.where(family_assignments == family_id)[0]
        family_features = [col_labels[i] for i in family_indices]
        
        if len(family_features) == 1:
            # Single feature family - use truncated feature name
            name = family_features[0][:max_name_length]
            
        else:
            # Multi-feature family - try to find common patterns
            # Look for common prefixes
            if len(family_features) > 1:
                # Find common words/prefixes
                words_sets = [set(name.lower().split()) for name in family_features]
                common_words = set.intersection(*words_sets) if words_sets else set()
                
                if common_words:
                    # Use most meaningful common word
                    meaningful_words = [w for w in common_words 
                                      if len(w) > 2 and w not in ['the', 'and', 'for']]
                    if meaningful_words:
                        name = meaningful_words[0].title()
                    else:
                        name = list(common_words)[0].title()
                else:
                    # No common words, use generic name
                    name = f"Group {family_id+1}"
            else:
                name = f"Family {family_id+1}"
        
        # Ensure name fits length limit
        if len(name) > max_name_length:
            name = name[:max_name_length-2] + ".."
        
        family_names[family_id] = name
    
    return family_names


if __name__ == "__main__":
    # Test the clustering with synthetic data
    print("Testing feature clustering with synthetic data...")
    
    # Create synthetic vector fields with known structure
    grid_res = 20
    n_features = 12
    
    # Generate test vector fields
    np.random.seed(42)
    grid_u_test = np.random.randn(n_features, grid_res, grid_res) * 0.1
    grid_v_test = np.random.randn(n_features, grid_res, grid_res) * 0.1
    
    # Add structured patterns
    x, y = np.meshgrid(np.linspace(-1, 1, grid_res), np.linspace(-1, 1, grid_res))
    
    # Family 1: Radial patterns (features 0-3)
    for i in range(4):
        grid_u_test[i] = x + np.random.randn(grid_res, grid_res) * 0.05
        grid_v_test[i] = y + np.random.randn(grid_res, grid_res) * 0.05
    
    # Family 2: Circular patterns (features 4-7)  
    for i in range(4, 8):
        grid_u_test[i] = -y + np.random.randn(grid_res, grid_res) * 0.05
        grid_v_test[i] = x + np.random.randn(grid_res, grid_res) * 0.05
    
    # Family 3: Uniform patterns (features 8-11)
    for i in range(8, 12):
        grid_u_test[i] = np.ones((grid_res, grid_res)) + np.random.randn(grid_res, grid_res) * 0.05
        grid_v_test[i] = np.ones((grid_res, grid_res)) * 0.5 + np.random.randn(grid_res, grid_res) * 0.05
    
    # Test clustering
    family_assignments, similarity_matrix, metrics = cluster_features_by_direction(
        grid_u_test, grid_v_test, n_families=3
    )
    
    # Create test feature labels
    test_labels = [f"test_feature_{i}" for i in range(n_features)]
    
    # Analyze results
    analysis = analyze_feature_families(family_assignments, test_labels, similarity_matrix)
    
    print(f"\nTest completed. Expected 3 families, got {analysis['n_families']}")
    print(f"Family assignments: {family_assignments}")