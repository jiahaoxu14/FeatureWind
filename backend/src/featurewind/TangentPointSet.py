import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import TangentPoint

class TangentPointSet:
    """
    Represents a cluster of Point objects sharing the same class label.
    Computes aggregated properties such as convex hull and anisotropy ellipse based on aggregated gradient vectors.
    Utilizes the Point class's plot method for visualization.
    """
    def __init__(self, label):
        """
        Initializes the Cluster object.
        
        Parameters:
        - points: List of Point objects belonging to this cluster.
        - label: The class label of the cluster.
        """
        self.label = label
        self.points = []

    def add_point(self, point):
        """
        Adds a Point object to the cluster.
        
        Parameters:
        - point: The Point object to add.
        """
        if point.tmap_label == self.label:
            self.points.append(point)
        else:
            print(f"Point with label {point.tmap_label} does not belong to cluster '{self.label}'.")

    def compute_aggregated_properties(self, scale_factor=15.0):
        # Calculate mean position for all points
        positions = np.array([point.position for point in self.points])
        self.position = positions.mean(axis=0)
            
        # Assume all points have the same number of features
        self.feature_names = self.points[0].feature_names
        num_features = len(self.feature_names)
        
        # Initialize list to store aggregated gradients per feature
        self.aggregated_gradients = np.zeros((num_features, 2))  # Shape: (num_features, 2)
        
        for feature_idx in range(num_features):
            # Collect all gradient vectors for this feature across points
            feature_gradients = []
            for point in self.points:
                if feature_idx < point.gradient_vectors_scaled.shape[0]:
                    feature_gradients.append(point.gradient_vectors_scaled[feature_idx])
                else:
                    print(f"Point with label {self.label} missing feature index {feature_idx}.")
            
            if feature_gradients:
                # Aggregate by averaging
                self.aggregated_gradients[feature_idx] = np.mean(feature_gradients, axis=0)
            else:
                # If no gradients found for this feature, leave as zero
                self.aggregated_gradients[feature_idx] = np.zeros(2)

        # Prepare a temporary tmap for the aggregated Point
        aggregated_tmap = {
            'range': self.position,  # Aggregated gradient vectors
            'class': self.label,
            'tangent': self.aggregated_gradients.T  # Shape: (2, N_features)
        }

        # Create a temporary Point instance for aggregated properties
        self.aggregated_point = TangentPoint.TangentPoint(aggregated_tmap, scale_factor, self.feature_names)

        self.anisotropy_index = self.aggregated_point.anisotropy_index

        self.ellipse_area = self.aggregated_point.ellipse_area
    
    def plot(self, 
             ax=None, 
             show_ellipse=True, 
             ellipse_color='none', 
             ellipse_face_color='none', 
             show_legend=True, 
             plot_individual_points=False, 
             point_color=None, 
             aggregated_color=None):
        # Create a new figure and axis if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # --- Plot Aggregated Properties Using Point's Plot Method ---
        
        # Plot the aggregated Point
        self.aggregated_point.plot(ax=ax, 
                                   show_ellipse=show_ellipse, 
                                   ellipse_color=ellipse_color, 
                                   ellipse_face_color=ellipse_face_color, 
                                   convex_hull=True, 
                                   vector=True, 
                                   vector_label=True, 
                                   point_color=aggregated_color)
        
        # --- Optionally Plot Individual Points ---
        if plot_individual_points:
            for point in self.points:
                point.plot(ax=ax, point_color=point_color)
        
        # --- Customize Overall Plot ---
        ax.set_xlabel('Gradient Vector X Component', fontsize=12)
        ax.set_ylabel('Gradient Vector Y Component', fontsize=12)
        ax.axis('equal')
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True)
        
        # Add legend
        if show_legend:
            ax.legend()
        
        # Show the plot if a new figure was created
        if ax is None:
            plt.tight_layout()
            plt.show()