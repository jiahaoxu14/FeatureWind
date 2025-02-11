import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class TangentPoint:
    """
    Represents a point with associated gradient vectors and computes properties
    such as anisotropy index, convex hull, and ellipse area based on the gradients.
    """
    def __init__(self, tmap, scale_factor, feature_names):
        """
        Initializes the Point object.

        Parameters:
        - tmap: dict containing 'range', 'class', and 'tangent' keys
            - 'range': list or array-like with at least two elements representing position
            - 'class': label of the point
            - 'tangent': array of tangent vectors (2 x N)
        - scale_factor: float, factor to scale gradient vectors
        - feature_names: list of feature names corresponding to gradients
        """

        # True if the dataset is prelabelled
        self.label = False

        # Start processing
        self.position = np.array(tmap['range'][:2])  # Shape: (2,)
        self.tmap_label = tmap['class']
        
        # Extract gradient (tangent) vectors and feature names
        self.gradient_vectors = np.array(tmap['tangent']).T  # Shape: (N, 2)
        self.feature_names = feature_names  # List of feature names
        self.gradient_vectors_negative = -self.gradient_vectors
        
        # Ensure there are enough vectors
        num_vectors = self.gradient_vectors.shape[0]
        if num_vectors < 2:
            self.valid = False
            return
        else:
            self.valid = True

        # Store scale factor
        self.scale_factor = scale_factor

        # Compute various properties
        self.scale_gradient_vectors(self.scale_factor)
        self.compute_endpoints()
        self.compute_combined_vectors()
        self.compute_gradient_sum()
        self.compute_covariance_matrix()
        self.perform_eigen_decomposition()
        self.compute_anisotropy_index()
        self.compute_convex_hull()
        self.compute_ellipse_area()

    def scale_gradient_vectors(self, scale_factor):
        """Scale gradient vectors by the scale factor."""
        self.gradient_vectors_scaled = self.gradient_vectors * scale_factor
        self.gradient_vectors_negative_scaled = self.gradient_vectors_negative * scale_factor

    def compute_endpoints(self):
        """Compute the endpoints of the scaled gradient vectors."""
        self.endpoints = self.position + self.gradient_vectors_scaled  # Shape: (N, 2)
        self.endpoints_negative = self.position + self.gradient_vectors_negative_scaled

    def compute_combined_vectors(self):
        """Combine positive and negative scaled gradient vectors and endpoints."""
        self.combined_vectors_scaled = np.vstack((self.gradient_vectors_scaled, self.gradient_vectors_negative_scaled))
        self.combined_endpoints = np.vstack((self.endpoints, self.endpoints_negative))

    def compute_gradient_sum(self):
        """Compute the sum of gradient vectors and its norm."""
        gradient_sum = np.sum(self.combined_vectors_scaled, axis=0)
        self.gradient_sum = np.linalg.norm(gradient_sum)

    def compute_covariance_matrix(self):
        """Compute the covariance matrix of the combined scaled gradient vectors."""
        self.covariance_matrix = np.cov(self.combined_vectors_scaled, rowvar=False)

    def perform_eigen_decomposition(self):
        """Perform eigenvalue decomposition on the covariance matrix."""
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_matrix)
        # Sort eigenvalues and eigenvectors in descending order
        idx_sorted = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx_sorted]
        self.eigenvectors = eigenvectors[:, idx_sorted]

    def compute_anisotropy_index(self):
        """Compute the anisotropy index based on eigenvalues."""
        lambda_max = self.eigenvalues[0]
        lambda_min = self.eigenvalues[1]
        if lambda_max == 0:
            self.anisotropy_index = 0  # Handle zero eigenvalue
        else:
            self.anisotropy_index = np.sqrt(1 - lambda_min / lambda_max)

    def compute_convex_hull(self):
        """Compute the convex hull of the combined endpoints."""
        self.hull = ConvexHull(self.combined_endpoints)
        self.boundary_indices_set = set(self.hull.vertices)

    def compute_ellipse_area(self):
        """Compute the area of the ellipse defined by the eigenvalues."""
        a = np.sqrt(self.eigenvalues[0])  # Semi-major axis length
        b = np.sqrt(self.eigenvalues[1])  # Semi-minor axis length
        self.ellipse_area = np.pi * a * b  # Ellipse area

    def update_scale_factor(self, new_scale_factor):
        """Update the scale factor and recompute dependent properties."""
        self.scale_factor = new_scale_factor
        self.scale_gradient_vectors(self.scale_factor)
        self.compute_endpoints()
        self.compute_combined_vectors()
        self.compute_gradient_sum()
        self.compute_covariance_matrix()
        self.perform_eigen_decomposition()
        self.compute_anisotropy_index()
        self.compute_convex_hull()
        self.compute_ellipse_area()

    def plot(self, 
             ax=None, 
             show_point=True, 
             show_ellipse=False, 
             ellipse_alpha = 1.0,  
             convex_hull=False, 
             vector=False, 
             vector_label=False, 
             point_color='white', 
             ellipse_color='white', 
             ellipse_face_color=None):
        """
        Plots the point's gradient vectors, convex hull, and anisotropy ellipse.

        Parameters:
        - ax: Matplotlib Axes object to plot on. If None, a new figure and axes are created.
        - ellipse: Boolean flag to plot the anisotropy ellipse.
        - convex_hull: Boolean flag to plot the convex hull boundary.
        - vector: Boolean flag to plot the gradient vectors.
        - vector_label: Boolean flag to label the gradient vectors.
        - point_color: Color for plotting the point and ellipse.
        """
        if not self.valid:
            print("Invalid Point: Not enough gradient vectors to plot.")
            return

        # Create a new figure and axis if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        # Unpack necessary data
        position = self.position
        endpoints = self.endpoints
        endpoints_negative = self.endpoints_negative
        combined_endpoints = self.combined_endpoints
        feature_names = self.feature_names
        eigenvalues = self.eigenvalues
        eigenvectors = self.eigenvectors
        anisotropy_value = self.anisotropy_index
        label = self.tmap_label
        hull = self.hull
        ellipse_area = self.ellipse_area

        # Plot gradient vectors
        if vector:
            num_vectors = self.gradient_vectors_scaled.shape[0]
            for idx in range(num_vectors):
                # Positive gradient vectors
                endpoint = endpoints[idx]
                feature_name = feature_names[idx] if idx < len(feature_names) else f'Feature{idx+1}'
                ax.plot(
                    [position[0], endpoint[0]],
                    [position[1], endpoint[1]],
                    color='red',
                    linewidth=1.5,
                    zorder=2
                )
                if vector_label:
                    direction = endpoint - position
                    norm = np.linalg.norm(direction)
                    if norm != 0:
                        direction /= norm
                        # Compute a perpendicular direction for label offset
                        perp_direction = np.array([-direction[1], direction[0]])
                        # Adjust the offset magnitude as needed
                        label_offset = 0.2 * self.scale_factor * perp_direction
                        label_pos = endpoint + label_offset
                        ax.text(
                            label_pos[0],
                            label_pos[1],
                            feature_name,
                            fontsize=8,
                            color='red',
                            ha='center',
                            va='center'
                        )
                # Negative gradient vectors
                endpoint_neg = endpoints_negative[idx]
                feature_name_neg = feature_name + '_neg'
                ax.plot(
                    [position[0], endpoint_neg[0]],
                    [position[1], endpoint_neg[1]],
                    color='orange',
                    linewidth=1.5,
                    zorder=2
                )
                if vector_label:
                    direction = endpoint_neg - position
                    norm = np.linalg.norm(direction)
                    if norm != 0:
                        direction /= norm
                        # Compute a perpendicular direction for label offset
                        perp_direction = np.array([-direction[1], direction[0]])
                        # Adjust the offset magnitude as needed
                        label_offset = 0.2 * self.scale_factor * perp_direction
                        label_pos = endpoint_neg + label_offset
                        ax.text(
                            label_pos[0],
                            label_pos[1],
                            feature_name_neg,
                            fontsize=8,
                            color='orange',
                            ha='center',
                            va='center'
                        )

        # Plot the convex hull boundary
        if convex_hull:
            hull_path = combined_endpoints[hull.vertices]
            hull_path = np.vstack([hull_path, hull_path[0]])  # Close the loop
            ax.plot(
                hull_path[:, 0],
                hull_path[:, 1],
                color='blue',
                linestyle='--',
                linewidth=1.5,
                label='Convex Hull'
            )

        # Plot the anisotropy ellipse
        if show_ellipse:
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * np.sqrt(eigenvalues)
            ellipse_patch = Ellipse(
                xy=position,
                width=width,
                height=height,
                angle=angle,
                edgecolor=ellipse_color,
                facecolor=ellipse_face_color,
                linewidth=2,
                zorder=4,
                alpha=ellipse_alpha
            )
            ax.add_patch(ellipse_patch)

        # Plot the point's position
        if show_point:
            ax.plot(position[0], position[1], 'o', markersize=4, color=point_color, zorder=5)

        # Set plot labels and title
        ax.set_title(
            f"Label: {label}\nAnisotropy Index: {anisotropy_value:.2f}\nEllipse Area: {ellipse_area:.2f}",
            fontsize=10
        )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.grid(True)

        # Add legend if convex hull is plotted
        if convex_hull:
            ax.legend()

        # Show the plot if created here
        if ax is None:
            plt.tight_layout()
            plt.show()