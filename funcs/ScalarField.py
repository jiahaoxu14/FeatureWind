import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
from matplotlib.tri import Triangulation

def reconstruct_scalar_field(points, grid_size=100, lambda_reg=1e-4, feature_index=0):
    """
    Reconstructs a scalar field from point positions and gradient vectors.

    Parameters:
    - points: List of Point objects with attributes 'valid', 'position', 'gradient_vectors'.
    - grid_size: Integer, the number of divisions in the grid (default is 100).
    - lambda_reg: Float, regularization parameter (default is 1e-4).
    - feature_index: Integer, the index feature vector

    Returns:
    - f_grid: 2D NumPy array of scalar field values reshaped to the grid.
    - X_grid: 2D NumPy array of x coordinates matching f_grid.
    - Y_grid: 2D NumPy array of y coordinates matching f_grid.
    - contour_levels: 1D NumPy array of contour levels used.
    """
    # Step 1: Data Preparation
    positions = []
    gradient_vectors = []
    
    for point in points:
        if point.valid:
            positions.append(point.position)
            # Select one gradient vector per point (e.g., the one with the largest magnitude)
            gradients = point.gradient_vectors
            selected_gradient = gradients[feature_index]
            gradient_vectors.append(selected_gradient)
    
    positions = np.array(positions)
    gradient_vectors = np.array(gradient_vectors)
    
    # Ensure there are valid points
    if positions.size == 0:
        raise ValueError("No valid points found.")
    
    # Step 2: Grid Construction
    x_min, y_min = positions.min(axis=0) - 0.1
    x_max, y_max = positions.max(axis=0) + 0.1
    x_grid = np.linspace(x_min, x_max, grid_size + 1)
    y_grid = np.linspace(y_min, y_max, grid_size + 1)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    vertices = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    # Construct the triangles
    triangles = []
    for i in range(grid_size):
        for j in range(grid_size):
            idx_bl = i * (grid_size + 1) + j
            idx_br = idx_bl + 1
            idx_tl = idx_bl + (grid_size + 1)
            idx_tr = idx_tl + 1
            triangles.append([idx_bl, idx_tl, idx_tr])
            triangles.append([idx_bl, idx_tr, idx_br])
    triangles = np.array(triangles)
    
    # Triangulation and TriFinder
    triang = Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    trifinder = triang.get_trifinder()
    
    # Step 3: Scalar Field Modeling
    point_indices = []
    for pos in positions:
        triangle_index = trifinder(pos[0], pos[1])
        if triangle_index == -1:
            print("Point outside grid:", pos)
            point_indices.append(None)
        else:
            point_indices.append(triangle_index)
    
    def compute_shape_function_gradients(tri_vertices):
        x = tri_vertices[:, 0]
        y = tri_vertices[:, 1]
        area = 0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if area == 0:
            raise ValueError("Degenerate triangle with zero area")
        b = np.zeros(3)
        c = np.zeros(3)
        b[0] = (y[1] - y[2]) / (2 * area)
        b[1] = (y[2] - y[0]) / (2 * area)
        b[2] = (y[0] - y[1]) / (2 * area)
        c[0] = (x[2] - x[1]) / (2 * area)
        c[1] = (x[0] - x[2]) / (2 * area)
        c[2] = (x[1] - x[0]) / (2 * area)
        grad_phi = np.array([b, c]).T
        return grad_phi
    
    # Step 4: Linear Constraints Setup
    num_vertices = vertices.shape[0]
    A = lil_matrix((2 * len(positions), num_vertices))
    b_vector = np.zeros(2 * len(positions))
    row = 0
    
    for idx_point, pos in enumerate(positions):
        triangle_index = point_indices[idx_point]
        if triangle_index is None or triangle_index == -1:
            continue
        tri_vertices_indices = triangles[triangle_index]
        tri_vertices_positions = vertices[tri_vertices_indices]
        grad_phi = compute_shape_function_gradients(tri_vertices_positions)
        B = grad_phi.T
        cols = tri_vertices_indices
        g = gradient_vectors[idx_point]
        for comp in range(2):
            A[row, cols] = B[comp, :]
            b_vector[row] = g[comp]
            row += 1
    
    # Step 5: Least-Squares Solution
    num_reg_eqs = num_vertices
    A_reg = lil_matrix((num_reg_eqs, num_vertices))
    b_reg = np.zeros(num_reg_eqs)
    A_reg.setdiag(lambda_reg * np.ones(num_vertices))
    A_total = lil_matrix((A.shape[0] + A_reg.shape[0], num_vertices))
    A_total[:A.shape[0], :] = A
    A_total[A.shape[0]:, :] = A_reg
    b_total = np.concatenate([b_vector, b_reg])
    A_total_csr = A_total.tocsr()
    result = lsqr(A_total_csr, b_total)
    f = result[0]
    
    # Reshape scalar field to grid
    f_grid = f.reshape((grid_size + 1, grid_size + 1))
    
    # Generate contour levels for visualization
    num_levels = 20
    contour_levels = np.linspace(np.nanmin(f_grid), np.nanmax(f_grid), num_levels)
    
    return f_grid, X_grid, Y_grid, contour_levels