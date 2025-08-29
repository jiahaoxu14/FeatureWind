import sys
# import DualNum
import csv
import numpy as np
from .tsne import tsne
import torch

class ProjectionRunner:
    def __init__(self, projection, params=None):
        self.params = params
        self.projection = projection
        self.firstRun = False

    def calculateValues(self, points, perturbations=None):
        self.points = points
        # self.origPoints = points

        # Convert data to PyTorch tensor with requires_grad=True
        data = torch.tensor(points, dtype=torch.float32, requires_grad=True)

        # Usage for DimReader:
        print("Step 1/3: Computing base t-SNE projection (999 iterations)...")
        with torch.no_grad():
            Y_base, params = tsne(data, 2, 999, 50, 30.0, save_params = True)
        print("Step 2/3: Computing projection with gradients (1 iteration)...")
        Y, params = tsne(data, no_dims=2, maxIter = 1, initial_dims=50, perplexity=30.0, save_params = False,
                            initY = params[0], initBeta = params[2], betaTries = 50, initIY =params[1])
        
        # Compute gradients and full Jacobian matrix
        self.outPoints = Y
        grads = []
        n_points, n_features = data.shape

        print(f"Step 3/3: Computing gradients for {n_points} points...")
        # Initialize Jacobian matrix: J[2*i:2*i+2, :] = gradients for point i
        self.jacobian = torch.zeros(2 * n_points, n_features, dtype=torch.float32)

        for i in range(len(Y)):
            # Show progress every 10% or every 50 points, whichever is more frequent
            progress_interval = min(50, max(1, n_points // 10))
            if i % progress_interval == 0 or i == n_points - 1:
                progress_pct = (i + 1) / n_points * 100
                print(f"  Computing gradients: {i+1}/{n_points} points ({progress_pct:.1f}%)")
            grad_x = torch.autograd.grad(Y[i, 0], data, retain_graph=True)[0][i]
            grad_y = torch.autograd.grad(Y[i, 1], data, retain_graph=True)[0][i]
            grads.append(torch.stack([grad_x, grad_y], dim=0))
            
            # Store in Jacobian matrix
            self.jacobian[2*i, :] = grad_x
            self.jacobian[2*i+1, :] = grad_y

        # Convert gradients to NumPy array (backward compatibility)
        self.grads = torch.stack(grads).detach().numpy()
        
        # Store Jacobian as NumPy array for easier integration
        self.jacobian_numpy = self.jacobian.detach().numpy()
        print("✓ Tangent map generation completed successfully!")

    def get_jacobian_for_point(self, point_idx):
        """Get the 2x d Jacobian matrix for a specific point."""
        if self.jacobian is None:
            raise ValueError("Jacobian not computed. Run calculateValues first.")
        
        return self.jacobian_numpy[2*point_idx:2*point_idx+2, :]
    
    def compute_metric_tensor(self, point_idx):
        """Compute the pullback metric G = J^T * J for a specific point."""
        J_i = self.get_jacobian_for_point(point_idx)
        return np.dot(J_i.T, J_i)
    
    def pushforward_vector(self, point_idx, high_d_vector):
        """Apply pushforward: map high-D vector to 2D using Jacobian."""
        J_i = self.get_jacobian_for_point(point_idx)
        return np.dot(J_i, high_d_vector)
    
    def metric_normalized_pushforward(self, point_idx, high_d_vector):
        """Apply metric-aware pushforward with normalization."""
        J_i = self.get_jacobian_for_point(point_idx)
        v_2d = np.dot(J_i, high_d_vector)
        
        # Compute metric tensor for normalization
        G = self.compute_metric_tensor(point_idx)
        
        # Normalize using metric (simplified: use trace for scaling)
        metric_scale = np.sqrt(np.trace(G))
        if metric_scale > 1e-10:
            v_2d = v_2d / metric_scale
            
        return v_2d

projections = ["tsne", "Tangent-Map"]
projectionClasses = [tsne, None]
projectionParamOpts = [["Perplexity", "Max_Iterations", "Number_of_Dimensions"], []]

# read points of cvs file
def readFile(filename):
    read = csv.reader(open(filename, 'rt'))

    points = []
    firstLine = next(read)
    headers = []
    rowDat = []
    head = False
    for i in range(0, len(firstLine)):
        try:
            rowDat.append(float(firstLine[i]))
        except:
            head = True
            break
    if head:
        headers = firstLine
    else:
        points.append(rowDat)

    for row in read:
        rowDat = []
        for i in range(0, len(row)):
            try:
                rowDat.append(float(row[i]))
            except:
                print("invalid data type - must be numeric")
                exit(0)
        points.append(rowDat)
    return points