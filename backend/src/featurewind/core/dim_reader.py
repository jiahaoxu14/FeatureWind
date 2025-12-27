import sys
# import DualNum
import csv
import numpy as np
from .tsne import tsne
from .mds_torch import mds, distance_matrix_HD_tensor
import torch

class ProjectionRunner:
    def __init__(self, projection, params=None):
        self.params = params
        self.projection = projection
        self.firstRun = False

    def calculateValues(self, points, perturbations=None):
        self.points = points
        # self.origPoints = points

        # Choose best available device (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using device: CUDA")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using device: MPS")
        else:
            device = torch.device('cpu')
            print("Using device: CPU")

        # Convert data to PyTorch tensor on selected device with requires_grad=True
        data = torch.tensor(points, dtype=torch.float32, device=device, requires_grad=True)

        # Usage for DimReader:
        if self.projection == tsne or (isinstance(self.projection, str) and self.projection.lower() == 'tsne'):
            # Optional perplexity override via self.params (supports 'perplexity=NN', 'perp=NN', or a lone number)
            perp_override = None
            try:
                if self.params:
                    for p in self.params:
                        s = str(p).strip()
                        if '=' in s:
                            k, v = s.split('=', 1)
                            if k.strip().lower() in ('perplexity', 'perp'):
                                perp_override = float(v)
                        else:
                            # no key, treat as numeric perplexity if possible
                            try:
                                perp_override = float(s)
                            except Exception:
                                pass
            except Exception:
                perp_override = perp_override

            perp_base = perp_override if (isinstance(perp_override, (int, float)) and perp_override > 0) else 40.0
            perp_grad = perp_override if (isinstance(perp_override, (int, float)) and perp_override > 0) else 40.0

            print("Step 1/3: Computing base t-SNE projection (1000 iterations)...")
            with torch.no_grad():
                Y_base, params = tsne(data, 2, 1000, 10, perp_base, save_params = True)
            print("Step 2/3: Computing projection with gradients (1 iteration)...")
            Y, params = tsne(data, no_dims=2, maxIter = 1, initial_dims=10, perplexity=perp_grad, save_params = False,
                                initY = params[0], initBeta = params[2], betaTries = 50, initIY =params[1])
        elif self.projection == mds or (isinstance(self.projection, str) and self.projection.lower() == 'mds'):
            print("Step 1/3: Computing base MDS projection (999 iterations)...")
            with torch.no_grad():
                dist_hd = distance_matrix_HD_tensor(data)
                Y_base = mds(dist_hd, n_components=2, max_iter=999)
            print("Step 2/3: Computing projection with gradients (1 iteration)...")
            # Recompute with graph enabled so autograd can backprop to input coordinates
            dist_hd = distance_matrix_HD_tensor(data)
            Y = mds(dist_hd, n_components=2, max_iter=1)
        else:
            raise ValueError("Unsupported projection method. Use 'tsne' or 'mds'.")
        
        # Compute gradients and full Jacobian matrix
        self.outPoints = Y
        grads = []
        n_points, n_features = data.shape

        print(f"Step 3/3: Computing gradients for {n_points} points...")
        # Initialize Jacobian matrix: J[2*i:2*i+2, :] = gradients for point i
        self.jacobian = torch.zeros(2 * n_points, n_features, dtype=torch.float32, device=device)

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
        self.grads = torch.stack(grads).detach().cpu().numpy()
        
        # Store Jacobian as NumPy array for easier integration
        self.jacobian_numpy = self.jacobian.detach().cpu().numpy()
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

projections = ["tsne", "mds", "Tangent-Map"]
projectionClasses = [tsne, mds, None]
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
