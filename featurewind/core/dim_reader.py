import sys
# import DualNum
import csv
import numpy as np
from .tsne import tsne
from .mds_torch import mds, distance_matrix_HD_tensor
import torch
import inspect

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
            print("Step 1/3: Computing base t-SNE projection (500 iterations)...")
            with torch.no_grad():
                Y_base, params = tsne(data, 2, 500, 10, 15.0, save_params = True)
            print("Step 2/3: Computing projection with gradients (1 iteration)...")
            Y, params = tsne(data, no_dims=2, maxIter = 1, initial_dims=10, perplexity=15.0, save_params = False,
                                initY = params[0], initBeta = params[2], betaTries = 20, initIY =params[1])
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
        n_points, n_features = data.shape

        print(f"Step 3/3: Computing gradients for {n_points} points...")
        # Initialize Jacobian matrix: J[2*i:2*i+2, :] = gradients for point i
        self.jacobian = torch.zeros(2 * n_points, n_features, dtype=torch.float32, device=device)

        # Try a batched autograd path (PyTorch >= 1.13 supports is_grads_batched)
        sig = inspect.signature(torch.autograd.grad)
        use_batched = 'is_grads_batched' in sig.parameters

        if use_batched:
            # Choose a batch size to control memory. 64–256 generally works well on A30.
            # Choose batch size based on N to cap grad_outputs memory.
            # grad_outputs tensor has shape (2*b, N, 2) => elements ~= 4*b*N
            # Aim to keep <= ~8e6 elements (~32MB in fp32) by default.
            max_elems = 8_000_000
            b_cap = max(1, max_elems // max(1, 4 * n_points))
            batch_size = max(1, min(256, b_cap))
            # Pre-allocate tensor to store gradients per point (n_points, 2, n_features)
            grads_tensor = torch.zeros(n_points, 2, n_features, dtype=torch.float32, device=device)

            print(f"  Using batched autograd with batch_size={batch_size}")
            for start in range(0, n_points, batch_size):
                end = min(start + batch_size, n_points)
                b = end - start

                # Build batched grad_outputs selecting each (i, dim) in the batch
                # Shape: (2*b, n_points, 2)
                G = torch.zeros(2 * b, n_points, 2, dtype=Y.dtype, device=device)
                for bi, i in enumerate(range(start, end)):
                    G[2 * bi, i, 0] = 1.0  # select Y[i, 0]
                    G[2 * bi + 1, i, 1] = 1.0  # select Y[i, 1]

                # Compute gradients for this batch in a single autograd call
                retain = end < n_points
                try:
                    grad_all = torch.autograd.grad(
                        outputs=Y,
                        inputs=data,
                        grad_outputs=G,
                        retain_graph=retain,
                        is_grads_batched=True,
                        allow_unused=False,
                    )[0]  # shape: (2*b, n_points, n_features)
                except TypeError:
                    # is_grads_batched not supported — fall back to per-point loop
                    use_batched = False
                    break

                # Extract only the needed rows (diagonal per selected i)
                for bi, i in enumerate(range(start, end)):
                    grad_x = grad_all[2 * bi, i, :]
                    grad_y = grad_all[2 * bi + 1, i, :]
                    grads_tensor[i, 0, :] = grad_x
                    grads_tensor[i, 1, :] = grad_y

                # Progress update
                progress_pct = end / n_points * 100
                print(f"  Computing gradients: {end}/{n_points} points ({progress_pct:.1f}%)")

            if use_batched:
                # Fill jacobian and numpy equivalents
                self.jacobian[0::2, :] = grads_tensor[:, 0, :]
                self.jacobian[1::2, :] = grads_tensor[:, 1, :]
                self.grads = grads_tensor.detach().cpu().numpy()
                self.jacobian_numpy = self.jacobian.detach().cpu().numpy()
            else:
                # Fall back to original per-point loop
                print("  Batched autograd not available; falling back to per-point gradients.")
                grads = []
                for i in range(n_points):
                    progress_interval = min(50, max(1, n_points // 10))
                    if i % progress_interval == 0 or i == n_points - 1:
                        progress_pct = (i + 1) / n_points * 100
                        print(f"  Computing gradients: {i+1}/{n_points} points ({progress_pct:.1f}%)")
                    grad_x = torch.autograd.grad(Y[i, 0], data, retain_graph=True)[0][i]
                    grad_y = torch.autograd.grad(Y[i, 1], data, retain_graph=True)[0][i]
                    grads.append(torch.stack([grad_x, grad_y], dim=0))
                    self.jacobian[2 * i, :] = grad_x
                    self.jacobian[2 * i + 1, :] = grad_y
                self.grads = torch.stack(grads).detach().cpu().numpy()
                self.jacobian_numpy = self.jacobian.detach().cpu().numpy()
        else:
            # Original per-point loop
            print("  Using per-point autograd loop for gradients.")
            grads = []
            for i in range(n_points):
                progress_interval = min(50, max(1, n_points // 10))
                if i % progress_interval == 0 or i == n_points - 1:
                    progress_pct = (i + 1) / n_points * 100
                    print(f"  Computing gradients: {i+1}/{n_points} points ({progress_pct:.1f}%)")
                grad_x = torch.autograd.grad(Y[i, 0], data, retain_graph=True)[0][i]
                grad_y = torch.autograd.grad(Y[i, 1], data, retain_graph=True)[0][i]
                grads.append(torch.stack([grad_x, grad_y], dim=0))
                self.jacobian[2 * i, :] = grad_x
                self.jacobian[2 * i + 1, :] = grad_y
            self.grads = torch.stack(grads).detach().cpu().numpy()
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
