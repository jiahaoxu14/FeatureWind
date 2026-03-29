import sys
import csv
import time
import numpy as np
from .tsne import tsne, tsne_1step
from .mds_torch import mds, distance_matrix_HD_tensor
import torch
from torch.autograd.functional import jacobian as autograd_jacobian

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
            t0 = time.time()
            with torch.no_grad():
                Y_base, params = tsne(data, 2, 1000, 10, perp_base, save_params=True)
            print(f"  Step 1 done in {time.time()-t0:.1f}s")

            # Save init state for tsne_1step (all detached from step-1 graph)
            init_Y    = params[0]  # (n, 2)
            init_iY   = params[1]  # (n, 2)
            init_beta = params[2]  # (n, 1)

            print("Step 2/3: Computing 1-step projection...")
            t1 = time.time()
            with torch.no_grad():
                Y = tsne_1step(data.detach(), init_Y, init_iY,
                               perplexity=perp_grad, init_beta=init_beta)
            print(f"  Step 2 done in {time.time()-t1:.1f}s")
            n_points, n_features = data.shape
            print(f"Step 3/3: Computing Jacobian for {n_points} points (vectorized)...")
            t2 = time.time()
            grads_x, grads_y = self._compute_jacobian_tsne(
                data, init_Y, init_iY, init_beta, perp_grad, device, n_points, n_features
            )
            print(f"  Step 3 done in {time.time()-t2:.1f}s  (total: {time.time()-t0:.1f}s)")

        elif self.projection == mds or (isinstance(self.projection, str) and self.projection.lower() == 'mds'):
            t0 = time.time()
            print("Step 1/3: Computing base MDS projection (999 iterations)...")
            with torch.no_grad():
                dist_hd = distance_matrix_HD_tensor(data)
                Y_base = mds(dist_hd, n_components=2, max_iter=999)
            print("Step 2/3: Computing projection with gradients (1 iteration)...")
            dist_hd = distance_matrix_HD_tensor(data)
            Y = mds(dist_hd, n_components=2, max_iter=1)

            n_points, n_features = data.shape
            print(f"Step 3/3: Computing Jacobian for {n_points} points (serial)...")
            t2 = time.time()
            def mds_fn(x):
                return mds(distance_matrix_HD_tensor(x), n_components=2, max_iter=1)
            grads_x, grads_y = self._compute_jacobian_serial(
                data, mds_fn, n_points, n_features, device
            )
            print(f"  Step 3 done in {time.time()-t2:.1f}s  (total: {time.time()-t0:.1f}s)")
        else:
            raise ValueError("Unsupported projection method. Use 'tsne' or 'mds'.")

        # Store results (shared by both branches)
        self.outPoints = Y

        # Build grads array (N, 2, d) — same shape expected downstream
        self.grads = torch.stack([grads_x, grads_y], dim=1).detach().cpu().numpy()

        # Build flat Jacobian matrix (2*N, d)
        self.jacobian = torch.zeros(2 * n_points, n_features, dtype=torch.float32, device=device)
        self.jacobian[0::2] = grads_x
        self.jacobian[1::2] = grads_y
        self.jacobian_numpy = self.jacobian.detach().cpu().numpy()

        print("✓ Tangent map generation completed successfully!")

    def _compute_jacobian_tsne(self, data, init_Y, init_iY, init_beta, perplexity,
                               device, n_points, n_features):
        """
        Compute diagonal Jacobian blocks ∂Y[i]/∂data[i] for all i.

        Tries vectorized batch via autograd.functional.jacobian(vectorize=True).
        Falls back to the original serial loop if that fails.

        Returns:
            grads_x (n_points, n_features): ∂Y[:,0]/∂data (diagonal blocks)
            grads_y (n_points, n_features): ∂Y[:,1]/∂data (diagonal blocks)
        """
        def tsne_1step_fn(x):
            return tsne_1step(x, init_Y, init_iY,
                              perplexity=perplexity, init_beta=init_beta)

        try:
            # Vectorized: all 2*N backward passes run in parallel via vmap
            J = autograd_jacobian(tsne_1step_fn, data, create_graph=False, vectorize=True)
            # J shape: (n_points, 2, n_points, n_features)
            idx = torch.arange(n_points, device=device)
            grads_x = J[idx, 0, idx, :]   # (n_points, n_features)
            grads_y = J[idx, 1, idx, :]   # (n_points, n_features)
            print("  (used vectorized Jacobian)")
            return grads_x, grads_y

        except Exception as e:
            print(f"  Vectorized Jacobian failed ({e}), falling back to serial loop...")
            return self._compute_jacobian_serial(data, tsne_1step_fn, n_points, n_features, device)

    def _compute_jacobian_serial(self, data, fn, n_points, n_features, device):
        """Original per-point backward loop — fallback only."""
        Y = fn(data)
        grads_x = torch.zeros(n_points, n_features, device=device)
        grads_y = torch.zeros(n_points, n_features, device=device)
        progress_interval = max(1, n_points // 10)
        for i in range(n_points):
            if i % progress_interval == 0 or i == n_points - 1:
                print(f"  Computing gradients: {i+1}/{n_points} ({(i+1)/n_points*100:.0f}%)")
            grads_x[i] = torch.autograd.grad(Y[i, 0], data, retain_graph=True)[0][i]
            grads_y[i] = torch.autograd.grad(Y[i, 1], data, retain_graph=True)[0][i]
        return grads_x, grads_y

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
