import torch
import numpy as np


def distance_matrix_HD_tensor(dataHDw):
    # Compute pairwise distances using torch on the same device as input
    return torch.cdist(dataHDw, dataHDw)


def distance_matrix_2D_tensor(dat2D):
    return torch.cdist(dat2D, dat2D)


def stress_tensor(distHD, dist2D):  # distHD, dist2D are tensors
    s = ((distHD - dist2D) ** 2).sum() / (distHD ** 2).sum()  # eliminate sqrt for efficiency
    return s


def mds(dataHDw, n_components=2, n_init=10, max_iter=1000, random_state=101, eps=1e-9, n_jobs=12):
    # we will denote the original distance metric with D
    def mds_stress(d):
        return ((dataHDw - d) ** 2).sum()

    def create_B(d):
        # Avoid division by zero
        d = d.clone()
        d.masked_fill_(d == 0, torch.inf)
        B = dataHDw / d
        # Zero diagonal before computing diag
        B.fill_diagonal_(0.0)
        diag = -B.sum(dim=0)
        # Set diagonal to diag values
        B = B.clone()
        for i in range(B.shape[0]):
            B[i, i] = diag[i]
        return B

    # steps of SMACOF
    # Device follows input tensor
    device = dataHDw.device
    n, m = dataHDw.shape
    # Random initialization on the correct device
    x_m = torch.randn(n, n_components, device=device)

    # denote the subspace distance matrix as d
    d = distance_matrix_HD_tensor(x_m)

    stress_old = mds_stress(d)

    tol = 1e-4
    for i in range(max_iter):
        B = create_B(d.clone())

        x_min = torch.matmul(B, x_m) / n

        d = distance_matrix_HD_tensor(x_min)
        stress_new = mds_stress(d)
        if (stress_old - stress_new) < tol:
            break
        else:
            x_m = x_min
            stress_old = stress_new

    return x_min[:, :n_components]
