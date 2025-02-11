import torch
import numpy as np


def distance_matrix_HD_tensor(dataHDw):

    return torch.cdist(dataHDw, dataHDw)


def distance_matrix_2D_tensor(dat2D):

    return torch.cdist(dat2D, dat2D)


def stress_tensor(distHD, dist2D):  # distHD, dist2D (numpy) -> stress (float)
    s = ((distHD - dist2D) ** 2).sum() / (distHD ** 2).sum()  # numpy, eliminate sqrt for efficiency
    return s


def mds(dataHDw, n_components=2, n_init=10, max_iter=1000, random_state=101, eps=1e-9, n_jobs=12):
    # we will denote the original distance metric with D
    def mds_stress(d):
        v = dataHDw - d

        return ((dataHDw - d) ** 2).sum()

    def create_B(d):
        d[d == 0.0] = np.inf
        B = dataHDw / d
        for i in range(len(B)):
            B[i][i] = 0.0

        diag = -B.sum(axis=0)

        for i in range(len(diag)):
            B[i][i] = diag[i]

        return (B)

    # steps of SMACOF
    np.random.seed(random_state)
    n, m = dataHDw.shape
    # choose a random pivot point
    x_m = torch.tensor(np.random.rand(n, n_components))

    # denote the subspace distance matrix as d
    d = distance_matrix_HD_tensor(x_m)

    stress_old = mds_stress(d)

    tol = 1e-4
    for i in range(max_iter):
        B = create_B(d.clone())

        x_min = torch.matmul(B, x_m) / n

        d = distance_matrix_HD_tensor(x_min)
        stress_new = mds_stress(d)
        if stress_old - stress_new < tol:
            break
        else:
            x_m = x_min
            stress_old = stress_new

    return x_min[:, :n_components]
