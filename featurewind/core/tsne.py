import numpy as np
import torch

'''
Computes the entropy H and the probability vector P for a given distance vector D and precision beta. 
This function is used in the process of computing the conditional probabilities in t-SNE.
'''
def Hbeta_torch(D, beta=1.0):
    # Use same device/dtype as inputs
    P = torch.exp(-D.clone() * beta)
    sumP = torch.sum(P)
    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p_torch(X, tol=1e-5, perplexity=10.0, max_tries = 400, init_beta = None):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape

    X.requires_grad_(True)
    device, dtype = X.device, X.dtype

    sum_X = torch.sum(X*X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

    P = torch.zeros(n, n, device=device, dtype=dtype)
    beta = torch.ones(n, 1, device=device, dtype=dtype)
    logU = torch.log(torch.tensor([perplexity], device=device, dtype=dtype))
    n_list = [i for i in range(n)]

    if init_beta is not None:
        beta = init_beta

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i]+n_list[i+1:n]]

        (H, thisP) = Hbeta_torch(Di, beta[i].clone())

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        # Use .item() to extract scalar for comparison
        while torch.abs(Hdiff).item() > tol and tries < max_tries:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta_torch(Di, beta[i].clone())

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    # Return final P-matrix
    return P, beta


def pca_torch(X, no_dims=50):
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - torch.mean(X, 0)

    (l, M) = torch.eig(torch.mm(X.t(), X), True)
    # split M real
    # this part may be some difference for complex eigenvalue
    # but complex eignevalue is meanless here, so they are replaced by their real part
    i = 0
    while i < d:
        if l[i, 1] != 0:
            M[:, i+1] = M[:, i]
            i += 2
        else:
            i += 1

    Y = torch.mm(X, M[:, 0:no_dims])
    return Y


def tsne(X, no_dims=2, maxIter = 999, initial_dims=50, perplexity=30.0, save_params = False, initY = None, initBeta = None, betaTries = 50, initIY =None):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should not have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    # X = pca_torch(X, initial_dims)
    (n, d) = X.shape
    device, dtype = X.device, X.dtype
    max_iter = maxIter
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01

    Y = torch.randn(n, no_dims, device=device, dtype=dtype)  # randomly initialize embedding
    dY = torch.zeros(n, no_dims, device=device, dtype=dtype) # gradient of the cost function with respect to Y
    iY = torch.zeros(n, no_dims, device=device, dtype=dtype) # Momentum term for updating Y
    gains = torch.ones(n, no_dims, device=device, dtype=dtype) # individual gains for each dimension to accelerate convergence


    if initY is not None:
        Y = initY
    if initIY is not None:
        iY = initIY

    # Compute P-values
    P, beta = x2p_torch(X, 1e-5, perplexity, betaTries, initBeta)
    P = P + P.t()
    P = P / torch.sum(P)
    P = P * 4.    # early exaggeration
    P = torch.max(P, torch.tensor([1e-21], device=device, dtype=dtype))

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = torch.sum(Y*Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / torch.sum(num)
        Q = torch.max(Q, torch.tensor([1e-12], device=device, dtype=dtype))

        # Compute gradient (vectorized over all points)
        # Original per-point form:
        #   dY[i, :] = sum_j ((P[j,i]-Q[j,i]) * num[j,i]) * (Y[i,:] - Y[j,:])
        # Let B = (P - Q) .* num (elementwise). Then for all i:
        #   dY = (sum_j B[i,j]) * Y - B @ Y
        # This matches the original loop when B is symmetric (as here).
        PQ = P - Q
        B = PQ * num                      # NxN
        sum_B = torch.sum(B, dim=1, keepdim=True)  # Nx1
        dY = sum_B * Y - torch.matmul(B, Y)        # Nxno_dims

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).float() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).float()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = torch.sum(P * torch.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.
    params = []
    if save_params:
        params = [Y.clone().detach(), iY.clone().detach(), beta.clone().detach()]
    # Return solution
    return Y, params
