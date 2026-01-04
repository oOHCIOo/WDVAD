import numpy as np

def rbf_kernel(X, Y, sigma=1.0):
    """Compute Gaussian kernel matrix"""
    XX = np.sum(X ** 2, axis=1).reshape(-1, 1)
    YY = np.sum(Y ** 2, axis=1).reshape(1, -1)
    distances = XX + YY - 2 * np.dot(X, Y.T)
    return np.exp(-distances / (2 * sigma ** 2))

def compute_mmd(X, Y, sigma=1.0):
    """Compute MMD^2 between X and Y using RBF kernel"""
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)

    n = X.shape[0]
    m = Y.shape[0]

    # Exclude diagonal elements
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)

    mmd = Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2 * Kxy.mean()
    return mmd

def permutation_test_mmd(X, Y, sigma=1.0, num_permutations=1000):
    """Non-parametric MMD significance test based on permutation test"""
    observed_mmd = compute_mmd(X, Y, sigma)
    combined = np.vstack([X, Y])
    n = X.shape[0]
    count = 0
    for _ in range(num_permutations):
        idx = np.random.permutation(len(combined))
        X_perm = combined[idx[:n]]
        Y_perm = combined[idx[n:]]
        mmd_perm = compute_mmd(X_perm, Y_perm, sigma)
        if mmd_perm >= observed_mmd:
            count += 1
    p_value = count / num_permutations
    return observed_mmd, p_value