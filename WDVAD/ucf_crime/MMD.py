import torch

def _rbf_kernel_torch(X, Y, sigma):
    """
    Computes the RBF kernel between two sets of samples using PyTorch.
    X, Y are torch tensors.
    """
    XX = X.pow(2).sum(dim=1, keepdim=True)
    YY = Y.pow(2).sum(dim=1, keepdim=True).t()
    distances = XX + YY - 2 * torch.matmul(X, Y.t())
    return torch.exp(-distances / (2 * sigma ** 2))

def _compute_mmd_torch(X, Y, sigma):
    """
    Computes the MMD^2 between two sets of samples using PyTorch tensors.
    """
    Kxx = _rbf_kernel_torch(X, X, sigma)
    Kyy = _rbf_kernel_torch(Y, Y, sigma)
    Kxy = _rbf_kernel_torch(X, Y, sigma)

    n = X.shape[0]
    m = Y.shape[0]

    # Exclude diagonal elements for unbiased estimation
    if n > 1:
        Kxx.fill_diagonal_(0)
    if m > 1:
        Kyy.fill_diagonal_(0)

    # Handle cases where n or m is 1 to avoid division by zero
    sum_kxx = Kxx.sum() / (n * (n - 1)) if n > 1 else 0.0
    sum_kyy = Kyy.sum() / (m * (m - 1)) if m > 1 else 0.0
    
    mmd = sum_kxx + sum_kyy - 2 * Kxy.mean()
    return mmd

def permutation_test_mmd(X, Y, device, sigma=1.0, num_permutations=1000):
    """
    Performs a non-parametric MMD significance test using a permutation test,
    accelerated with PyTorch on the specified device (GPU/CPU).

    Args:
        X (np.ndarray): First set of samples.
        Y (np.ndarray): Second set of samples.
        device (torch.device): The device to perform computation on.
        sigma (float): The RBF kernel bandwidth.
        num_permutations (int): The number of permutations to perform.

    Returns:
        tuple: A tuple containing the observed MMD^2 value (float) and the p-value (float).
    """
    # Convert numpy arrays to torch tensors on the target device
    X_torch = torch.from_numpy(X).float().to(device)
    Y_torch = torch.from_numpy(Y).float().to(device)

    observed_mmd = _compute_mmd_torch(X_torch, Y_torch, sigma)
    
    # Combine samples for permutation
    combined = torch.cat([X_torch, Y_torch], dim=0)
    n = X.shape[0]
    count = 0
    
    for _ in range(num_permutations):
        # Permute the combined samples
        idx = torch.randperm(len(combined), device=device)
        X_perm = combined[idx[:n]]
        Y_perm = combined[idx[n:]]
        
        # Compute MMD for the permuted samples
        mmd_perm = _compute_mmd_torch(X_perm, Y_perm, sigma)
        
        if mmd_perm >= observed_mmd:
            count += 1
            
    p_value = count / num_permutations
    return observed_mmd.item(), p_value