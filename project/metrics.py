import torch



def covariance_matrix(data):
    """
    Calculate the covariance matrix for the given data using PyTorch.

    Parameters:
        data (torch.Tensor): A 2D tensor where each row represents an observation,
                             and each column represents a variable.

    Returns:
        torch.Tensor: The covariance matrix.
    """

    # Subtract the mean from each variable (column)
    mean_centered_data = data - torch.mean(data, dim=0, keepdim=True)
    
    # Calculate the covariance matrix
    # (n-1) in the denominator for an unbiased estimate
    n_samples = data.shape[0]
    cov_matrix = 1 / (n_samples - 1) * mean_centered_data.t().mm(mean_centered_data)
    
    return cov_matrix

def matrix_sqrt(A, eps=1e-9):
    """Compute the square root of a positive semi-definite matrix using eigendecomposition.
    Regularization is added to handle numerical instabilities."""
    # Symmetrize matrix
    A = (A + A.t()) / 2 

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    
    # Threshold eigenvalues (set negative eigenvalues to eps)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    
    # Compute the sqrt of eigenvalues
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    
    # Compute the square root of the matrix
    sqrt_matrix = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.t()

    return sqrt_matrix


def frechet_distance(data1, data2, eps=1e-6):
    """PyTorch implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Params:
    -- data1: PyTorch tensor 1 x n representing the activations of a model.
    -- data2: PyTorch tensor 1 x n representing the activations of a model.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = torch.mean(data1, dim=0).float()
    mu2 = torch.mean(data2, dim=0).float()

    sigma1 = covariance_matrix(data1)
    sigma2 = covariance_matrix(data2)

    mu_diff = mu1 - mu2

    # Adding a small identity matrix to avoid numerical instabilities
    sigma1 += torch.eye(sigma1.shape[0], device=sigma1.device) * eps
    sigma2 += torch.eye(sigma2.shape[0], device=sigma2.device) * eps

    # Computing the square root of the product of covariance matrices
    covmean = matrix_sqrt(sigma1.mm(sigma2))

    if torch.is_complex(covmean):
        if not torch.allclose(covmean.imag, torch.zeros_like(covmean.imag), atol=1e-3):
            return torch.tensor(float('nan'))
        covmean = covmean.real

    tr_covmean = torch.trace(covmean)

    return (mu_diff.dot(mu_diff) + torch.trace(sigma1)
            + torch.trace(sigma2) - 2 * tr_covmean)