import torch
import numpy as np
import yaml
from pathlib import Path
import sys
from Partial_Information_Decomposition.PID_util import create_cov_matrix






def make_random_true_cov(
    config: dict,
    rng:  torch.Generator | None = None,
) -> np.ndarray:
    """
    Construct a generic positive-definite Gaussian covariance.

    Block order is [X0, X1, Y].
    Inputs: 
        config: dict, configuration dictionary with keys:
            n0: int, dimension of source 1 (X0).
            n1: int, dimension of source 2 (X1).
            n2: int, dimension of target (Y).
            q_scale: float, scaling factor for Q block.
            r_scale: float, scaling factor for R block.
            p_scale: float, scaling factor for P block.
        rng: torch.Generator or None, random number generator for reproducibility.


    Outputs: np.ndarray, covariance matrix of shape (n0+n1+n2, n0+n1+n2).
    """
    q_scale = config['q_scale']
    r_scale = config['r_scale']
    p_scale = config['p_scale']
    n0 = config['n0']
    n1 = config['n1']
    n2 = config['n2']
    A = torch.randn((n0, n2), generator=rng, dtype=torch.float64,device=config['device'])
    B = torch.randn((n1, n2), generator=rng, dtype=torch.float64,device=config['device'])
    C = torch.randn((n0, n1), generator=rng, dtype=torch.float64,device=config['device'])

    A_norm = torch.linalg.norm(A, ord=2)
    B_norm = torch.linalg.norm(B, ord=2)
    C_norm = torch.linalg.norm(C, ord=2)   
    if A_norm == 0 or B_norm == 0 or C_norm == 0:
        raise RuntimeError("Unexpected zero spectral norm in random construction.")

    Q = q_scale * A / A_norm
    R = r_scale * B / B_norm
    P = p_scale * C / C_norm

    # Construct M8 covariance - sample coavraiance after whitening each block
    row1 = torch.cat([torch.eye(n0, device=config['device']), P, Q], dim=1)
    row2 = torch.cat([P.T, torch.eye(n1, device=config['device']), R], dim=1)
    row3 = torch.cat([Q.T, R.T, torch.eye(n2, device=config['device'])], dim=1)   
    true_cov = torch.cat([row1, row2, row3], dim=0)

    eigvals = torch.linalg.eigvalsh(true_cov)
    if torch.min(eigvals) <= 1e-10:
        raise ValueError(
            f"Constructed covariance not sufficiently PD. min eig={torch.min(eigvals):.3e}"
        )


    return true_cov


def sample_from_cov(true_cov: torch.Tensor, n_samples: int, rng: torch.Generator) -> torch.Tensor:
    """
    Sample from a Gaussian distribution with the given covariance.

    Inputs:
        true_cov: torch.Tensor, covariance matrix of shape (d, d).
        n_samples: int, number of samples to generate.
        rng: torch.Generator, random number generator for reproducibility."""
    
    d = true_cov.shape[0]
    mean = torch.zeros(d, device=true_cov.device)
    samples = np.random.default_rng(rng).multivariate_normal(mean.cpu().numpy(), true_cov.cpu().numpy(), size=n_samples)
    return torch.from_numpy(samples).to(true_cov.device)


def change_covariance_order(cov: torch.Tensor, new_order: list[int],dims:list[int]) -> torch.Tensor:
    """
    Permute the covariance matrix to change the order of variables.

    Inputs:
        cov: torch.Tensor, covariance matrix of shape (d, d).
        new_order: list of int, new order of variable indices.
        dim: list of int, dimensions of each variable block in the original order. Used for validation.

    For example, if the original order is [X0, X1, Y] and new_order is 
    [2, 0, 1], the output covariance will be ordered as [Y, X0, X1].

    Outputs:
        torch.Tensor, permuted covariance matrix.
    """

    new_cov = cov[new_order][:, new_order]

    #Asser new order: 
    d = cov.shape[0]

    new_dim = [dims[i] for i in new_order]
    old_cov_dict = create_cov_matrix(Sigma=cov,dims=dims)

    new_cov_dict = create_cov_matrix(Sigma=new_cov,dims=new_dim)


    
    return cov[new_order][:, new_order]