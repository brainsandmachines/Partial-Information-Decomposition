import torch
import numpy as np
import yaml
from pathlib import Path
import sys

STORY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STORY_ROOT.parents[2]
DEFAULT_CONFIG_PATH = STORY_ROOT.parent / "rv_config.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
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
    dx1 = config['dx1']
    dx2 = config['dx2']
    dt = config['dt']
    A = torch.randn((dx1, dt), generator=rng, dtype=torch.float64,device=config['device'])
    B = torch.randn((dx2, dt), generator=rng, dtype=torch.float64,device=config['device'])
    C = torch.randn((dx1, dx2), generator=rng, dtype=torch.float64,device=config['device'])

    A_norm = torch.linalg.norm(A, ord=2)
    B_norm = torch.linalg.norm(B, ord=2)
    C_norm = torch.linalg.norm(C, ord=2)   
    if A_norm == 0 or B_norm == 0 or C_norm == 0:
        raise RuntimeError("Unexpected zero spectral norm in random construction.")

    Q = q_scale * A / A_norm 
    R = r_scale * B / B_norm 
    P = p_scale * C / C_norm 

    # Construct M8 covariance - sample coavraiance after whitening each block
    row1 = torch.cat([torch.eye(dx1, device=config['device']), P, Q], dim=1)
    row2 = torch.cat([P.T, torch.eye(dx2, device=config['device']), R], dim=1)
    row3 = torch.cat([Q.T, R.T, torch.eye(dt, device=config['device'])], dim=1)   
    true_cov = torch.cat([row1, row2, row3], dim=0)

    eigvals = torch.linalg.eigvalsh(true_cov)
    if torch.min(eigvals) <= 1e-10:
        raise ValueError(
            f"Constructed covariance not sufficiently PD. min eig={torch.min(eigvals):.3e}"
        )


    return true_cov




import torch


import torch


def rectangular_identity(
    d_a: int,
    d_b: int,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Create a rectangular identity-like matrix of shape (d_a, d_b).

    Example:
        d_a = 5, d_b = 2

        [[1, 0],
         [0, 1],
         [0, 0],
         [0, 0],
         [0, 0]]
    """
    M = torch.zeros((d_a, d_b), dtype=dtype, device=device)
    m = min(d_a, d_b)
    M[:m, :m] = torch.eye(m, dtype=dtype, device=device)
    return M


def make_direct_true_cov_from_config(
    config: dict,
    dtype: torch.dtype = torch.float64,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Create an interpretable covariance matrix for [X1, X2, T]
    directly from a merged config dictionary.

    Expected merged config keys:
        p_scale  : correlation between X1 and X2
        q_scale  : correlation between X1 and T
        r_scale  : correlation between X2 and T

        dx1      : dimension of X1
        dx2      : dimension of X2
        dt       : dimension of T

        device   : torch device, optional. Default is "cpu".

    Block order:
        [X1, X2, T]

    Returns
    -------
    Sigma:
        Positive definite covariance matrix of shape
        (dx1 + dx2 + dt, dx1 + dx2 + dt).
    """

    device = config.get("device", "cpu")

    # Interpret scales as block correlations
    rho_12 = config["p_scale"]  # X1-X2
    rho_1t = config["q_scale"]  # X1-T
    rho_2t = config["r_scale"]  # X2-T

    dx1 = config["dx1"]
    dx2 = config["dx2"]
    dt = config["dt"]

    # Marginal covariance blocks.
    # Each variable is internally whitened:
    # Cov(X1)=I, Cov(X2)=I, Cov(T)=I.
    I1 = torch.eye(dx1, dtype=dtype, device=device)
    I2 = torch.eye(dx2, dtype=dtype, device=device)
    It = torch.eye(dt, dtype=dtype, device=device)

    # Cross-covariance blocks.
    # These correlate matching coordinates only, up to min(dim_a, dim_b).
    C12 = rho_12 * rectangular_identity(dx1, dx2, dtype=dtype, device=device)
    C1t = rho_1t * rectangular_identity(dx1, dt, dtype=dtype, device=device)
    C2t = rho_2t * rectangular_identity(dx2, dt, dtype=dtype, device=device)

    top = torch.cat([I1, C12, C1t], dim=1)
    mid = torch.cat([C12.T, I2, C2t], dim=1)
    bot = torch.cat([C1t.T, C2t.T, It], dim=1)

    Sigma = torch.cat([top, mid, bot], dim=0)

    eigvals = torch.linalg.eigvalsh(Sigma)
    min_eig = eigvals.min().item()

    if min_eig <= eps:
        raise ValueError(
            f"The requested covariance is not positive definite. "
            f"Minimum eigenvalue = {min_eig:.3e}. "
            f"Try reducing p_scale, q_scale, or r_scale.\n"
            f"Current values: "
            f"p_scale={rho_12}, q_scale={rho_1t}, r_scale={rho_2t}"
        )

    return Sigma

def make_both_unique_true_cov_from_config(
    config: dict,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Create the paper's Gaussian both-unique covariance in [X1, X2, T].

    Inputs:
        config: dict containing dx1, dx2, dt, device, and optionally
        both_unique_connection_probability (default 0.1).
        rng: torch.Generator or None controlling the two channel matrices.

    Outputs:
        torch.Tensor: float64 population covariance with shape
        (dx1 + dx2 + dt, dx1 + dx2 + dt), ordered [X1, X2, T].
    """
    dx1 = config['dx1']
    dx2 = config['dx2']
    dt = config['dt']
    device = config['device']
    connection_probability = config.get('both_unique_connection_probability', 0.1)

    H1 = torch.bernoulli(
        torch.full((dx1, dt), connection_probability, dtype=torch.float64, device=device),
        generator=rng,
    )  # Bernoulli probabilities (dx1, dt) -> channel matrix (dx1, dt)
    H2 = torch.bernoulli(
        torch.full((dx2, dt), connection_probability, dtype=torch.float64, device=device),
        generator=rng,
    )  # Bernoulli probabilities (dx2, dt) -> channel matrix (dx2, dt)

    cov_x1 = H1 @ H1.T + torch.eye(dx1, dtype=torch.float64, device=device)  # (dx1, dt) @ (dt, dx1) -> (dx1, dx1)
    cov_x2 = H2 @ H2.T + torch.eye(dx2, dtype=torch.float64, device=device)  # (dx2, dt) @ (dt, dx2) -> (dx2, dx2)
    cov_x1_x2 = H1 @ H2.T  # (dx1, dt) @ (dt, dx2) -> (dx1, dx2)
    cov_t = torch.eye(dt, dtype=torch.float64, device=device)  # scalar dimension dt -> (dt, dt)

    top = torch.cat([cov_x1, cov_x1_x2, H1], dim=1)  # three covariance blocks -> (dx1, dx1 + dx2 + dt)
    middle = torch.cat([cov_x1_x2.T, cov_x2, H2], dim=1)  # three covariance blocks -> (dx2, dx1 + dx2 + dt)
    bottom = torch.cat([H1.T, H2.T, cov_t], dim=1)  # three covariance blocks -> (dt, dx1 + dx2 + dt)
    return torch.cat([top, middle, bottom], dim=0)  # three block rows -> (dx1 + dx2 + dt, dx1 + dx2 + dt)

def sample_from_cov(config,true_cov: torch.Tensor, n_samples: int, rng: torch.Generator) -> torch.Tensor:
    """
    Sample from a Gaussian distribution with the given covariance.

    Inputs:
        true_cov: torch.Tensor, covariance matrix of shape (d, d).
        n_samples: int, number of samples to generate.
        rng: torch.Generator, random number generator for reproducibility."""
    
    d = true_cov.shape[0]

    mean = torch.zeros(d, device=true_cov.device, dtype=true_cov.dtype)

    dist = torch.distributions.MultivariateNormal(
        loc=mean,
        covariance_matrix=true_cov
    )

    samples = dist.sample((n_samples,))

    sample_cov = torch.cov(samples.T, correction=1)

    dx1 = config['dx1']
    dx2 = config['dx2']
    dt = config['dt']
    rvs = [samples[:, :dx1], samples[:, dx1:dx1+dx2], samples[:, dx1+dx2:dx1+dx2+dt]]
    
    return sample_cov,rvs

def change_covariance_order(cov: torch.Tensor, new_order: list[int],dims:list[int]) -> torch.Tensor:
    """
    Permute the covariance matrix to change the order of variables.

    Inputs:
        cov: torch.Tensor, covariance matrix of shape (d, d).
        new_order: list of int, new order of covariance block indices.
        dims: list of int, dimensions of each variable block in the original order. Used for validation.

    For example, if the original order is [X0, X1, Y] and new_order is 
    [2, 0, 1], the output covariance will be ordered as [Y, X0, X1].

    Outputs:
        torch.Tensor, permuted covariance matrix.
    """

    assert cov.ndim == 2 and cov.shape[0] == cov.shape[1], "cov must be a square matrix"
    assert len(dims) in (2, 3), "dims must describe either two or three covariance blocks"
    assert len(new_order) == len(dims), "new_order must contain one entry per covariance block"
    assert sorted(new_order) == list(range(len(dims))), "new_order must be a permutation of block indices"
    assert sum(dims) == cov.shape[0], f"sum(dims) must match covariance size, got {sum(dims)} and {cov.shape[0]}"

    block_indices = []
    start = 0
    for dim in dims:
        block_indices.append(torch.arange(start, start + dim, device=cov.device))
        start += dim

    idx = torch.cat([block_indices[i] for i in new_order])
    new_cov = cov[idx][:, idx]
    new_dim = [dims[i] for i in new_order]
    old_cov_dict = create_cov_matrix(Sigma=cov,dims=dims,check_singular=False)
    new_cov_dict = create_cov_matrix(Sigma=new_cov,dims=new_dim,check_singular=False)

    assert new_cov.shape == (sum(dims), sum(dims)), "reordered covariance shape changed unexpectedly"
    assert torch.allclose(new_cov, new_cov.T, atol=1e-8), "reordered covariance is not symmetric"
    assert torch.allclose(new_cov_dict["full_cov"], cov[idx][:, idx], atol=1e-8), "full covariance reorder mismatch"

    diag_keys = ["cov_x1", "cov_x2"] if len(dims) == 2 else ["cov_x1", "cov_x2", "cov_t"]
    for new_position, old_position in enumerate(new_order):
        assert torch.allclose(
            new_cov_dict[diag_keys[new_position]],
            old_cov_dict[diag_keys[old_position]],
            atol=1e-8,
        ), f"auto-covariance block mismatch for new position {new_position}"

    pair_keys = {(0, 1): "cross_x1_x2"}
    if len(dims) == 3:
        pair_keys.update({(0, 2): "cross_x1_t", (1, 2): "cross_x2_t"})

    for new_pair, new_key in pair_keys.items():
        old_pair = (new_order[new_pair[0]], new_order[new_pair[1]])
        if old_pair[0] < old_pair[1]:
            expected = old_cov_dict[pair_keys[old_pair]]
        else:
            expected = old_cov_dict[pair_keys[(old_pair[1], old_pair[0])]].T

        assert torch.allclose(
            new_cov_dict[new_key],
            expected,
            atol=1e-8,
        ), f"cross-covariance block mismatch for new pair {new_pair}"

    return new_cov
