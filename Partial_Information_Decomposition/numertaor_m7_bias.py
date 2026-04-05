import torch
from PID_util import whiten_block


def _extract_blocks_from_S(S: torch.Tensor, n0: int, n1: int, n2: int):
    """
    Extract the blocks needed for the M7 numerator from the full covariance S
    in block order [X0, X1, X2].
    """
    d01 = n0 + n1
    d = d01 + n2

    if S.shape != (d, d):
        raise ValueError(f"Expected full covariance shape {(d, d)}, got {S.shape}.")

    S00 = S[:n0, :n0]
    S11 = S[n0:d01, n0:d01]
    S22 = S[d01:d, d01:d]

    S02 = S[:n0, d01:d]
    S12 = S[n0:d01, d01:d]

    return S00, S11, S22, S02, S12


def _m7_numerator_from_full_cov(S: torch.Tensor, n0: int, n1: int, n2: int) -> torch.Tensor:
    """
    Compute the M7 numerator from the raw full covariance matrix S:

        f(S) = 0.5 * logdet(I - P(S)^T P(S))

    where:
        Q(S) = whiten_block(S00, S02, S22)
        R(S) = whiten_block(S11, S12, S22)
        P(S) = Q(S) @ R(S).T
    """
    S = 0.5 * (S + S.mT)

    S00, S11, S22, S02, S12 = _extract_blocks_from_S(S, n0, n1, n2)

    Q = whiten_block(S00, S02, S22)
    R = whiten_block(S11, S12, S22)
    P = Q @ R.mT

    A = torch.eye(n1, dtype=S.dtype, device=S.device) - P.mT @ P
    sign, ld = torch.linalg.slogdet(A)
    if torch.any(sign <= 0):
        raise RuntimeError(
            "I - P^T P is not positive definite in _m7_numerator_from_full_cov."
        )

    return 0.5 * ld


def _get_triu_indices(d: int, device: torch.device):
    """
    Return upper-triangular indices for a d x d symmetric matrix.
    """
    tri_i, tri_j = torch.triu_indices(d, d, offset=0, device=device)
    return tri_i, tri_j


def _sym_matrix_from_theta(theta: torch.Tensor, d: int) -> torch.Tensor:
    """
    Reconstruct a symmetric d x d matrix from its upper-triangular entries.

    theta has length m = d(d+1)/2 in the order returned by torch.triu_indices.
    """
    tri_i, tri_j = _get_triu_indices(d, theta.device)

    S = torch.zeros((d, d), dtype=theta.dtype, device=theta.device)
    S[tri_i, tri_j] = theta
    S[tri_j, tri_i] = theta
    return S


def _theta_from_sym_matrix(S: torch.Tensor) -> torch.Tensor:
    """
    Extract the upper-triangular entries of a symmetric matrix S into a vector theta.
    """
    d = S.shape[0]
    tri_i, tri_j = _get_triu_indices(d, S.device)
    return S[tri_i, tri_j]


def _hessian_m7_numerator_wrt_theta(
    S: torch.Tensor,
    n0: int,
    n1: int,
    n2: int,
) -> torch.Tensor:
    """
    Exact Hessian of the M7 numerator with respect to the unique symmetric
    parameters of S (upper-triangular entries only).

    This is faster than differentiating with respect to all d*d entries.
    """
    S = 0.5 * (S + S.mT)
    S = S.detach().clone().to(torch.float64)

    d = S.shape[0]
    theta0 = _theta_from_sym_matrix(S).detach().clone().requires_grad_(True)

    def scalar_stat(theta: torch.Tensor) -> torch.Tensor:
        Smat = _sym_matrix_from_theta(theta, d)
        return _m7_numerator_from_full_cov(Smat, n0, n1, n2)

    try:
        H = torch.autograd.functional.hessian(
            scalar_stat,
            theta0,
            vectorize=True,
        )
    except TypeError:
        H = torch.autograd.functional.hessian(scalar_stat, theta0)

    return H.detach()


def _cov_theta_sample_cov_gaussian(Sigma: torch.Tensor, df: int) -> torch.Tensor:
    """
    Covariance of theta(S), where theta(S) is the vector of upper-triangular
    entries of the unbiased Gaussian sample covariance S.

    For Gaussian data:
        Cov(S_ij, S_kl) = (Sigma_ik Sigma_jl + Sigma_il Sigma_jk) / df

    This function returns the covariance matrix of the unique symmetric entries only,
    shape (m, m), where m = d(d+1)/2.
    """
    Sigma = 0.5 * (Sigma + Sigma.mT).to(torch.float64)
    d = Sigma.shape[0]

    tri_i, tri_j = _get_triu_indices(d, Sigma.device)

    i = tri_i[:, None]
    j = tri_j[:, None]
    k = tri_i[None, :]
    l = tri_j[None, :]

    cov_theta = (
        Sigma[i, k] * Sigma[j, l] +
        Sigma[i, l] * Sigma[j, k]
    ) / df

    return cov_theta


def bias_m7_nume_second_order(config: dict) -> dict:
    """
    Faster exact second-order delta-method bias correction for the M7 numerator,
    done in Sigma-space and parameterized only by the unique symmetric entries
    of the raw sample covariance S.

    We treat the active M7 numerator estimator as a function of the raw sample
    covariance S:

        h(S) = 0.5 * logdet(I - P(S)^T P(S))

    with:
        Q(S) = whiten_block(S00, S02, S22)
        R(S) = whiten_block(S11, S12, S22)
        P(S) = Q(S) @ R(S).T

    Delta method:
        Bias[h(S)] ≈ 0.5 * trace( H_theta(S_hat) @ Cov(theta(S_hat)) )

    where theta(S) is the vector of upper-triangular entries of S.

    Required config keys
    --------------------
    config['Sigma_raw'] : the raw sample covariance before M7 projection/whitening
    config['n0'], config['n1'], config['n2']
    config['n_samples']

    Optional config key
    -------------------
    config['device'] : 'cpu' or 'cuda'
    """
    if 'Sigma_raw' not in config:
        raise KeyError(
            "config['Sigma_raw'] is required for the Sigma-space delta method."
        )

    n0 = int(config['n0'])
    n1 = int(config['n1'])
    n2 = int(config['n2'])
    N = int(config['n_samples'])
    df = N - 1

    if df <= 0:
        raise ValueError(f"Need n_samples >= 2, got {N}.")

    device = config.get('device', None)

    Sigma_raw = config['Sigma_raw']
    if not torch.is_tensor(Sigma_raw):
        Sigma_raw = torch.as_tensor(Sigma_raw, dtype=torch.float64)
    else:
        Sigma_raw = Sigma_raw.detach().clone().to(torch.float64)

    if device is not None:
        Sigma_raw = Sigma_raw.to(device)

    Sigma_raw = 0.5 * (Sigma_raw + Sigma_raw.mT)

    H_theta = _hessian_m7_numerator_wrt_theta(Sigma_raw, n0, n1, n2)
    cov_theta = _cov_theta_sample_cov_gaussian(Sigma_raw, df=df)

    if H_theta.shape != cov_theta.shape:
        raise ValueError(
            f"Hessian shape {H_theta.shape} does not match Cov(theta) shape {cov_theta.shape}."
        )

    bias = 0.5 * torch.trace(H_theta @ cov_theta)

    return {'bias': bias.item()}
