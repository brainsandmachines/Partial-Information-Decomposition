import torch
from pathlib import Path
import sys


root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from Partial_Information_Decomposition.PID_util import create_cov_matrix
from eigen_pid import eigen_pid


def logdet_wishart_bias(df: int, d: int) -> float:
    """
    Exact finite-sample bias for log|S| when S is the unbiased sample covariance
    from Gaussian data and (df) * S ~ Wishart_d(Sigma, df).

    Returns
    -------
    bias : float
        E[log|S|] - log|Sigma|
    """
    if df <= d - 1:
        raise ValueError(f"Need df > d-1. Got df={df}, d={d}.")

    i = torch.arange(1, d + 1, dtype=torch.float64)
    term = torch.special.digamma((df - i + 1) / 2.0)

    bias = torch.sum(term) + d * torch.log(torch.tensor(2.0 / df, dtype=torch.float64))

    return bias.item()

def mi_wishart_bias(dims: list, n_samples: int):
    """
    Bias correction for Gaussian mutual information estimates
    from unbiased sample covariance.

    Assumes order: [X1, X2, T]
    and torch.cov(..., correction=1), so df = n_samples - 1.

    Returns biases in nats.
    """
    df = n_samples - 1

    if len(dims) == 2:
        d1, d2 = dims

        bias_x1 = logdet_wishart_bias(df, d1)
        bias_x2 = logdet_wishart_bias(df, d2)
        bias_x1x2 = logdet_wishart_bias(df, d1 + d2)

        bias_mi = 0.5 * (bias_x1 + bias_x2 - bias_x1x2)
        return bias_mi

    if len(dims) == 3:
        d1, d2, dt = dims

        bias_x1 = logdet_wishart_bias(df, d1)
        bias_x2 = logdet_wishart_bias(df, d2)
        bias_t = logdet_wishart_bias(df, dt)

        bias_x1x2 = logdet_wishart_bias(df, d1 + d2)
        bias_x1t = logdet_wishart_bias(df, d1 + dt)
        bias_x2t = logdet_wishart_bias(df, d2 + dt)
        bias_x1x2t = logdet_wishart_bias(df, d1 + d2 + dt)

        # Bias of I(X1; T)
        bias_mi_1_t = 0.5 * (bias_x1 + bias_t - bias_x1t)

        # Bias of I(X2; T)
        bias_mi_2_t = 0.5 * (bias_x2 + bias_t - bias_x2t)

        # Bias of I((X1, X2); T)
        bias_tri_mi = 0.5 * (bias_x1x2 + bias_t - bias_x1x2t)

        # Optional: bias of I(X1; X2), if you need it
        bias_mi_12 = 0.5 * (bias_x1 + bias_x2 - bias_x1x2)

        return {
            "bias_mi_1_t": bias_mi_1_t,
            "bias_mi_2_t": bias_mi_2_t,
            "bias_tri_mi": bias_tri_mi,
            "bias_mi_12": bias_mi_12,
        }

    raise ValueError(f"dims must have length 2 or 3. Got len(dims)={len(dims)}.")

    


def broja_eigen_bias(config):
    """Calculate one raw Gaussian BROJA statistic for permutation debiasing.

    Inputs:
        config: dict containing:
            X1_perm, X2_perm, T_perm: torch.Tensor or array-like sample matrices
                with shapes (N, dx1), (N, dx2), and (N, dt).
            dx1, dx2, dt: int, source and target dimensions.
            n_samples: int, number of rows in each sample matrix.
            bias_correction: bool, retained in the config; this helper returns
                raw statistics for the surrounding correction routines.
            pid_bias_term: optional str, either ``'obj'`` (default) or ``'syn'``.

    Outputs:
        float, either raw Eigen PID ``obj`` or the plugin Eigen PID synergy in bits.
    """

    dm , dx, dy = config['dt'] , config['dx1'] , config['dx2']
    X_1, X_2, T = config['X1_perm'], config['X2_perm'], config['T_perm']
    data = [T, X_1, X_2]  # [T, X1, X2]
    dict_cov = create_cov_matrix(data)
    cov = dict_cov["full_cov"].detach().cpu().numpy()  # torch (D, D) -> NumPy (D, D)
    output = eigen_pid(cov, dm, dx, dy)
    # (D, D) -> scalar information tensors and (dt,) generalized eigenvalues
    statistic = config.get('pid_bias_term', 'obj')
    if statistic == 'obj':
        return float(output["union_info"])
    if statistic == 'syn':
        return float(output["syn"])
    raise ValueError(
        "config['pid_bias_term'] must be 'obj' or 'syn'; "
        f"got {statistic!r}."
    )


def permutation_null_debias(config,func):
    """Debias an MI-like estimator by subtracting its permutation null floor.

    X and Y can each be a single array or a tuple of paired tensors.
    Samples are assumed to be along axis 0.

    The function computes:

        raw = func(X, Y)
        perm_mean = mean(func(X, permuted_Y))
        debiased = raw - perm_mean

    If Y is a tuple, all Y blocks are permuted with the same permutation.
    X is kept fixed, so any internal structure within X is preserved.

    If n_perm=0, the function returns the raw estimate with perm_mean=0."""

    X1,X2,T = config['X1'],config['X2'],config['T']
    n = config['n_samples']
    dx1 = config['dx1']
    dx2 = config['dx2']
    dt = config['dt']
    device = config['device']
    n_perm = config['n_perm']

    rng = config.get('rng', None)
    if rng is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(config['rng_seed'])


    if n_perm == 0:
        return {
            "debiased": 0.0,
            "perm_mean": 0.0,
            "perm_std": 0.0,
            "perm_se": 0.0,
            "perm_values": torch.empty(0, dtype=float),
            "n_perm": n_perm,
        }
    
    config_sigma_perm = config.copy()
    perm_values = torch.empty(n_perm, dtype=float, device=device)
    
    for i in range(n_perm):
        idx = torch.randperm(n,generator=rng)

        if isinstance(T, tuple):
            T_perm = tuple(t[idx] for t in T)
        else:
            T_perm = T[idx]
        
        config_sigma_perm['X1_perm'] = X1 #X1 is not permuted
        config_sigma_perm['X2_perm'] = X2 #X2 is not permuted
        config_sigma_perm['T_perm'] = T_perm #T is permuted

        perm_values[i] = float(func(config=config_sigma_perm))

    perm_mean = float(torch.mean(perm_values))
    perm_std = float(torch.std(perm_values, unbiased=True)) if n_perm > 1 else 0.0
    perm_se = perm_std / torch.sqrt(torch.tensor(n_perm, dtype=torch.float64)) if n_perm > 0 else 0.0

    return {
        "bias": perm_mean,
        "perm_mean": perm_mean,
        "perm_std": perm_std,
        "perm_se": perm_se,
        "perm_values": perm_values,
        "n_perm": n_perm,
    }
