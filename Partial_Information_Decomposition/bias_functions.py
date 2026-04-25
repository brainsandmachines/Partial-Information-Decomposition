import torch
from pathlib import Path
import sys
from Idep_Simulations.wrapper_M7_M8_models import make_random_true_cov,create_m7_cov
from Partial_Information_Decomposition.Idep_Simulations.Simulation_utils import build_m7_terms
from mi_functions import calcualte_mi

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from PID_util import create_cov_matrix









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



def permuteation_debiased(config,term = 'nume'):
    n0 = config['n0']
    n1 = config['n1']
    n2 = config['n2']
    device = config['device']
    X1,X2,T = config['X1_perm'],config['X2_perm'],config['T_perm']
    
    Z = create_cov_matrix(rvs=[X1,X2,T],device=device)

    if config['model'] == 'M7':
        M7_cov = create_m7_cov(config,Z['full_cov'],whitening_normalize=True) #Also Whiten Normalize
        M7_cov_dict = create_cov_matrix(Sigma=M7_cov,dims = [n0, n1, n2],device=device)
        m7_dict = build_m7_terms(config,M7_cov_dict,whiten='True') # Because of one row above we can assume it's already whitened
        mi_terms = calcualte_mi(config,m7_dict)
        value = mi_terms[term]
    
    return value


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
    n0 = config['n0']
    n1 = config['n1']
    n2 = config['n2']
    device = config['device']
    n_perm = config['n_perm']
    rng = config['rng']

    if n_perm == 0:
        return {
            "debiased": None,
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