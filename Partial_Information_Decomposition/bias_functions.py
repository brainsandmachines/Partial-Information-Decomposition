import torch
from pathlib import Path
import sys

from Partial_Information_Decomposition.mi_functions import calcualte_mi
from functools import partial

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from Partial_Information_Decomposition.PID_util import create_cov_matrix
from Partial_Information_Decomposition.Idep.Idep_Simulations.simulation_wrapper import make_random_true_cov,create_m7_cov
from Partial_Information_Decomposition.Idep.Idep_Simulations.Simulation_utils import build_m7_terms








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

    



def permuteation_debiased(config,term = 'nume'):
    dx1 = config['dx1']
    dx2 = config['dx2']
    dt = config['dt']
    device = config['device']
    X1,X2,T = config['X1_perm'],config['X2_perm'],config['T_perm']
    
    Z = create_cov_matrix(rvs=[X1,X2,T],device=device)

    if config['model'] == 'M7':
        M7_cov = create_m7_cov(config,Z['full_cov'],whitening_normalize=True) #Also Whiten Normalize
        M7_cov_dict = create_cov_matrix(Sigma=M7_cov,dims = [dx1, dx2, dt],device=device)
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
    dx1 = config['dx1']
    dx2 = config['dx2']
    dt = config['dt']
    device = config['device']
    n_perm = config['n_perm']
    rng = config['rng']

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



def unique_bias(config,functions_dict:dict = None):

    nodes = {'M7':['i','h'],'M8':['k','j']} #The unique nodes for each model, used to extract the relevant bias correction for each statistic
    assert type(functions_dict) == dict, "Expected bias_corr_func to be a dict with keys 'M7' and 'M8'."
    
    bias_dict ={}
    for model,bc_func in zip(nodes.keys(), functions_dict.values()):
        config['model'] = model
        bias = bc_func(config=config,model=model)

        node_0 = nodes[model][0] # i or k depending on the model
        node_1 = nodes[model][1] # h or j depending on the model

        bias_dict[node_0] = bias[node_0]
        bias_dict[node_1] = bias[node_1]

        
    return bias_dict


def bias_func(config,model):
    dx1 = config['dx1']
    dx2 = config['dx2']
    dt = config['dt']
    n_samples = config['n_samples']
    d = dx1 + dx2 + dt
    df = n_samples - 1

    bias_x0 = logdet_wishart_bias(df, dx1)
    bias_x1 = logdet_wishart_bias(df, dx2)
    bias_y  = logdet_wishart_bias(df, dt)
    # M7 (Structural) Biases
    bias_02 = logdet_wishart_bias(df, dx1 + dt) # Clique 0
    bias_12 = logdet_wishart_bias(df, dx2 + dt) # Clique 1
    bias_2 = bias_y # seperator 2
    bi_variate_mix2t_bias = 0.5*logdet_wishart_bias(df, dx2) + 0.5*logdet_wishart_bias(df, dt) - 0.5*logdet_wishart_bias(df, dx2+dt)
    bi_variate_mix1t_bias = 0.5*logdet_wishart_bias(df, dx1) + 0.5*logdet_wishart_bias(df, dt) - 0.5*logdet_wishart_bias(df, dx1+dt)

    if model == 'M7':
        nume_bias = permutation_null_debias(config,partial(permuteation_debiased,term='nume'))['bias']
        bias_m7_structural = bias_02 + bias_12 - bias_2
        deno_bias = 0.5 * (bias_m7_structural - (bias_x0 + bias_x1 + bias_y))
        return {'i': (nume_bias - deno_bias) - bi_variate_mix2t_bias,'h': (nume_bias - deno_bias) - bi_variate_mix1t_bias}
    
    elif model == 'M8':
        deno_bias = 0.5*(logdet_wishart_bias(df, d)-(bias_x0 + bias_x1 + bias_y))
        nume_bias = 0.5*(logdet_wishart_bias(df, dx1 + dx2) - (bias_x0 + bias_x1))
        return {'k': (nume_bias - deno_bias) - bi_variate_mix2t_bias,'j': (nume_bias - deno_bias) - bi_variate_mix1t_bias}

    

