import torch
import numpy as np
import pandas as pd
import sys
import os
import time
from pathlib import Path
from PID_util import *
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root)) 
from Idep_Simulations.Simulation_utils import *
from PID_util import *

def jackkinfe_func(config,cov_loo,calculate_statistic_func):
    """Calculate the jackknife bias correction for a given statistic calculated on leave-one-out covariance matrices.
    Inputs:
    - cov_loo: a numpy array of shape (n_samples, d, d) containing the leave-one-out covariance matrices for each sample.
    - calculate_statistic_func: a function that takes a covariance matrix as input and returns the statistic of interest (e.g. logdet,
    mutual information, etc). This function should be able to handle a batch of covariance matrices and return a 1D array of statistics, one per leave-one-out sample.
    Returns:
    - bias_correction: the jackknife bias correction estimate for the statistic, calculated as (n_samples - 1) * (mean of leave-one-out statistics - statistic calculated on full sample covariance).
    """
    values = calculate_statistic_func(cov_loo)
    assert values.ndim == 1, "Expected calculate_statistic_func to return a 1D array of statistics, one per leave-one-out sample"
    assert values.shape[0] == config['n_samples'], f"Expected number of leave-one-out statistics to match n_samples. Got {values.shape[0]} statistics but expected {config['n_samples']}."
    values_mean = torch.mean(values).item()
    raw_value = config['sample_statistic']
    bias = (config['n_samples'] - 1) * (values_mean - raw_value)
    return bias

def jackknife_resample(config:dict) -> list:
    """
    Compute the full covnarice matrix across smaples 
    and the covariance matrix of the left out ovbesrvation. 
    Using the formula for covariance matrix 
    Σ(-j)=N-2/(S(-j)-(1/N)*s(-j)s(-j)T)
    Where S(-j)=S-ZjZjT
     and s(-j)=s-Zj
   
    S = Sum of the outer product of the samples
    s = sum of the samples

    Input: 
    N - number of samples
    Z - a list of RVs each with shape (N, p_i) where p_i is the dimension of the i-th variable. The list should be in the order [M1, M2, T]   
          """
    rvs = config['rvs_list']
    N = config['n_samples']
    Sigma = config.get('Sigma', None)
    assert type(rvs) == list, "Input Z should be a list of torch tensors"
    d = sum([rv.shape[1] for rv in rvs])  # Total dimension across all variables
    if type(rvs[0]) == np.ndarray:
        rvs = [torch.from_numpy(rv).to(torch.float64) for rv in rvs]
    Z = torch.hstack(rvs).to(torch.float64)   # shape (N, len(rvs)*len(rvs)*p)
    S = Z.T @ Z  # shape (len(rvs)*p, len(rvs)*p)
    s = torch.sum(Z, axis=0)
    s_outer = torch.outer(s, s)
    Sigma_full = (S - (1/N)*s_outer) / (N-1)
    Sigma_full_dict = para_create_cov_matrix([config['n0'],config['n1'],config['n2']], Sigma_full.unsqueeze(0))
    Sigma_full = jackknife_whiten(config,Sigma_full_dict)
    if type(Sigma_full) == np.ndarray:
        Sigma_full = torch.from_numpy(Sigma_full).to(torch.float64)
    assert torch.allclose(Sigma_full.to(torch.float32), Sigma, atol=1e-7, rtol=1e-5), "The covariance matrix computed using the formula does not match the one computed using torch"
    # All z_j z_j^T at once
    outer_all = Z[:, :, None] * Z[:, None, :]   # (N, d, d)
    assert outer_all.shape == (N, d, d), f"Expected outer_all to have shape (N, d, d) but got {outer_all.shape}"
    # All S^{(-j)}
    S_minus_all = S.unsqueeze(0) - outer_all    # (N, d, d)
    assert S_minus_all.shape == (N, d, d), f"Expected S_minus_all to have shape (N, d, d) but got {S_minus_all.shape}"
    # All s^{(-j)}
    s_minus_all = s.unsqueeze(0) - Z            # (N, d)
    assert s_minus_all.shape == (N, d), f"Expected s_minus_all to have shape (N, d) but got {s_minus_all.shape}"
    # All s^{(-j)} s^{(-j)T}
    s_outer_all = s_minus_all[:, :, None] * s_minus_all[:, None, :]  # (N, d, d)
    assert s_outer_all.shape == (N, d, d), f"Expected s_outer_all to have shape (N, d, d) but got {s_outer_all.shape}"
    # All leave-one-out covariances
    cov_loo_all = (S_minus_all - s_outer_all / (N - 1)) / (N - 2)
    assert cov_loo_all.shape == (N, d, d), f"Expected cov_loo_all to have shape (N, d, d) but got {cov_loo_all.shape}"
    cov_loo_all_dict = para_create_cov_matrix([config['n0'],config['n1'],config['n2']], cov_loo_all)
    cov_loo_all_whiten = jackknife_whiten(config, cov_loo_all_dict)
    return Sigma_full, cov_loo_all_whiten

def jackknife_whiten(config,m7_cov_dict):
    Q = para_whiten_block(m7_cov_dict['cov_x0'], m7_cov_dict['cross_x0_x2'], m7_cov_dict['cov_x2']).to(config['device'])
    R = para_whiten_block(m7_cov_dict['cov_x1'], m7_cov_dict['cross_x1_x2'], m7_cov_dict['cov_x2']).to(config['device'])
    P = para_whiten_block(m7_cov_dict['cov_x0'], m7_cov_dict['cross_x0_x1'], m7_cov_dict['cov_x1']).to(config['device'])

    if config['model'] == 'M7':
        P = Q @ R.mT 

    I0 = torch.eye(config['n0'], device=config['device']).repeat(P.shape[0], 1, 1) 
    I1 = torch.eye(config['n1'], device=config['device']).repeat(Q.shape[0], 1, 1) 
    I2 = torch.eye(config['n2'], device=config['device']).repeat(R.shape[0], 1, 1) 

    # 1. Build the three "rows" of the block matrix by concatenating along the column dimension (dim=-1)
    row1 = torch.cat([I0,   P,    Q],    dim=-1)
    row2 = torch.cat([P.mT, I1,   R],    dim=-1)
    row3 = torch.cat([Q.mT, R.mT, I2],   dim=-1)

    # 2. Stack the rows vertically by concatenating along the row dimension (dim=-2)
    cov_whiten = torch.cat([row1, row2, row3], dim=-2)
    return cov_whiten
