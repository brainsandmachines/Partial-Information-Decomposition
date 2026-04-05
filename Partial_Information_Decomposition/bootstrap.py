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
from PID_util import *


_BOOTSTRAP_COUNT_KEYS = (
    'n_bootstrap',
    'n_bootstraps',
    'n_resamples',
    'n_boot',
    'bootstrap_samples',
    'n_bootstrap_samples',
)


def _get_bootstrap_count(config: dict) -> int:
    """Infer the number of bootstrap replicates from config.

    Falls back to n_samples so the behaviour mirrors the jackknife path
    without requiring changes in the calling code.
    """
    for key in _BOOTSTRAP_COUNT_KEYS:
        if key in config and config[key] is not None:
            n_boot = int(config[key])
            if n_boot < 1:
                raise ValueError(f"{key} must be a positive integer. Got {n_boot}.")
            return n_boot
    return int(config['n_samples'])



def _to_tensor_list(rvs_list: list, device: str | torch.device) -> list[torch.Tensor]:
    tensors = []
    for rv in rvs_list:
        if isinstance(rv, np.ndarray):
            rv = torch.from_numpy(rv)
        tensors.append(rv.to(device=device, dtype=torch.float64))
    return tensors



def _estimate_fitted_model_cov(config: dict) -> torch.Tensor:
    """Return the fitted covariance used by the parametric bootstrap.

    Priority:
    1. Use config['Sigma'] if the current simulation already computed the
       whitened/projected covariance for the requested model.
    2. Otherwise estimate it from config['rvs_list'] using the same whitening
       logic as the jackknife branch.
    """
    device = config.get('device', 'cpu')
    Sigma = config.get('Sigma', None)

    if Sigma is not None:
        if isinstance(Sigma, np.ndarray):
            Sigma = torch.from_numpy(Sigma)
        return Sigma.to(device=device, dtype=torch.float64)

    rvs = _to_tensor_list(config['rvs_list'], device=device)
    Z = torch.hstack(rvs).to(torch.float64)
    Sigma_full = torch.cov(Z.T, correction=1)
    Sigma_full_dict = para_create_cov_matrix(
        [config['n0'], config['n1'], config['n2']],
        Sigma_full.unsqueeze(0),
    )
    Sigma_model = bootstrap_whiten(config, Sigma_full_dict).squeeze(0)
    return Sigma_model





def bootstrap_func(config: dict, cov_bootstrap: torch.Tensor, calculate_statistic_func: callable):
    """Estimate parametric-bootstrap bias for a statistic.

    Parameters
    ----------
    config:
        Must contain 'sample_statistic'. If one of the recognised bootstrap-count
        keys is present it is used, otherwise the function defaults to n_samples.
    cov_bootstrap:
        Batched covariance estimates of shape (B, d, d).
    calculate_statistic_func:
        Function that accepts a batch of covariance matrices and returns a
        1D tensor/array with one statistic per bootstrap replicate.

    Returns
    -------
    float
        Estimated bias E[T* | theta_hat] - T(data), which fits the existing
        corrected-statistic convention raw - bias.
    """
    bias_dict = {}
    config['cov_bootstrap'] = cov_bootstrap
    values_dict = calculate_statistic_func(config)
    for key in values_dict.keys():
        values = values_dict[key]
        if not torch.is_tensor(values):
            values = torch.as_tensor(values, dtype=torch.float64, device=cov_bootstrap.device)
        values = values.reshape(-1)

        n_boot = _get_bootstrap_count(config)
        assert values.ndim == 1, (
            "Expected calculate_statistic_func to return a 1D array/tensor of statistics, "
            "one per bootstrap replicate."
        )
        assert values.shape[0] == n_boot, (
            f"Expected {n_boot} bootstrap statistics but received {values.shape[0]}."
        )

        raw_value = config['sample_statistic']


        if torch.is_tensor(raw_value):
            raw_value = raw_value.item()
        if type(raw_value) == dict:
            raw_value = raw_value[key]
        bias = values.mean().item() - float(raw_value)
        bias_dict[key] = bias

    return bias_dict



def bootstrap_resample(config: dict) -> list:
    """Generate parametric-bootstrap covariance estimates.

    The bootstrap model is the fitted Gaussian covariance in config['Sigma']
    (preferred) or, if unavailable, the covariance estimated from config['rvs_list']
    and then projected/whitened according to the requested model.

    Returns
    -------
    (Sigma_model, cov_bootstrap_whitened)
        Sigma_model has shape (d, d).
        cov_bootstrap_whitened has shape (B, d, d).
    """
    device = config.get('device', 'cpu')
    N = int(config['n_samples'])
    B = _get_bootstrap_count(config)

    Sigma_model = _estimate_fitted_model_cov(config)
    if Sigma_model.ndim != 2:
        raise ValueError(f"Expected fitted covariance to have shape (d, d), got {Sigma_model.shape}.")

    d = Sigma_model.shape[0]
    mean = torch.zeros(d, dtype=torch.float64, device=device)
    dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=Sigma_model)
    samples = dist.sample((B, N))
    cov_list = []
    for sample in samples: 
        cov = torch.cov(sample.T, correction=1)
        cov_list.append(cov)
    cov_bootstrap = torch.stack(cov_list, dim=0)
    #centered = samples - samples.mean(dim=1, keepdim=True)
    cov_bootstrap_handy = samples.transpose(1, 2) @ samples / (N - 1)

    cov_bootstrap_dict = para_create_cov_matrix(
        [config['n0'], config['n1'], config['n2']],
        cov_bootstrap,
    )
    cov_bootstrap_whitened = bootstrap_whiten(config, cov_bootstrap_dict)
    return Sigma_model, cov_bootstrap_whitened



def bootstrap_whiten(config: dict, cov_dict: dict) -> torch.Tensor:
    """Project batched covariance estimates onto the M7/M8 whitened model space."""
    device = config.get('device', 'cpu')

    Q = para_whiten_block(
        cov_dict['cov_x0'],
        cov_dict['cross_x0_x2'],
        cov_dict['cov_x2'],
    ).to(device)
    R = para_whiten_block(
        cov_dict['cov_x1'],
        cov_dict['cross_x1_x2'],
        cov_dict['cov_x2'],
    ).to(device)
    P = para_whiten_block(
        cov_dict['cov_x0'],
        cov_dict['cross_x0_x1'],
        cov_dict['cov_x1'],
    ).to(device)

    if config['model'] == 'M7':
        P = Q @ R.mT

    batch_size = P.shape[0]
    I0 = torch.eye(config['n0'], dtype=torch.float64, device=device).repeat(batch_size, 1, 1)
    I1 = torch.eye(config['n1'], dtype=torch.float64, device=device).repeat(batch_size, 1, 1)
    I2 = torch.eye(config['n2'], dtype=torch.float64, device=device).repeat(batch_size, 1, 1)

    row1 = torch.cat([I0, P, Q], dim=-1)
    row2 = torch.cat([P.mT, I1, R], dim=-1)
    row3 = torch.cat([Q.mT, R.mT, I2], dim=-1)
    return torch.cat([row1, row2, row3], dim=-2)
