import torch
import numpy as np
from scipy.linalg import sqrtm, inv
from scipy.special import digamma
from pathlib import Path
import sys
import os
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from PID_util import *
import pandas as pd



def mean_std_csv_results(results_dict):
    """ Helper: Compute mean results across seeds """
    df = pd.DataFrame.from_dict(results_dict, orient="index")
    mean_results = df.mean()
    std_results = df.std()
    return mean_results, std_results



def N_P_variation_simulation(config):
    """ Helper: Run simulations across different N and p values 
    and then create a heatamp of the results. """
    N_values = config['N_values']
    p_values = config['p_values']
    simulation_func = config['simulation_func']
    all_results = []
    len_N = len(N_values)
    len_P = len(p_values)
    i=1
    for N in N_values:
        for p in p_values:
            print(f"\nRunning simulation for N={N}, p={p} ({i}/{len_N*len_P})")
            results_dict = simulation_func(config)
            mean_results, std_results = mean_std_csv_results(results_dict)
            row = {
            "N": N,
            "p": p[0],  # Assuming all p values in the list are the same for simplicity
        }

            for key in mean_results.index:
                row[f"{key}_mean"] = mean_results[key]
                row[f"{key}_std"] = std_results[key]

                all_results.append(row)
            print(f"Completed combination N={N}, p={p} ({i}/{len_N * len_P})")
            i += 1

    return all_results



def sample_data_from_cov(true_cov: np.ndarray, n_samples: int, rng: int | None = None) -> np.ndarray:
    """
    Sample multivariate Gaussian data from the specified covariance.
    and return it's covariance matrix. This is a helper function for the m7_whiten bias simulation.
    """
    d = true_cov.shape[0]
    mean = np.zeros(d)
    data =  rng.multivariate_normal(mean, true_cov, size=n_samples)
    return np.cov(data, rowvar=False, bias=False) # Unbiased estimator with N-1 in the denominator


def safe_logdet(A: np.ndarray) -> float:
    """
    Compute log determinant and raise if matrix is not positive definite.
    """
    sign, ld = np.linalg.slogdet(A)
    if sign <= 0:
        eigmin = np.min(np.linalg.eigvalsh(0.5 * (A + A.T)))
        raise np.linalg.LinAlgError(
            f"Matrix not positive definite in logdet. sign={sign}, min_eig={eigmin:.3e}"
        )
    return ld

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
    return np.sum([digamma((df - i + 1) / 2.0) for i in range(1,d+1)]) + d * np.log(2.0 / df)
    