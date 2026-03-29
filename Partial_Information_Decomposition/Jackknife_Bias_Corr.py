import torch
import numpy as np 
from matplotlib.pylab import eigvals
import matplotlib.pyplot as plt
from sklearn.covariance import OAS
from scipy.special import digamma
import torch
from sklearn.model_selection import LeaveOneOut 
import pandas as pd
from joblib import Parallel, delayed
import time
from PID_util import *
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from Toy_Simulations.Bias_Corr_simulations import theoretical_covariance, sample_cov_simulation
from Partial_Information_Decomposition.Idep_multivariate_gauss import Idep_multivariate_gauss
from parallel_Idep_multivariate_gauss import para_Idep_multivariate_gauss
from Partial_Information_Decomposition.Toy_Simulations.Bias_Corr_simulations import N_P_variation_simulation
from utils import (
    extract_all_components,
    print_seed_summary,
    run_multi_seed_experiment,
    get_seed_runs_csv_path,
    save_seed_summary_csv,
    create_test_histograms_with_kde,
    save_seed_summary_table_image,
    load_csv_and_add_data
)

def get_run_config() -> dict:
    return {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        "n_seeds": 10000,
        "seed_start": 0,
        "n": 1000,
        'N_values': [800, 1000, 2000, 3000],
        "p_values": [[5,5,10],[50, 50, 55],[100, 50, 105]],  # Dimensions for X1, X2, X3
        "p": 0,
        'q': 0,
        'r': 0,
        "results_dir": "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/para_jackknife_pid/with_groundtruth",
        "results_prefix": "seed_summary",
        "all_runs_results_prefix": "seed_runs",
        "progress_print_every": 100,
        "test_name": "NoWishart_jackknife_pid",
    }
def lo_cov(rvs:list,N:int,Sigma=None):
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
    assert type(rvs) == list, "Input Z should be a list of torch tensors"
    d = sum([rv.shape[1] for rv in rvs])  # Total dimension across all variables
    Z = torch.hstack(rvs).to(torch.float64)   # shape (N, len(rvs)*len(rvs)*p)
    cov = torch.cov((Z).T, correction=1)
    S = Z.T @ Z  # shape (len(rvs)*p, len(rvs)*p)
    s = torch.sum(Z, axis=0)
    s_outer = torch.outer(s, s)
    Sigma_full = (S - (1/N)*s_outer) / (N-1) if Sigma is None else Sigma
    assert torch.allclose(Sigma_full, cov, atol=1e-10, rtol=1e-8), "The covariance matrix computed using the formula does not match the one computed using torch"
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
    return Sigma_full, cov_loo_all


def idep_parallel(device,sources, target, N):

    assert len(sources) == 2, "Expected exactly two source variables"
    assert len(target) == 1, "Expected exactly one target variable"
    rvs = sources + target
    Sigma_full, cov_loo_all = lo_cov(rvs, N)
    Sigma_raw = Sigma_full.unsqueeze(0)
    dims = [source.shape[1] for source in sources] + [target[0].shape[1]]
    #Calculate the PID for the full covariance matrix
    idep_raw = para_Idep_multivariate_gauss(N=1,df = N-1,device=device,cov_matrix=Sigma_raw,dims=dims,bias_correction=False)
    pid_raw,mi_raw = idep_raw.idep()


    #Calculate the PID for each leave-one-out covariance matrix
    idep_loo = para_Idep_multivariate_gauss(N=N,df=N-1,device=device,cov_matrix=cov_loo_all,dims=dims,bias_correction=False)
    pid_loo,mi_loo = idep_loo.idep() 

    mean_pid_loo = {key: torch.mean(torch.stack([pid_loo[key][i] for i in range(N)])) for key in pid_loo.keys()}
    mean_mi_loo = {key: torch.mean(torch.stack([mi_loo[key][i] for i in range(N)])) for key in mi_loo.keys()}

    pid_bias_term = {key: (N-1)*(mean_pid_loo[key] - pid_raw[key]) for key in pid_raw.keys()}
    mi_bias_term = {key: (N-1)*(mean_mi_loo[key] - mi_raw[key]) for key in mi_raw.keys()}

    pid_values = {key: (pid_raw[key] - pid_bias_term[key]).item() for key in pid_raw.keys()}
    mi_values = {key: (mi_raw[key] - mi_bias_term[key]).item() for key in mi_raw.keys()}

    return pid_values, mi_values

def run_single_seed(seed: int,config:dict) -> dict:

    device = config['device']

    #Sample from the multivariate Gaussian distribution defined by the covariance matrix
    rv_list, sample_cov = sample_cov_simulation(seed, config['n'], config['dims'], config['true_cov'])

    #Convert the sampled random variables to torch tensors and move to device
    torch_rv_list = [torch.from_numpy(rv).to(device) for rv in rv_list]  # Convert to torch tensors

    #Calculate the bias-corrected PID values using the parallelized idep function
    pid_values, mi_values = idep_parallel(device = device, sources=torch_rv_list[:2], target=torch_rv_list[2:], N=config['n'])

    return extract_all_components(global_results={},ca_results={} ,pid_results=pid_values, mi_results=mi_values)


def run_simulation(config:dict) -> dict:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = get_run_config()
    #Create the ground truth covariance matrix based on the config parameters and pid values
    p = config['p']
    q = config['q']
    r = config['r']
    corr_matrix =np.array([
        [1.0,   p,  q  ],  # Row 1: X1
        [p, 1.0,    r  ],  # Row 2: X2
        [q,     r,      1.0]   # Row 3: X3
    ])

    config['true_cov'] = theoretical_covariance(config['dims'], corr_matrix)


    true_cov = theoretical_covariance(config['dims'], corr_matrix)
    torch_true_cov = torch.from_numpy(true_cov).to(device)
    true_value,_ = para_Idep_multivariate_gauss(N=1,df=config['n']-1,device=device,cov_matrix=torch_true_cov.unsqueeze(0),dims=config['dims']).idep()
    true_value = {f'true_{key}': true_value[key].item() for key in true_value.keys()}


    summary, seed_rows = run_multi_seed_experiment(config, per_seed_runner=run_single_seed)
    all_runs_path = get_seed_runs_csv_path(config)
    summary = {**summary, **true_value}

    summary_path = save_seed_summary_csv(summary, config)
    print(f"\nSaved all seed run results to: {all_runs_path}")
    print(f"Saved summary to: {summary_path}")
    return summary



def main():
    config = get_run_config()
    summarry = N_P_variation_simulation(config=config)
    summary = run_simulation(config)
    standard_summary = {
    "N": config["n"],
    "p": config["p"],
    "mi_theoretical_mean": ...,
    "mi_theoretical_std": ...,
    "mi_sample_no_bias_mean": ...,
    "mi_sample_no_bias_std": ...,
    "mi_sample_with_bias_mean": ...,
    "mi_sample_with_bias_std": ...,
}
    df = pd.DataFrame([summary])
    df.to_csv("summary_single_run.csv", index=False)

if __name__ == "__main__":
    main()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    q = 0
    r = 0
    p = 0
    N = 1000
    seed = 1
    dims = [5, 5, 5]  # Dimensions for X1, X2, X3
    # According to the image, Cov(X1, X2) is Q * R^T. 
    # In scalar terms, this is simply q * r.
    corr_matrix = np.array([
        [1.0,   p,  q  ],  # Row 1: X1
        [p, 1.0,    r  ],  # Row 2: X2
        [q,     r,      1.0]   # Row 3: X3
    ])

    true_cov = theoretical_covariance(dims, corr_matrix)
    torch_true_cov = torch.from_numpy(true_cov).to(device)
    rv_list, sample_cov = sample_cov_simulation(seed, N, dims, true_cov)
    torch_rv_list = [torch.from_numpy(rv).to(device) for rv in rv_list]  # Convert to torch tensors
    # assert len(torch_rv_list) == 3, "Expected 3 random variables in the list"
    # lo_cov_matrix = lo_cov(torch_rv_list,N)  # Example for the first random variable
    true_value,mi_true = para_Idep_multivariate_gauss(N=1,device=device,cov_matrix=torch_true_cov.unsqueeze(0),dims=dims).idep()
    pid_values, mi_values = idep_parallel(device=device,sources=torch_rv_list[:2],target=torch_rv_list[2:], N=N)

    # print("True PID values (no bias correction):")
    # for key, value in true_value.items():
    #     print(f"{key}: {torch.round(torch.tensor(value),decimals=4).item()}")
    # print("\nPID values (bias-corrected):" )
    # for key, value in pid_values.items():
    #     print(f"{key}: {torch.round(torch.tensor(value),decimals=4).item()}")
    # print(f"\nMI values (bias-corrected):")
    # for key, value in mi_values.items():
    #     print(f"{key}: {torch.round(torch.tensor(value),decimals=4).item()}")