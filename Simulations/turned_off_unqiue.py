import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root)) 
from toy_examples.toy_example import commonality_analysis
from Partial_Information_Decomposition.Idep_multivariate_gauss import Idep_multivariate_gauss
from Partial_Information_Decomposition.PID_util import compare_results
from utils import (
    extract_all_components,
    print_seed_summary,
    run_multi_seed_experiment,
    get_seed_runs_csv_path,
    save_seed_summary_csv,
)


def get_run_config() -> dict:
    return {
        "method": "ridge_cv",
        "n_seeds": 1000,
        "seed_start": 0,
        "snr": 10,
        "n": 10000,
        "p": 100,
        "r_str": 30,
        "u1_str": 15,
        "u2_str": 2,
        "results_dir": "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/turned_off_unique",
        "results_prefix": "seed_summary",
        "all_runs_results_prefix": "seed_runs",
        "progress_print_every": 100,
        "test_name": "turned_off_unique",
    }



def feature_creation(rng,r_str,u1_str,u2_str,unique_method = 'orthogonal', n=1024, p=100, mixing_dimension=None, snr=10.0, method='standard', show_diagnostic_plots=False):
    """
    Creates dummy predictors and a target
    
    Args:
        rng: Random number generator
        r_str: strength of redundant features
        u1_str: strength of unique features in source 1
        u2_str: strength of unique features in source 2
        n: Number of samples
        p: Number of features per source
        snr: Signal-to-noise ratio (signal_std / noise_std)
        method: Which R² computation to use: 'standard', 'ols_cv', or 'ridge_cv'
        
    Returns:
        dict: Commonality analysis results
    """
    # Generate the four feature tensors
    R = rng.standard_normal((n, p))
    U1 = rng.standard_normal((n, p))
    U2 = rng.standard_normal((n, p))
    
    

    signal = r_str * R + u1_str * U1 + u2_str * U2

    noise_std = np.std(signal) / snr

    y_real  = signal + noise_std * rng.standard_normal((signal.shape[0], signal.shape[1]))


    X_M1 = r_str * R + u1_str * U1 
    X_M1 += 0*noise_std * rng.standard_normal((X_M1.shape[0], X_M1.shape[1]))
    X_M2 = r_str * R + u2_str * U2
    X_M2 += 0*noise_std * rng.standard_normal((X_M2.shape[0], X_M2.shape[1]))

    return X_M1, X_M2, y_real




def test(rng, r_str, u1_str, u2_str, n=1024, p=100, snr=10.0, method='standard'):
    M1, M2, y_real = feature_creation(rng,r_str,u1_str,u2_str, n=n, p=p, snr=snr, method=method)
    ca_results = commonality_analysis(M1, M2, y_real, method=method)
    M1 = torch.tensor(M1)
    M2 = torch.tensor(M2)
    T = torch.tensor(y_real)
    pid_results,mi_results = Idep_multivariate_gauss(sources=[M1, M2], targets=[T], bias_correction=True).idep()

    return ca_results, pid_results, mi_results


def run_single_seed(seed: int, config: dict) -> dict:
    rng = np.random.default_rng(seed=seed)
    ca_results, pid_results, mi_results = test(
        rng,
        config["r_str"],
        config["u1_str"],
        config["u2_str"],
        n=config["n"],
        p=config["p"],
        snr=config["snr"],
        method=config["method"],
    )
    return extract_all_components(ca_results, pid_results, mi_results)



def main():
    config = get_run_config()
    summary, seed_rows = run_multi_seed_experiment(
        config,
        per_seed_runner=run_single_seed,
    )
    print_seed_summary(summary, n_seeds=config["n_seeds"], seed_start=config["seed_start"])
    all_runs_path = get_seed_runs_csv_path(config)
    summary_path = save_seed_summary_csv(summary, config)
    print(f"\nSaved all seed run results to: {all_runs_path}")
    print(f"Saved summary to: {summary_path}")

if __name__ == "__main__":
    main()