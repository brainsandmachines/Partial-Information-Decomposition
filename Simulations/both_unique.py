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
    create_test_histograms_with_kde,
    save_seed_summary_table_image,
)


def get_run_config() -> dict:
    return {
        "method": "ridge_cv",
        "n_seeds": 10000,
        "seed_start": 0,
        "snr": 10,
        "n": 10000,
        "p": 100,
        "unique_ratio": 0.5,
        "results_dir": "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/Equal_Unique_10snr",
        "results_prefix": "seed_summary",
        "all_runs_results_prefix": "seed_runs",
        "progress_print_every": 100,
        "test_name": "10snrboth_unique",
    }

def half_permute(rng,features,snr=10):
    n,p = features.shape
    
    n_real_dim = 0.5
    real_dim = int(p*n_real_dim) 
    idx = rng.permutation(p)
    real_dim_indices = idx[:real_dim]
    spurious_dim_indices = idx[real_dim:]
    real_feature_1 = features[:,real_dim_indices]
    real_feature_2 = features[:,spurious_dim_indices]
    rand_perm_1 = rng.permutation(n)
    rand_perm_2 = rng.permutation(n)
    noise_std = np.std(features) / snr
    noise_1 = rng.standard_normal((n,real_dim))
    noise_2 = rng.standard_normal((n,real_dim))

    X_M1 = np.hstack([real_feature_1, noise_std*noise_1])
    X_M2 = np.hstack([noise_std*noise_2, real_feature_2])

    return X_M1, X_M2

def orthogonal_vectors(rng, n, p,features,noise=None,singal=None,unique_ratio=None,function=None):
    d = int(p*unique_ratio)

    z1 = features[:,:d]
    z2 = features[:,d:]


    if function is not None:
        z1 = function(z1)
        z2 = function(z2)
    q,_ = np.linalg.qr(rng.standard_normal((p, p)))

    A = q[:,:d]
    B = q[:,d:]
    
    X_M1 = z1 @ A.T
    X_M2 = z2 @ B.T
    

    #Make targets: 
    q_noise,_ = np.linalg.qr(rng.standard_normal((p, p)))
    W1 = q_noise[:d,:d]
    W2 = q_noise[d:,d:]
    
    target = np.hstack([z1 @ W1.T, z2 @ W2.T])


    #Make orthogonal noise:
    if noise is  not None: 
        noise = np.linalg.qr(rng.standard_normal((n, 2*p)))[0]
        ortho_noise_1 = noise[:,:p]
        ortho_noise_2 = noise[:,p:2*p]
   
        X_M1 += ortho_noise_1 
        X_M2 += ortho_noise_2 


    return X_M1, X_M2,target



def feature_creation(rng,unique_ratio,unique_method = 'orthogonal', n=1024, p=100, mixing_dimension=None, snr=10.0, method='standard', show_diagnostic_plots=False):
    """
    Creates dummy predictors and a target
    
    Args:
        rng: Random number generator
        unique_ratio : number between 0 and 1, indicating the proportion of unique features in each source. For example, 0.5 means that half of the features in each source are unique, and the other half are shared.
        n: Number of samples
        p: Number of features per source
        mixing_dimension: If not None, apply a mixing matrix with this dimension to entangle features
        snr: Signal-to-noise ratio (signal_std / noise_std)
        method: Which R² computation to use: 'standard', 'ols_cv', or 'ridge_cv'
        
    Returns:
        dict: Commonality analysis results
    """
    # Generate the four feature tensors
    real_features = rng.standard_normal((n, p))
    
   
    # Target: only real features contribute
    betas = rng.standard_normal((p, p))
    signal = real_features @ betas
    noise_std = np.std(signal) / snr

    y_real  = signal + noise_std * rng.standard_normal((n,p))


    if unique_method == 'half_permute':
        X_M1, X_M2 = half_permute(rng, real_features)
    
    else:
        noise = noise_std * rng.standard_normal((n,p))
        X_M1,X_M2,target = orthogonal_vectors(rng, n, p,features=real_features ,noise=noise,unique_ratio=unique_ratio)
        y_real = target + + noise_std * rng.standard_normal((n,p))

    return X_M1, X_M2, y_real




def test_both_unique(rng, unique_ratio, n=1024, p=100, snr=10.0, method='standard'):
    M1, M2, y_real = feature_creation(rng,unique_ratio, n=n, p=p, snr=snr, method=method)
    ca_results = commonality_analysis(M1, M2, y_real, method=method)
    M1 = torch.tensor(M1)
    M2 = torch.tensor(M2)
    T = torch.tensor(y_real)
    pid_results,mi_results = Idep_multivariate_gauss(sources=[M1, M2], targets=[T], bias_correction=True).idep()

    return ca_results, pid_results, mi_results


def run_single_seed(seed: int, config: dict) -> dict:
    rng = np.random.default_rng(seed=seed)
    ca_results, pid_results, mi_results = test_both_unique(
        rng,
        config["unique_ratio"],
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
    #main()
    csv_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/Equal_Unique_10snr/seed_runs_10snrboth_unique.csv"
    output_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/Simulation_figs/Equal_Unique10snr"
    create_test_histograms_with_kde(csv_path, output_path,bar_color="#FA8100", kde_color="#000000")

    summary_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/Equal_Unique_10snr/seed_summary_10snrboth_unique.csv"
    save_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/Equal_Unique_10snr/10snrboth_unique_seed_summary_table.png"
    save_seed_summary_table_image(summary_path,image_path=save_path) 