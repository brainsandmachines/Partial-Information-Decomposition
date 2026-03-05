import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
import csv
import sys
from pathlib import Path
from pathlib import Path
from typing import Sequence, Union, Optional

import pandas as pd
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root)) 
from Commonality_Analysis.CA import commonality_analysis
from Partial_Information_Decomposition.Idep_multivariate_gauss import Idep_multivariate_gauss
from Partial_Information_Decomposition.PID_util import compare_results
from utils import (
    create_test_histograms_with_kde,
    extract_all_components,
    print_seed_summary,
    run_multi_seed_experiment,
    get_seed_runs_csv_path,
    save_seed_summary_csv,
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
        "r_str": 30,
        "u1_str": 15,
        "u2_str": 2    ,
        "results_dir": "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/Turned_off_unique_exp2",
        "results_prefix": "seed_summary",
        "all_runs_results_prefix": "seed_runs",
        "progress_print_every": 100,
        "test_name": "turned_off_unique_Exp2",
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


    X_M1 =  r_str * R + u1_str * U1 
    #X_M1 += noise_std * rng.standard_normal((X_M1.shape[0], X_M1.shape[1]))
    X_M2 = r_str * R + u2_str * U2
    #X_M2 += noise_std * rng.standard_normal((X_M2.shape[0], X_M2.shape[1]))

    return X_M1, X_M2, y_real

def standardize(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Standardize columns of X to zero mean and unit variance.

    X shape: (N, P)
    """
    mean = np.mean(X, axis=0, keepdims=True)
    std  = np.std(X, axis=0, ddof=0, keepdims=True)
    return (X - mean) / (std + 1e-12)


def test(rng, r_str, u1_str, u2_str, n=1024, p=100, snr=10.0, method='standard'):
    M1, M2, y_real = feature_creation(rng,r_str,u1_str,u2_str, n=n, p=p, snr=snr, method=method)
    M1 = standardize(M1)
    M2 = standardize(M2)
    y_real = standardize(y_real)
    CA = commonality_analysis(M1, M2, y_real, method=method)
    ca_results = CA.ca(M1, M2, y_real, method=method)
    betas_dict = extract_betas(ca_results)
    M1 = torch.tensor(M1)
    M2 = torch.tensor(M2)
    T = torch.tensor(y_real)
    pid_results,mi_results = Idep_multivariate_gauss(sources=[M1, M2], targets=[T], bias_correction=True).idep()

    return ca_results,betas_dict, pid_results, mi_results

def extract_betas(ca_results):
    X1_betas = ca_results['betas_X1']
    X2_betas = ca_results['betas_X2']
    X12_betas = ca_results['betas_X12']
    y_features,p1 = X1_betas.shape
    _,p2 = X2_betas.shape
    _,p12 = X12_betas.shape

    X12_1_betas = X12_betas[:, :p1]
    X12_2_betas = X12_betas[:, p1:p12]

    return {
        "X1_betas": np.mean(np.mean(X1_betas, axis=1)),
        "X2_betas": np.mean(np.mean(X2_betas, axis=1)),
        "X12_betas": np.mean(np.mean(X12_betas, axis=1)),
        "X12_1_betas": np.mean(np.mean(X12_1_betas, axis=1)),
        "X12_2_betas": np.mean(np.mean(X12_2_betas, axis=1)),
    }

def run_single_seed(seed: int, config: dict) -> dict:
    rng = np.random.default_rng(seed=seed)
    ca_results,betas_dict, pid_results, mi_results = test(
        rng,
        config["r_str"],
        config["u1_str"],
        config["u2_str"],
        n=config["n"],
        p=config["p"],
        snr=config["snr"],
        method=config["method"],
    )
    outputs = extract_all_components(ca_results, pid_results, mi_results, betas_dict)
    results = {
        "ca_results": ca_results,
        "pid_results": pid_results,
        "mi_results": mi_results,
    }
    return outputs, results

def test_regularization_term(seed: int, config: dict):
    rng = np.random.default_rng(seed=seed)
    M1, M2, y_real = feature_creation(rng,config["r_str"],config["u1_str"],config["u2_str"], n=config["n"], p=config["p"], snr=config["snr"], method=config["method"])
    M1 = standardize(M1)
    M2 = standardize(M2)
    y_real = standardize(y_real)

    #Commonality analysis with RidgeCV regularization
    CA = commonality_analysis(M1, M2, y_real, method=config["method"])
    best_alpha = CA.find_best_alpha(alphas=None)
    alphas =  np.linspace(0.001, best_alpha, 5000)
    alpha_results = {}
    for i,alpha in enumerate(alphas):
        if i % 100 == 0:
            print(f"Testing alpha={alpha:.5f} ({i+1}/{len(alphas)})")
        ca_results = CA.ca(M1, M2, y_real, method=config["method"], alphas=[alpha])
        #Partial Information Decomposition with Idep 
        X_M1 = torch.tensor(M1)
        X_M2 = torch.tensor(M2)
        T = torch.tensor(y_real)
        pid_results,mi_results = Idep_multivariate_gauss(sources=[X_M1, X_M2], targets=[T], bias_correction=True).idep()
        alpha_results[alpha] = {
            "ca_results": ca_results,
            "pid_results": pid_results,
            "mi_results": mi_results,
        }
    return alpha_results


def save_term_results_csv(x_axis: str, term_results: dict, output_csv_path: str | Path) -> Path:
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for x_value, results in term_results.items():
        flat_metrics = extract_all_components(
            results.get("ca_results", {}),
            results.get("pid_results", {}),
            results.get("mi_results", {}),
        )
        row = {x_axis: float(x_value)}
        row.update(flat_metrics)
        rows.append(row)


    metric_columns = sorted({key for row in rows for key in row.keys() if key != x_axis})
    fieldnames = [x_axis, *metric_columns]

    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path

def test_u2str(seed: int, config: dict, final_ratio: float):

    results = {}
    u2_str = config['u1_str']
    current_ratio = u2_str / config['u1_str']
    while current_ratio >= final_ratio:
        rng = np.random.default_rng(seed=seed)
        print(f"Testing u2_str={u2_str:.4f} (ratio={current_ratio:.4f})")
        test_results = test(
            rng,
            config["r_str"],
            config["u1_str"],
            u2_str,
            n=config["n"],
            p=config["p"],
            snr=config["snr"],
            method=config["method"],
        )
        ca_results,betas_dict, pid_results, mi_results = test_results
        results[current_ratio] = {
            "ca_results": ca_results,
            "pid_results": pid_results,
            "mi_results": mi_results,
        }
        u2_str *= 0.9 
        current_ratio = u2_str / config['u1_str']
    return results




def plot_keys_vs_alpha(
    csv_path: Union[str, Path],
    keys: Sequence[str],
    *,
    x_col: str = "alpha",
    sort_alpha: bool = False,
    logx: bool = False,
    figsize: tuple[float, float] = (8, 4.5),
    marker: Optional[str] = None,   # e.g. ".", "o", or None
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot selected columns (keys) vs x_col from a CSV file.

    Args:
        csv_path: Path to CSV.
        keys: Column names to plot (must exist in CSV).
        x_col: Name of the x-axis column (default: "alpha").
        sort_alpha: Sort rows by x_col before plotting.
        logx: Use log scale on x-axis (recommended for ridge grids).
        figsize: Figure size.
        marker: Optional marker style.
    """
    df = pd.read_csv(csv_path)

    if x_col not in df.columns:
        raise ValueError(f"CSV must include an '{x_col}' column. Found: {list(df.columns)}")

    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"Missing keys in CSV: {missing}\nAvailable columns: {list(df.columns)}")

    # Keep only needed columns, coerce to numeric in case something was read as string
    cols = [x_col] + list(keys)
    df = df[cols].copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with NaNs in required columns
    df = df.dropna(subset=cols)

    if sort_alpha:
        df = df.sort_values(x_col)

    alpha = df[x_col].to_numpy()

    plt.figure(figsize=figsize)
    for k in keys:
        y_value = df[k].to_numpy()
        plt.plot(alpha, np.round(y_value,decimals=5), label=k, marker=marker,color=np.random.rand(3,))  # Random color for each key

    if logx:
        plt.yscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel(x_col)
    plt.ylabel("value")
    plt.title(f"Selected Keys vs {x_col}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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

    #Regular test of this example: 
    # _,results = run_single_seed(seed=0, config=get_run_config())
    # compare_results(results["ca_results"], results["pid_results"], results["mi_results"])
    # csv_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/turned_off_unique/seed_runs_turned_off_unique.csv"
    # output_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/Simulation_figs/turned_off_unique"
    # create_test_histograms_with_kde(csv_path, output_path,bar_color="#C4D200", kde_color="#000000")

    # summary_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/turned_off_unique/seed_summary_turned_off_unique.csv"
    # save_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/turned_off_unique/turned_off_unique_seed_summary_table.png"
    # save_seed_summary_table_image(summary_path,image_path=save_path) 

    #Regularization term testing str are all consant
    csv_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/turned_off_unique_exp2/alpha_results.csv"
    save_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/turned_off_unique_exp2/alpha_function.png"
    test_regularization_term(seed=0, config=get_run_config())
    plot_keys_vs_alpha(x_col="alpha", csv_path=csv_path, keys=['CA_unique_X2'], logx=False,save_path=save_path, sort_alpha=False)


    #Testing different u2_str ratios to u1_str:

    # u2str_results = test_u2str(seed=0, config=get_run_config(), final_ratio=0.000001)
    # save_term_results_csv(x_axis="unique_ratio", term_results=u2str_results, output_csv_path="/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/turned_off_unique_exp2/u2_str_results.csv")
    # csv_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/turned_off_unique_exp2/u2_str_results.csv"
    # save_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/turned_off_unique_exp2"
    # keys = ['CA_unique_X2']
    # plot_keys_vs_alpha(x_col="unique_ratio", csv_path=csv_path, keys=keys, save_path=f"{save_path}/ca_unique.png",sort_alpha=False)
    # keys = ['PID_unq2']
    # plot_keys_vs_alpha(x_col="unique_ratio", csv_path=csv_path, keys=keys, save_path=f"{save_path}/pid_unique.png",sort_alpha=False)
