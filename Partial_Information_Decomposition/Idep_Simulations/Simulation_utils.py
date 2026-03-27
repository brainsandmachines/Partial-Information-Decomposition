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


def m7_m8_mean_std_csv_results(results_dict):
    """ Helper: Compute mean results across seeds """
    mean_results = {}
    std_results = {}
    for model in results_dict.keys():
        mean_results[model] = results_dict[model]['corrected_statistic']
        std_results[model] = results_dict[model]['std']
    return mean_results, std_results


def N_P_variation_simulation(config,mean_std_func=m7_m8_mean_std_csv_results):
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
            config['n_samples'] = N
            config['n0'] = p[0]
            config['n1'] = p[1]
            config['n2'] = p[2]
            results_dict = simulation_func(config)
            mean_results, std_results = mean_std_func(results_dict)
            row = {
            "N": N,
            "p": p,  
        }

            for key in mean_results.keys():
                row[f"{key}_mean"] = mean_results[key]
                row[f"{key}_std"] = std_results[key]
                row[f"{key}_ground_truth"] = results_dict[key]['ground_truth']

            all_results.append(row)
            print(f"Completed combination N={N}, p={p} ({i}/{len_N * len_P})")
            i += 1

    return all_results



def sample_data_from_cov(true_cov: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
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
    

def plot_heatmap_mean_std(
    results,
    x_col="N",
    y_col="p",
    mean_col="mean",
    std_col="std",
    ground_truth_col="ground_truth",
    title=None,
    cmap="viridis",
    figsize=(8, 6),
    save_path=None,
    mean_fmt=".3f",
    std_fmt=".3f",
):
    """
    Create a heatmap where:
        x-axis = N
        y-axis = p
        color  = mean
        text   = mean ± std

    Parameters
    ----------
    results : list[dict] or pd.DataFrame
        Each entry should look like:
        {'N': ..., 'p': ..., 'mean': ..., 'std': ...}

        Note:
        p can also be a list/tuple/array, and it will be converted
        to a string label automatically.
    """

    df = pd.DataFrame(results).copy()

    # Convert y values to hashable / displayable labels
    def make_label(v):
        if isinstance(v, np.ndarray):
            return str(v.tolist())
        if isinstance(v, (list, tuple)):
            return str(list(v))
        return str(v)

    df[y_col] = df[y_col].apply(make_label)

    # Safer than pivot if duplicates ever appear
    mean_mat = df.pivot_table(index=y_col, columns=x_col, values=mean_col, aggfunc="mean")
    std_mat = df.pivot_table(index=y_col, columns=x_col, values=std_col, aggfunc="mean")
    ground_truth_mat = df.pivot_table(index=y_col, columns=x_col, values=ground_truth_col, aggfunc="mean")

    # sort axes
    mean_mat = mean_mat.sort_index()
    mean_mat = mean_mat.reindex(sorted(mean_mat.columns), axis=1)

    std_mat = std_mat.reindex(index=mean_mat.index, columns=mean_mat.columns)
    ground_truth_mat = ground_truth_mat.reindex(index=mean_mat.index, columns=mean_mat.columns)
    data = mean_mat.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(len(mean_mat.columns)))
    ax.set_xticklabels(mean_mat.columns)
    ax.set_yticks(np.arange(len(mean_mat.index)))
    ax.set_yticklabels(mean_mat.index)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    if title is not None:
        ax.set_title(title)

    threshold = np.nanmean(data)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            m = mean_mat.iloc[i, j]
            s = std_mat.iloc[i, j]
            gt = ground_truth_mat.iloc[i, j]

            if pd.isna(m):
                text = "NA"
            elif pd.isna(s):
                text = f"{m:{mean_fmt}}\n\nGT={gt:{mean_fmt}}"
            else:
                text = f"{m:{mean_fmt}}\n±{s:{std_fmt}}\n\nGT={gt:{mean_fmt}}"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if pd.notna(m) and m < threshold else "black",
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean")

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(f'{save_path}/{title}.png', dpi=300, bbox_inches="tight")

    plt.show()


def corrected_statistic(statistics: np.ndarray, bias_correction: float) -> np.ndarray:
    """
    Apply bias correction to the raw statistics.
    """
    return statistics - bias_correction


