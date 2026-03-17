from matplotlib.pylab import eigvals
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import OAS
from scipy.special import digamma
import torch
from sklearn.model_selection import LeaveOneOut 
from PID_util import *
import pandas as pd
from bias_corr import entropy_bias_term,asymptotic_entropy_bias


import numpy as np
import torch

def theoretical_covariance(dims, corr_matrix):
    """ 
    Helper: Create a covariance matrix for N variables with specified block correlations.
    dims: list of integers representing the dimensions of each variable (e.g., [px, py, pz])
    corr_matrix: a 2D array (len(dims) x len(dims)) with pairwise correlations.
    """
    n_vars = len(dims)
    total_dim = sum(dims)
    theoretical_cov = np.eye(total_dim)
    
    # Calculate starting indices for each block
    starts = np.cumsum([0] + dims[:-1])
    
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            pi, pj = dims[i], dims[j]
            corr = corr_matrix[i, j]
            
            # Create cross-covariance block using the minimum dimension
            cross_block = corr * np.eye(min(pi, pj))
            block = np.zeros((pi, pj))
            block[:min(pi, pj), :min(pi, pj)] = cross_block
            
            start_i, start_j = starts[i], starts[j]
            
            # Assign to the upper and lower triangles of the full matrix
            theoretical_cov[start_i:start_i+pi, start_j:start_j+pj] = block
            theoretical_cov[start_j:start_j+pj, start_i:start_i+pi] = block.T

    eigvals = np.linalg.eigvalsh(theoretical_cov)
    if np.any(eigvals <= 0):
        raise ValueError("Theoretical covariance is not positive definite. Check your correlation matrix.")
        
    return theoretical_cov


def sample_cov_simulation(seed, N, dims, theo_cov_matrix):
    """ Helper: Sample data from a Gaussian and compute sample covariance """
    rng = np.random.default_rng(seed)
    total_dim = sum(dims)
    mean = np.zeros(total_dim)
    
    # Sample the joint distribution
    X = rng.multivariate_normal(mean, theo_cov_matrix, size=N)
    
    # Split into individual random variables based on dims
    starts = np.cumsum([0] + dims[:-1])
    rv_list = [X[:, start:start+d] for start, d in zip(starts, dims)]
    
    # Note: Using standard numpy covariance here for self-containment.
    # Replace the below line with your custom `create_cov_matrix` if you prefer your PyTorch implementation.
    sample_covariance = np.cov(X, rowvar=False) 
    
    return rv_list, sample_covariance


def mi_simulation(seed, N, dims, corr_matrix):
    
    # 1. Setup Covariances
    theoretical_cov = theoretical_covariance(dims, corr_matrix)
    rvs, sample_cov = sample_cov_simulation(seed, N, dims, theoretical_cov)

    assert sample_cov.shape == theoretical_cov.shape, "Sample covariance shape does not match theoretical."
    total_dim = sum(dims)
    
    # 2. Calculate Theoretical Entropies (Log Determinants)
    _, log_det_theo_joint = np.linalg.slogdet(theoretical_cov)
    
    log_dets_theo_marginal = []
    starts = np.cumsum([0] + dims[:-1])
    for start, d in zip(starts, dims):
        _, ld = np.linalg.slogdet(theoretical_cov[start:start+d, start:start+d])
        log_dets_theo_marginal.append(ld)

    # 3. Calculate Sample Entropies (Log Determinants)
    _, log_det_sample_joint = np.linalg.slogdet(sample_cov)
    
    log_dets_sample_marginal = []
    for start, d in zip(starts, dims):
        _, ld = np.linalg.slogdet(sample_cov[start:start+d, start:start+d])
        log_dets_sample_marginal.append(ld)

    # 4. Calculate Total Correlation (Multivariate MI)
    mi_theoretical = 0.5 * (sum(log_dets_theo_marginal) - log_det_theo_joint)
    mi_sample_no_bias = 0.5 * (sum(log_dets_sample_marginal) - log_det_sample_joint)

    # 5. Analytical Bias Correction
    df = N - 1
    # Note: Ensure `asymptotic_entropy_bias` is defined in your environment
    bias_marginals = sum(entropy_bias_term(df, d) for d in dims)
    bias_joint = entropy_bias_term(df, total_dim)
    
    bias = bias_marginals - bias_joint
    mi_sample_with_bias = mi_sample_no_bias + bias

    return mi_theoretical, mi_sample_no_bias, mi_sample_with_bias


def theoretical_cov_simulation(seeds, N, dims, corr_matrix):
    """ Helper: Run multiple simulations to compare theoretical vs sample covariance """
    results_dict = {}
    for seed in seeds:
        mi_theoretical, mi_sample_no_bias, mi_sample_with_bias = mi_simulation(seed, N, dims, corr_matrix)
        if seed % 100 == 0:
            print(f"Completed seed {seed}/{len(seeds)}")
            
        results_dict[seed] = {
            'mi_theoretical': mi_theoretical, 
            'mi_sample_no_bias': mi_sample_no_bias, 
            'mi_sample_with_bias': mi_sample_with_bias
        }
    return results_dict

def mean_std_csv_results(results_dict):
    """ Helper: Compute mean results across seeds """
    df = pd.DataFrame.from_dict(results_dict, orient="index")
    mean_results = df.mean()
    std_results = df.std()
    return mean_results, std_results


def N_P_variation_simulation(seeds, N_values, p_values, corr_matrix,cross_cov=None):
    """ Helper: Run simulations across different N and p values """
    all_results = []
    len_N = len(N_values)
    len_P = len(p_values)
    i=1
    for N in N_values:
        for p in p_values:
            print(f"\nRunning simulation for N={N}, p={p} ({i}/{len_N*len_P})")
            results_dict = theoretical_cov_simulation(seeds, N, dims=p, corr_matrix=corr_matrix)
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


if __name__ == "__main__":
    seeds_list = range(1000)
    N = 500
    px = 50
    py = 50
    correlation = 0
    rv_num = 3
    exp_title = f"3rvs_TMI_Simulation=0"
    save_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/Bias_Corr_Sim"
    p_list = [[100]*rv_num,[200]*rv_num,[250]*rv_num]
    corr_matrix = np.eye(rv_num)
    N_p_var_results = N_P_variation_simulation(seeds_list, N_values=[500,1000,1500,3000], p_values=p_list, corr_matrix=corr_matrix)
    df = pd.DataFrame(N_p_var_results)
    df.index.name = "seed"
    df.to_csv(f"{save_path}/{exp_title}_simulation.csv",index=False)  

    csv_path = f"{save_path}/{exp_title}_simulation.csv"
    # plot_mi_heatmap(csv_path=csv_path, value_col='mi_theoretical',title='Theoretical MI',save_path=save_path)
    # plot_mi_heatmap(csv_path=csv_path, value_col='mi_sample_no_bias',title='Sample MI (No Bias Correction)',save_path=save_path)
    # plot_mi_heatmap(csv_path=csv_path, value_col='mi_sample_with_bias',title='Sample MI (With Bias Correction)',save_path=save_path)
    plot_all_mi_heatmaps(csv_path=csv_path, save_path=save_path,title=f'{exp_title}',log_scale=False)
    plot_all_mi_heatmaps(csv_path=csv_path, save_path=save_path,title=f'Logscaled_{exp_title}',log_scale=True)