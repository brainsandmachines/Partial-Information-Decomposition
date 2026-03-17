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


def theortical_covraince(dx1,dx2,dt,correlation,cross_cov=None):
    """ Helper: Create a covariance matrix with specified correlation between X and Y """
    theoretical_cov = np.eye(dx1 + dx2)
    if cross_cov is None:
        cross_cov = correlation * np.eye(min(dx1, dx2))

    theoretical_cov[:dx1, dx1:dx1+dx2] = cross_cov
    theoretical_cov[dx1:dx1+dx2, :dx1] = cross_cov.T

    eigvals = np.linalg.eigvalsh(theoretical_cov)
    if np.any(eigvals <= 0):
        raise ValueError("Theoretical covariance is not positive definite.")
    return theoretical_cov


def sample_cov_simulation(seed, N, px, py,theo_cov_matrix=None):
    """ Helper: Sample data from a Gaussian with specified covariance """
    rng = np.random.default_rng(seed)
    mean = np.zeros(px + py)
    X = rng.multivariate_normal(mean, theo_cov_matrix, size=N)

    X1 = X[:, :px]
    X2 = X[:, px:]


    X1_torch = torch.from_numpy(X1).to(torch.float64)
    X2_torch = torch.from_numpy(X2).to(torch.float64)

    rv_list = [X1, X2] 
    cov_blocks = create_cov_matrix(rvs=[X1_torch, X2_torch],verbose=False)
    sample_covariance = cov_blocks['full_cov'].numpy()
    return rv_list, sample_covariance

def mi_simulation(seed, N, px, py, correlation, cross_cov=None):
    
    theoretical_cov = theortical_covraince(px, py,dt=0 ,correlation=correlation,cross_cov=cross_cov)
    rvs, sample_cov = sample_cov_simulation(seed, N, px, py, theoretical_cov)

    assert sample_cov.shape == theoretical_cov.shape, "Sample covariance shape does not match theoretical covariance shape."
    assert sample_cov.shape[0] == sample_cov.shape[1], "Sample covariance dimension does not match total dimension (px + py)."
    p = px + py

    # Theoretical Entropy Constants
    _, log_det_theo = np.linalg.slogdet(theoretical_cov)
    _, log_det_theox1 = np.linalg.slogdet(theoretical_cov[:px, :px])
    _, log_det_theox2 = np.linalg.slogdet(theoretical_cov[px:, px:])

    # Sample Entropy
    _, log_det_sample = np.linalg.slogdet(sample_cov)
    _, log_det_samplex1 = np.linalg.slogdet(sample_cov[:px, :px])
    _, log_det_samplex2 = np.linalg.slogdet(sample_cov[px:, px:])

    # Theoretical MI
    mi_theoretical = 0.5 * (log_det_theox1 + log_det_theox2 - log_det_theo)
    # No bias correction MI
    mi_sample_no_bias = 0.5 * (log_det_samplex1 + log_det_samplex2 - log_det_sample)

    # Analytical Bias Correction
    df = N-1
    bias =   asymptotic_entropy_bias(df, px) + asymptotic_entropy_bias(df, py) - asymptotic_entropy_bias(df, px + py)
    mi_sample_with_bias = mi_sample_no_bias + bias


    return mi_theoretical, mi_sample_no_bias, mi_sample_with_bias

def theoretical_cov_simulation(seeds, N, px, py, correlation, cross_cov=None):
    """ Helper: Run multiple simulations to compare theoretical vs sample covariance """
    results_dict = {}
    for seed in seeds:
        mi_theoretical, mi_sample_no_bias, mi_sample_with_bias = mi_simulation(seed, N, px, py, correlation, cross_cov)
        if seed % 100 == 0:
            print(f"Completed seed {seed}/{len(seeds)}")
            
        results_dict[seed] = {'mi_theoretical': mi_theoretical, 'mi_sample_no_bias': mi_sample_no_bias, 'mi_sample_with_bias': mi_sample_with_bias}
    return results_dict

def mean_std_csv_results(results_dict):
    """ Helper: Compute mean results across seeds """
    df = pd.DataFrame.from_dict(results_dict, orient="index")
    mean_results = df.mean()
    std_results = df.std()
    return mean_results, std_results


def N_P_variation_simulation(seeds, N_values, p_values, correlation,cross_cov=None):
    """ Helper: Run simulations across different N and p values """
    all_results = []
    len_N = len(N_values)
    len_P = len(p_values)
    i=1
    for N in N_values:
        for p in p_values:
            print(f"\nRunning simulation for N={N}, p={p} ({i}/{len_N*len_P})")
            px = py = p // 2
            cross_cov_seed = 10001
            rng = np.random.default_rng(cross_cov_seed)
            A = rng.normal(scale=0.01, size=(min(px, py), min(px, py)))
            #cross_cov = A @ A.T
            results_dict = theoretical_cov_simulation(seeds, N, px, py, correlation, cross_cov)
            mean_results, std_results = mean_std_csv_results(results_dict)
            row = {
            "N": N,
            "p": p,
        }

            for key in mean_results.index:
                row[f"{key}_mean"] = mean_results[key]
                row[f"{key}_std"] = std_results[key]

                all_results.append(row)
            print(f"Completed combination N={N}, p={p} ({i}/{len_N * len_P})")
            i += 1

    return all_results


def create_heat_map(results_dict, save_path):
    """ Helper: Create heatmap of each results """
    df = pd.DataFrame.from_dict(results_dict, orient="index")
    df.index.name = "seed"
    plt.figure(figsize=(10, 6))
    plt.plot(df['mi_theoretical'], label='Theoretical MI', marker='o')
    plt.plot(df['mi_sample_no_bias'], label='Sample MI (No Bias Correction)', marker='x')
    plt.plot(df['mi_sample_with_bias'], label='Sample MI (With Bias Correction)', marker='s')
    plt.xlabel('Seed')
    plt.ylabel('Mutual Information')
    plt.title('Mutual Information Simulation Results')
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_path}/MI_bias_corr_simulation_results.png")
    plt.show()

if __name__ == "__main__":
    seeds_list = range(1500)
    N = 500
    px = 50
    py = 50
    correlation = 0
    exp_title = f"MI_Simulation_with_Correlation_{correlation}"
    save_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/Bias_Corr_Sim"

    N_p_var_results = N_P_variation_simulation(seeds_list, N_values=[1000,1500,3000], p_values=[100,200,300,400], correlation=correlation)
    df = pd.DataFrame(N_p_var_results)
    df.index.name = "seed"
    df.to_csv(f"{save_path}/{exp_title}_simulation.csv",index=False)  

    csv_path = f"{save_path}/{exp_title}_asymptotic_simulation.csv"
    # plot_mi_heatmap(csv_path=csv_path, value_col='mi_theoretical',title='Theoretical MI',save_path=save_path)
    # plot_mi_heatmap(csv_path=csv_path, value_col='mi_sample_no_bias',title='Sample MI (No Bias Correction)',save_path=save_path)
    # plot_mi_heatmap(csv_path=csv_path, value_col='mi_sample_with_bias',title='Sample MI (With Bias Correction)',save_path=save_path)
    plot_all_mi_heatmaps(csv_path=csv_path, save_path=save_path,title=f'{exp_title}',log_scale=False)
    plot_all_mi_heatmaps(csv_path=csv_path, save_path=save_path,title=f'Logscaled_{exp_title}',log_scale=True)