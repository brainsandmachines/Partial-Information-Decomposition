
"""
compare_mutual_information.py

Simulation comparing mutual information estimates under the same
conditions as the logdet simulation. This script assumes all required
functions are available from:

    from g_wishart_bias_corr import *

No functions are redefined here — everything is imported and reused.
"""

from py_compile import main

import torch
import numpy as np
import argparse
import yaml
from functools import partial
# Import all existing utilities from the user's module
from Simulation_utils import *
from wrapper_M7_M8_models import simulation
from Simulation_utils import *
from logdet_m7_m8 import  sort_m7_m8_results


def simulate_m7_m8_mi(
    m8_true_cov: np.ndarray,
    m7_true_cov: np.ndarray,
    n_samples: int,
    n0: int,
    n1: int,
    n2: int,
    n_trials: int = 1000,
    rng: np.random.Generator | None = None,
):
    """
    Run MI simulation under the same covariance construction used
    in the logdet experiments.
    """

    
    if n_samples < 3:
            raise ValueError("Need at least 3 samples.")

    d = n0 + n1 + n2
    df = n_samples - 1

    if df <= d - 1:
        raise ValueError(
            f"Need df > d-1 for stable logdet expectation. Got n_samples={n_samples}, df={df}, d={d}."
        )



    m8_true_cov_torch = torch.from_numpy(m8_true_cov).to(torch.float64)
    m8_true_cov_dict = create_cov_matrix(Sigma=m8_true_cov_torch, dims=[n0, n1, n2])

    p_m8 = (m8_true_cov_dict['cross_x0_x1']).numpy()#True covraince already whitened, so this is the P matrix for M8
    r_m8 = (m8_true_cov_dict['cross_x1_x2']).numpy() #True covraince already whitened, so this is the R matrix for M8
    q_m8 = (m8_true_cov_dict['cross_x0_x2']).numpy() #True covraince already whitened, so this is the Q matrix for M8
    nume8_true = safe_logdet(np.eye(n1) - (p_m8.T @ p_m8))
    deno8_true = safe_logdet(m8_true_cov)
    m8_MI_true = 0.5*(nume8_true-deno8_true)

    m7_true_cov_torch = torch.from_numpy(m7_true_cov).to(torch.float64)
    m7_true_cov_dict = create_cov_matrix(Sigma=m7_true_cov_torch, dims=[n0, n1, n2])

    p_m7 = (m7_true_cov_dict['cross_x0_x1']).numpy() #True covraince already whitened, so this is the P matrix for M7
    r_m7 = (m7_true_cov_dict['cross_x1_x2']).numpy() #True covraince already whitened, so this is the R matrix for M7
    q_m7 = (m7_true_cov_dict['cross_x0_x2']).numpy() #True covraince already whitened, so this is the Q matrix for M7
    nume7_true = safe_logdet(np.eye(n1) - (p_m7.T @ p_m7))
    deno7_true = safe_logdet(m7_true_cov)
    m7_MI_true = 0.5*(nume7_true-deno7_true)



    mi_m8_list = []
    mi_m7_naive_list = []

    for i in range(n_trials):
        print(f"Trial {i+1}/{n_trials}", end="\r")
        # Build true covariance exactly the same way

        # Sample data
        Z = sample_data_from_cov(m8_true_cov, n_samples,rng=rng)

        # Sample covariance
        Z_torch = torch.from_numpy(Z).to(torch.float64)
        Z_dict = create_cov_matrix(Sigma=Z_torch, dims=[n0, n1, n2])


        Q_m8_m7 = whiten_block(Z_dict['cov_x0'], Z_dict['cross_x0_x2'], Z_dict['cov_x2']).numpy()
        R_m8_m7 = whiten_block(Z_dict['cov_x1'], Z_dict['cross_x1_x2'], Z_dict['cov_x2']).numpy()
        P = whiten_block(Z_dict['cov_x0'], Z_dict['cross_x0_x1'], Z_dict['cov_x1']).numpy()

        #M8 true MI
        m8_white = np.block([
            [np.eye(n0),         P, Q_m8_m7],
            [P.T,  np.eye(n1),        R_m8_m7],
            [Q_m8_m7.T,           R_m8_m7.T,          np.eye(n2)],
        ])
        nume8 = safe_logdet(np.eye(n1) - (P.T @ P))
        deno8 = safe_logdet(m8_white)
        m8_raw = 0.5*(nume8-deno8) 

        #M7 true MI
        P_m7 = Q_m8_m7 @ R_m8_m7.T
        m7_white = np.block([
            [np.eye(n0),         P_m7, Q_m8_m7],
            [P_m7.T,  np.eye(n1),        R_m8_m7],
            [Q_m8_m7.T,           R_m8_m7.T,          np.eye(n2)],
        ])

        nume7 = safe_logdet(np.eye(n1) - (P_m7.T @ P_m7))
        deno7 = safe_logdet(np.eye(n2) - (Q_m8_m7.T @ Q_m8_m7)) + safe_logdet(np.eye(n2) - (R_m8_m7.T @ R_m8_m7))
        m7_raw = 0.5*(nume7-deno7)

        mi_m8_list.append(m8_raw)
        mi_m7_naive_list.append(m7_raw)

    
    mi_m8_sample = np.asarray(mi_m8_list)
    mi_m7_sample = np.asarray(mi_m7_naive_list)

    
    avg_m8 = np.mean(mi_m8_sample)
    avg_m7 = np.mean(mi_m7_sample)


    emp_bias_m8 = avg_m8 - m8_MI_true
    emp_bias_m7 = avg_m7 - m7_MI_true

    m8_dict= {'sample': mi_m8_sample, 
              'avg': avg_m8,
              'std': np.std(mi_m8_sample),
              'emp_bias': emp_bias_m8,
              'ground_truth': m8_MI_true}
    m7_dict= {'sample': mi_m7_sample, 
              'avg': avg_m7,
              'std': np.std(mi_m7_sample),
              'emp_bias': emp_bias_m7,
              'ground_truth': m7_MI_true}
    return {'M8': m8_dict, 'M7': m7_dict}




def calculate_bias(config: dict,m8:bool,m7:bool,m7_wishart:bool,bias_correction:bool=True) -> list[dict]:
    """
    Run the specified simulation function over combinations of N and p values, calculating mean and std of results.
    """
    n0 = config['n0']
    n1 = config['n1']
    n2 = config['n2']
    n_samples = config['n_samples']
    d = n0 + n1 + n2
    df = n_samples - 1
    # A. DEFINE BIAS TERMS
    # Marginal Biases (Fixed by whitening)
    b_x0 = logdet_wishart_bias(df, n0)
    b_x1 = logdet_wishart_bias(df, n1)
    b_y  = logdet_wishart_bias(df, n2)
    
    # M8 (Saturated) Biases
    if m8 or m7_wishart:
        b_pred_m8 = logdet_wishart_bias(df, n0 + n1)
        b_joint_m8 = logdet_wishart_bias(df, d)
    
                          # Separator
    
    # Final MI Bias Corrections (Whitened Scale)
    # M8 MI Bias = 0.5 * ( (B_pred - B_marginals) - (B_joint - B_marginals) )
        return {'bias': 0.5 * ( (b_pred_m8 - (b_x0 + b_x1)) - (b_joint_m8 - (b_x0 + b_x1 + b_y)) ) if bias_correction else 0.0}

    if m7:
        # M7 (Structural) Biases
        b_c0 = logdet_wishart_bias(df, n0 + n2) # Clique 0
        b_c1 = logdet_wishart_bias(df, n1 + n2) # Clique 1
        b_sep = b_y   
    
        # M7 MI Bias = 0.5 * ( (B_pred_struct - B_marginals) - (B_joint_struct - B_marginals) )
        # Note: b_joint_m7 = b_c0 + b_c1 - b_sep
        return {'bias': 0.5 * (b_x0 + b_x1 + 2*b_sep - b_c0 - b_c1) if bias_correction else 0.0}


def simulation_wrapper(config: dict) -> dict:
    """
    Run the logdet bias simulation for M7 and M8 models, returning a summary of results.
    """
    seed = config['seed']
    sim_func = simulate_m7_m8_mi
    m8_bias_func = partial(calculate_bias, m8=True, m7=False, m7_wishart=False,bias_correction=False)
    m7_bias_func = partial(calculate_bias, m8=False, m7=True, m7_wishart=False,bias_correction=False)
    bias_corr_func = {'M8': m8_bias_func, 'M7': m7_bias_func}
    corr_value_func  = corrected_statistic
    functions_dict = {'s_simulation': sim_func, 'bias_correction': bias_corr_func, 'corrected_statistic': corr_value_func}
    results_dict = simulation(config,functions_dict,seed=seed)
    return results_dict
    

if __name__ == "__main__":
    print("Running m7_whiten and M8 Simulation Mutual Information comparison simulation...")
    save_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/figures/MI_sim"
    yaml_file = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/configs/sim.yaml"
    with open(yaml_file, 'r') as f:
        super_config = yaml.safe_load(f)

        config = super_config['Mutual_Information_Simulation']

        n_p_config = super_config['N_P_variations']
        p_values = n_p_config['p_values']
        N_values = n_p_config['N_values']
        config['p_values'] = p_values
        config['N_values'] = N_values
        config['simulation_func'] = simulation_wrapper
    results = N_P_variation_simulation(config)
    m7_results_list, m8_results_list = sort_m7_m8_results(results)


        #m8_results, m7_results = simulation_wrapper(config)
    plot_heatmap_mean_std(m7_results_list, title="smalltest_Corrected_MI_M7",save_path=save_path)
    plot_heatmap_mean_std(m8_results_list, title="smalltest_Corrected_MI_M8",save_path=save_path)
    #Save config for file
    with open(f'{save_path}/MI_config.yaml', 'w') as f:
         yaml_config = {
            'simulation': 'logdet bias comparison for M7 and M8 models',
            'seed': config['seed'],
            'N_samples_values': N_values,
            'p_values': p_values,
            'p_scale': config['p_scale'],
            'q_scale': config['q_scale'],
            'r_scale': config['r_scale'],
         }
         yaml.safe_dump(yaml_config, f, sort_keys=False, allow_unicode=True)
    print("\nFinished simulation.")