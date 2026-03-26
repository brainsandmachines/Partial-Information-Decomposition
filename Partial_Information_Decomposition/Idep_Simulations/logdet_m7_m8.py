import numpy as np
from scipy.special import digamma
import sys
import os
from pathlib import Path
from Simulation_utils import *
from M7_M8_models import make_random_true_cov
import yaml
from functools import partial

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from PID_util import *
from M7_M8_models import *



def simulate_m7_m8_log_det(
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
    Compare logdet bias for:
      1) Full sample covariance S                     (exact Wishart case)
      2) Whitened paper-style m7_whiten estimator           (not Wishart)
      3) Original-scale paper-style m7_whiten estimator     (not Wishart)

    Returns
    -------
    dict with empirical and corrected bias summaries.
    """
    if n_samples < 3:
        raise ValueError("Need at least 3 samples.")

    d = n0 + n1 + n2
    df = n_samples - 1

    if df <= d - 1:
        raise ValueError(
            f"Need df > d-1 for stable logdet expectation. Got n_samples={n_samples}, df={df}, d={d}."
        )


    #True logdet values 
    m8_true_logdet_full = safe_logdet(m8_true_cov)
    m7_true_cov_logdet = safe_logdet(m7_true_cov)


    logdets_m8_sample = []
    logdets_m7_sample = []
    log_dets_m7_org_structural = []

    for i in range(n_trials):
        if (i+1) % 100 == 0:
            print(f"Trial {i+1}/{n_trials}...")

        #Sample data and get sample covariance
        S = sample_data_from_cov(true_cov=m8_true_cov, n_samples=n_samples, rng=rng)
        S_torch = torch.from_numpy(S).to(torch.float64)
        S_dict = create_cov_matrix(Sigma=S_torch, dims=[n0, n1, n2])




        # #Calculate m7 not whiten model logdets
        # Q_m7 = S_dict['cross_x0_x2']
        # R_m7 = S_dict['cross_x1_x2']
        # P_m7 = Q_m7 @ np.linalg.inv(S_dict['cov_x2']) @ R_m7.T
        # m7_org = np.block([
        #     [S_dict['cov_x0'].numpy(), P_m7.numpy(), S_dict['cross_x0_x2'].numpy()],
        #     [P_m7.numpy().T, S_dict['cov_x1'].numpy(), S_dict['cross_x1_x2'].numpy()],
        #     [S_dict['cross_x0_x2'].numpy().T, S_dict['cross_x1_x2'].numpy().T, S_dict['cov_x2'].numpy()]
        # ])


        #Calculate m7_whiten model logdets
        Q_m7_whiten = whiten_block(S_dict['cov_x0'], S_dict['cross_x0_x2'], S_dict['cov_x2']).numpy()
        R_m7_whiten = whiten_block(S_dict['cov_x1'], S_dict['cross_x1_x2'], S_dict['cov_x2']).numpy()
        P = Q_m7_whiten @ R_m7_whiten.T
        m7_whiten_white = np.block([
            [np.eye(n0),         P, Q_m7_whiten],
            [P.T,  np.eye(n1),        R_m7_whiten],
            [Q_m7_whiten.T,           R_m7_whiten.T,          np.eye(n2)],
        ])


        m8_logdet_raw = safe_logdet(S)
        #m7_org_logdet_raw = safe_logdet(m7_org)
        log_det_m7_raw = safe_logdet(m7_whiten_white)

            
        logdets_m8_sample.append(m8_logdet_raw)
        #log_dets_m7_org_structural.append(m7_org_logdet_raw)
        logdets_m7_sample.append(log_det_m7_raw)



    logdets_m8_sample = np.asarray(logdets_m8_sample)
    logdets_m7_sample = np.asarray(logdets_m7_sample)

    # Calculate mean raw values
    avg_m8 = np.mean(logdets_m8_sample)
    avg_m7_whiten_naive = np.mean(logdets_m7_sample)

    # Calculate Emperical Biases
    emp_bias_m8 = avg_m8 - m8_true_logdet_full
    emp_bias_m7_whiten_naive = avg_m7_whiten_naive - m7_true_cov_logdet

    m8_dict= {'m8_sample': logdets_m8_sample, 'm8_avg': avg_m8,'m8_std': np.std(logdets_m8_sample) ,'m8_emp_bias': emp_bias_m8,'ground_truth': m8_true_logdet_full}
    
    m7_dict = {'m7_sample': logdets_m7_sample, 'm7_avg': avg_m7_whiten_naive, 'm7_std': np.std(logdets_m7_sample), 'm7_emp_bias': emp_bias_m7_whiten_naive, 'ground_truth': m7_true_cov_logdet}

    return {'M8': m8_dict, 'M7': m7_dict}


def calculate_bias(config: dict,m8: bool,m7: bool,m7_wishart: bool,bias_correction: bool=True) -> list[dict]:
    """
    Run the specified simulation function over combinations of N and p values, calculating mean and std of results.
    """
    n0 = config['n0']
    n1 = config['n1']
    n2 = config['n2']
    n_samples = config['n_samples']
    d = n0 + n1 + n2
    df = n_samples - 1
    if m8:
        m8_wishart_bias_corr = logdet_wishart_bias(df, d) if bias_correction else 0.0
     # 1. Calculate Marginal Biases
        return {'m8_bias' : m8_wishart_bias_corr}
    if m7:
        bias_x0 = logdet_wishart_bias(df=df, d=n0)
        bias_x1 = logdet_wishart_bias(df=df, d=n1)
        bias_y  = logdet_wishart_bias(df=df, d=n2)

        # Bias correction for Chrodal Graphs: 
        bias_02 = logdet_wishart_bias(df=df, d=n0+n2)
        bias_12 = logdet_wishart_bias(df=df, d=n1+n2)
        bias_2 = logdet_wishart_bias(df=df, d=n2)
        bias_m7_structural = bias_02 + bias_12 - bias_2
        bias_m7_structural = bias_m7_structural - (bias_x0 + bias_x1 + bias_y) if bias_correction else 0.0
        return {'m7_bias': bias_m7_structural}

    if m7_wishart:
        bias_m7_whiten_structural = bias_m7_structural - (bias_x0 + bias_x1 + bias_y) if bias_correction else 0.0
        return {'m7_wishart_bias': bias_m7_whiten_structural}




def simulation_wrapper(config: dict) -> dict:
    """
    Run the logdet bias simulation for M7 and M8 models, returning a summary of results.
    """
    seed = config['seed']
    sim_func = simulate_m7_m8_log_det
    m8_bias_func = partial(calculate_bias, m8=True, m7=False, m7_wishart=False,bias_correction=False)
    m7_bias_func = partial(calculate_bias, m8=False, m7=True, m7_wishart=False,bias_correction=False)
    bias_corr_func = {'m8': m8_bias_func, 'm7': m7_bias_func}
    corr_value_func  = corrected_statistic
    functions_dict = {'s_simulation': sim_func, 'bias_correction': bias_corr_func, 'corrected_statistic': corr_value_func}
    m8_results, m7_results = simulation(config,functions_dict,seed=seed)
    return {"M8": m8_results, "M7": m7_results}
    
def sort_m7_m8_results(results_list):
    """ Helper: Sort results list by N and p values for  sperate by m7 and m8."""
    m7_results_list = []
    m8_results_list = []
    for res in results_list:
        N = res['N']
        p = res['p']
        m7_mean = res['M7_mean']
        m7_std = res['M7_std']
        m7_results_list.append({'N': N, 'p': p, 'mean': m7_mean, 'std': m7_std, 'ground_truth': res['M7_ground_truth']})

        m8_mean = res['M8_mean']
        m8_std = res['M8_std']
        m8_results_list.append({'N': N, 'p': p, 'mean': m8_mean, 'std': m8_std, 'ground_truth': res['M8_ground_truth']})
    return m7_results_list, m8_results_list

if __name__ == "__main__":
    print("Running m7_whiten and M8 Simulation logdet bias comparison simulation...")
    save_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/figures/logdet_sim"
    yaml_file = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/configs/sim.yaml"
    with open(yaml_file, 'r') as f:
        super_config = yaml.safe_load(f)

        config = super_config['logdet_simulation']

        n_p_config = super_config['N_P_variations']
        p_values = n_p_config['p_values']
        N_values = n_p_config['N_values']
        config['p_values'] = p_values
        config['N_values'] = N_values
        config['simulation_func'] = simulation_wrapper
    results = N_P_variation_simulation(config)
    m7_results_list, m8_results_list = sort_m7_m8_results(results)

        
    #m8_results, m7_results = simulation_wrapper(config)
    plot_heatmap_mean_std(m7_results_list, title="Bias_Corrected_MI_M7",save_path=save_path)
    plot_heatmap_mean_std(m8_results_list, title="Bias_Corrected_MI_M8",save_path=save_path)
    #Save config for file
    with open(f'{save_path}/exp_config.yaml', 'w') as f:
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


    