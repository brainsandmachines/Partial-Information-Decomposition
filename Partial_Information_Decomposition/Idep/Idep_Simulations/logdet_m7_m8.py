import numpy as np
from scipy.special import digamma
import sys
import os
from pathlib import Path
from Partial_Information_Decomposition.Idep.Idep_Simulations.Simulation_utils import *
from Partial_Information_Decomposition.Idep.Idep_Simulations.simulation_wrapper import simulation
import yaml
from functools import partial
from Partial_Information_Decomposition.Idep.non_parametric_bias_corr.resampling_wrapper import bias_resampling
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from PID_util import *




def simulate_m7_m8_log_det(
    data: list,
    sim_config: dict,
    rng: torch.Generator | None = None,
):
    """
    Inputs: 
    data: list containing the true covariance matrices for M7 and M8 models, in the form [m7_true_cov, m8_true_cov]
    
    config: dictionary containing simulation parameters:
        n_samples: number of samples to draw for each trial
        n0, n1, n2: dimensions of the X0, X1, Y blocks respectively
        n_trials: number of simulation trials to run

    rng: torch random generator for reproducibility
   
    Compare logdet bias for:
      1) Full sample covariance S                     (exact Wishart case)
      2) Whitened paper-style m7_whiten estimator           (not Wishart)
      3) Original-scale paper-style m7_whiten estimator     (not Wishart)

    Returns
    -------
    dict with empirical and corrected bias summaries.
    """
    n_samples = sim_config['n_samples']
    n0 = sim_config['n0']
    n1 = sim_config['n1']
    n2 = sim_config['n2']
    n_trials = sim_config['n_trials']
    bias_method = sim_config['bias_method']
    device = sim_config['device']

    if n_samples < 3:
        raise ValueError("Need at least 3 samples.")

    d = n0 + n1 + n2
    df = n_samples - 1

    #Extract true covariances for m7 and m8 models
    m8_true_cov, m7_true_cov = data

    if df <= d - 1:
        raise ValueError(
            f"Need df > d-1 for stable logdet expectation. Got n_samples={n_samples}, df={df}, d={d}."
        )


    #True logdet values 
    m8_true_logdet = safe_logdet(m8_true_cov)
    m7_true_cov_logdet = safe_logdet(m7_true_cov)


    logdets_m8_sample = []
    logdets_m7_sample = []

    logdets_m8_corrected = {}
    logdets_m7_corrected = {}
    

    for i in range(n_trials):
        if (i+1) % 100 == 0:
            print(f"Trial {i+1}/{n_trials}...")

        #Sample data and get sample covariance
        S,rv_list = sample_data_from_cov(config,true_cov=m8_true_cov,rng=rng)
        S_dict = create_cov_matrix(Sigma=S, dims=[n0, n1, n2],device=device)

        Q_m8_whiten = whiten_block(S_dict['cov_x0'], S_dict['cross_x0_x2'], S_dict['cov_x2'])
        R_m8_whiten = whiten_block(S_dict['cov_x1'], S_dict['cross_x1_x2'], S_dict['cov_x2'])
        P_m8_whiten = whiten_block(S_dict['cov_x0'], S_dict['cross_x0_x1'], S_dict['cov_x1'])
        row1_m8 = torch.cat([torch.eye(n0, device=device), P_m8_whiten, Q_m8_whiten], dim=1)
        row2_m8 = torch.cat([P_m8_whiten.T, torch.eye(n1, device=device), R_m8_whiten], dim=1)
        row3_m8 = torch.cat([Q_m8_whiten.T, R_m8_whiten.T, torch.eye(n2, device=device)], dim=1)
        m8_Sigma = torch.cat([row1_m8, row2_m8, row3_m8], dim=0)



        #Calculate m7_whiten model logdets
        Q_m7_whiten = whiten_block(S_dict['cov_x0'], S_dict['cross_x0_x2'], S_dict['cov_x2'])
        R_m7_whiten = whiten_block(S_dict['cov_x1'], S_dict['cross_x1_x2'], S_dict['cov_x2'])
        P_m7 = Q_m7_whiten @ R_m7_whiten.T
        row1_m7 = torch.cat([torch.eye(n0,device=device), P_m7, Q_m7_whiten], dim=1)
        row2_m7 = torch.cat([P_m7.T, torch.eye(n1,device=device), R_m7_whiten], dim=1)
        row3_m7 = torch.cat([Q_m7_whiten.T, R_m7_whiten.T, torch.eye(n2,device=device)], dim=1)   
        m7_Sigma = torch.cat([row1_m7, row2_m7, row3_m7], dim=0)


                # #Calculate m7 not whiten model logdets
        # Q_m7 = S_dict['cross_x0_x2']
        # R_m7 = S_dict['cross_x1_x2']
        # P_m7 = Q_m7 @ np.linalg.inv(S_dict['cov_x2']) @ R_m7.T
        # m7_org = np.block([
        #     [S_dict['cov_x0'].numpy(), P_m7.numpy(), S_dict['cross_x0_x2'].numpy()],
        #     [P_m7.numpy().T, S_dict['cov_x1'].numpy(), S_dict['cross_x1_x2'].numpy()],
        #     [S_dict['cross_x0_x2'].numpy().T, S_dict['cross_x1_x2'].numpy().T, S_dict['cov_x2'].numpy()]
        # ])




        m8_logdet_raw = safe_logdet(m8_Sigma)
        #m7_org_logdet_raw = safe_logdet(m7_org)
        m7_logdet_raw = safe_logdet(m7_Sigma)

        logdets_m8_sample.append(m8_logdet_raw)
        #log_dets_m7_org_structural.append(m7_org_logdet_raw)
        logdets_m7_sample.append(m7_logdet_raw)

        if bias_method[0] in ['jackknife', 'bootstrap']:
            sim_config_m8 = sim_config.copy()
            sim_config_m7 = sim_config.copy()

            sim_config_m8['model'] = 'M8'
            sim_config_m7['model'] = 'M7'

            sim_config_m8['sample_statistic'] = m8_logdet_raw
            sim_config_m7['sample_statistic'] = m7_logdet_raw
            
            sim_config_m8['calc_statistic_func'] = safe_logdet
            sim_config_m7['calc_statistic_func'] = safe_logdet

            sim_config_m8['rvs_list'] = rv_list
            sim_config_m7['rvs_list'] = rv_list

            sim_config_m8['Sigma'] = S
            sim_config_m7['Sigma'] = m7_Sigma
            for bc_method in bias_method:
                sim_config_m8['bias_method'] = [bc_method]
                sim_config_m7['bias_method'] = [bc_method]

                bias_corr_func = sim_config['bias_correction_func']
                m8_bias_corr = bias_corr_func['M8'](config=sim_config_m8)
                m7_bias_corr = bias_corr_func['M7'](config=sim_config_m7)

                m8_logdet = m8_logdet_raw - m8_bias_corr
                m7_logdet = m7_logdet_raw - m7_bias_corr
                
                if bc_method not in logdets_m8_corrected:
                    logdets_m8_corrected[bc_method] = [m8_logdet.item()]
                    logdets_m7_corrected[bc_method] = [m7_logdet.item()]
                else:  
                    logdets_m8_corrected[bc_method].append(m8_logdet.item()) 
                    logdets_m7_corrected[bc_method].append(m7_logdet.item())
            





    logdets_m8_sample = torch.tensor(logdets_m8_sample,device=device)
    logdets_m7_sample = torch.tensor(logdets_m7_sample,device=device)

    # Calculate mean raw values
    avg_m8 = torch.mean(logdets_m8_sample)
    avg_m7 = torch.mean(logdets_m7_sample)

    # Calculate Emperical Biases
    emp_bias_m8 = avg_m8 - m8_true_logdet
    emp_bias_m7_whiten_naive = avg_m7 - m7_true_cov_logdet
    avg_resmaple_m8 = {}
    avg_resmaple_m7 = {}

    if bias_method[0] in ['jackknife', 'bootstrap']:
        for bc_method in bias_method:

            avg_m8_corrected = torch.mean(torch.tensor(logdets_m8_corrected[bc_method], device=device))
            avg_m7_corrected = torch.mean(torch.tensor(logdets_m7_corrected[bc_method], device=device))

            avg_resmaple_m8[bc_method] = avg_m8_corrected
            avg_resmaple_m7[bc_method] = avg_m7_corrected
    else:
        avg_m8_corrected = None
        avg_m7_corrected = None

    m8_dict= {'sample': logdets_m8_sample, 'avg': avg_m8,'std': torch.std(logdets_m8_sample),'avg_resample': avg_m8_corrected, 'emp_bias': emp_bias_m8,'ground_truth': m8_true_logdet}
    
    m7_dict = {'sample': logdets_m7_sample, 'avg': avg_m7, 'std': torch.std(logdets_m7_sample), 'avg_resample': avg_m7_corrected, 'emp_bias': emp_bias_m7_whiten_naive, 'ground_truth': m7_true_cov_logdet}

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
    bias_x0 = logdet_wishart_bias(df=df, d=n0)
    bias_x1 = logdet_wishart_bias(df=df, d=n1)
    bias_y  = logdet_wishart_bias(df=df, d=n2)

    if m8:
        m8_wishart_bias_corr = logdet_wishart_bias(df, d) - (bias_x0 + bias_x1 + bias_y) if bias_correction else 0.0
     # 1. Calculate Marginal Biases
        return {'bias' : m8_wishart_bias_corr}
    if m7:
        # Bias correction for Chrodal Graphs: 
        bias_02 = logdet_wishart_bias(df=df, d=n0+n2)
        bias_12 = logdet_wishart_bias(df=df, d=n1+n2)
        bias_2 = logdet_wishart_bias(df=df, d=n2)
        bias_m7_structural = bias_02 + bias_12 - bias_2
        bias_m7_structural = bias_m7_structural - (bias_x0 + bias_x1 + bias_y) if bias_correction else 0.0
        return {'bias': bias_m7_structural}

    if m7_wishart:
        bias_m7_whiten_structural = bias_m7_structural - (bias_x0 + bias_x1 + bias_y) if bias_correction else 0.0
        return {'bias': bias_m7_whiten_structural}




def simulation_wrapper(config: dict) -> dict:
    """
    Run the logdet bias simulation for M7 and M8 models, returning a summary of results.
    """
    seed = config['seed']
    sim_func = simulate_m7_m8_log_det
    bias_method = config['bias_method']
    if bias_method[0] == 'analytic':
        print("Using analytic bias correction...")
        m8_bias_func = partial(calculate_bias, m8=True, m7=False, m7_wishart=False,bias_correction=config['bias_correction'])
        m7_bias_func = partial(calculate_bias, m8=False, m7=True, m7_wishart=False,bias_correction=config['bias_correction'])

    else:
        print("Using resampling bias correction...")
        m8_bias_func = bias_resampling
        m7_bias_func = bias_resampling
    bias_corr_func = {'M8': m8_bias_func, 'M7': m7_bias_func}
    corr_value_func  = corrected_statistic
    functions_dict = {'s_simulation': sim_func, 'bias_correction': bias_corr_func, 'corrected_statistic': corr_value_func}
    results = simulation(config,functions_dict,seed=seed)

    return results
    
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
    save_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/figures/logdet_sim/smalltest"
    yaml_file = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/configs/small_test.yaml"
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
    plot_heatmap_mean_std(m7_results_list, title="M7_LogDet_NoBias_Corrected_Highdim",save_path=save_path)
    plot_heatmap_mean_std(m8_results_list, title="M8_LogDet_NoBias_Corrected_Highdim",save_path=save_path)
    #Save config for file
    with open(f'{save_path}/aftersim_config.yaml', 'w') as f:
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


    