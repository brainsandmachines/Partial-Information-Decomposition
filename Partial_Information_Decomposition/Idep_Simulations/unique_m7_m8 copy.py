

import pathlib
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
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from Partial_Information_Decomposition.resampling_wrapper import bias_resampling
from Partial_Information_Decomposition.numertaor_m7_bias import  bias_m7_nume_second_order

def simulate_m7_m8_mi(
    data: list,
    sim_config: dict,
    rng: torch.Generator | None = None
):
    """
    Run MI simulation under the same covariance construction used
    in the logdet experiments.
    """
    n_samples = sim_config['n_samples']
    n0 = sim_config['n0']
    n1 = sim_config['n1']
    n2 = sim_config['n2']
    n_trials = sim_config['n_trials']
    device = sim_config['device']
    
    if n_samples < 3:
            raise ValueError("Need at least 3 samples.")

    d = n0 + n1 + n2
    df = n_samples - 1




    if df <= d - 1:
        raise ValueError(
            f"Need df > d-1 for stable logdet expectation. Got n_samples={n_samples}, df={df}, d={d}."
        )

        #Extract true covariances for m7 and m8 models
    m8_true_cov, m7_true_cov = data

    m8_true_cov_dict = create_cov_matrix(Sigma=m8_true_cov, dims=[n0, n1, n2])
    

    Q_m8_whiten = whiten_block(m8_true_cov_dict['cov_x0'], m8_true_cov_dict['cross_x0_x2'], m8_true_cov_dict['cov_x2'])
    R_m8_whiten = whiten_block(m8_true_cov_dict['cov_x1'], m8_true_cov_dict['cross_x1_x2'], m8_true_cov_dict['cov_x2'])
    P_m8_whiten = whiten_block(m8_true_cov_dict['cov_x0'], m8_true_cov_dict['cross_x0_x1'], m8_true_cov_dict['cov_x1'])
    row1_m8 = torch.cat([torch.eye(n0, device=device), P_m8_whiten, Q_m8_whiten], dim=1)
    row2_m8 = torch.cat([P_m8_whiten.T, torch.eye(n1, device=device), R_m8_whiten], dim=1)
    row3_m8 = torch.cat([Q_m8_whiten.T, R_m8_whiten.T, torch.eye(n2, device=device)], dim=1)
    m8_Sigma = torch.cat([row1_m8, row2_m8, row3_m8], dim=0)

    nume_m8_true = 0.5*safe_logdet(torch.eye(n1,device=device) - (P_m8_whiten.T @ P_m8_whiten)).item()
    deno_m8_true = 0.5*safe_logdet(m8_true_cov).item()
    m8_MI_true = (nume_m8_true-deno_m8_true)

    #Mutual Information between M1 or M2 with T
    i_m1_t_true = -0.5*safe_logdet(torch.eye(n2, device=device) - (Q_m8_whiten.T @ Q_m8_whiten)).item()
    i_m2_t_true = -0.5*safe_logdet(torch.eye(n2, device=device) - (R_m8_whiten.T @ R_m8_whiten)).item()


    m7_true_cov_dict = create_cov_matrix(Sigma=m7_true_cov, dims=[n0, n1, n2])
    Q_m7_whiten = whiten_block(m7_true_cov_dict['cov_x0'], m7_true_cov_dict['cross_x0_x2'], m7_true_cov_dict['cov_x2'])
    R_m7_whiten = whiten_block(m7_true_cov_dict['cov_x1'], m7_true_cov_dict['cross_x1_x2'], m7_true_cov_dict['cov_x2'])
    P_m7 = Q_m7_whiten @ R_m7_whiten.T
    row1_m7 = torch.cat([torch.eye(n0,device=device), P_m7, Q_m7_whiten], dim=1)
    row2_m7 = torch.cat([P_m7.T, torch.eye(n1,device=device), R_m7_whiten], dim=1)
    row3_m7 = torch.cat([Q_m7_whiten.T, R_m7_whiten.T, torch.eye(n2,device=device)], dim=1)   
    m7_Sigma = torch.cat([row1_m7, row2_m7, row3_m7], dim=0)

    nume_m7_true = 0.5*safe_logdet(torch.eye(n1,device=device) - (P_m7.T @ P_m7)).item()
    deno_m7_true = 0.5*safe_logdet(m7_true_cov).item()

    m7_MI_true = (nume_m7_true-deno_m7_true)

    #Unique 1
    i_true = m7_MI_true - i_m2_t_true
    k_true = m8_MI_true - i_m2_t_true
    
    #Unique 2
    h_true = m7_MI_true - i_m1_t_true
    j_true = m8_MI_true - i_m1_t_true


    unq1_dict_values = {'i':[],'k':[]}
    unq2_dict_values = {'h':[],'j':[]}

    unq1_corrected = {'i':[],'k':[]}
    unq2_corrected = {'h':[],'j':[]}

    pid_config = sim_config.copy()
    bias_calc_func = sim_config['bias_correction_func']

    m7_analytic_bias = calculate_bias(sim_config, m7=True, bias_correction=True)['bias']
    m8_analytic_bias = calculate_bias(sim_config, m8=True, bias_correction=True)['bias']
    m1_t_analytic_bias = 0.5*logdet_wishart_bias(df, n0) + 0.5*logdet_wishart_bias(df, n2) - 0.5*logdet_wishart_bias(df, n0+n2)
    m2_t_analytic_bias = 0.5*logdet_wishart_bias(df, n1) + 0.5*logdet_wishart_bias(df, n2) - 0.5*logdet_wishart_bias(df, n1+n2)

    for i in range(n_trials):
        print(f"Trial {i+1}/{n_trials}", end="\r")
        # Build true covariance exactly the same way

        # Sample data
        Z,rv_list = sample_data_from_cov(sim_config,m8_true_cov,rng=rng)
        Z_m7 = oas_cov_torch(Z, N=n_samples-1)
        # Sample covariance
        Z_dict = create_cov_matrix(Sigma=Z, dims=[n0, n1, n2])
        Z_m7 = create_cov_matrix(Sigma=Z_m7, dims=[n0, n1, n2])

        #Graph Model M8 
        Q_m8_whiten = whiten_block(Z_dict['cov_x0'], Z_dict['cross_x0_x2'], Z_dict['cov_x2'])
        R_m8_whiten = whiten_block(Z_dict['cov_x1'], Z_dict['cross_x1_x2'], Z_dict['cov_x2'])
        P_m8_whiten = whiten_block(Z_dict['cov_x0'], Z_dict['cross_x0_x1'], Z_dict['cov_x1'])
        row1_m8 = torch.cat([torch.eye(n0, device=device), P_m8_whiten, Q_m8_whiten], dim=1)
        row2_m8 = torch.cat([P_m8_whiten.T, torch.eye(n1, device=device), R_m8_whiten], dim=1)
        row3_m8 = torch.cat([Q_m8_whiten.T, R_m8_whiten.T, torch.eye(n2, device=device)], dim=1)
        m8_Sigma = torch.cat([row1_m8, row2_m8, row3_m8], dim=0)
        nume8_raw = 0.5*safe_logdet((torch.eye(n1, device=device) - (P_m8_whiten.T @ P_m8_whiten)))
        deno8_q = torch.eye(n2, device=device)-(Q_m8_whiten.T @ Q_m8_whiten)
        deno8_r = torch.eye(n2, device=device)-(R_m8_whiten.T @ R_m8_whiten)
        deno8_raw = 0.5*safe_logdet(m8_Sigma)
        mi_m8_raw = (nume8_raw - deno8_raw).item()
        
        i_m1_t_raw = -0.5*safe_logdet(torch.eye(n2, device=device) - (Q_m8_whiten.T @ Q_m8_whiten)).item()
        i_m2_t_raw = -0.5*safe_logdet(torch.eye(n2, device=device) - (R_m8_whiten.T @ R_m8_whiten)).item()

        #Graph Model M7 denominator
        Q_m7_whiten = whiten_block(Z_dict['cov_x0'], Z_dict['cross_x0_x2'], Z_dict['cov_x2'])
        R_m7_whiten = whiten_block(Z_dict['cov_x1'], Z_dict['cross_x1_x2'], Z_dict['cov_x2'])
        P_m7 = Q_m7_whiten @ R_m7_whiten.T
        row1_m7 = torch.cat([torch.eye(n0,device=device), P_m7, Q_m7_whiten], dim=1)
        row2_m7 = torch.cat([P_m7.T, torch.eye(n1,device=device), R_m7_whiten], dim=1)
        row3_m7 = torch.cat([Q_m7_whiten.T, R_m7_whiten.T, torch.eye(n2,device=device)], dim=1)   
        m7_Sigma = torch.cat([row1_m7, row2_m7, row3_m7], dim=0)
        nume7_raw = 0.5*safe_logdet(torch.eye(n1, device=device) - (P_m7.T @ P_m7))
        deno7_q = torch.eye(n2, device=device)-(Q_m7_whiten.T @ Q_m7_whiten)
        deno7_r = torch.eye(n2, device=device)-(R_m7_whiten.T @ R_m7_whiten)
        deno7_raw = 0.5*safe_logdet(deno7_q) + 0.5*safe_logdet(deno7_r)
        mi_m7_raw = (nume7_raw - deno7_raw).item()
        
        
        i_bias = (m7_analytic_bias - m2_t_analytic_bias)
        k_bias = (m8_analytic_bias - m2_t_analytic_bias)
        h_bias = (m7_analytic_bias - m1_t_analytic_bias)
        j_bias = (m8_analytic_bias - m1_t_analytic_bias)
        pid_config['analytic_bias'] = {'i': i_bias, 'k': k_bias,
                                            'h': h_bias, 'j': j_bias}
        #Raw PID Values
        i_raw =  mi_m7_raw - i_m2_t_raw
        i_raw -= i_bias
        k_raw = mi_m8_raw - i_m2_t_raw
        k_raw -= k_bias

        h_raw = mi_m7_raw - i_m1_t_raw
        h_raw -= h_bias

        j_raw = mi_m8_raw - i_m1_t_raw
        j_raw -= j_bias




        pid_config['sample_statistic'] = {'i': i_raw, 'k': k_raw,'h': h_raw, 'j': j_raw}

        pid_config['rvs_list'] = rv_list
        pid_config['Sigma'] = Z

        pid_config['model'] = 'pid'
        pid_bias_dict = bias_calc_func(pid_config)

        unq1_corrected['i'].append(i_raw - pid_bias_dict['i'])
        unq1_corrected['k'].append(k_raw - pid_bias_dict['k'])
        
        unq2_corrected['h'].append(h_raw - pid_bias_dict['h'])
        unq2_corrected['j'].append(j_raw - pid_bias_dict['j'])



        #Save raw values as well.
        unq1_dict_values['i'].append(i_raw)
        unq1_dict_values['k'].append(k_raw)
        
        unq2_dict_values['j'].append(j_raw)
        unq2_dict_values['h'].append(h_raw)

    
    i_sample = torch.tensor(unq1_dict_values['i'])
    k_sample = torch.tensor(unq1_dict_values['k'])
    h_sample = torch.tensor(unq2_dict_values['h'])
    j_sample = torch.tensor(unq2_dict_values['j'])

    
    avg_i_ = torch.mean(i_sample)
    avg_k_ = torch.mean(k_sample)
    avg_h_ = torch.mean(h_sample)
    avg_j_ = torch.mean(j_sample)

    avg_corrected_i = torch.mean(torch.tensor(unq1_corrected['i']))
    avg_corrected_k = torch.mean(torch.tensor(unq1_corrected['k']))

    avg_corrected_j = torch.mean(torch.tensor(unq2_corrected['j']))
    avg_corrected_h = torch.mean(torch.tensor(unq2_corrected['h']))


    i_dict= {'sample': i_sample, 
              'avg': avg_i_,
              'corrected_avg': avg_corrected_i,
              'std': torch.std(i_sample),
              'ground_truth': i_true}
    k_dict = {'sample': k_sample,
                    'avg': avg_k_,
                    'corrected_avg': avg_corrected_k,
                    'std': torch.std(k_sample),
                    'ground_truth': k_true}
    h_dict = {'sample': h_sample,
                    'avg': avg_h_,
                    'corrected_avg': avg_corrected_h,
                    'std': torch.std(h_sample),
                    'ground_truth': h_true}
    j_dict= {'sample': j_sample, 
              'avg': avg_j_,
                'corrected_avg': avg_corrected_j,
              'std': torch.std(j_sample),
              'ground_truth': j_true}


    
    return {'i': i_dict, 'k': k_dict, 'h': h_dict, 'j': j_dict}




def calculate_bias(config: dict,m8:bool=False,m8_nume:bool=False,m8_deno:bool=False, 
                   m7:bool=False,m7_deno:bool=False,m7_nume:bool=False,bias_correction:bool=True) -> dict:
   
    """
    Run the specified simulation function over combinations of N and p values, calculating mean and std of results.
    """
    if not bias_correction:
        return {'bias': 0.0} 
    
    assert m7 or m8 or m8_nume or m8_deno or m7_deno or m7_nume, "Must specify at least one of m7, m8, m8_nume, m8_deno, m7_deno, or m7_nume for bias calculation."
    n0 = config['n0']
    n1 = config['n1']
    n2 = config['n2']
    n_samples = config['n_samples']
    d = n0 + n1 + n2
    df = n_samples - 1
    # A. DEFINE BIAS TERMS
    # Marginal Biases (Fixed by whitening)
    bias_x0 = logdet_wishart_bias(df, n0)
    bias_x1 = logdet_wishart_bias(df, n1)
    bias_y  = logdet_wishart_bias(df, n2)
    

    if m7 or m7_deno:
        # M7 (Structural) Biases
        bias_02 = logdet_wishart_bias(df, n0 + n2) # Clique 0
        bias_12 = logdet_wishart_bias(df, n1 + n2) # Clique 1
        bias_2 = bias_y # seperator 2

        # M7 MI Bias = 0.5 * ( (B_pred_struct - B_marginals) - (B_joint_struct - B_marginals) )
        # Note: b_joint_m7 = b_c0 + b_c1 - b_sep
        bias_m7_structural = bias_02 + bias_12 - bias_2
        bias_m7 = 0.5 * (bias_m7_structural - (bias_x0 + bias_x1 + bias_y))
        mi_bias_m7 = 0.5 * (bias_x0 + bias_x1 + 2*bias_2 - bias_02 - bias_12)
        return {'bias': mi_bias_m7 if m7 else bias_m7}
    
    if m7_nume:
        c0 = n0/(n_samples)
        c1 = n1/(n_samples)
        bias_m7_nume = (
    -0.5 * torch.log(torch.tensor(1.0 - c0 - c1, dtype=torch.float64))
    + 0.5 * c0 * torch.log(torch.tensor(1.0 - c1, dtype=torch.float64))
    + 0.5 * c1 * torch.log(torch.tensor(1.0 - c0, dtype=torch.float64))).item()
        return {'bias': bias_m7_nume}
    
    # M8 (Saturated) Biases
    else:
        b_pred_m8 = logdet_wishart_bias(df, n0 + n1)
        b_joint_m8 = logdet_wishart_bias(df, d)
    
        nume_m8_bias = 0.5*(b_pred_m8 - (bias_x0 + bias_x1))
        deno_m8_bias =0.5*(b_joint_m8 - (bias_x0 + bias_x1 + bias_y))
        bias_m8_mi = (nume_m8_bias - deno_m8_bias)

        bias_model = bias_m8_mi if m8 else (nume_m8_bias if m8_nume else deno_m8_bias)
        return {'bias': bias_model} 
    
def sort_m7_m8_results(results_list):
    """ Helper: Sort results list by N and p values for  sperate by m7 and m8."""
    mi_m7_results_list = []
    nome_m7_results_list = []
    deno_m7_results_list = []

    mi_m8_results_list = []
    nome_m8_results_list = []
    deno_m8_results_list = []
    for res in results_list:
        N = res['N']
        p = res['p']
        mi_m7_results_list.append({'N': N, 'p': p, 'mean': res['M7_mi_mean'], 'std': res['M7_mi_std'], 'ground_truth': res['M7_mi_ground_truth']})
        nome_m7_results_list.append({'N': N, 'p': p, 'mean': res['M7_nume_mean'], 'std': res['M7_nume_std'], 'ground_truth': res['M7_nume_ground_truth']})
        deno_m7_results_list.append({'N': N, 'p': p, 'mean': res['M7_deno_mean'], 'std': res['M7_deno_std'], 'ground_truth': res['M7_deno_ground_truth']})

        mi_m8_results_list.append({'N': N, 'p': p, 'mean': res['M8_mi_mean'], 'std': res['M8_mi_std'], 'ground_truth': res['M8_mi_ground_truth']})
        nome_m8_results_list.append({'N': N, 'p': p, 'mean': res['M8_nume_mean'], 'std': res['M8_nume_std'], 'ground_truth': res['M8_nume_ground_truth']})
        deno_m8_results_list.append({'N': N, 'p': p, 'mean': res['M8_deno_mean'], 'std': res['M8_deno_std'], 'ground_truth': res['M8_deno_ground_truth']})

    return [mi_m7_results_list, nome_m7_results_list, deno_m7_results_list], [mi_m8_results_list, nome_m8_results_list, deno_m8_results_list]


def simulation_wrapper(config: dict) -> dict:
    """
    Run the logdet bias simulation for M7 and M8 models, returning a summary of results.
    """
    seed = config['seed']
    sim_func = simulate_m7_m8_mi


    pid_bootstrap_func = partial(bias_resampling,calc_func=para_unique_bias_calc)
    
    corr_value_func  = corrected_statistic
    functions_dict = {'s_simulation': sim_func, 'bias_correction': pid_bootstrap_func, 'corrected_statistic': corr_value_func}
    results_dict = simulation(config,functions_dict,seed=seed)
    return results_dict
    

if __name__ == "__main__":
    print("Running m7_whiten and M8 Simulation Mutual Information comparison simulation...")
    
    exp_name = 'NoBias_bigtest'
    yaml_file = f"/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/configs/sim.yaml"
    folder_path = f"/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/figures/MI_sim2.0"
    save_path = pathlib.Path(f"{folder_path}/{exp_name}")
    save_path.mkdir(parents=True, exist_ok=True)    
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

    mi_m7_result,nome_m7_list,deno_m7_list =m7_results_list[0] ,m7_results_list[1], m7_results_list[2]
    mi_m8_result,nome_m8_list,deno_m8_list =m8_results_list[0] ,m8_results_list[1], m8_results_list[2]
   
    #Plt Mutual Information results
    plot_heatmap_mean_std(mi_m7_result, title=f"Mutual Information M7 -{exp_name} - Mutual Information M7",save_path=save_path)
    plot_heatmap_mean_std(mi_m8_result, title=f"Mutual Information M8 -{exp_name} - Mutual Information M8",save_path=save_path)
    
    #Plot numerator
    plot_heatmap_mean_std(nome_m7_list, title=f"numerator M7 -{exp_name} - numerator M7",save_path=save_path)
    plot_heatmap_mean_std(nome_m8_list, title=f"numerator M8 -{exp_name} - numerator M8",save_path=save_path)
    
    #Plot denominator
    plot_heatmap_mean_std(deno_m7_list, title=f"denominator M7 -{exp_name} - denominator M7",save_path=save_path)
    plot_heatmap_mean_std(deno_m8_list, title=f"denominator M8 -{exp_name} - denominator M8",save_path=save_path)
    #Save config for file
    with open(f'{save_path}/{exp_name}_config.yaml', 'w') as f:
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