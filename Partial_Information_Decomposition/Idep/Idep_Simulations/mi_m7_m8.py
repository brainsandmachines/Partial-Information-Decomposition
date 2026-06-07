

import pathlib
from py_compile import main
import torch
import numpy as np
import argparse
import yaml
from functools import partial
# Import all existing utilities from the user's module

from Partial_Information_Decomposition.Idep.Idep_Simulations.Simulation_utils import *
from Partial_Information_Decomposition.Idep.Idep_Simulations.simulation_wrapper import simulation
from Partial_Information_Decomposition.Idep.Idep_Simulations.Simulation_utils import *
from Partial_Information_Decomposition.Idep.Idep_Simulations.logdet_m7_m8 import  sort_m7_m8_results
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from Partial_Information_Decomposition.Idep.non_parametric_bias_corr.resampling_wrapper import bias_resampling
from Partial_Information_Decomposition.bias_functions import logdet_wishart_bias, permutation_null_debias, permuteation_debiased
from mi_functions import calcualte_mi,para_calcualte_mi
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
    sim_config['rng'] = rng
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
    

    Q_m8_whiten = whiten_block(m8_true_cov_dict['cov_x1'], m8_true_cov_dict['cross_x1_xt'], m8_true_cov_dict['cov_xt'])
    R_m8_whiten = whiten_block(m8_true_cov_dict['cov_x2'], m8_true_cov_dict['cross_x2_xt'], m8_true_cov_dict['cov_xt'])
    P_m8_whiten = whiten_block(m8_true_cov_dict['cov_x1'], m8_true_cov_dict['cross_x1_x2'], m8_true_cov_dict['cov_x2'])
    row1_m8 = torch.cat([torch.eye(n0, device=device), P_m8_whiten, Q_m8_whiten], dim=1)
    row2_m8 = torch.cat([P_m8_whiten.T, torch.eye(n1, device=device), R_m8_whiten], dim=1)
    row3_m8 = torch.cat([Q_m8_whiten.T, R_m8_whiten.T, torch.eye(n2, device=device)], dim=1)
    m8_Sigma = torch.cat([row1_m8, row2_m8, row3_m8], dim=0)

    nume_m8_true = 0.5*safe_logdet(torch.eye(n1,device=device) - (P_m8_whiten.T @ P_m8_whiten)).item()
    deno_m8_true = 0.5*safe_logdet(m8_true_cov).item()
    m8_MI_true = (nume_m8_true-deno_m8_true)

    m7_true_cov_dict = create_cov_matrix(Sigma=m7_true_cov, dims=[n0, n1, n2])

    Q_m7_whiten = whiten_block(m7_true_cov_dict['cov_x1'], m7_true_cov_dict['cross_x1_xt'], m7_true_cov_dict['cov_xt'])
    R_m7_whiten = whiten_block(m7_true_cov_dict['cov_x2'], m7_true_cov_dict['cross_x2_xt'], m7_true_cov_dict['cov_xt'])
    P_m7 = Q_m7_whiten @ R_m7_whiten.T
    row1_m7 = torch.cat([torch.eye(n0,device=device), P_m7, Q_m7_whiten], dim=1)
    row2_m7 = torch.cat([P_m7.T, torch.eye(n1,device=device), R_m7_whiten], dim=1)
    row3_m7 = torch.cat([Q_m7_whiten.T, R_m7_whiten.T, torch.eye(n2,device=device)], dim=1)   
    m7_Sigma = torch.cat([row1_m7, row2_m7, row3_m7], dim=0)

    nume_m7_true = 0.5*safe_logdet(torch.eye(n1,device=device) - (P_m7.T @ P_m7)).item()
    deno_m7_true = 0.5*safe_logdet(m7_true_cov).item()

    m7_MI_true = (nume_m7_true-deno_m7_true)



    m8_dict_values = {'mi':[],'nume':[],'deno':[]}
    m7_dict_values = {'mi':[],'nume':[],'deno':[]}


    mi_m8_corrected = {'mi':[],'nume':[],'deno':[]}
    mi_m7_corrected = {'mi':[],'nume':[],'deno':[]}

    sim_config_m8 = sim_config.copy()
    sim_config_m7 = sim_config.copy()

    sim_config_m8['model'] = 'M8'
    sim_config_m7['model'] = 'M7'

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
        Q_m8_whiten = whiten_block(Z_dict['cov_x1'], Z_dict['cross_x1_xt'], Z_dict['cov_xt'])
        R_m8_whiten = whiten_block(Z_dict['cov_x2'], Z_dict['cross_x2_xt'], Z_dict['cov_xt'])
        P_m8_whiten = whiten_block(Z_dict['cov_x1'], Z_dict['cross_x1_x2'], Z_dict['cov_x2'])
        row1_m8 = torch.cat([torch.eye(n0, device=device), P_m8_whiten, Q_m8_whiten], dim=1)
        row2_m8 = torch.cat([P_m8_whiten.T, torch.eye(n1, device=device), R_m8_whiten], dim=1)
        row3_m8 = torch.cat([Q_m8_whiten.T, R_m8_whiten.T, torch.eye(n2, device=device)], dim=1)
        m8_Sigma = torch.cat([row1_m8, row2_m8, row3_m8], dim=0)
        nume8_raw = 0.5*safe_logdet((torch.eye(n1, device=device) - (P_m8_whiten.T @ P_m8_whiten)))
        deno8_q = torch.eye(n2, device=device)-(Q_m8_whiten.T @ Q_m8_whiten)
        deno8_r = torch.eye(n2, device=device)-(R_m8_whiten.T @ R_m8_whiten)
        deno8_raw = 0.5*safe_logdet(m8_Sigma)
        mi_m8_raw = (nume8_raw - deno8_raw).item()


        # #Graph Model M7 numerator
        # Q_m7_whiten = whiten_block(Z_m7['cov_x0'], Z_m7['cross_x0_x2'], Z_m7['cov_x2'])
        # R_m7_whiten = whiten_block(Z_m7['cov_x1'], Z_m7['cross_x1_x2'], Z_m7['cov_x2'])
        # P_m7 = Q_m7_whiten @ R_m7_whiten.T
        # row1_m7 = torch.cat([torch.eye(n0,device=device), P_m7, Q_m7_whiten], dim=1)
        # row2_m7 = torch.cat([P_m7.T, torch.eye(n1,device=device), R_m7_whiten], dim=1)
        # row3_m7 = torch.cat([Q_m7_whiten.T, R_m7_whiten.T, torch.eye(n2,device=device)], dim=1)   
        # m7_Sigma_oracle = torch.cat([row1_m7, row2_m7, row3_m7], dim=0)
        

        #Graph Model M7 denominator
        Q_m7_whiten = whiten_block(Z_dict['cov_x1'], Z_dict['cross_x1_xt'], Z_dict['cov_xt'])
        R_m7_whiten = whiten_block(Z_dict['cov_x2'], Z_dict['cross_x2_xt'], Z_dict['cov_xt'])
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
        




        sim_config_m8['sample_statistic'] = {'mi': mi_m8_raw, 'nume': nume8_raw, 'deno': deno8_raw}
        sim_config_m7['sample_statistic'] = {'mi': mi_m7_raw, 'nume': nume7_raw, 'deno': deno7_raw}

        #sim_config_m8['calc_statistic_func'] = mi_calculation_from_cov
        #sim_config_m7['calc_statistic_func'] = mi_calculation_from_cov

        sim_config_m8['rvs_list'] = rv_list
        sim_config_m7['rvs_list'] = rv_list

        sim_config_m8['Sigma'] = Z_dict['full_cov']
        sim_config_m7['Sigma'] = Z
        sim_config_m7['p'] = P_m7.detach().clone()
        
        sim_config_m7['X1'],sim_config_m7['X2'],sim_config_m7['T'] = rv_list[0],rv_list[1],rv_list[2]


        bias_corr_func = sim_config['bias_correction_func']
        # m8_bias_corr = bias_corr_func['M8'](config=sim_config_m8)
        # m7_bias_corr = bias_corr_func['M7'](config=sim_config_m7)
        
        m8_bias_dict = mi_bias_calc(sim_config_m8)
        m7_bias_dict = mi_bias_calc(sim_config_m7)

        mi_m8_corrected['mi'].append(mi_m8_raw - m8_bias_dict['mi'])
        mi_m8_corrected['nume'].append(nume8_raw - m8_bias_dict['nume'])
        mi_m8_corrected['deno'].append(deno8_raw - m8_bias_dict['deno'])

        mi_m7_corrected['mi'].append(mi_m7_raw - m7_bias_dict['mi'])
        mi_m7_corrected['nume'].append(nume7_raw - m7_bias_dict['nume'])
        mi_m7_corrected['deno'].append(deno7_raw - m7_bias_dict['deno'])


        # if bc_method not in mi_m8_corrected:
        #     mi_m8_corrected[bc_method] = [m8_logdet.item()]
        #     mi_m7_corrected[bc_method] = [m7_logdet.item()]
        # else:  
        #     mi_m8_corrected[bc_method].append(m8_logdet.item()) 
        #     mi_m7_corrected[bc_method].append(m7_logdet.item())

        #Save raw values as well.
        m8_dict_values['mi'].append(mi_m8_raw)
        m8_dict_values['nume'].append(nume8_raw)
        m8_dict_values['deno'].append(deno8_raw)

        m7_dict_values['mi'].append(mi_m7_raw)
        m7_dict_values['nume'].append(nume7_raw)
        m7_dict_values['deno'].append(deno7_raw)

    
    mi_m8_sample = torch.tensor(m8_dict_values['mi'])
    nume_m8_sample = torch.tensor(m8_dict_values['nume'])
    deno_m8_sample = torch.tensor(m8_dict_values['deno'])
    mi_m7_sample = torch.tensor(m7_dict_values['mi'])
    nume_m7_sample = torch.tensor(m7_dict_values['nume'])
    deno_m7_sample = torch.tensor(m7_dict_values['deno'])

    
    avg_m8_mi = torch.mean(mi_m8_sample)
    avg_m8_nume = torch.mean(nume_m8_sample)
    avg_m8_deno = torch.mean(deno_m8_sample)

    avg_m7_mi = torch.mean(mi_m7_sample)
    avg_m7_nume = torch.mean(nume_m7_sample)
    avg_m7_deno = torch.mean(deno_m7_sample)

    avg_corrected_m8_mi = torch.mean(torch.tensor(mi_m8_corrected['mi']))
    avg_corrected_m7_mi = torch.mean(torch.tensor(mi_m7_corrected['mi']))

    avg_corrected_m8_nume = torch.mean(torch.tensor(mi_m8_corrected['nume']))
    avg_corrected_m7_nume = torch.mean(torch.tensor(mi_m7_corrected['nume']))

    avg_corrected_m8_deno = torch.mean(torch.tensor(mi_m8_corrected['deno']))
    avg_corrected_m7_deno = torch.mean(torch.tensor(mi_m7_corrected['deno']))


    emp_bias_m8_mi = avg_m8_mi - m8_MI_true
    emp_bias_m8_nume = avg_m8_nume - nume_m8_true
    emp_bias_m8_deno = avg_m8_deno - deno_m8_true
    emp_bias_m7_mi = avg_m7_mi - m7_MI_true
    emp_bias_m7_nume = avg_m7_nume - nume_m7_true
    emp_bias_m7_deno = avg_m7_deno - deno_m7_true

    mi_m8_dict= {'sample': mi_m8_sample, 
              'avg': avg_m8_mi,
              'corrected_avg': avg_corrected_m8_mi,
              'std': torch.std(mi_m8_sample),
              'emp_bias': emp_bias_m8_mi,
              'ground_truth': m8_MI_true}
    nume_m8_dict = {'sample': nume_m8_sample,
                    'avg': avg_m8_nume,
                    'corrected_avg': avg_corrected_m8_nume,
                    'std': torch.std(nume_m8_sample),
                    'emp_bias': emp_bias_m8_nume,
                    'ground_truth': nume_m8_true}
    deno_m8_dict = {'sample': deno_m8_sample,
                    'avg': avg_m8_deno,
                    'corrected_avg': avg_corrected_m8_deno,
                    'std': torch.std(deno_m8_sample),
                    'emp_bias': emp_bias_m8_deno,
                    'ground_truth': deno_m8_true}
    mi_m7_dict= {'sample': mi_m7_sample, 
              'avg': avg_m7_mi  ,
                'corrected_avg': avg_corrected_m7_mi,
              'std': torch.std(mi_m7_sample),
              'emp_bias': emp_bias_m7_mi,
              'ground_truth': m7_MI_true}
    nume_m7_dict = {'sample': nume_m7_sample,
                    'avg': avg_m7_nume,
                    'corrected_avg': avg_corrected_m7_nume,
                    'std': torch.std(nume_m7_sample),
                    'emp_bias': emp_bias_m7_nume,
                    'ground_truth': nume_m7_true}
    deno_m7_dict = {'sample': deno_m7_sample,
                    'avg': avg_m7_deno,
                    'corrected_avg': avg_corrected_m7_deno,
                    'std': torch.std(deno_m7_sample),
                    'emp_bias': emp_bias_m7_deno,
                    'ground_truth': deno_m7_true}


    
    return {'M8_mi': mi_m8_dict, 'M8_nume': nume_m8_dict, 'M8_deno': deno_m8_dict,
            'M7_mi': mi_m7_dict, 'M7_nume': nume_m7_dict, 'M7_deno': deno_m7_dict}




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

    # M7 (Structural) Biases
    bias_02 = logdet_wishart_bias(df, n0 + n2) # Clique 0
    bias_12 = logdet_wishart_bias(df, n1 + n2) # Clique 1
    bias_2 = bias_y # seperator 2

    if m7 or m7_deno:


        # M7 MI Bias = 0.5 * ( (B_pred_struct - B_marginals) - (B_joint_struct - B_marginals) )
        # Note: b_joint_m7 = b_c0 + b_c1 - b_sep
        bias_m7_structural = bias_02 + bias_12 - bias_2
        bias_m7 = 0.5 * (bias_m7_structural - (bias_x0 + bias_x1 + bias_y))
        mi_bias_m7 = 0.5 * (bias_x0 + bias_x1 + 2*bias_2 - bias_02 - bias_12)
        return {'bias': mi_bias_m7 if m7 else bias_m7}
    
    if m7_nume:
        bias_m7_nume = 0.5 * (bias_02 + bias_12 - bias_2 - bias_x0 - bias_x1)
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
        mi_m7_results_list.append({'N': N, 'p': p, 'mean': res['M7_mi_mean'], 'std': res['M7_mi_std'], 'ground_truth': res['M7_mi_ground_truth'],'emp_bias': res['M7_mi_emp_bias']})
        nome_m7_results_list.append({'N': N, 'p': p, 'mean': res['M7_nume_mean'], 'std': res['M7_nume_std'], 'ground_truth': res['M7_nume_ground_truth'],'emp_bias': res['M7_nume_emp_bias']})
        deno_m7_results_list.append({'N': N, 'p': p, 'mean': res['M7_deno_mean'], 'std': res['M7_deno_std'], 'ground_truth': res['M7_deno_ground_truth'],'emp_bias': res['M7_deno_emp_bias']})

        mi_m8_results_list.append({'N': N, 'p': p, 'mean': res['M8_mi_mean'], 'std': res['M8_mi_std'], 'ground_truth': res['M8_mi_ground_truth'],'emp_bias': res['M8_mi_emp_bias']})
        nome_m8_results_list.append({'N': N, 'p': p, 'mean': res['M8_nume_mean'], 'std': res['M8_nume_std'], 'ground_truth': res['M8_nume_ground_truth'],'emp_bias': res['M8_nume_emp_bias']})
        deno_m8_results_list.append({'N': N, 'p': p, 'mean': res['M8_deno_mean'], 'std': res['M8_deno_std'], 'ground_truth': res['M8_deno_ground_truth'],'emp_bias': res['M8_deno_emp_bias']})

    return [mi_m7_results_list, nome_m7_results_list, deno_m7_results_list], [mi_m8_results_list, nome_m8_results_list, deno_m8_results_list]


def simulation_wrapper(config: dict) -> dict:
    """
    Run the logdet bias simulation for M7 and M8 models, returning a summary of results.
    """
    seed = config['seed']
    sim_func = simulate_m7_m8_mi

    #Set every bias correction function to it needs.
    m8_bias_func = partial(calculate_bias,m8=True)
    m7_bias_func = partial(calculate_bias,m7=True)
    m8_nume_fuc = partial(calculate_bias,m8_nume=True)
    m8_deno_func = partial(calculate_bias, m8_deno=True)
    m7_nume_func = partial(bias_resampling,calc_func = partial(para_calcualte_mi,term='nume'))
    m7_deno_func = partial(calculate_bias, m7_deno=True)

    bias_corr_func = {'M8': {'mi': m8_bias_func,'nume': m8_nume_fuc, 'deno': m8_deno_func}, 
                      'M7': {'mi': m7_bias_func, 'nume': m7_nume_func, 'deno': m7_deno_func}}
    
    corr_value_func  = corrected_statistic
    functions_dict = {'s_simulation': sim_func, 'bias_correction': bias_corr_func, 'corrected_statistic': corr_value_func}
    results_dict = simulation(config,functions_dict,seed=seed)
    return results_dict
    

if __name__ == "__main__":
    print("Running m7_whiten and M8 Simulation Mutual Information comparison simulation...")
    
    exp_name = 'MI>0_nume_bootstrap_debias'
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