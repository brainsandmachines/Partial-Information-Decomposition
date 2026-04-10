

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
    deno8_true = 0.5 * safe_logdet(m8_true_cov)
    #Numerator
    joint_x0_x1 = m8_true_cov_dict['joint_x0_x1']
    cov_x2 = m8_true_cov_dict['cov_x2']
    nume8_joint_true = 0.5 * safe_logdet(joint_x0_x1)
    nume8_target_true = 0.5 * safe_logdet(cov_x2)
    nume8_true = nume8_joint_true + nume8_target_true
    mi_m8_true = nume8_true - deno8_true


    m7_true_cov_dict = create_cov_matrix(Sigma=m7_true_cov, dims=[n0, n1, n2])
    deno7_true = 0.5 * safe_logdet(m7_true_cov)
    nume7_joint_true = 0.5 * safe_logdet(m7_true_cov_dict['joint_x0_x1'])
    nume7_target_true = 0.5 * safe_logdet(m7_true_cov_dict['cov_x2'])
    nume7_true = nume7_joint_true + nume7_target_true
    mi_m7_true = nume7_true - deno7_true




    m8_dict_values = {'mi':[],'nume':[],'nume_joint':[],'nume_target':[],'deno':[]}
    m7_dict_values = {'mi':[],'nume':[],'nume_joint':[],'nume_target':[],'deno':[]}


    mi_m8_corrected = {'mi':[],'nume':[],'nume_joint':[],'nume_target':[],'deno':[]}
    mi_m7_corrected = {'mi':[],'nume':[],'nume_joint':[],'nume_target':[],'deno':[]}



    for i in range(n_trials):
        print(f"Trial {i+1}/{n_trials}", end="\r")
        # Build true covariance exactly the same way

        #Sample data and get sample covariance
        S,rv_list = sample_data_from_cov(config,true_cov=m8_true_cov,rng=rng)
        S_dict = create_cov_matrix(Sigma=S, dims=[n0, n1, n2],device=device)

        sim_config['Sigma'] = S.unsqueeze(0) #(1, d, d) for bias resampling
        sim_config['model'] = 'M8_M7'
        raw_results,sigmas = mi_calculation_not_whiten(config=sim_config)
        sigma_m8 = sigmas['M8']
        sigma_m7 = sigmas['M7']
        #M8
        mi_m8_raw = raw_results['M8']['mi']
        nume8_raw = raw_results['M8']['nume']
        nume_m8_joint_raw = raw_results['M8']['nume_joint']
        nume_m8_target_raw = raw_results['M8']['nume_target']
        deno8_raw = raw_results['M8']['deno']

        #M7
        mi_m7_raw = raw_results['M7']['mi']
        nume7_raw = raw_results['M7']['nume']
        nume_m7_joint_raw = raw_results['M7']['nume_joint']
        nume_m7_target_raw = raw_results['M7']['nume_target']
        deno7_raw = raw_results['M7']['deno']

        sim_config_m8 = sim_config.copy()
        sim_config_m7 = sim_config.copy()
        sim_config_m8['model'] = 'M8'
        sim_config_m7['model'] = 'M7'
        sim_config_m7['boots_xtra_bias'] = logdet_wishart_bias(df, n0 + n1) #We know the bias for the M7 numerator is the same as the bias of the M8 predictor clique, so we can correct it directly here.


        sim_config_m8['sample_statistic'] = {'mi': mi_m8_raw,'nume': nume8_raw,'nume_joint': nume_m8_joint_raw,'nume_target': nume_m8_target_raw, 'deno': deno8_raw}
        sim_config_m7['sample_statistic'] = {'mi': mi_m7_raw, 'nume': nume7_raw,'nume_joint': nume_m7_joint_raw,'nume_target': nume_m7_target_raw, 'deno': deno7_raw}

        sim_config_m8['calc_statistic_func'] = mi_calculation_not_whiten
        sim_config_m7['calc_statistic_func'] = mi_calculation_not_whiten

        sim_config_m8['rvs_list'] = rv_list
        sim_config_m7['rvs_list'] = rv_list

        sim_config_m8['Sigma'] = S
        sim_config_m7['Sigma'] = sigma_m7




        #bias_corr_func = sim_config['bias_correction_func']
        # m8_bias_corr = bias_corr_func['M8'](config=sim_config_m8)
        # m7_bias_corr = bias_corr_func['M7'](config=sim_config_m7)
        
        m8_bias_dict = mi_bias_calc(sim_config_m8)
        m7_bias_dict = mi_bias_calc(sim_config_m7)

        mi_m8_corrected['mi'].append(mi_m8_raw - m8_bias_dict['mi'])
        mi_m8_corrected['nume'].append(nume8_raw - m8_bias_dict['nume'])
        mi_m8_corrected['nume_joint'].append(nume_m8_joint_raw - m8_bias_dict['nume_joint'])
        mi_m8_corrected['nume_target'].append(nume_m8_target_raw - m8_bias_dict['nume_target'])
        mi_m8_corrected['deno'].append(deno8_raw - m8_bias_dict['deno'])

        mi_m7_corrected['mi'].append(mi_m7_raw - m7_bias_dict['mi'])
        mi_m7_corrected['nume'].append(nume7_raw - m7_bias_dict['nume'])
        mi_m7_corrected['nume_joint'].append(nume_m7_joint_raw - m7_bias_dict['nume_joint']) #We know the bias for the joint numerator is the same as the bias of the M8 predictor clique, so we can correct it directly here.
        mi_m7_corrected['nume_target'].append(nume_m7_target_raw - m7_bias_dict['nume_target'])
        mi_m7_corrected['deno'].append(deno7_raw - m7_bias_dict['deno'])



        #Save raw values as well.
        m8_dict_values['mi'].append(mi_m8_raw)
        m8_dict_values['nume'].append(nume8_raw)
        m8_dict_values['nume_joint'].append(nume_m8_joint_raw)
        m8_dict_values['nume_target'].append(nume_m8_target_raw)
        m8_dict_values['deno'].append(deno8_raw)

        m7_dict_values['mi'].append(mi_m7_raw)
        m7_dict_values['nume'].append(nume7_raw)
        m7_dict_values['nume_joint'].append(nume_m7_joint_raw)
        m7_dict_values['nume_target'].append(nume_m7_target_raw)
        m7_dict_values['deno'].append(deno7_raw)

    
    mi_m8_sample = torch.tensor(m8_dict_values['mi'])
    nume_m8_sample = torch.tensor(m8_dict_values['nume'])
    nume_m8_joint_sample = torch.tensor(m8_dict_values['nume_joint'])
    nume_m8_target_sample = torch.tensor(m8_dict_values['nume_target'])
    deno_m8_sample = torch.tensor(m8_dict_values['deno'])
    mi_m7_sample = torch.tensor(m7_dict_values['mi'])
    nume_m7_sample = torch.tensor(m7_dict_values['nume'])
    nume_m7_joint_sample = torch.tensor(m7_dict_values['nume_joint'])
    nume_m7_target_sample = torch.tensor(m7_dict_values['nume_target'])
    deno_m7_sample = torch.tensor(m7_dict_values['deno'])

    
    avg_m8_mi = torch.mean(mi_m8_sample)
    avg_m8_nume = torch.mean(nume_m8_sample)
    avg_m8_joint = torch.mean(nume_m8_joint_sample)
    avg_m8_target = torch.mean(nume_m8_target_sample)
    avg_m8_deno = torch.mean(deno_m8_sample)

    avg_m7_mi = torch.mean(mi_m7_sample)
    avg_m7_nume = torch.mean(nume_m7_sample)
    avg_m7_joint = torch.mean(nume_m7_joint_sample)
    avg_m7_target = torch.mean(nume_m7_target_sample)
    avg_m7_deno = torch.mean(deno_m7_sample)

    avg_corrected_m8_mi = torch.mean(torch.tensor(mi_m8_corrected['mi']))
    avg_corrected_m7_mi = torch.mean(torch.tensor(mi_m7_corrected['mi']))

    avg_corrected_m8_nume = torch.mean(torch.tensor(mi_m8_corrected['nume']))
    avg_corrected_m7_nume = torch.mean(torch.tensor(mi_m7_corrected['nume']))

    avg_corrected_m8_nume_joint = torch.mean(torch.tensor(mi_m8_corrected['nume_joint']))
    avg_corrected_m7_nume_joint = torch.mean(torch.tensor(mi_m7_corrected['nume_joint']))

    avg_corrected_m8_nume_target = torch.mean(torch.tensor(mi_m8_corrected['nume_target']))
    avg_corrected_m7_nume_target = torch.mean(torch.tensor(mi_m7_corrected['nume_target']))

    avg_corrected_m8_deno = torch.mean(torch.tensor(mi_m8_corrected['deno']))
    avg_corrected_m7_deno = torch.mean(torch.tensor(mi_m7_corrected['deno']))


    emp_bias_m8_mi = avg_m8_mi - mi_m8_true
    emp_bias_m8_nume = avg_m8_nume - nume8_true

    emp_bias_m8_joint = avg_m8_joint - nume8_joint_true
    emp_bias_m8_target = avg_m8_target - nume8_target_true
    emp_bias_m8_deno = avg_m8_deno - deno8_true

    emp_bias_m7_mi = avg_m7_mi - mi_m7_true
    emp_bias_m7_nume = avg_m7_nume - nume7_true
    emp_bias_m7_joint = avg_m7_joint - nume7_joint_true
    emp_bias_m7_target = avg_m7_target - nume7_target_true
    emp_bias_m7_deno = avg_m7_deno - deno7_true

    mi_m8_dict= {'sample': mi_m8_sample, 
              'avg': avg_m8_mi,
              'corrected_avg': avg_corrected_m8_mi,
              'std': torch.std(mi_m8_sample),
              'emp_bias': emp_bias_m8_mi,
              'ground_truth': mi_m8_true}
    nume_m8_dict = {'sample': nume_m8_sample,
                    'avg': avg_m8_nume,
                    'corrected_avg': avg_corrected_m8_nume,
                    'std': torch.std(nume_m8_sample),
                    'emp_bias': emp_bias_m8_nume,
                    'ground_truth': nume8_true}
    nume_joint_m8_dict = {'sample': nume_m8_joint_sample,
                    'avg': avg_m8_joint,
                    'corrected_avg': avg_corrected_m8_nume_joint,
                    'std': torch.std(nume_m8_joint_sample),
                    'emp_bias': emp_bias_m8_joint,
                    'ground_truth': nume8_joint_true}
    nume_target_m8_dict = {'sample': nume_m8_target_sample,
                    'avg': avg_m8_target,
                    'corrected_avg': avg_corrected_m8_nume_target,
                    'std': torch.std(nume_m8_target_sample),
                    'emp_bias': emp_bias_m8_target,
                    'ground_truth': nume8_target_true}
    deno_m8_dict = {'sample': deno_m8_sample,
                    'avg': avg_m8_deno,
                    'corrected_avg': avg_corrected_m8_deno,
                    'std': torch.std(deno_m8_sample),
                    'emp_bias': emp_bias_m8_deno,
                    'ground_truth': deno8_true}
    mi_m7_dict= {'sample': mi_m7_sample, 
              'avg': avg_m7_mi  ,
                'corrected_avg': avg_corrected_m7_mi,
              'std': torch.std(mi_m7_sample),
              'emp_bias': emp_bias_m7_mi,
              'ground_truth': mi_m7_true}
    nume_m7_dict = {'sample': nume_m7_sample,
                    'avg': avg_m7_nume,
                    'corrected_avg': avg_corrected_m7_nume,
                    'std': torch.std(nume_m7_sample),
                    'emp_bias': emp_bias_m7_nume,
                    'ground_truth': nume7_true}
    
    nume_joint_m7_dict = {'sample': nume_m7_joint_sample,
                    'avg': avg_m7_joint,
                    'corrected_avg': avg_corrected_m7_nume_joint,
                    'std': torch.std(nume_m7_joint_sample),
                    'emp_bias': emp_bias_m7_joint,
                    'ground_truth': nume7_joint_true}  
    nume_target_m7_dict = {'sample': nume_m7_target_sample,
                    'avg': avg_m7_target,
                    'corrected_avg': avg_corrected_m7_nume_target,
                    'std': torch.std(nume_m7_target_sample),
                    'emp_bias': emp_bias_m7_target,
                    'ground_truth': nume7_target_true}  
    deno_m7_dict = {'sample': deno_m7_sample,
                    'avg': avg_m7_deno,
                    'corrected_avg': avg_corrected_m7_deno,
                    'std': torch.std(deno_m7_sample),
                    'emp_bias': emp_bias_m7_deno,
                    'ground_truth': deno7_true}


    
    return {'M8_mi': mi_m8_dict, 'M8_nume': nume_m8_dict,'M8_joint': nume_joint_m8_dict, 'M8_target': nume_target_m8_dict, 'M8_deno': deno_m8_dict,
            'M7_mi': mi_m7_dict, 'M7_nume': nume_m7_dict, 'M7_joint': nume_joint_m7_dict, 'M7_target': nume_target_m7_dict, 'M7_deno': deno_m7_dict}




def calculate_bias(config: dict, mi = False,nume=False,nume_joint=False,nume_target=False,deno=False,bias_correction=True) -> dict:
   
    """
    Run the specified simulation function over combinations of N and p values, calculating mean and std of results.
    """
    if not bias_correction:
        return {'bias': 0.0} 
    
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
    bias_2 = bias_y # separator 2
    b_pred_m8 = logdet_wishart_bias(df, n0 + n1)
    b_joint_m8 = logdet_wishart_bias(df, d)
    
    

    if mi:
        bias_m8_mi = 0.5 * ((b_pred_m8+bias_2) - (b_joint_m8))
        bias_m7_mi = 0.5* ((bias_02 + bias_12 - bias_2))
        bias  = bias_m8_mi if config['model'] == 'M8' else bias_m7_mi
        st = 'mi'
    elif nume:
        bias_m8_nume = 0.5 * (b_pred_m8 +  bias_2)
        bias_m7_nume = 0.5 * (bias_02 + bias_12)
        bias = bias_m8_nume if config['model'] == 'M8' else bias_m7_nume
        st = 'nume'
    elif nume_joint:
        bias = 0.5 * (b_pred_m8)
        #bias = 0 #We don't have it right now
        st = 'nume_joint'
    elif nume_target:
        bias = 0.5 * (bias_2) #Bias is the same for M8 and M7
        st = 'nume_target'
    elif deno:
        bias_m8_deno = 0.5 * (b_joint_m8)
        bias_m7_deno = 0.5 * (bias_02 + bias_12 - bias_2) #Same as MI beacuse we dont know the analytical bias for nume7
        bias = bias_m8_deno if config['model'] == 'M8' else bias_m7_deno
        st = 'deno'
    return {st: bias} 
    
def sort_m7_m8_results(results_list):
    """ Helper: Sort results list by N and p values for  sperate by m7 and m8."""
    mi_m7_results_list = []
    nome_m7_results_list = []
    nome_joint_m7_results_list = []
    nome_target_m7_results_list = []
    deno_m7_results_list = []

    mi_m8_results_list = []
    nome_m8_results_list = []
    nome_joint_m8_results_list = []
    nome_target_m8_results_list = []
    deno_m8_results_list = []
    for res in results_list:
        N = res['N']
        p = res['p']
        mi_m7_results_list.append({'N': N, 'p': p, 'mean': res['M7_mi_mean'], 'std': res['M7_mi_std'], 'ground_truth': res['M7_mi_ground_truth']})
        nome_m7_results_list.append({'N': N, 'p': p, 'mean': res['M7_nume_mean'], 'std': res['M7_nume_std'], 'ground_truth': res['M7_nume_ground_truth']})
        nome_joint_m7_results_list.append({'N': N, 'p': p, 'mean': res['M7_joint_mean'], 'std': res['M7_joint_std'], 'ground_truth': res['M7_joint_ground_truth']})
        nome_target_m7_results_list.append({'N': N, 'p': p, 'mean': res['M7_target_mean'], 'std': res['M7_target_std'], 'ground_truth': res['M7_target_ground_truth']})
        deno_m7_results_list.append({'N': N, 'p': p, 'mean': res['M7_deno_mean'], 'std': res['M7_deno_std'], 'ground_truth': res['M7_deno_ground_truth']})
        
        mi_m8_results_list.append({'N': N, 'p': p, 'mean': res['M8_mi_mean'], 'std': res['M8_mi_std'], 'ground_truth': res['M8_mi_ground_truth']})
        nome_m8_results_list.append({'N': N, 'p': p, 'mean': res['M8_nume_mean'], 'std': res['M8_nume_std'], 'ground_truth': res['M8_nume_ground_truth']})
        nome_joint_m8_results_list.append({'N': N, 'p': p, 'mean': res['M8_joint_mean'], 'std': res['M8_joint_std'], 'ground_truth': res['M8_joint_ground_truth']})
        nome_target_m8_results_list.append({'N': N, 'p': p, 'mean': res['M8_target_mean'], 'std': res['M8_target_std'], 'ground_truth': res['M8_target_ground_truth']})
        deno_m8_results_list.append({'N': N, 'p': p, 'mean': res['M8_deno_mean'], 'std': res['M8_deno_std'], 'ground_truth': res['M8_deno_ground_truth']})

    return [mi_m7_results_list, nome_m7_results_list, nome_joint_m7_results_list, nome_target_m7_results_list, deno_m7_results_list], [mi_m8_results_list, nome_m8_results_list, nome_joint_m8_results_list, nome_target_m8_results_list, deno_m8_results_list]


def simulation_wrapper(config: dict) -> dict:
    """
    Run the logdet bias simulation for M7 and M8 models, returning a summary of results.
    """
    seed = config['seed']
    sim_func = simulate_m7_m8_mi

    #Set every bias correction function to it needs.
    m8_bias_func = partial(calculate_bias,mi=True)
    m7_bias_func = partial(calculate_bias,mi=True) #Assume no bias or numerator (NOT)
    m8_nume_fuc = partial(calculate_bias, nume=True)
    m8_nume_joint_func = partial(calculate_bias, nume_joint=True)
    m8_nume_target_func = partial(calculate_bias,nume_target=True)
    m8_deno_func = partial(calculate_bias, deno=True)
    m7_nume_func = partial(calculate_bias,nume=True) #Assume no bias or numerator (NOT)
    m7_nume_joint_func = partial(bias_resampling) 
    m7_nume_target_func = partial(calculate_bias,nume_target=True) 
    m7_deno_func = partial(calculate_bias,deno=True)

    bias_corr_func = {'M8': {'mi': m8_bias_func,'nume': m8_nume_fuc,'nume_joint': m8_nume_joint_func, 'nume_target': m8_nume_target_func, 'deno': m8_deno_func}, 
                      'M7': {'mi': m7_bias_func, 'nume': m7_nume_func, 'nume_joint': m7_nume_joint_func, 'nume_target': m7_nume_target_func, 'deno': m7_deno_func}}
    
    corr_value_func  = corrected_statistic
    functions_dict = {'s_simulation': sim_func, 'bias_correction': bias_corr_func, 'corrected_statistic': corr_value_func}
    results_dict = simulation(config,functions_dict,seed=seed)
    return results_dict
    

if __name__ == "__main__":
    print("Running m7_whiten and M8 Simulation Mutual Information comparison simulation...")

    exp_name = f"MI>0Mid_dim_smalltest_Sigmam7_NoWhitened"
    yaml_file = f"/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/configs/sim.yaml"
    folder_path = f"/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/figures/LogDet_NotWhitened/{exp_name}"
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

    mi_m7_result,nome_m7_list,nume_joint_m7_list,nume_target_m7_list,deno_m7_list =m7_results_list[0] ,m7_results_list[1], m7_results_list[2], m7_results_list[3], m7_results_list[4]
    mi_m8_result,nome_m8_list,nume_joint_m8_list,nume_target_m8_list,deno_m8_list =m8_results_list[0] ,m8_results_list[1], m8_results_list[2], m8_results_list[3], m8_results_list[4]

    #Plt Mutual Information results
    plot_heatmap_mean_std(mi_m7_result, title=f"Mutual Information M7 -{exp_name} - Mutual Information M7",save_path=save_path)
    plot_heatmap_mean_std(mi_m8_result, title=f"Mutual Information M8 -{exp_name} - Mutual Information M8",save_path=save_path)
    
    #Plot numerator
    plot_heatmap_mean_std(nome_m7_list, title=f"numerator M7 -{exp_name} - numerator M7",save_path=save_path)
    plot_heatmap_mean_std(nome_m8_list, title=f"numerator M8 -{exp_name} - numerator M8",save_path=save_path)

    #Plot numerator joint
    plot_heatmap_mean_std(nume_joint_m7_list, title=f"numerator joint M7 -{exp_name} - numerator joint M7",save_path=save_path)
    plot_heatmap_mean_std(nume_joint_m8_list, title=f"numerator joint M8 -{exp_name} - numerator joint M8",save_path=save_path)

    #Plot numerator target
    plot_heatmap_mean_std(nume_target_m7_list, title=f"numerator target M7 -{exp_name} - numerator target M7",save_path=save_path)
    plot_heatmap_mean_std(nume_target_m8_list, title=f"numerator target M8 -{exp_name} - numerator target M8",save_path=save_path)
    
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