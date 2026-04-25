

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
from Partial_Information_Decomposition.mi_functions import mi_calculation_not_whiten,safe_logdet,logdet_wishart_bias,mi_wrapper


def simulate_m7_m8_idep(
    data: list,
    sim_config: dict,
    rng: torch.Generator | None = None,
    intermediate_func: callable = None,):
    """
    Run MI simulation under the same covariance construction used
    in the logdet experiments.

    Inputs: 
        data - True Covriances for M7 and M8 models, in the form of a list [m8_cov, m7_cov]
        sim_config - Simulation configuration dictionary containing parameters for the simulation.
        rng - Optional random number generator for reproducibility.
        intermediate_func - Optional function to apply transformations to the sampled data before calculating covariances.
        mi_func - Mutual information calculation function to use for the simulation. Must accept sim_config and
    """
    n_samples = sim_config['n_samples']
    n0 = sim_config['n0']
    n1 = sim_config['n1']
    n2 = sim_config['n2']
    n_trials = sim_config['n_trials']
    device = sim_config['device']
    analytic_bias_correction = sim_config['analytic_bias_correction']
    resample_bias_correction = sim_config['resample_bias_correction']
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
    m7_true_cov_dict = create_cov_matrix(Sigma=m7_true_cov, dims=[n0, n1, n2])


    m8 = build_m8_terms(sim_config, m8_true_cov_dict, whiten=sim_config['normalization'])
    m7 = build_m7_terms(sim_config, m7_true_cov_dict, whiten=sim_config['normalization'])

    mi_m8 = mi_wrapper(sim_config,m8_true_cov_dict,m8)
    mi_m7 = mi_wrapper(sim_config,m7_true_cov_dict,m7)

    m8_MI_true = mi_m8['mi_tri']
    m7_MI_true = mi_m7['mi_tri']

    #Calculate bi_variate MIs for bias calculations
    mi_bi_true = mi_wrapper(sim_config,m8_true_cov_dict,m8,tri_variate=False) #Calculated differently
    i_x1_t_true = mi_bi_true['mi_bi_1'] 
    i_x2_t_true = mi_bi_true['mi_bi_2'] 
    #Unique 1
    i_true = m7_MI_true - i_x2_t_true
    k_true = m8_MI_true - i_x2_t_true
    
    #Unique 2
    h_true = m7_MI_true - i_x1_t_true
    j_true = m8_MI_true - i_x1_t_true


    unq1_dict_values = {'i':[],'k':[]}
    unq2_dict_values = {'h':[],'j':[]}

    unq1_corrected = {'i':[],'k':[]}
    unq2_corrected = {'h':[],'j':[]}

    pid_config = sim_config.copy()

    if analytic_bias_correction:
        bias_calc_func = sim_config['bias_correction_func']
        m7_analytic_bias = calculate_bias(sim_config, m7=True, bias_correction=True)['bias']
        m8_analytic_bias = calculate_bias(sim_config, m8=True, bias_correction=True)['bias']
        m1_t_analytic_bias = 0.5*logdet_wishart_bias(df, n0) + 0.5*logdet_wishart_bias(df, n2) - 0.5*logdet_wishart_bias(df, n0+n2)
        m2_t_analytic_bias = 0.5*logdet_wishart_bias(df, n1) + 0.5*logdet_wishart_bias(df, n2) - 0.5*logdet_wishart_bias(df, n1+n2)

    for i in range(n_trials):
        print(f"Trial {i+1}/{n_trials}", end="\r")
        # Build true covariance exactly the same way

        # Sample data
        data_raw = sample_data_from_cov(sim_config,m8_true_cov,rng=rng) # (sample_cov,rv_list)
        inter_vars  = intermediate_func(sim_config,data_raw) #Intermediate function can be used to apply shrinkage or other covariance transformations before calculating the sample covariance. It should return the transformed data and the corresponding RV list.
        Z = inter_vars.get('cov', data_raw[0]) #If the intermediate function does not return a new covariance, use the original one from data_raw.
        rv_list = inter_vars.get('rv_list', data_raw[1]) #If the intermediate function does not return a new rv_list, use the original one from data_raw.
        Z_dict = inter_vars.get('cov_dict', create_cov_matrix(Sigma=Z, dims=[n0, n1, n2]))

        Z = Z.squeeze(0) if Z.ndim == 3 and Z.shape[0] == 1 else Z

        Z_raw_dict = create_cov_matrix(Sigma=data_raw[0], dims=[n0, n1, n2])
        #Graph Model M8 
        m8_sample = build_m8_terms(sim_config, Z_dict, whiten='whiten_ver') 
        
        mi_m8_dict = mi_wrapper(sim_config,Z_dict,m8_sample)
        mi_m8_raw = mi_m8_dict['mi_tri']
 
        

        #Graph Model M7 denominator
        m7_sample = build_m7_terms(sim_config, Z_dict, whiten='whiten_ver')
        mi_m7_dict = mi_wrapper(sim_config,Z_dict,m7_sample)
        mi_m7_raw = mi_m7_dict['mi_tri']

        #Calculate bi-variate MIs 
        mi_bi_dict = mi_wrapper(sim_config,Z_raw_dict,m8_sample,tri_variate=False) #Calculated differently
        i_x1_t_raw = mi_bi_dict['mi_bi_1']
        i_x2_t_raw = mi_bi_dict['mi_bi_2']

        
        if analytic_bias_correction:
            i_bias = (m7_analytic_bias - m2_t_analytic_bias)
            k_bias = (m8_analytic_bias - m2_t_analytic_bias)
            h_bias = (m7_analytic_bias - m1_t_analytic_bias)
            j_bias = (m8_analytic_bias - m1_t_analytic_bias)
        
        else:
            i_bias = k_bias = h_bias = j_bias = 0.0

        pid_config['analytic_bias'] = {'i': i_bias, 'k': k_bias,
                                            'h': h_bias, 'j': j_bias}
        #Raw PID Values
        i_raw =  mi_m7_raw - i_x2_t_raw
        i_corr = i_raw - i_bias
        k_raw = mi_m8_raw - i_x2_t_raw
        k_corr = k_raw - k_bias

        h_raw = mi_m7_raw - i_x1_t_raw
        h_corr = h_raw - h_bias

        j_raw = mi_m8_raw - i_x1_t_raw
        j_corr = j_raw - j_bias




        pid_config['sample_statistic'] = {'i': i_raw, 'k': k_raw,'h': h_raw, 'j': j_raw}

        pid_config['rvs_list'] = rv_list
        pid_config['Sigma'] = Z

        pid_config['model'] = 'pid'
        
        if resample_bias_correction:
            pid_bias_dict = bias_calc_func(pid_config)

        else:
            pid_bias_dict = {'i': 0.0, 'k': 0.0, 'h': 0.0, 'j': 0.0}

        unq1_corrected['i'].append(i_corr - pid_bias_dict['i'])
        unq1_corrected['k'].append(k_corr - pid_bias_dict['k'])
        
        unq2_corrected['h'].append(h_corr - pid_bias_dict['h'])
        unq2_corrected['j'].append(j_corr - pid_bias_dict['j'])



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

    emp_bias_i = avg_i_ - i_true
    emp_bias_k = avg_k_ - k_true
    emp_bias_h = avg_h_ - h_true
    emp_bias_j = avg_j_ - j_true

    after_corr_bias_i = avg_corrected_i - i_true
    after_corr_bias_k = avg_corrected_k - k_true
    after_corr_bias_h = avg_corrected_h - h_true
    after_corr_bias_j = avg_corrected_j - j_true

    i_dict= {'sample': i_sample, 
              'avg': avg_i_,
              'corrected_avg': avg_corrected_i,
              'std': torch.std(i_sample),
              'emp_bias': emp_bias_i,
              'after_corr_bias': after_corr_bias_i,
              'ground_truth': i_true}
    k_dict = {'sample': k_sample,
                    'avg': avg_k_,
                    'corrected_avg': avg_corrected_k,
                    'std': torch.std(k_sample),
                    'emp_bias': emp_bias_k,
                    'after_corr_bias': after_corr_bias_k,
                    'ground_truth': k_true}
    h_dict = {'sample': h_sample,
                    'avg': avg_h_,
                    'corrected_avg': avg_corrected_h,
                    'std': torch.std(h_sample),
                    'emp_bias': emp_bias_h,
                    'after_corr_bias': after_corr_bias_h,
                    'ground_truth': h_true}
    j_dict= {'sample': j_sample, 
              'avg': avg_j_,
                'corrected_avg': avg_corrected_j,
              'std': torch.std(j_sample),
                'emp_bias': emp_bias_j,
                'after_corr_bias': after_corr_bias_j,
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
        pass
        
    
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
    i_results_list = []
    j_results_list = []
    k_results_list = []
    h_results_list = []

    for res in results_list:
        N = res['N']
        p = res['p']
        i_results_list.append({'N': N, 'p': p, 'mean': res['i_mean'], 'std': res['i_std'], 'ground_truth': res['i_ground_truth'],'emp_bias': res['i_emp_bias'],'after_corr_bias': res['i_after_corr_bias']})
        j_results_list.append({'N': N, 'p': p, 'mean': res['j_mean'], 'std': res['j_std'], 'ground_truth': res['j_ground_truth'],'emp_bias': res['j_emp_bias'],'after_corr_bias': res['j_after_corr_bias']})
        k_results_list.append({'N': N, 'p': p, 'mean': res['k_mean'], 'std': res['k_std'], 'ground_truth': res['k_ground_truth'],'emp_bias': res['k_emp_bias'],'after_corr_bias': res['k_after_corr_bias']})
        h_results_list.append({'N': N, 'p': p, 'mean': res['h_mean'], 'std': res['h_std'], 'ground_truth': res['h_ground_truth'],'emp_bias': res['h_emp_bias'],'after_corr_bias': res['h_after_corr_bias']})

    return [i_results_list, j_results_list, k_results_list, h_results_list]


def simulation_wrapper(config: dict) -> dict:
    """
    Run the logdet bias simulation for M7 and M8 models, returning a summary of results.
    """
    seed = config['seed']
    intermediate_func = config['intermediate_func']
    sim_func = partial(simulate_m7_m8_idep, intermediate_func=intermediate_func)


    pid_bootstrap_func = partial(bias_resampling,calc_func=para_unique_bias_calc)
    
    corr_value_func  = corrected_statistic
    functions_dict = {'s_simulation': sim_func, 'bias_correction': pid_bootstrap_func, 
                      'corrected_statistic': corr_value_func}
    results_dict = simulation(config,functions_dict,seed=seed)
    return results_dict

def run(main_func,exp_name, config,save_path,plot_heatmaps:bool=True):
        config['simulation_func'] = main_func
        results = N_P_variation_simulation(config)
        nodes_results_list = sort_m7_m8_results(results)

        i_result,j_result,k_result,h_result =nodes_results_list[0] ,nodes_results_list[1], nodes_results_list[2], nodes_results_list[3]
        if plot_heatmaps:
            #Plot Unique 1 results  
            plot_heatmap_mean_std(i_result, title=f"Unique-1-i-node-{exp_name}",save_path=save_path)
            plot_heatmap_mean_std(k_result, title=f"Unique-1-k-node-{exp_name}",save_path=save_path)

            #Plot Unique 2 results
            plot_heatmap_mean_std(j_result, title=f"Unique-2-j-node-{exp_name}",save_path=save_path)
            plot_heatmap_mean_std(h_result, title=f"Unique-2-h-node-{exp_name}",save_path=save_path)
            #Save config for file
            with open(f'{save_path}/{exp_name}_config.yaml', 'w') as f:
                yaml_config = {key: value for key, value in config.items() if not callable(value)}
                
                yaml.safe_dump(yaml_config, f, sort_keys=False, allow_unicode=True)
        print("\nFinished simulation.")

        return nodes_results_list
    

