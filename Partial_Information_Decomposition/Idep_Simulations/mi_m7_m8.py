

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
    
    p_m8 = whiten_block(m8_true_cov_dict['cov_x0'], m8_true_cov_dict['cross_x0_x1'], m8_true_cov_dict['cov_x1']).numpy() #True covraince already whitened, so this is the P matrix for M8
    r_m8 = whiten_block(m8_true_cov_dict['cov_x1'], m8_true_cov_dict['cross_x1_x2'], m8_true_cov_dict['cov_x2']).numpy() #True covraince already whitened, so this is the R matrix for M8
    q_m8 = whiten_block(m8_true_cov_dict['cov_x0'], m8_true_cov_dict['cross_x0_x2'], m8_true_cov_dict['cov_x2']).numpy() #True covraince already whitened, so this is the Q matrix for M8
    nume_m8_true = safe_logdet(np.eye(n1) - (p_m8.T @ p_m8))
    deno_m8_true = safe_logdet(m8_true_cov)
    m8_MI_true = 0.5*(nume_m8_true-deno_m8_true)

    m7_true_cov_torch = torch.from_numpy(m7_true_cov).to(torch.float64)
    m7_true_cov_dict = create_cov_matrix(Sigma=m7_true_cov_torch, dims=[n0, n1, n2])

    p_m7 = whiten_block(m7_true_cov_dict['cov_x0'], m7_true_cov_dict['cross_x0_x1'], m7_true_cov_dict['cov_x1']).numpy() #True covraince already whitened, so this is the P matrix for M7
    r_m7 = whiten_block(m7_true_cov_dict['cov_x1'], m7_true_cov_dict['cross_x1_x2'], m7_true_cov_dict['cov_x2']).numpy() #True covraince already whitened, so this is the R matrix for M7
    q_m7 = whiten_block(m7_true_cov_dict['cov_x0'], m7_true_cov_dict['cross_x0_x2'], m7_true_cov_dict['cov_x2']).numpy() #True covraince already whitened, so this is the Q matrix for M7
    nume_m7_true = safe_logdet(np.eye(n1) - (p_m7.T @ p_m7))
    deno_m7_true = safe_logdet(m7_true_cov)

    m7_MI_true = 0.5*(nume_m7_true-deno_m7_true)



    m8_dict_values = {'mi':[],'nume':[],'deno':[]}
    m7_dict_values = {'mi':[],'nume':[],'deno':[]}

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

        m8_dict_values['mi'].append(m8_raw)
        m8_dict_values['nume'].append(nume8)
        m8_dict_values['deno'].append(deno8)

        m7_dict_values['mi'].append(m7_raw)
        m7_dict_values['nume'].append(nume7)
        m7_dict_values['deno'].append(deno7)

    
    mi_m8_sample = np.asarray(m8_dict_values['mi'])
    nume_m8_sample = np.asarray(m8_dict_values['nume'])
    deno_m8_sample = np.asarray(m8_dict_values['deno'])
    mi_m7_sample = np.asarray(m7_dict_values['mi'])
    nume_m7_sample = np.asarray(m7_dict_values['nume'])
    deno_m7_sample = np.asarray(m7_dict_values['deno'])

    
    avg_m8_mi = np.mean(mi_m8_sample)
    avg_m8_nume = np.mean(nume_m8_sample)
    avg_m8_deno = np.mean(deno_m8_sample)

    avg_m7_mi = np.mean(mi_m7_sample)
    avg_m7_nume = np.mean(nume_m7_sample)
    avg_m7_deno = np.mean(deno_m7_sample)


    emp_bias_m8_mi = avg_m8_mi - m8_MI_true
    emp_bias_m8_nume = avg_m8_nume - nume_m8_true
    emp_bias_m8_deno = avg_m8_deno - deno_m8_true
    emp_bias_m7_mi = avg_m7_mi - m7_MI_true
    emp_bias_m7_nume = avg_m7_nume - nume_m7_true
    emp_bias_m7_deno = avg_m7_deno - deno_m7_true

    mi_m8_dict= {'sample': mi_m8_sample, 
              'avg': avg_m8_mi,
              'std': np.std(mi_m8_sample),
              'emp_bias': emp_bias_m8_mi,
              'ground_truth': m8_MI_true}
    nume_m8_dict = {'sample': nume_m8_sample,
                    'avg': avg_m8_nume,
                    'std': np.std(nume_m8_sample),
                    'emp_bias': emp_bias_m8_nume,
                    'ground_truth': nume_m8_true}
    deno_m8_dict = {'sample': deno_m8_sample,
                    'avg': avg_m8_deno,
                    'std': np.std(deno_m8_sample),
                    'emp_bias': emp_bias_m8_deno,
                    'ground_truth': deno_m8_true}
    mi_m7_dict= {'sample': mi_m7_sample, 
              'avg': avg_m7_mi  ,
              'std': np.std(mi_m7_sample),
              'emp_bias': emp_bias_m7_mi,
              'ground_truth': m7_MI_true}
    nume_m7_dict = {'sample': nume_m7_sample,
                    'avg': avg_m7_nume,
                    'std': np.std(nume_m7_sample),
                    'emp_bias': emp_bias_m7_nume,
                    'ground_truth': nume_m7_true}
    deno_m7_dict = {'sample': deno_m7_sample,
                    'avg': avg_m7_deno,
                    'std': np.std(deno_m7_sample),
                    'emp_bias': emp_bias_m7_deno,
                    'ground_truth': deno_m7_true}


    
    return {'M8_mi': mi_m8_dict, 'M8_nume': nume_m8_dict, 'M8_deno': deno_m8_dict,
            'M7_mi': mi_m7_dict, 'M7_nume': nume_m7_dict, 'M7_deno': deno_m7_dict}




def calculate_bias(config: dict,m8:bool=False,m8_nume:bool=False,m8_deno:bool=False, m7:bool=False,m7_deno:bool=False, bias_correction:bool=True) -> dict:
   
    """
    Run the specified simulation function over combinations of N and p values, calculating mean and std of results.
    """
    if not bias_correction:
        return {'bias': 0.0} 
    
    assert m7 or m8 or m8_nume or m8_deno or m7_deno, "Must specify at least one of m7, m8, m8_nume, or m8_deno for bias calculation."
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
        bias_m7_structural = bias_m7_structural - (bias_x0 + bias_x1 + bias_y)
        bias_m7 = 0.5 * (bias_x0 + bias_x1 + 2*bias_2 - bias_02 - bias_12)
        return {'bias': bias_m7 if m7 else bias_m7_structural}
    
    # M8 (Saturated) Biases
    else:
        b_pred_m8 = logdet_wishart_bias(df, n0 + n1)
        b_joint_m8 = logdet_wishart_bias(df, d)
    
        nume_m8_bias = b_pred_m8 - (bias_x0 + bias_x1) 
        deno_m8_bias =b_joint_m8 - (bias_x0 + bias_x1 + bias_y)
        bias_m8_mi = 0.5*(nume_m8_bias - deno_m8_bias)

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

    #Set every bias correction function to it needs.
    m8_bias_func = partial(calculate_bias, m8=True)
    m7_bias_func = partial(calculate_bias, m7=True)
    m8_nume_fuc = partial(calculate_bias,m8_nume=True)
    m8_deno_func = partial(calculate_bias, m8_deno=True)
    m7_nume_func = partial(calculate_bias,bias_correction=False) #Assume no bias or numerator (NOT)
    m7_deno_func = partial(calculate_bias,m7_deno=True)

    bias_corr_func = {'M8_mi': m8_bias_func, 'M7_mi': m7_bias_func,
                      'M8_nume': m8_nume_fuc, 'M8_deno': m8_deno_func,
                      'M7_nume': m7_nume_func, 'M7_deno': m7_deno_func}
    
    corr_value_func  = corrected_statistic
    functions_dict = {'s_simulation': sim_func, 'bias_correction': bias_corr_func, 'corrected_statistic': corr_value_func}
    results_dict = simulation(config,functions_dict,seed=seed)
    return results_dict
    

if __name__ == "__main__":
    print("Running m7_whiten and M8 Simulation Mutual Information comparison simulation...")
    
    exp_name = 'MI_bias_corrected'
    yaml_file = f"/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/configs/sim.yaml"
    folder_path = f"/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/figures/MI_sim"
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