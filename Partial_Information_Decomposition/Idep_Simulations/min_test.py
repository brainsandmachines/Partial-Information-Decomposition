import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from unique_m7_m8 import *
from Simulation_utils import *
from resampling_wrapper import bias_resampling
from bootstrap import _estimate_fitted_model_cov
from Partial_Information_Decomposition.Idep_Simulations.simulation_wrapper import *
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from PID_util import *



def find_minimum(config: dict, rng: torch.Generator | None = None) -> dict:
    """
    Run a single simulation for the M7_whiten and M8_Whiten models.

    Returns a dictionary with keys 'M7_whiten' and 'M8_Whiten', each containing
    the results for that model.
    """
    n0 = config['n0']
    n1 = config['n1']
    n2 = config['n2']

    true_covs = make_random_true_cov(config, n0, n1, n2, rng=rng)
    
    outputs = simulate_m7_m8_idep(true_covs,config,rng)

    #Check min for ground truth for unq1 and unq2

    #Extract Ground truth values
    min_dict_unq1 = {}
    min_dict_unq2 = {}
    #Unique 1
    i_true = outputs['i']['ground_truth']
    k_true = outputs['k']['ground_truth']
    atol = 1e-3
    #Unique 1 -> Torch.Tensor
    i_sample = outputs['i']['sample']
    k_sample = outputs['k']['sample']
    unq1_sample = torch.where(i_sample < k_sample, 1, -1)
    unq1_equality = torch.where(torch.isclose(i_sample, k_sample, atol=atol), 1, 0)
    if i_true < k_true:
        min_dict_unq1['i'] = i_true
        min_dict_unq1['N_mins'] = torch.sum(unq1_sample == 1).item()
        min_dict_unq1['N_mins'] += torch.sum(unq1_equality == 1).item() 
        print(f"Unique 1 minimum is i with {min_dict_unq1['N_mins']} out of {config['n_trials']} trials.")
        
    elif k_true <= i_true:
        min_dict_unq1['k'] = k_true
        min_dict_unq1['N_mins'] = torch.sum(unq1_sample == -1).item()
        min_dict_unq1['N_mins'] += torch.sum(unq1_equality == 1).item() 
        print(f"Unique 1 minimum is k with {min_dict_unq1['N_mins']} out of {config['n_trials']} trials.")

    #Unique 2
    j_true = outputs['j']['ground_truth']
    h_true = outputs['h']['ground_truth']
    #Unique 2 sample -> Torch.Tensor
    j_sample = outputs['j']['sample']
    h_sample = outputs['h']['sample']
    unq2_sample = torch.where(h_sample < j_sample, 1, -1)
    unq2_equality = torch.where(torch.isclose(h_sample, j_sample, atol=atol), 1, 0)
    if h_true <= j_true:
        min_dict_unq2['h'] = h_true
        min_dict_unq2['N_mins'] = torch.sum(unq2_sample == 1).item()
        min_dict_unq2['N_mins'] += torch.sum(unq2_equality == 1).item()
        print(f"Unique 2 minimum is h with {min_dict_unq2['N_mins']} out of {config['n_trials']} trials.")
    elif j_true <= h_true:
        min_dict_unq2['j'] = j_true
        min_dict_unq2['N_mins'] = torch.sum(unq2_sample == -1).item()
        min_dict_unq2['N_mins'] += torch.sum(unq2_equality == 1).item()
        print(f"Unique 2 minimum is j with {min_dict_unq2['N_mins']} out of {config['n_trials']} trials.")


    return {'unq1': min_dict_unq1, 'unq2': min_dict_unq2, 'full_outputs': outputs}


def plot_minimums(config,results_dict: dict,title: str,save_path: str | None = None):


    unq1_mins = results_dict['unq1']
    unq2_mins = results_dict['unq2']

    # Plotting Unique 1 minimums
    plt.figure(figsize=(12, 6))
    plt.bar(list(unq1_mins.keys())[0], unq1_mins['N_mins'])
    plt.bar(list(unq2_mins.keys())[0], unq2_mins['N_mins'])
    plt.axhline(config['n_trials'], color='black', linewidth=0.8, linestyle='--')
    plt.title(f'Unique 1 & 2 Minimums' + title)
    plt.ylabel('Number of Minimums')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{title}.png"))



if __name__ == "__main__":
    config = {
        'device': 'cuda',
        'seed':2,
        'n0': 1,
        'n1': 1,
        'n2': 1,
        'q_scale': 0.5,
        'r_scale': 0.5,
        'p_scale': 0.6,
        'n_samples': 1,
        'n_trials': 100,
        'analytic_bias_correction': False,
        'resample_bias_correction': False,
        'on_covariance': False,
    }

    
# for scale in [(0,0,0),(0.9,0,0),(0,0.9,0),(0,0,0.9),(0.35,0.25,15),(0.25,0.35,15),(0.25,15,0.35),(0,0.3,0.6),(0.3,0,0.6),(0.6,0.3,0)]:
#     config['q_scale'], config['r_scale'], config['p_scale'] = scale
#     for seed in range(1, 11):
        # config['seed'] = seed
    for n in range(13,100):
        config['n_samples'] = n
        rng = torch.Generator(device=config['device']).manual_seed(config['seed'])
        results_dict = find_minimum(config, rng)
        save_folder = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/figures/min_check/1dimRvs"
        save_path =  pathlib.Path(f"{save_folder}/seed_{config['seed']}")
        save_path.mkdir(parents=True, exist_ok=True)  
        title = f" - Simulation with n_samples={config['n_samples']}_Dim=[{config['n0']}, {config['n1']}, {config['n2']}] - p_scale, q_scale, r_scale = {config['p_scale']}, {config['q_scale']}, {config['r_scale']}"
        plot_minimums(config, results_dict, 
                    title=title, 
                    save_path=save_path)
        
