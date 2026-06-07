import torch

import torch
import sys
from pathlib import Path
import yaml
import numpy as np
root = Path(__file__).resolve().parents[4]
sys.path.append(str(root))  

from Partial_Information_Decomposition.PID_calc import pid_calc
from Partial_Information_Decomposition.PID_util import create_cov_matrix,pid_comparison_table,save_pid_comparison_table
from Partial_Information_Decomposition.mi_functions import calculate_mi_raw
from Partial_Information_Decomposition.bias_functions import mi_wishahrt_bias




def true_mi_pid(sources, target, covariance=None):
    """Because we know Unique 2 is zero,
    We can know the true values of the PID components,
    As well because we know the it all gaussian we can calculate the true MI values as well.
    """
    x1, x2 = sources
    t = target[0]
    dims = [x1.shape[1], x2.shape[1], t.shape[1]]
    n_samples = x1.shape[0]

    cov_dict = create_cov_matrix(rvs=[x1, x2, t],dims=dims)
    cov = cov_dict['full_cov']

    calculated_mi = calculate_mi_raw(device=x1.device, sigma=cov, dims=dims)
    bias_dict = mi_wishahrt_bias(dims,n_samples)
    
    mi_tri = calculated_mi['tri_mi'] - bias_dict['bias_tri_mi']
    mi_bi_1 = calculated_mi['bi_mi_1_t'] - bias_dict['bias_mi_1_t']
    mi_bi_2 = calculated_mi['bi_mi_2_t'] - bias_dict['bias_mi_2_t']

    unq2 = 0
    red = mi_bi_2
    unq1 = mi_bi_1 - red
    syn = mi_tri - red - unq1 - unq2
    pid = {'red': red, 'unq1': unq1, 'unq2': unq2, 'syn': syn}
    mi = {'tri_mi': mi_tri, 'bi_mi_1': mi_bi_1, 'bi_mi_2': mi_bi_2}
    
    return pid, mi




def main_func(config,function_to_run):
    seed = config['seed']
    n = config['n']
    p = config['p']
    config['dx1'] = config['dx2'] = config['dt'] = p
    noise_std = config['noise_std']


    rng = np.random.default_rng(seed)
    x1, x2, t = function_to_run(rng, n, p, noise_std)

    sources = [torch.from_numpy(x1), torch.from_numpy(x2)]
    target = [torch.from_numpy(t)]
    
    true_values = true_mi_pid(sources, target, covariance=None)
    print(f"\nTrue PID values: {true_values[0]}")
    print(f"True MI values: {true_values[1]}")
    pid_tilde = pid_calc(config, sources, target, covariance=None, rng=rng, on_rvs=None,method='tilde')
    print(f"\nFinished calculating PID with tilde method")
    print(f"="*70)
    pid_delta = pid_calc(config, sources, target, covariance=None, rng=rng, on_rvs=None,method='delta')
    print(f"\nFinished calculating PID with delta method")
    print(f"="*70)

    cov_dict = create_cov_matrix(rvs=sources+[target[0]],dims=[s.shape[1] for s in sources]+[target[0].shape[1]])
    cov = cov_dict['full_cov']
    pid_flow = pid_calc(config, sources, target, covariance=cov, rng=rng, on_rvs=None,method='flow')
    print(f"\nFinished calculating PID with flow method")
    print(f"="*70)

    #Save into table
    results_dict = {'True Values':true_values
                    ,'Tilde':pid_tilde
                    ,'Delta':pid_delta
                    ,'Flow':pid_flow}

    return results_dict

