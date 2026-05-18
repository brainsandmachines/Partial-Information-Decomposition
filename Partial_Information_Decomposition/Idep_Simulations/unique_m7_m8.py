
import torch
import yaml
from functools import partial
from Simulation_utils import (
    N_P_variation_simulation,
    sample_data_from_cov,
    build_m8_terms,
    build_m7_terms,
    corrected_statistic,
    plot_heatmap_mean_std,
    save_nodes_results_csv,
    plot_pid_trajectory_vs_p_over_N,
    _build_pid_rows_from_node,
)
from Partial_Information_Decomposition.Idep_Simulations.simulation_wrapper import simulation
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from PID_util import create_cov_matrix
from Partial_Information_Decomposition.mi_functions import mi_wrapper,pid_components
from Partial_Information_Decomposition.bias_functions import unique_bias,logdet_wishart_bias,bias_func

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
    bias_correction = sim_config['bias_correction']
    
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
    true_unq1 = min(i_true, k_true)
    #Unique 2
    h_true = m7_MI_true - i_x1_t_true
    j_true = m8_MI_true - i_x1_t_true
    true_unq2 = min(h_true, j_true)
    true_pid_conffig = {'mi_tri': m8_MI_true, 'mi_bi_1': i_x1_t_true, 'mi_bi_2': i_x2_t_true,'unq1': true_unq1, 'unq2': true_unq2}
    _ = pid_components(true_pid_conffig, print_results=True)
    print(f"\nTrue i node value: {i_true:.6f}")
    print(f"True k node value: {k_true:.6f}")
    print(f"True h node value: {h_true:.6f}")
    print(f"True j node value: {j_true:.6f}")


    unq1_dict_values = {'i':[],'k':[]}
    unq2_dict_values = {'h':[],'j':[]}

    unq1_corrected = {'i':[],'k':[]}
    unq2_corrected = {'h':[],'j':[]}

    pid_config = sim_config.copy()

    mi_m7_samples = []
    mi_m8_samples = []
    mi_bi_1_samples = []
    mi_bi_2_samples = []


    for i in range(n_trials):
        print(f"Trial {i+1}/{n_trials}", end="\r")
        # Build true covariance exactly the same way

        # Sample data
        data_raw = sample_data_from_cov(sim_config,m8_true_cov,rng=rng) # (sample_cov,rv_list)
        inter_vars  = intermediate_func(sim_config,data_raw[0]) #Intermediate function can be used to apply shrinkage or other covariance transformations before calculating the sample covariance. It should return the transformed data and the corresponding RV list.
        Z = inter_vars.get('cov', data_raw[0]) #If the intermediate function does not return a new covariance, use the original one from data_raw.
        rv_list = inter_vars.get('rv_list', data_raw[1]) #If the intermediate function does not return a new rv_list, use the original one from data_raw.
        Z_dict = inter_vars.get('cov_dict', create_cov_matrix(Sigma=Z, dims=[n0, n1, n2]))

        Z = Z.squeeze(0) if Z.ndim == 3 and Z.shape[0] == 1 else Z

        Z_raw_dict = create_cov_matrix(Sigma=Z, dims=[n0, n1, n2])
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

        mi_m7_samples.append(mi_m7_raw)
        mi_m8_samples.append(mi_m8_raw)
        mi_bi_1_samples.append(i_x1_t_raw)
        mi_bi_2_samples.append(i_x2_t_raw)


        sim_config['M8_raw'] = mi_m8_dict
        sim_config['M7_raw'] = mi_m7_dict
        sim_config['X1'],sim_config['X2'],sim_config['T'] = rv_list
        sim_config['rng'] = rng
        if bias_correction:
            bias_dict = sim_config['bias_correction_func'](config=sim_config)
            i_bias,k_bias,j_bias,h_bias = bias_dict['i'],bias_dict['k'],bias_dict['j'],bias_dict['h']
        else:
            i_bias = k_bias = h_bias = j_bias = 0.0

        #Raw PID Values
        i_raw =  mi_m7_raw - i_x2_t_raw
        i_corr = i_raw - i_bias
        k_raw = mi_m8_raw - i_x2_t_raw
        k_corr = k_raw - k_bias

        h_raw = mi_m7_raw - i_x1_t_raw
        h_corr = h_raw - h_bias

        j_raw = mi_m8_raw - i_x1_t_raw
        j_corr = j_raw - j_bias



        unq1_corrected['i'].append(i_corr)
        unq1_corrected['k'].append(k_corr)
        
        unq2_corrected['h'].append(h_corr)
        unq2_corrected['j'].append(j_corr)

        i_corr_sample = torch.tensor(unq1_corrected['i'])
        k_corr_sample = torch.tensor(unq1_corrected['k'])
        h_corr_sample = torch.tensor(unq2_corrected['h'])
        j_corr_sample = torch.tensor(unq2_corrected['j'])



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

    i_corr_var = torch.var(i_corr_sample, correction=1)
    k_corr_var = torch.var(k_corr_sample, correction=1)
    h_corr_var = torch.var(h_corr_sample, correction=1)
    j_corr_var = torch.var(j_corr_sample, correction=1)

    i_corr_mse = torch.mean((i_corr_sample - i_true) ** 2)
    k_corr_mse = torch.mean((k_corr_sample - k_true) ** 2)
    h_corr_mse = torch.mean((h_corr_sample - h_true) ** 2)
    j_corr_mse = torch.mean((j_corr_sample - j_true) ** 2)

    bias_x0 = logdet_wishart_bias(df, n0)
    bias_x1 = logdet_wishart_bias(df, n1)
    bias_y  = logdet_wishart_bias(df, n2)
    tri_mi_bias = 0.5*(logdet_wishart_bias(df, n0 + n1) - (bias_x0 + bias_x1)) - 0.5*(logdet_wishart_bias(df, d)-(bias_x0 + bias_x1 + bias_y))
    bi_mi_bias_1 = 0.5*logdet_wishart_bias(df, n0) + 0.5*logdet_wishart_bias(df, n2) - 0.5*logdet_wishart_bias(df, n0+n2)
    bi_mi_bias_2 = 0.5*logdet_wishart_bias(df, n1) + 0.5*logdet_wishart_bias(df, n2) - 0.5*logdet_wishart_bias(df, n1+n2)
    avg_mi_m8 = torch.mean(torch.tensor(mi_m8_samples)-tri_mi_bias)
    avg_mi_bi_1 = torch.mean(torch.tensor(mi_bi_1_samples)-bi_mi_bias_1)
    avg_mi_bi_2 = torch.mean(torch.tensor(mi_bi_2_samples)-bi_mi_bias_2)

    i_dict= {'sample': i_sample, 
              'avg': avg_i_,
              'corrected_avg': avg_corrected_i,
              'std': torch.std(i_sample),
              'emp_bias': emp_bias_i,
              'after_corr_bias': after_corr_bias_i,
              'ground_truth': i_true,
              'var': i_corr_var,
                            'mse': i_corr_mse,
                            'mi_tri_avg': avg_mi_m8,
                            'mi_bi_1_avg': avg_mi_bi_1,
                            'mi_bi_2_avg': avg_mi_bi_2,
                            'mi_tri_ground_truth': m8_MI_true,
                            'mi_bi_1_ground_truth': i_x1_t_true,
                            'mi_bi_2_ground_truth': i_x2_t_true}
    k_dict = {'sample': k_sample,
                    'avg': avg_k_,
                    'corrected_avg': avg_corrected_k,
                    'std': torch.std(k_sample),
                    'emp_bias': emp_bias_k,
                    'after_corr_bias': after_corr_bias_k,
                    'ground_truth': k_true,
                    'var': k_corr_var,
                                        'mse': k_corr_mse,
                                        'mi_tri_avg': avg_mi_m8,
                                        'mi_bi_1_avg': avg_mi_bi_1,
                                        'mi_bi_2_avg': avg_mi_bi_2,
                                        'mi_tri_ground_truth': m8_MI_true,
                                        'mi_bi_1_ground_truth': i_x1_t_true,
                                        'mi_bi_2_ground_truth': i_x2_t_true}
    h_dict = {'sample': h_sample,
                    'avg': avg_h_,
                    'corrected_avg': avg_corrected_h,
                    'std': torch.std(h_sample),
                    'emp_bias': emp_bias_h,
                    'after_corr_bias': after_corr_bias_h,
                    'ground_truth': h_true,
                    'var': h_corr_var,
                                        'mse': h_corr_mse,
                                        'mi_tri_avg': avg_mi_m8,
                                        'mi_bi_1_avg': avg_mi_bi_1,
                                        'mi_bi_2_avg': avg_mi_bi_2,
                                        'mi_tri_ground_truth': m8_MI_true,
                                        'mi_bi_1_ground_truth': i_x1_t_true,
                                        'mi_bi_2_ground_truth': i_x2_t_true}
    j_dict= {'sample': j_sample, 
              'avg': avg_j_,
                'corrected_avg': avg_corrected_j,
              'std': torch.std(j_sample),
                'emp_bias': emp_bias_j,
                'after_corr_bias': after_corr_bias_j,
              'ground_truth': j_true,
              'var': j_corr_var,
                            'mse': j_corr_mse,
                            'mi_tri_avg': avg_mi_m8,
                            'mi_bi_1_avg': avg_mi_bi_1,
                            'mi_bi_2_avg': avg_mi_bi_2,
                            'mi_tri_ground_truth': m8_MI_true,
                            'mi_bi_1_ground_truth': i_x1_t_true,
                            'mi_bi_2_ground_truth': i_x2_t_true}


    
    return {'i': i_dict, 'k': k_dict, 'h': h_dict, 'j': j_dict}



    

def sort_m7_m8_results(results_list):
    """ Helper: Sort results list by N and p values for  sperate by m7 and m8."""
    i_results_list = []
    j_results_list = []
    k_results_list = []
    h_results_list = []

    for res in results_list:
        N = res['N']
        p = res['p']
        i_results_list.append({'N': N, 'p': p, 'mean': res['i_mean'], 'std': res['i_std'], 'ground_truth': res['i_ground_truth'],'emp_bias': res['i_emp_bias'],'after_corr_bias': res['i_after_corr_bias'],'var': res['i_var'],'mse': res['i_mse'], 'mi_tri': res.get('i_mi_tri_avg'), 'mi_bi_1': res.get('i_mi_bi_1_avg'), 'mi_bi_2': res.get('i_mi_bi_2_avg'), 'mi_tri_ground_truth': res.get('i_mi_tri_ground_truth'), 'mi_bi_1_ground_truth': res.get('i_mi_bi_1_ground_truth'), 'mi_bi_2_ground_truth': res.get('i_mi_bi_2_ground_truth')})
        j_results_list.append({'N': N, 'p': p, 'mean': res['j_mean'], 'std': res['j_std'], 'ground_truth': res['j_ground_truth'],'emp_bias': res['j_emp_bias'],'after_corr_bias': res['j_after_corr_bias'],'var': res['j_var'],'mse': res['j_mse'], 'mi_tri': res.get('j_mi_tri_avg'), 'mi_bi_1': res.get('j_mi_bi_1_avg'), 'mi_bi_2': res.get('j_mi_bi_2_avg'), 'mi_tri_ground_truth': res.get('j_mi_tri_ground_truth'), 'mi_bi_1_ground_truth': res.get('j_mi_bi_1_ground_truth'), 'mi_bi_2_ground_truth': res.get('j_mi_bi_2_ground_truth')})
        k_results_list.append({'N': N, 'p': p, 'mean': res['k_mean'], 'std': res['k_std'], 'ground_truth': res['k_ground_truth'],'emp_bias': res['k_emp_bias'],'after_corr_bias': res['k_after_corr_bias'],'var': res['k_var'],'mse': res['k_mse'], 'mi_tri': res.get('k_mi_tri_avg'), 'mi_bi_1': res.get('k_mi_bi_1_avg'), 'mi_bi_2': res.get('k_mi_bi_2_avg'), 'mi_tri_ground_truth': res.get('k_mi_tri_ground_truth'), 'mi_bi_1_ground_truth': res.get('k_mi_bi_1_ground_truth'), 'mi_bi_2_ground_truth': res.get('k_mi_bi_2_ground_truth')})
        h_results_list.append({'N': N, 'p': p, 'mean': res['h_mean'], 'std': res['h_std'], 'ground_truth': res['h_ground_truth'],'emp_bias': res['h_emp_bias'],'after_corr_bias': res['h_after_corr_bias'],'var': res['h_var'],'mse': res['h_mse'], 'mi_tri': res.get('h_mi_tri_avg'), 'mi_bi_1': res.get('h_mi_bi_1_avg'), 'mi_bi_2': res.get('h_mi_bi_2_avg'), 'mi_tri_ground_truth': res.get('h_mi_tri_ground_truth'), 'mi_bi_1_ground_truth': res.get('h_mi_bi_1_ground_truth'), 'mi_bi_2_ground_truth': res.get('h_mi_bi_2_ground_truth')})

    return [i_results_list, j_results_list, k_results_list, h_results_list]


def simulation_wrapper(config: dict) -> dict:
    """
    Run the logdet bias simulation for M7 and M8 models, returning a summary of results.
    """
    seed = config['seed']
    intermediate_func = config['intermediate_func']
    sim_func = partial(simulate_m7_m8_idep, intermediate_func=intermediate_func)

    bias_functions_dict = {
        'M7': partial(bias_func, model='M7'),
        'M8': partial(bias_func, model='M8')}
    
    corr_value_func  = corrected_statistic
    functions_dict = {'s_simulation': sim_func, 'bias_correction': partial(unique_bias, functions_dict=bias_functions_dict),
                      'corrected_statistic': corr_value_func}
    results_dict = simulation(config,functions_dict,seed=seed)
    return results_dict


def _save_config_yaml(config, save_path, exp_name):
    with open(f'{save_path}/{exp_name}_config.yaml', 'w') as f:
        yaml_config = {key: value for key, value in config.items() if not callable(value)}
        yaml.safe_dump(yaml_config, f, sort_keys=False, allow_unicode=True)


def _render_heatmaps(i_result, j_result, k_result, h_result, save_path, exp_name):
    plot_heatmap_mean_std(i_result, title=f"Unique-1-i-node-{exp_name}", save_path=save_path)
    plot_heatmap_mean_std(k_result, title=f"Unique-1-k-node-{exp_name}", save_path=save_path)
    plot_heatmap_mean_std(j_result, title=f"Unique-2-j-node-{exp_name}", save_path=save_path)
    plot_heatmap_mean_std(h_result, title=f"Unique-2-h-node-{exp_name}", save_path=save_path)


def _render_pid_trajectories(config, i_result, j_result, k_result, h_result, save_path, exp_name):
    single_p = len(config.get('p_values', [])) == 1
    multi_n = len(config.get('N_values', [])) > 1
    if not (single_p and multi_n):
        print("Skipping PID trajectory plots: requires exactly one p value and multiple N values.")
        return

    plot_specs = [
        (i_result, 'unq1', f"PID components vs p/N - Unique-1-i-node-{exp_name}"),
        (k_result, 'unq1', f"PID components vs p/N - Unique-1-k-node-{exp_name}"),
        (j_result, 'unq2', f"PID components vs p/N - Unique-2-j-node-{exp_name}"),
        (h_result, 'unq2', f"PID components vs p/N - Unique-2-h-node-{exp_name}"),
    ]

    for node_rows, known_component, title in plot_specs:
        pid_rows, pid_ground_truth = _build_pid_rows_from_node(node_rows, known_component)
        if not pid_rows:
            print(f"Skipping PID trajectory plot for {title}: missing MI terms.")
            continue

        plot_pid_trajectory_vs_p_over_N(
            results=pid_rows,
            ground_truth=pid_ground_truth,
            save_path=save_path,
            title=title,
        )

def run(main_func,exp_name, config,save_path,plot_heatmaps:bool=True,plot_graphs:bool=False):
        config['simulation_func'] = main_func
        results = N_P_variation_simulation(config)
        nodes_results_list = sort_m7_m8_results(results)

        i_result,j_result,k_result,h_result =nodes_results_list[0] ,nodes_results_list[1], nodes_results_list[2], nodes_results_list[3]
        save_nodes_results_csv(i_result,j_result,k_result,h_result,save_path)
        if plot_heatmaps:
            _render_heatmaps(i_result, j_result, k_result, h_result, save_path, exp_name)
            _save_config_yaml(config, save_path, exp_name)

        if plot_graphs:
            _render_pid_trajectories(config, i_result, j_result, k_result, h_result, save_path, exp_name)

        print("\nFinished simulation.")

        return nodes_results_list
    

