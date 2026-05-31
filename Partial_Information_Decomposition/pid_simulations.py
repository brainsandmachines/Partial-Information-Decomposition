import torch 
from pathlib import Path
import sys
import os
import yaml
import pandas as pd
from heatmap_plot import plot_pid_and_mi_heatmaps_from_csv
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from PID_calc import pid_calc
from Idep_Simulations.simulation_wrapper import make_random_true_cov
from Idep_Simulations.Simulation_utils import already_exists_in_csv, make_pre_config, sample_data_from_cov, flatten_pid_results,append_row_to_csv
from utils import Tee
from mi_functions import calculate_mi_raw
from PID_util import reorder_cov_blocks

log = open("Simulation_DeltaVsIIdepVsBroja.log", "w")

sys.stdout = Tee(sys.stdout, log)
sys.stderr = Tee(sys.stderr, log)

def pid_simulation(config,rng,cov,pid_ver,true_values = None):
    """Run PID simulation with know ground truth PID values from the covariance matrix
    sample for the true covariance matrix, calculate the PID using the specified method, and return the results along with the ground truth PID values calculated from the covariance matrix.
    """

    if true_values is None:
        pid_true,mi_true = pid_calc(config=config, covariance=cov, rng=rng, method=pid_ver)

    else: 
        pid_true = true_values['pid']
        mi_true = true_values['mi']



    n_trials = config['n_trials']
    unq1_sample, unq2_sample, syn_sample, red_sample = [], [], [], []
    tri_mi_sample, bi_mi_1_sample, bi_mi_2_sample = [], [], []

    for i in range(n_trials):
        if (i+1) % max(1, n_trials//10) == 0:
            print(f"Trial {i+1}/{n_trials}", end="\r")
        # Build true covariance exactly the same way

        # Sample data
        data_raw = sample_data_from_cov(config,cov,rng=rng) # (sample_cov,rv_list)
        sample_cov, rv_raw = data_raw

        pid,mi = pid_calc(config,sources=rv_raw[:2],target=rv_raw[2:],rng=rng,method=pid_ver) #Calculate PID for the sampled data

        unq1_sample.append(pid['unq1'])
        unq2_sample.append(pid['unq2'])
        syn_sample.append(pid['syn'])
        red_sample.append(pid['red'])

        tri_mi_sample.append(mi['tri_mi'])
        bi_mi_1_sample.append(mi['bi_mi_1'])
        bi_mi_2_sample.append(mi['bi_mi_2'])

    avg_unq1 = torch.mean(torch.tensor(unq1_sample))
    avg_unq2 = torch.mean(torch.tensor(unq2_sample))
    avg_syn = torch.mean(torch.tensor(syn_sample))
    avg_red = torch.mean(torch.tensor(red_sample))
    avg_tri_mi = torch.mean(torch.tensor(tri_mi_sample))
    avg_bi_mi_1 = torch.mean(torch.tensor(bi_mi_1_sample))
    avg_bi_mi_2 = torch.mean(torch.tensor(bi_mi_2_sample))

    var_unq1 = torch.var(torch.tensor(unq1_sample))
    var_unq2 = torch.var(torch.tensor(unq2_sample))
    var_syn = torch.var(torch.tensor(syn_sample))
    var_red = torch.var(torch.tensor(red_sample))
    var_tri_mi = torch.var(torch.tensor(tri_mi_sample))
    var_bi_mi_1 = torch.var(torch.tensor(bi_mi_1_sample))
    var_bi_mi_2 = torch.var(torch.tensor(bi_mi_2_sample))

    unq1_emp_bias = avg_unq1 - pid_true['unq1']
    unq2_emp_bias = avg_unq2 - pid_true['unq2']
    syn_emp_bias = avg_syn - pid_true['syn']
    red_emp_bias = avg_red - pid_true['red']

    tri_mi_emp_bias = avg_tri_mi - mi_true['tri_mi']
    bi_mi_1_emp_bias = avg_bi_mi_1 - mi_true['bi_mi_1']
    bi_mi_2_emp_bias = avg_bi_mi_2 - mi_true['bi_mi_2']

    mse_unq1 = (unq1_emp_bias)**2 + var_unq1
    mse_unq2 = (unq2_emp_bias)**2 + var_unq2
    mse_syn = (syn_emp_bias)**2 + var_syn
    mse_red = (red_emp_bias)**2 + var_red
    mse_tri_mi = (tri_mi_emp_bias)**2 + var_tri_mi
    mse_bi_mi_1 = (bi_mi_1_emp_bias)**2 + var_bi_mi_1
    mse_bi_mi_2 = (bi_mi_2_emp_bias)**2 + var_bi_mi_2

    return {
        'unq1': {'mean': avg_unq1, 'ground_truth': pid_true['unq1'], 'emp_bias': unq1_emp_bias, 'var': var_unq1, 'mse': mse_unq1},
        'unq2': {'mean': avg_unq2, 'ground_truth': pid_true['unq2'], 'emp_bias': unq2_emp_bias, 'var': var_unq2, 'mse': mse_unq2},
        'syn': {'mean': avg_syn, 'ground_truth': pid_true['syn'], 'emp_bias': syn_emp_bias, 'var': var_syn, 'mse': mse_syn},
        'red': {'mean': avg_red, 'ground_truth': pid_true['red'], 'emp_bias': red_emp_bias, 'var': var_red, 'mse': mse_red},
        'mi_tri': {'mean': avg_tri_mi, 'ground_truth': mi_true['tri_mi'], 'emp_bias': tri_mi_emp_bias, 'var': var_tri_mi, 'mse': mse_tri_mi},
        'mi_bi_1': {'mean': avg_bi_mi_1, 'ground_truth': mi_true['bi_mi_1'], 'emp_bias': bi_mi_1_emp_bias, 'var': var_bi_mi_1, 'mse': mse_bi_mi_1},
        'mi_bi_2': {'mean': avg_bi_mi_2, 'ground_truth': mi_true['bi_mi_2'], 'emp_bias': bi_mi_2_emp_bias, 'var': var_bi_mi_2, 'mse': mse_bi_mi_2},
    }


def trials_simulation(config,title):
    pid_ver = config['pid_ver']
    rng = torch.Generator().manual_seed(config['seed'])
    print("Running PID simulations with PID version:", pid_ver)


    N_values = config['N_values']
    P_values = config['P_values']
    len_N = len(N_values)
    len_P = len(P_values)
    i = 1
    rows = []
    output_csv = config.get("output_csv", "pid_results.csv")

    for N in N_values:
        for p in P_values:
            print(f"\nRunning simulation for N={N}, p={p} ({i}/{len_N*len_P})")
            config['n_samples'] = N
            config['dx1'] = p[0]
            config['dx2'] = p[1]
            config['dt'] = p[2]
            m8_true_cov, m7_true_cov = make_random_true_cov(config,rng=rng)
            #Calculate MI outside of PID definition: 
            true_mi_dict = calculate_mi_raw(device=config['device'],sigma=m8_true_cov,dims=[config['dx1'],config['dx2'],config['dt']])
            true_pid_dict = {'unq1' : true_mi_dict['bi_mi_1'], 'unq2' : true_mi_dict['bi_mi_2'], 'syn' : true_mi_dict['tri_mi'] - true_mi_dict['bi_mi_1'] - true_mi_dict['bi_mi_2'], 'red' : 0.0} #For Gaussian variables, the redundancy is 0, because the unique information is defined as the difference between the mutual information of each source with the target and the mutual information of both sources together with the target, and for Gaussian variables, the mutual information of both sources together with the target is equal to the sum of the mutual information of each source with the target.
            true_values = {'pid': true_pid_dict, 'mi': true_mi_dict}
            
            for pid_ver in config['pid_ver']:

                if already_exists_in_csv(output_csv, N, p, pid_ver, config["seed"],csv_title=title):
                    print(
                        f"Already exists in CSV, skipping: "
                        f"N={N}, p={p}, pid_ver={pid_ver}"
                    )
                    continue
                print(f"\nRunning PID simulation for {pid_ver}...")
                pid_results = pid_simulation(config=config,rng=rng, cov=m8_true_cov, pid_ver=pid_ver,true_values=true_values)


                row = {
                                "N": N,
                                "dx1": p[0],
                                "dx2": p[1],
                                "dt": p[2],
                                "pid_ver": pid_ver,
                                "seed": config["seed"],
                            }

                row.update(flatten_pid_results(pid_results))
                append_row_to_csv(row,output_folder=output_csv,csv_title=title)

            
                df = pd.DataFrame(rows)


    print(f"\nSaved PID results to: {output_csv}")

    return output_csv



def main(config,single=True,multi=False,exp_name=None):
    pid_ver = config['pid_ver']
    rng = torch.Generator().manual_seed(config['seed'])
    print("Running PID simulations with PID version:", pid_ver)


    if single:
        m8_true_cov, m7_true_cov = make_random_true_cov(config,rng=rng)
        
        for pid_ver in config['pid_ver']:
            #print(f"\nRunning PID simulation for {pid_ver}...")
            pid_results = pid_simulation(config,rng, m8_true_cov,pid_ver)
            #print(f"Finished PID simulation for {pid_ver}.")

    elif multi:
        output_csv = trials_simulation(config,title=exp_name)
        print("Finished all PID simulations.")
        return output_csv



if __name__ == "__main__":
    config_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/configs/pid_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    exp_name = ['unknown']


    for exp in exp_name:
        pre_config = make_pre_config(exp, config["Mutual_Information_Simulation"], config["0=Mutual_Information_Simulation"], config["M7_Mutual_Information_Simulation"], config["M8_Mutual_Information_Simulation"], config["N_P_variations"], config["Unknown_Mutual_Information_Simulation"], config.get("DE_config", {}))
        output = main(pre_config,single=False,multi=True,exp_name='2.0NoRedundancy')
    
        for pid_ver in pre_config['pid_ver']:
            
            figures = plot_pid_and_mi_heatmaps_from_csv(csv_path = f'{pre_config["output_csv"]}/{'2.0NoRedundancy'}_{pid_ver}.csv',save_dir=f"{pre_config['output_csv']}", show=False,base_title=f"{exp}_Heatmap_")