
import numpy as np
import yaml
from pathlib import Path
import sys
import os
from unique_m7_m8 import run, simulation_wrapper
from lr_Idep import create_cov_lr
from Partial_Information_Decomposition.Idep_Simulations.Simulation_utils import on_covariance, plot_nodes_as_alpha 
from functools import partial
    



def load_config_parts(yaml_file):
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)

    return (
        cfg["Mutual_Information_Simulation"].copy(),
        cfg["0=Mutual_Information_Simulation"].copy(),
        cfg["0<Mutual_Information_Simulation"].copy(),
        cfg["N_P_variations"].copy(),
    )


def make_pre_config(exp, MI_config, mi0_config, above0_mi_config, n_p_config):
    if exp == "MI=0":
        return {**MI_config, **mi0_config, **n_p_config}
    return {**MI_config, **above0_mi_config, **n_p_config}


def shrinkage_simulation(exp_list, shrinkage_list, alpha_list, yaml_file, folder_path, plot_heat_map):
    print("Running M7 and M8 shrinkage simulation...")
    main_func = partial(simulation_wrapper, intermediate_func=on_covariance)

    MI_config, mi0_config, above0_mi_config, n_p_config = load_config_parts(yaml_file)

    for exp in exp_list:
        for shrink in shrinkage_list:
            pre_config = make_pre_config(exp, MI_config, mi0_config, above0_mi_config, n_p_config)
            pre_config["on_covariance"] = shrink

            node_dict = {}
            bias_dict = {}
            after_corr_bias_dict = {}

            if shrink == "shrunk_cov":
                for alpha in alpha_list:
                    alpha = float(alpha)
                    pre_config["alpha"] = alpha

                    exp_name = f"ahatahat_{exp}_shrinkage_{shrink}_alpha_{alpha}"
                    print(f"\nRunning experiment: {exp_name}")

                    run_result = run(
                        main_func,
                        exp_name,
                        pre_config.copy(),
                        save_path=None,
                        plot_heatmaps=plot_heat_map,
                    )

                    i_result, j_result, k_result, h_result = run_result[:4]

                    node_dict[alpha] = {
                        "i": i_result[0]["mean"],
                        "j": j_result[0]["mean"],
                        "k": k_result[0]["mean"],
                        "h": h_result[0]["mean"],
                    }
                    bias_dict[alpha] = {
                        "i": i_result[0]["emp_bias"],
                        "j": j_result[0]["emp_bias"],
                        "k": k_result[0]["emp_bias"],
                        "h": h_result[0]["emp_bias"],
                    }
                    after_corr_bias_dict[alpha] = {
                        "i": i_result[0]["after_corr_bias"],
                        "j": j_result[0]["after_corr_bias"],
                        "k": k_result[0]["after_corr_bias"],
                        "h": h_result[0]["after_corr_bias"],
                    }

                save_path = folder_path / f"3.0Bootstrap_{exp}_shrinkage_{shrink}"
                save_path.mkdir(parents=True, exist_ok=True)

                title = (
                    f"Bootstrap_seed{pre_config['seed']}_"
                    f"{exp}_sample-{pre_config['N_values'][0]}_"
                    f"dim-{pre_config['p_values'][0]}"
                )

                plot_nodes_as_alpha(node_dict, title=title, save_path=save_path)
                plot_nodes_as_alpha(bias_dict, title="Bias_" + title, save_path=save_path)
                plot_nodes_as_alpha(after_corr_bias_dict, title="After_Corr_Bias_" + title, save_path=save_path)

                with open(save_path / f"{title}_config.yaml", "w") as f:
                    yaml.safe_dump(pre_config.copy(), f, sort_keys=False, allow_unicode=True)

            else:
                exp_name = f"ahatahat_{exp}_shrinkage_{shrink}"
                save_path = folder_path / exp_name
                save_path.mkdir(parents=True, exist_ok=True)

                print(f"\nRunning experiment: {exp_name}")
                run(main_func, exp_name, pre_config.copy(), save_path=save_path, plot_heatmaps=False)


def linear_regression_simulation(exp_list,yaml_file,folder_path):
    print("Running M7 and M8 linear regression simulation...")

    MI_config, mi0_config, above0_mi_config, n_p_config = load_config_parts(yaml_file)

    for exp in exp_list:
        pre_config = make_pre_config(exp, MI_config, mi0_config, above0_mi_config, n_p_config)

        exp_name = f"4.0LR_{exp}"
        save_path = folder_path / exp_name
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\nRunning experiment: {exp_name}")
        func = simulation_wrapper
        pre_config['intermediate_func'] = create_cov_lr
        run(func, exp_name, pre_config.copy(), save_path=save_path, plot_heatmaps=True)

        title = (
            f"LR_seed{pre_config['seed']}_"
            f"{exp}_sample-{pre_config['N_values'][0]}_"
            f"dim-{pre_config['p_values'][0]}"
        )

        with open(save_path / f"{title}_config.yaml", "w") as f:
            yaml.safe_dump(pre_config.copy(), f, sort_keys=False, allow_unicode=True)

def idep_simulation(exp_list, yaml_file, folder_path):
    print("Running M7 and M8 Idep simulation...")

    MI_config, mi0_config, above0_mi_config, n_p_config = load_config_parts(yaml_file)

    for exp in exp_list:
        pre_config = make_pre_config(exp, MI_config, mi0_config, above0_mi_config, n_p_config)

        exp_name = f"5.0Idep_{exp}"
        save_path = folder_path / exp_name
        save_path.mkdir(parents=True, exist_ok=True)
        pre_config['intermediate_func'] = on_covariance  # Set whiten to False for Idep simulation
        print(f"\nRunning experiment: {exp_name}")
        run(simulation_wrapper, exp_name, pre_config.copy(), save_path=save_path, plot_heatmaps=True)

        title = (
            f"Idep_seed{pre_config['seed']}_"
            f"{exp}_sample-{pre_config['N_values'][0]}_"
            f"dim-{pre_config['p_values'][0]}"
        )

        with open(save_path / f"{title}_config.yaml", "w") as f:
            save_config = {key: value for key, value in pre_config.items() if not callable(value)}
            yaml.safe_dump(save_config, f, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":

    exp_list = ["MI>0", "MI=0"]
    yaml_file_lr = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/configs/shrinkage.yaml"
    folder_path = Path("/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/figures/lr")


   # linear_regression_simulation(exp_list, yaml_file_lr, folder_path)

    shrinkage_list = ["shrunk_cov"]
    alpha_list = np.linspace(0.00001, 1.0, 50)
    plot_heat_map = False
    #shrinkage_simulation(exp_list, shrinkage_list, alpha_list, yaml_file, folder_path, plot_heat_map)


    folder_path = Path("/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/figures/idep_LB")
    yaml_bias_correction = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/configs/sim.yaml'
    idep_simulation(exp_list, yaml_bias_correction, folder_path)