import torch
import numpy as np
import yaml
from pathlib import Path
import sys
STORY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STORY_ROOT.parents[2]
DEFAULT_CONFIG_PATH = STORY_ROOT.parent / "rv_config.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STORY_ROOT) not in sys.path:
    sys.path.insert(0, str(STORY_ROOT))
from cov_functions import make_direct_true_cov_from_config, make_random_true_cov, change_covariance_order,sample_from_cov
from Partial_Information_Decomposition.PID_calc import pid_calc
from save_results import save_sample_simulation_results_table

"""This file generates a covriance matrix, and sample from it and then calculate PID values. 
But it is also calculate the PID values for the true covariance, so it is not only a sampling script but also a script for calculating the true PID values for the covariance.
it is to check the Bias out of sampling, and also to check the true PID values for the covariance.
"""

DEFAULT_CONFIG_PATH = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Simulations/Theoretical_Examples/rv_config.yaml"

def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def csv_save(config: dict, experiment_name: str, method: str, theoretical_values: tuple, sampled_values: dict) -> Path:
    """Save theoretical and sampled PID/MI component values to one method CSV.

    Inputs:
        config: dict, simulation configuration containing output_dir.
        experiment_name: str, name used for the output folder and CSV prefix.
        method: str, PID definition/method name used in the CSV filename.
        theoretical_values: tuple, output of pid_calc for the theoretical covariance as (pid_dict, mi_dict).
        sampled_values: dict, component arrays keyed by PID/MI component names, one value per trial.

    Outputs:
        Path, path to the saved CSV file.
    """
    import csv

    values = ['red', 'unq1', 'unq2', 'syn', 'bi_mi_1', 'bi_mi_2', 'tri_mi']
    pid_values, mi_values = theoretical_values
    theoretical_row = {key: {**pid_values, **mi_values}.get(key, np.nan) for key in values}

    n_trials = max((len(sampled_values[key]) for key in values if key in sampled_values), default=0)
    rows = [theoretical_row]
    for trial in range(n_trials):
        rows.append({key: sampled_values.get(key, np.full(n_trials, np.nan))[trial] for key in values})

    csv_path = Path(config['output_dir']) / experiment_name / f"{experiment_name}_{method}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=values)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path






def simulation(config: dict, methods: list, experiment_name: str | None = None) -> dict:
    """Run theoretical-covariance PID calculations and sampled trial summaries.

    Inputs:
        config: dict, simulation configuration containing dimensions, n_trials, n_samples, device, and output_dir.
        methods: list, PID method names to evaluate.
        experiment_name: str | None, optional name used for CSV output folder and filename prefix.

    Outputs:
        dict, method-keyed results containing theoretical, mean_sampled, bias, variance, and mse component values.


    Raises:
        FileExistsError: if config['output_dir']/experiment_name already exists.
    """

    experiment_name = experiment_name or config.get("exp_name", "sample_simulation")
    experiment_dir = Path(config['output_dir']) / experiment_name
    if experiment_dir.exists():
        raise FileExistsError(f"Experiment folder already exists: {experiment_dir}")


    rng = torch.Generator(device=config['device'])
    experiment_name = experiment_name or config.get("exp_name", "sample_simulation")

    true_cov = make_direct_true_cov_from_config(config)

    dims = [config['dx1'], config['dx2'], config['dt']]
    reordered_cov = change_covariance_order(true_cov, new_order=[2,0,1], dims=dims)

    reorederd_methods = ['flow','Flow' ,'Tilde','tilde' ,'Delta','delta'] #They all assume the same order of variables, so they should be calculated with the same covariance, which is the reordered_cov. The rest of the methods should be calculated with the true_cov.
    theoretical_values = {}
    #Calculate true PID
    for method in methods:
        if method in reorederd_methods:
            cov = reordered_cov
        else:            
            cov = true_cov
        theoretical_values[method] = pid_calc(config,rng=rng,covariance = cov, method=method)


    values = ['red', 'unq1', 'unq2', 'syn', 'bi_mi_1', 'bi_mi_2', 'tri_mi']
    sampled_value = {method: {k: np.zeros((config['n_trials'],)) for k in values} for method in methods}

    n_trials = config['n_trials']
    for trial in range(n_trials):
        if trial % max(1, n_trials // 10) == 0:
            print(f"Trial {trial}/{n_trials} ({(trial / n_trials) * 100:.1f}%)")
        # Sample from the covariance and calculate PID for the samples
        sampled_cov, rvs = sample_from_cov(config,true_cov, config['n_samples'], rng)
        reordered_sampled_cov = change_covariance_order(sampled_cov, new_order=[2,0,1], dims=dims)
        sources = [rvs[0], rvs[1]]
        target = [rvs[2]]
        for method in methods:
            if method in reorederd_methods:
                cov = reordered_sampled_cov
            else:
                cov = sampled_cov
            pid_results,mi_results = pid_calc(config,sources=sources,target=target, method=method)

            results = {**pid_results, **mi_results}

            for k in values:
                sampled_value[method][k][trial] = results[k]

    for method in methods:
        csv_save(config, experiment_name, method, theoretical_values[method], sampled_value[method])


    theoretical_component_values = {
        method: {**theoretical_values[method][0], **theoretical_values[method][1]}
        for method in methods
    }
    mean_sampled_values = {
        method: {key: np.mean(sampled_value[method][key]) for key in values}
        for method in methods
    }
    bias = {
        method: {key: mean_sampled_values[method][key] - theoretical_component_values[method][key] for key in values}
        for method in methods
    }
    var = {
        method: {key: np.var(sampled_value[method][key]) for key in values}
        for method in methods
    }
    mse = {
        method: {key: bias[method][key]**2 + var[method][key] for key in values}
        for method in methods
    }

    results = {k: {"theoretical": theoretical_values[k], "mean_sampled": mean_sampled_values[k], "bias": bias[k], "variance": var[k], "mse": mse[k]} 
               for k in methods}
        
    return results
    


if __name__ == "__main__":

    #Config and experiment setup
    config = load_config()
    config_params = config['parameters']
    config_cov = config['covariance']
    config = {**config_params, **config_cov}
    methods = config.get("methods", ["flow", "tilde", "delta", "idep"])

    exp_name = f'Tilde-Debiased_{config["n_samples"]}_samples_{config["n_trials"]}_trials'
    #exp_name = 'testt'
    results = simulation(config, methods, experiment_name=exp_name)
    
    save_sample_simulation_results_table(
        results,
        config,
        Path(config['output_dir']) / exp_name / f"{exp_name}_sample_simulation_results.png",
    )
