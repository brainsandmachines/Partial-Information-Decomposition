import csv
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
from cov_functions import (
    change_covariance_order,
    make_both_unique_true_cov_from_config,
    make_direct_true_cov_from_config,
    make_random_true_cov,
    sample_from_cov,
)
from pid.mutual_information import gaussian_mutual_information
from pid.calculation import pid_calc
from pid.utils import create_cov_matrix
from save_results import save_sample_simulation_results_table

OWN_MI_KEYS = (
    'own_imx_raw_bits',
    'own_imy_raw_bits',
    'own_imxy_raw_bits',
    'own_imx_wishart_bias_bits',
    'own_imy_wishart_bias_bits',
    'own_imxy_wishart_bias_bits',
    'own_imx_wishart_bits',
    'own_imy_wishart_bits',
    'own_imxy_wishart_bits',
)

RESULT_KEYS = (
    'red',
    'unq1',
    'unq2',
    'syn',
    'bi_mi_1',
    'bi_mi_2',
    'tri_mi',
    'union_info',
    'obj',
    'obj_bias',
) + OWN_MI_KEYS  # (10 result keys,) + (9 own-MI keys,) -> (19 result keys,)

"""This file generates a covriance matrix, and sample from it and then calculate PID values. 
But it is also calculate the PID values for the true covariance, so it is not only a sampling script but also a script for calculating the true PID values for the covariance.
it is to check the Bias out of sampling, and also to check the true PID values for the covariance.
"""

DEFAULT_CONFIG_PATH = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulations/theoretical/rv_config.yaml"


def calculate_sample_whitened_wishart_mi_bits(
    sources: list[torch.Tensor],
    target: list[torch.Tensor],
) -> dict[str, float]:
    """Calculate raw and Wishart-corrected sample Gaussian MIs in bits.

    Inputs:
        sources: list[torch.Tensor], exactly two two-dimensional sample tensors
            ordered as ``[X1, X2]``, with shapes ``(N, dx1)`` and ``(N, dx2)``.
        target: list[torch.Tensor], exactly one two-dimensional sample tensor
            ``T`` with shape ``(N, dt)``.

    Outputs:
        dict[str, float], raw MI, exact Wishart MI bias, and bias-corrected MI
        for ``imx = I(T; X1)``, ``imy = I(T; X2)``, and
        ``imxy = I(T; [X1, X2])``. Every returned value is in bits. Corrected
        values are not clipped at zero.

    Notes:
        The sample covariance uses ``N - 1`` in the denominator. Wishart
        correction is applied to each Gaussian MI without clipping.
    """
    if len(sources) != 2:
        raise ValueError(f"sources must contain [X1, X2]; got {len(sources)} tensors.")
    if len(target) != 1:
        raise ValueError(f"target must contain [T]; got {len(target)} tensors.")

    x1, x2 = sources
    t = target[0]
    sample_tensors = (x1, x2, t)
    if any(tensor.ndim != 2 for tensor in sample_tensors):
        raise ValueError("X1, X2, and T must all have shape (N, dimension).")
    n_samples = x1.shape[0]
    if any(tensor.shape[0] != n_samples for tensor in sample_tensors[1:]):
        raise ValueError("X1, X2, and T must contain the same number of samples.")
    if any(tensor.device != x1.device for tensor in sample_tensors[1:]):
        raise ValueError("X1, X2, and T must be on the same torch device.")

    dx1, dx2 = x1.shape[1], x2.shape[1]
    covariance = create_cov_matrix(
        rvs=[x1, x2, t],
        device=x1.device,
    )  # [(N, dx1), (N, dx2), (N, dt)] -> covariance blocks for (D, D)
    imx = gaussian_mutual_information(
        covariance['joint_x1_t'],
        dx1,
        n_samples=n_samples,
        bias_correction="wishart",
    )
    imy = gaussian_mutual_information(
        covariance['joint_x2_t'],
        dx2,
        n_samples=n_samples,
        bias_correction="wishart",
    )
    imxy = gaussian_mutual_information(
        covariance['full_cov'],
        dx1 + dx2,
        n_samples=n_samples,
        bias_correction="wishart",
    )

    return {
        'own_imx_raw_bits': imx['raw_bits'],
        'own_imy_raw_bits': imy['raw_bits'],
        'own_imxy_raw_bits': imxy['raw_bits'],
        'own_imx_wishart_bias_bits': imx['bias_bits'],
        'own_imy_wishart_bias_bits': imy['bias_bits'],
        'own_imxy_wishart_bias_bits': imxy['bias_bits'],
        'own_imx_wishart_bits': imx['corrected_bits'],
        'own_imy_wishart_bits': imy['corrected_bits'],
        'own_imxy_wishart_bits': imxy['corrected_bits'],
    }


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

    values = list(RESULT_KEYS)
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
        config: dict, simulation configuration containing dimensions, n_trials,
            n_samples, device, output_dir, and optional
            use_both_unique_covariance.
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


    simulation_seed = int(config.get('seed', 42))
    torch.manual_seed(simulation_seed)
    rng = torch.Generator(device=config['device'])
    rng.manual_seed(simulation_seed)
    experiment_name = experiment_name or config.get("exp_name", "sample_simulation")

    if config.get('use_both_unique_covariance', False):
        true_cov = make_both_unique_true_cov_from_config(config, rng=rng)
    else:
        true_cov = make_direct_true_cov_from_config(config)

    dims = [config['dx1'], config['dx2'], config['dt']]
    reordered_cov = change_covariance_order(true_cov, new_order=[2,0,1], dims=dims)
    population_covariance = create_cov_matrix(
        Sigma=true_cov,
        dims=dims,
        check_singular=False,
    )  # population covariance (D, D) -> covariance blocks for (D, D)
    population_imx_bits = gaussian_mutual_information(
        population_covariance['joint_x1_t'],
        dims[0],
    )['raw_bits']
    population_imy_bits = gaussian_mutual_information(
        population_covariance['joint_x2_t'],
        dims[1],
    )['raw_bits']
    population_imxy_bits = gaussian_mutual_information(
        population_covariance['full_cov'],
        dims[0] + dims[1],
    )['raw_bits']
    own_theoretical_mi = {
        'own_imx_raw_bits': population_imx_bits,
        'own_imy_raw_bits': population_imy_bits,
        'own_imxy_raw_bits': population_imxy_bits,
        'own_imx_wishart_bias_bits': 0.0,
        'own_imy_wishart_bias_bits': 0.0,
        'own_imxy_wishart_bias_bits': 0.0,
        'own_imx_wishart_bits': population_imx_bits,
        'own_imy_wishart_bits': population_imy_bits,
        'own_imxy_wishart_bits': population_imxy_bits,
    }

    theoretical_values = {}
    #Calculate true PID
    for method in methods:
        cov = reordered_cov  # (D, D) [X1, X2, T] -> (D, D) [T, X1, X2]
        theoretical_pid, theoretical_mi = pid_calc(
            {**config, 'bias_correction': False},
            rng=rng,
            covariance=cov,
            method=method,
        )
        theoretical_pid = theoretical_pid.copy()
        theoretical_pid['obj_bias'] = (
            theoretical_pid.get('obj', np.nan)
            - theoretical_pid.get('union_info', np.nan)
        )
        theoretical_mi = {**theoretical_mi, **own_theoretical_mi}
        theoretical_values[method] = (theoretical_pid, theoretical_mi)


    values = list(RESULT_KEYS)
    sampled_value = {
        method: {
            key: np.zeros((config['n_trials'],))
            for key in values
        }
        for method in methods
    }  # scalar trial count -> method/component NumPy arrays with shape (n_trials,)

    n_trials = config['n_trials']
    for trial in range(n_trials):
        if trial % max(1, n_trials // 10) == 0:
            print(f"Trial {trial}/{n_trials} ({(trial / n_trials) * 100:.1f}%)")
        # Sample from the covariance and calculate PID for the samples
        sampled_cov, rvs = sample_from_cov(config,true_cov, config['n_samples'], rng)
        sources = [rvs[0], rvs[1]]
        target = [rvs[2]]
        own_sample_mi = calculate_sample_whitened_wishart_mi_bits(
            sources,
            target,
        )
        for method in methods:
            pid_results, mi_results = pid_calc(
                config,
                sources=sources,
                target=target,
                rng=rng,
                method=method,
            )

            results = {**pid_results, **mi_results, **own_sample_mi}
            results['obj_bias'] = (
                results.get('obj', np.nan)
                - results.get('union_info', np.nan)
            )

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
    bias_methods = ['Venkatesh']
    d = 80
    for bias_method in bias_methods:
        config_path = Path(__file__).resolve().parents[1] / 'rv_config_analytic.yaml'

        loaded_config = load_config(config_path)
        config = {
            **loaded_config['parameters'],
            **loaded_config['covariance'],
        }

        config['dx1'] = d
        config['dx2'] = d
        config['dt'] = d

        exp_name = f"ZeroMIChannels_{bias_method}-Dim{d}_trials{config['n_trials']}"
        results = simulation(config, config['methods'], experiment_name=exp_name)

        save_sample_simulation_results_table(
            results,
            config,
            Path(config['output_dir']) / exp_name / f"{exp_name}_sample_simulation_results.png",
        )
