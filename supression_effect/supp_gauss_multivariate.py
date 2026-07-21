from xml.parsers.expat import model

from sklearn.linear_model import LinearRegression
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path


root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))  
from encoding_model.commonality import commonality_analysis
from toy_examples.toy_example import run_experiment
from Partial_Information_Decomposition.Idep.Idep_multivariate_gauss import Idep_multivariate_gauss
from Partial_Information_Decomposition.PID_util import compare_results
from my_utils import (
    Tee,
    extract_all_components,
    get_seed_runs_csv_path,
    print_seed_summary,
    run_multi_seed_experiment,
    save_seed_summary_csv,
    seed_summary_to_table,
    save_csv_column_means
)

from Partial_Information_Decomposition.Idep.Idep_Simulations.Simulation_utils import CCA_reduction



""""This module implements the supression effect for gaussian univariate sorces and targets. 
and computes Variance Partitioning and Partial Information Decomposition using the Idep method."""

log = open("p1_pidvsvp.log", "w")

sys.stdout = Tee(sys.stdout, log)
sys.stderr = Tee(sys.stderr, log)


from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import numpy as np
import torch

def crossfit_residualize(Y_raw, X_raw, n_splits=5, seed=0):
    """
    Residualize Y_raw against X_raw using cross-fitted linear regression.
    Returns residuals Y - E[Y|X] predicted out-of-fold.
    """
    Y_raw = np.asarray(Y_raw)
    X_raw = np.asarray(X_raw)

    residuals = np.zeros_like(Y_raw, dtype=np.float64)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X_raw):
        model = LinearRegression().fit(X_raw[train_idx], Y_raw[train_idx])
        pred = model.predict(X_raw[test_idx])
        residuals[test_idx] = Y_raw[test_idx] - pred

    return residuals
def test_suppression(N=1000,P=1,suppression_strength=0.5,rng_seed=1,mode='simple', snr=1.0,method='ridge_cv',mixing_dimension=None):
    """Run a simple Gaussian example with no mixing deminsion each experiement has different random seeds.
    Parameters
    ----------
    N : int
        Number of samples.
    P : int
        Number of features.
    suppresion_strength : float
        Strength of suppression effect.
    rng_seed : int
        Seed for the random number generator.
    noise_seed : int
        Seed for the noise generator.
    snr : float
        Signal-to-noise ratio.
    method : str
        Method for variance partitioning ('ols', 'ridge_cv', etc.).
    mixing_dimension : int or None
        Dimension of the mixing matrix. If None, no mixing is applied.

    Returns
    -------
    Partial Information Decomposition (using Idep method) results printed to console.
    Varaince Partitioning results are returned from run_experiment function.
    """

    rng = np.random.default_rng(seed=rng_seed)
    rng_torch = torch.Generator().manual_seed(rng_seed)
    # --- generate data ---
    vp_results , t_m_dict = run_experiment(rng,suppression_strength,mode,N,P,mixing_dimension,snr,method) #Variance Partitioning and sources and target

    M1_raw = t_m_dict['X_M1']
    M2_raw = t_m_dict['X_M2']
    T = t_m_dict['y']
    # vif_summary(np.array([M1,M2]))
    # vif_summary(np.hstack((M1,M2)))
    # std_scaling_summary(np.array([M1,M2]))
    # std_scaling_summary(np.hstack((M1,M2)))
    # std_scaling_summary(T)
    # --- compute PID ---

    #cca_output = CCA_reduction(device=M1_raw.device, rv_list=[M1_raw,M2_raw], n_components=M1_raw.shape[1]//3)
    #M1, M2 = cca_output['X0'], cca_output['X1']
    #M1_resid = crossfit_residualize(M1_raw, M2_raw, n_splits=5, seed=rng_seed)
    #Change to tensors: 
    M1 = torch.tensor(M1_raw)
    M2 = torch.tensor(M2_raw)
    T = torch.tensor(T)

    sources = [M1,M2]
    
    targets = [T]
    idep_class = Idep_multivariate_gauss(rng=rng_torch, sources=sources, targets=targets, 
                                         bias_correction=False)
    pid , mi = idep_class.idep()
    return vp_results,pid,mi




def plot_pid_results(mi_results= None, pid_results=None, sub_title=None):
    """Plot bar chart for PID results.
    
    Parameters
    ----------
    pid_results : dict
        Dictionary containing PID components (unq0, unq1, red, syn).
    title : str
        Title for the plot.
    """
    if pid_results is not None:
    # Extract the main PID components
        components = {
            'Unique(M1)': pid_results.get('unq0', 0),
            'Unique(M2)': pid_results.get('unq1', 0),
            'Redundant': pid_results.get('red', 0),
            'Synergy': pid_results.get('syn', 0)
            
        }
        title = "PID Components"
    elif mi_results is not None:
        components = {
            'I(M1;T)': mi_results.get('I(M0;T)', 0),
            'I(M2;T)': mi_results.get('I(M1;T)', 0),
            'I(M1,M2;T)': mi_results.get('I(M0,M1;T)', 0)
        }
        title = "Mutual Information Components"
    else:
        raise ValueError("Either pid_results or mi_results must be provided.")
    
    
    keys = list(components.keys())
    values = list(components.values())
    
    # Create a colormap with different colors for each component
    colors = plt.cm.Set2(np.linspace(0, 1, len(values)))
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(keys)), values, color=colors)
    plt.xticks(range(len(keys)), keys, rotation=0, ha='center', fontsize=18)
    plt.yticks(fontsize=16)
    plt.ylabel('Information (bits)', fontsize=18)
    plt.title(title, fontsize=20)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(f"/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/figures_/{title}_{sub_title}.pdf", transparent=True)


def get_seed_sweep_config() -> dict:
    """Configuration for fixed-parameter suppression simulations across seeds."""
    return {
        "n_seeds": 100,
        "seed_start": 0,
        "N": 500000,
        "P": 50,
        "suppression_strength": 0.5,
        "simple_example": False,
        "snr": 5,
        "method": "ridge_cv",
        'mode': 'only_unq2_zero',
        "mixing_dimension": None,
        "jackknife": False,
        "results_dir": "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/Toy_Example/only_unq2_zero",
        "results_prefix": "seed_summary",
        "all_runs_results_prefix": "seed_runs",
        "progress_print_every": 10,
        "test_name": "cca_toyexample_supp_gauss_multivariate",
    }


def run_single_seed_fixed(seed: int, config: dict) -> dict:
    """Run one suppression experiment seed with all other parameters fixed."""
    vp_results, pid_results, mi_results = test_suppression(
        N=config["N"],
        P=config["P"],
        suppression_strength=config["suppression_strength"],
        rng_seed=seed,
        mode=config["mode"],
        snr=config["snr"],
        method=config["method"],
        mixing_dimension=config["mixing_dimension"],
    )
    return extract_all_components(vp_results, pid_results, mi_results)


def run_fixed_params_across_seeds(config: dict | None = None) -> tuple[dict, list[dict]]:
    """
    Sweep over seeds while keeping all other simulation parameters fixed,
    then save the per-seed results and mean/std summary.
    """
    run_config = get_seed_sweep_config() if config is None else config

    summary, seed_rows = run_multi_seed_experiment(
        run_config,
        per_seed_runner=run_single_seed_fixed,
    )

    print_seed_summary(
        summary,
        n_seeds=run_config["n_seeds"],
        seed_start=run_config["seed_start"],
    )
    all_runs_path = get_seed_runs_csv_path(run_config)
    summary_path = save_seed_summary_csv(summary, run_config)
    print(f"\nSaved all seed run results to: {all_runs_path}")
    print(f"Saved summary to: {summary_path}")

    return summary, seed_rows
        




def main():
    """Main function to run the Gaussian simple example and compare results."""
    N, P = 1000000, 1
    rng_seed = 56
    snr = 0.5
    method = 'ridge_cv'
    mixing_dimension = None
    mode = 'simple'
    suppression_strength = 0.5
    print("\n" + "="*70)
    print(f"Running Test with N,P = {N,P} and mode = {mode}")
    print(f"\nExperiment 1: snr = {snr}")
    vp_results, pid_results, mi_results = test_suppression(N, P, suppression_strength, rng_seed, mode=mode, snr=snr, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results, mi_results)
    #plot_pid_results(pid_results=pid_results, sub_title="Experiment 2")
    #plot_pid_results(mi_results=mi_results, sub_title="Experiment 2")

    print("\n" + "="*70)
    print(f"Experiment 2: snr = {snr} multivariate gaussian with different seeds")
    vp_results, pid_results, mi_results = test_suppression(N, P, suppression_strength, rng_seed+2, mode=mode, snr=snr, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results, mi_results)

    print("\n" + "="*70)
    snr = 0.9
    print(f"Experiment 3: snr = {snr}")
    vp_results, pid_results, mi_results = test_suppression(N, P, suppression_strength, rng_seed, mode=mode, snr=snr, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results, mi_results)

    print("\n" + "="*70)
    print(f"Experiment 4: snr = {snr} multivariate gaussian with different seeds")
    vp_results, pid_results, mi_results = test_suppression(N, P, suppression_strength, rng_seed+2, mode=mode, snr=snr, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results, mi_results)

    print("\n" + "="*70)
    snr = 1.2
    print(f"Experiment 5: snr = {snr} multivariate gaussian with different seeds")
    vp_results, pid_results, mi_results = test_suppression(N, P, suppression_strength, rng_seed+1, mode=mode, snr=snr, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results, mi_results)

    print("\n" + "="*70)
    snr = 3
    print(f"Experiment 6: snr = {snr}")
    vp_results, pid_results, mi_results = test_suppression(N, P, suppression_strength, rng_seed, mode=mode, snr=snr, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results, mi_results)

    print("\n" + "="*70)
    snr = 5
    print(f"Experiment 7: snr = {snr} multivariate gaussian with different seeds")
    vp_results, pid_results, mi_results = test_suppression(N, P, suppression_strength, rng_seed+1, mode=mode, snr=snr, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results, mi_results)

    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
    # For fixed-parameter seed analysis and saved mean/std across seeds, run:
    # run_fixed_params_across_seeds()
    # csv_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/Toy_Example/only_unq2_zero/seed_runs_toyexample_supp_gauss_multivariate.csv'
    # means_save_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/Toy_Example/only_unq2_zero/seed_summary_toyexample_supp_gauss_multivariate.csv'
    # save_csv_column_means(csv_path=csv_path, output_csv_path=means_save_path)
