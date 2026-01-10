import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))  
from encoding_model.toy_example import run_all_methods,run_experiment,commonality_analysis
from Partial_Information_Decomposition.Idep_univariabe_gauss import Idep_univariate_gauss
""""This module implements the supression effect for gaussian univariate sorces and targets. 
and computes Variance Partitioning and Partial Information Decomposition using the Idep method."""




def gauss_simple_example(N=1000,P=1,rng_seed=1, noise_seed=1,simple_example=True, snr=1.0,method='ridge_cv',mixing_dimension=None):
    """Run a simple Gaussian example with no mixing deminsion each experiement has different random seeds.
    Parameters
    ----------
    N : int
        Number of samples.
    P : int
        Number of features.
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

    print("\n" + "="*70)
    print(f"Simple Example with RNG seed {rng_seed} and Noise seed {noise_seed}")
    print("="*70)
    rng = np.random.default_rng(seed=rng_seed)
    noise_rng = np.random.default_rng(seed=noise_seed)
    # --- generate data ---
    vp_results , t_m_dict = run_experiment(rng,noise_rng,simple_example,N,P,mixing_dimension,snr,method) #Variance Partitioning and sources and target

    M1 = t_m_dict['X_M1']
    M2 = t_m_dict['X_M2']
    T = t_m_dict['y']
    # --- compute PID ---

    #Change to tensors: 
    M1 = torch.tensor(M1, dtype=torch.float64)
    M2 = torch.tensor(M2, dtype=torch.float64)
    T = torch.tensor(T, dtype=torch.float64)

    sources = [M1,M2]
    targets = [T]
    idep_class = Idep_univariate_gauss(sources,targets)
    pid = idep_class.idep()
    print("\nIdep PID results:")
    for key, value in pid.items():
        print(f"- {key}: {value:.4f}")

    return vp_results,pid


def compare_results(vp_results,pid_results):
    """Compare Variance Partitioning and Partial Information Decomposition results.
    Parameters
    ----------
    vp_results : dict
        Results from variance partitioning.
    pid_results : dict
        Results from Partial Information Decomposition.

    Returns
    -------
    None
    """
    print("\n" + "="*70)
    print("Comparison of Variance Partitioning and PID Results")
    print("="*70)
    print("Results:")
    print(f"M1 variance explained (VP): {vp_results['R²_A']:.4f} | M1 mutual information (PID): {pid_results['unq0'] + pid_results['red'] :.4f}")
    print(f"M2 variance explained (VP): {vp_results['R²_B']:.4f} | M2 mutual information (PID): {pid_results['unq1'] + pid_results['red'] :.4f}")
    print(f"Both M1 and M2 variance explained (VP): {vp_results['R²_AB']:.4f} | Total mutual information (PID): {pid_results['unq0'] + pid_results['unq1'] + pid_results['red'] + pid_results['syn']:.4f}")
    print(f"\nUnique to M1 (VP): {vp_results['unique_A']:.4f} | Unique to M1 (PID): {pid_results['unq0']:.4f}")
    print(f"Unique to M2 (VP): {vp_results['unique_B']:.4f} | Unique to M2 (PID): {pid_results['unq1']:.4f}")
    print(f"Common (VP): {vp_results['common']:.4f} | Redundant (PID): {pid_results['red']:.4f}")
    print(f"Synergy (PID): {pid_results['syn']:.4f}")





def main():
    """Main function to run the Gaussian simple example and compare results."""
    N, P = 1000, 1
    rng_seed, noise_seed = 42, 24
    snr = 1
    method = 'ridge_cv'
    mixing_dimension = None

    print("\nRunning Gaussian Univariate target and sources simple example...")

    print("\n" + "="*70)
    print("Experiment 1:")
    vp_results, pid_results = gauss_simple_example(N, P, rng_seed, noise_seed, simple_example=True, snr=snr, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results)


if __name__ == "__main__":
    main()