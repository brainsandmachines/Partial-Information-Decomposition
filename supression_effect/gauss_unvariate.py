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

def check_supression_effect(vp_results,pid_results):
    """Check for suppression effect in the results.
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
    R_A = vp_results['R²_A']
    R_B = vp_results['R²_B']
    R_AB = vp_results['R²_AB']
    unique_A = vp_results['unique_A']
    unique_B = vp_results['unique_B']
    common = vp_results['common']

    unq0 = pid_results['unq0']
    unq1 = pid_results['unq1']
    red = pid_results['red']
    syn = pid_results['syn']

    if np.isclose(R_A,0) or R_A < 0 and unique_A > 0:
        print("\nSuppression effect detected: M1 is a suppressor variable.❗❗❗")
        if not np.isclose(unq0,0,atol=1e-5):
            print("PID fell to supression effect ❌")
        else: 
            if syn > 0:
                print("PID did not fall to supression effect and detected synergy ✅✅✅")
            else:
                print("PID did not fall to supression effect ✅ (No synergy detected) ❌")
            
    elif np.isclose(R_B,0) or R_B < 0 and unique_B > 0:
        print("\nSuppression effect detected: M2 is a suppressor variable.❗❗❗")
        if not np.isclose(unq1,0,atol=1e-5):
            print("PID fell to supression effect ❌")
        else: 
            if syn > 0:
                print("PID did not fall to supression effect and detected synergy ✅✅✅")
            else:
                print("PID did not fall to supression effect ✅ (No synergy detected) ❌")
    else:
        print("\nNo suppression effect detected: One of the unique contributions is zero.")
    return 


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
    print(f"M1 R² (VP): {vp_results['R²_A']:.4f} | I(T;M1): {pid_results['unq0'] + pid_results['red'] :.4f}")
    print(f"M2 R² (VP): {vp_results['R²_B']:.4f} | I(M2;T): {pid_results['unq1'] + pid_results['red'] :.4f}")
    print(f"Both M1 and M2 R² (VP): {vp_results['R²_AB']:.4f} | I(M1,M2;T): {pid_results['unq0'] + pid_results['unq1'] + pid_results['red'] + pid_results['syn']:.4f}")
    print(f"\nUnique to M1 (VP): {vp_results['unique_A']:.4f} | Unique(T;M1\\M2): {pid_results['unq0']:.4f}")
    print(f"Unique to M2 (VP): {vp_results['unique_B']:.4f} | Unique(T;M2\\M1): {pid_results['unq1']:.4f}")
    print(f"Common (VP): {vp_results['common']:.4f} | Redundant (PID): {pid_results['red']:.4f}")
    print(f"Synergy (PID): {pid_results['syn']:.4f}")

    
    check_supression_effect(vp_results,pid_results)
        




def main():
    """Main function to run the Gaussian simple example and compare results."""
    N, P = 10000, 1
    rng_seed, noise_seed = 42, 24
    snr = 1
    method = 'ridge_cv'
    mixing_dimension = None

    print("\nRunning Gaussian Univariate target and sources simple example...")

    print("\n" + "="*70)
    print("Experiment 1: snr = 1")
    vp_results, pid_results = gauss_simple_example(N, P, rng_seed, noise_seed, simple_example=True, snr=snr, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results)

    print("\n" + "="*70)
    print("Experiment 2: snr = 1 univariate gaussian with different seeds")
    vp_results, pid_results = gauss_simple_example(N, P, rng_seed+1, noise_seed+1, simple_example=True, snr=snr, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results)

    print("\n" + "="*70)
    print("Experiment 3: snr = 10")
    vp_results, pid_results = gauss_simple_example(N, P, rng_seed, noise_seed, simple_example=True, snr=10, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results)

    print("\n" + "="*70)
    print("Experiment 4: snr = 10 univariate gaussian with different seeds")
    vp_results, pid_results = gauss_simple_example(N, P, rng_seed+1, noise_seed+1, simple_example=True, snr=10, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results)

    print("\n" + "="*70)
    print("Experiment 5: snr = 100")
    vp_results, pid_results = gauss_simple_example(N, P, rng_seed, noise_seed, simple_example=True, snr=100, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results)

    print("\n" + "="*70)
    print("Experiment 6: snr = 100 univariate gaussian with different seeds")
    vp_results, pid_results = gauss_simple_example(N, P, rng_seed+1, noise_seed+1, simple_example=True, snr=100, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results)

    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()