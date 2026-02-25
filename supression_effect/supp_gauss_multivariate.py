import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path


root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))  
from toy_examples.toy_example import run_all_methods,run_experiment,commonality_analysis
from Partial_Information_Decomposition.Idep_multivariate_gauss import Idep_multivariate_gauss
from utils import Tee
from Partial_Information_Decomposition.PID_util import standardize, singularity_report,vif_summary,std_scaling_summary
""""This module implements the supression effect for gaussian univariate sorces and targets. 
and computes Variance Partitioning and Partial Information Decomposition using the Idep method."""

log = open("pidvsvp.log", "w")

sys.stdout = Tee(sys.stdout, log)
sys.stderr = Tee(sys.stderr, log)


def standardize(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Standardize columns of X to zero mean and unit variance.

    X shape: (N, P)
    """
    mean = X.mean(dim=0, keepdim=True)
    std  = X.std(dim=0, unbiased=False, keepdim=True)
    return (X - mean) / (std + eps)


def test_suppresion(N=1000,P=1,suppresion_strength=0.5,rng_seed=1,simple_example=True, snr=1.0,method='ridge_cv',mixing_dimension=None):
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
    # --- generate data ---
    vp_results , t_m_dict = run_experiment(rng,suppresion_strength,simple_example,N,P,mixing_dimension,snr,method) #Variance Partitioning and sources and target

    M1 = t_m_dict['X_M1']
    M2 = t_m_dict['X_M2']
    T = t_m_dict['y']
    # vif_summary(np.array([M1,M2]))
    # vif_summary(np.hstack((M1,M2)))
    # std_scaling_summary(np.array([M1,M2]))
    # std_scaling_summary(np.hstack((M1,M2)))
    # std_scaling_summary(T)
    # --- compute PID ---

    #Change to tensors: 
    M1 = torch.tensor(M1)
    M2 = torch.tensor(M2)
    T = torch.tensor(T)

    # M1 = standardize(M1)
    # M2 = standardize(M2) 
    # T = standardize(T)
    
    sources = [M1,M2]
    targets = [T]
    idep_class = Idep_multivariate_gauss(sources,targets,bias_correction=True)
    pid , mi = idep_class.idep()
    return vp_results,pid,mi

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
    R_A = vp_results['R²_X1']
    R_B = vp_results['R²_X2']
    R_AB = vp_results['R²_X12']
    unique_A = vp_results['unique_X1']
    unique_B = vp_results['unique_X2']
    common = vp_results['common']

    unq0 = pid_results['unq0']
    unq1 = pid_results['unq1']
    red = pid_results['red']
    syn = pid_results['syn']

    if np.isclose(R_A,0) or R_A < 0 and unique_A > 0:
        print("\nSuppression effect detected: M1 is a suppressor variable.❗❗❗")
        # if not np.isclose(unq0,0,atol=1e-5):
        #     print("PID fell to supression effect ❌")
        # else: 
        #     if syn > 0:
        #         print("PID did not fall to supression effect and detected synergy ✅✅✅")
        #     else:
        #         print("PID did not fall to supression effect ✅ (No synergy detected) ❌")
            
    elif np.isclose(R_B,0) or R_B < 0 and unique_B > 0:
        print("\nSuppression effect detected: M2 is a suppressor variable.❗❗❗")
        # if not np.isclose(unq1,0,atol=1e-5):
        #     print("PID fell to supression effect ❌")
        # else: 
        #     if syn > 0:
        #         print("PID did not fall to supression effect and detected synergy ✅✅✅")
        #     else:
        #         print("PID did not fall to supression effect ✅ (No synergy detected) ❌")
    else:
        print("\nNo suppression effect detected for VP: One of the unique contributions is zero.")
    return 


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
    print(f"M1 R² (VP): {vp_results['R²_X1']:.4f} | I(T;M1): {pid_results['unq0'] + pid_results['red'] :.4f}")
    print(f"M2 R² (VP): {vp_results['R²_X2']:.4f} | I(T;M2): {pid_results['unq1'] + pid_results['red'] :.4f}")
    print(f"Both M1 and M2 R² (VP): {vp_results['R²_X12']:.4f} | I(T;M1,M2): {pid_results['unq0'] + pid_results['unq1'] + pid_results['red'] + pid_results['syn']:.4f}")
    print(f"\nUnique to M1 (VP): {vp_results['unique_X1']:.4f} | Unique(T;M1\\M2): {pid_results['unq0']:.4f}")
    print(f"Unique to M2 (VP): {vp_results['unique_X2']:.4f} | Unique(T;M2\\M1): {pid_results['unq1']:.4f}")
    print(f"Common (VP): {vp_results['common']:.4f} | Redundant (PID): {pid_results['red']:.4f}")
    print(f"Synergy (PID): {pid_results['syn']:.4f}")

    
    check_supression_effect(vp_results,pid_results)
        




def main():
    """Main function to run the Gaussian simple example and compare results."""
    N, P = 8000, 200
    rng_seed = 10
    snr = 2
    method = 'ridge_cv'
    mixing_dimension = None
    simple_example = False
    suppresion_strength = 0.8
    print("\n" + "="*70)
    print(f"Experiment 1: snr = {snr}")
    vp_results, pid_results, mi_results = test_suppresion(N, P,suppresion_strength ,rng_seed, simple_example=simple_example, snr=1, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results)
    #plot_pid_results(pid_results=pid_results, sub_title="Experiment 2")
    #plot_pid_results(mi_results=mi_results, sub_title="Experiment 2")

    print("\n" + "="*70)
    print(f"Experiment 2: snr = {snr} multivariate gaussian with different seeds")
    vp_results, pid_results, mi_results = test_suppresion(N, P,suppresion_strength ,rng_seed+2, simple_example=simple_example, snr=snr, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results)

    print("\n" + "="*70)
    snr = 5
    print(f"Experiment 3: snr = {snr}")
    vp_results, pid_results, mi_results = test_suppresion(N, P, suppresion_strength, rng_seed, simple_example=simple_example, snr=snr, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results)

    print("\n" + "="*70)
    print(f"Experiment 4: snr = {snr} multivariate gaussian with different seeds")
    vp_results, pid_results, mi_results = test_suppresion(N, P,suppresion_strength ,rng_seed+2, simple_example=simple_example, snr=snr, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results)

    print("\n" + "="*70)
    snr = 10
    print(f"Experiment 5: snr = {snr}")
    vp_results, pid_results, mi_results = test_suppresion(N, P, suppresion_strength, rng_seed, simple_example=simple_example, snr=snr, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results)

    print("\n" + "="*70)
    print(f"Experiment 6: snr = {snr} multivariate gaussian with different seeds")
    vp_results, pid_results, mi_results = test_suppresion(N, P,suppresion_strength ,rng_seed+1, simple_example=simple_example, snr=snr, method=method, mixing_dimension=mixing_dimension)
    compare_results(vp_results, pid_results)

    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()