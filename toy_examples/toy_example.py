import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from encoding_model.commonality import commonality_analysis
from Partial_Information_Decomposition.PID_util import diagnostic_plots
from encoding_model.suppression_core import permutate_models





def unq2_zero_with_red_unq1_syn(rng, n, p, noise_std=0.9):
    """
    Continuous Gaussian-like example where theoretically:

        unq2 = 0
        redundancy > 0
        unq1 > 0
        synergy > 0

    The output dimensions are 3p:
        block 1: redundant information
        block 2: unique information for X1
        block 3: synergistic/suppressor information
    """

    # -------------------------
    # 1. Redundant block
    # -------------------------
    print(f"Generating unq2=0 example with concatenation")
    print('\n Function name: unq2_zero_with_red_unq1_syn')
    R = rng.standard_normal((n, p))

    noise_r_y = noise_std * rng.standard_normal((n, p))
    noise_r_x1 = noise_std * rng.standard_normal((n, p))
    noise_r_x2 = noise_std * rng.standard_normal((n, p))

    y_red = R + noise_r_y

    # X1 observes R
    X1_red = R + noise_r_x1

    # X2 is a degraded version of X1, not an independent new measurement of R
    X2_red = X1_red + noise_r_x2

    # -------------------------
    # 2. Unique-to-X1 block
    # -------------------------
    U = rng.standard_normal((n, p))

    noise_u_y = noise_std * rng.standard_normal((n, p))
    noise_u_x1 = noise_std * rng.standard_normal((n, p))

    y_unq1 = U 
    X1_unq1 = U 

    # X2 has no U signal
    X2_unq1 = noise_std * rng.standard_normal((n, p))

    # -------------------------
    # 3. Synergy / suppressor block
    # -------------------------
    A = rng.standard_normal((n, p))
    N = rng.standard_normal((n, p))

    #noise_s_y = noise_std * rng.standard_normal((n, p))
    #noise_s_x1 = noise_std * rng.standard_normal((n, p))
    #noise_s_x2 = noise_std * rng.standard_normal((n, p))

    y_syn = A 

    # X1 contains signal + nuisance
    X1_syn = A + N 

    # X2 contains only the nuisance/suppressor variable
    X2_syn = N 
    # -------------------------
    # Concatenate independent blocks
    # -------------------------
    y = np.hstack([y_red, y_unq1, y_syn])
    y += noise_std * rng.standard_normal(y.shape)  # Add noise to the target
    
    X_M1 = np.hstack([X1_red, X1_unq1, X1_syn])
    X_M2 = np.hstack([X2_red, X2_unq1, X2_syn])

    return X_M1, X_M2, y

def run_experiment(
    rng,
    suppresion_strength,
    mode="permuted",
    n=1024,
    p=100,
    mixing_dimension=None,
    snr=10.0,
    method='standard',
    show_diagnostic_plots=False,
):
    """
    Run commonality analysis experiment.
    
    Args:
        rng: Random number generator
        n: Number of samples
        p: Number of features per source
        mixing_dimension: If not None, apply a mixing matrix with this dimension to entangle features
        snr: Signal-to-noise ratio (signal_std / noise_std)
        method: Which R² computation to use: 'standard', 'ols_cv', or 'ridge_cv'
        
    Returns:
        dict: Commonality analysis results
    """
    real_features = rng.standard_normal((n, p))

    # Target: only real features contribute
    betas = rng.standard_normal((p, p))
    signal = real_features @ betas
    noise_std = np.std(signal) / snr
    y_real  = signal + noise_std * rng.standard_normal((n,p))

    if mode == "simple":
        common_noise = noise_std * rng.standard_normal((n,p))
        X_M1 = signal + common_noise
        X_M2 = common_noise
    elif mode == "permuted":
        X_M1, X_M2 = permutate_models(
            rng,
            real_features,
            suppression_strength=suppresion_strength,
        )
    elif mode == 'only_unq2_zero':
        X_M1, X_M2, y_real = unq2_zero(rng, n, p, noise_std)

    elif mode == 'unq2_zero_with_red_unq1_syn':
        X_M1, X_M2, y_real = unq2_zero_with_red_unq1_syn(rng, n, p, noise_std)
    # else: 
    #     X_M1, X_M2 = half_permute(rng, signal, suppression_strength=0.3)
    # make sure joint covariance matrix is not singular
    #sing_report = singularity_report(X_M1, X_M2, y_real)
    #assert not sing_report["M1+M2+Y"]["is_singular"], \
        "Joint covariance matrix is singular or ill-conditioned!"

    if mixing_dimension is not None:
        # Create mixed features: entangle real and spurious with a mixing matrix
        mixing_matrix_M1 = rng.standard_normal((X_M1.shape[1], mixing_dimension))
        X_M1 = X_M1 @ mixing_matrix_M1
        mixing_matrix_M2 = rng.standard_normal((X_M2.shape[1], mixing_dimension))
        X_M2 = X_M2 @ mixing_matrix_M2
    # plot correlation matrices
    if show_diagnostic_plots:
        diagnostic_plots(X_M1, X_M2, y_real, method, mixing_dimension)

    

    # make sure joint covariance matrix is not singular
    #sing_report = singularity_report(X_M1, X_M2, y_real)
    #assert not sing_report["M1+M2+Y"]["is_singular"], \
        "Joint covariance matrix is singular or ill-conditioned!"

    # Commonality analysis
    decomp = commonality_analysis(X_M1, X_M2, y_real, method=method)


    return decomp ,{'X_M1'  :X_M1,'X_M2':X_M2,'y':y_real}



# =============================================================================
# 2x3 Factorial Design: SNR (low/high) x Mixing (none/invertible/lossy)
# =============================================================================

def run_ridge_toy_method(rng_seed,n, p, mixing_dimension, snr):
    """Run the ridge-CV analysis for this specialized toy example."""
    decomp_dict = {}
    for method in ['ridge_cv']: #,'ols_cv','standard']:
        print(f"\n--- {method.upper()} ---")
        rng = np.random.default_rng(seed=rng_seed)
        decomp,_ = run_experiment(
            rng,
            suppresion_strength=0.5,
            n=n,
            p=p,
            mixing_dimension=mixing_dimension,
            snr=snr,
            method=method,
        )
    
    for key, value in decomp.items():
        if key not in decomp_dict:
            decomp_dict[key] = value
    return decomp_dict

def main():
    """Run the 2x3 factorial experiment design."""
    # Common parameters
    N, P, SEED = 1000, 200, 42

    # =============================================================================
    # LOW SNR experiments (SNR = 1.0)
    # =============================================================================

    print("\n" + "="*70)
    print("Experiment 1: LOW SNR + NO MIXING")
    print("="*70)
    run_ridge_toy_method(SEED, N, P, mixing_dimension=None, snr=0.9)

    print("\n" + "="*70)
    print("Experiment 2: LOW SNR + INVERTIBLE MIXING (200→200)")
    print("="*70)
    run_ridge_toy_method(SEED, N, P, mixing_dimension=200, snr=1.0)

    print("\n" + "="*70)
    print("Experiment 3: LOW SNR + LOSSY MIXING (200→100)")
    print("="*70)
    run_ridge_toy_method(SEED, N, P, mixing_dimension=100, snr=1.0)

    # =============================================================================
    # HIGH SNR experiments (SNR = 10.0)
    # =============================================================================

    print("\n" + "="*70)
    print("Experiment 4: HIGH SNR + NO MIXING")
    print("="*70)
    run_ridge_toy_method(SEED, N, P, mixing_dimension=None, snr=10.0)

    print("\n" + "="*70)
    print("Experiment 5: HIGH SNR + INVERTIBLE MIXING (200→200)")
    print("="*70)
    run_ridge_toy_method(SEED, N, P, mixing_dimension=200, snr=10.0)

    print("\n" + "="*70)
    print("Experiment 6: HIGH SNR + LOSSY MIXING (200→100)")
    print("="*70)
    run_ridge_toy_method(SEED, N, P, mixing_dimension=100, snr=10.0)


def plot_components(comp_dict, title="Variance Decomposition"):
    """Plot bar chart for component dictionary."""
    keys = list(comp_dict.keys())
    values = list(comp_dict.values())
    
    # Create a colormap with different colors for each bar
    colors = plt.cm.Set3(np.linspace(0, 1, len(values)))
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(keys)), values, color=colors)
    plt.xticks(range(len(keys)), keys, rotation=45, ha='right',fontsize=18)
    plt.ylabel('Value',fontsize=16)
    plt.title(title)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/figures_/variance_decomposition_{title}.pdf", transparent=True)


def gauss_simple_example():
    """Run a simple Gaussian example with no mixing deminsion each experiement has different random seeds."""
    N, P = 500000, 50
    rng_seed = 42 
    noise_seed = 24

    print("\n" + "="*70)
    print(f"Simple Example with RNG seed {rng_seed} and Noise seed {noise_seed}")
    print("="*70)
    rng = np.random.default_rng(seed=rng_seed)
    comp_dict = run_ridge_toy_method(rng, n=N, p=P, mixing_dimension=30, snr=0.9)

    # Plot the results
    plot_components(comp_dict, f"Variance Decomposition Example 2")
if __name__ == '__main__':
    main()
    #gauss_simple_example()
