import numpy as np
from sklearn.linear_model import RidgeCV, LinearRegression
import matplotlib.pyplot as plt

def correlation_matrix(X):
    """Compute the correlation matrix of the columns of X."""
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    stddev = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(stddev, stddev)
    return corr_matrix

def diagnostic_plots(X_M1, X_M2, y_real, method, mixing_dimension):
    def cross_correlation(X, Y):
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        n = Xc.shape[0] - 1
        cov = (Xc.T @ Yc) / n
        sx = np.sqrt(np.diag((Xc.T @ Xc) / n))
        sy = np.sqrt(np.diag((Yc.T @ Yc) / n))
        with np.errstate(divide='ignore', invalid='ignore'):
            return cov / np.outer(sx, sy)

    blocks, labels = [X_M1, X_M2, y_real], ["M1", "M2", "Y"]
    counts = [b.shape[1] for b in blocks]
    fig = plt.figure(figsize=(9, 9))
    gs = plt.GridSpec(3, 3, width_ratios=counts, height_ratios=counts, wspace=0.05, hspace=0.05)
    axes, im = [], None
    for i in range(3):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            corr = cross_correlation(blocks[i], blocks[j])
            im = ax.imshow(corr, cmap='bwr', vmin=-1, vmax=1, aspect='auto')
            max_text = "max: n/a" if np.all(np.isnan(corr)) else f"max: {np.nanmax(corr):.2f}"
            ax.text(0.02, 0.98, max_text, transform=ax.transAxes, ha='left', va='top', fontsize=8,
                    color='black', bbox={'boxstyle': 'round,pad=0.2', 'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none'})
            if i == 0:
                ax.set_title(labels[j])
            if j == 0:
                ax.set_ylabel(labels[i])
            ax.set_xticks([]); ax.set_yticks([])
            axes.append(ax)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02).set_label('Correlation Coefficient')
    fig.suptitle(f'Correlation Matrix - Method: {method}, Mixing Dim: {mixing_dimension}')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def compute_ols_cv_r2(X, y):
    """
    Compute cross-validated R² using leave-one-out cross-validation.
    
    Uses RidgeCV with near-zero regularization (alpha=1e-16) which is
    effectively OLS but leverages the efficient GCV formula.
    
    Args:
        X (np.ndarray): Design matrix WITHOUT intercept (shape: [n, p]).
        y (np.ndarray): Target variable (shape: [n,]).
        
    Returns:
        float: Cross-validated R² (can be negative if model overfits badly)
    """
    ridge_cv = RidgeCV(alphas=[1e-16], fit_intercept=True, scoring='r2', cv=None)
    ridge_cv.fit(X, y)
    return ridge_cv,ridge_cv.best_score_


def compute_ridge_cv_r2(X, y, alphas=None):
    """
    Compute cross-validated R² using RidgeCV with efficient LOO cross-validation.
    
    RidgeCV uses generalized cross-validation (GCV) which is an efficient 
    approximation to leave-one-out CV for ridge regression.
    
    Args:
        X (np.ndarray): Design matrix WITHOUT intercept (shape: [n, p]).
        y (np.ndarray): Target variable (shape: [n,]).
        alphas (array-like, optional): Array of alpha values to try.
            Defaults to DEFAULT_RIDGE_ALPHAS.
        
    Returns:
        float: Best cross-validated R² across all alpha values.
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 50)
    
    # RidgeCV with leave-one-out CV (efficient GCV approximation)
    # cv=None means use efficient LOO via GCV
    ridge_cv = RidgeCV(alphas=alphas, fit_intercept=True, scoring='r2', cv=None)
    ridge_cv.fit(X, y)
    
    return ridge_cv,ridge_cv.best_score_


def compute_r2(X, y):
    """
    Compute in-sample R² for OLS regression.
    
    Args:
        X (np.ndarray): Design matrix WITHOUT intercept (shape: [n, p]).
        y (np.ndarray): Target variable (shape: [n,]).
        
    Returns:
        float: In-sample R².
    """
    model = LinearRegression()
    model.fit(X, y)
    return model,model.score(X, y)


def commonality_analysis(features_A, features_B, target, method='standard', alphas=None):
    """
    Decomposes the variance of the target variable into contributions from features_A and features_B.
    
    This is done using commonality analysis, which does not assume uncorrelated sources.
    Supports three methods:
    - 'standard': In-sample R² (prone to overfitting)
    - 'ols_cv': Cross-validated R² using PRESS residuals
    - 'ridge_cv': Cross-validated R² using RidgeCV with GCV
    
    Args:
        features_A (np.ndarray): Feature matrix A (shape: [n, p_A] or [n,] for 1D).
        features_B (np.ndarray): Feature matrix B (shape: [n, p_B] or [n,] for 1D).
        target (np.ndarray): Target variable (shape: [n,]).
        method (str): Which R² computation method to use: 'standard', 'ols_cv', or 'ridge_cv'.
        alphas (array-like, optional): Alpha values for RidgeCV (only used if method='ridge_cv').
        
    Returns:
        dict: A dictionary with R² values and variance decomposition.
    """
    n = len(target)
    
    # Ensure 2D
    if features_A.ndim == 1:
        features_A = features_A.reshape(-1, 1)
    if features_B.ndim == 1:
        features_B = features_B.reshape(-1, 1)
    
    # Total sum of squares (using N-1 for unbiased sample variance)
    tss = np.sum((target - target.mean())**2)
    var_y = tss / (n - 1)
    
    # Build combined features
    features_AB = np.hstack([features_A, features_B])
    
    # Select R² computation function based on method
    if method == 'standard':
        compute_r2_fn = compute_r2
    elif method == 'ols_cv':
        compute_r2_fn = compute_ols_cv_r2
    elif method == 'ridge_cv':
        compute_r2_fn = lambda X, y: compute_ridge_cv_r2(X, y, alphas)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'standard', 'ols_cv', or 'ridge_cv'.")
    
    # Compute R² for each model
    modelA,r2_A = compute_r2_fn(features_A, target)
    modelB,r2_B = compute_r2_fn(features_B, target)
    modelAB,r2_AB = compute_r2_fn(features_AB, target)
    
    # Commonality analysis decomposition
    unique_A = (r2_AB - r2_B) 
    unique_B = (r2_AB - r2_A) 
    common_AB = (r2_A + r2_B - r2_AB) 
    unexplained = (1 - r2_AB) 
    
    return {
        'R²_A': r2_A,
        'R²_B': r2_B,
        'R²_AB': r2_AB,
        'unique_A': unique_A,
        'unique_B': unique_B,
        'common': common_AB,
        'unexplained': unexplained,
        'betas_A': modelA.coef_ if hasattr(modelA, 'coef_') else None,
        'betas_B': modelB.coef_ if hasattr(modelB, 'coef_') else None,
        'betas_AB': modelAB.coef_ if hasattr(modelAB, 'coef_') else None
    }

def run_experiment(rng,noise_rng,simple_example = True, n=1024, p=100, mixing_dimension=None, snr=10.0, method='standard', show_diagnostic_plots=False):
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
    # Generate the four feature tensors
    real_features = rng.standard_normal((n, p))
    spurious_features = rng.standard_normal((n, p))
    rand_perm = rng.permutation(n)
    
    shuffled_real = real_features[rand_perm]
    shuffled_spurious = spurious_features[rand_perm]
    
    # Target: only real features contribute
    betas = rng.standard_normal((p, p))
    signal = real_features @ betas
    noise_std = np.std(signal) / snr
    signa_noise_rng = np.random.default_rng(seed=1000)
    y_real  = signal + noise_std * signa_noise_rng.standard_normal((n,p))
    
    if simple_example:
        noise = noise_std * noise_rng.standard_normal((n,p))
        X_M1 = y_real + noise
        X_M2 = noise
    else:
        X_M1 = np.hstack([real_features, shuffled_spurious])
        X_M2 = np.hstack([shuffled_real, shuffled_spurious])




    if mixing_dimension is not None:
        # Create mixed features: entangle real and spurious with a mixing matrix
        mixing_matrix_M1 = rng.standard_normal((X_M1.shape[1], mixing_dimension))
        X_M1 = X_M1 @ mixing_matrix_M1
        mixing_matrix_M2 = rng.standard_normal((X_M2.shape[1], mixing_dimension))
        X_M2 = X_M2 @ mixing_matrix_M2
    # plot correlation matrices
    if show_diagnostic_plots:
        diagnostic_plots(X_M1, X_M2, y_real, method, mixing_dimension)

    # Commonality analysis
    decomp = commonality_analysis(X_M1, X_M2, y_real, method=method)


    print(f"{method} analysis of target (y_real):")
    for key, value in decomp.items():
        if key.startswith('R²') or key in ['unique_A', 'unique_B', 'common', 'unexplained']:
            print(f"- {key}: {value:.4f}")
    return decomp ,{'X_M1'  :X_M1,'X_M2':X_M2,'y':y_real}
    # Verify sum (only variance components, not R² values)
    # Note: CV version may not sum exactly due to negative R² being possible
    # if method == 'standard':
    #     total_variance = np.var(y_real, ddof=1)
    #     sum_of_components = decomp['unique_A'] + decomp['unique_B'] + decomp['common'] + decomp['unexplained']
    #     assert np.isclose(total_variance, sum_of_components), \
    #         "Decomposed components do not sum to total variance!"
    
    return decomp



# =============================================================================
# 2x3 Factorial Design: SNR (low/high) x Mixing (none/invertible/lossy)
# =============================================================================

def run_all_methods(rng_seed,noise_seed,n, p, mixing_dimension, snr):
    """Run all three analysis methods with the same random seed."""
    coef_dict = {}
    decomp_dict = {}
    for method in ['ridge_cv']: #,'ols_cv','standard']:
        print(f"\n--- {method.upper()} ---")
        rng = np.random.default_rng(seed=rng_seed)
        noise_rng = np.random.default_rng(seed=noise_seed)
        decomp,_ = run_experiment(rng,noise_rng=noise_rng, n=n, p=p, mixing_dimension=mixing_dimension, snr=snr, method=method)
    
    for key, value in decomp.items():
        if key.startswith('betas'):
            coef_dict[method + '_' + key] = value
        if key not in decomp_dict:
            decomp_dict[key] = value
    return coef_dict,decomp_dict

def main():
    """Run the 2x3 factorial experiment design."""
    # Common parameters
    N, P, SEED,NOISE_SEED = 1000, 100, 42, 24

    # =============================================================================
    # LOW SNR experiments (SNR = 1.0)
    # =============================================================================

    print("\n" + "="*70)
    print("Experiment 1: LOW SNR + NO MIXING")
    print("="*70)
    run_all_methods(SEED, NOISE_SEED, N, P, mixing_dimension=None, snr=1.0)

    print("\n" + "="*70)
    print("Experiment 2: LOW SNR + INVERTIBLE MIXING (200→200)")
    print("="*70)
    run_all_methods(SEED, NOISE_SEED, N, P, mixing_dimension=200, snr=1.0)

    print("\n" + "="*70)
    print("Experiment 3: LOW SNR + LOSSY MIXING (200→100)")
    print("="*70)
    run_all_methods(SEED, NOISE_SEED, N, P, mixing_dimension=100, snr=1.0)

    # =============================================================================
    # HIGH SNR experiments (SNR = 10.0)
    # =============================================================================

    print("\n" + "="*70)
    print("Experiment 4: HIGH SNR + NO MIXING")
    print("="*70)
    run_all_methods(SEED, NOISE_SEED, N, P, mixing_dimension=None, snr=10.0)

    print("\n" + "="*70)
    print("Experiment 5: HIGH SNR + INVERTIBLE MIXING (200→200)")
    print("="*70)
    run_all_methods(SEED, NOISE_SEED, N, P, mixing_dimension=200, snr=10.0)

    print("\n" + "="*70)
    print("Experiment 6: HIGH SNR + LOSSY MIXING (200→100)")
    print("="*70)
    run_all_methods(SEED, NOISE_SEED, N, P, mixing_dimension=100, snr=10.0)


def plot_coefficients(coef_dict, title="Coefficient Comparison"):
    """Plot bar chart for coefficient dictionary."""
    keys = list(coef_dict.keys())
    # Compute mean value for each key (handle both arrays and scalars)
    # Special handling for ridge_cv_betas_AB to plot both items
    plot_keys = []
    plot_values = []
    
    for key, value in coef_dict.items():
        if value.shape[0] > 1:
            for i in range(value.shape[0]):
                plot_keys.append(f'{key}_predictor_{i}')
                plot_values.append(value[i])

        else:
            plot_keys.append(f'{key}_predictor')
            plot_values.append(np.mean(value))
            
    # Create a colormap with different colors for each bar
    colors = plt.cm.tab20(np.linspace(0, 1, len(plot_values)))
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(plot_keys)), plot_values, color=colors)
    plt.xticks(range(len(plot_keys)), plot_keys, rotation=45, ha='right')
    plt.ylabel('Mean Coefficient Value')
    plt.title(title)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/figures_/coefficient_comparison_{title}.pdf")


def plot_components(comp_dict, title="Variance Decomposition"):
    """Plot bar chart for component dictionary, excluding the last 3 keys (betas)."""
    # Filter out the beta keys
    filtered_dict = {k: v for k, v in comp_dict.items() if not k.startswith('betas')}
    
    keys = list(filtered_dict.keys())
    values = list(filtered_dict.values())
    
    # Create a colormap with different colors for each bar
    colors = plt.cm.Set3(np.linspace(0, 1, len(values)))
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(keys)), values, color=colors)
    plt.xticks(range(len(keys)), keys, rotation=45, ha='right')
    plt.ylabel('Value')
    plt.title(title)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/figures_/variance_decomposition_{title}.pdf")


def gauss_simple_example():
    """Run a simple Gaussian example with no mixing deminsion each experiement has different random seeds."""
    N, P = 1000, 1
    for rng_seed, noise_seed in [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10)]:
        print("\n" + "="*70)
        print(f"Simple Example with RNG seed {rng_seed} and Noise seed {noise_seed}")
        print("="*70)
        rng = np.random.default_rng(seed=rng_seed)
        noise_rng = np.random.default_rng(seed=noise_seed)
        coef_dict,comp_dict = run_all_methods(rng, noise_rng, n=N, p=P, mixing_dimension=None, snr=1)
        print("Coefficient details:")
        for key, value in coef_dict.items():
            print(f"for {key}, the coefficients are: {np.mean(value,axis=1) if len(value.shape) > 1 else value}")
        
        # Plot the results
        plot_coefficients(coef_dict, f"Coefficients - Seed {rng_seed}")
        plot_components(comp_dict, f"Variance Decomposition - Seed {rng_seed}")
if __name__ == '__main__':
    main()
    #gauss_simple_example()