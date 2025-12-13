import numpy as np
from sklearn.linear_model import RidgeCV, LinearRegression


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
    return ridge_cv.best_score_


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
    
    return ridge_cv.best_score_


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
    return model.score(X, y)


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
    r2_A = compute_r2_fn(features_A, target)
    r2_B = compute_r2_fn(features_B, target)
    r2_AB = compute_r2_fn(features_AB, target)
    
    # Commonality analysis decomposition
    unique_A = (r2_AB - r2_B) * var_y
    unique_B = (r2_AB - r2_A) * var_y
    common_AB = (r2_A + r2_B - r2_AB) * var_y
    unexplained = (1 - r2_AB) * var_y
    
    return {
        'R²_A': r2_A,
        'R²_B': r2_B,
        'R²_AB': r2_AB,
        'unique_A': unique_A,
        'unique_B': unique_B,
        'common': common_AB,
        'unexplained': unexplained
    }

def run_experiment(rng, n=1024, p=100, mixing_dimension=None, snr=10.0, method='standard'):
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
    betas = rng.standard_normal(p)
    signal = real_features @ betas
    noise_std = np.std(signal) / snr
    y_real = signal + noise_std * rng.standard_normal(n)
    
    X_M1 = np.hstack([real_features, shuffled_spurious])
    X_M2 = np.hstack([shuffled_real, shuffled_spurious])




    if mixing_dimension is not None:
        # Create mixed features: entangle real and spurious with a mixing matrix
        mixing_matrix_M1 = rng.standard_normal((2 * p, mixing_dimension))
        X_M1 = X_M1 @ mixing_matrix_M1
        mixing_matrix_M2 = rng.standard_normal((2 * p, mixing_dimension))
        X_M2 = X_M2 @ mixing_matrix_M2
    
    # Commonality analysis
    decomp = commonality_analysis(X_M1, X_M2, y_real, method=method)
    
    # Print results
    print(f"{method} analysis of target (y_real):")
    for key, value in decomp.items():
        print(f"- {key}: {value:.4f}")
    
    # Verify sum (only variance components, not R² values)
    # Note: CV version may not sum exactly due to negative R² being possible
    if method == 'standard':
        total_variance = np.var(y_real, ddof=1)
        sum_of_components = decomp['unique_A'] + decomp['unique_B'] + decomp['common'] + decomp['unexplained']
        assert np.isclose(total_variance, sum_of_components), \
            "Decomposed components do not sum to total variance!"
    
    return decomp



# =============================================================================
# 2x3 Factorial Design: SNR (low/high) x Mixing (none/invertible/lossy)
# =============================================================================

def run_all_methods(rng_seed, n, p, mixing_dimension, snr):
    """Run all three analysis methods with the same random seed."""
    for method in ['standard', 'ols_cv', 'ridge_cv']:
        print(f"\n--- {method.upper()} ---")
        rng = np.random.default_rng(seed=rng_seed)
        run_experiment(rng, n=n, p=p, mixing_dimension=mixing_dimension, snr=snr, method=method)


def main():
    """Run the 2x3 factorial experiment design."""
    # Common parameters
    N, P, SEED = 2000, 100, 42

    # =============================================================================
    # LOW SNR experiments (SNR = 1.0)
    # =============================================================================

    print("\n" + "="*70)
    print("Experiment 1: LOW SNR + NO MIXING")
    print("="*70)
    run_all_methods(SEED, N, P, mixing_dimension=None, snr=1.0)

    print("\n" + "="*70)
    print("Experiment 2: LOW SNR + INVERTIBLE MIXING (200→200)")
    print("="*70)
    run_all_methods(SEED, N, P, mixing_dimension=200, snr=1.0)

    print("\n" + "="*70)
    print("Experiment 3: LOW SNR + LOSSY MIXING (200→100)")
    print("="*70)
    run_all_methods(SEED, N, P, mixing_dimension=100, snr=1.0)

    # =============================================================================
    # HIGH SNR experiments (SNR = 10.0)
    # =============================================================================

    print("\n" + "="*70)
    print("Experiment 4: HIGH SNR + NO MIXING")
    print("="*70)
    run_all_methods(SEED, N, P, mixing_dimension=None, snr=10.0)

    print("\n" + "="*70)
    print("Experiment 5: HIGH SNR + INVERTIBLE MIXING (200→200)")
    print("="*70)
    run_all_methods(SEED, N, P, mixing_dimension=200, snr=10.0)

    print("\n" + "="*70)
    print("Experiment 6: HIGH SNR + LOSSY MIXING (200→100)")
    print("="*70)
    run_all_methods(SEED, N, P, mixing_dimension=100, snr=10.0)


if __name__ == '__main__':
    main()
