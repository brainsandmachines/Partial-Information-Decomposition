import numpy as np
import torch
import pandas as pd
import sys
import joblib
from pathlib import Path
from encoding_model.encoding_utils import diagnostic_plots, singularity_report

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.linear_model import LinearRegression
from encoding_model.encoding_utils import compute_r2,compute_ols_cv_r2,compute_ridge_cv_r2





def create_predictions(reg_lh, reg_rh, features):
    """
    Create fMRI predictions using trained regression models.
    
    Args:
        reg_lh: Trained regression model for left hemisphere.
        reg_rh: Trained regression model for right hemisphere.
        features: Feature matrix (shape: [n_samples, n_features]).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Predicted fMRI responses for left and right hemispheres.
    """
    y_hat_lh = reg_lh.predict(features) if reg_lh is not None else None
    y_hat_rh = reg_rh.predict(features) if reg_rh is not None else None
    return y_hat_lh, y_hat_rh

def create_encoder(rng,features,target,n_features):
    """Create and train a linear regression encoder.
    mostly usable when the number of model features is larger than the number of samples.
    there for we randomly select a subset of features to use for training.
    
    Args:
        features: Feature matrix (shape: [n_samples, n_features]).
        target: Target matrix (shape: [n_samples, n_targets]).
        n_features: Number of features to use from the feature matrix.

    Returns:
        Tuple[LinearRegression, np.ndarray]: Trained regression model and used features.
    """
    assert n_features <= features.shape[1], "n_features must be less than or equal to the number of available features."


    model = LinearRegression()
    features_idx = rng.permutation(features.shape[1])
    features_idx = features_idx[:n_features]
    selected_features = features[:,features_idx]
    model.fit(selected_features, target)
    return model, selected_features


def permutate_models(rng,features,suppression_strength):
    n,p = features.shape
    
    n_real_dim = 1-suppression_strength
    real_dim = int(p*n_real_dim) 
    idx = rng.permutation(p)
    real_dim_indices = idx[:real_dim]
    spurious_dim_indices = idx[real_dim:]
    real_feature = features[:,real_dim_indices]
    spurious_feature = features[:,spurious_dim_indices]
    rand_perm = rng.permutation(n)

    shuffled_real = real_feature[rand_perm]
    shuffled_spurious = spurious_feature[rand_perm]

    X_M1 = np.hstack([real_feature, shuffled_spurious])
    X_M2 = np.hstack([shuffled_real, shuffled_spurious])

    return X_M1, X_M2

def partition_correlation(rng,features,target,permuation,suppression_strength,semi_partial=False):
    """This function creates two predictors X_M1 and X_M2 and removes any linear predictability between them.
    and remove linear predictability from the real features to X_M2.
    where X_M2 is the the model that shouldn't be with correlation to the target.
    WARNING: This function is useless right now!!!
    
    args:
        features: Feature matrix (shape: [n_samples, n_features]).
        target: Target matrix (shape: [n_samples, n_targets]).
        permuation: Whether to permute the features or not.
        suppression_strength: Proportion of features to suppress (between 0 and 1).
        semi_partial: If True, only remove predictability from X_M2 to features.
        
        returns:
        X_M1: Feature matrix for model 1.
        X_M2: Feature matrix for model 2.
            """
    n,p = features.shape
    
    if permuation:
        X_M1, X_M2 = permutate_models(rng,features,suppression_strength)

    else:
        noise_component = rng.standard_normal(features.shape) 
        X_M1 = features.copy()
        X_M2 = features.copy() + noise_component * suppression_strength

    # Remove any linear predictability of X_M1 from X_M2
    target_proj = LinearRegression(fit_intercept=True)
    target_proj.fit(target, X_M2)            # X = target, y = X_M2
    X_M2 = X_M2 - target_proj.predict(target)

    
    if semi_partial:
        return X_M1, X_M2
    
    # Residualize X_M2 with respect to X_M1 (remove linear predictability of X_M2 from X_M1)
    X_M2_proj = LinearRegression(fit_intercept=False)
    X_M2_proj.fit(X_M1, X_M2)
    X_M2_pred = X_M2_proj.predict(X_M1)
    X_M2 = X_M2 - X_M2_pred

    return X_M1, X_M2

def noise_component(rng,features,suppression_strength,permutation):
    """Create suppresion model using only noise component.
    args:
        rng: Random number generator.
        features: Feature matrix (shape: [n_samples, n_features]).
        suppression_strength: Proportion of features to suppress (between 0 and 1).
        permutation: Whether to permute the features or not.
    returns:
        X_M1: Feature matrix for model 1. (real feautres + noise)
        X_M2: Feature matrix for model 2. (noise only)"""
    noise_component = rng.standard_normal(features.shape)
    if permutation:
        X_M1, X_M2 = permutate_models(rng,features,suppression_strength)


    else:
        X_M1 = features.copy() 
        X_M2 = noise_component


    X_M1 = X_M1 + suppression_strength * (noise_component)
    

    return X_M1, X_M2
    

def create_supression_model(rng,signal,suppresion_method, features, suppression_strength=0.5,snr=1.0,mixing_dimension=None):
    """Create suppression model features X_M1 and X_M2 based on the given parameters.
    Args:
        rng: Random number generator.
        signal: The target signal (fMRI predictions).
        suppresion_method: Method to create suppression ('partition_correlation', 'semi_partition_correlation', or 'permutate').
        features: The original feature matrix.
        suppression_strength: Proportion of features to suppress (between 0 and 1).
        snr: Signal-to-noise ratio for adding noise to the target.
        mixing_dimension: Dimension to which features are mixed (if None, no mixing is applied).
    Returns:
        X_M1: Feature matrix for model 1.
        X_M2: Feature matrix for model 2.
        target: Noisy target signal.
    """
    

    std = np.std(signal)
    noise_std = std.item() / snr
    signal_dim1 , signal_dim2 = signal.shape[0], signal.shape[1]
    target = signal +  noise_std * rng.standard_normal((signal_dim1 , signal_dim2))


    X_M1, X_M2 = permutate_models(rng,features,suppression_strength)
        
    if mixing_dimension is not None:
        # Create mixed features: entangle real and spurious with a mixing matrix
        mixing_matrix_M1 = rng.standard_normal((X_M1.shape[1], mixing_dimension))
        X_M1 = X_M1 @ mixing_matrix_M1
        mixing_matrix_M2 = rng.standard_normal((X_M2.shape[1], mixing_dimension))
        X_M2 = X_M2 @ mixing_matrix_M2

        # make sure joint covariance matrix is not singular
    # sing_report = singularity_report(X_M1, X_M2, target)
    # assert not sing_report["M1+M2+Y"]["is_singular"], \
    #     "Joint covariance matrix is singular or ill-conditioned!"

    return X_M1, X_M2,target


def commonality_analysis(features_A, features_B, target, method='standard', alphas=None, snr=1.0):
    # Select R² computation function based on method
    if method == 'standard':
        compute_r2_fn = compute_r2
    elif method == 'ols_cv':
        compute_r2_fn = compute_ols_cv_r2
    elif method == 'ridge_cv':
        compute_r2_fn = lambda X, y: compute_ridge_cv_r2(X, y, alphas)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'standard', 'ols_cv', or 'ridge_cv'.")
    
    #Define joint model features
    features_AB = np.hstack([features_A, features_B])
    # Compute R² for each model
    r2_A = compute_r2_fn(features_A, target)
    r2_B = compute_r2_fn(features_B, target)
    r2_AB = compute_r2_fn(features_AB, target)
    
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
        'unexplained': unexplained
    }

def run_all_methods(rng_seed,suppresion_method ,mixing_dimension, snr,suppression_strength,models_and_features_dict=None):
    methods_outputs = {}
    """Run all three analysis methods with the same random seed."""
    X_M1,X_M2,target,signal,real_feature = models_and_features_dict['X_M1'],models_and_features_dict['X_M2'],models_and_features_dict['target'],models_and_features_dict['signal'],models_and_features_dict['real_feature']
    if X_M1 is None or X_M2 is None or target is None:
        print("Creating suppression model...")
        X_M1, X_M2,target = create_supression_model(rng=rng_seed,signal = signal,suppresion_method=suppresion_method,features=real_feature,suppression_strength=suppression_strength,mixing_dimension=mixing_dimension,snr=snr)
    for method in ['standard', 'ols_cv', 'ridge_cv']: #@['standard', 'ols_cv', 'ridge_cv']
        print(f"\n--- {method.upper()} ---")
        outputs = commonality_analysis(X_M1, X_M2, target, method=method)
        df = pd.DataFrame.from_dict(outputs, orient='index', columns=['value']) 
        print(df)
        methods_outputs[method] = outputs
    return methods_outputs