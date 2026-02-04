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
    if rng is not None: #If we need to create a permutated model
        features_idx = rng.permutation(features.shape[1])
        features_idx = features_idx[:n_features]
        
    else: #Just to lower number of features
        features_idx = np.arange(n_features)

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


def suppression_analysis_pipeline(
    features,
    reg_lh=None,
    reg_rh=None,
    hemisphere='both',
    suppression_strength=0.5,
    snr=1.0,
    mixing_dimension=None,
    suppresion_method='permutate',
    analysis_methods=['standard', 'ols_cv', 'ridge_cv'],
    rng_seed=None,
    alphas=None
):
    """
    Complete pipeline that takes model features, creates predictions via regression,
    generates suppression models, and performs commonality analysis.
    
    Args:
        features: Feature matrix (shape: [n_samples, n_features]).
        reg_lh: Trained regression model for left hemisphere (optional, creates predictions if provided).
        reg_rh: Trained regression model for right hemisphere (optional, creates predictions if provided).
        hemisphere: Which hemisphere to analyze ('left', 'right', or 'both').
        suppression_strength: Proportion of features to suppress (between 0 and 1).
        snr: Signal-to-noise ratio for adding noise to the target.
        mixing_dimension: Dimension to which features are mixed (if None, no mixing is applied).
        suppresion_method: Method to create suppression ('permutate' is default).
        analysis_methods: List of methods to use for commonality analysis.
        rng_seed: Random seed for reproducibility (if None, uses random).
        alphas: Alpha values for ridge regression (if None, uses default range).
        
    Returns:
        dict: Dictionary containing:
            - predictions: Dictionary with 'lh' and/or 'rh' predictions
            - suppression_models: Dictionary with X_M1, X_M2, target for each hemisphere
            - commonality_results: Dictionary with analysis results for each hemisphere and method
    """
    # Initialize RNG
    rng = np.random.default_rng(rng_seed) if rng_seed is not None else np.random.default_rng()
    
    # Initialize output dictionary
    pipeline_results = {
        'predictions': {},
        'suppression_models': {},
        'commonality_results': {}
    }
    
    # Step 1: Create predictions if regression models are provided
    print("=" * 70)
    print("STEP 1: Creating predictions from regression models")
    print("=" * 70)
    
    hemispheres_to_process = []
    if hemisphere in ['left', 'both'] and reg_lh is not None:
        hemispheres_to_process.append(('lh', reg_lh))
    if hemisphere in ['right', 'both'] and reg_rh is not None:
        hemispheres_to_process.append(('rh', reg_rh))
    
    if not hemispheres_to_process:
        raise ValueError("No regression models provided. Need at least one of reg_lh or reg_rh.")
    
    for hemi_name, reg_model in hemispheres_to_process:
        predictions = reg_model.predict(features)
        pipeline_results['predictions'][hemi_name] = predictions
        print(f"  {hemi_name.upper()} predictions shape: {predictions.shape}")
    
    # Step 2: Create suppression models for each hemisphere
    print("\n" + "=" * 70)
    print("STEP 2: Creating suppression models")
    print("=" * 70)
    print(f"  Suppression strength: {suppression_strength}")
    print(f"  SNR: {snr}")
    print(f"  Mixing dimension: {mixing_dimension}")
    print(f"  Method: {suppresion_method}")
    
    for hemi_name in pipeline_results['predictions'].keys():
        signal = pipeline_results['predictions'][hemi_name]
        
        print(f"\n  Creating suppression model for {hemi_name.upper()}...")
        X_M1, X_M2, target = create_supression_model(
            rng=rng,
            signal=signal,
            suppresion_method=suppresion_method,
            features=features,
            suppression_strength=suppression_strength,
            snr=snr,
            mixing_dimension=mixing_dimension
        )
        
        pipeline_results['suppression_models'][hemi_name] = {
            'X_M1': X_M1,
            'X_M2': X_M2,
            'target': target,
            'signal': signal
        }
        
        print(f"    X_M1 shape: {X_M1.shape}")
        print(f"    X_M2 shape: {X_M2.shape}")
        print(f"    Target shape: {target.shape}")
    
    # Step 3: Perform commonality analysis for each hemisphere
    print("\n" + "=" * 70)
    print("STEP 3: Performing commonality analysis")
    print("=" * 70)
    
    for hemi_name in pipeline_results['suppression_models'].keys():
        X_M1 = pipeline_results['suppression_models'][hemi_name]['X_M1']
        X_M2 = pipeline_results['suppression_models'][hemi_name]['X_M2']
        target = pipeline_results['suppression_models'][hemi_name]['target']
        
        pipeline_results['commonality_results'][hemi_name] = {}
        
        print(f"\n  Analyzing {hemi_name.upper()} hemisphere:")
        print("-" * 70)
        
        for method in analysis_methods:
            print(f"\n    Method: {method.upper()}")
            
            analysis_result = commonality_analysis(
                X_M1, X_M2, target,
                method=method,
                alphas=alphas,
                snr=snr
            )
            
            pipeline_results['commonality_results'][hemi_name][method] = analysis_result
            
            # Print results as DataFrame
            df = pd.DataFrame.from_dict(analysis_result, orient='index', columns=['value'])
            print(df.to_string())
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Processed hemispheres: {list(pipeline_results['predictions'].keys())}")
    print(f"Analysis methods used: {analysis_methods}")
    
    return pipeline_results

