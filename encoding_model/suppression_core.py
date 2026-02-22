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


def grid_search_suppression_analysis(
    features,
    reg_lh=None,
    reg_rh=None,
    suppression_strength_list=None,
    snr_list=None,
    mixing_dimension_list=None,
    rng_seed_list=None,
    hemisphere='both',
    suppresion_method='permutate',
    output_dir='./grid_search_results',
    grid_name='NoName',
    verbose=True
):
    """
    Perform a grid search over suppression analysis parameters using ridge regression.
    
    Args:
        features: Feature matrix (shape: [n_samples, n_features]).
        reg_lh: Trained regression model for left hemisphere.
        reg_rh: Trained regression model for right hemisphere.
        suppression_strength_list: List of suppression strengths to test.
        snr_list: List of SNR values to test.
        mixing_dimension_list: List of mixing dimensions to test.
        rng_seed_list: List of random seeds to test.
        hemisphere: Which hemisphere(s) to analyze ('left', 'right', or 'both').
        suppresion_method: Method to create suppression (default: 'permutate').
        output_dir: Directory to save results.
        grid_name: The name of the grid_search
        verbose: Whether to print progress messages.
        
    Returns:
        dict: Dictionary containing:
            - results_df: Comprehensive dataframe with all results
            - results_by_seed: Dictionary with results grouped by rng_seed
            - file_paths: List of saved file paths
    """
    # Set defaults
    suppression_strength_list = suppression_strength_list or [0.3, 0.5, 0.7]
    snr_list = snr_list or [1.0, 5.0, 10.0]
    mixing_dimension_list = mixing_dimension_list or [None, 50, 100]
    rng_seed_list = rng_seed_list or [42, 123, 456]
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 80)
        print("GRID SEARCH SUPPRESSION ANALYSIS")
        print("=" * 80)
        print(f"Parameters to test:")
        print(f"  Suppression strengths: {suppression_strength_list}")
        print(f"  SNR values: {snr_list}")
        print(f"  Mixing dimensions: {mixing_dimension_list}")
        print(f"  Random seeds: {rng_seed_list}")
        print(f"  Total combinations: {len(suppression_strength_list) * len(snr_list) * len(mixing_dimension_list) * len(rng_seed_list)}")
        print("=" * 80)
    
    # Initialize containers
    all_results = []
    results_by_seed = {}
    file_paths = []
    alphas = np.logspace(-3, 3, 50)
    
    # Master output file for all seeds
    master_output_file = output_dir / f'grid_search_{grid_name}.csv'
    
    # Load existing results if file exists
    if master_output_file.exists():
        if verbose:
            print(f"\nFound existing results file: {master_output_file}")
            print("Loading and continuing from previous results...")
        all_results = [pd.read_csv(master_output_file)]
        if verbose:
            print(f"Loaded {len(all_results[0])} existing rows")
    
    # Get total combinations for progress tracking
    total_combos = (len(suppression_strength_list) * len(snr_list) * 
                    len(mixing_dimension_list) * len(rng_seed_list))
    current_combo = 0
    
    # Grid search loop
    for rng_seed in rng_seed_list:
        seed_results = []
        
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Processing RNG Seed: {rng_seed}")
            print(f"{'=' * 80}")
        
        for suppression_strength in suppression_strength_list:
            for snr in snr_list:
                for mixing_dimension in mixing_dimension_list:
                    current_combo += 1
                    
                    if verbose:
                        print(f"\n[{current_combo}/{total_combos}] SS={suppression_strength}, SNR={snr}, MD={mixing_dimension}, Seed={rng_seed}")
                    
                    try:
                        # Run pipeline with ridge regression only
                        results = suppression_analysis_pipeline(
                            features=features,
                            reg_lh=reg_lh,
                            reg_rh=reg_rh,
                            hemisphere=hemisphere,
                            suppression_strength=suppression_strength,
                            snr=snr,
                            mixing_dimension=mixing_dimension,
                            suppresion_method=suppresion_method,
                            analysis_methods=['ridge_cv'],  # Only ridge regression
                            rng_seed=rng_seed,
                            alphas=alphas
                        )
                        
                        # Extract results from dict and convert to DataFrame
                        result_dict = results['commonality_results']['lh']['ridge_cv'].copy() if 'lh' in results['commonality_results'] else results['commonality_results']['rh']['ridge_cv'].copy()
                        
                        # Add grid search parameters
                        result_dict['suppression_strength'] = suppression_strength
                        result_dict['snr'] = snr
                        result_dict['mixing_dimension'] = mixing_dimension
                        result_dict['rng_seed'] = rng_seed
                        
                        # Convert to DataFrame with single row
                        results_df = pd.DataFrame([result_dict])
                        
                        # Append to lists
                        all_results.append(results_df)
                        seed_results.append(results_df)
                        
                        if verbose:
                            print(f"  ✓ Completed")
                        
                    except Exception as e:
                        if verbose:
                            print(f"  ✗ Error: {str(e)}")
                        continue
                
                # After each mixing_dimension loop, update the master CSV file with all results so far
                if all_results:
                    master_df = pd.concat(all_results, ignore_index=True)
                    
                    # Save/update master results file
                    master_df.to_csv(master_output_file, index=False)
                    
                    if verbose:
                        print(f"\n  ✓ Master CSV updated: {master_output_file}")
                        print(f"    Total rows so far: {len(master_df)}")
        
        # Add to results_by_seed for reference
        if seed_results:
            results_by_seed[rng_seed] = seed_results
    
    # Final results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        file_paths.append(str(master_output_file))
        
        if verbose:
            print(f"\n{'=' * 80}")
            print("GRID SEARCH COMPLETED")
            print(f"{'=' * 80}")
            print(f"Results saved to: {master_output_file}")
            print(f"Total parameter combinations tested: {len(final_df)}")
            print(f"\nResults Summary:")
            print(final_df.groupby('rng_seed').size())
        
        return {
            'results_df': final_df,
            'results_by_seed': results_by_seed,
            'file_paths': file_paths,
            'output_dir': str(output_dir)
        }
    else:
        if verbose:
            print("\n✗ No results generated!")
        return {
            'results_df': None,
            'results_by_seed': {},
            'file_paths': file_paths,
            'output_dir': str(output_dir)
        }

