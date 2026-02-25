"""
Example script demonstrating the use of the suppression analysis pipeline.

This script shows how to:
1. Load or generate model features
2. Train regression models (or use existing ones)
3. Create predictions as targets
4. Generate suppression models
5. Perform commonality analysis
"""

import numpy as np
import sys
from pathlib import Path

# Add root to path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from sklearn.linear_model import RidgeCV
from encoding_model.suppression_core import suppression_analysis_pipeline


def example_with_synthetic_data():
    """Example using synthetic data to demonstrate the pipeline."""
    
    print("=" * 80)
    print("SUPPRESSION ANALYSIS PIPELINE - SYNTHETIC DATA EXAMPLE")
    print("=" * 80)
    
    # Generate synthetic data
    n_samples = 1000
    n_features = 100
    n_voxels_lh = 50
    n_voxels_rh = 50
    
    print("\nGenerating synthetic data...")
    rng = np.random.default_rng(42)
    
    # Create features (e.g., DNN activations)
    features = rng.standard_normal((n_samples, n_features))
    
    # Create synthetic fMRI data
    true_weights_lh = rng.standard_normal((n_features, n_voxels_lh))
    true_weights_rh = rng.standard_normal((n_features, n_voxels_rh))
    
    fmri_lh = features @ true_weights_lh + 0.1 * rng.standard_normal((n_samples, n_voxels_lh))
    fmri_rh = features @ true_weights_rh + 0.1 * rng.standard_normal((n_samples, n_voxels_rh))
    
    print(f"  Features shape: {features.shape}")
    print(f"  fMRI LH shape: {fmri_lh.shape}")
    print(f"  fMRI RH shape: {fmri_rh.shape}")
    
    # Train regression models
    print("\nTraining regression models...")
    alphas = np.logspace(-3, 3, 50)
    
    reg_lh = RidgeCV(alphas=alphas, fit_intercept=True, scoring='r2', cv=5)
    reg_lh.fit(features, fmri_lh)
    print(f"  LH model trained. Best alpha: {reg_lh.alpha_:.4f}")
    
    reg_rh = RidgeCV(alphas=alphas, fit_intercept=True, scoring='r2', cv=5)
    reg_rh.fit(features, fmri_rh)
    print(f"  RH model trained. Best alpha: {reg_rh.alpha_:.4f}")
    
    # Run the complete pipeline
    print("\n" + "=" * 80)
    print("RUNNING SUPPRESSION ANALYSIS PIPELINE")
    print("=" * 80)
    
    results = suppression_analysis_pipeline(
        features=features,
        reg_lh=reg_lh,
        reg_rh=reg_rh,
        hemisphere='both',
        suppression_strength=0.5,
        snr=5.0,
        mixing_dimension=50,
        suppresion_method='permutate',
        analysis_methods=['standard', 'ridge_cv'],
        rng_seed=42,
        alphas=alphas
    )
    
    # Access and display results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    for hemi in results['commonality_results'].keys():
        print(f"\n{hemi.upper()} Hemisphere Results:")
        print("-" * 80)
        
        for method in results['commonality_results'][hemi].keys():
            print(f"\n  {method.upper()} method:")
            result = results['commonality_results'][hemi][method]
            
            print(f"    R² Model A (M1):     {result['R²_A']:.4f}")
            print(f"    R² Model B (M2):     {result['R²_B']:.4f}")
            print(f"    R² Combined (AB):    {result['R²_AB']:.4f}")
            print(f"    Unique A:            {result['unique_A']:.4f}")
            print(f"    Unique B:            {result['unique_B']:.4f}")
            print(f"    Common:              {result['common']:.4f}")
            print(f"    Unexplained:         {result['unexplained']:.4f}")
    
    return results


def example_with_pretrained_models():
    """Example showing how to use the pipeline with pre-trained models."""
    
    print("=" * 80)
    print("SUPPRESSION ANALYSIS PIPELINE - PRE-TRAINED MODELS EXAMPLE")
    print("=" * 80)
    
    # This example shows the structure when loading pre-trained models
    # In practice, you would load these from saved files
    
    n_samples = 500
    n_features = 150
    
    print("\nGenerating test features...")
    rng = np.random.default_rng(123)
    features = rng.standard_normal((n_samples, n_features))
    
    # Simulate loading a pre-trained model
    # In practice: reg_lh = joblib.load('path/to/saved/model_lh.pkl')
    from sklearn.linear_model import LinearRegression
    reg_lh = LinearRegression()
    fake_weights = rng.standard_normal((n_features, 100))
    fake_fmri = features @ fake_weights
    reg_lh.fit(features, fake_fmri)
    
    print("  Loaded pre-trained model (simulated)")
    
    # Run pipeline for single hemisphere
    results = suppression_analysis_pipeline(
        features=features,
        reg_lh=reg_lh,
        reg_rh=None,  # Only analyzing left hemisphere
        hemisphere='left',
        suppression_strength=0.3,
        snr=10.0,
        mixing_dimension=None,  # No mixing
        analysis_methods=['standard'],  # Just one method
        rng_seed=123
    )
    
    print("\nPipeline completed for left hemisphere only.")
    return results


if __name__ == "__main__":
    # Run the synthetic data example
    print("\n\n")
    results_synthetic = example_with_synthetic_data()
    
    print("\n\n")
    print("*" * 80)
    print("\n\n")
    
    # Run the pre-trained model example
    results_pretrained = example_with_pretrained_models()
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 80)
