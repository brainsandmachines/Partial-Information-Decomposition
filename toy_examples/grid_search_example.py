"""
Example script demonstrating the grid search suppression analysis.

This script shows how to run a comprehensive grid search over multiple
parameters and save results organized by random seed.
"""

import numpy as np
import sys
from pathlib import Path

# Add root to path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from sklearn.linear_model import RidgeCV
from encoding_model.suppression_core import grid_search_suppression_analysis


def example_grid_search():
    """Example demonstrating grid search functionality."""
    
    print("=" * 80)
    print("GRID SEARCH EXAMPLE - SUPPRESSION ANALYSIS")
    print("=" * 80)
    
    # Generate synthetic data
    n_samples = 200
    n_features = 50
    n_voxels = 30
    
    print("\nGenerating synthetic data...")
    rng = np.random.default_rng(42)
    
    # Create features
    features = rng.standard_normal((n_samples, n_features))
    
    # Create synthetic fMRI data
    true_weights = rng.standard_normal((n_features, n_voxels))
    fmri = features @ true_weights + 0.1 * rng.standard_normal((n_samples, n_voxels))
    
    print(f"  Features shape: {features.shape}")
    print(f"  fMRI shape: {fmri.shape}")
    
    # Train regression model (using RidgeCV)
    print("\nTraining regression model...")
    alphas = np.logspace(-3, 3, 50)
    
    reg_model = RidgeCV(alphas=alphas, fit_intercept=True, scoring='r2', cv=5)
    reg_model.fit(features, fmri)
    print(f"  Model trained. Best alpha: {reg_model.alpha_:.4f}")
    
    # Run grid search
    print("\n" + "=" * 80)
    print("RUNNING GRID SEARCH")
    print("=" * 80)
    
    grid_search_results = grid_search_suppression_analysis(
        features=features,
        reg_lh=reg_model,
        reg_rh=None,
        suppression_strength_list=[0.3, 0.5, 0.7],
        snr_list=[1.0, 5.0, 10.0],
        mixing_dimension_list=[None, 30, 50],
        rng_seed_list=[42, 123, 456],
        hemisphere='left',
        suppresion_method='permutate',
        output_dir='./encoding_model/grid_search_results',
        verbose=True
    )
    
    # Access results
    print("\n" + "=" * 80)
    print("ACCESSING RESULTS")
    print("=" * 80)
    
    results_df = grid_search_results['results_df']
    results_by_seed = grid_search_results['results_by_seed']
    file_paths = grid_search_results['file_paths']
    
    print(f"\nTotal results: {len(results_df)}")
    print(f"\nSaved files:")
    for fp in file_paths:
        print(f"  - {fp}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("RESULTS BY RANDOM SEED")
    print("=" * 80)
    
    for seed, df in results_by_seed.items():
        print(f"\nSeed {seed}: {len(df)} combinations")
        print(f"  Unique suppression strengths: {df['suppression_strength'].nunique()}")
        print(f"  Unique SNR values: {df['snr'].nunique()}")
        print(f"  Unique mixing dimensions: {df['mixing_dimension'].nunique()}")
        
        # Show best result for this seed (highest common variance)
        best_idx = df['common'].idxmax()
        best_row = df.loc[best_idx]
        print(f"\n  Best result (highest common variance):")
        print(f"    SS={best_row['suppression_strength']}, SNR={best_row['snr']}, " +
              f"MD={best_row['mixing_dimension']}")
        print(f"    Common: {best_row['common']:.4f}, Unique A: {best_row['unique_A']:.4f}, " +
              f"Unique B: {best_row['unique_B']:.4f}")
    
    return grid_search_results


def example_focused_grid_search():
    """Example with a smaller grid for quick testing."""
    
    print("\n\n")
    print("=" * 80)
    print("FOCUSED GRID SEARCH - QUICK EXAMPLE")
    print("=" * 80)
    
    # Generate smaller synthetic data
    n_samples = 100
    n_features = 30
    n_voxels = 20
    
    print("\nGenerating synthetic data...")
    rng = np.random.default_rng(999)
    
    features = rng.standard_normal((n_samples, n_features))
    true_weights = rng.standard_normal((n_features, n_voxels))
    fmri = features @ true_weights + 0.05 * rng.standard_normal((n_samples, n_voxels))
    
    # Train model
    print("Training model...")
    alphas = np.logspace(-3, 3, 50)
    reg_model = RidgeCV(alphas=alphas, fit_intercept=True, scoring='r2', cv=5)
    reg_model.fit(features, fmri)
    
    # Run quick grid search
    print("\nRunning focused grid search...")
    results = grid_search_suppression_analysis(
        features=features,
        reg_lh=reg_model,
        suppression_strength_list=[0.5],
        snr_list=[5.0, 10.0],
        mixing_dimension_list=[None, 20],
        rng_seed_list=[42, 100],
        output_dir='./encoding_model/grid_search_results_quick',
        verbose=True
    )
    
    print("\n✓ Quick grid search completed!")
    
    return results


if __name__ == "__main__":
    # Run the full grid search example
    results = example_grid_search()
    
    # Run the focused grid search example
    results_quick = example_focused_grid_search()
    
    print("\n\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 80)
