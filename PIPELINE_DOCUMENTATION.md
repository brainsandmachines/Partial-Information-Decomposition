# Suppression Analysis Pipeline

This document describes the complete suppression analysis pipeline that integrates feature extraction, regression modeling, suppression model creation, and commonality analysis.

## Overview

The `suppression_analysis_pipeline` function provides an end-to-end workflow for analyzing neural representations using suppression models and commonality analysis. This is particularly useful for understanding how different feature sets contribute to neural predictions.

## Pipeline Steps

1. **Prediction Generation**: Uses trained regression models to create predictions from features
2. **Suppression Model Creation**: Generates two models (M1 and M2) with controlled suppression
3. **Commonality Analysis**: Decomposes variance into unique and shared components

## Usage

### Basic Example

```python
from encoding_model.suppression_core import suppression_analysis_pipeline
from sklearn.linear_model import RidgeCV
import numpy as np

# Assume you have:
# - features: shape (n_samples, n_features)
# - reg_lh: trained regression model for left hemisphere
# - reg_rh: trained regression model for right hemisphere

results = suppression_analysis_pipeline(
    features=features,
    reg_lh=reg_lh,
    reg_rh=reg_rh,
    hemisphere='both',
    suppression_strength=0.5,
    snr=5.0,
    mixing_dimension=50,
    analysis_methods=['standard', 'ridge_cv']
)
```

### Parameters

- **features** (*np.ndarray*): Feature matrix with shape `(n_samples, n_features)`
  - Typically DNN activations or other model representations

- **reg_lh** (*sklearn model, optional*): Trained regression model for left hemisphere
  - Must implement `.predict()` method
  - Use `None` if not analyzing left hemisphere

- **reg_rh** (*sklearn model, optional*): Trained regression model for right hemisphere
  - Must implement `.predict()` method
  - Use `None` if not analyzing right hemisphere

- **hemisphere** (*str*): Which hemisphere(s) to analyze
  - Options: `'left'`, `'right'`, `'both'`
  - Default: `'both'`

- **suppression_strength** (*float*): Proportion of features to suppress
  - Range: [0, 1]
  - Default: `0.5`
  - Higher values = more suppression

- **snr** (*float*): Signal-to-noise ratio for target
  - Default: `1.0`
  - Higher values = less noise added to predictions

- **mixing_dimension** (*int or None*): Dimensionality for feature mixing
  - Default: `None` (no mixing)
  - When not None, features are projected to this dimension

- **suppresion_method** (*str*): Method for creating suppression
  - Default: `'permutate'`
  - Creates M1 and M2 by permuting feature subsets

- **analysis_methods** (*list*): Methods for commonality analysis
  - Options: `'standard'`, `'ols_cv'`, `'ridge_cv'`
  - Default: `['standard', 'ols_cv', 'ridge_cv']`

- **rng_seed** (*int or None*): Random seed for reproducibility
  - Default: `None` (random)

- **alphas** (*np.ndarray or None*): Alpha values for ridge regression
  - Default: `None` (uses default range)

### Returns

Dictionary with three main components:

```python
{
    'predictions': {
        'lh': np.ndarray,  # Predictions for left hemisphere
        'rh': np.ndarray   # Predictions for right hemisphere
    },
    'suppression_models': {
        'lh': {
            'X_M1': np.ndarray,    # Model 1 features
            'X_M2': np.ndarray,    # Model 2 features
            'target': np.ndarray,  # Target with noise
            'signal': np.ndarray   # Original predictions
        },
        'rh': { ... }  # Same structure for right hemisphere
    },
    'commonality_results': {
        'lh': {
            'standard': {
                'R²_A': float,       # R² for model A
                'R²_B': float,       # R² for model B
                'R²_AB': float,      # R² for combined model
                'unique_A': float,   # Unique variance of A
                'unique_B': float,   # Unique variance of B
                'common': float,     # Shared variance
                'unexplained': float # Unexplained variance
            },
            'ols_cv': { ... },
            'ridge_cv': { ... }
        },
        'rh': { ... }  # Same structure for right hemisphere
    }
}
```

## Commonality Analysis Interpretation

The commonality analysis decomposes the total variance explained into:

- **Unique A**: Variance explained only by Model 1 (features with real signal)
- **Unique B**: Variance explained only by Model 2 (features with spurious signal)
- **Common**: Variance explained by both models (shared information)
- **Unexplained**: Variance not explained by either model

### Mathematical Formulation

```
R²(AB) = unique_A + unique_B + common
unique_A = R²(AB) - R²(B)
unique_B = R²(AB) - R²(A)
common = R²(A) + R²(B) - R²(AB)
unexplained = 1 - R²(AB)
```

## Complete Workflow Example

```python
import numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
import joblib

# 1. Load your features (e.g., from DNN)
features = np.load('path/to/features.npy')  # shape: (n_samples, n_features)

# 2. Load or train regression models
# Option A: Load pre-trained
reg_lh = joblib.load('path/to/reg_lh.pkl')
reg_rh = joblib.load('path/to/reg_rh.pkl')

# Option B: Train new models
# fmri_lh = np.load('path/to/fmri_lh.npy')
# fmri_rh = np.load('path/to/fmri_rh.npy')
# reg_lh = RidgeCV(alphas=np.logspace(-3, 3, 50)).fit(features, fmri_lh)
# reg_rh = RidgeCV(alphas=np.logspace(-3, 3, 50)).fit(features, fmri_rh)

# 3. Run the pipeline
results = suppression_analysis_pipeline(
    features=features,
    reg_lh=reg_lh,
    reg_rh=reg_rh,
    hemisphere='both',
    suppression_strength=0.5,
    snr=5.0,
    mixing_dimension=50,
    analysis_methods=['standard', 'ridge_cv'],
    rng_seed=42
)

# 4. Extract results
for hemi in ['lh', 'rh']:
    for method in ['standard', 'ridge_cv']:
        result = results['commonality_results'][hemi][method]
        print(f"\n{hemi.upper()} - {method}:")
        print(f"  Unique A: {result['unique_A']:.4f}")
        print(f"  Unique B: {result['unique_B']:.4f}")
        print(f"  Common:   {result['common']:.4f}")

# 5. Save results
output_dir = Path('results/suppression_analysis')
output_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(results, output_dir / 'pipeline_results.pkl')
```

## Integration with Existing Code

The pipeline integrates with existing functions:

### From `fmri_model.py`
- Train models using the `encoding_model` class
- Extract features from DNNs

### From `encoding_utils.py`
- Use `compute_r2()`, `compute_ols_cv_r2()`, `compute_ridge_cv_r2()`
- These are called internally by `commonality_analysis()`

### From `suppression_core.py`
- `create_supression_model()`: Creates M1 and M2 models
- `commonality_analysis()`: Decomposes variance
- `suppression_analysis_pipeline()`: Orchestrates the entire workflow

## Example Use Cases

### 1. Single Hemisphere Analysis
```python
results = suppression_analysis_pipeline(
    features=features,
    reg_lh=reg_lh,
    reg_rh=None,  # Only left hemisphere
    hemisphere='left',
    suppression_strength=0.3,
    snr=10.0
)
```

### 2. High SNR (Less Noise)
```python
results = suppression_analysis_pipeline(
    features=features,
    reg_lh=reg_lh,
    reg_rh=reg_rh,
    hemisphere='both',
    suppression_strength=0.5,
    snr=20.0  # High SNR = less noise
)
```

### 3. Grid Search Over Parameters
```python
import pandas as pd

results_list = []
for suppression in [0.3, 0.5, 0.7]:
    for snr in [1.0, 5.0, 10.0]:
        results = suppression_analysis_pipeline(
            features=features,
            reg_lh=reg_lh,
            hemisphere='left',
            suppression_strength=suppression,
            snr=snr,
            analysis_methods=['standard']
        )
        
        result_dict = results['commonality_results']['lh']['standard']
        result_dict['suppression_strength'] = suppression
        result_dict['snr'] = snr
        results_list.append(result_dict)

df = pd.DataFrame(results_list)
print(df)
```

## Troubleshooting

### Error: "No regression models provided"
- Ensure at least one of `reg_lh` or `reg_rh` is not None
- Check that `hemisphere` parameter matches the provided models

### Singular Matrix Errors
- Try reducing `mixing_dimension`
- Increase `snr` to add less noise
- Check that features are not perfectly correlated

### Negative Commonality Values
- This is possible in commonality analysis
- Indicates suppression effects
- Common when features are highly correlated

## See Also

- [examples/suppression_pipeline_example.py](../examples/suppression_pipeline_example.py) - Working examples
- [encoding_model/suppression_core.py](../encoding_model/suppression_core.py) - Source code
- [encoding_model/grid_search.py](../encoding_model/grid_search.py) - Grid search implementation
