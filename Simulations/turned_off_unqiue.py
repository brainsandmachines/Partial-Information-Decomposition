import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root)) 
from toy_examples.toy_example import commonality_analysis
from Partial_Information_Decomposition.Idep_multivariate_gauss import Idep_multivariate_gauss
from Partial_Information_Decomposition.PID_util import compare_results



def feature_creation(rng,r_str,u1_str,u2_str,unique_method = 'orthogonal', n=1024, p=100, mixing_dimension=None, snr=10.0, method='standard', show_diagnostic_plots=False):
    """
    Creates dummy predictors and a target
    
    Args:
        rng: Random number generator
        r_str: strength of redundant features
        u1_str: strength of unique features in source 1
        u2_str: strength of unique features in source 2
        n: Number of samples
        p: Number of features per source
        mixing_dimension: If not None, apply a mixing matrix with this dimension to entangle features
        snr: Signal-to-noise ratio (signal_std / noise_std)
        method: Which R² computation to use: 'standard', 'ols_cv', or 'ridge_cv'
        
    Returns:
        dict: Commonality analysis results
    """
    # Generate the four feature tensors
    R = rng.standard_normal((n, p))
    U1 = rng.standard_normal((n, p))
    U2 = rng.standard_normal((n, p))
    
    

    signal = r_str * R + u1_str * U1 + u2_str * U2
    # Target: only real features contribute
    #betas = rng.standard_normal((p, p))
    #signal = real_features @ betas
    noise_std = np.std(signal) / snr

    y_real  = signal + noise_std * rng.standard_normal((signal.shape[0], signal.shape[1]))


    X_M1 = r_str * R + u1_str * U1 
    X_M1 += 0*noise_std * rng.standard_normal((X_M1.shape[0], X_M1.shape[1]))
    X_M2 = r_str * R + u2_str * U2
    X_M2 += 0*noise_std * rng.standard_normal((X_M2.shape[0], X_M2.shape[1]))

    return X_M1, X_M2, y_real




def test(rng, r_str, u1_str, u2_str, n=1024, p=100, snr=10.0, method='standard'):
    M1, M2, y_real = feature_creation(rng,r_str,u1_str,u2_str, n=n, p=p, snr=snr, method=method)
    ca_results = commonality_analysis(M1, M2, y_real, method=method)
    M1 = torch.tensor(M1)
    M2 = torch.tensor(M2)
    T = torch.tensor(y_real)
    pid_results,mi_results = Idep_multivariate_gauss(sources=[M1, M2], targets=[T], bias_correction=True).idep()

    return ca_results, pid_results, mi_results



def main():
    rng = np.random.default_rng(seed=42)
    r_str = 10
    u1_str = 10
    u2_str = 0.85
    ca_results, pid_results, mi_results = test(rng, r_str, u1_str, u2_str, n=10000000, p=2, snr=10, method='lasso_cv')
    compare_results(ca_results, pid_results,mi_results)

if __name__ == "__main__":
    main()