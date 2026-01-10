import numpy as np
import torch
import pytest
from sklearn.linear_model import LinearRegression
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))  
from Partial_Information_Decomposition import PID_util

create_cov_matrix = PID_util.create_cov_matrix

"""This file tests the Idep implementation using known toy examples from:
Venkatesh et al 2023. (NeurIPS-2023-gaussian-partial-information-decomposition-bias-correction-and-application-to-high-dimensional-data-Paper-Conference)
Ince et al. 2018: (Exact Partial Information Decompositions for Gaussian Systems Based on Dependency Constraints)"""


@pytest.fixture #It marks the function random_data() as a data provider.
def random_data():
    """Fixture to provide random tensors for testing."""
    N = 100
    d_t = 1
    d_m1 = 1
    d_m2 = 1
    T = torch.randn(N, d_t)
    M1 = torch.randn(N, d_m1)
    M2 = torch.randn(N, d_m2)
    return T, M1, M2

def test_output_keys(random_data):
    """Check if all required keys are present."""
    T, M1, M2 = random_data
    cov_dict = create_cov_matrix(T, M1, M2)
    expected_keys = [
        'cov_t', 'cov_m1', 'cov_m2', 
        'cov_t_m1', 'cov_t_m2', 'cov_tm2',
        'cross_cov_m1_m2', 'cross_cov_m12_t', 
        'auto_cov_m12', 'cov_tm1', 'full_cov'
    ]
    for key in expected_keys:
        assert key in cov_dict

def test_shapes(random_data):
    """Test if output sub-matrices have correct dimensions."""
    T, M1, M2 = random_data
    d_t = T.shape[1]
    d_m1 = M1.shape[1]
    d_m2 = M2.shape[1]
    
    cov_dict = create_cov_matrix(T, M1, M2)
    
    # Check T-M1 covariance shape
    assert cov_dict['cov_t_m1'].shape == (d_t, d_m1)
    
    # Check T-M2 manual reconstruction shape
    assert cov_dict['cov_tm2'].shape == (d_t + d_m2, d_t + d_m2)
    
    # Check auto covariance of M12
    expected_dim = d_m1 + d_m2
    assert cov_dict['auto_cov_m12'].shape == (expected_dim, expected_dim)

def test_reconstruction_of_tm2(random_data):
    """
    CRITICAL: Verify manual construction of cov_tm2 matches direct calculation.
    """
    T, M1, M2 = random_data
    cov_dict = create_cov_matrix(T, M1, M2)
    computed_tm2 = cov_dict['cov_tm2']
    
    # Ground Truth: Calculate covariance of JUST T and M2 directly
    Z_tm2 = torch.hstack([T, M2])
    expected_tm2 = torch.cov(Z_tm2.T, correction=1)
    
    assert torch.allclose(computed_tm2, expected_tm2, atol=1e-6), \
        "Manual reconstruction of T-M2 covariance is incorrect!"

def test_perfect_correlation():
    """Test values with deterministic linear relationships."""
    T = torch.randn(50, 1)
    M1 = 2.0 * T  # Perfectly correlated
    
    # We need M2 just to run the function, logic doesn't depend on it
    M2 = torch.randn(50, 1) 
    
    cov_dict = create_cov_matrix(T, M1, M2)
    
    var_t = cov_dict['cov_t'].item()
    cov_t_m1 = cov_dict['cov_t_m1'].item()
    
    # Cov(T, 2T) should be 2 * Var(T)
    assert abs(cov_t_m1 - 2 * var_t) < 1e-5, f"Expected 2*Var(T), got {cov_t_m1}"

def test_symmetry(random_data):
    """Full covariance matrix must be symmetric."""
    T, M1, M2 = random_data
    cov_dict = create_cov_matrix(T, M1, M2)
    full_cov = cov_dict['full_cov']
    
    assert torch.allclose(full_cov, full_cov.T, atol=1e-6), \
        "Full covariance matrix is not symmetric"