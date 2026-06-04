import numpy as np
import torch
import pytest
from sklearn.linear_model import LinearRegression
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))  
from Partial_Information_Decomposition.Idep.Idep_univariabe_gauss import Idep_univariate_gauss



"""This file tests the Idep implementation using known toy examples from:
Venkatesh et al 2023. (NeurIPS-2023-gaussian-partial-information-decomposition-bias-correction-and-application-to-high-dimensional-data-Paper-Conference)
Ince et al. 2018: (Exact Partial Information Decompositions for Gaussian Systems Based on Dependency Constraints)"""

def test_table3_models_normalized(random_data):
    """
    Verifies that the dependency_matrix function reproduces the logic
    defined in Table 3, assuming normalized inputs.
    """
    solver = Idep_univariate_gauss(None, None, None)
    
    # Define a VALID Correlation matrix (Diagonals must be 1.0)
    # p = 0.5, q = 0.4, r = 0.2
    cov_matrix = torch.tensor([
        [1.0, 0.5, 0.4],
        [0.5, 1.0, 0.2],
        [0.4, 0.2, 1.0]
    ])
    
    p, q, r = 0.5, 0.4, 0.2

    # --- Test Model 5 (U5) ---
    # Fix p, q -> Induce r = p*q
    res_u5 = solver.dependency_matrix(['c_model_5'], cov_matrix=cov_matrix)
    res_u5 = res_u5['c_model_5']
    assert torch.isclose(res_u5[1,2], torch.tensor(p * q)), \
        f"Model 5 failed. Expected induced r={p*q:.2f}, got {res_u5[1,2]:.2f}"
    # Check if diagonals are preserved (still 1.0)
    assert torch.allclose(torch.diag(res_u5), torch.ones(3)), "Model 5 lost normalization!"

    # --- Test Model 7 (U7) ---
    # Fix q, r -> Induce p = q*r
    res_u7 = solver.dependency_matrix(['c_model_7'], cov_matrix=cov_matrix)
    res_u7 = res_u7['c_model_7']
    assert torch.isclose(res_u7[0,1], torch.tensor(q * r)), \
        f"Model 7 failed. Expected induced p={q*r:.2f}, got {res_u7[0,1]:.2f}"
    
    # -- Test Model 1 (U1)
    res_u1 = solver.dependency_matrix(['c_model_1'], cov_matrix=cov_matrix)
    res_u1 = res_u1['c_model_1']
    assert torch.allclose(res_u1, torch.eye(3)), "Model 1 should return Identity matrix"

    # -- Test Model 2 (U2)
    res_u2 = solver.dependency_matrix(['c_model_2'], cov_matrix=cov_matrix)
    res_u2 = res_u2['c_model_2']
    assert torch.isclose(res_u2[0,1], torch.tensor(p)), \
        f"Model 2 failed. Expected p={p:.2f}, got {res_u2[0,1]:.2f}"

    # -- Test Model 3 (U3)
    res_u3 = solver.dependency_matrix(['c_model_3'], cov_matrix=cov_matrix)
    res_u3 = res_u3['c_model_3']
    assert torch.isclose(res_u3[0,2], torch.tensor(q)), \
        f"Model 3 failed. Expected q={q:.2f}, got {res_u3[0,2]:.2f}"
def test_identity_default():
    """Test that U1 (c_model_1) returns Identity."""
    solver = Idep_univariate_gauss(None, None, None)
    cov_matrix = torch.tensor([
        [1.0, 0.8, 0.7],
        [0.8, 1.0, 0.6],
        [0.7, 0.6, 1.0]
    ])
    
    res_u1 = solver.dependency_matrix(['c_model_1'], cov_matrix=cov_matrix)
    res_u1 = res_u1['c_model_1']
    assert torch.allclose(res_u1, torch.eye(3)), "Model 1 should return Identity matrix"
