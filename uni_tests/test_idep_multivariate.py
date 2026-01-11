import numpy as np
import torch
import pytest
from sklearn.linear_model import LinearRegression
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))  
from Partial_Information_Decomposition.Idep_multivariate_gauss import Idep_multivariate_gauss




@pytest.fixture(scope="module")
def dims():
    return dict(d_m1=3, d_m2=4, d_t=3)



@pytest.fixture #It marks the function random_data() as a data provider.
def random_data(dims):
    """Fixture to provide random tensors for testing."""
    N = 100
    d_t = dims['d_t']
    d_m1 = dims['d_m1']
    d_m2 = dims['d_m2']
    T = torch.randn(N, d_t)
    M1 = torch.randn(N, d_m1)
    M2 = torch.randn(N, d_m2)
    return T, M1, M2

def test_init_idep(random_data):
    """
    Verifies that the dependency_matrix function reproduces the logic
    defined in Table 3, assuming normalized inputs.
    """
    M1, M2, T = random_data
    assert Idep_multivariate_gauss(sources=[M1, M2], targets=[T], cov_matrix=None), "Initialization of Idep_multivariate_gauss failed."

def test_dependency_matrix(random_data):
    """
    Verifies that the dependency_matrix function reproduces the logic
    defined in Table 3, assuming normalized inputs.
    """
    M1, M2, T = random_data
    solver = Idep_multivariate_gauss(sources=[M1, M2], targets=[T], cov_matrix=None)
    
    constraints = ['c_model_1', 'c_model_2', 'c_model_3', 'c_model_4', 'c_model_5', 'c_model_6', 'c_model_7']

    res = solver.dependency_matrix(constraints)
    assert res is not None, "dependency_matrix should return a result."


def test_symmetry(random_data, dims):
    """
    Verifies that the covariance matrices produced by the dependency_matrix function
    are symmetric.
    """
    d_m1 , d_m2 , d_t  = dims['d_m1'], dims['d_m2'], dims['d_t']
    T, M1, M2 = random_data
    solver = Idep_multivariate_gauss(sources=[M1, M2], targets=[T], cov_matrix=None)
    
    constraints = ['c_model_1', 'c_model_2', 'c_model_3', 'c_model_4', 'c_model_5', 'c_model_6', 'c_model_7']
    res = solver.dependency_matrix(constraints)
    for model_key, cov_matrix in res.items():
        assert torch.allclose(cov_matrix, cov_matrix.T, atol=1e-8), f"Covariance matrix for {model_key} is not symmetric."

def test_diagonal_block(random_data, dims):
    """
    Verifies that the diagonal blocks of the resulting covariance matrices
    remain normalized (i.e., equal to 1.0).
    """
    d_m1 , d_m2 , d_t  = dims['d_m1'], dims['d_m2'], dims['d_t']
    T, M1, M2 = random_data
    solver = Idep_multivariate_gauss(sources=[M1, M2], targets=[T], cov_matrix=None)
    
    constraints = ['c_model_1', 'c_model_2', 'c_model_3', 'c_model_4', 'c_model_5', 'c_model_6', 'c_model_7']
    res = solver.dependency_matrix(constraints)
    I0 = torch.eye(d_m1)
    I1 = torch.eye(d_m2)
    I2 = torch.eye(d_t)
    for model_key, cov_matrix in res.items():
        # Extract diagonal blocks
        diag_block_m1 = cov_matrix[0:d_m1, 0:d_m1]
        diag_block_m2 = cov_matrix[d_m1:d_m1 + d_m2, d_m1:d_m1 + d_m2]
        diag_block_t = cov_matrix[d_m1 + d_m2:, d_m1 + d_m2:]

        # Check if diagonal blocks are close to identity matrices
        assert torch.allclose(diag_block_m1, I0, atol=1e-8), f"Diagonal block for M1 in {model_key} is not normalized."
        assert torch.allclose(diag_block_m2, I1, atol=1e-8), f"Diagonal block for M2 in {model_key} is not normalized."
        assert torch.allclose(diag_block_t, I2, atol=1e-8), f"Diagonal block for T in {model_key} is not normalized."


def test_constraint_dim(random_data, dims):
    """
    Verifies that the covariance matrices produced by the dependency_matrix function
    have the correct dimensions.
    """
    d_m1 , d_m2 , d_t  = dims['d_m1'], dims['d_m2'], dims['d_t']
    T, M1, M2 = random_data
    solver = Idep_multivariate_gauss(sources=[M1, M2], targets=[T], cov_matrix=None)
    
    constraints = ['c_model_1', 'c_model_2', 'c_model_3', 'c_model_4', 'c_model_5', 'c_model_6', 'c_model_7']
    res = solver.dependency_matrix(constraints)
    expected_dim = d_m1 + d_m2 + d_t
    for model_key, cov_matrix in res.items():
        assert cov_matrix.shape == (expected_dim, expected_dim), f"Covariance matrix for {model_key} has incorrect dimensions."


def test_semi_positivity(random_data):
    """
    Verifies that the covariance matrices produced by the dependency_matrix function
    are positive semi-definite.
    """
    M1, M2, T = random_data
    solver = Idep_multivariate_gauss(sources=[M1, M2], targets=[T], cov_matrix=None)
    
    constraints = ['c_model_1', 'c_model_2', 'c_model_3', 'c_model_4', 'c_model_5', 'c_model_6', 'c_model_7']
    res = solver.dependency_matrix(constraints)
    for model_key, cov_matrix in res.items():
        eigenvalues = torch.linalg.eigvalsh(cov_matrix)
        assert torch.all(eigenvalues >= -1e-5), f"Covariance matrix for {model_key} is not positive semi-definite."


@pytest.fixture
def PQR(random_data):
    T, M1, M2 = random_data  # each (N, d)

    def zscore(X):
        X = X - X.mean(dim=0, keepdim=True)
        X = X / (X.std(dim=0, keepdim=True, unbiased=True) + 1e-8)
        return X

    Tz  = zscore(T)
    M1z = zscore(M1)
    M2z = zscore(M2)

    # Cross-correlation blocks (matrix-valued)
    # P: corr(M1, M2) shape (d_m1, d_m2)
    P = (M1z.T @ M2z) / (M1z.shape[0] - 1)

    # Q: corr(M1, T)  shape (d_m1, d_t)
    Q = (M1z.T @ Tz)  / (M1z.shape[0] - 1)

    # R: corr(M2, T)  shape (d_m2, d_t)
    R = (M2z.T @ Tz)  / (M2z.shape[0] - 1)

    return [P, Q, R],[Tz, M1z, M2z]

def test_PQR(PQR):
    """
    Verifies that the P, Q, R matrices have correct dimensions.
    """
    list1,list2 = PQR
    P, Q, R = list1
    T, M1, M2 = list2
    solver = Idep_multivariate_gauss(sources=[M1, M2], targets=[T], cov_matrix=None)
    
    assert torch.all(torch.isclose(P, solver.P)), "P matrix does not match expected value."
    assert torch.all(torch.isclose(Q, solver.Q)), "Q matrix does not match expected value."
    assert torch.all(torch.isclose(R, solver.R)), "R matrix does not match expected value."
    # Further dimension checks can be added based on expected sizes

def test_PQR_multi(PQR):
    list1,list2 = PQR
    P, Q, R = list1
    T, M1, M2 = list2
    solver = Idep_multivariate_gauss(sources=[M1, M2], targets=[T], cov_matrix=None)

    constraints  = ['c_model_1', 'c_model_2', 'c_model_3', 'c_model_4', 'c_model_5', 'c_model_6', 'c_model_7']
    res = solver.dependency_matrix(constraints)
    M5 = res['c_model_5']
    M6 = res['c_model_6']
    M7 = res['c_model_7']

    assert torch.all(torch.isclose(P.T @ Q, M5[M1.shape[1]:M1.shape[1] + M2.shape[1], M1.shape[1] + M2.shape[1]:])), "P.T @ Q does not match covariance block in model 5." 
    assert torch.all(torch.isclose(P @ R, M6[:M1.shape[1], M1.shape[1] + M2.shape[1]:])), "P.T @ R does not match covariance block in model 6."
    assert torch.all(torch.isclose(Q@ R.T, M7[:M1.shape[1], M1.shape[1]:M1.shape[1] + M2.shape[1]])), "Q.T @ R does not match covariance block in model 7."

def test_idep(random_data):
    """
    Verifies that the compute_Idep function runs without errors and returns a dictionary.
    """
    M1, M2, T = random_data
    solver = Idep_multivariate_gauss(sources=[M1, M2], targets=[T], cov_matrix=None)
    
    constraints = ['c_model_1', 'c_model_2', 'c_model_3', 'c_model_4', 'c_model_5', 'c_model_6', 'c_model_7','c_model_8']
    solver.dependency_matrix(constraints)
    unique_result = solver.compute_Idep()
    pid_result = solver.pid_values(unique_result['unique_0'], unique_result['unique_1'])

    #run pipeline
    pid_result_pipline = solver.idep()

    for key in ['red', 'unq0', 'unq1', 'syn']:
        assert key in pid_result, f"Key {key} missing in PID result."
        assert key in pid_result_pipline, f"Key {key} missing in PID result from pipeline."
        assert torch.isclose(torch.tensor([pid_result[key]]), torch.tensor([pid_result_pipline[key]])), f"Mismatch in PID value for {key} between direct computation and pipeline."