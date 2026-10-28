import pytest
import torch


@pytest.fixture
def random_data():
    """Return three (100, 1) torch tensors for covariance utility tests."""
    N = 100
    d_t = d_m1 = d_m2 = 1

    # Scalar dimensions -> three (100, 1) tensors.
    T = torch.randn(N, d_t)
    M1 = torch.randn(N, d_m1)
    M2 = torch.randn(N, d_m2)
    return T, M1, M2
