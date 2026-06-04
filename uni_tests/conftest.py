import pytest
import torch


@pytest.fixture(scope="module")
def dims():
    return dict(d_m1=3, d_m2=4, d_t=3)


@pytest.fixture
def random_data(request):
    """Provide random tensors for univariate and multivariate PID tests."""
    N = 100

    if request.module.__name__.endswith("test_idep_multivariate"):
        test_dims = request.getfixturevalue("dims")
        d_t = test_dims["d_t"]
        d_m1 = test_dims["d_m1"]
        d_m2 = test_dims["d_m2"]
    else:
        d_t = 1
        d_m1 = 1
        d_m2 = 1

    T = torch.randn(N, d_t)
    M1 = torch.randn(N, d_m1)
    M2 = torch.randn(N, d_m2)
    return T, M1, M2
