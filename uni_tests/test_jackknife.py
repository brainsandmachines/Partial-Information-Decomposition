import sys
from pathlib import Path
import torch
import pytest
from sklearn.model_selection import LeaveOneOut
import time
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))  
from Partial_Information_Decomposition.PID_util import *
from Partial_Information_Decomposition.Toy_Simulations.Bias_Corr_simulations import theoretical_covariance, sample_cov_simulation
from Partial_Information_Decomposition.Jackknife_Bias_Corr import lo_cov



def reference_loo_covariances_with_sklearn(rvs):
    """
    Reference implementation using sklearn LeaveOneOut + explicit loop.
    """
    Z = torch.hstack(rvs).to(torch.float64)  # shape (N, d)
    N, d = Z.shape
    full_cov = create_cov_matrix(rvs=rvs)['full_cov']
    assert torch.allclose(full_cov, torch.cov(Z.T, correction=1), atol=1e-10, rtol=1e-8), "Full covariance from create_cov"
    loo = LeaveOneOut()
    covs = []

    for train_idx, _ in loo.split(Z):
        Z_train = Z[train_idx]  # shape (N-1, d)

        # unbiased covariance, same convention as torch.cov(..., correction=1)
        cov_j = torch.cov(Z_train.T, correction=1)
        covs.append(cov_j)

    return full_cov,torch.stack(covs, dim=0)  # shape (N, d, d)


@pytest.mark.parametrize(
    "N,p,q,r,dims,seed",
    [
        (100, 0.6, 0.5, 0.4, [10,10,10], 42),
        (100, 0.7, 0.6, 0.5, [10,10,5], 42),
        (100, 0.8, 0.7, 0.6, [10,5,10], 42),
        (100, 0.9, 0.8, 0.7, [5,8,10], 42),
        (100, 0.5, 0.4, 0.3, [10,20,30], 42),
    ],
)

def test_lo_cov(N, p, q, r, dims, seed):
    corr_matrix = np.array([
        [1.0,   q * r,  q  ],  # Row 1: X1
        [q * r, 1.0,    r  ],  # Row 2: X2
        [q,     r,      1.0]   # Row 3: X3
    ])
    true_cov = theoretical_covariance(dims, corr_matrix)
    rv_list, sample_cov = sample_cov_simulation(seed, N, dims, true_cov)
    sample_cov = torch.from_numpy(sample_cov).to(torch.float64)
    torch_rv_list = [torch.from_numpy(rv).to(torch.float64) for rv in rv_list]  # Convert to torch tensors
    ref_full_cov, ref_Sigma = reference_loo_covariances_with_sklearn(torch_rv_list)
    full_cov_lo_cov, loo_covs = lo_cov(torch_rv_list, N)
    assert torch.allclose(full_cov_lo_cov, sample_cov, atol=1e-10, rtol=1e-8), "Full covariance from lo_cov does not match sample covariance"
    assert torch.allclose(ref_full_cov, full_cov_lo_cov, atol=1e-10, rtol=1e-8), "Full covariance from lo_cov does not match reference full covariance"
    assert torch.allclose(ref_Sigma, loo_covs, atol=1e-10, rtol=1e-8), "Leave-one-out covariance matrices do not match the reference implementation"



@pytest.mark.parametrize(
    "N,p,q,r,dims,seed",
    [
        (1000, 0.6, 0.5, 0.4, [100,100,500], 42),
        (2000, 0.6, 0.5, 0.4, [100,100,500], 42),
        (3000, 0.6, 0.5, 0.4, [100,100,500], 42),
    ],
)

def test_speed(N,p,q,r,dims,seed):

    corr_matrix = np.array([
        [1.0,   q * r,  q  ],  # Row 1: X1
        [q * r, 1.0,    r  ],  # Row 2: X2
        [q,     r,      1.0]   # Row 3: X3
    ])
    true_cov = theoretical_covariance(dims, corr_matrix)
    rv_list, sample_cov = sample_cov_simulation(seed, N, dims, true_cov)
    torch_rv_list = [torch.from_numpy(rv).to(torch.float64) for rv in rv_list]  # Convert to torch tensors
    

    start_time_loo = time.time()
    a, b = lo_cov(torch_rv_list, N)
    end_time_loo = time.time()

    start_time_ref = time.time()
    a, b = reference_loo_covariances_with_sklearn(torch_rv_list)
    end_time_ref = time.time()

    loo_total_time = end_time_loo - start_time_loo
    ref_total_time = end_time_ref - start_time_ref

    print(f"\nLOO Time: {loo_total_time:.4f}s")
    print(f"Ref Time: {ref_total_time:.4f}s")
    print(f"Speedup: {ref_total_time / loo_total_time:.2f}x")
    assert loo_total_time < ref_total_time, f"lo_cov took {loo_total_time:.4f} seconds, which is not faster than the reference implementation that took {ref_total_time:.4f} seconds."