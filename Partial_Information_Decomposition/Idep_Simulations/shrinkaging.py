import torch
import numpy as np 
from sklearn.covariance import LedoitWolf,ShrunkCovariance,OAS
from Idep_Simulations.Simulation_utils import *
from wrapper_M7_M8_models import create_m7_cov, make_random_true_cov
import torch
from typing import Dict, Union
from functools import partial 

def ledoit_wolf_cov(X):
    lw = LedoitWolf().fit(X)
    return lw.covariance_

def oracle_shrinkage_cov(X, assume_centered=False, return_shrinkage=False):
    """
    Oracle Approximating Shrinkage covariance estimator.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.
    assume_centered : bool, default=False
        If True, data are assumed to be centered already.
    return_shrinkage : bool, default=False
        If True, also return the estimated shrinkage coefficient.

    Returns
    -------
    cov : ndarray of shape (n_features, n_features)
        Shrunk covariance estimate.

    shrinkage : float, optional
        Estimated shrinkage coefficient, returned only if
        return_shrinkage=True.
    """
    oas = OAS(assume_centered=assume_centered).fit(X)
    if return_shrinkage:
        return oas.covariance_, oas.shrinkage_
    return oas.covariance_

def shrunk_cov(X, alpha=0.1):
    if type(alpha) == list:
        sc_list = []
        for shrinkage in alpha:
            sc = ShrunkCovariance(shrinkage=shrinkage).fit(X)
            sc_list.append(sc.covariance_)
        return sc_list
    else:
        sc = (ShrunkCovariance(shrinkage=alpha).fit(X))
    return sc.covariance_



def shrinkage_covariance(X, method='ledoit_wolf', alpha=0.1):
    if method == 'ledoit_wolf':
        return ledoit_wolf_cov(X)
    elif method == 'shrunk_cov':
        return shrunk_cov(X, alpha)
    else:
        raise ValueError("Unsupported method. Use 'ledoit_wolf' or 'shrunk_cov'.")
    

def custom_shrunk_cov(X, alpha=0.1, target=None, assume_centered=False, ddof=0):
    """
    Shrink sample covariance toward a user-supplied symmetric target matrix.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    alpha : float in [0, 1]
        Shrinkage intensity.
    target : ndarray, shape (p, p)
        Symmetric target matrix.
    assume_centered : bool, default=False
        If False, center X first.
    ddof : int, default=0
        Use ddof=0 to match sklearn covariance scaling more closely.
        Use ddof=1 for unbiased sample covariance.

    Returns
    -------
    Sigma_hat : ndarray, shape (p, p)
    """
    if not assume_centered:
        X = X - X.mean(axis=0, keepdims=True)

    n, p = X.shape
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    S = np.cov(X, rowvar=False, bias=(ddof == 0), ddof=ddof)

    if target is None:
        raise ValueError("You must provide a target matrix")

    T = np.asarray(target, dtype=float)
    if T.shape != (p, p):
        raise ValueError(f"target must have shape {(p, p)}, got {T.shape}")

    if not np.allclose(T, T.T, atol=1e-10):
        raise ValueError("target must be symmetric")

    Sigma_hat = (1 - alpha) * S + alpha * T
    return Sigma_hat
    


def shrinkage_m7_m8_simulation(config:dict,evluation_func:callable = None,data=None):
    """This function takes true covriances and return a smaple with shrinkage covariance estimation for both M7 and M8 models. 
    It also returns the true covariances for both models. The function can be used to evaluate the performance of shrinkage covariance estimation methods in the context of M7 and M8 models."""
    n0 = config['n0'] #X0 dim
    n1 = config['n1'] #X1 dim
    n2 = config['n2'] #T dim
    seed = config['seed'] 
    device = config.get('device', 'cpu')
    white_normalize = config.get('white_normalize', True)
    alphas = config['alpha_list'] #List of alpha values to test for shrinkage
    rng = torch.Generator(device=config['device']).manual_seed(seed)

    #Set true covrainces for m7 and m8
    true_cov_m8, true_cov_m7 = data

    sample_cov_m8,_ = sample_data_from_cov(config, true_cov_m8, rng)
    ledoit_wolf_cov_m8 = ledoit_wolf_cov(sample_cov_m8)
    shrunk_cov_m8 = shrunk_cov(sample_cov_m8, alphas)

    sample_cov_m7 = create_m7_cov(config,sample_cov_m8,whitening_normalize=white_normalize)
    ledoit_wolf_cov_m7 = ledoit_wolf_cov(sample_cov_m7)
    shrunk_cov_m7 = shrunk_cov(sample_cov_m7, alphas)

    return {'M8': {'true_cov': true_cov_m8, 'sample_cov': sample_cov_m8, 'ledoit_wolf_cov': ledoit_wolf_cov_m8, 'shrunk_cov': shrunk_cov_m8},
            'M7': {'true_cov': true_cov_m7, 'sample_cov': sample_cov_m7, 'ledoit_wolf_cov': ledoit_wolf_cov_m7, 'shrunk_cov': shrunk_cov_m7}}




def evaluate_shrinkage(config:dict,results_dict:dict):
    """Will evaluate the preformance of the shrinkage covriance according to 
    some evaluation function (e.g. Frobenius norm between the true covariance and the estimated covariance)"""

    evaluation_func = config['evaluation_func']
    evaluation_results = {}
    for model in ['M7', 'M8']:
        true_cov = results_dict[model]['true_cov']
        sample_cov = results_dict[model]['sample_cov']
        ledoit_wolf_cov = results_dict[model]['ledoit_wolf_cov']
        shrunk_cov_list = results_dict[model]['shrunk_cov']
        evaluation_results[model] = {
            'ledoit_wolf': evaluation_func(true_cov, ledoit_wolf_cov),
            'shrunk_cov': [evaluation_func(true_cov, shrunk_cov) for shrunk_cov in shrunk_cov_list],
            'sample_cov': evaluation_func(true_cov, sample_cov)
        }
    return evaluation_results





TensorLike = Union[torch.Tensor, "np.ndarray"]


def _to_torch_float64(X: TensorLike, device: str = None) -> torch.Tensor:
    """
    Convert input to torch.float64 tensor.
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float64)
    else:
        X = X.to(torch.float64)

    if device is not None:
        X = X.to(device)
    return X


def _check_same_shape(A: torch.Tensor, B: torch.Tensor):
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Both inputs must be square matrices, got {A.shape}")


def _check_spd(A: torch.Tensor, name: str = "matrix", eps: float = 1e-12):
    """
    Check positive definiteness via eigenvalues.
    """
    eigvals = torch.linalg.eigvalsh(A)
    min_eig = eigvals.min().item()
    if min_eig <= eps:
        raise ValueError(f"{name} is not numerically SPD. min eigenvalue = {min_eig:.3e}")


def covariance_frobenius_error(
    Sigma_true: TensorLike,
    Sigma_hat: TensorLike,
    device: str = None
) -> float:
    """
    ||Sigma_hat - Sigma_true||_F
    """
    Sigma_true = _to_torch_float64(Sigma_true, device=device)
    Sigma_hat = _to_torch_float64(Sigma_hat, device=device)
    _check_same_shape(Sigma_true, Sigma_hat)

    return torch.linalg.norm(Sigma_hat - Sigma_true, ord="fro").item()


def covariance_relative_frobenius_error(
    Sigma_true: TensorLike,
    Sigma_hat: TensorLike,
    device: str = None,
    eps: float = 1e-12
) -> float:
    """
    ||Sigma_hat - Sigma_true||_F / ||Sigma_true||_F
    """
    Sigma_true = _to_torch_float64(Sigma_true, device=device)
    Sigma_hat = _to_torch_float64(Sigma_hat, device=device)
    _check_same_shape(Sigma_true, Sigma_hat)

    num = torch.linalg.norm(Sigma_hat - Sigma_true, ord="fro")
    den = torch.linalg.norm(Sigma_true, ord="fro").clamp_min(eps)
    return (num / den).item()


def covariance_operator_error(
    Sigma_true: TensorLike,
    Sigma_hat: TensorLike,
    device: str = None
) -> float:
    """
    Spectral/operator norm error: ||Sigma_hat - Sigma_true||_2
    """
    Sigma_true = _to_torch_float64(Sigma_true, device=device)
    Sigma_hat = _to_torch_float64(Sigma_hat, device=device)
    _check_same_shape(Sigma_true, Sigma_hat)

    diff = Sigma_hat - Sigma_true
    return torch.linalg.svdvals(diff).max().item()


def precision_frobenius_error(
    Sigma_true: TensorLike,
    Sigma_hat: TensorLike,
    device: str = None,
    check_spd: bool = True
) -> float:
    """
    ||Sigma_hat^{-1} - Sigma_true^{-1}||_F
    """
    Sigma_true = _to_torch_float64(Sigma_true, device=device)
    Sigma_hat = _to_torch_float64(Sigma_hat, device=device)
    _check_same_shape(Sigma_true, Sigma_hat)

    if check_spd:
        _check_spd(Sigma_true, "Sigma_true")
        _check_spd(Sigma_hat, "Sigma_hat")

    K_true = torch.linalg.inv(Sigma_true)
    K_hat = torch.linalg.inv(Sigma_hat)
    return torch.linalg.norm(K_hat - K_true, ord="fro").item()


def precision_relative_frobenius_error(
    Sigma_true: TensorLike,
    Sigma_hat: TensorLike,
    device: str = None,
    check_spd: bool = True,
    eps: float = 1e-12
) -> float:
    """
    ||Sigma_hat^{-1} - Sigma_true^{-1}||_F / ||Sigma_true^{-1}||_F
    """
    Sigma_true = _to_torch_float64(Sigma_true, device=device)
    Sigma_hat = _to_torch_float64(Sigma_hat, device=device)
    _check_same_shape(Sigma_true, Sigma_hat)

    if check_spd:
        _check_spd(Sigma_true, "Sigma_true")
        _check_spd(Sigma_hat, "Sigma_hat")

    K_true = torch.linalg.inv(Sigma_true)
    K_hat = torch.linalg.inv(Sigma_hat)
    num = torch.linalg.norm(K_hat - K_true, ord="fro")
    den = torch.linalg.norm(K_true, ord="fro").clamp_min(eps)
    return (num / den).item()


def logdet_error(
    Sigma_true: TensorLike,
    Sigma_hat: TensorLike,
    device: str = None,
    check_spd: bool = True
) -> float:
    """
    |log|Sigma_hat| - log|Sigma_true||
    """
    Sigma_true = _to_torch_float64(Sigma_true, device=device)
    Sigma_hat = _to_torch_float64(Sigma_hat, device=device)
    _check_same_shape(Sigma_true, Sigma_hat)

    if check_spd:
        _check_spd(Sigma_true, "Sigma_true")
        _check_spd(Sigma_hat, "Sigma_hat")

    sign_t, logdet_true = torch.linalg.slogdet(Sigma_true)
    sign_h, logdet_hat = torch.linalg.slogdet(Sigma_hat)

    if sign_t <= 0:
        raise ValueError("Sigma_true has non-positive determinant.")
    if sign_h <= 0:
        raise ValueError("Sigma_hat has non-positive determinant.")

    return torch.abs(logdet_hat - logdet_true).item()


def gaussian_kl_true_to_hat(
    Sigma_true: TensorLike,
    Sigma_hat: TensorLike,
    device: str = None,
    check_spd: bool = True
) -> float:
    """
    KL( N(0, Sigma_true) || N(0, Sigma_hat) )

    = 0.5 * [ tr(Sigma_hat^{-1} Sigma_true) - p + log|Sigma_hat| - log|Sigma_true| ]
    """
    Sigma_true = _to_torch_float64(Sigma_true, device=device)
    Sigma_hat = _to_torch_float64(Sigma_hat, device=device)
    _check_same_shape(Sigma_true, Sigma_hat)

    if check_spd:
        _check_spd(Sigma_true, "Sigma_true")
        _check_spd(Sigma_hat, "Sigma_hat")

    p = Sigma_true.shape[0]

    sign_t, logdet_true = torch.linalg.slogdet(Sigma_true)
    sign_h, logdet_hat = torch.linalg.slogdet(Sigma_hat)

    if sign_t <= 0 or sign_h <= 0:
        raise ValueError("One of the matrices has non-positive determinant.")

    trace_term = torch.trace(torch.linalg.solve(Sigma_hat, Sigma_true))
    kl = 0.5 * (trace_term - p + (logdet_hat - logdet_true))
    return kl.item()


def gaussian_kl_hat_to_true(
    Sigma_true: TensorLike,
    Sigma_hat: TensorLike,
    device: str = None,
    check_spd: bool = True
) -> float:
    """
    KL( N(0, Sigma_hat) || N(0, Sigma_true) )
    """
    return gaussian_kl_true_to_hat(
        Sigma_true=Sigma_hat,
        Sigma_hat=Sigma_true,
        device=device,
        check_spd=check_spd
    )


def gaussian_symmetric_kl(
    Sigma_true: TensorLike,
    Sigma_hat: TensorLike,
    device: str = None,
    check_spd: bool = True
) -> float:
    """
    Symmetric KL = KL(true || hat) + KL(hat || true)
    """
    kl1 = gaussian_kl_true_to_hat(
        Sigma_true=Sigma_true,
        Sigma_hat=Sigma_hat,
        device=device,
        check_spd=check_spd
    )
    kl2 = gaussian_kl_hat_to_true(
        Sigma_true=Sigma_true,
        Sigma_hat=Sigma_hat,
        device=device,
        check_spd=check_spd
    )
    return kl1 + kl2


def eigenvalue_l2_error(
    Sigma_true: TensorLike,
    Sigma_hat: TensorLike,
    device: str = None
) -> float:
    """
    ||eig(Sigma_hat) - eig(Sigma_true)||_2
    Uses sorted eigenvalues from eigvalsh.
    """
    Sigma_true = _to_torch_float64(Sigma_true, device=device)
    Sigma_hat = _to_torch_float64(Sigma_hat, device=device)
    _check_same_shape(Sigma_true, Sigma_hat)

    evals_true = torch.linalg.eigvalsh(Sigma_true)
    evals_hat = torch.linalg.eigvalsh(Sigma_hat)
    return torch.linalg.norm(evals_hat - evals_true, ord=2).item()


def eigenvalue_relative_l2_error(
    Sigma_true: TensorLike,
    Sigma_hat: TensorLike,
    device: str = None,
    eps: float = 1e-12
) -> float:
    """
    ||eig(Sigma_hat) - eig(Sigma_true)||_2 / ||eig(Sigma_true)||_2
    """
    Sigma_true = _to_torch_float64(Sigma_true, device=device)
    Sigma_hat = _to_torch_float64(Sigma_hat, device=device)
    _check_same_shape(Sigma_true, Sigma_hat)

    evals_true = torch.linalg.eigvalsh(Sigma_true)
    evals_hat = torch.linalg.eigvalsh(Sigma_hat)

    num = torch.linalg.norm(evals_hat - evals_true, ord=2)
    den = torch.linalg.norm(evals_true, ord=2).clamp_min(eps)
    return (num / den).item()


def evaluate_covariance_estimator(
    Sigma_true: TensorLike,
    Sigma_hat: TensorLike,
    device: str = None,
    check_spd: bool = True
) -> Dict[str, float]:
    """
    Return all standard covariance-estimation distances in one dictionary.
    """
    Sigma_true = _to_torch_float64(Sigma_true, device=device)
    Sigma_hat = _to_torch_float64(Sigma_hat, device=device)
    _check_same_shape(Sigma_true, Sigma_hat)

    results = {
        "cov_fro_error": covariance_frobenius_error(Sigma_true, Sigma_hat, device=device),
        "cov_rel_fro_error": covariance_relative_frobenius_error(Sigma_true, Sigma_hat, device=device),
        "cov_operator_error": covariance_operator_error(Sigma_true, Sigma_hat, device=device),
        "eig_l2_error": eigenvalue_l2_error(Sigma_true, Sigma_hat, device=device),
        "eig_rel_l2_error": eigenvalue_relative_l2_error(Sigma_true, Sigma_hat, device=device),
    }

    if check_spd:
        results.update({
            "precision_fro_error": precision_frobenius_error(Sigma_true, Sigma_hat, device=device, check_spd=True),
            "precision_rel_fro_error": precision_relative_frobenius_error(Sigma_true, Sigma_hat, device=device, check_spd=True),
            "logdet_error": logdet_error(Sigma_true, Sigma_hat, device=device, check_spd=True),
            "kl_true_to_hat": gaussian_kl_true_to_hat(Sigma_true, Sigma_hat, device=device, check_spd=True),
            "kl_hat_to_true": gaussian_kl_hat_to_true(Sigma_true, Sigma_hat, device=device, check_spd=True),
            "symmetric_kl": gaussian_symmetric_kl(Sigma_true, Sigma_hat, device=device, check_spd=True),
        })

    return results



if __name__ == "__main__":
    # Example usage
    config = {
        'n0': 50,
        'n1': 55,
        'n2': 70,
        'n_samples': 500,
        'seed': 42,
        'device': 'cpu',
        'white_normalize': True,
        'alpha_list': [0.05,0.1,0.2,0.5,0.8,0.9,0.95],
        'evaluation_func': logdet_error
    }

    true_cov_m8, true_cov_m7 = make_random_true_cov(config, n0=config['n0'], n1=config['n1'], n2=config['n2'], rng=torch.Generator(device=config['device']).manual_seed(config['seed']))
    results_dict = shrinkage_m7_m8_simulation(config, data=(true_cov_m8, true_cov_m7))
    evaluation_results = evaluate_shrinkage(config, results_dict)
    
    for model in ['M7', 'M8']:
        print(f"\nEvaluation results for {model}:")
        for method, error in evaluation_results[model].items():
            print(f"  {method}: {min(error) if isinstance(error, list) else error:.4f}")