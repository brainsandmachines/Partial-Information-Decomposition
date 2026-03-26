import torch 
import numpy as np
import matplotlib.pyplot as plt
import sys
import os 
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from PID_util import *




def _build_whitened_blocks_from_cov(S, n0, n1, n2):
    """
    Given a covariance matrix S in block order [X0, X1, Y],
    return the whitened blocks P, Q, R using the same helpers as the main code.
    """
    S_torch = torch.from_numpy(S).to(torch.float64)
    S_dict = create_cov_matrix(Sigma=S_torch, dims=[n0, n1, n2])

    P = whiten_block(S_dict["cov_x0"], S_dict["cross_x0_x1"], S_dict["cov_x1"]).numpy()
    Q = whiten_block(S_dict["cov_x0"], S_dict["cross_x0_x2"], S_dict["cov_x2"]).numpy()
    R = whiten_block(S_dict["cov_x1"], S_dict["cross_x1_x2"], S_dict["cov_x2"]).numpy()
    return P, Q, R, S_dict


def make_random_true_cov(
    n0: int,
    n1: int,
    n2: int,
    q_scale: float = 0.25,
    r_scale: float = 0.25,
    p_scale: float = 0.25,
    rng: np.random.Generator | None = None,
    m7_whiten_structural: bool = True,
) -> np.ndarray:
    """
    Construct a generic positive-definite Gaussian M7_whiten and M8_Whiten covariance.

    Block order is [X0, X1, Y].

    m7_whiten population structure:
        Sigma_01 = Sigma_02 @ Sigma_22^{-1} @ Sigma_21
    and here Sigma_22 = I, so:
        P = Q @ R.T
    """

    A = rng.standard_normal((n0, n2))
    B = rng.standard_normal((n1, n2))
    C =  rng.standard_normal((n0, n1))

    A_norm = np.linalg.norm(A, ord=2)
    B_norm = np.linalg.norm(B, ord=2)
    C_norm = np.linalg.norm(C, ord=2)   
    if A_norm == 0 or B_norm == 0 or C_norm == 0:
        raise RuntimeError("Unexpected zero spectral norm in random construction.")

    Q = q_scale * A / A_norm
    R = r_scale * B / B_norm
    P = p_scale * C / C_norm

    # Construct M8 covariance - sample coavraiance after whitening each block
    true_cov_M8 = np.block([
        [np.eye(n0), P,          Q],
        [P.T,        np.eye(n1), R],
        [Q.T,        R.T,        np.eye(n2)]
    ])

    eigvals = np.linalg.eigvalsh(true_cov_M8)
    if np.min(eigvals) <= 1e-10:
        raise ValueError(
            f"Constructed covariance not sufficiently PD. min eig={np.min(eigvals):.3e}"
        )

    # Construct M7 covariance - sample coavraiance after whitening each block
    P_m7 = Q @ R.T
    true_cov_M7 = np.block([
        [np.eye(n0), P_m7,          Q],
        [P_m7.T,        np.eye(n1), R],
        [Q.T,        R.T,        np.eye(n2)]
    ])


    
    eigvals_m7 = np.linalg.eigvalsh(true_cov_M7)
    if np.min(eigvals_m7) <= 1e-10:
        raise ValueError(
            f"Constructed M7 covariance not sufficiently PD. min eig={np.min(eigvals_m7):.3e}"
        )

    # # Check precision-matrix m7_whiten condition: K_{X0,X1} = 0
    # K = np.linalg.inv(true_cov)
    # K01 = K[:n0, n0:n0+n1]
    # if not np.allclose(K01, 0, atol=1e-10):
    #     raise ValueError("Constructed covariance does not satisfy the m7_whiten precision condition.")

    return true_cov_M8, true_cov_M7




def simulation(config,functions_dict:dict,seed=None):
    """
    Run a simulation over combinations of N and p values, computing the specified statistic and bias correction.
    """
    results_dict = {}
    rng = np.random.default_rng(seed)

    s_simulation_func = functions_dict["s_simulation"]
    bias_correction_func = functions_dict["bias_correction"]
    corrected_statistic_func = functions_dict["corrected_statistic"]

    m8_true_cov, m7_true_cov = make_random_true_cov(config["n0"], config["n1"], config["n2"], q_scale=config["q_scale"], r_scale=config["r_scale"], p_scale=config["p_scale"], rng=rng)

    statistic = s_simulation_func(m8_true_cov, m7_true_cov,config['n_samples'] ,config["n0"], config["n1"], config["n2"], config["n_trials"], rng)

    for statistic_key in statistic.keys():
        print(f"Calculating bias for {statistic_key}...")
        statistic_model = statistic[statistic_key]

        
        model_config = config.copy()
        model_config['statistics'] = statistic_model
        
        model_bc_func = bias_correction_func[statistic_key]
        model_bias_correction = model_bc_func(model_config)

        model_corr_values = corrected_statistic_func(statistic_model['avg'], model_bias_correction['bias'])
        
        results_dict[statistic_key] = {
            'sample': statistic_model['sample'],
            'avg': statistic_model['avg'],
            'std': statistic_model['std'],
            'emp_bias': model_bias_correction[f'bias'],
            'corrected_statistic': model_corr_values,
            'ground_truth': statistic_model['ground_truth']
        }


    return results_dict






