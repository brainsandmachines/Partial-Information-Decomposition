import torch 
import numpy as np
import matplotlib.pyplot as plt
import sys
import os 
from pathlib import Path


root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from PID_util import whiten_block, create_cov_matrix
from mi_functions import calcualte_mi



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
    config: dict,
    rng: torch.Generator | None = None,
):
    """
    Construct a whitened Gaussian M8 covariance and its corresponding M7 covariance.

    New config parameters:
        mode: one of {"exact_m7", "m8_side", "m7_side"}
        delta: size of perturbation away from P_m7 = Q @ R.T
        max_tries: retry budget to find a covariance in the requested regime
        delta_margin: minimum separation in MI gap to accept the sample

    Notes:
        - "exact_m7" gives P = Q @ R.T exactly.
        - "m8_side" means keep only draws with I_M8 - I_M7 < 0.
        - "m7_side" means keep only draws with I_M8 - I_M7 > 0.
    """
    q_scale = config["q_scale"]
    r_scale = config["r_scale"]
    n0 = config["n0"]
    n1 = config["n1"]
    n2 = config["n2"]
    device = config["device"]

    mode = config.get("mode", "exact_m7")       # "exact_m7", "m8_side", "m7_side"
    delta = config.get("delta", 0.0)            # perturbation size
    max_tries = config.get("max_tries", 1000)
    delta_margin = config.get("delta_margin", 1e-3)
    alpha = config.get("alpha", 0.5)            # only used for shrinkage versions, ignored otherwise
    dtype = torch.float64

    for _ in range(max_tries):
        A = torch.randn((n0, n2), generator=rng, dtype=dtype, device=device)
        B = torch.randn((n1, n2), generator=rng, dtype=dtype, device=device)
        C = torch.randn((n0, n1), generator=rng, dtype=dtype, device=device)

        A_norm = torch.linalg.norm(A, ord=2)
        B_norm = torch.linalg.norm(B, ord=2)
        C_norm = torch.linalg.norm(C, ord=2)

        if A_norm == 0 or B_norm == 0 or C_norm == 0:
            continue

        Q = q_scale * A / A_norm
        R = r_scale * B / B_norm

        # Exact M7 block
        P_m7 = Q @ R.T

        # Choose P according to desired regime
        if mode == "exact_m7":
            P = P_m7
        elif mode in {"m8_side", "m7_side"}:
            D = C / C_norm
            P = alpha*P_m7 + delta * D
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Build M8 covariance
        row1_m8 = torch.cat([torch.eye(n0, device=device, dtype=dtype), P, Q], dim=1)
        row2_m8 = torch.cat([P.T, torch.eye(n1, device=device, dtype=dtype), R], dim=1)
        row3_m8 = torch.cat([Q.T, R.T, torch.eye(n2, device=device, dtype=dtype)], dim=1)
        true_cov_m8 = torch.cat([row1_m8, row2_m8, row3_m8], dim=0)

        # Check PD for M8
        eigvals_m8 = torch.linalg.eigvalsh(true_cov_m8)
        if torch.min(eigvals_m8) <= 1e-10:
            continue

        # Build M7 covariance from your existing helper
        true_cov_m7 = create_m7_cov(config, true_cov_m8, whitening_normalize=True)

        # Check PD for M7
        eigvals_m7 = torch.linalg.eigvalsh(true_cov_m7)
        if torch.min(eigvals_m7) <= 1e-10:
            continue

        # exact M7 case: accept immediately
        if mode == "exact_m7":
            return true_cov_m8, true_cov_m7

        # Otherwise check which side won
        try:
            dict_terms_M8 = {'P': P, 'Q': Q, 'R': R, 'Sigma': true_cov_m8}
            dict_terms_M7 = {'P': P_m7, 'Q': Q, 'R': R, 'Sigma': true_cov_m7}
            mi_m8 = calcualte_mi(config, dict_terms_M8, term='mi_tri')['mi_tri']
            mi_m7 = calcualte_mi(config, dict_terms_M7, term='mi_tri')['mi_tri']
            delta_mi = mi_m8 - mi_m7
        except RuntimeError:
            continue

        if mode == "m8_side" and delta_mi < -delta_margin:
            return true_cov_m8, true_cov_m7

        if mode == "m7_side" and delta_mi > delta_margin:
            return true_cov_m8, true_cov_m7

    return true_cov_m8, true_cov_m7  # return the last one even if it doesn't meet criteria after max_tries

# def make_random_true_cov(
#     config: dict,
#     rng:  torch.Generator | None = None,
#     m7_whiten_structural: bool = True,
# ) -> np.ndarray:
#     """
#     Construct a generic positive-definite Gaussian M7_whiten and M8_Whiten covariance.

#     Block order is [X0, X1, Y].

#     m7_whiten population structure:
#         Sigma_01 = Sigma_02 @ Sigma_22^{-1} @ Sigma_21
#     and here Sigma_22 = I, so:
#         P = Q @ R.T
#     """
#     q_scale = config['q_scale']
#     r_scale = config['r_scale']
#     p_scale = config['p_scale']
#     n0 = config['n0']
#     n1 = config['n1']
#     n2 = config['n2']
#     A = torch.randn((n0, n2), generator=rng, dtype=torch.float64,device=config['device'])
#     B = torch.randn((n1, n2), generator=rng, dtype=torch.float64,device=config['device'])
#     C = torch.randn((n0, n1), generator=rng, dtype=torch.float64,device=config['device'])

#     A_norm = torch.linalg.norm(A, ord=2)
#     B_norm = torch.linalg.norm(B, ord=2)
#     C_norm = torch.linalg.norm(C, ord=2)   
#     if A_norm == 0 or B_norm == 0 or C_norm == 0:
#         raise RuntimeError("Unexpected zero spectral norm in random construction.")

#     Q = q_scale * A / A_norm
#     R = r_scale * B / B_norm
#     P = p_scale * C / C_norm

#     # Construct M8 covariance - sample coavraiance after whitening each block
#     row1_m8 = torch.cat([torch.eye(n0, device=config['device']), P, Q], dim=1)
#     row2_m8 = torch.cat([P.T, torch.eye(n1, device=config['device']), R], dim=1)
#     row3_m8 = torch.cat([Q.T, R.T, torch.eye(n2, device=config['device'])], dim=1)   
#     true_cov_m8 = torch.cat([row1_m8, row2_m8, row3_m8], dim=0)

#     eigvals = torch.linalg.eigvalsh(true_cov_m8)
#     if torch.min(eigvals) <= 1e-10:
#         raise ValueError(
#             f"Constructed covariance not sufficiently PD. min eig={torch.min(eigvals):.3e}"
#         )
 
#     true_cov_m7 = create_m7_cov(config,true_cov_m8, whitening_normalize=True)


    
#     eigvals_m7 = torch.linalg.eigvalsh(true_cov_m7)
#     if torch.min(eigvals_m7) <= 1e-10:
#         raise ValueError(
#             f"Constructed M7 covariance not sufficiently PD. min eig={torch.min(eigvals_m7):.3e}"
#         )

#     return true_cov_m8, true_cov_m7


def create_m7_cov(config:dict,cov_m8,whitening_normalize:bool = True):
    "Takes covariance of M8 and creates M7 out of M8"

    cov_m8_dict = create_cov_matrix(Sigma=cov_m8, dims=[config['n0'], config['n1'], config['n2']])
    diag0 = torch.eye(config['n0'],device=config['device']) if whitening_normalize else cov_m8_dict["cov_x1"]
    diag1 = torch.eye(config['n1'],device=config['device']) if whitening_normalize else cov_m8_dict["cov_x2"]
    diag2 = torch.eye(config['n2'],device=config['device']) if whitening_normalize else cov_m8_dict["cov_xt"]
    if whitening_normalize: 
        Q = whiten_block(cov_m8_dict["cov_x1"], cov_m8_dict["cross_x1_xt"], cov_m8_dict["cov_xt"])
        R = whiten_block(cov_m8_dict["cov_x2"], cov_m8_dict["cross_x2_xt"], cov_m8_dict["cov_xt"])
        P = Q @ R.T

    else:
        Q = cov_m8_dict["cross_x1_xt"]
        R = cov_m8_dict["cross_x2_xt"]
        P = Q @ torch.linalg.inv(cov_m8_dict["cov_xt"]) @ R.T

    #Construct M7 covariance 
    row1 = torch.cat([diag0, P, Q], dim=1)
    row2 = torch.cat([P.T, diag1, R], dim=1)
    row3 = torch.cat([Q.T, R.T, diag2], dim=1)
    return torch.cat([row1, row2, row3], dim=0)

def simulation(config,functions_dict:dict,seed=None):
    """
    Run a simulation over combinations of N and p values, computing the specified statistic and bias correction.
    """
    results_dict = {}
    device = config['device']
    rng = torch.Generator(device=device).manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    bias_method = config['bias_method']
    s_simulation_func = functions_dict["s_simulation"]
    bias_correction_func = functions_dict["bias_correction"]
    corrected_statistic_func = functions_dict["corrected_statistic"]

    m8_true_cov, m7_true_cov = make_random_true_cov(config, rng=rng)

    data = [m8_true_cov, m7_true_cov]
    sim_config = config.copy()
    sim_config['bias_correction_func'] = bias_correction_func
    statistic = s_simulation_func(data, sim_config, rng)

    for statistic_key in statistic.keys():
        print(f"Finishing for {statistic_key}...")
        statistic_model = statistic[statistic_key]

        
        model_config = config.copy()
        model_config['statistics'] = statistic_model

        if  f"corrected_avg" in statistic_model:
            model_corr_values = statistic_model['corrected_avg']
        
        else:
            if bias_method[0] == 'analytic':
                model_bc_func = bias_correction_func[statistic_key]
                model_bias_correction = model_bc_func(model_config)
                
                model_corr_values = corrected_statistic_func(statistic_model['avg'], model_bias_correction['bias'])
            
            else: 
                    model_corr_values = statistic_model['avg_resample']

       
                
        results_dict[statistic_key] = {
            'sample': statistic_model['sample'],
            'avg': statistic_model['avg'],
            'std': statistic_model['std'],
            'emp_bias': statistic_model['emp_bias'],
            'after_corr_bias': statistic_model.get('after_corr_bias', 100000),
            'corrected_statistic': model_corr_values,
            'ground_truth': statistic_model['ground_truth'],
            'var': statistic_model['var'],
            'mse': statistic_model['mse'],
        }


    return results_dict






