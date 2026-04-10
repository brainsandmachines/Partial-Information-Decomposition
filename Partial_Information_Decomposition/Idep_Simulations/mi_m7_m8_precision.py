import pathlib
import torch
import numpy as np
import yaml
from functools import partial

from Simulation_utils import *
from wrapper_M7_M8_models import simulation
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

# Optional imports from the user's larger project. Fallbacks keep this file self-contained.
try:
    from Partial_Information_Decomposition.resampling_wrapper import bias_resampling  # type: ignore
except Exception:
    def bias_resampling(config):
        st = config.get('st', 'bias')
        return {st: 0.0}


def _as_batched_cov(S: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return covariance as (B,d,d) and a flag saying whether the input was already batched."""
    if S.ndim == 2:
        return S.unsqueeze(0), False
    if S.ndim == 3:
        return S, True
    raise ValueError(f"Sigma must be 2D or 3D, got shape {tuple(S.shape)}")


def _first_available(d: dict, *keys: str):
    for key in keys:
        if key in d:
            return d[key]
    raise KeyError(f"None of the keys {keys} found. Available keys: {list(d.keys())}")


def _m7_inverse_df_checks(df: int, n0: int, n1: int, n2: int, need_variance: bool = False) -> None:
    """
    Validate df requirements for inverse-covariance and inverse-covariance variance formulas.

    For unbiased inverse covariance mean correction we need df > p + 1.
    For the inverse-Wishart variance formula used in the delta-method correction we need df > p + 3.
    """
    required_gap = 3 if need_variance else 1
    req_02 = n0 + n2 + required_gap
    req_12 = n1 + n2 + required_gap
    req_t = n2 + required_gap
    needed = max(req_02, req_12, req_t)
    strict_needed = needed + 1
    if df < strict_needed:
        context = "variance/delta" if need_variance else "inverse-mean"
        raise ValueError(
            f"Need df > max(d0+dT+{required_gap}, d1+dT+{required_gap}, dT+{required_gap}) "
            f"for M7 {context} formulas. Got df={df}, need at least {strict_needed}."
        )


def _select_tt_covariance_tensor(cov4: torch.Tensor, n2: int) -> torch.Tensor:
    """Select the T,T block covariance tensor from a full 4D covariance tensor."""
    p = cov4.shape[0]
    idx = torch.arange(p - n2, p, device=cov4.device)
    out = cov4.index_select(0, idx)
    out = out.index_select(1, idx)
    out = out.index_select(2, idx)
    out = out.index_select(3, idx)
    return out


def _inverse_covariance_covariance_unbiased(K_u: torch.Tensor, df: int) -> torch.Tensor:
    """
    Approximate Cov(U_ij, U_kl) for the unbiased inverse-covariance estimator U.

    If S is the unbiased sample covariance of a p-variate Gaussian sample with df=N-1,
    then the unbiased inverse-covariance estimator is U = ((df-p-1)/df) * S^{-1}.

    Using the standard inverse-Wishart second-moment formula and plugging in K_u for the
    unknown true precision gives the approximation below.

    Returns a 4D tensor C with C[i,j,k,l] = Cov(U_ij, U_kl).
    """
    p = K_u.shape[0]
    if df <= p + 3:
        raise ValueError(f"Need df > p+3 for inverse-covariance variance formula. Got df={df}, p={p}.")

    denom = (df - p) * (df - p - 3)
    coeff = (df - p - 1)

    term1 = 2.0 * K_u[:, :, None, None] * K_u[None, None, :, :]
    term2a = K_u[:, None, :, None] * K_u[None, :, None, :]
    term2b = K_u[:, None, None, :] * K_u.T[None, :, :, None]
    cov4 = (term1 + coeff * (term2a + term2b)) / denom
    return cov4


def _m7_precision_components_from_sample(S: torch.Tensor, n0: int, n1: int, n2: int, df: int) -> dict:
    """
    Build raw and unbiased clique/separator precision pieces for the M7 decomposable graph.

    For each clique/separator C, with dimension p_C:
        E[S_C^{-1}] = (df / (df - p_C - 1)) * Sigma_C^{-1}
    so the unbiased inverse-covariance estimator is
        U_C = ((df - p_C - 1) / df) * S_C^{-1}.

    We return both raw and unbiased T,T pieces, together with approximate covariance tensors.
    """
    _m7_inverse_df_checks(df, n0, n1, n2, need_variance=True)

    S = S.to(torch.float64)
    S_dict = create_cov_matrix(Sigma=S, dims=[n0, n1, n2])

    S_0T = S_dict['auto_x02']
    S_1T = _first_available(S_dict, 'joint_x1_x2', 'auto_x12')
    S_T = S_dict['cov_x2']

    p_0T = n0 + n2
    p_1T = n1 + n2
    p_T = n2

    inv_0T = torch.linalg.inv(S_0T)
    inv_1T = torch.linalg.inv(S_1T)
    inv_T = torch.linalg.inv(S_T)

    s0 = df / (df - p_0T - 1)
    s1 = df / (df - p_1T - 1)
    st = df / (df - p_T - 1)

    a0 = 1.0 / s0
    a1 = 1.0 / s1
    at = 1.0 / st

    K_0T_u_full = a0 * inv_0T
    K_1T_u_full = a1 * inv_1T
    K_T_u_full = at * inv_T

    A_raw = inv_0T[-n2:, -n2:]
    B_raw = inv_1T[-n2:, -n2:]
    C_raw = inv_T

    A_u = K_0T_u_full[-n2:, -n2:]
    B_u = K_1T_u_full[-n2:, -n2:]
    C_u = K_T_u_full

    Ktt_raw = A_raw + B_raw - C_raw
    Ktt_u = A_u + B_u - C_u

    cov_A_u = _select_tt_covariance_tensor(_inverse_covariance_covariance_unbiased(K_0T_u_full, df), n2)
    cov_B_u = _select_tt_covariance_tensor(_inverse_covariance_covariance_unbiased(K_1T_u_full, df), n2)
    cov_C_u = _inverse_covariance_covariance_unbiased(K_T_u_full, df)

    cov_A_raw = (s0 ** 2) * cov_A_u
    cov_B_raw = (s1 ** 2) * cov_B_u
    cov_C_raw = (st ** 2) * cov_C_u

    return {
        'A_raw': A_raw, 'B_raw': B_raw, 'C_raw': C_raw,
        'A_u': A_u, 'B_u': B_u, 'C_u': C_u,
        'Ktt_raw': Ktt_raw, 'Ktt_u': Ktt_u,
        'cov_A_raw': cov_A_raw, 'cov_B_raw': cov_B_raw, 'cov_C_raw': cov_C_raw,
        'cov_A_u': cov_A_u, 'cov_B_u': cov_B_u, 'cov_C_u': cov_C_u,
        'scale_raw_0T': s0, 'scale_raw_1T': s1, 'scale_raw_T': st,
    }


def _delta_logdet_second_order_term(K_plug: torch.Tensor, cov4: torch.Tensor) -> torch.Tensor:
    """
    Second-order delta-method contribution for log|K|:
        -1/2 E tr(K^{-1} E K^{-1} E)
    """
    K_plug = 0.5 * (K_plug + K_plug.mT)
    invK = torch.linalg.inv(K_plug)
    quad = torch.einsum('ij,kl,jkli->', invK, invK, cov4)
    return -0.5 * quad


def _delta_logdet_bias_from_bias_and_covariance(K_plug: torch.Tensor, bias_mat: torch.Tensor, cov4: torch.Tensor) -> torch.Tensor:
    """
    First + second order delta-method bias approximation for log|K_hat| around K_plug:
        tr(K^{-1} Bias[K_hat]) - 1/2 E tr(K^{-1} E K^{-1} E)
    where E has mean zero and covariance cov4.
    """
    K_plug = 0.5 * (K_plug + K_plug.mT)
    bias_mat = 0.5 * (bias_mat + bias_mat.mT)
    invK = torch.linalg.inv(K_plug)
    first = torch.trace(invK @ bias_mat)
    second = _delta_logdet_second_order_term(K_plug, cov4)
    return first + second


def _m7_precision_mi_bias(config: dict) -> dict:
    """
    Bias approximation for the *raw* M7 joint mutual information estimator.

    The raw M7 MI equals
        0.5 * log|S_T| + 0.5 * log|K_TT,MLE^{(7)}|
    where K_TT,MLE^{(7)} is the T,T block of the clique/separator MLE concentration.

    We approximate the bias of the second term by a delta method using:
    - exact matrix bias of the raw clique/separator inverse-covariance blocks,
    - approximate covariance tensors (cross-covariances ignored),
    - unbiased precision pieces as plug-in targets.
    """
    n0 = config['n0']
    n1 = config['n1']
    n2 = config['n2']
    df = config['n_samples'] - 1
    S = config['Sigma']
    if S.ndim == 3:
        if S.shape[0] != 1:
            raise ValueError("M7 precision-delta bias expects a single sample covariance matrix.")
        S = S.squeeze(0)

    comps = _m7_precision_components_from_sample(S, n0, n1, n2, df)

    K_plug = comps['Ktt_u']
    bias_mat = ((comps['scale_raw_0T'] - 1.0) * comps['A_u']
                + (comps['scale_raw_1T'] - 1.0) * comps['B_u']
                - (comps['scale_raw_T'] - 1.0) * comps['C_u'])

    cov_Ktt_raw = comps['cov_A_raw'] + comps['cov_B_raw'] + comps['cov_C_raw']

    bias_logdet_ktt = _delta_logdet_bias_from_bias_and_covariance(K_plug, bias_mat, cov_Ktt_raw)
    bias_logdet_st = torch.as_tensor(logdet_wishart_bias(df, n2), dtype=torch.float64, device=S.device)

    mi_bias = 0.5 * bias_logdet_st + 0.5 * bias_logdet_ktt
    return {'mi': mi_bias.item()}


def _m7_precision_mi_from_cov(S: torch.Tensor, n0: int, n1: int, n2: int, n_samples: int) -> torch.Tensor:
    """
    Exact identity for the *raw* M7 estimator:
        I_M7(raw) = 0.5 * log|S_T| + 0.5 * log|K_TT,MLE^{(7)}|
    where K_TT,MLE^{(7)} is built from the clique/separator MLE concentration.
    """
    df = n_samples - 1
    comps = _m7_precision_components_from_sample(S, n0, n1, n2, df)
    S_dict = create_cov_matrix(Sigma=S, dims=[n0, n1, n2])
    cov_t = S_dict['cov_x2']
    mi_val = 0.5 * safe_logdet(cov_t) + 0.5 * safe_logdet(comps['Ktt_raw'])
    return mi_val


def mi_calculation_not_whiten_precision_delta(config) -> float:
    """
    Compute MI statistics.

    M8 is left exactly as in the original file.
    For M7, only the joint MI is replaced by the precision-based estimator using the
    unbiased K_TT estimator; the auxiliary nume/joint/target/deno fields are kept in the
    original covariance-based form so that the rest of the pipeline keeps working.
    """
    n0 = config['n0']
    n1 = config['n1']
    n2 = config['n2']
    n_samples = config['n_samples']
    S_in = config['Sigma']

    S_batch, was_batched = _as_batched_cov(S_in)
    S_dict = para_create_cov_matrix([n0, n1, n2], S_batch)

    if config['model'] == 'M8' or config['model'] == 'M8_M7':
        m8_sigma = S_batch
        deno8_raw = 0.5 * safe_logdet(m8_sigma)
        joint_x0_x1 = _first_available(S_dict, 'joint_x0_x1', 'auto_x01')
        cov_x2 = S_dict['cov_x2']
        nume_m8_joint_raw = 0.5 * safe_logdet(joint_x0_x1)
        nume_m8_target_raw = 0.5 * safe_logdet(cov_x2)
        nume8_raw = nume_m8_joint_raw + nume_m8_target_raw
        mi_m8_raw = nume8_raw - deno8_raw
        final_dict_m8 = {
            'mi': mi_m8_raw,
            'nume': nume8_raw,
            'nume_joint': nume_m8_joint_raw,
            'nume_target': nume_m8_target_raw,
            'deno': deno8_raw,
        }

    if config['model'] == 'M7' or config['model'] == 'M8_M7':
        cross_x0_x1_m7 = S_dict['cross_x0_x2'] @ torch.linalg.inv(S_dict['cov_x2']) @ S_dict['cross_x1_x2'].mT
        cross_x1_x0_m7 = cross_x0_x1_m7.mT
        S_m7 = S_batch.clone()
        S_m7[:, :n0, n0:n0+n1] = cross_x0_x1_m7
        S_m7[:, n0:n0+n1, :n0] = cross_x1_x0_m7

        S_m7_dict = para_create_cov_matrix([n0, n1, n2], S_m7)
        deno7_raw = 0.5 * safe_logdet(S_m7)
        nume_m7_joint_raw = 0.5 * safe_logdet(_first_available(S_m7_dict, 'joint_x0_x1', 'auto_x01'))
        nume_m7_target_raw = 0.5 * safe_logdet(S_m7_dict['cov_x2'])
        nume7_raw = nume_m7_joint_raw + nume_m7_target_raw

        # Keep the raw M7 estimator exactly as in the original file.
        mi_m7_raw = nume7_raw - deno7_raw

        final_dict_m7 = {
            'mi': mi_m7_raw,
            'nume': nume7_raw,
            'nume_joint': nume_m7_joint_raw,
            'nume_target': nume_m7_target_raw,
            'deno': deno7_raw,
        }

    if config['model'] == 'M8_M7':
        final_dict = {'M8': final_dict_m8, 'M7': final_dict_m7}
    else:
        final_dict = final_dict_m8 if config['model'] == 'M8' else final_dict_m7

    if not was_batched and config['model'] != 'M8_M7':
        # Keep behavior close to the original code for single matrices.
        for key, value in final_dict.items():
            if isinstance(value, torch.Tensor) and value.ndim == 1 and value.shape[0] == 1:
                final_dict[key] = value.squeeze(0)

    return final_dict


def simulate_m7_m8_mi(
    data: list,
    sim_config: dict,
    rng: torch.Generator | None = None
):
    """
    Run MI simulation under the same covariance construction used in the logdet experiments.

    Relative to the original file:
    - M8 is unchanged.
    - M7 raw joint MI is unchanged; only its analytic bias correction is replaced.
    - result structure is preserved.
    """
    n_samples = sim_config['n_samples']
    n0 = sim_config['n0']
    n1 = sim_config['n1']
    n2 = sim_config['n2']
    n_trials = sim_config['n_trials']
    device = sim_config['device']

    if n_samples < 3:
        raise ValueError("Need at least 3 samples.")

    d = n0 + n1 + n2
    df = n_samples - 1
    if df <= d - 1:
        raise ValueError(
            f"Need df > d-1 for stable logdet expectation. Got n_samples={n_samples}, df={df}, d={d}."
        )

    m8_true_cov, m7_true_cov = data

    # Ground truth M8 as before.
    m8_true_cov_dict = create_cov_matrix(Sigma=m8_true_cov, dims=[n0, n1, n2])
    deno8_true = 0.5 * safe_logdet(m8_true_cov)
    joint_x0_x1 = _first_available(m8_true_cov_dict, 'joint_x0_x1', 'auto_x01')
    cov_x2 = m8_true_cov_dict['cov_x2']
    nume8_joint_true = 0.5 * safe_logdet(joint_x0_x1)
    nume8_target_true = 0.5 * safe_logdet(cov_x2)
    nume8_true = nume8_joint_true + nume8_target_true
    mi_m8_true = nume8_true - deno8_true

    # Ground truth M7: covariance and precision representations agree; keep the same target quantity.
    m7_true_cov_dict = create_cov_matrix(Sigma=m7_true_cov, dims=[n0, n1, n2])
    deno7_true = 0.5 * safe_logdet(m7_true_cov)
    nume7_joint_true = 0.5 * safe_logdet(_first_available(m7_true_cov_dict, 'joint_x0_x1', 'auto_x01'))
    nume7_target_true = 0.5 * safe_logdet(m7_true_cov_dict['cov_x2'])
    nume7_true = nume7_joint_true + nume7_target_true
    m7_true_precision = torch.linalg.inv(m7_true_cov)
    ktt_true_m7 = m7_true_precision[-n2:, -n2:]
    mi_m7_true = 0.5 * safe_logdet(m7_true_cov_dict['cov_x2']) + 0.5 * safe_logdet(ktt_true_m7)

    m8_dict_values = {'mi': [], 'nume': [], 'nume_joint': [], 'nume_target': [], 'deno': []}
    m7_dict_values = {'mi': [], 'nume': [], 'nume_joint': [], 'nume_target': [], 'deno': []}

    mi_m8_corrected = {'mi': [], 'nume': [], 'nume_joint': [], 'nume_target': [], 'deno': []}
    mi_m7_corrected = {'mi': [], 'nume': [], 'nume_joint': [], 'nume_target': [], 'deno': []}

    for i in range(n_trials):
        print(f"Trial {i+1}/{n_trials}", end="\r")

        S, rv_list = sample_data_from_cov(sim_config, true_cov=m8_true_cov, rng=rng)

        sim_config['Sigma'] = S.unsqueeze(0)
        sim_config['model'] = 'M8_M7'
        raw_results = mi_calculation_not_whiten_precision_delta(config=sim_config)

        mi_m8_raw = raw_results['M8']['mi']
        nume8_raw = raw_results['M8']['nume']
        nume_m8_joint_raw = raw_results['M8']['nume_joint']
        nume_m8_target_raw = raw_results['M8']['nume_target']
        deno8_raw = raw_results['M8']['deno']

        mi_m7_raw = raw_results['M7']['mi']
        nume7_raw = raw_results['M7']['nume']
        nume_m7_joint_raw = raw_results['M7']['nume_joint']
        nume_m7_target_raw = raw_results['M7']['nume_target']
        deno7_raw = raw_results['M7']['deno']

        sim_config_m8 = sim_config.copy()
        sim_config_m7 = sim_config.copy()
        sim_config_m8['model'] = 'M8'
        sim_config_m7['model'] = 'M7'

        sim_config_m8['sample_statistic'] = {
            'mi': mi_m8_raw,
            'nume': nume8_raw,
            'nume_joint': nume_m8_joint_raw,
            'nume_target': nume_m8_target_raw,
            'deno': deno8_raw,
        }
        sim_config_m7['sample_statistic'] = {
            'mi': mi_m7_raw,
            'nume': nume7_raw,
            'nume_joint': nume_m7_joint_raw,
            'nume_target': nume_m7_target_raw,
            'deno': deno7_raw,
        }

        sim_config_m8['calc_statistic_func'] = mi_calculation_not_whiten_precision_delta
        sim_config_m7['calc_statistic_func'] = mi_calculation_not_whiten_precision_delta
        sim_config_m8['rvs_list'] = rv_list
        sim_config_m7['rvs_list'] = rv_list
        sim_config_m8['Sigma'] = S
        sim_config_m7['Sigma'] = S

        m8_bias_dict = mi_bias_calc(sim_config_m8)
        m7_bias_dict = mi_bias_calc(sim_config_m7)

        mi_m8_corrected['mi'].append(mi_m8_raw - m8_bias_dict['mi'])
        mi_m8_corrected['nume'].append(nume8_raw - m8_bias_dict['nume'])
        mi_m8_corrected['nume_joint'].append(nume_m8_joint_raw - m8_bias_dict['nume_joint'])
        mi_m8_corrected['nume_target'].append(nume_m8_target_raw - m8_bias_dict['nume_target'])
        mi_m8_corrected['deno'].append(deno8_raw - m8_bias_dict['deno'])

        mi_m7_corrected['mi'].append(mi_m7_raw - m7_bias_dict['mi'])
        mi_m7_corrected['nume'].append(nume7_raw - m7_bias_dict['nume'])
        mi_m7_corrected['nume_joint'].append(nume_m7_joint_raw - m7_bias_dict['nume_joint'])
        mi_m7_corrected['nume_target'].append(nume_m7_target_raw - m7_bias_dict['nume_target'])
        mi_m7_corrected['deno'].append(deno7_raw - m7_bias_dict['deno'])

        m8_dict_values['mi'].append(mi_m8_raw)
        m8_dict_values['nume'].append(nume8_raw)
        m8_dict_values['nume_joint'].append(nume_m8_joint_raw)
        m8_dict_values['nume_target'].append(nume_m8_target_raw)
        m8_dict_values['deno'].append(deno8_raw)

        m7_dict_values['mi'].append(mi_m7_raw)
        m7_dict_values['nume'].append(nume7_raw)
        m7_dict_values['nume_joint'].append(nume_m7_joint_raw)
        m7_dict_values['nume_target'].append(nume_m7_target_raw)
        m7_dict_values['deno'].append(deno7_raw)

    mi_m8_sample = torch.tensor(m8_dict_values['mi'])
    nume_m8_sample = torch.tensor(m8_dict_values['nume'])
    nume_m8_joint_sample = torch.tensor(m8_dict_values['nume_joint'])
    nume_m8_target_sample = torch.tensor(m8_dict_values['nume_target'])
    deno_m8_sample = torch.tensor(m8_dict_values['deno'])
    mi_m7_sample = torch.tensor(m7_dict_values['mi'])
    nume_m7_sample = torch.tensor(m7_dict_values['nume'])
    nume_m7_joint_sample = torch.tensor(m7_dict_values['nume_joint'])
    nume_m7_target_sample = torch.tensor(m7_dict_values['nume_target'])
    deno_m7_sample = torch.tensor(m7_dict_values['deno'])

    avg_m8_mi = torch.mean(mi_m8_sample)
    avg_m8_nume = torch.mean(nume_m8_sample)
    avg_m8_joint = torch.mean(nume_m8_joint_sample)
    avg_m8_target = torch.mean(nume_m8_target_sample)
    avg_m8_deno = torch.mean(deno_m8_sample)

    avg_m7_mi = torch.mean(mi_m7_sample)
    avg_m7_nume = torch.mean(nume_m7_sample)
    avg_m7_joint = torch.mean(nume_m7_joint_sample)
    avg_m7_target = torch.mean(nume_m7_target_sample)
    avg_m7_deno = torch.mean(deno_m7_sample)

    avg_corrected_m8_mi = torch.mean(torch.tensor(mi_m8_corrected['mi']))
    avg_corrected_m7_mi = torch.mean(torch.tensor(mi_m7_corrected['mi']))
    avg_corrected_m8_nume = torch.mean(torch.tensor(mi_m8_corrected['nume']))
    avg_corrected_m7_nume = torch.mean(torch.tensor(mi_m7_corrected['nume']))
    avg_corrected_m8_nume_joint = torch.mean(torch.tensor(mi_m8_corrected['nume_joint']))
    avg_corrected_m7_nume_joint = torch.mean(torch.tensor(mi_m7_corrected['nume_joint']))
    avg_corrected_m8_nume_target = torch.mean(torch.tensor(mi_m8_corrected['nume_target']))
    avg_corrected_m7_nume_target = torch.mean(torch.tensor(mi_m7_corrected['nume_target']))
    avg_corrected_m8_deno = torch.mean(torch.tensor(mi_m8_corrected['deno']))
    avg_corrected_m7_deno = torch.mean(torch.tensor(mi_m7_corrected['deno']))

    emp_bias_m8_mi = avg_m8_mi - mi_m8_true
    emp_bias_m8_nume = avg_m8_nume - nume8_true
    emp_bias_m8_joint = avg_m8_joint - nume8_joint_true
    emp_bias_m8_target = avg_m8_target - nume8_target_true
    emp_bias_m8_deno = avg_m8_deno - deno8_true

    emp_bias_m7_mi = avg_m7_mi - mi_m7_true
    emp_bias_m7_nume = avg_m7_nume - nume7_true
    emp_bias_m7_joint = avg_m7_joint - nume7_joint_true
    emp_bias_m7_target = avg_m7_target - nume7_target_true
    emp_bias_m7_deno = avg_m7_deno - deno7_true

    mi_m8_dict = {
        'sample': mi_m8_sample,
        'avg': avg_m8_mi,
        'corrected_avg': avg_corrected_m8_mi,
        'std': torch.std(mi_m8_sample),
        'emp_bias': emp_bias_m8_mi,
        'ground_truth': mi_m8_true,
    }
    nume_m8_dict = {
        'sample': nume_m8_sample,
        'avg': avg_m8_nume,
        'corrected_avg': avg_corrected_m8_nume,
        'std': torch.std(nume_m8_sample),
        'emp_bias': emp_bias_m8_nume,
        'ground_truth': nume8_true,
    }
    nume_joint_m8_dict = {
        'sample': nume_m8_joint_sample,
        'avg': avg_m8_joint,
        'corrected_avg': avg_corrected_m8_nume_joint,
        'std': torch.std(nume_m8_joint_sample),
        'emp_bias': emp_bias_m8_joint,
        'ground_truth': nume8_joint_true,
    }
    nume_target_m8_dict = {
        'sample': nume_m8_target_sample,
        'avg': avg_m8_target,
        'corrected_avg': avg_corrected_m8_nume_target,
        'std': torch.std(nume_m8_target_sample),
        'emp_bias': emp_bias_m8_target,
        'ground_truth': nume8_target_true,
    }
    deno_m8_dict = {
        'sample': deno_m8_sample,
        'avg': avg_m8_deno,
        'corrected_avg': avg_corrected_m8_deno,
        'std': torch.std(deno_m8_sample),
        'emp_bias': emp_bias_m8_deno,
        'ground_truth': deno8_true,
    }

    mi_m7_dict = {
        'sample': mi_m7_sample,
        'avg': avg_m7_mi,
        'corrected_avg': avg_corrected_m7_mi,
        'std': torch.std(mi_m7_sample),
        'emp_bias': emp_bias_m7_mi,
        'ground_truth': mi_m7_true,
    }
    nume_m7_dict = {
        'sample': nume_m7_sample,
        'avg': avg_m7_nume,
        'corrected_avg': avg_corrected_m7_nume,
        'std': torch.std(nume_m7_sample),
        'emp_bias': emp_bias_m7_nume,
        'ground_truth': nume7_true,
    }
    nume_joint_m7_dict = {
        'sample': nume_m7_joint_sample,
        'avg': avg_m7_joint,
        'corrected_avg': avg_corrected_m7_nume_joint,
        'std': torch.std(nume_m7_joint_sample),
        'emp_bias': emp_bias_m7_joint,
        'ground_truth': nume7_joint_true,
    }
    nume_target_m7_dict = {
        'sample': nume_m7_target_sample,
        'avg': avg_m7_target,
        'corrected_avg': avg_corrected_m7_nume_target,
        'std': torch.std(nume_m7_target_sample),
        'emp_bias': emp_bias_m7_target,
        'ground_truth': nume7_target_true,
    }
    deno_m7_dict = {
        'sample': deno_m7_sample,
        'avg': avg_m7_deno,
        'corrected_avg': avg_corrected_m7_deno,
        'std': torch.std(deno_m7_sample),
        'emp_bias': emp_bias_m7_deno,
        'ground_truth': deno7_true,
    }

    return {
        'M8_mi': mi_m8_dict,
        'M8_nume': nume_m8_dict,
        'M8_joint': nume_joint_m8_dict,
        'M8_target': nume_target_m8_dict,
        'M8_deno': deno_m8_dict,
        'M7_mi': mi_m7_dict,
        'M7_nume': nume_m7_dict,
        'M7_joint': nume_joint_m7_dict,
        'M7_target': nume_target_m7_dict,
        'M7_deno': deno_m7_dict,
    }


def calculate_bias(config: dict, mi=False, nume=False, nume_joint=False, nume_target=False, deno=False, bias_correction=True) -> dict:
    """
    Same interface as the original file.

    Only change:
    - for M7 + mi=True, use the precision-based unbiased-K_TT + delta-method bias.
    - M8 and all other statistics keep the original formulas.
    """
    if not bias_correction:
        return {'bias': 0.0}

    n0 = config['n0']
    n1 = config['n1']
    n2 = config['n2']
    n_samples = config['n_samples']
    d = n0 + n1 + n2
    df = n_samples - 1

    bias_y = logdet_wishart_bias(df, n2)
    bias_02 = logdet_wishart_bias(df, n0 + n2)
    bias_12 = logdet_wishart_bias(df, n1 + n2)
    bias_2 = bias_y
    b_pred_m8 = logdet_wishart_bias(df, n0 + n1)
    b_joint_m8 = logdet_wishart_bias(df, d)

    if mi:
        if config['model'] == 'M8':
            bias = 0.5 * ((b_pred_m8 + bias_2) - b_joint_m8)
        else:
            bias = _m7_precision_mi_bias(config)['mi']
        st = 'mi'
    elif nume:
        bias_m8_nume = 0.5 * (b_pred_m8 + bias_2)
        bias_m7_nume = 0.5 * (bias_02 + bias_12)
        bias = bias_m8_nume if config['model'] == 'M8' else bias_m7_nume
        st = 'nume'
    elif nume_joint:
        bias = 0.0
        st = 'nume_joint'
    elif nume_target:
        bias = 0.5 * bias_2
        st = 'nume_target'
    elif deno:
        bias_m8_deno = 0.5 * b_joint_m8
        bias_m7_deno = 0.5 * (bias_02 + bias_12 - bias_2)
        bias = bias_m8_deno if config['model'] == 'M8' else bias_m7_deno
        st = 'deno'
    else:
        raise ValueError("At least one statistic flag must be True.")

    return {st: bias}


def sort_m7_m8_results(results_list):
    """Helper: sort results list by N and p values, separated by M7 and M8."""
    mi_m7_results_list = []
    nome_m7_results_list = []
    nome_joint_m7_results_list = []
    nome_target_m7_results_list = []
    deno_m7_results_list = []

    mi_m8_results_list = []
    nome_m8_results_list = []
    nome_joint_m8_results_list = []
    nome_target_m8_results_list = []
    deno_m8_results_list = []
    for res in results_list:
        N = res['N']
        p = res['p']
        mi_m7_results_list.append({'N': N, 'p': p, 'mean': res['M7_mi_mean'], 'std': res['M7_mi_std'], 'ground_truth': res['M7_mi_ground_truth']})
        nome_m7_results_list.append({'N': N, 'p': p, 'mean': res['M7_nume_mean'], 'std': res['M7_nume_std'], 'ground_truth': res['M7_nume_ground_truth']})
        nome_joint_m7_results_list.append({'N': N, 'p': p, 'mean': res['M7_joint_mean'], 'std': res['M7_joint_std'], 'ground_truth': res['M7_joint_ground_truth']})
        nome_target_m7_results_list.append({'N': N, 'p': p, 'mean': res['M7_target_mean'], 'std': res['M7_target_std'], 'ground_truth': res['M7_target_ground_truth']})
        deno_m7_results_list.append({'N': N, 'p': p, 'mean': res['M7_deno_mean'], 'std': res['M7_deno_std'], 'ground_truth': res['M7_deno_ground_truth']})

        mi_m8_results_list.append({'N': N, 'p': p, 'mean': res['M8_mi_mean'], 'std': res['M8_mi_std'], 'ground_truth': res['M8_mi_ground_truth']})
        nome_m8_results_list.append({'N': N, 'p': p, 'mean': res['M8_nume_mean'], 'std': res['M8_nume_std'], 'ground_truth': res['M8_nume_ground_truth']})
        nome_joint_m8_results_list.append({'N': N, 'p': p, 'mean': res['M8_joint_mean'], 'std': res['M8_joint_std'], 'ground_truth': res['M8_joint_ground_truth']})
        nome_target_m8_results_list.append({'N': N, 'p': p, 'mean': res['M8_target_mean'], 'std': res['M8_target_std'], 'ground_truth': res['M8_target_ground_truth']})
        deno_m8_results_list.append({'N': N, 'p': p, 'mean': res['M8_deno_mean'], 'std': res['M8_deno_std'], 'ground_truth': res['M8_deno_ground_truth']})

    return [mi_m7_results_list, nome_m7_results_list, nome_joint_m7_results_list, nome_target_m7_results_list, deno_m7_results_list], [mi_m8_results_list, nome_m8_results_list, nome_joint_m8_results_list, nome_target_m8_results_list, deno_m8_results_list]


def simulation_wrapper(config: dict) -> dict:
    """
    Run the MI simulation for M7 and M8.

    Only M7 joint MI bias correction path is changed to the unbiased-K_TT + delta-method route.
    """
    seed = config['seed']
    sim_func = simulate_m7_m8_mi

    m8_bias_func = partial(calculate_bias, mi=True)
    m7_bias_func = partial(calculate_bias, mi=True)
    m8_nume_fuc = partial(bias_resampling)
    m8_nume_joint_func = partial(bias_resampling)
    m8_nume_target_func = partial(calculate_bias, nume_target=True)
    m8_deno_func = partial(calculate_bias, deno=True)
    m7_nume_func = partial(bias_resampling)
    m7_nume_joint_func = partial(bias_resampling)
    m7_nume_target_func = partial(calculate_bias, nume_target=True)
    m7_deno_func = partial(calculate_bias, deno=True)

    bias_corr_func = {
        'M8': {
            'mi': m8_bias_func,
            'nume': m8_nume_fuc,
            'nume_joint': m8_nume_joint_func,
            'nume_target': m8_nume_target_func,
            'deno': m8_deno_func,
        },
        'M7': {
            'mi': m7_bias_func,
            'nume': m7_nume_func,
            'nume_joint': m7_nume_joint_func,
            'nume_target': m7_nume_target_func,
            'deno': m7_deno_func,
        },
    }

    corr_value_func = corrected_statistic
    functions_dict = {
        's_simulation': sim_func,
        'bias_correction': bias_corr_func,
        'corrected_statistic': corr_value_func,
    }
    results_dict = simulation(config, functions_dict, seed=seed)
    return results_dict

if __name__ == "__main__":
    print("Running M7/M8 mutual-information simulation with M7 precision-delta correction...")

    print(f"\nRunning simulation...")
    exp_name = f"_precision_delta"
    yaml_file = f"/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/configs/small_test.yaml"
    folder_path = f"/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/figures/MI_precision_matrix"
    save_path = pathlib.Path(f"{folder_path}/{exp_name}")
    save_path.mkdir(parents=True, exist_ok=True)
    with open(yaml_file, 'r') as f:
        super_config = yaml.safe_load(f)

        config = super_config['Mutual_Information_Simulation']

        n_p_config = super_config['N_P_variations']
        p_values = n_p_config['p_values']
        N_values = n_p_config['N_values']
        config['p_values'] = p_values
        config['N_values'] = N_values
        config['simulation_func'] = simulation_wrapper
    results = N_P_variation_simulation(config)
    m7_results_list, m8_results_list = sort_m7_m8_results(results)

    mi_m7_result, nome_m7_list, nume_joint_m7_list, nume_target_m7_list, deno_m7_list = m7_results_list[0], m7_results_list[1], m7_results_list[2], m7_results_list[3], m7_results_list[4]
    mi_m8_result, nome_m8_list, nume_joint_m8_list, nume_target_m8_list, deno_m8_list = m8_results_list[0], m8_results_list[1], m8_results_list[2], m8_results_list[3], m8_results_list[4]

    plot_heatmap_mean_std(mi_m7_result, title=f"Mutual Information M7 -{exp_name} - Mutual Information M7", save_path=save_path)
    plot_heatmap_mean_std(mi_m8_result, title=f"Mutual Information M8 -{exp_name} - Mutual Information M8", save_path=save_path)
    plot_heatmap_mean_std(nome_m7_list, title=f"numerator M7 -{exp_name} - numerator M7", save_path=save_path)
    plot_heatmap_mean_std(nome_m8_list, title=f"numerator M8 -{exp_name} - numerator M8", save_path=save_path)
    plot_heatmap_mean_std(nume_joint_m7_list, title=f"numerator joint M7 -{exp_name} - numerator joint M7", save_path=save_path)
    plot_heatmap_mean_std(nume_joint_m8_list, title=f"numerator joint M8 -{exp_name} - numerator joint M8", save_path=save_path)
    plot_heatmap_mean_std(nume_target_m7_list, title=f"numerator target M7 -{exp_name} - numerator target M7", save_path=save_path)
    plot_heatmap_mean_std(nume_target_m8_list, title=f"numerator target M8 -{exp_name} - numerator target M8", save_path=save_path)
    plot_heatmap_mean_std(deno_m7_list, title=f"denominator M7 -{exp_name} - denominator M7", save_path=save_path)
    plot_heatmap_mean_std(deno_m8_list, title=f"denominator M8 -{exp_name} - denominator M8", save_path=save_path)

    with open(f'{save_path}/{exp_name}_config.yaml', 'w') as f:
        yaml_config = {
            'simulation': 'MI comparison for M7/M8 with M7 precision-delta correction',
            'seed': config['seed'],
            'N_samples_values': N_values,
            'p_values': p_values,
            'p_scale': config['p_scale'],
            'q_scale': config['q_scale'],
            'r_scale': config['r_scale'],
        }
        yaml.safe_dump(yaml_config, f, sort_keys=False, allow_unicode=True)
    print("\nFinished simulation.")
