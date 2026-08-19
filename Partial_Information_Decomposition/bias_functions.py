import torch
from pathlib import Path
import sys


from functools import partial

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from Partial_Information_Decomposition.PID_util import create_cov_matrix
from Partial_Information_Decomposition.Idep.covariance_utils import create_m7_cov
from Partial_Information_Decomposition.mi_functions import calcualte_mi
from external.gpid.src.gpid import tilde_pid








def logdet_wishart_bias(df: int, d: int) -> float:
    """
    Exact finite-sample bias for log|S| when S is the unbiased sample covariance
    from Gaussian data and (df) * S ~ Wishart_d(Sigma, df).

    Returns
    -------
    bias : float
        E[log|S|] - log|Sigma|
    """
    if df <= d - 1:
        raise ValueError(f"Need df > d-1. Got df={df}, d={d}.")

    i = torch.arange(1, d + 1, dtype=torch.float64)
    term = torch.special.digamma((df - i + 1) / 2.0)

    bias = torch.sum(term) + d * torch.log(torch.tensor(2.0 / df, dtype=torch.float64))

    return bias.item()

def mi_wishart_bias(dims: list, n_samples: int):
    """
    Bias correction for Gaussian mutual information estimates
    from unbiased sample covariance.

    Assumes order: [X1, X2, T]
    and torch.cov(..., correction=1), so df = n_samples - 1.

    Returns biases in nats.
    """
    df = n_samples - 1

    if len(dims) == 2:
        d1, d2 = dims

        bias_x1 = logdet_wishart_bias(df, d1)
        bias_x2 = logdet_wishart_bias(df, d2)
        bias_x1x2 = logdet_wishart_bias(df, d1 + d2)

        bias_mi = 0.5 * (bias_x1 + bias_x2 - bias_x1x2)
        return bias_mi

    if len(dims) == 3:
        d1, d2, dt = dims

        bias_x1 = logdet_wishart_bias(df, d1)
        bias_x2 = logdet_wishart_bias(df, d2)
        bias_t = logdet_wishart_bias(df, dt)

        bias_x1x2 = logdet_wishart_bias(df, d1 + d2)
        bias_x1t = logdet_wishart_bias(df, d1 + dt)
        bias_x2t = logdet_wishart_bias(df, d2 + dt)
        bias_x1x2t = logdet_wishart_bias(df, d1 + d2 + dt)

        # Bias of I(X1; T)
        bias_mi_1_t = 0.5 * (bias_x1 + bias_t - bias_x1t)

        # Bias of I(X2; T)
        bias_mi_2_t = 0.5 * (bias_x2 + bias_t - bias_x2t)

        # Bias of I((X1, X2); T)
        bias_tri_mi = 0.5 * (bias_x1x2 + bias_t - bias_x1x2t)

        # Optional: bias of I(X1; X2), if you need it
        bias_mi_12 = 0.5 * (bias_x1 + bias_x2 - bias_x1x2)

        return {
            "bias_mi_1_t": bias_mi_1_t,
            "bias_mi_2_t": bias_mi_2_t,
            "bias_tri_mi": bias_tri_mi,
            "bias_mi_12": bias_mi_12,
        }

    raise ValueError(f"dims must have length 2 or 3. Got len(dims)={len(dims)}.")

    



def permuteation_debiased(config,term = 'nume'):
    dx1 = config['dx1']
    dx2 = config['dx2']
    dt = config['dt']
    device = config['device']
    X1,X2,T = config['X1_perm'],config['X2_perm'],config['T_perm']
    
    Z = create_cov_matrix(rvs=[X1,X2,T],device=device)

    if config['model'] == 'M7':
        M7_cov = create_m7_cov(config,Z['full_cov'],whitening_normalize=True) #Also Whiten Normalize
        M7_cov_dict = create_cov_matrix(Sigma=M7_cov,dims = [dx1, dx2, dt],device=device)
        m7_dict = {
            'P': M7_cov_dict['cross_x1_x2'],
            'Q': M7_cov_dict['cross_x1_t'],
            'R': M7_cov_dict['cross_x2_t'],
            'Sigma': M7_cov_dict['full_cov'],
        }
        mi_terms = calcualte_mi(config,m7_dict)
        value = mi_terms[term]
    
    return value

def broja_venkatesh_bias(config):
    """Calculate one raw Gaussian BROJA statistic for permutation debiasing.

    Inputs:
        config: dict containing:
            X1_perm, X2_perm, T_perm: torch.Tensor or array-like sample matrices
                with shapes (N, dx1), (N, dx2), and (N, dt).
            dx1, dx2, dt: int, source and target dimensions.
            n_samples: int, number of rows in each sample matrix.
            bias_correction: bool, passed to the legacy estimator. Lorenz
                correction callers must set this to ``False``.
            pid_bias_term: optional str, either ``'obj'`` (default) or ``'syn'``.

    Outputs:
        float, either raw Venkatesh ``obj`` or the plugin gPID synergy in bits.
    """

    dm , dx, dy = config['dt'] , config['dx1'] , config['dx2']
    X_1, X_2, T = config['X1_perm'], config['X2_perm'], config['T_perm']
    N = config['n_samples']
    #bias_corr = config['bias_corr']
    data = [T, X_1, X_2]  # [T, X1, X2]
    dict_cov = create_cov_matrix(data)
    cov = dict_cov["full_cov"].detach().cpu().numpy()  # torch (D, D) -> NumPy (D, D)
    output = tilde_pid.exact_gauss_tilde_pid(
        cov,
        dm,
        dx,
        dy,
        unbiased=config['bias_correction'],
        sample_size=N,
    )
    statistic = config.get('pid_bias_term', 'obj')
    if statistic == 'obj':
        return float(output[4])
    if statistic == 'syn':
        return float(output[8])
    raise ValueError(
        "config['pid_bias_term'] must be 'obj' or 'syn'; "
        f"got {statistic!r}."
    )



def permutation_null_debias(config,func):
    """Debias an MI-like estimator by subtracting its permutation null floor.

    X and Y can each be a single array or a tuple of paired tensors.
    Samples are assumed to be along axis 0.

    The function computes:

        raw = func(X, Y)
        perm_mean = mean(func(X, permuted_Y))
        debiased = raw - perm_mean

    If Y is a tuple, all Y blocks are permuted with the same permutation.
    X is kept fixed, so any internal structure within X is preserved.

    If n_perm=0, the function returns the raw estimate with perm_mean=0."""

    X1,X2,T = config['X1'],config['X2'],config['T']
    n = config['n_samples']
    dx1 = config['dx1']
    dx2 = config['dx2']
    dt = config['dt']
    device = config['device']
    n_perm = config['n_perm']

    rng = config.get('rng', None)
    if rng is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(config['rng_seed'])


    if n_perm == 0:
        return {
            "debiased": 0.0,
            "perm_mean": 0.0,
            "perm_std": 0.0,
            "perm_se": 0.0,
            "perm_values": torch.empty(0, dtype=float),
            "n_perm": n_perm,
        }
    
    config_sigma_perm = config.copy()
    perm_values = torch.empty(n_perm, dtype=float, device=device)
    
    for i in range(n_perm):
        idx = torch.randperm(n,generator=rng)

        if isinstance(T, tuple):
            T_perm = tuple(t[idx] for t in T)
        else:
            T_perm = T[idx]
        
        config_sigma_perm['X1_perm'] = X1 #X1 is not permuted
        config_sigma_perm['X2_perm'] = X2 #X2 is not permuted
        config_sigma_perm['T_perm'] = T_perm #T is permuted

        perm_values[i] = float(func(config=config_sigma_perm))

    perm_mean = float(torch.mean(perm_values))
    perm_std = float(torch.std(perm_values, unbiased=True)) if n_perm > 1 else 0.0
    perm_se = perm_std / torch.sqrt(torch.tensor(n_perm, dtype=torch.float64)) if n_perm > 0 else 0.0

    return {
        "bias": perm_mean,
        "perm_mean": perm_mean,
        "perm_std": perm_std,
        "perm_se": perm_se,
        "perm_values": perm_values,
        "n_perm": n_perm,
    }



def unique_bias(config,functions_dict:dict = None):

    nodes = {'M7':['i','h'],'M8':['k','j']} #The unique nodes for each model, used to extract the relevant bias correction for each statistic
    assert type(functions_dict) == dict, "Expected bias_corr_func to be a dict with keys 'M7' and 'M8'."
    
    bias_dict ={}
    for model,bc_func in zip(nodes.keys(), functions_dict.values()):
        config['model'] = model
        bias = bc_func(config=config,model=model)

        node_0 = nodes[model][0] # i or k depending on the model
        node_1 = nodes[model][1] # h or j depending on the model

        bias_dict[node_0] = bias[node_0]
        bias_dict[node_1] = bias[node_1]

        
    return bias_dict


def bias_func(config,model):
    dx1 = config['dx1']
    dx2 = config['dx2']
    dt = config['dt']
    n_samples = config['n_samples']
    d = dx1 + dx2 + dt
    df = n_samples - 1

    bias_x0 = logdet_wishart_bias(df, dx1)
    bias_x1 = logdet_wishart_bias(df, dx2)
    bias_y  = logdet_wishart_bias(df, dt)
    # M7 (Structural) Biases
    bias_02 = logdet_wishart_bias(df, dx1 + dt) # Clique 0
    bias_12 = logdet_wishart_bias(df, dx2 + dt) # Clique 1
    bias_2 = bias_y # seperator 2
    bi_variate_mix2t_bias = 0.5*logdet_wishart_bias(df, dx2) + 0.5*logdet_wishart_bias(df, dt) - 0.5*logdet_wishart_bias(df, dx2+dt)
    bi_variate_mix1t_bias = 0.5*logdet_wishart_bias(df, dx1) + 0.5*logdet_wishart_bias(df, dt) - 0.5*logdet_wishart_bias(df, dx1+dt)

    if model == 'M7':
        nume_bias = permutation_null_debias(config,partial(permuteation_debiased,term='nume'))['bias']
        bias_m7_structural = bias_02 + bias_12 - bias_2
        deno_bias = 0.5 * (bias_m7_structural - (bias_x0 + bias_x1 + bias_y))
        return {'i': (nume_bias - deno_bias) - bi_variate_mix2t_bias,'h': (nume_bias - deno_bias) - bi_variate_mix1t_bias}
    
    elif model == 'M8':
        deno_bias = 0.5*(logdet_wishart_bias(df, d)-(bias_x0 + bias_x1 + bias_y))
        nume_bias = 0.5*(logdet_wishart_bias(df, dx1 + dx2) - (bias_x0 + bias_x1))
        return {'k': (nume_bias - deno_bias) - bi_variate_mix2t_bias,'j': (nume_bias - deno_bias) - bi_variate_mix1t_bias}


def parametric_bootstrap_obj_debias(
    config: dict,
    covariance=None,
    raw_obj: float | None = None,
    statistic: str = 'obj',
) -> dict:
    """Estimate Gaussian BROJA objective or synergy bias by parametric bootstrap.

    The function fits a zero-mean Gaussian model using the supplied covariance,
    or using the covariance of ``config['T']``, ``config['X1']``, and
    ``config['X2']``. It then draws parametric-bootstrap datasets of the same
    sample size and fully re-optimizes Gaussian BROJA for every dataset.
    Venkatesh's multiplicative debias factor and the mutual-information bias
    option are both disabled in every calculation.

    Inputs:
        config: dict containing:
            dx1: int, dimension of source X1.
            dx2: int, dimension of source X2.
            dt: int, dimension of target T.
            n_samples: int, sample size represented by the fitted covariance.
            covariance_is_sample: bool, which must be exactly ``True`` when
                ``covariance`` is supplied. This prevents accidental correction
                of a theoretical/population or shrinkage covariance.
            X1, X2, T: optional torch.Tensor or array-like values with shapes
                (n_samples, dx1), (n_samples, dx2), and (n_samples, dt).
                They are required only when ``covariance`` is not supplied.
            n_obj_bootstrap: optional int, number of Gaussian bootstrap
                replicates. It falls back to ``n_bootstrap``, then 20.
            obj_bootstrap_seed: optional int, local bootstrap seed. It falls
                back to ``rng_seed``, then ``seed``, then 56.
            device: optional str or torch.device used for sampling.
        covariance: optional torch.Tensor or array-like unbiased sample
            covariance with shape (dt + dx1 + dx2, dt + dx1 + dx2), ordered as
            [T, X1, X2]. When omitted, it is estimated from the sample arrays
            in ``config``.
        raw_obj: optional float, the already calculated value of ``statistic``
            for exactly the supplied/estimated covariance. A mismatch with the
            value recalculated here raises an error. The name is retained for
            backward compatibility with objective-only callers.
        statistic: str, either ``'obj'`` (default) or ``'syn'``. Synergy uses
            the plugin gPID synergy returned by the estimator, as required by
            the Lorenz resampling and shuffle correction.

    Outputs:
        dict containing:
            method: str, correction method name.
            statistic: str, selected raw statistic.
            raw_obj: float, uncorrected selected statistic in bits; the key
                name is retained for backward compatibility.
            bias: float, bootstrap_mean - raw_obj, to subtract from raw_obj.
            debiased: float, raw_obj - bias, without clipping.
            corrected_obj: float, alias of ``debiased``.
            bootstrap_mean: float, mean bootstrap objective in bits.
            bootstrap_std: float, bootstrap objective standard deviation.
            bootstrap_se: float, Monte Carlo standard error of the mean.
            relative_mc_error: float, bootstrap_se / abs(bias).
            mc_stable: bool, whether at least 20 replicates were used and the
                relative Monte Carlo error is at most 10 percent.
            bootstrap_values: torch.Tensor with shape (n_obj_bootstrap,).
            n_bootstrap: int, number of bootstrap replicates.
            n_samples: int, sample size per bootstrap replicate.
            seed: int, local bootstrap seed.
            covariance_min_eigenvalue: float, fitted covariance diagnostic.
            covariance_condition_number: float, fitted covariance diagnostic.
            covariance_is_sample: bool, always ``True`` after validation.

    Notes:
        The returned correction uses the basic parametric-bootstrap formula
        ``bias = mean(value*) - value`` and ``debiased = value - bias``. The
        function deliberately does not clip the corrected value or impose PID
        bounds.
    """

    if statistic not in ('obj', 'syn'):
        raise ValueError(
            "statistic must be 'obj' or 'syn'; "
            f"got {statistic!r}."
        )

    dx = int(config['dx1'])
    dy = int(config['dx2'])
    dm = int(config['dt'])
    total_dimension = dm + dx + dy

    if covariance is None:
        missing_keys = [key for key in ('T', 'X1', 'X2') if key not in config]
        if missing_keys:
            raise KeyError(
                "covariance was not supplied, so config must contain "
                f"'T', 'X1', and 'X2'. Missing: {missing_keys}."
            )

        sample_device = config.get('device', 'cpu')
        target = torch.as_tensor(
            config['T'],
            dtype=torch.float64,
            device=sample_device,
        )  # array-like (N, dt) -> torch (N, dt)
        source_x1 = torch.as_tensor(
            config['X1'],
            dtype=torch.float64,
            device=sample_device,
        )  # array-like (N, dx1) -> torch (N, dx1)
        source_x2 = torch.as_tensor(
            config['X2'],
            dtype=torch.float64,
            device=sample_device,
        )  # array-like (N, dx2) -> torch (N, dx2)

        sample_arrays = {'T': target, 'X1': source_x1, 'X2': source_x2}
        expected_dimensions = {'T': dm, 'X1': dx, 'X2': dy}
        for name, values in sample_arrays.items():
            if values.ndim != 2:
                raise ValueError(
                    f"config['{name}'] must have shape (N, d); got {tuple(values.shape)}."
                )
            if values.shape[1] != expected_dimensions[name]:
                raise ValueError(
                    f"config['{name}'] has {values.shape[1]} columns; "
                    f"expected {expected_dimensions[name]}."
                )

        observed_sample_size = target.shape[0]
        if source_x1.shape[0] != observed_sample_size or source_x2.shape[0] != observed_sample_size:
            raise ValueError("T, X1, and X2 must contain the same number of samples.")

        configured_sample_size = int(config.get('n_samples', observed_sample_size))
        if configured_sample_size != observed_sample_size:
            raise ValueError(
                f"config['n_samples']={configured_sample_size} does not match "
                f"the sample arrays' size {observed_sample_size}."
            )

        fitted_covariance = create_cov_matrix(
            rvs=[target, source_x1, source_x2],
            device=sample_device,
        )['full_cov']  # [three (N, d_i) tensors] -> torch (D, D)
        n_samples = observed_sample_size
    else:
        if 'n_samples' not in config:
            raise KeyError("config['n_samples'] is required when covariance is supplied.")
        if config.get('covariance_is_sample') is not True:
            raise ValueError(
                "A supplied covariance must be an ordinary unbiased sample "
                "covariance and requires config['covariance_is_sample'] = True. "
                "Use a theoretical covariance only to generate repeated samples "
                "and define ground truth."
            )

        covariance_device = (
            covariance.device
            if torch.is_tensor(covariance)
            else config.get('device', 'cpu')
        )
        fitted_covariance = torch.as_tensor(
            covariance,
            dtype=torch.float64,
            device=covariance_device,
        )  # array-like (D, D) -> torch (D, D)
        n_samples = int(config['n_samples'])

    if fitted_covariance.ndim != 2 or fitted_covariance.shape != (
        total_dimension,
        total_dimension,
    ):
        raise ValueError(
            "covariance must be a square [T, X1, X2] matrix with shape "
            f"({total_dimension}, {total_dimension}); got {tuple(fitted_covariance.shape)}."
        )
    if n_samples <= total_dimension:
        raise ValueError(
            "Parametric-bootstrap sample covariances are singular unless "
            f"n_samples > total dimension. Got n_samples={n_samples}, "
            f"dimension={total_dimension}."
        )
    if not bool(torch.isfinite(fitted_covariance).all()):
        raise ValueError("covariance contains NaN or infinite values.")

    covariance_scale = max(float(torch.max(torch.abs(fitted_covariance))), 1.0)
    maximum_asymmetry = float(
        torch.max(torch.abs(fitted_covariance - fitted_covariance.T))
    )
    if maximum_asymmetry > 1e-8 * covariance_scale:
        raise ValueError(
            "covariance must be symmetric; maximum absolute asymmetry is "
            f"{maximum_asymmetry:.3e}."
        )

    fitted_covariance = (
        fitted_covariance + fitted_covariance.T
    ) / 2.0  # torch (D, D) -> torch (D, D)
    covariance_eigenvalues = torch.linalg.eigvalsh(
        fitted_covariance
    )  # torch (D, D) -> torch (D,)
    minimum_eigenvalue = float(covariance_eigenvalues[0])
    maximum_eigenvalue = float(covariance_eigenvalues[-1])
    if minimum_eigenvalue <= 0.0:
        raise ValueError(
            "covariance must be positive definite for an unregularized "
            f"parametric bootstrap; minimum eigenvalue is {minimum_eigenvalue:.3e}."
        )
    covariance_condition_number = maximum_eigenvalue / minimum_eigenvalue
    covariance_cholesky = torch.linalg.cholesky(
        fitted_covariance
    )  # torch (D, D) -> torch (D, D)

    raw_covariance_numpy = (
        fitted_covariance.detach().cpu().numpy()
    )  # torch (D, D) -> NumPy (D, D)
    raw_output = tilde_pid.exact_gauss_tilde_pid(
        raw_covariance_numpy,
        dm,
        dx,
        dy,
        unbiased=False,
        sample_size=n_samples,
    )
    fitted_obj = (
        float(raw_output[4])
        if statistic == 'obj'
        else float(raw_output[8])
    )
    if not bool(torch.isfinite(torch.tensor(fitted_obj))):
        raise RuntimeError(
            f"The BROJA optimizer returned a non-finite raw {statistic}."
        )

    if raw_obj is None:
        raw_obj_value = fitted_obj
    else:
        raw_obj_value = float(raw_obj)
        objective_tolerance = 1e-8 * max(1.0, abs(fitted_obj))
        if abs(raw_obj_value - fitted_obj) > objective_tolerance:
            raise ValueError(
                "raw_obj does not match the selected statistic recalculated "
                "from covariance: "
                f"raw_obj={raw_obj_value}, recalculated={fitted_obj}."
            )

    n_bootstrap = int(
        config.get(
            'n_obj_bootstrap',
            config.get('n_bootstrap', 20),
        )
    )
    if n_bootstrap < 2:
        raise ValueError(
            f"n_obj_bootstrap must be at least 2; got {n_bootstrap}."
        )

    seed = int(
        config.get(
            'obj_bootstrap_seed',
            config.get('rng_seed', config.get('seed', 56)),
        )
    )
    generator = torch.Generator(device=fitted_covariance.device)
    generator.manual_seed(seed)

    bootstrap_values = torch.empty(
        n_bootstrap,
        dtype=torch.float64,
        device=fitted_covariance.device,
    )  # empty scalar storage -> torch (B,)

    for bootstrap_index in range(n_bootstrap):
        standard_samples = torch.randn(
            (n_samples, total_dimension),
            dtype=torch.float64,
            device=fitted_covariance.device,
            generator=generator,
        )  # scalar Gaussian draws -> torch (N, D)
        bootstrap_samples = (
            standard_samples @ covariance_cholesky.T
        )  # torch (N, D) @ torch (D, D) -> torch (N, D)
        centered_samples = (
            bootstrap_samples - bootstrap_samples.mean(dim=0, keepdim=True)
        )  # torch (N, D) - torch (1, D) -> torch (N, D)
        bootstrap_covariance = (
            centered_samples.T @ centered_samples
        ) / (n_samples - 1)  # torch (D, N) @ torch (N, D) -> torch (D, D)
        bootstrap_covariance_numpy = (
            bootstrap_covariance.detach().cpu().numpy()
        )  # torch (D, D) -> NumPy (D, D)

        bootstrap_output = tilde_pid.exact_gauss_tilde_pid(
            bootstrap_covariance_numpy,
            dm,
            dx,
            dy,
            unbiased=False,
            sample_size=n_samples,
        )
        bootstrap_obj = (
            float(bootstrap_output[4])
            if statistic == 'obj'
            else float(bootstrap_output[8])
        )
        if not bool(torch.isfinite(torch.tensor(bootstrap_obj))):
            raise RuntimeError(
                f"The BROJA optimizer returned a non-finite {statistic} for "
                f"bootstrap replicate {bootstrap_index}."
            )
        bootstrap_values[bootstrap_index] = bootstrap_obj

    bootstrap_mean = float(torch.mean(bootstrap_values))
    bootstrap_std = float(torch.std(bootstrap_values, unbiased=True))
    bootstrap_se = bootstrap_std / (n_bootstrap ** 0.5)
    estimated_bias = bootstrap_mean - raw_obj_value
    corrected_obj = raw_obj_value - estimated_bias
    relative_mc_error = (
        bootstrap_se / abs(estimated_bias)
        if estimated_bias != 0.0
        else float('inf')
    )
    mc_stable = n_bootstrap >= 20 and relative_mc_error <= 0.10
    bootstrap_values_cpu = (
        bootstrap_values.detach().cpu()
    )  # torch (B,) on sampling device -> torch (B,) on CPU

    return {
        'method': f'gaussian_parametric_bootstrap_{statistic}',
        'statistic': statistic,
        'raw_obj': raw_obj_value,
        'bias': estimated_bias,
        'debiased': corrected_obj,
        'corrected_obj': corrected_obj,
        'bootstrap_mean': bootstrap_mean,
        'bootstrap_std': bootstrap_std,
        'bootstrap_se': bootstrap_se,
        'relative_mc_error': relative_mc_error,
        'mc_stable': mc_stable,
        'bootstrap_values': bootstrap_values_cpu,
        'n_bootstrap': n_bootstrap,
        'n_samples': n_samples,
        'seed': seed,
        'covariance_min_eigenvalue': minimum_eigenvalue,
        'covariance_condition_number': covariance_condition_number,
        'covariance_is_sample': True,
    }


def lorenz_gaussian_obj_debias(
    config: dict,
    covariance=None,
    raw_obj: float | None = None,
) -> dict:
    """Apply the Lorenz et al. Gaussian merged correction to raw BROJA ``obj``.

    The implementation follows the Gaussian procedure in Lorenz et al. (2026):
    exact Goodman/Wishart correction for Gaussian mutual informations,
    parametric-resampling and target-shuffle estimates for synergy bias, and an
    equally weighted merge by default. Lorenz corrects the plugin gPID union
    through ``union = I(T; X1, X2) - synergy``. The raw optimizer objective is
    retained separately, and the returned additive correction is:

    ``obj_bias = raw_obj - corrected_plugin_union``.

    Inputs:
        config: dict containing:
            X1, X2, T: torch.Tensor or array-like sample matrices with shapes
                (N, dx1), (N, dx2), and (N, dt).
            dx1, dx2, dt: int, source and target dimensions.
            n_samples: int, number of observations.
            covariance_ddof: optional int, must be 1.
            covariance_estimator: optional str, must be ``'sample'``.
            n_lorenz_resamples: optional int, Gaussian resamples; defaults to
                ``n_obj_bootstrap`` and then 20.
            n_lorenz_shuffles: optional int, target shuffles; defaults to
                ``n_perm`` and then 20.
            lorenz_merge_weight: optional float in [0, 1], weight assigned to
                the resampling synergy bias; defaults to 0.5.
            lorenz_seed: optional int, root seed.
            obj_bootstrap_seed: optional int, per-dataset root-seed fallback.
            lorenz_resampling_seed: optional int, resampling seed.
            lorenz_shuffle_seed: optional int, independent shuffle seed.
        covariance: optional torch.Tensor or array-like ordinary sample
            covariance with shape (dt + dx1 + dx2, dt + dx1 + dx2), ordered
            [T, X1, X2].
        raw_obj: optional float, raw Venkatesh ``output[4]`` for the supplied
            samples.

    Outputs:
        dict containing:
            method: str, ``'lorenz_gaussian_merged'``.
            raw_obj: float, uncorrected Venkatesh objective in bits.
            bias: float, additive objective bias to subtract.
            debiased: float, corrected objective without clipping.
            corrected_obj: float, alias of ``debiased``.
            configuration: dict with dimensions, iteration counts, weight,
                covariance convention, units, and seeds.
            plugin: dict with raw MIs, raw PID atoms, synergy, and objective.
            bias_components: dict with three Goodman MI biases, resampling,
                shuffle, merged synergy, and derived objective biases.
            corrected: dict with corrected MIs, PID atoms, synergy, and union.
            surrogates: dict with resampling and shuffle synergy tensors.

    Notes:
        This function assumes an ordinary unbiased sample covariance (ddof=1).
        Shrinkage covariances are not supported. Venkatesh's proportional
        correction is never used. The underlying gPID plugin applies its
        existing PID bounds before synergy correction; the Lorenz-corrected
        values are not clipped.
    """

    if int(config.get('covariance_ddof', 1)) != 1:
        raise NotImplementedError(
            "Lorenz/Goodman correction currently requires covariance_ddof=1."
        )
    if config.get('covariance_estimator', 'sample') != 'sample':
        raise NotImplementedError(
            "Lorenz/Goodman correction currently supports only the ordinary "
            "sample covariance; shrinkage estimators are not supported."
        )

    dx = int(config['dx1'])
    dy = int(config['dx2'])
    dm = int(config['dt'])
    expected_dimensions = {'T': dm, 'X1': dx, 'X2': dy}
    missing_keys = [key for key in expected_dimensions if key not in config]
    if missing_keys:
        raise KeyError(
            "Lorenz Gaussian correction requires sample-level T, X1, and X2. "
            f"Missing: {missing_keys}."
        )

    samples = {
        name: torch.as_tensor(
            config[name],
            dtype=torch.float64,
            device='cpu',
        )  # array-like (N, d_i) -> CPU torch (N, d_i)
        for name in expected_dimensions
    }
    for name, values in samples.items():
        if values.ndim != 2:
            raise ValueError(
                f"config['{name}'] must have shape (N, d); "
                f"got {tuple(values.shape)}."
            )
        if values.shape[1] != expected_dimensions[name]:
            raise ValueError(
                f"config['{name}'] has {values.shape[1]} columns; "
                f"expected {expected_dimensions[name]}."
            )
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"config['{name}'] contains NaN or infinite values.")

    target = samples['T']
    source_x1 = samples['X1']
    source_x2 = samples['X2']
    n_samples = int(target.shape[0])
    if source_x1.shape[0] != n_samples or source_x2.shape[0] != n_samples:
        raise ValueError("T, X1, and X2 must contain the same number of rows.")
    if n_samples < 2:
        raise ValueError(f"At least two samples are required; got {n_samples}.")
    configured_n_samples = int(config.get('n_samples', n_samples))
    if configured_n_samples != n_samples:
        raise ValueError(
            f"config['n_samples']={configured_n_samples} does not match the "
            f"sample arrays' size {n_samples}."
        )

    total_dimension = dm + dx + dy
    if n_samples - 1 < total_dimension:
        raise ValueError(
            "The exact Goodman correction and positive-definite sample "
            "covariance require n_samples - 1 >= total dimension. "
            f"Got {n_samples - 1} < {total_dimension}."
        )

    if covariance is None:
        sample_covariance = create_cov_matrix(
            rvs=[target, source_x1, source_x2],
            device='cpu',
        )['full_cov'].to(dtype=torch.float64)  # [three (N, d_i)] -> torch (D, D)
    else:
        sample_covariance = torch.as_tensor(
            covariance,
            dtype=torch.float64,
            device='cpu',
        )  # array-like (D, D) -> CPU torch (D, D)

    covariance_numpy = (
        sample_covariance.detach().cpu().numpy()
    )  # CPU torch (D, D) -> NumPy (D, D)
    raw_output = tilde_pid.exact_gauss_tilde_pid(
        covariance_numpy,
        dm,
        dx,
        dy,
        unbiased=False,
        sample_size=n_samples,
    )
    raw_imx = float(raw_output[0])
    raw_imy = float(raw_output[1])
    raw_imxy = float(raw_output[2])
    plugin_union = float(raw_output[3])
    recalculated_raw_obj = float(raw_output[4])
    plugin_synergy = float(raw_output[8])
    raw_obj_value = (
        recalculated_raw_obj if raw_obj is None else float(raw_obj)
    )

    n_resamples = int(
        config.get(
            'n_lorenz_resamples',
            config.get('n_obj_bootstrap', 20),
        )
    )
    n_shuffles = int(
        config.get(
            'n_lorenz_shuffles',
            config.get('n_perm', 20),
        )
    )
    if n_resamples < 2:
        raise ValueError(
            f"n_lorenz_resamples must be at least 2; got {n_resamples}."
        )
    if n_shuffles < 1:
        raise ValueError(
            f"n_lorenz_shuffles must be at least 1; got {n_shuffles}."
        )

    merge_weight = float(config.get('lorenz_merge_weight', 0.5))
    if not 0.0 <= merge_weight <= 1.0:
        raise ValueError(
            f"lorenz_merge_weight must be in [0, 1]; got {merge_weight}."
        )

    root_seed = int(
        config.get(
            'lorenz_seed',
            config.get(
                'obj_bootstrap_seed',
                config.get('rng_seed', config.get('seed', 56)),
            ),
        )
    )
    resampling_seed = int(
        config.get('lorenz_resampling_seed', root_seed)
    )
    shuffle_seed = int(
        config.get('lorenz_shuffle_seed', root_seed + 1)
    )
    resampling_config = config.copy()
    resampling_config.update(
        {
            'n_samples': n_samples,
            'n_obj_bootstrap': n_resamples,
            'obj_bootstrap_seed': resampling_seed,
            'covariance_is_sample': True,
            'device': 'cpu',
        }
    )
    resampling = parametric_bootstrap_obj_debias(
        resampling_config,
        covariance=sample_covariance,
        raw_obj=plugin_synergy,
        statistic='syn',
    )

    shuffle_generator = torch.Generator(device='cpu')
    shuffle_generator.manual_seed(shuffle_seed)
    shuffle_config = config.copy()
    shuffle_config.update(
        {
            'T': target,
            'X1': source_x1,
            'X2': source_x2,
            'n_samples': n_samples,
            'n_perm': n_shuffles,
            'rng': shuffle_generator,
            'device': 'cpu',
            'bias_correction': False,
            'pid_bias_term': 'syn',
        }
    )
    shuffle = permutation_null_debias(
        shuffle_config,
        func=broja_venkatesh_bias,
    )

    synergy_resampling_bias = float(resampling['bias'])
    synergy_shuffle_bias = float(shuffle['bias'])
    synergy_merged_bias = (
        merge_weight * synergy_resampling_bias
        + (1.0 - merge_weight) * synergy_shuffle_bias
    )

    natural_log_two = float(
        torch.log(torch.tensor(2.0, dtype=torch.float64))
    )
    wishart_bias_nats = mi_wishart_bias(
        [dx, dy, dm],
        n_samples,
    )
    mi_x1_bias = wishart_bias_nats['bias_mi_1_t'] / natural_log_two
    mi_x2_bias = wishart_bias_nats['bias_mi_2_t'] / natural_log_two
    joint_mi_bias = wishart_bias_nats['bias_tri_mi'] / natural_log_two

    corrected_imx = raw_imx - mi_x1_bias
    corrected_imy = raw_imy - mi_x2_bias
    corrected_imxy = raw_imxy - joint_mi_bias
    corrected_synergy = plugin_synergy - synergy_merged_bias
    corrected_obj = corrected_imxy - corrected_synergy
    plugin_union_bias = plugin_union - corrected_obj
    raw_obj_gap = raw_obj_value - plugin_union
    objective_bias = raw_obj_value - corrected_obj

    corrected_redundancy = corrected_imx + corrected_imy - corrected_obj
    corrected_unique_x1 = corrected_obj - corrected_imy
    corrected_unique_x2 = corrected_obj - corrected_imx

    return {
        'method': 'lorenz_gaussian_merged',
        'raw_obj': raw_obj_value,
        'bias': objective_bias,
        'debiased': corrected_obj,
        'corrected_obj': corrected_obj,
        'configuration': {
            'n_samples': n_samples,
            'dim_target': dm,
            'dim_source_1': dx,
            'dim_source_2': dy,
            'n_resamples': n_resamples,
            'n_shuffles': n_shuffles,
            'merge_weight': merge_weight,
            'covariance_ddof': 1,
            'information_unit': 'bits',
            'resampling_seed': resampling_seed,
            'shuffle_seed': shuffle_seed,
        },
        'plugin': {
            'mi_target_source_1': raw_imx,
            'mi_target_source_2': raw_imy,
            'mi_target_joint_sources': raw_imxy,
            'red': float(raw_output[7]),
            'unq1': float(raw_output[5]),
            'unq2': float(raw_output[6]),
            'syn': plugin_synergy,
            'union_info': plugin_union,
            'obj': raw_obj_value,
        },
        'bias_components': {
            'mi_target_source_1_goodman': mi_x1_bias,
            'mi_target_source_2_goodman': mi_x2_bias,
            'mi_target_joint_sources_goodman': joint_mi_bias,
            'syn_resampling': synergy_resampling_bias,
            'syn_shuffle': synergy_shuffle_bias,
            'syn_merged': synergy_merged_bias,
            'plugin_union_merged': plugin_union_bias,
            'raw_obj_to_plugin_union': raw_obj_gap,
            'obj_merged': objective_bias,
        },
        'corrected': {
            'mi_target_source_1': corrected_imx,
            'mi_target_source_2': corrected_imy,
            'mi_target_joint_sources': corrected_imxy,
            'red': corrected_redundancy,
            'unq1': corrected_unique_x1,
            'unq2': corrected_unique_x2,
            'syn': corrected_synergy,
            'union_info': corrected_obj,
        },
        'surrogates': {
            'resampling_synergies': resampling['bootstrap_values'],
            'shuffle_synergies': shuffle['perm_values'].detach().cpu(),
        },
    }


def equal_direct_wishart_control_obj_debias(
    config: dict,
    covariance,
    raw_obj: float | None = None,
) -> dict:
    """Estimate bias of the raw Venkatesh objective under a known direct tie.

    This objective-only estimator fits the repository's direct Gaussian
    covariance family while imposing the population constraint that the two
    source-target channels are equal. Gaussian bootstrap datasets are drawn
    from that constrained fit and the complete Venkatesh optimization is rerun
    for every dataset with ``unbiased=False``.

    The exact Wishart bias of the mean marginal mutual information is used only
    as a control variate for estimating the objective bias. The function does
    not return corrected marginal mutual informations, apply clipping, or
    replace the observed objective with a pooled mutual-information estimate.

    Inputs:
        config: dict containing:
            known_equal_direct_covariance: bool, which must be exactly ``True``.
            covariance_is_sample: bool, which must be exactly ``True``.
            dx1: int, dimension of source X1.
            dx2: int, dimension of source X2.
            dt: int, dimension of target T.
            n_samples: int, observations represented by ``covariance``.
            n_obj_bootstrap: optional int, number of bootstrap replicates;
                defaults to 20.
            obj_bootstrap_seed: optional int, local bootstrap seed.
            device: optional str or torch.device for bootstrap sampling.
        covariance: torch.Tensor or array-like unbiased sample covariance with
            shape (dt + dx1 + dx2, dt + dx1 + dx2), ordered [T, X1, X2].
        raw_obj: optional float, raw optimized Venkatesh objective for exactly
            ``covariance``. A mismatch with a fresh calculation raises an error.

    Outputs:
        dict containing:
            method: str, correction method name.
            raw_obj: float, uncorrected optimized objective in bits.
            bias: float, additive objective bias to subtract in bits.
            debiased: float, ``raw_obj - bias`` without clipping.
            corrected_obj: float, alias of ``debiased``.
            fitted_parameters: dict with fitted p, unrestricted q/r, and the
                pooled equal-channel correlation.
            fitted_residual: float, objective-minus-mean-MI at the constrained
                covariance.
            marginal_control_bias: float, exact mean marginal-MI bias in bits.
            bootstrap_residual_mean: float, mean optimizer residual in bits.
            bootstrap_residual_std: float, residual standard deviation in bits.
            bootstrap_residual_se: float, Monte Carlo SE in bits.
            bootstrap_residuals: torch.Tensor with shape (n_obj_bootstrap,).
            n_bootstrap: int, number of bootstrap replicates.
            n_samples: int, sample size per replicate.
            seed: int, local bootstrap seed.
            fitted_covariance_min_eigenvalue: float, constrained-fit diagnostic.
            covariance_is_sample: bool, always ``True`` after validation.

    Notes:
        The method is valid only when the direct matching-coordinate covariance
        family and equal source-target channels are known before observing the
        sample. It must not be selected by testing equality on the same sample.
    """

    if config.get('known_equal_direct_covariance') is not True:
        raise ValueError(
            "equal_direct_wishart_control_obj_debias requires "
            "config['known_equal_direct_covariance'] = True."
        )
    if config.get('covariance_is_sample') is not True:
        raise ValueError(
            "covariance must be an ordinary unbiased sample covariance and "
            "requires config['covariance_is_sample'] = True."
        )

    dx = int(config['dx1'])
    dy = int(config['dx2'])
    dm = int(config['dt'])
    n_samples = int(config['n_samples'])
    total_dimension = dm + dx + dy
    covariance_device = (
        covariance.device
        if torch.is_tensor(covariance)
        else config.get('device', 'cpu')
    )
    sample_covariance = torch.as_tensor(
        covariance,
        dtype=torch.float64,
        device=covariance_device,
    )  # array-like (D, D) -> torch (D, D)

    expected_shape = (total_dimension, total_dimension)
    if sample_covariance.ndim != 2 or tuple(sample_covariance.shape) != expected_shape:
        raise ValueError(
            "covariance must be ordered [T, X1, X2] with shape "
            f"{expected_shape}; got {tuple(sample_covariance.shape)}."
        )
    if n_samples <= total_dimension:
        raise ValueError(
            "Bootstrap sample covariances require n_samples > total dimension. "
            f"Got n_samples={n_samples}, dimension={total_dimension}."
        )
    if not bool(torch.isfinite(sample_covariance).all()):
        raise ValueError("covariance contains NaN or infinite values.")

    covariance_scale = max(float(torch.max(torch.abs(sample_covariance))), 1.0)
    maximum_asymmetry = float(
        torch.max(torch.abs(sample_covariance - sample_covariance.T))
    )
    if maximum_asymmetry > 1e-8 * covariance_scale:
        raise ValueError(
            "covariance must be symmetric; maximum absolute asymmetry is "
            f"{maximum_asymmetry:.3e}."
        )
    sample_covariance = (
        sample_covariance + sample_covariance.T
    ) / 2.0  # torch (D, D) -> torch (D, D)
    sample_eigenvalues = torch.linalg.eigvalsh(
        sample_covariance
    )  # torch (D, D) -> torch (D,)
    if float(sample_eigenvalues[0]) <= 0.0:
        raise ValueError(
            "sample covariance must be positive definite; minimum eigenvalue "
            f"is {float(sample_eigenvalues[0]):.3e}."
        )

    standard_deviations = torch.sqrt(
        torch.diag(sample_covariance)
    )  # torch (D, D) -> torch (D,)
    correlation = sample_covariance / torch.outer(
        standard_deviations,
        standard_deviations,
    )  # torch (D, D) / torch (D, D) -> torch (D, D)

    p_dimension = min(dx, dy)
    q_dimension = min(dm, dx)
    r_dimension = min(dm, dy)
    p_indices = torch.arange(
        p_dimension,
        device=sample_covariance.device,
    )  # scalar dimension -> torch (min(dx1, dx2),)
    q_indices = torch.arange(
        q_dimension,
        device=sample_covariance.device,
    )  # scalar dimension -> torch (min(dt, dx1),)
    r_indices = torch.arange(
        r_dimension,
        device=sample_covariance.device,
    )  # scalar dimension -> torch (min(dt, dx2),)
    fitted_p = float(
        torch.mean(
            correlation[
                dm + p_indices,
                dm + dx + p_indices,
            ]
        )
    )
    fitted_q = float(
        torch.mean(
            correlation[
                q_indices,
                dm + q_indices,
            ]
        )
    )
    fitted_r = float(
        torch.mean(
            correlation[
                r_indices,
                dm + dx + r_indices,
            ]
        )
    )
    fitted_equal_channel = 0.5 * (fitted_q + fitted_r)

    target_covariance = torch.eye(
        dm,
        dtype=torch.float64,
        device=sample_covariance.device,
    )  # scalar dimension -> torch (dt, dt)
    source_x1_covariance = torch.eye(
        dx,
        dtype=torch.float64,
        device=sample_covariance.device,
    )  # scalar dimension -> torch (dx1, dx1)
    source_x2_covariance = torch.eye(
        dy,
        dtype=torch.float64,
        device=sample_covariance.device,
    )  # scalar dimension -> torch (dx2, dx2)
    target_x1_cross = torch.zeros(
        (dm, dx),
        dtype=torch.float64,
        device=sample_covariance.device,
    )  # two scalar dimensions -> torch (dt, dx1)
    target_x2_cross = torch.zeros(
        (dm, dy),
        dtype=torch.float64,
        device=sample_covariance.device,
    )  # two scalar dimensions -> torch (dt, dx2)
    source_cross = torch.zeros(
        (dx, dy),
        dtype=torch.float64,
        device=sample_covariance.device,
    )  # two scalar dimensions -> torch (dx1, dx2)
    target_x1_cross[q_indices, q_indices] = fitted_equal_channel
    target_x2_cross[r_indices, r_indices] = fitted_equal_channel
    source_cross[p_indices, p_indices] = fitted_p

    fitted_target_row = torch.cat(
        (target_covariance, target_x1_cross, target_x2_cross),
        dim=1,
    )  # three torch (dt, d_i) blocks -> torch (dt, D)
    fitted_x1_row = torch.cat(
        (target_x1_cross.T, source_x1_covariance, source_cross),
        dim=1,
    )  # three torch (dx1, d_i) blocks -> torch (dx1, D)
    fitted_x2_row = torch.cat(
        (target_x2_cross.T, source_cross.T, source_x2_covariance),
        dim=1,
    )  # three torch (dx2, d_i) blocks -> torch (dx2, D)
    fitted_covariance = torch.cat(
        (fitted_target_row, fitted_x1_row, fitted_x2_row),
        dim=0,
    )  # three torch (d_i, D) rows -> torch (D, D)
    fitted_eigenvalues = torch.linalg.eigvalsh(
        fitted_covariance
    )  # torch (D, D) -> torch (D,)
    fitted_minimum_eigenvalue = float(fitted_eigenvalues[0])
    if fitted_minimum_eigenvalue <= 0.0:
        raise ValueError(
            "constrained direct covariance is not positive definite; minimum "
            f"eigenvalue is {fitted_minimum_eigenvalue:.3e}."
        )
    fitted_cholesky = torch.linalg.cholesky(
        fitted_covariance
    )  # torch (D, D) -> torch (D, D)

    sample_covariance_numpy = (
        sample_covariance.detach().cpu().numpy()
    )  # torch (D, D) -> NumPy (D, D)
    observed_output = tilde_pid.exact_gauss_tilde_pid(
        sample_covariance_numpy,
        dm,
        dx,
        dy,
        unbiased=False,
        sample_size=n_samples,
    )
    recalculated_raw_obj = float(observed_output[4])
    if raw_obj is None:
        raw_obj_value = recalculated_raw_obj
    else:
        raw_obj_value = float(raw_obj)
        objective_tolerance = 1e-8 * max(1.0, abs(recalculated_raw_obj))
        if abs(raw_obj_value - recalculated_raw_obj) > objective_tolerance:
            raise ValueError(
                "raw_obj does not match the objective recalculated from "
                f"covariance: raw_obj={raw_obj_value}, "
                f"recalculated={recalculated_raw_obj}."
            )

    fitted_covariance_numpy = (
        fitted_covariance.detach().cpu().numpy()
    )  # torch (D, D) -> NumPy (D, D)
    fitted_output = tilde_pid.exact_gauss_tilde_pid(
        fitted_covariance_numpy,
        dm,
        dx,
        dy,
        unbiased=False,
        sample_size=n_samples,
    )
    fitted_residual = float(
        fitted_output[4] - 0.5 * (fitted_output[0] + fitted_output[1])
    )

    natural_log_two = float(
        torch.log(torch.tensor(2.0, dtype=torch.float64))
    )
    marginal_control_bias = 0.5 * (
        mi_wishart_bias([dm, dx], n_samples)
        + mi_wishart_bias([dm, dy], n_samples)
    ) / natural_log_two

    n_bootstrap = int(config.get('n_obj_bootstrap', 20))
    if n_bootstrap < 2:
        raise ValueError(
            f"n_obj_bootstrap must be at least 2; got {n_bootstrap}."
        )
    seed = int(
        config.get(
            'obj_bootstrap_seed',
            config.get('rng_seed', config.get('seed', 56)),
        )
    )
    generator = torch.Generator(device=fitted_covariance.device)
    generator.manual_seed(seed)
    bootstrap_residuals = torch.empty(
        n_bootstrap,
        dtype=torch.float64,
        device=fitted_covariance.device,
    )  # empty scalar storage -> torch (B,)

    for bootstrap_index in range(n_bootstrap):
        standard_samples = torch.randn(
            (n_samples, total_dimension),
            dtype=torch.float64,
            device=fitted_covariance.device,
            generator=generator,
        )  # scalar Gaussian draws -> torch (N, D)
        bootstrap_samples = (
            standard_samples @ fitted_cholesky.T
        )  # torch (N, D) @ torch (D, D) -> torch (N, D)
        centered_samples = (
            bootstrap_samples - bootstrap_samples.mean(dim=0, keepdim=True)
        )  # torch (N, D) - torch (1, D) -> torch (N, D)
        bootstrap_covariance = (
            centered_samples.T @ centered_samples
        ) / (n_samples - 1)  # torch (D, N) @ torch (N, D) -> torch (D, D)
        bootstrap_covariance_numpy = (
            bootstrap_covariance.detach().cpu().numpy()
        )  # torch (D, D) -> NumPy (D, D)
        bootstrap_output = tilde_pid.exact_gauss_tilde_pid(
            bootstrap_covariance_numpy,
            dm,
            dx,
            dy,
            unbiased=False,
            sample_size=n_samples,
        )
        bootstrap_residuals[bootstrap_index] = (
            bootstrap_output[4]
            - 0.5 * (bootstrap_output[0] + bootstrap_output[1])
        )

    bootstrap_residual_mean = float(torch.mean(bootstrap_residuals))
    bootstrap_residual_std = float(
        torch.std(bootstrap_residuals, unbiased=True)
    )
    bootstrap_residual_se = bootstrap_residual_std / (n_bootstrap ** 0.5)
    estimated_bias = (
        marginal_control_bias
        + bootstrap_residual_mean
        - fitted_residual
    )
    corrected_obj = raw_obj_value - estimated_bias
    bootstrap_residuals_cpu = (
        bootstrap_residuals.detach().cpu()
    )  # torch (B,) on sampling device -> torch (B,) on CPU

    return {
        'method': 'equal_direct_wishart_control_obj',
        'raw_obj': raw_obj_value,
        'bias': estimated_bias,
        'debiased': corrected_obj,
        'corrected_obj': corrected_obj,
        'fitted_parameters': {
            'p_scale': fitted_p,
            'q_scale_unrestricted': fitted_q,
            'r_scale_unrestricted': fitted_r,
            'equal_channel_scale': fitted_equal_channel,
        },
        'fitted_residual': fitted_residual,
        'marginal_control_bias': marginal_control_bias,
        'bootstrap_residual_mean': bootstrap_residual_mean,
        'bootstrap_residual_std': bootstrap_residual_std,
        'bootstrap_residual_se': bootstrap_residual_se,
        'bootstrap_residuals': bootstrap_residuals_cpu,
        'n_bootstrap': n_bootstrap,
        'n_samples': n_samples,
        'seed': seed,
        'fitted_covariance_min_eigenvalue': fitted_minimum_eigenvalue,
        'covariance_is_sample': True,
    }


def equivalent_channels_obj_debias(
    config: dict,
    covariance=None,
    raw_obj: float | None = None,
) -> dict:
    """Debias Gaussian BROJA PID under a known equivalent-channel constraint.

    This estimator is intended only for models in which the two source-to-target
    Gaussian channels are known a priori to be information-equivalent. It
    replaces the nonregular optimized union objective with the average of the
    two exactly Wishart-corrected source-target mutual informations. The joint
    mutual information is corrected in the same units, and all PID components
    are then reconstructed from the PID identities without clipping.

    Inputs:
        config: dict containing:
            equivalent_channels: bool, which must be exactly ``True`` as an
                explicit assertion that the population channels are equivalent.
            dx1: int, dimension of source X1.
            dx2: int, dimension of source X2.
            dt: int, dimension of target T.
            n_samples: int, number of observations represented by a supplied
                sample covariance.
            covariance_is_sample: bool, which must be exactly ``True`` when
                ``covariance`` is supplied. This prevents accidental correction
                of a theoretical/population or shrinkage covariance.
            X1, X2, T: optional torch.Tensor or array-like sample matrices with
                shapes (n_samples, dx1), (n_samples, dx2), and
                (n_samples, dt). They are required when ``covariance`` is not
                supplied.
            device: optional str or torch.device used when converting samples.
        covariance: optional torch.Tensor or array-like unbiased sample
            covariance with shape (dt + dx1 + dx2, dt + dx1 + dx2), ordered as
            [T, X1, X2]. A population/theoretical covariance should be used to
            generate repeated samples, not passed directly for bias correction.
        raw_obj: optional float, a previously computed uncorrected BROJA
            objective in bits for exactly the same covariance. If supplied, it
            is checked against a fresh calculation.

    Outputs:
        dict containing:
            method: str, correction method name.
            assumption: str, the required equivalent-channel condition.
            raw_obj: float, uncorrected optimized BROJA objective in bits.
            bias: float, raw_obj - corrected_obj, which is the amount to
                subtract from raw_obj.
            debiased: float, the corrected union objective in bits.
            corrected_obj: float, alias of ``debiased``.
            raw_mi: dict with uncorrected pairwise and joint MIs in bits.
            corrected_mi: dict with exact Wishart-corrected MIs in bits.
            wishart_bias_bits: dict with the subtracted MI biases in bits.
            pid: dict with corrected unq1, unq2, red, syn, and union_info. Its
                obj entry retains the raw optimizer objective to match the
                existing PID wrapper convention.
            n_samples: int, sample size represented by the covariance.
            equivalent_channels: bool, always ``True`` after validation.
            covariance_is_sample: bool, always ``True`` after validation.

    Notes:
        The correction is exact in expectation for Gaussian sample covariances
        when the population channels are equivalent. It is not a generic
        estimator: applying it to unequal channels biases the union objective
        by half of the population pairwise-MI difference.
    """

    if config.get('equivalent_channels') is not True:
        raise ValueError(
            "equivalent_channels_obj_debias requires the explicit setting "
            "config['equivalent_channels'] = True. Do not infer channel "
            "equivalence from a noisy sample."
        )

    dx = int(config['dx1'])
    dy = int(config['dx2'])
    dm = int(config['dt'])
    total_dimension = dm + dx + dy

    if covariance is None:
        missing_keys = [key for key in ('T', 'X1', 'X2') if key not in config]
        if missing_keys:
            raise KeyError(
                "covariance was not supplied, so config must contain "
                f"'T', 'X1', and 'X2'. Missing: {missing_keys}."
            )

        sample_device = config.get('device', 'cpu')
        target = torch.as_tensor(
            config['T'],
            dtype=torch.float64,
            device=sample_device,
        )  # array-like (N, dt) -> torch (N, dt)
        source_x1 = torch.as_tensor(
            config['X1'],
            dtype=torch.float64,
            device=sample_device,
        )  # array-like (N, dx1) -> torch (N, dx1)
        source_x2 = torch.as_tensor(
            config['X2'],
            dtype=torch.float64,
            device=sample_device,
        )  # array-like (N, dx2) -> torch (N, dx2)

        sample_arrays = {'T': target, 'X1': source_x1, 'X2': source_x2}
        expected_dimensions = {'T': dm, 'X1': dx, 'X2': dy}
        for name, values in sample_arrays.items():
            if values.ndim != 2:
                raise ValueError(
                    f"config['{name}'] must have shape (N, d); got {tuple(values.shape)}."
                )
            if values.shape[1] != expected_dimensions[name]:
                raise ValueError(
                    f"config['{name}'] has {values.shape[1]} columns; "
                    f"expected {expected_dimensions[name]}."
                )

        observed_sample_size = target.shape[0]
        if source_x1.shape[0] != observed_sample_size or source_x2.shape[0] != observed_sample_size:
            raise ValueError("T, X1, and X2 must contain the same number of samples.")

        configured_sample_size = int(config.get('n_samples', observed_sample_size))
        if configured_sample_size != observed_sample_size:
            raise ValueError(
                f"config['n_samples']={configured_sample_size} does not match "
                f"the sample arrays' size {observed_sample_size}."
            )

        sample_covariance = create_cov_matrix(
            rvs=[target, source_x1, source_x2],
            device=sample_device,
        )['full_cov']  # [three (N, d_i) tensors] -> torch (D, D)
        n_samples = observed_sample_size
    else:
        if 'n_samples' not in config:
            raise KeyError("config['n_samples'] is required when covariance is supplied.")
        if config.get('covariance_is_sample') is not True:
            raise ValueError(
                "A supplied covariance must be an ordinary unbiased sample "
                "covariance and requires config['covariance_is_sample'] = True. "
                "Use a theoretical covariance only to generate repeated samples "
                "and define ground truth."
            )

        covariance_device = (
            covariance.device
            if torch.is_tensor(covariance)
            else config.get('device', 'cpu')
        )
        sample_covariance = torch.as_tensor(
            covariance,
            dtype=torch.float64,
            device=covariance_device,
        )  # array-like (D, D) -> torch (D, D)
        n_samples = int(config['n_samples'])

    if sample_covariance.ndim != 2 or sample_covariance.shape != (
        total_dimension,
        total_dimension,
    ):
        raise ValueError(
            "covariance must be a square [T, X1, X2] matrix with shape "
            f"({total_dimension}, {total_dimension}); got {tuple(sample_covariance.shape)}."
        )
    if n_samples <= total_dimension:
        raise ValueError(
            "The exact Wishart correction requires n_samples > total "
            f"dimension. Got n_samples={n_samples}, dimension={total_dimension}."
        )
    if not bool(torch.isfinite(sample_covariance).all()):
        raise ValueError("covariance contains NaN or infinite values.")

    covariance_scale = max(float(torch.max(torch.abs(sample_covariance))), 1.0)
    maximum_asymmetry = float(
        torch.max(torch.abs(sample_covariance - sample_covariance.T))
    )
    if maximum_asymmetry > 1e-8 * covariance_scale:
        raise ValueError(
            "covariance must be symmetric; maximum absolute asymmetry is "
            f"{maximum_asymmetry:.3e}."
        )

    sample_covariance = (
        sample_covariance + sample_covariance.T
    ) / 2.0  # torch (D, D) -> torch (D, D)
    covariance_eigenvalues = torch.linalg.eigvalsh(
        sample_covariance
    )  # torch (D, D) -> torch (D,)
    minimum_eigenvalue = float(covariance_eigenvalues[0])
    if minimum_eigenvalue <= 0.0:
        raise ValueError(
            "covariance must be positive definite; minimum eigenvalue is "
            f"{minimum_eigenvalue:.3e}."
        )

    target_covariance = sample_covariance[
        :dm, :dm
    ]  # torch (D, D) -> torch (dt, dt)
    source_x1_covariance = sample_covariance[
        dm:dm + dx, dm:dm + dx
    ]  # torch (D, D) -> torch (dx1, dx1)
    source_x2_covariance = sample_covariance[
        dm + dx:, dm + dx:
    ]  # torch (D, D) -> torch (dx2, dx2)
    source_joint_covariance = sample_covariance[
        dm:, dm:
    ]  # torch (D, D) -> torch (dx1 + dx2, dx1 + dx2)
    target_x1_covariance = sample_covariance[
        :dm + dx, :dm + dx
    ]  # torch (D, D) -> torch (dt + dx1, dt + dx1)
    target_x2_indices = torch.cat(
        (
            torch.arange(dm, device=sample_covariance.device),
            torch.arange(dm + dx, total_dimension, device=sample_covariance.device),
        )
    )  # two index tensors -> torch (dt + dx2,)
    target_x2_covariance = sample_covariance.index_select(
        0, target_x2_indices
    ).index_select(
        1, target_x2_indices
    )  # torch (D, D) -> torch (dt + dx2, dt + dx2)

    covariance_blocks = {
        'T': target_covariance,
        'X1': source_x1_covariance,
        'X2': source_x2_covariance,
        'X1X2': source_joint_covariance,
        'TX1': target_x1_covariance,
        'TX2': target_x2_covariance,
        'TX1X2': sample_covariance,
    }
    log_determinants = {}
    for name, covariance_block in covariance_blocks.items():
        sign, log_abs_determinant = torch.linalg.slogdet(covariance_block)
        if float(sign) <= 0.0:
            raise ValueError(
                f"The {name} covariance block must have positive determinant."
            )
        log_determinants[name] = float(log_abs_determinant)

    natural_log_two = float(torch.log(torch.tensor(2.0, dtype=torch.float64)))
    raw_imx = 0.5 * (
        log_determinants['T']
        + log_determinants['X1']
        - log_determinants['TX1']
    ) / natural_log_two
    raw_imy = 0.5 * (
        log_determinants['T']
        + log_determinants['X2']
        - log_determinants['TX2']
    ) / natural_log_two
    raw_imxy = 0.5 * (
        log_determinants['T']
        + log_determinants['X1X2']
        - log_determinants['TX1X2']
    ) / natural_log_two

    sample_covariance_numpy = (
        sample_covariance.detach().cpu().numpy()
    )  # torch (D, D) -> NumPy (D, D)
    raw_output = tilde_pid.exact_gauss_tilde_pid(
        sample_covariance_numpy,
        dm,
        dx,
        dy,
        unbiased=False,
        sample_size=n_samples,
    )
    fitted_obj = float(raw_output[4])
    if not bool(torch.isfinite(torch.tensor(fitted_obj))):
        raise RuntimeError("The BROJA optimizer returned a non-finite raw objective.")

    if raw_obj is None:
        raw_obj_value = fitted_obj
    else:
        raw_obj_value = float(raw_obj)
        objective_tolerance = 1e-8 * max(1.0, abs(fitted_obj))
        if abs(raw_obj_value - fitted_obj) > objective_tolerance:
            raise ValueError(
                "raw_obj does not match the objective recalculated from covariance: "
                f"raw_obj={raw_obj_value}, recalculated={fitted_obj}."
            )

    imx_bias_bits = mi_wishart_bias([dm, dx], n_samples) / natural_log_two
    imy_bias_bits = mi_wishart_bias([dm, dy], n_samples) / natural_log_two
    imxy_bias_bits = (
        mi_wishart_bias([dm, dx + dy], n_samples) / natural_log_two
    )

    corrected_imx = raw_imx - imx_bias_bits
    corrected_imy = raw_imy - imy_bias_bits
    corrected_imxy = raw_imxy - imxy_bias_bits
    corrected_obj = 0.5 * (corrected_imx + corrected_imy)

    unique_x1 = corrected_obj - corrected_imy
    unique_x2 = corrected_obj - corrected_imx
    redundancy = corrected_imx + corrected_imy - corrected_obj
    synergy = corrected_imxy - corrected_obj
    estimated_bias = raw_obj_value - corrected_obj

    raw_mi = {
        'bi_mi_1': raw_imx,
        'bi_mi_2': raw_imy,
        'tri_mi': raw_imxy,
    }
    corrected_mi = {
        'bi_mi_1': corrected_imx,
        'bi_mi_2': corrected_imy,
        'tri_mi': corrected_imxy,
    }
    wishart_bias_bits = {
        'bi_mi_1': imx_bias_bits,
        'bi_mi_2': imy_bias_bits,
        'tri_mi': imxy_bias_bits,
    }
    corrected_pid = {
        'unq1': unique_x1,
        'unq2': unique_x2,
        'red': redundancy,
        'syn': synergy,
        'union_info': corrected_obj,
        'obj': raw_obj_value,
    }

    return {
        'method': 'equivalent_channels_wishart_obj',
        'assumption': 'population source-target channels are information-equivalent',
        'raw_obj': raw_obj_value,
        'bias': estimated_bias,
        'debiased': corrected_obj,
        'corrected_obj': corrected_obj,
        'raw_mi': raw_mi,
        'corrected_mi': corrected_mi,
        'wishart_bias_bits': wishart_bias_bits,
        'pid': corrected_pid,
        'n_samples': n_samples,
        'equivalent_channels': True,
        'covariance_is_sample': True,
    }
