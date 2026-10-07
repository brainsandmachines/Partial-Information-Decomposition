"""Eigen PID calculation with the project's existing sample/covariance interface."""

from pathlib import Path
import sys

import numpy as np
import torch
from eigen_pid import debias_eigen_pid, eigen_pid

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

from Partial_Information_Decomposition.bias_functions import (
    broja_eigen_bias,
    permutation_null_debias,
)
from Partial_Information_Decomposition.PID_util import create_cov_matrix


def pid_calc(config=None, sources=None, target=None, rng=torch.Generator().manual_seed(56),
             method=None, on_rvs: callable = None, covariance: torch.Tensor = None,
             param_bias=False):
    """Calculate Eigen PID while preserving the existing calling convention.

    Inputs:
        config: dict with bias settings and dt/dx1/dx2 for covariance input.
        sources: list [X1, X2] of tensors shaped (N, dx1) and (N, dx2).
        target: list [T] containing a tensor shaped (N, dt).
        rng: random generator retained for caller compatibility.
        method: str or None; only eigen/eigen_pid (or the default) is supported.
        on_rvs: optional callable transforming the source list once.
        covariance: optional tensor/array shaped (D, D), ordered [T, X1, X2].
        param_bias: bool enabling the configured objective correction.
    Outputs:
        tuple[dict, dict]: PID atoms and mutual information, in bits.
    """
    if method not in (None, "eigen", "eigen_pid"):
        raise ValueError(f"Only Eigen PID is supported; received {method!r}.")
    config = dict(config or {})
    config.setdefault("bias_correction", False)
    if on_rvs is not None:
        sources = on_rvs(sources)  # [(N, dx1), (N, dx2)] -> transformed source shapes
    if sources is not None and target is not None:
        config.update(dt=target[0].shape[1], dx1=sources[0].shape[1], dx2=sources[1].shape[1])
    print("\nCalculating PID using Eigen PID...")
    return eigen_pid_wrapper(config, sources, target, covariance, rng,
                             param_bias=param_bias)


def eigen_pid_wrapper(config: dict, sources: list, target: list,
                      covariance: torch.Tensor, rng: torch.Generator,
                      on_rvs: callable = None, param_bias: bool = False):
    """Adapt the installed eigen_pid estimator to the project's PID/MI schema.

    Inputs:
        config: dict containing dt/dx1/dx2 and bias_correction. With param_bias,
            param_bias_method selects 'permutation' (existing local correction)
            or 'combined' (eigen_pid.debias_eigen_pid). The latter accepts
            n_shuffles/n_resamples (positive ints, default 20) and bias_seed
            (int, defaults to rng_seed, seed, then 56).
        sources: list [X1, X2] of (N, dx1)/(N, dx2) sample tensors, or None.
        target: list [T] of one (N, dt) sample tensor, or None.
        covariance: tensor/array (D, D) ordered [T, X1, X2], or None.
        rng: random generator accepted for compatibility.
        on_rvs: callable accepted for compatibility; pid_calc applies it.
        param_bias: bool selecting permutation or library combined correction.
    Outputs:
        tuple[dict, dict]: Python numeric PID/MI values in bits, raw obj,
        corrected union_info, obj_bias, and Eigen diagnostics under extra.

    A supplied covariance is a population input unless an explicit objective
    correction is requested. The ordinary sample-bias option retains the
    existing Cai MI correction, proportional union scaling, and bounds.
    Objective-corrected atoms are not clipped.
    Combined correction delegates to the library and is not a reproduction
    of the retired local Lorenz correction.
    """
    del rng, on_rvs
    dm, dx, dy = config["dt"], config["dx1"], config["dx2"]
    if covariance is None:
        if sources is None or target is None:
            raise ValueError("Eigen PID requires samples or a covariance.")
        cov = create_cov_matrix([target[0], sources[0], sources[1]])["full_cov"]
        # [(N, dt), (N, dx1), (N, dx2)] -> (D, D), D = dt + dx1 + dx2
        sample_size = target[0].shape[0]
        bias_correction = bool(config["bias_correction"])
    else:
        cov = covariance
        sample_size = config.get("n_samples")
        bias_correction = False
    cov = torch.as_tensor(cov, dtype=torch.float64)  # (D, D) -> float64 (D, D)
    if cov.ndim != 2:
        raise ValueError("pid_calc expects one covariance matrix, not a batch.")
    result = eigen_pid(cov, dm, dx, dy)
    # (D, D) -> scalar information tensors and (dt,) generalized eigenvalues
    imx, imy, imxy = (float(result[key]) for key in ("mix", "miy", "mixy"))
    obj = float(result["union_info"])
    union_info = obj
    correction = None

    if param_bias:
        correction_config = dict(config)
        correction_config["n_samples"] = sample_size
        correction_method = config.get("param_bias_method", "permutation")
        if correction_method not in ("permutation", "combined"):
            raise ValueError(f"Unsupported objective correction: {correction_method!r}.")
        if sources is None or target is None:
            raise ValueError("Objective correction requires source and target samples.")
        correction_config.update(X1=sources[0], X2=sources[1], T=target[0])
        if correction_method == "combined":
            data = torch.cat([
                torch.as_tensor(rv, dtype=cov.dtype, device=cov.device)
                for rv in (target[0], sources[0], sources[1])
            ], dim=1)  # [(N, dt), (N, dx1), (N, dx2)] -> (N, D)
            sample_cov = torch.cov(data.T)  # (N, D) -> (D, D), correction=1
            if not torch.allclose(cov, sample_cov, rtol=1e-7, atol=1e-10):
                raise ValueError("Combined correction requires the covariance of the supplied samples.")
            correction = debias_eigen_pid(
                data, dm, dx, dy,
                n_shuffles=config.get("n_shuffles", 20),
                n_resamples=config.get("n_resamples", 20),
                seed=config.get("bias_seed", config.get("rng_seed", config.get("seed", 56))),
            )  # (N, D) -> dictionaries of scalar information estimates in bits
            obj = correction["raw"]["union_info"]
            corrected = correction["combined"]
            imx, imy, imxy = (corrected[key] for key in ("mix", "miy", "mixy"))
            union_info = corrected["union_info"]
        else:
            correction = permutation_null_debias(correction_config, func=broja_eigen_bias)
            union_info = obj - correction.get("bias", correction["perm_mean"])
    elif bias_correction:
        if sample_size <= dm + dx + dy:
            raise ValueError("Sample-bias correction needs n_samples > dt + dx1 + dx2.")
        # Preserve the existing finite-sample correction rather than selecting
        # a different resampling experiment when changing the PID estimator.
        entropy_biases = {
            d: float(np.log1p(-np.arange(1, d + 1) / sample_size).sum() / (2 * np.log(2)))
            for d in (dm, dx, dy, dx + dy, dm + dx, dm + dy, dm + dx + dy)
        }  # scalar dimensions -> (d,) log terms -> scalar entropy biases
        raw_joint = imxy
        imx = max(imx - entropy_biases[dm] - entropy_biases[dx] + entropy_biases[dm + dx], 0.0)
        imy = max(imy - entropy_biases[dm] - entropy_biases[dy] + entropy_biases[dm + dy], 0.0)
        imxy = max(imxy - entropy_biases[dm] - entropy_biases[dx + dy]
                   + entropy_biases[dm + dx + dy], 0.0, imx, imy)
        factor = imxy / raw_joint if abs(raw_joint) > np.finfo(float).eps else 1.0
        union_info = min(max(obj * factor, imx, imy), imx + imy, imxy)

    pid = {
        "red": imx + imy - union_info,
        "unq1": union_info - imy,
        "unq2": union_info - imx,
        "syn": imxy - union_info,
        "obj": obj,
        "union_info": union_info,
        "obj_bias": obj - union_info,
        "method": "eigen_pid",
        "extra": {
            "generalized_eigenvalues": result["gen_ev"].detach().cpu().tolist(),
            "log2_generalized_eigenvalues": result["log2_gen_ev"].detach().cpu().tolist(),
            # (dt,) tensors -> (dt,) Python lists
        },
    }
    if correction is not None and "mi_bias" in correction:
        pid["extra"].update(bias_method="combined", mi_bias=correction["mi_bias"],
                            union_bias=correction["union_bias"])
    return pid, {"tri_mi": imxy, "bi_mi_1": imx, "bi_mi_2": imy}
