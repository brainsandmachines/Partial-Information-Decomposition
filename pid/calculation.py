"""Eigen-PID calculation with the project's sample/covariance interface."""

import torch
from eigen_pid import debias_eigen_pid, eigen_pid

from pid.utils import create_cov_matrix


def pid_calc(
    config: dict | None = None,
    sources: list | None = None,
    target: list | None = None,
    rng: torch.Generator | None = None,
    method: str | None = None,
    on_rvs: callable = None,
    covariance: torch.Tensor | None = None,
) -> tuple[dict, dict]:
    """Calculate Gaussian BROJA PID in bits with Eigen-PID.

    Inputs:
        config: dict or None, dimensions and bias settings. ``bias_correction``
            selects the package's combined permutation/bootstrap correction;
            ``n_shuffles``, ``n_resamples``, and ``bias_seed`` are optional.
        sources: list [X1, X2] of tensors shaped (N, dx1) and (N, dx2).
        target: list [T] containing a tensor shaped (N, dt).
        rng: torch.Generator or None, fallback source for the correction seed.
        method: str or None; only eigen/eigen_pid (or the default) is supported.
        on_rvs: optional callable transforming the source list once.
        covariance: optional tensor/array shaped (D, D), ordered [T, X1, X2].

    Outputs:
        tuple[dict, dict]: PID atoms and mutual information values in bits.

    Bias correction requires samples because ``debias_eigen_pid`` constructs
    permutation and parametric-bootstrap surrogates from the observations.
    Population covariance input is therefore supported only without correction.
    """
    if method not in (None, "eigen", "eigen_pid"):
        raise ValueError(f"Only Eigen PID is supported; received {method!r}.")

    config = dict(config or {})
    bias_correction = bool(config.get("bias_correction", False))
    if on_rvs is not None:
        sources = on_rvs(sources)  # [(N, dx1), (N, dx2)] -> transformed source shapes

    if sources is not None and target is not None:
        dm, dx, dy = target[0].shape[1], sources[0].shape[1], sources[1].shape[1]
    else:
        try:
            dm, dx, dy = config["dt"], config["dx1"], config["dx2"]
        except KeyError as error:
            raise ValueError("Covariance input requires config dimensions dt, dx1, and dx2.") from error

    print("\nCalculating PID using Eigen PID...")
    if bias_correction:
        if sources is None or target is None:
            raise ValueError("Bias-corrected Eigen PID requires source and target samples.")
        reference = torch.as_tensor(target[0])
        data = torch.cat(
            [
                torch.as_tensor(rv, dtype=torch.float64, device=reference.device)
                for rv in (target[0], sources[0], sources[1])
            ],
            dim=1,
        )  # [(N, dt), (N, dx1), (N, dx2)] -> (N, D) ordered [T, X1, X2]
        fallback_seed = rng.initial_seed() if rng is not None else 56
        correction = debias_eigen_pid(
            data,
            dm,
            dx,
            dy,
            n_shuffles=config.get("n_shuffles", 20),
            n_resamples=config.get("n_resamples", 20),
            seed=config.get(
                "bias_seed",
                config.get("rng_seed", config.get("seed", fallback_seed)),
            ),
        )  # (N, D) -> raw, permutation, and combined scalar estimates in bits
        result = correction["combined"]
        raw_union = correction["raw"]["union_info"]
        extra = {
            "bias_method": "combined",
            "raw": correction["raw"],
            "permutation": correction["permutation"],
            "mi_bias": correction["mi_bias"],
            "union_bias": correction["union_bias"],
            "timings": correction["timings"],
            "iteration_seconds": correction["iteration_seconds"],
            "analytical_correction_approximate": correction[
                "analytical_correction_approximate"
            ],
        }
    else:
        if covariance is None:
            if sources is None or target is None:
                raise ValueError("Eigen PID requires samples or a covariance.")
            covariance = create_cov_matrix([target[0], sources[0], sources[1]])["full_cov"]
            # [(N, dt), (N, dx1), (N, dx2)] -> (D, D) ordered [T, X1, X2]
        covariance = torch.as_tensor(covariance, dtype=torch.float64)
        # array-like (D, D) -> float64 tensor (D, D)
        if covariance.ndim != 2:
            raise ValueError("pid_calc expects one covariance matrix, not a batch.")
        result = eigen_pid(covariance, dm, dx, dy)
        # (D, D) -> scalar information tensors and (dt,) generalized eigenvalues
        raw_union = float(result["union_info"])
        extra = {
            "generalized_eigenvalues": result["gen_ev"].detach().cpu().tolist(),
            "log2_generalized_eigenvalues": result["log2_gen_ev"].detach().cpu().tolist(),
            # (dt,) tensors -> (dt,) Python lists
        }

    values = {
        key: float(result[key])
        for key in ("mix", "miy", "mixy", "uix", "uiy", "red", "syn", "union_info")
    }
    pid = {
        "red": values["red"],
        "unq1": values["uix"],
        "unq2": values["uiy"],
        "syn": values["syn"],
        "obj": float(raw_union),
        "union_info": values["union_info"],
        "obj_bias": float(raw_union) - values["union_info"],
        "method": "eigen_pid",
        "extra": extra,
    }
    mi = {
        "tri_mi": values["mixy"],
        "bi_mi_1": values["mix"],
        "bi_mi_2": values["miy"],
    }
    return pid, mi
