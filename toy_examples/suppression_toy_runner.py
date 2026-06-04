"""Shared runners for suppression toy examples."""

import numpy as np

from encoding_model.commonality import commonality_analysis

DEFAULT_METHODS = ("standard", "ols_cv", "ridge_cv")
SPLIT_SIGNAL = "split_signal"
FEATURE_CORRELATION = "feature_correlation"


def generate_correlated_features(n, p, rho, rng):
    """Generate samples with an AR(1)-style covariance structure."""
    ii, jj = np.indices((p, p))
    cov_matrix = rho ** np.abs(ii - jj)
    return rng.multivariate_normal(mean=np.zeros(p), cov=cov_matrix, size=n)


def _apply_mixing(rng, X_M1, X_M2, mixing_dimension):
    if mixing_dimension is None:
        return X_M1, X_M2

    mixing_matrix_M1 = rng.standard_normal((X_M1.shape[1], mixing_dimension))
    mixing_matrix_M2 = rng.standard_normal((X_M2.shape[1], mixing_dimension))
    return X_M1 @ mixing_matrix_M1, X_M2 @ mixing_matrix_M2


def _split_signal_sources(rng, n, p, snr):
    if n % 2 != 0:
        raise ValueError("split_signal experiments require an even n.")

    signal = rng.standard_normal((n, p))
    real_features = signal[: n // 2, :]
    spurious_features = signal[n // 2 :, :]
    rand_perm = rng.permutation(n // 2)

    shuffled_real = real_features[rand_perm]
    shuffled_spurious = spurious_features[rand_perm]

    betas = rng.standard_normal(p)
    target_signal = signal @ betas
    noise_std = np.std(target_signal) / snr
    target = target_signal + noise_std * rng.standard_normal(n)

    X_M1 = np.concatenate([real_features, shuffled_spurious])
    X_M2 = np.concatenate([shuffled_real, shuffled_spurious])
    return X_M1, X_M2, target


def _feature_correlation_sources(rng, n, p, snr, rho):
    features = generate_correlated_features(2 * n, 2 * p, rho=rho, rng=rng)
    real_features = features[:n, :p].copy()
    spurious_features = features[n:, p:].copy()
    rand_perm = rng.permutation(n)

    shuffled_real = real_features[rand_perm]
    shuffled_spurious = spurious_features[rand_perm]

    betas = rng.standard_normal(p)
    target_signal = real_features @ betas
    noise_std = np.std(target_signal) / snr
    target = target_signal + noise_std * rng.standard_normal(n)

    X_M1 = np.hstack([real_features, shuffled_spurious])
    X_M2 = np.hstack([shuffled_real, shuffled_spurious])
    return X_M1, X_M2, target


def build_toy_sources(rng, n, p, snr, experiment_kind, rho=1):
    if experiment_kind == SPLIT_SIGNAL:
        return _split_signal_sources(rng, n, p, snr)
    if experiment_kind == FEATURE_CORRELATION:
        return _feature_correlation_sources(rng, n, p, snr, rho)
    raise ValueError(
        f"Unknown experiment_kind: {experiment_kind}. "
        f"Use '{SPLIT_SIGNAL}' or '{FEATURE_CORRELATION}'."
    )


def run_toy_experiment(
    rng,
    n=1024,
    p=100,
    mixing_dimension=None,
    snr=10.0,
    method="standard",
    experiment_kind=SPLIT_SIGNAL,
    rho=1,
):
    """Run one toy suppression/commonality experiment."""
    X_M1, X_M2, target = build_toy_sources(
        rng=rng,
        n=n,
        p=p,
        snr=snr,
        experiment_kind=experiment_kind,
        rho=rho,
    )
    X_M1, X_M2 = _apply_mixing(rng, X_M1, X_M2, mixing_dimension)

    decomp = commonality_analysis(
        X_M1,
        X_M2,
        target,
        method=method,
        scale_by_target_variance=(experiment_kind == SPLIT_SIGNAL),
    )

    print(f"{method} analysis of target:")
    for key, value in decomp.items():
        print(f"- {key}: {value:.4f}")

    if experiment_kind == SPLIT_SIGNAL and method == "standard":
        total_variance = np.var(target, ddof=1)
        sum_of_components = (
            decomp["unique_X1"]
            + decomp["unique_X2"]
            + decomp["common"]
            + decomp["unexplained"]
        )
        assert np.isclose(total_variance, sum_of_components), (
            "Decomposed components do not sum to total variance!"
        )

    return decomp


def run_all_toy_methods(
    rng_seed,
    n,
    p,
    mixing_dimension,
    snr,
    experiment_kind,
    methods=DEFAULT_METHODS,
    report_negative_common=False,
    rho=1,
):
    """Run all requested commonality methods with a fixed seed."""
    results = {}
    common_negative = []

    for method in methods:
        print(f"\n--- {method.upper()} ---")
        rng = np.random.default_rng(seed=rng_seed)
        decomp = run_toy_experiment(
            rng,
            n=n,
            p=p,
            mixing_dimension=mixing_dimension,
            snr=snr,
            method=method,
            experiment_kind=experiment_kind,
            rho=rho,
        )
        results[method] = decomp
        common_negative.append(decomp["common"] < 0)

    if report_negative_common:
        if all(common_negative):
            print("\nAll common variance estimates are negative")
            print(
                f"With mixing_dimension={mixing_dimension} and snr={snr}, "
                "variance partitioning was broken."
            )
        else:
            print("\nNot all common variance estimates are negative")

    return results


def run_default_factorial_scenarios(
    experiment_kind,
    n=1000,
    p=100,
    seed=42,
    report_negative_common=False,
):
    """Run the standard low/high SNR by mixing-dimension toy scenarios."""
    scenarios = (
        ("Experiment 1: LOW SNR + NO MIXING", None, 1.0),
        ("Experiment 2: LOW SNR + INVERTIBLE MIXING (200->200)", 200, 1.0),
        ("Experiment 3: LOW SNR + LOSSY MIXING (200->100)", 100, 1.0),
        ("Experiment 4: HIGH SNR + NO MIXING", None, 10.0),
        ("Experiment 5: HIGH SNR + INVERTIBLE MIXING (200->200)", 200, 10.0),
        ("Experiment 6: HIGH SNR + LOSSY MIXING (200->100)", 100, 10.0),
    )

    for label, mixing_dimension, snr in scenarios:
        print("\n" + "=" * 70)
        print(label)
        print("=" * 70)
        run_all_toy_methods(
            seed,
            n,
            p,
            mixing_dimension=mixing_dimension,
            snr=snr,
            experiment_kind=experiment_kind,
            report_negative_common=report_negative_common,
        )
