"""Compare Pearson-RDM RSA and Gaussian BROJA unique information on matched sources.

Construction (all variables d-dimensional, P = I - 11^T / d):
    T = Z + M 1 + eps_T,   X = Z + M 1 + eps_X,   Y = Z + N 1 + eps_Y,
with Z ~ N(0, P), M, N ~ N(0, 1), and eps ~ N(0, noise_variance I). Pearson RDMs
remove each stimulus pattern's mean across features, so they cannot see M or N.
Gaussian BROJA PID uses the full covariance and can.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm, compare

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pid.calculation import pid_calc
from pid.utils import create_cov_matrix
from simulations.rsa_vs_pid.plot_results import plot_all


CONFIG = {
    "dimensions": [50,100,500],
    "sample_sizes": [2000,10000,50000],
    "n_trials": 100,
    "base_seed": 56,
    "noise_variance": 0.1,
    "n_shuffles": 20,
    "n_resamples": 20,
    "output_dir": "simulation_results/rsa_vs_pid",
    "overwrite_outputs": True,
}

PID_KEYS = ("red", "unq_x", "unq_y", "syn", "mi_tx", "mi_ty", "mi_txy")
TRIAL_FIELDS = (
    "dimension",
    "n_samples",
    "trial",
    "seed",
    "noise_variance",
    "rsa_x",
    "rsa_y",
    "delta_rsa",
    *PID_KEYS,
    "delta_ui",
    *(f"{key}_raw" for key in PID_KEYS),
    "delta_ui_raw",
    "cov_min_eig",
    "cov_condition",
    "bias_method",
    "n_shuffles",
    "n_resamples",
    "rsa_seconds",
    "pid_seconds",
)
SUMMARY_METRICS = (
    "rsa_x",
    "rsa_y",
    "delta_rsa",
    *PID_KEYS,
    "delta_ui",
    *(f"{key}_raw" for key in PID_KEYS),
    "delta_ui_raw",
)
POSITIVE_FRACTION_METRICS = ("delta_ui", "delta_ui_raw")
IDENTITY_ATOL = 1e-9


def validate_config(config: dict) -> None:
    """Validate the experiment configuration.

    Inputs:
        config: dict with the keys of ``CONFIG``. Dimensions, sample sizes,
            trial and resampling counts must be positive integers, and every
            sample size must satisfy n >= 3d + 1 so the (3d, 3d) joint sample
            covariance can be full rank.

    Outputs:
        None. Raises TypeError or ValueError when a setting is invalid.
    """
    missing_keys = set(CONFIG).difference(config)
    if missing_keys:
        raise ValueError(f"Missing config keys: {sorted(missing_keys)}")

    for key in ("dimensions", "sample_sizes"):
        values = config[key]
        if not isinstance(values, (list, tuple)) or not values:
            raise TypeError(f"config[{key!r}] must be a non-empty list of integers")
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values):
            raise ValueError(f"Every entry of config[{key!r}] must be a positive integer")
    for key in ("n_trials", "n_shuffles", "n_resamples"):
        value = config[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"config[{key!r}] must be a positive integer")
    if not isinstance(config["base_seed"], int) or isinstance(config["base_seed"], bool):
        raise TypeError("config['base_seed'] must be an integer")
    if not isinstance(config["noise_variance"], (int, float)) or config["noise_variance"] <= 0:
        raise ValueError("config['noise_variance'] must be a positive number")
    if not isinstance(config["output_dir"], (str, Path)):
        raise TypeError("config['output_dir'] must be a string or pathlib.Path")
    if not isinstance(config["overwrite_outputs"], bool):
        raise TypeError("config['overwrite_outputs'] must be True or False")

    too_small = [
        (dimension, n_samples)
        for dimension in config["dimensions"]
        for n_samples in config["sample_sizes"]
        if n_samples < 3 * dimension + 1
    ]
    if too_small:
        raise ValueError(
            "Each sample size must satisfy n >= 3d + 1 for a full-rank joint "
            f"covariance; invalid (d, n) pairs: {too_small}"
        )


def sample_shift_invariant_example(
    rng: np.random.Generator,
    n: int,
    d: int,
    noise_variance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the construction in which X and Y differ only along the ones direction.

    Inputs:
        rng: np.random.Generator controlling every draw. The draw order is
            fixed: G (for Z), M, N, eps_T, eps_X, eps_Y.
        n: int, number of stimulus observations.
        d: int, dimension of T, X, and Y.
        noise_variance: float, variance of each independent noise coordinate.

    Outputs:
        tuple (x, y, target) of float64 arrays, each shaped (n, d), ordered
        [X1, X2, T]. X shares Z and M with the target; Y shares only Z and
        carries the target-independent N along the ones direction.
    """
    ones = np.ones(d)
    projection = np.eye(d) - np.outer(ones, ones) / d  # (d,) -> (d, d) centering projection P
    z = rng.standard_normal((n, d)) @ projection  # (n, d) @ (d, d) -> (n, d); Cov(Z) = P P^T = P
    m = rng.standard_normal((n, 1))  # -> (n, 1)
    n_shift = rng.standard_normal((n, 1))  # -> (n, 1)
    noise_std = math.sqrt(noise_variance)
    eps_target = noise_std * rng.standard_normal((n, d))  # -> (n, d)
    eps_x = noise_std * rng.standard_normal((n, d))  # -> (n, d)
    eps_y = noise_std * rng.standard_normal((n, d))  # -> (n, d)

    target = z + m * ones + eps_target  # (n, 1) * (d,) -> (n, d)
    x = z + m * ones + eps_x  # (n, 1) * (d,) -> (n, d)
    y = z + n_shift * ones + eps_y  # (n, 1) * (d,) -> (n, d)
    return x, y, target


def rsa_scores(target: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Calculate Spearman RSA between Pearson-correlation RDMs with rsatoolbox.

    Inputs:
        target: np.ndarray shaped (n_stimuli, n_features), the target T.
        x: np.ndarray shaped (n_stimuli, n_features), source X.
        y: np.ndarray shaped (n_stimuli, n_features), source Y.
        Rows are stimuli (rsatoolbox observations); columns are features
        (rsatoolbox channels).

    Outputs:
        dict with float ``rsa_x`` = Spearman(D_T, D_X) and ``rsa_y`` =
        Spearman(D_T, D_Y), where D_A(i, j) = 1 - Pearson(A_i, A_j).
    """
    n_stimuli = target.shape[0]
    datasets = [
        Dataset(measurements=array, obs_descriptors={"stimulus": np.arange(n_stimuli)})
        for array in (target, x, y)
    ]  # three (n_stimuli, n_features) arrays -> three datasets with n_stimuli observations
    rdm_target = calc_rdm(datasets[0], method="correlation")
    # (n_stimuli, n_features) -> one RDM with n_stimuli (n_stimuli - 1) / 2 dissimilarities
    rdm_sources = calc_rdm(datasets[1:], method="correlation")
    # two (n_stimuli, n_features) datasets -> two RDMs, ordered [X, Y]

    n_pairs = n_stimuli * (n_stimuli - 1) // 2
    for name, rdms, n_rdm in (("T", rdm_target, 1), ("X, Y", rdm_sources, 2)):
        if rdms.n_cond != n_stimuli or rdms.dissimilarities.shape != (n_rdm, n_pairs):
            raise RuntimeError(
                f"RDM for {name} has n_cond={rdms.n_cond} and dissimilarities "
                f"{rdms.dissimilarities.shape}; expected {n_stimuli} and {(n_rdm, n_pairs)}."
            )

    scores = compare(rdm_target, rdm_sources, method="spearman")  # 1 RDM vs 2 RDMs -> (1, 2)
    return {"rsa_x": float(scores[0, 0]), "rsa_y": float(scores[0, 1])}


def run_trial(dimension: int, n_samples: int, trial: int, seed: int, config: dict) -> dict:
    """Run RSA and Gaussian BROJA PID on one set of generated observations.

    Inputs:
        dimension: int d, dimension of T, X, and Y.
        n_samples: int n, number of stimulus observations.
        trial: int, trial index within the (d, n) condition.
        seed: int, seed for data generation and for the PID bias correction.
        config: dict with ``noise_variance``, ``n_shuffles``, and ``n_resamples``.

    Outputs:
        dict, one row with the fields in ``TRIAL_FIELDS``. PID values are in
        bits; unsuffixed atoms are bias-corrected (combined) and ``_raw``
        atoms are uncorrected. Raises RuntimeError when a check fails.
    """
    # 1. Sample T, X, Y. RSA and PID below use exactly these arrays.
    rng = np.random.default_rng(seed)
    x, y, target = sample_shift_invariant_example(rng, n_samples, dimension, config["noise_variance"])

    # 2. Validate the generated data.
    for name, array in (("T", target), ("X", x), ("Y", y)):
        if array.shape != (n_samples, dimension):
            raise RuntimeError(f"{name} has shape {array.shape}; expected {(n_samples, dimension)}.")
        if not np.isfinite(array).all():
            raise RuntimeError(f"{name} contains non-finite values (seed={seed}).")

    # 3. RSA: Pearson-correlation RDMs compared with Spearman correlation.
    rsa_start = time.perf_counter()
    rsa = rsa_scores(target, x, y)
    rsa_seconds = time.perf_counter() - rsa_start
    if not all(math.isfinite(value) for value in rsa.values()):
        raise RuntimeError(f"Non-finite RSA scores {rsa} (seed={seed}).")

    # 4. Joint sample covariance ordered [T, X, Y], checked before PID.
    target_t, x_t, y_t = (torch.from_numpy(array) for array in (target, x, y))
    # three float64 (n, d) arrays -> three float64 (n, d) tensors
    covariance = create_cov_matrix([target_t, x_t, y_t])["full_cov"]
    # three (n, d) tensors -> (3d, 3d) unbiased covariance ordered [T, X, Y]
    joint_dim = 3 * dimension
    if covariance.shape != (joint_dim, joint_dim):
        raise RuntimeError(f"Covariance shape {tuple(covariance.shape)}; expected {(joint_dim, joint_dim)}.")
    if not torch.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12):
        raise RuntimeError(f"Sample covariance is not symmetric (seed={seed}).")
    eigenvalues = torch.linalg.eigvalsh(covariance)  # (3d, 3d) -> (3d,) ascending eigenvalues
    cov_min_eig = float(eigenvalues[0])
    if cov_min_eig <= 0:
        raise RuntimeError(f"Sample covariance is not positive definite: min eigenvalue {cov_min_eig:.3e}.")
    cov_condition = float(eigenvalues[-1] / eigenvalues[0])

    # 5. Gaussian BROJA PID with the project's Eigen-PID wrapper (X1 = X, X2 = Y).
    pid_start = time.perf_counter()
    pid, mi = pid_calc(
        config={
            "bias_correction": True,
            "n_shuffles": config["n_shuffles"],
            "n_resamples": config["n_resamples"],
            "bias_seed": seed,
        },
        sources=[x_t, y_t],
        target=[target_t],
    )
    pid_seconds = time.perf_counter() - pid_start
    raw = pid["extra"]["raw"]
    corrected_values = {
        "red": pid["red"],
        "unq_x": pid["unq1"],
        "unq_y": pid["unq2"],
        "syn": pid["syn"],
        "mi_tx": mi["bi_mi_1"],
        "mi_ty": mi["bi_mi_2"],
        "mi_txy": mi["tri_mi"],
    }
    raw_values = {
        "red": raw["red"],
        "unq_x": raw["uix"],
        "unq_y": raw["uiy"],
        "syn": raw["syn"],
        "mi_tx": raw["mix"],
        "mi_ty": raw["miy"],
        "mi_txy": raw["mixy"],
    }

    # 6. Check finiteness and the PID consistency identities for both estimates.
    for label, values in (("corrected", corrected_values), ("raw", raw_values)):
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError(f"Non-finite {label} PID values {values} (seed={seed}).")
        residuals = {
            "R + U_X - I(T;X)": values["red"] + values["unq_x"] - values["mi_tx"],
            "R + U_Y - I(T;Y)": values["red"] + values["unq_y"] - values["mi_ty"],
            "R + U_X + U_Y + S - I(T;X,Y)": (
                values["red"] + values["unq_x"] + values["unq_y"] + values["syn"] - values["mi_txy"]
            ),
        }
        failed = {name: value for name, value in residuals.items() if abs(value) > IDENTITY_ATOL}
        if failed:
            raise RuntimeError(f"{label} PID identities fail (seed={seed}): {failed}")

    # 7. Build the result row.
    return {
        "dimension": dimension,
        "n_samples": n_samples,
        "trial": trial,
        "seed": seed,
        "noise_variance": config["noise_variance"],
        "rsa_x": rsa["rsa_x"],
        "rsa_y": rsa["rsa_y"],
        "delta_rsa": rsa["rsa_x"] - rsa["rsa_y"],
        **corrected_values,
        "delta_ui": corrected_values["unq_x"] - corrected_values["unq_y"],
        **{f"{key}_raw": value for key, value in raw_values.items()},
        "delta_ui_raw": raw_values["unq_x"] - raw_values["unq_y"],
        "cov_min_eig": cov_min_eig,
        "cov_condition": cov_condition,
        "bias_method": pid["extra"]["bias_method"],
        "n_shuffles": config["n_shuffles"],
        "n_resamples": config["n_resamples"],
        "rsa_seconds": rsa_seconds,
        "pid_seconds": pid_seconds,
    }


def summarize_trials(trials_csv: str | Path, summary_csv: str | Path) -> Path:
    """Summarize trial results per (dimension, sample size) with trial quartiles.

    Inputs:
        trials_csv: str or Path, CSV written by ``run_experiment``.
        summary_csv: str or Path, output CSV path.

    Outputs:
        Path to a long-format CSV with columns dimension, n_samples, metric,
        mean, median, sd, q25, q75, frac_positive, n_trials. ``q25`` and
        ``q75`` are the 25th and 75th percentiles of the trial values (the
        interquartile range of the estimates across trials, not a confidence
        interval of the mean). ``frac_positive`` is filled only for the
        unique-information differences.
    """
    trials = pd.read_csv(trials_csv)
    rows = []
    for (dimension, n_samples), group in trials.groupby(["dimension", "n_samples"], sort=True):
        n_trials = len(group)
        for metric in SUMMARY_METRICS:
            values = group[metric].to_numpy(dtype=np.float64)  # (n_trials,)
            q25, median, q75 = np.percentile(values, [25, 50, 75])  # (n_trials,) -> three scalars
            rows.append(
                {
                    "dimension": int(dimension),
                    "n_samples": int(n_samples),
                    "metric": metric,
                    "mean": float(values.mean()),
                    "median": float(median),
                    "sd": float(values.std(ddof=1)) if n_trials > 1 else float("nan"),
                    "q25": float(q25),
                    "q75": float(q75),
                    "frac_positive": (
                        float((values > 0).mean()) if metric in POSITIVE_FRACTION_METRICS else float("nan")
                    ),
                    "n_trials": n_trials,
                }
            )
    summary_path = Path(summary_csv)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path


def run_experiment(config: dict) -> dict[str, Path]:
    """Run every (dimension, sample size, trial), save the CSVs, and plot.

    Inputs:
        config: dict with the keys of ``CONFIG``. A relative ``output_dir`` is
            resolved from the repository root. Trial seeds are
            base_seed + condition_index * n_trials + trial, with conditions
            ordered by dimension, then sample size.

    Outputs:
        dict with Paths ``trials_csv``, ``summary_csv``, and one entry per figure.
    """
    validate_config(config)
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    trials_csv = output_dir / "rsa_vs_pid_trials.csv"
    summary_csv = output_dir / "rsa_vs_pid_summary.csv"
    if trials_csv.exists() and not config["overwrite_outputs"]:
        raise FileExistsError(f"{trials_csv} exists and overwrite_outputs is False.")
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = [(d, n) for d in config["dimensions"] for n in config["sample_sizes"]]
    n_trials = config["n_trials"]
    with trials_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        for condition_index, (dimension, n_samples) in enumerate(conditions):
            condition_rows = []
            for trial in range(n_trials):
                seed = config["base_seed"] + condition_index * n_trials + trial
                row = run_trial(dimension, n_samples, trial, seed, config)
                writer.writerow(row)
                handle.flush()
                condition_rows.append(row)
            print(
                f"[d={dimension}, n={n_samples}] {n_trials} trials: "
                f"mean delta_rsa={np.mean([r['delta_rsa'] for r in condition_rows]):+.4f}, "
                f"mean delta_ui={np.mean([r['delta_ui'] for r in condition_rows]):+.4f} bits "
                f"(raw {np.mean([r['delta_ui_raw'] for r in condition_rows]):+.4f})"
            )

    summarize_trials(trials_csv, summary_csv)
    figures = plot_all(trials_csv, summary_csv, output_dir)
    print(f"Saved trial results to: {trials_csv}")
    print(f"Saved summary to: {summary_csv}")
    for path in figures.values():
        print(f"Saved figure: {path}")
    return {"trials_csv": trials_csv, "summary_csv": summary_csv, **figures}


def parse_args() -> argparse.Namespace:
    """Parse optional command-line overrides of ``CONFIG``.

    Inputs:
        No inputs; values are read from the process command line.

    Outputs:
        argparse.Namespace with optional overrides (None when not given).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dimensions", nargs="+", type=int)
    parser.add_argument("--sample-sizes", nargs="+", type=int)
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite-outputs", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> dict:
    """Apply explicitly supplied command-line values to a copy of ``CONFIG``.

    Inputs:
        args: argparse.Namespace returned by ``parse_args``.

    Outputs:
        dict, an independent experiment configuration.
    """
    config = dict(CONFIG)
    for key in ("dimensions", "sample_sizes", "n_trials", "base_seed", "output_dir", "overwrite_outputs"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return config


def main() -> dict[str, Path]:
    """Run the configured RSA-versus-PID experiment.

    Inputs:
        No inputs. Uses ``CONFIG`` with command-line overrides.

    Outputs:
        dict with the saved CSV and figure paths.
    """
    return run_experiment(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
