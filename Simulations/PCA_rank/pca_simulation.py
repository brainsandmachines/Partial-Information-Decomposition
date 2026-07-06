from __future__ import annotations

from itertools import product
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from library_wrappers.missmda_ncp import estimate_ncp_pca

METHOD, METHOD_CV, NBSIM = "EM", "Kfold", 5
OUTPUT_DIR = ROOT / "results/pca_cv_simulations"

FULL_GRID = dict(
    n_samples=[50, 100, 300], n_features=[50, 100], rank=[1, 3, 5],
    loading_corr=[0.0, 0.3, 0.6, 0.9], noise_std=[0.01, 0.1, 0.2],
    seed=list(range(20)),
)
GRID = dict(
    n_samples=[100], n_features=[55,70], rank=[1, 3,20,50],
    loading_corr=[0.0,0.3, 0.9], noise_std=[0.01, 0.1], seed=list(range(3)),
)


def create_T_and_P(
    n_samples: int,
    n_features: int,
    rank: int,
    loading_corr: float = 0.0,
    component_strengths: np.ndarray | list[float] | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create scores and correlated loadings. Inputs define sizes, rank, correlation, strengths, and seed; outputs are T and P arrays."""
    rng = np.random.default_rng(random_state)
    T = rng.normal(size=(n_samples, rank))
    T = (T - T.mean(0, keepdims=True)) / T.std(0, keepdims=True)
    if component_strengths is not None:
        T *= np.asarray(component_strengths)[None, :]
    A = rng.normal(size=(n_features, rank))
    Q, _ = np.linalg.qr(A - A.mean(0, keepdims=True))
    C = np.full((rank, rank), loading_corr)
    np.fill_diagonal(C, 1.0)
    P = Q[:, :rank] @ np.linalg.cholesky(C).T
    P -= P.mean(0, keepdims=True)
    P /= np.linalg.norm(P, axis=0, keepdims=True)
    return T, P


def generate_rank_simulation_data(
    n_samples: int,
    n_features: int,
    rank: int,
    loading_corr: float,
    noise_std: float,
    random_state: int,
    component_strengths: list[float] | np.ndarray | None = None,
    center_columns: bool = True,
) -> dict:
    """Generate noisy known-rank data. Inputs define the condition; output contains X, T, P, normalized signal, and metadata."""
    seeds = np.random.SeedSequence(random_state).generate_state(2)
    T, P = create_T_and_P(
        n_samples, n_features, rank, loading_corr, component_strengths, int(seeds[0])
    )
    signal = T @ P.T
    signal /= signal.std()
    X = signal + noise_std * np.random.default_rng(int(seeds[1])).normal(size=signal.shape)
    if center_columns:
        X -= X.mean(0, keepdims=True)
    return dict(
        X=X, T=T, P=P, X_signal=signal, true_rank=rank,
        loading_corr=loading_corr, noise_std=noise_std, random_state=random_state,
    )


def run_rank_simulation(
    grid: dict[str, list],
    output_dir: str | Path = OUTPUT_DIR,
    nbsim: int = NBSIM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run EM/K-fold rank selection. Inputs are grid, output path, and K-fold count; outputs are raw/summary tables and saved CSV/heatmap files."""
    names = ("n_samples", "n_features", "true_rank", "loading_corr", "noise_std", "seed")
    values = (grid["n_samples"], grid["n_features"], grid["rank"],
              grid["loading_corr"], grid["noise_std"], grid["seed"])
    rows = []
    n_rows = np.prod([len(v) for v in values])
    finished = 0
    start = perf_counter()
    for condition in product(*values):
        row, start = dict(zip(names, condition)), perf_counter()
        n, p, rank, corr, noise, seed = condition
        try:
            print(f"\nRunning condition: n={n}, p={p}, rank={rank}, corr={corr}, noise={noise}, seed={seed}")
            X = generate_rank_simulation_data(n, p, rank, corr, noise, seed)["X"]
            result = estimate_ncp_pca(
                X, ncp_max=min(rank + 5, min(n, p) - 1), method=METHOD,
                method_cv=METHOD_CV, nbsim=nbsim, seed=seed,
            )
            estimated, status, message = result["ncp"], "ok", ""
        except Exception as error:
            estimated, status, message = np.nan, "failed", str(error)
        rank_error = estimated - rank
        row.update(
            estimated_rank=estimated, rank_error=rank_error,
            is_success=estimated == rank, is_underfit=estimated < rank,
            is_overfit=estimated > rank, method=METHOD, method_cv=METHOD_CV,
            runtime_seconds=perf_counter() - start, wrapper_status=status,
            error_message=message,
        )
        rows.append(row)
        finished +=1
        print(f"Finished condition number: {finished}/{n_rows} in {perf_counter() - start:.2f} seconds")

    raw = pd.DataFrame(rows)
    groups = ["method", "method_cv", "n_samples", "n_features", "true_rank",
              "loading_corr", "noise_std"]
    summary = raw.groupby(groups, as_index=False).agg(
        n_runs=("seed", "size"), n_success=("is_success", "sum"),
        success_rate=("is_success", "mean"), underfit_rate=("is_underfit", "mean"),
        overfit_rate=("is_overfit", "mean"), mean_estimated_rank=("estimated_rank", "mean"),
        std_estimated_rank=("estimated_rank", "std"),
        median_estimated_rank=("estimated_rank", "median"),
        mean_abs_rank_error=("rank_error", lambda x: x.abs().mean()),
        mean_runtime_seconds=("runtime_seconds", "mean"),
        n_failed_runs=("wrapper_status", lambda x: (x != "ok").sum()),
    ).sort_values(["true_rank", "n_samples", "n_features", "loading_corr", "noise_std"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "estim_ncpPCA_EM_raw.csv", index=False)
    summary.to_csv(output_dir / "estim_ncpPCA_EM_summary.csv", index=False)

    panels = list(summary.groupby(["n_samples", "n_features", "true_rank"]))
    figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    axes = axes.ravel()
    for axis, (condition, panel) in zip(axes, panels):
        heat = panel.pivot(index="loading_corr", columns="noise_std", values="success_rate")
        image = axis.imshow(heat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set(xticks=range(len(heat.columns)), yticks=range(len(heat.index)),
                 xlabel="noise_std", ylabel="loading_corr",
                 title=f"n={condition[0]}, p={condition[1]}, rank={condition[2]}")
        axis.set_xticklabels(heat.columns)
        axis.set_yticklabels(heat.index)
        for y, x in product(range(heat.shape[0]), range(heat.shape[1])):
            axis.text(x, y, f"{heat.iloc[y, x]:.2f}", ha="center", va="center", color="white")
    figure.colorbar(image, ax=axes.tolist(), label="success_rate")
    figure.savefig(output_dir / "estim_ncpPCA_EM_success_heatmap.png",
                   dpi=200, bbox_inches="tight")
    plt.close(figure)
    return raw, summary


if __name__ == "__main__":
    smoke = generate_rank_simulation_data(50, 20, 3, 0.0, 0.01, 0)
    check = estimate_ncp_pca(
        smoke["X"], ncp_max=8, method=METHOD, method_cv=METHOD_CV,
        nbsim=NBSIM, seed=0,
    )
    global_min = min(check["criterion"], key=check["criterion"].get)
    assert check["ncp"] == global_min
    print("Smoke:", check["ncp"], check["criterion"], METHOD, METHOD_CV)
    raw_results, summary_results = run_rank_simulation(GRID)
    print(raw_results.head().to_string(index=False))
    print(summary_results.to_string(index=False))
    print()
