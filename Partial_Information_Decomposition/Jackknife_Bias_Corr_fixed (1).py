import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from Toy_Simulations.Bias_Corr_simulations import theoretical_covariance, sample_cov_simulation
from parallel_Idep_multivariate_gauss import para_Idep_multivariate_gauss


def get_run_config() -> dict:
    return {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_seeds': 1000,
        'seed_start': 0,
        'n': 1000,
        'N_values': [500,800, 1000],
        'p_values': [[5, 5, 10], [50, 50, 55], [100, 50, 105]],  # Dimensions for X1, X2, X3
        'p': 0,
        'q': 0,
        'r': 0,
        'results_dir': '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/para_jackknife_pid/with_groundtruth',
        'results_prefix': 'seed_summary',
        'all_runs_results_prefix': 'seed_runs',
        'progress_print_every': 100,
        'test_name': 'heatmap_jackknife_pid',
    }


def plot_all_mi_heatmaps(
    csv_path,
    title="Mutual Information Heatmaps",
    *,
    n_col="N",
    dims_col="dims",
    figsize=(16, 5),
    save_path=None,
    annotate=True,
    mean_fmt=".2f",
    std_fmt=".2f",
    log_scale=False,
    cmap="viridis",
    annotation_mode="pm",
    fontsize=9,
    aggfunc="mean",
):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    required_cols = [
        n_col, dims_col,
        "mi_theoretical_mean", "mi_theoretical_std",
        "mi_sample_no_bias_mean", "mi_sample_no_bias_std",
        "mi_sample_with_bias_mean", "mi_sample_with_bias_std",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df[n_col] = pd.to_numeric(df[n_col], errors="coerce")
    df[dims_col] = df[dims_col].astype(str)

    mean_std_pairs = [
        ("mi_theoretical_mean", "mi_theoretical_std", "Theoretical MI"),
        ("mi_sample_no_bias_mean", "mi_sample_no_bias_std", "Naive MI"),
        ("mi_sample_with_bias_mean", "mi_sample_with_bias_std", "Bias-corrected MI"),
    ]

    dup_mask = df.duplicated(subset=[n_col, dims_col], keep=False)
    if dup_mask.any():
        print("Warning: duplicate (N, dims) pairs found. Aggregating with pivot_table.")
        print(df.loc[dup_mask, [n_col, dims_col]].value_counts().sort_index())

    mean_columns = [pair[0] for pair in mean_std_pairs]
    all_mean_values = df[mean_columns].to_numpy(dtype=float)

    if log_scale:
        positive_vals = all_mean_values[all_mean_values > 0]
        if positive_vals.size == 0:
            raise ValueError("No positive mean values found. Cannot use log scale.")
        norm = LogNorm(vmin=positive_vals.min(), vmax=positive_vals.max())
    else:
        norm = plt.Normalize(vmin=np.nanmin(all_mean_values), vmax=np.nanmax(all_mean_values))

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    fig.suptitle(title, fontsize=14)
    last_im = None

    for ax, (mean_col, std_col, panel_title) in zip(axes, mean_std_pairs):
        mean_df = df.pivot_table(index=dims_col, columns=n_col, values=mean_col, aggfunc=aggfunc)
        std_df = df.pivot_table(index=dims_col, columns=n_col, values=std_col, aggfunc=aggfunc)

        mean_df = mean_df.sort_index().sort_index(axis=1)
        std_df = std_df.reindex(index=mean_df.index, columns=mean_df.columns)

        mean_values = mean_df.to_numpy(dtype=float)
        std_values = std_df.to_numpy(dtype=float)
        plot_values = mean_values.copy()
        if log_scale:
            plot_values[plot_values <= 0] = np.nan

        im = ax.imshow(plot_values, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        last_im = im

        ax.set_xticks(np.arange(len(mean_df.columns)))
        ax.set_xticklabels(mean_df.columns.astype(int))

        ax.set_yticks(np.arange(len(mean_df.index)))
        ax.set_yticklabels([str(idx) for idx in mean_df.index])

        ax.set_xlabel("N")
        ax.set_ylabel("dims")
        ax.set_title(panel_title)

        if annotate:
            for i in range(mean_values.shape[0]):
                for j in range(mean_values.shape[1]):
                    mean_val = mean_values[i, j]
                    std_val = std_values[i, j]

                    if np.isnan(mean_val):
                        text = "nan"
                    else:
                        mean_text = format(mean_val, mean_fmt)
                        std_text = "nan" if np.isnan(std_val) else format(std_val, std_fmt)
                        if annotation_mode == "pm":
                            text = f"{mean_text}\n±{std_text}"
                        elif annotation_mode == "paren":
                            text = f"{mean_text}\n({std_text})"
                        else:
                            raise ValueError("annotation_mode must be 'pm' or 'paren'")

                    ax.text(j, i, text, ha="center", va="center", color="black", fontsize=fontsize)

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.85)
    cbar.set_label("Mutual Information (mean)")

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, f"{title}.png")
        plt.savefig(full_path, dpi=300, bbox_inches="tight")

    plt.show()


def lo_cov(rvs, N):
    """
    Compute the full covariance matrix across samples and all leave-one-out covariance matrices.
    """
    assert isinstance(rvs, list), 'Input Z should be a list of torch tensors'
    d = sum(rv.shape[1] for rv in rvs)
    Z = torch.hstack(rvs).to(torch.float64)
    cov = torch.cov(Z.T, correction=1)
    S = Z.T @ Z
    s = torch.sum(Z, axis=0)
    s_outer = torch.outer(s, s)
    Sigma_full = (S - (1 / N) * s_outer) / (N - 1)
    assert torch.allclose(Sigma_full, cov, atol=1e-10, rtol=1e-8), (
        'The covariance matrix computed using the formula does not match torch.cov'
    )

    outer_all = Z[:, :, None] * Z[:, None, :]
    S_minus_all = S.unsqueeze(0) - outer_all
    s_minus_all = s.unsqueeze(0) - Z
    s_outer_all = s_minus_all[:, :, None] * s_minus_all[:, None, :]
    cov_loo_all = (S_minus_all - s_outer_all / (N - 1)) / (N - 2)

    assert outer_all.shape == (N, d, d)
    assert S_minus_all.shape == (N, d, d)
    assert s_minus_all.shape == (N, d)
    assert s_outer_all.shape == (N, d, d)
    assert cov_loo_all.shape == (N, d, d)
    return Sigma_full, cov_loo_all


def infer_total_mi_key(mi_keys):
    """
    Infer the total/joint mutual information key from the MI dictionary returned by idep().
    """
    keys = list(mi_keys)
    if not keys:
        raise ValueError('No MI keys found.')

    def score(key: str):
        s = key.lower().replace(' ', '')
        val = 0
        if 'm1,m2' in s or 'm1m2' in s:
            val += 5
        if 'x0,x1' in s or 'x0x1' in s:
            val += 5
        if ';t' in s or 't)' in s or ',t' in s or ';x2' in s or 'x2)' in s:
            val += 3
        if 'total' in s or 'joint' in s or 'all' in s:
            val += 2
        if 'm1;t' in s or 'm2;t' in s or 'x0;x2' in s or 'x1;x2' in s:
            val -= 2
        return val, len(key)

    best = max(keys, key=score)
    return best


@torch.no_grad()
def compute_true_total_mi(config: dict):
    device = config['device']
    corr_matrix = np.array([
        [1.0, config['p'], config['q']],
        [config['p'], 1.0, config['r']],
        [config['q'], config['r'], 1.0],
    ])
    true_cov = theoretical_covariance(config['dims'], corr_matrix)
    config['true_cov'] = true_cov

    torch_true_cov = torch.from_numpy(true_cov).to(device=device, dtype=torch.float64)
    _, mi_true = para_Idep_multivariate_gauss(
        N=1,
        df=config['n'] - 1,
        device=device,
        cov_matrix=torch_true_cov.unsqueeze(0),
        dims=config['dims'],
        bias_correction=False,
    ).idep()

    total_key = infer_total_mi_key(mi_true.keys())
    true_total_mi = mi_true[total_key].item()
    return total_key, true_total_mi, true_cov


@torch.no_grad()
def compute_mi_raw_and_jackknife(device, sources, target, N):
    assert len(sources) == 2, 'Expected exactly two source variables'
    assert len(target) == 1, 'Expected exactly one target variable'

    rvs = sources + target
    Sigma_full, cov_loo_all = lo_cov(rvs, N)
    Sigma_raw = Sigma_full.unsqueeze(0)
    dims = [source.shape[1] for source in sources] + [target[0].shape[1]]

    idep_raw = para_Idep_multivariate_gauss(
        N=1,
        df=N - 1,
        device=device,
        cov_matrix=Sigma_raw,
        dims=dims,
        bias_correction=False,
    )
    pid_raw, mi_raw = idep_raw.idep()

    idep_loo = para_Idep_multivariate_gauss(
        N=N,
        df=N - 1,
        device=device,
        cov_matrix=cov_loo_all,
        dims=dims,
        bias_correction=False,
    )
    pid_loo, mi_loo = idep_loo.idep()

    mean_pid_loo = {key: torch.mean(pid_loo[key]) for key in pid_loo.keys()}
    mean_mi_loo = {key: torch.mean(mi_loo[key]) for key in mi_loo.keys()}

    pid_bias_term = {key: (N - 1) * (mean_pid_loo[key] - pid_raw[key]) for key in pid_raw.keys()}
    mi_bias_term = {key: (N - 1) * (mean_mi_loo[key] - mi_raw[key]) for key in mi_raw.keys()}

    pid_bc = {key: (pid_raw[key] - pid_bias_term[key]).item() for key in pid_raw.keys()}
    mi_bc = {key: (mi_raw[key] - mi_bias_term[key]).item() for key in mi_raw.keys()}
    mi_raw_out = {key: mi_raw[key].item() for key in mi_raw.keys()}
    return pid_bc, mi_bc, mi_raw_out


def run_single_seed(seed: int, config: dict, total_mi_key: str, true_total_mi: float) -> dict:
    device = config['device']
    rv_list, _ = sample_cov_simulation(seed, config['n'], config['dims'], config['true_cov'])
    torch_rv_list = [torch.from_numpy(rv).to(device=device, dtype=torch.float64) for rv in rv_list]

    _, mi_bc, mi_raw = compute_mi_raw_and_jackknife(
        device=device,
        sources=torch_rv_list[:2],
        target=torch_rv_list[2:],
        N=config['n'],
    )

    return {
        'seed': seed,
        'mi_theoretical': true_total_mi,
        'mi_sample_no_bias': mi_raw[total_mi_key],
        'mi_sample_with_bias': mi_bc[total_mi_key],
    }


def summarize_seed_rows(seed_rows):
    df = pd.DataFrame(seed_rows)
    summary = {
        'mi_theoretical_mean': df['mi_theoretical'].mean(),
        'mi_theoretical_std': df['mi_theoretical'].std(ddof=1),
        'mi_sample_no_bias_mean': df['mi_sample_no_bias'].mean(),
        'mi_sample_no_bias_std': df['mi_sample_no_bias'].std(ddof=1),
        'mi_sample_with_bias_mean': df['mi_sample_with_bias'].mean(),
        'mi_sample_with_bias_std': df['mi_sample_with_bias'].std(ddof=1),
    }
    return summary, df


def run_simulation(config: dict):
    total_mi_key, true_total_mi, true_cov = compute_true_total_mi(config)
    config['true_cov'] = true_cov

    seeds = range(config['seed_start'], config['seed_start'] + config['n_seeds'])
    seed_rows = []
    for idx, seed in enumerate(seeds, start=1):
        row = run_single_seed(seed, config, total_mi_key, true_total_mi)
        seed_rows.append(row)
        if idx % config['progress_print_every'] == 0:
            print(f"Completed seed {idx}/{config['n_seeds']} for N={config['n']}, dims={config['dims']}")

    summary, seed_df = summarize_seed_rows(seed_rows)
    return summary, seed_df, total_mi_key


def save_results(config: dict, all_summary_rows, all_seed_rows):
    os.makedirs(config['results_dir'], exist_ok=True)

    summary_df = pd.DataFrame(all_summary_rows)
    seed_df = pd.DataFrame(all_seed_rows)

    summary_csv_path = os.path.join(
        config['results_dir'],
        f"{config['test_name']}_heatmap_summary.csv",
    )
    seed_csv_path = os.path.join(
        config['results_dir'],
        f"{config['test_name']}_all_seed_runs.csv",
    )

    summary_df.to_csv(summary_csv_path, index=False)
    seed_df.to_csv(seed_csv_path, index=False)

    print(f"Saved heatmap summary CSV to: {summary_csv_path}")
    print(f"Saved all seed-level runs CSV to: {seed_csv_path}")
    return summary_csv_path, seed_csv_path


def N_P_variation_simulation(config: dict):
    all_summary_rows = []
    all_seed_rows = []

    N_values = config['N_values']
    p_values = config['p_values']
    total_jobs = len(N_values) * len(p_values)
    job_idx = 1

    for N in N_values:
        for dims in p_values:
            local_config = dict(config)
            local_config['n'] = N
            local_config['dims'] = list(dims)

            print(f"\nRunning simulation for N={N}, dims={dims} ({job_idx}/{total_jobs})")
            summary, seed_df, total_mi_key = run_simulation(local_config)

            p_sources = dims[0] + dims[1]
            summary_row = {
                'N': N,
                'p': p_sources,
                'dims': str(list(dims)),
                'total_mi_key': total_mi_key,
                **summary,
            }
            all_summary_rows.append(summary_row)

            seed_df = seed_df.copy()
            seed_df['N'] = N
            seed_df['p'] = p_sources
            seed_df['dims'] = str(list(dims))
            seed_df['total_mi_key'] = total_mi_key
            all_seed_rows.append(seed_df)

            print(f"Completed combination N={N}, dims={dims} ({job_idx}/{total_jobs})")
            job_idx += 1

    all_seed_df = pd.concat(all_seed_rows, ignore_index=True) if all_seed_rows else pd.DataFrame()
    return all_summary_rows, all_seed_df


def main():
    config = get_run_config()
    os.makedirs(config['results_dir'], exist_ok=True)

    all_summary_rows, all_seed_df = N_P_variation_simulation(config=config)
    summary_csv_path, seed_csv_path = save_results(config, all_summary_rows, all_seed_df)

    plot_all_mi_heatmaps(
        csv_path=summary_csv_path,
        save_path=config['results_dir'],
        title=config['test_name'],
        log_scale=False,
    )
    plot_all_mi_heatmaps(
        csv_path=summary_csv_path,
        save_path=config['results_dir'],
        title=f"Logscaled_{config['test_name']}",
        log_scale=True,
    )

    return summary_csv_path, seed_csv_path


if __name__ == '__main__':
    main()
