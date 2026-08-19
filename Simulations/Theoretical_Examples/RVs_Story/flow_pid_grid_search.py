import argparse
import csv
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

ROOT = Path(__file__).resolve().parents[3]
FLOW_PID_ROOT = ROOT / "external" / "flow-pid"
for path in (ROOT, FLOW_PID_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Partial_Information_Decomposition.PID_calc import flow_pid_wrapper
from Simulations.Theoretical_Examples.RVs_Story.regular_examples.equal_unique import (
    equal_unique,
)
from Simulations.Theoretical_Examples.RVs_Story.story_pid_utils import (
    truth_pid_equal_unique,
    truth_pid_suppression,
)
from Simulations.Theoretical_Examples.RVs_Story.suppresion_examples.full_suppresion import (
    full_suppresion,
)
from Simulations.Theoretical_Examples.RVs_Story.suppresion_examples.unq12_zero import (
    unq12_zero,
)
from Simulations.Theoretical_Examples.RVs_Story.suppresion_examples.unq2_zero import (
    unq2_zero,
)


FLOW_GRID = {
    "n_flows": [1, 2, 3, 4, 5, 10],
    "n_epochs": [20,50,100,200,500,1000],
    "batch_size": [16,32,64,128],
    "lr": [2e-4,1e-3,5e-3,0.01,2e-2],
}

EXAMPLES = {
    "equal_unique": {
        "generator": equal_unique,
        "truth": truth_pid_equal_unique,
    },
    "full_suppresion": {
        "generator": full_suppresion,
        "truth": truth_pid_suppression,
    },
    "unq2_zero": {
        "generator": unq2_zero,
        "truth": truth_pid_suppression,
    },
    "unq12_zero": {
        "generator": unq12_zero,
        "truth": truth_pid_suppression,
    },
}


def load_example_and_truth(example_name):
    """Return the registered sample generator and analytical truth function.

    Inputs:
        example_name: str key in ``EXAMPLES``.

    Outputs:
        tuple of generator and truth callables used by the Flow-PID grid search.
    """

    spec = EXAMPLES[example_name]
    return spec["generator"], spec["truth"]


def grid_items(grid):
    keys = list(grid)
    for values in product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def standardize_train_val(train, val):
    train = np.asarray(train, dtype=np.float32)
    val = np.asarray(val, dtype=np.float32)
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (train - mean) / std, (val - mean) / std


def make_folds(n, k_folds, seed):
    indices = np.random.default_rng(seed).permutation(n)
    return [fold for fold in np.array_split(indices, k_folds) if len(fold)]





def true_synergy(truth_func, x1, x2, t):
    sources = [torch.from_numpy(x1), torch.from_numpy(x2)]
    target = [torch.from_numpy(t)]
    pid, _ = truth_func(sources, target)
    return float(pid["syn"])


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_grid_search(config, example_name, k_folds, grid, results_dir, device):
    generator, truth_func = load_example_and_truth(example_name)
    rng = np.random.default_rng(config["seed"])
    x1, x2, t = generator(rng, config["n"], config["p"], config["noise_std"])
    folds = make_folds(config["n"], k_folds, config["seed"])
    min_fold_size = min(len(fold) for fold in folds)
    if min_fold_size <= 3 * config["p"]:
        raise ValueError(
            f"Each validation fold needs more than 3*p samples for the ground-truth calculation; "
            f"got min fold size {min_fold_size} and p={config['p']}."
        )
    all_idx = np.arange(config["n"])
    fold_rows = []

    for grid_id, params in enumerate(grid_items(grid)):
        for fold_id, val_idx in enumerate(folds):
            train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=True)
            m_train, m_val = standardize_train_val(t[train_idx], t[val_idx])
            x_train, x_val = standardize_train_val(x1[train_idx], x1[val_idx])
            y_train, y_val = standardize_train_val(x2[train_idx], x2[val_idx])
            # Use flow_pid_wrapper so PID calculation uses the same flow implementation
            wrapper_config = {
                "dt": t.shape[1],
                "dx1": x1.shape[1],
                "dx2": x2.shape[1],
                "n_flows": params["n_flows"],
                "n_epochs": params["n_epochs"],
                "batch_size": params["batch_size"],
                "lr": params["lr"],
                "verbose": False,
                "device": device,
                "bias_correction": config.get("bias_correction", False),
            }

            # flow_pid_wrapper accepts raw samples: sources=[x1, x2], target=[t]
            pid, mi = flow_pid_wrapper(
                wrapper_config,
                sources=[x1[val_idx], x2[val_idx]],
                target=[t[val_idx]],
                covariance=None,
                rng=None,
                on_rvs=None,
            )
            # flow_pid_wrapper does not return a validation loss; keep as NaN
            val_loss = float("nan")
            truth_syn = true_synergy(truth_func, x1[val_idx], x2[val_idx], t[val_idx])
            fold_rows.append({
                "example": example_name,
                "grid_id": grid_id,
                "fold": fold_id,
                **params,
                "flow_syn": float(pid["syn"]),
                "true_syn": truth_syn,
                "syn_abs_error": abs(float(pid["syn"]) - truth_syn),
                "val_loss": val_loss,
            })
            print(f"grid={grid_id} fold={fold_id} syn_abs_error={fold_rows[-1]['syn_abs_error']:.6f}")

    summary_rows = []
    for grid_id, params in enumerate(grid_items(grid)):
        rows = [row for row in fold_rows if row["grid_id"] == grid_id]
        summary_rows.append({
            "example": example_name,
            "grid_id": grid_id,
            **params,
            "mean_syn_abs_error": float(np.mean([row["syn_abs_error"] for row in rows])),
            "std_syn_abs_error": float(np.std([row["syn_abs_error"] for row in rows])),
            "mean_val_loss": float(np.mean([row["val_loss"] for row in rows])),
        })
    best = min(summary_rows, key=lambda row: row["mean_syn_abs_error"])
    best_params = {key: best[key] for key in grid}

    m_all, _ = standardize_train_val(t, t)
    x_all, _ = standardize_train_val(x1, x1)
    y_all, _ = standardize_train_val(x2, x2)
    # Compute final PID on all data using the selected best hyperparameters
    wrapper_config_all = {
        "dt": t.shape[1],
        "dx1": x1.shape[1],
        "dx2": x2.shape[1],
        "n_flows": best_params["n_flows"],
        "n_epochs": best_params["n_epochs"],
        "batch_size": best_params["batch_size"],
        "lr": best_params["lr"],
        "verbose": False,
        "device": device,
        "bias_correction": config.get("bias_correction", False),
    }
    pid, mi = flow_pid_wrapper(
        wrapper_config_all,
        sources=[x1, x2],
        target=[t],
        covariance=None,
        rng=None,
        on_rvs=None,
    )
    final_loss = float("nan")
    truth_syn = true_synergy(truth_func, x1, x2, t)
    best_row = {
        "example": example_name,
        **best_params,
        "red": float(pid["red"]),
        "unq1": float(pid["unq1"]),
        "unq2": float(pid["unq2"]),
        "syn": float(pid["syn"]),
        "true_syn": truth_syn,
        "syn_abs_error": abs(float(pid["syn"]) - truth_syn),
        "tri_mi": float(mi["tri_mi"]),
        "bi_mi_1": float(mi["bi_mi_1"]),
        "bi_mi_2": float(mi["bi_mi_2"]),
        "loss": final_loss,
    }

    fold_fields = ["example", "grid_id", "fold", *grid, "flow_syn", "true_syn", "syn_abs_error", "val_loss"]
    summary_fields = ["example", "grid_id", *grid, "mean_syn_abs_error", "std_syn_abs_error", "mean_val_loss"]
    best_fields = ["example", *grid, "red", "unq1", "unq2", "syn", "true_syn", "syn_abs_error", "tri_mi", "bi_mi_1", "bi_mi_2", "loss"]
    write_csv(results_dir / f"flow_pid_grid_{example_name}_folds.csv", fold_rows, fold_fields)
    write_csv(results_dir / f"flow_pid_grid_{example_name}_summary.csv", summary_rows, summary_fields)
    write_csv(results_dir / f"flow_pid_grid_{example_name}_best.csv", [best_row], best_fields)
    return best


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search Flow-PID hyperparameters by k-fold synergy error.")
    parser.add_argument("--example", choices=sorted(EXAMPLES), default="equal_unique")
    parser.add_argument("--config", type=Path, default=ROOT / "Simulations" / "Theoretical_Examples" / "rv_config.yaml")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n", type=int)
    parser.add_argument("--p", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--noise-std", type=float)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)["parameters"]
    for key in ("n", "p", "seed"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.noise_std is not None:
        config["noise_std"] = args.noise_std
    grid = FLOW_GRID
    k_folds = args.k_folds
    if args.quick:
        grid = {"n_flows": [1,2,3,4,5,10], "n_epochs": [20,100,500], "batch_size": [16,32,64], "lr": [2e-4,1e-3,5e-3,0.01,2e-2]}
        config["n"] = args.n or 120
        config["p"] = args.p or 2
        k_folds = min(k_folds, 2)
    results_dir = args.results_dir or Path(config["results_dir"])
    device = args.device or config.get("device", "cpu")
    best = run_grid_search(config, args.example, k_folds, grid, results_dir, device)
    print(f"\nBest hyperparameters for {args.example}: {best}")


if __name__ == "__main__":
    main()
