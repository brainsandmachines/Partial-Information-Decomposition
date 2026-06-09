"""Seed-loop and CSV helpers for RVs_Story example batches."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PID_METHODS = ("True Values", "Tilde", "Delta", "Flow")
VALUE_KEYS = ("Red", "Unq1", "Unq2", "Syn", "I(X1;T)", "I(X2;T)", "I(X1,X2;T)")


def _as_float(value) -> float:
    """Convert scalar numeric values from tensors or Python numbers to float.

    Inputs:
        value: object, scalar torch.Tensor or numeric value.

    Outputs:
        float, converted scalar value.
    """
    return float(value.detach().cpu().numpy()) if isinstance(value, torch.Tensor) else float(value)


def csv_path(results_dir: Path, example: str, method: str) -> Path:
    """Build the per-example, per-method seed CSV path.

    Inputs:
        results_dir: Path, directory where result CSVs are stored.
        example: str, example display name.
        method: str, PID method display name.

    Outputs:
        Path, CSV file path for that example and method.
    """
    example_name = example.replace(" ", "_")
    method_name = method.replace(" ", "_")
    return results_dir / f"{example_name}_{method_name}_seeds.csv"


def csv_has_seed(path: Path, seed: int) -> bool:
    """Check whether a seed is already present in a seed CSV.

    Inputs:
        path: Path, CSV path to inspect.
        seed: int, seed value to find.

    Outputs:
        bool, True when the CSV contains that seed.
    """
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        return any(int(row["seed"]) == seed for row in csv.DictReader(handle))


def seed_is_done(seed: int, example_names: list[str], results_dir: Path, methods=PID_METHODS) -> bool:
    """Check whether all expected example/method CSV rows exist for a seed.

    Inputs:
        seed: int, seed value to inspect.
        example_names: list[str], example names that should be complete.
        results_dir: Path, directory where result CSVs are stored.
        methods: iterable[str], method names expected for each example.

    Outputs:
        bool, True when every expected row exists.
    """
    return all(
        csv_has_seed(csv_path(results_dir, example, method), seed)
        for example in example_names
        for method in methods
    )


def loop_examples(
    config: dict,
    functions_to_run: list[Callable],
    example_names: list[str],
    main_func: Callable,
    save_image: bool = True,
) -> dict:
    """Run a list of RV examples once with one config.

    Inputs:
        config: dict, simulation and PID configuration values.
        functions_to_run: list[Callable], RV generators to execute.
        example_names: list[str], output names aligned with functions_to_run.
        main_func: Callable, runner accepting (config, generator).
        save_image: bool, whether to save one image per example.

    Outputs:
        dict, result dictionaries keyed by example name.
    """
    from Partial_Information_Decomposition.PID_util import save_pid_comparison_table

    all_results = {}
    for func, name in zip(functions_to_run, example_names):
        print(f"\nRunning example {func.__name__}...")
        results = main_func(config, func)
        all_results[name] = results
        if save_image:
            save_pid_comparison_table(results, f"{config['results_dir']}/{name}.png", config=config)
        print(f"Finished example {func.__name__}.")
    return all_results


def save_seed_csvs(seed: int, all_results: dict, results_dir: Path) -> None:
    """Save one seed of PID results into method-specific CSV files.

    Inputs:
        seed: int, seed used to generate all_results.
        all_results: dict, nested result dictionaries keyed by example name.
        results_dir: Path, directory where CSVs should be written.

    Outputs:
        None.
    """
    from Partial_Information_Decomposition.PID_util import pid_comparison_table

    for example, results in all_results.items():
        for row in pid_comparison_table(results, print_table=False):
            method = row.pop("method")
            path = csv_path(results_dir, example, method)
            fieldnames = ["seed", *row.keys()]
            old_rows = []
            if path.exists():
                with path.open(newline="", encoding="utf-8") as handle:
                    old_rows = [old for old in csv.DictReader(handle) if int(old["seed"]) != seed]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(old_rows)
                writer.writerow({"seed": seed, **{key: _as_float(value) for key, value in row.items()}})


def mean_results_from_csvs(results_dir: Path, example: str, seeds: list[int]) -> dict:
    """Average saved seed CSVs back into a PID result dictionary.

    Inputs:
        results_dir: Path, directory containing seed CSVs.
        example: str, example name to load.
        seeds: list[int], seeds to include in the average.

    Outputs:
        dict, averaged PID results keyed by method name.
    """
    seed_set = {int(seed) for seed in seeds}
    results = {}
    for method in PID_METHODS:
        path = csv_path(results_dir, example, method)
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if int(row["seed"]) in seed_set]
        if not rows:
            continue
        mean = {key: np.mean([float(row[key]) for row in rows]) for key in VALUE_KEYS}
        pid = {"red": mean["Red"], "unq1": mean["Unq1"], "unq2": mean["Unq2"], "syn": mean["Syn"]}
        mi = {"bi_mi_1": mean["I(X1;T)"], "bi_mi_2": mean["I(X2;T)"], "tri_mi": mean["I(X1,X2;T)"]}
        results[method] = (pid, mi)
    return results


def loop_examples_over_seeds(
    config: dict,
    functions_to_run: list[Callable],
    example_names: list[str],
    main_func: Callable,
    num_seeds: int | None = None,
    seeds: list[int] | None = None,
) -> dict:
    """Run examples over seeds, save seed CSVs, then save averaged figures.

    Inputs:
        config: dict, simulation and PID configuration values.
        functions_to_run: list[Callable], RV generators to execute.
        example_names: list[str], output names aligned with functions_to_run.
        main_func: Callable, runner accepting (config, generator).
        num_seeds: int | None, number of sequential seeds to run.
        seeds: list[int] | None, explicit seed list overriding num_seeds.

    Outputs:
        dict, averaged result dictionaries keyed by example name.
    """
    from Partial_Information_Decomposition.PID_util import save_pid_comparison_table

    base_seed = config.get("seed", 0)
    seed_values = list(seeds) if seeds is not None else list(range(base_seed, base_seed + (num_seeds or 1)))
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    for seed in seed_values:
        if seed_is_done(seed, example_names, results_dir):
            print(f"\nSeed {seed} has already been done. Skipping to the next seed.")
            continue
        print(f"\n{'=' * 70}\nRunning all examples with seed={seed}\n{'=' * 70}")
        all_results = loop_examples({**config, "seed": seed}, functions_to_run, example_names, main_func, save_image=False)
        save_seed_csvs(seed, all_results, results_dir)

    mean_config = {**config, "seed": f"{seed_values[0]}-{seed_values[-1]}"}
    mean_by_example = {}
    for name in example_names:
        mean_results = mean_results_from_csvs(results_dir, name, seed_values)
        if not mean_results:
            continue
        mean_by_example[name] = mean_results
        save_pid_comparison_table(
            mean_results,
            f"{results_dir}/{name}_mean_over_{len(seed_values)}_seeds.png",
            title=f"PID Method Comparison - Mean Over {len(seed_values)} Seeds",
            config=mean_config,
        )
    return mean_by_example
