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

from encoding_model.commonality import commonality_analysis
from Partial_Information_Decomposition.PID_util import (
    pid_comparison_table,
    save_commonality_comparison_table,
    save_pid_comparison_table,
)
try:
    from .story_pid_utils import pid_method_display_name
except ImportError:  # pragma: no cover - direct script compatibility
    from story_pid_utils import pid_method_display_name

PID_METHODS = (
    "True Values",
    "Tilde",
    "Delta",
    "Analytical BROJA",
    "Flow",
)
VALUE_KEYS = ("Red", "Unq1", "Unq2", "Syn", "I(X1;T)", "I(X2;T)", "I(X1,X2;T)")
COMMONALITY_KEY = "Commonality_Analysis"
COMMONALITY_FIELDS = (
    ("R²_X1", "R²_X1"),
    ("R²_X2", "R²_X2"),
    ("R²_X12", "R²_X12"),
    ("Unique_X1", "unique_X1"),
    ("Unique_X2", "unique_X2"),
    ("Common", "common"),
    ("Unexplained", "unexplained"),
)


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


def commonality_csv_path(results_dir: Path, example: str = COMMONALITY_KEY) -> Path:
    """Build the shared per-seed commonality CSV path.

    Inputs:
        results_dir: Path, directory where result CSVs are stored.
        example: str, commonality result-group name used in the filename.

    Outputs:
        Path, CSV file path containing commonality rows for all examples.
    """
    example_name = example.replace(" ", "_")
    return results_dir / f"{example_name}_commonality_seeds.csv"


def csv_has_seed(path: Path, seed: int, example: str | None = None) -> bool:
    """Check whether a seed, optionally for one example, is in a CSV.

    Inputs:
        path: Path, CSV path to inspect.
        seed: int, seed value to find.
        example: str | None, optional example value required in the same row.

    Outputs:
        bool, True when the CSV contains the requested row.
    """
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        return any(
            int(row["seed"]) == seed
            and (example is None or row.get("example") == example)
            for row in csv.DictReader(handle)
        )


def seed_is_done(
    seed: int,
    example_names: list[str],
    results_dir: Path,
    methods=PID_METHODS,
    require_commonality: bool = False,
) -> bool:
    """Check whether all expected example/method CSV rows exist for a seed.

    Inputs:
        seed: int, seed value to inspect.
        example_names: list[str], example names that should be complete.
        results_dir: Path, directory where result CSVs are stored.
        methods: iterable[str], method names expected for each example.
        require_commonality: bool, whether every example also needs a
            commonality row for this seed.

    Outputs:
        bool, True when every expected row exists.
    """
    pid_is_done = all(
        csv_has_seed(csv_path(results_dir, example, method), seed)
        for example in example_names
        for method in methods
    )
    if not pid_is_done or not require_commonality:
        return pid_is_done

    path = commonality_csv_path(results_dir)
    return all(csv_has_seed(path, seed, example) for example in example_names)


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
    all_results = {}
    commonality_results = {}
    for func, name in zip(functions_to_run, example_names):
        print(f"\nRunning example {func.__name__}...")
        results, rvs = main_func(config, func)
        all_results[name] = results
        if save_image:
            save_pid_comparison_table(results, f"{config['results_dir']}/{name}.png", config=config)

        if config.get("commonality_analysis", False):
            print("\nRunning commonality analysis...")
            x1, x2, target = rvs
            commonality_results[name] = commonality_analysis(
                x1,
                x2,
                target,
                method=config.get("commonality_method", "ridge_cv"),
                alphas=config.get("commonality_alphas"),
                scale_by_target_variance=config.get(
                    "commonality_scale_by_target_variance",
                    False,
                ),
            )
            print("Finished commonality analysis.")
        print(f"Finished example {func.__name__}.")

    if commonality_results:
        all_results[COMMONALITY_KEY] = commonality_results
    return all_results


def save_seed_csvs(
    seed: int,
    all_results: dict,
    results_dir: Path,
    save_commonality_csv: bool = False,
) -> None:
    """Save one seed of PID and optional commonality results to CSV.

    Inputs:
        seed: int, seed used to generate all_results.
        all_results: dict, nested result dictionaries keyed by example name.
        results_dir: Path, directory where CSVs should be written.
        save_commonality_csv: bool, whether to persist commonality rows.

    Outputs:
        None.
    """
    for example, results in all_results.items():
        if example == COMMONALITY_KEY:
            if not save_commonality_csv:
                continue

            path = commonality_csv_path(results_dir, example)
            fieldnames = [
                "seed",
                "example",
                *(csv_name for csv_name, _ in COMMONALITY_FIELDS),
            ]
            current_examples = set(results)
            old_rows = []
            if path.exists():
                with path.open(newline="", encoding="utf-8") as handle:
                    old_rows = [
                        old
                        for old in csv.DictReader(handle)
                        if not (
                            int(old["seed"]) == seed
                            and (
                                old.get("example") in current_examples
                                or not old.get("example")
                            )
                        )
                    ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(old_rows)
                for commonality_example, commonality_result in results.items():
                    row = {
                        csv_name: commonality_result[result_key]
                        for csv_name, result_key in COMMONALITY_FIELDS
                    }
                    writer.writerow({
                        "seed": seed,
                        "example": commonality_example,
                        **{key: _as_float(value) for key, value in row.items()},
                    })
            continue

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


def mean_commonality_from_csvs(
    results_dir: Path,
    example: str,
    seeds: list[int],
) -> dict:
    """Average one example's saved commonality rows across selected seeds.

    Inputs:
        results_dir: Path, directory containing the commonality seed CSV.
        example: str, example name whose rows should be averaged.
        seeds: list[int], seed values to include.

    Outputs:
        dict, averaged commonality payload using canonical analysis keys, or an
        empty dict when no matching rows exist.
    """
    path = commonality_csv_path(results_dir)
    if not path.exists():
        return {}

    seed_set = {int(seed) for seed in seeds}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["seed"]) in seed_set
            and (row.get("example") == example or not row.get("example"))
        ]
    if not rows:
        return {}

    return {
        result_key: float(np.mean([float(row[csv_name]) for row in rows]))
        for csv_name, result_key in COMMONALITY_FIELDS
    }


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
    base_seed = config.get("seed", 0)
    seed_values = list(seeds) if seeds is not None else list(range(base_seed, base_seed + (num_seeds or 1)))
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    save_commonality = bool(config.get("commonality_analysis", False))
    configured_methods = tuple(
        pid_method_display_name(method)
        for method in config.get("methods", ())
    )
    expected_methods = ("True Values", *configured_methods)

    for seed in seed_values:
        if seed_is_done(
            seed,
            example_names,
            results_dir,
            methods=expected_methods,
            require_commonality=save_commonality,
        ):
            print(f"\nSeed {seed} has already been done. Skipping to the next seed.")
            continue
        print(f"\n{'=' * 70}\nRunning all examples with seed={seed}\n{'=' * 70}")
        all_results = loop_examples({**config, "seed": seed}, functions_to_run, example_names, main_func, save_image=False)
        save_seed_csvs(
            seed,
            all_results,
            results_dir,
            save_commonality_csv=save_commonality,
        )

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
            title="PID Method Comparison",
            config=mean_config,
        )

    if save_commonality:
        mean_commonality = {
            name: result
            for name in example_names
            if (result := mean_commonality_from_csvs(results_dir, name, seed_values))
        }
        if mean_commonality:
            save_commonality_comparison_table(
                mean_commonality,
                f"{results_dir}/commonality_over_{len(seed_values)}_seeds.png",
                title="Commonality Results",
                config=mean_config,
            )
    return mean_by_example
