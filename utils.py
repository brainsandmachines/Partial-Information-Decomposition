import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from nilearn import datasets, plotting
from PIL import Image
from pathlib import Path
import pandas as pd
import csv
import json
import hashlib
from typing import Callable


RUN_SIGNATURE_COLUMN = "__run_signature__"


def check_file_exists(file_path):
    """Check if a file exists at the given path.
    if it exists change it's name by adding a number at the end.

    Args:
        file_path (str): The path to the file."""
    
    base, ext = os.path.splitext(file_path)
    counter = 1
    new_file_path = file_path
    while os.path.exists(new_file_path):
        new_file_path = f"{base}_{counter}{ext}"
        counter += 1
    return new_file_path

def check_folder_exists(folder_path):
    """Check if a folder exists at the given path.
    if it doesn't exist, create it.

    Args:
        folder_path (str): The path to the folder."""
    base, ext = os.path.splitext(folder_path)
    counter = 1
    new_folder_path = folder_path
    while os.path.exists(new_folder_path):
        new_folder_path = f"{base}_{counter}{ext}"
        counter += 1
    os.makedirs(new_folder_path)
    return new_folder_path
    
def create_permuation(list_to_permute):
    """This function take a range of indices 
    and return a permuted version of it.
    Args:    
        list_to_permute (list,np.array,torch.Tensor): list to permute
        
    Returns:
        permuted_list (list,np.array,torch.Tensor): permuted list
    """
    permute_type = type(list_to_permute)

    if not isinstance(list_to_permute, (np.ndarray)):
        list_to_permute = np.array(list_to_permute)

    list_to_permute = list_to_permute[np.random.permutation(len(list_to_permute))]

    return permute_type(list_to_permute) 



class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

def check_equal_type_invariance(a,b) -> bool:
    """Check if two inputs are equal in value and type invariance.
    
    Args:
        a: First input.
        b: Second input."""
    if type(a) == type(b):
        return a == b
    # if b is None:
    #     
    element_a = a[0] if isinstance(a, (list, np.ndarray, torch.Tensor)) and len(a) > 0 else a
    
    if pd.isna(element_a) and pd.isna(b):
        return True
    try:
        b_converted = type(element_a)(b)
        return b_converted == a
    except (ValueError, TypeError):
        return False

def meta_exists(meta_data: dict, csv_path) -> bool:
    """
    Check whether a row with identical meta_data already exists in a CSV file. 
    it is invariant to type differences (e.g., int vs float vs str).
    
    Args:
        meta_data (dict): hyperparameter dictionary
        csv_path (Path or str): path to csv file
    
    Returns:
        bool: True if meta_data already exists, False otherwise
    """
    if not csv_path.exists():
        return False

    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    if df.empty:
        return False


    cols = meta_data.keys()

    
    for record in records:
        mask_list = []
        is_equal = False
        for col in cols:
            is_equal = check_equal_type_invariance(record[col], meta_data[col])
            mask_list.append(is_equal)
        if all(mask_list):
            return True
    else:
        return False


def _to_float_or_none(value):
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.item())
    return None


def extract_all_components(ca_results: dict, pid_results: dict, mi_results: dict) -> dict:
    combined = {}

    for key, value in ca_results.items():
        numeric_value = _to_float_or_none(value)
        if numeric_value is not None:
            combined[f"CA_{key}"] = numeric_value

    for key, value in pid_results.items():
        numeric_value = _to_float_or_none(value)
        if numeric_value is not None:
            combined[f"PID_{key}"] = numeric_value

    for key, value in mi_results.items():
        numeric_value = _to_float_or_none(value)
        if numeric_value is not None:
            combined[f"{key}"] = numeric_value

    return combined


def summarize_seed_results(results: list[dict]) -> dict:
    if not results:
        return {}

    metric_names = results[0].keys()
    summary = {}
    for metric in metric_names:
        values = np.array([row[metric] for row in results], dtype=float)
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def print_seed_summary(summary: dict, n_seeds: int, seed_start: int) -> None:
    print("\n" + "=" * 70)
    print(f"CA + PID component summary across {n_seeds} seeds (start={seed_start})")
    print("=" * 70)
    for metric, stats in summary.items():
        print(f"{metric}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")


def _normalize_config_value(value):
    if isinstance(value, np.random.Generator):
        return "np.random.Generator"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def get_experiment_name(config: dict) -> str:
    explicit_name = config.get("test_name")
    if explicit_name is not None:
        return explicit_name

    normalized = {
        key: _normalize_config_value(value)
        for key, value in sorted(config.items(), key=lambda item: item[0])
    }
    config_blob = json.dumps(normalized, sort_keys=True, default=str)
    digest = hashlib.sha1(config_blob.encode("utf-8")).hexdigest()[:12]
    return f"exp_{digest}"


def _parse_csv_numeric(value: str):
    if value is None:
        return ""
    stripped = value.strip()
    if stripped == "":
        return ""
    try:
        return float(stripped)
    except ValueError:
        return stripped


def get_seed_runs_csv_path(config: dict) -> Path:
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    prefix = config.get("all_runs_results_prefix", "seed_runs")
    experiment_name = get_experiment_name(config)
    return results_dir / f"{prefix}_{experiment_name}.csv"


def get_seed_summary_csv_path(config: dict) -> Path:
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    prefix = config.get("results_prefix", "seed_summary")
    experiment_name = get_experiment_name(config)
    return results_dir / f"{prefix}_{experiment_name}.csv"


def load_seed_run_checkpoint(config: dict) -> tuple[Path, list[dict], list[str]]:
    file_path = get_seed_runs_csv_path(config)
    if not file_path.exists() or file_path.stat().st_size == 0:
        return file_path, [], []

    with open(file_path, "r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))

    header = []
    data_start_index = None
    for index, row in enumerate(rows):
        if row and row[0] == "seed":
            header = row
            data_start_index = index + 1
            break

    if not header or data_start_index is None:
        return file_path, [], []

    seed_rows = []
    for row in rows[data_start_index:]:
        if not row:
            continue
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        parsed = {column: _parse_csv_numeric(value) for column, value in zip(header, row)}
        seed_value = parsed.get("seed")
        if seed_value == "":
            continue
        parsed["seed"] = int(float(seed_value))
        for key, value in list(parsed.items()):
            if key == "seed":
                continue
            if isinstance(value, str) and value == "":
                parsed.pop(key)
        seed_rows.append(parsed)

    metric_names = [
        column
        for column in header
        if column not in {"seed", RUN_SIGNATURE_COLUMN}
    ]
    return file_path, seed_rows, metric_names


def _ensure_seed_runs_header(file_path: Path, config: dict, metric_names: list[str]) -> None:
    if file_path.exists() and file_path.stat().st_size > 0:
        return

    config_to_save = {
        key: _normalize_config_value(value)
        for key, value in dict(config).items()
    }
    config_json = json.dumps(config_to_save, sort_keys=True, default=str)
    header = ["seed", *metric_names]

    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["config_json", config_json])
        writer.writerow([])
        writer.writerow(header)


def append_seed_run_checkpoint(config: dict, row: dict, metric_names: list[str]) -> Path:
    file_path = get_seed_runs_csv_path(config)
    _ensure_seed_runs_header(file_path, config, metric_names)

    header = ["seed", *metric_names]
    with open(file_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([row.get(column, "") for column in header])

    return file_path


def save_seed_summary_csv(summary: dict, config: dict) -> Path:
    file_path = get_seed_summary_csv_path(config)
    config_to_save = {
        key: _normalize_config_value(value)
        for key, value in dict(config).items()
    }
    config_json = json.dumps(config_to_save, sort_keys=True, default=str)

    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["config_json", config_json])
        writer.writerow([])
        writer.writerow(["metric", "mean", "std"])
        for metric, stats in summary.items():
            writer.writerow([metric, stats["mean"], stats["std"]])

    return file_path


def run_multi_seed_experiment(
    config: dict,
    per_seed_runner: Callable[[int, dict], dict],
) -> tuple[dict, list[dict]]:
    all_seed_runs_path, seed_rows, metric_names = load_seed_run_checkpoint(config)
    completed_seeds = {int(row["seed"]) for row in seed_rows}
    all_component_results = [
        {
            key: value
            for key, value in row.items()
            if key not in {"seed", RUN_SIGNATURE_COLUMN}
        }
        for row in seed_rows
    ]

    seed_start = config["seed_start"]
    n_seeds = config["n_seeds"]
    progress_print_every = config.get("progress_print_every", 100)
    target_seeds = set(range(seed_start, seed_start + n_seeds))
    completed_target_runs = len(completed_seeds.intersection(target_seeds))

    if seed_rows:
        print(f"Loaded {len(completed_seeds)} completed seeds from: {all_seed_runs_path}")

    for seed in range(seed_start, seed_start + n_seeds):
        if seed in completed_seeds:
            print(f"Skipping seed {seed} (already completed).")
            continue

        print(f"\nRunning seed {seed} ({completed_target_runs + 1}/{n_seeds})...")
        single_run_results = per_seed_runner(seed, config)
        all_component_results.append(single_run_results)

        row = {"seed": seed}
        row.update(single_run_results)
        seed_rows.append(row)
        completed_seeds.add(seed)
        completed_target_runs += 1

        if not metric_names:
            metric_names = list(single_run_results.keys())
        append_seed_run_checkpoint(config, row=row, metric_names=metric_names)

        if progress_print_every > 0 and completed_target_runs % progress_print_every == 0:
            running_summary = summarize_seed_results(all_component_results)
            print(f"\nIntermediate summary after {completed_target_runs} runs:")
            print_seed_summary(running_summary, completed_target_runs, seed_start)

    return summarize_seed_results(all_component_results), seed_rows
            

