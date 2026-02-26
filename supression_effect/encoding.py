import torch
import numpy as np
import joblib
import sys
import csv
import json
from pathlib import Path
from datetime import datetime


def _append_project_root_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.append(str(root))


_append_project_root_to_path()

from encoding_model.algoanut_data import argObj
from encoding_model.suppresion_model import train_save_or_load   
from encoding_model.suppression_core import *
from Partial_Information_Decomposition.Idep_multivariate_gauss import Idep_multivariate_gauss
from encoding_model.encoding_utils import get_specific_roi_fmri
from Partial_Information_Decomposition.PID_util import *


def get_run_config() -> dict:
    return {
        "data_dir": "/mnt/data4tb/data_algonauts/",
        "parent_submission_dir": "/mnt/data4tb/data_algonauts/submissions",
        "subj": 1,
        "method": "ridge_cv",
        "n_s": 7000,
        "n_f": 500,
        "rng_seed": np.random.default_rng(seed=30),
        "n_seeds": 10000,
        "seed_start": 0,
        "snr": 10,
        "mixing_dimension": 50,
        "suppression_strength": 0.5,
        "suppression_method": "permutate",
        "path_to_load": "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/trained_models/roi_models/FBA-1_alexnet_features.8_subj01_1.pth/FBA-1_alexnet_features.8_subj01.pth_encoding_model.joblib",
        "fmri_dict_path": "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/fmri_dicts/subj1_fmri_dicts.joblib",
        "roi_name": "FBA-1",
        "results_dir": "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/Suppresion_FBA_Encoding",
        "results_prefix": "seed_summary",
        "all_runs_results_prefix": "seed_runs",
        "progress_print_every": 100,
        "test_name": 'FBA-Encoding-suppresion_exp1',  # Optional: specify a custom name for the summary file; if None, uses timestampW
    }


def load_model_and_fmri(config: dict):
    loaded_model = train_save_or_load(path_to_load=config["path_to_load"])
    real_features = loaded_model["features_train"]
    fmri_dict = joblib.load(config["fmri_dict_path"])
    return real_features, fmri_dict


def prepare_inputs(config: dict, real_features: np.ndarray, fmri_dict: dict):
    args = argObj(config["data_dir"], config["parent_submission_dir"], config["subj"])
    n_s = config["n_s"]
    n_f = config["n_f"]

    lh_fmri_train = fmri_dict["lh_fmri_train"][:n_s, :]
    rh_fmri_train = fmri_dict["rh_fmri_train"][:n_s, :]
    lh_fmri, _ = get_specific_roi_fmri(
        args=args,
        lh_fmri=lh_fmri_train,
        rh_fmri=rh_fmri_train,
        roi_name=config["roi_name"],
    )

    features = real_features[:n_s, :]
    encoder, selected_features = create_encoder(
        rng=config["rng_seed"],
        features=features,
        target=lh_fmri,
        n_features=n_f,
    )
    return encoder, selected_features


def run_suppression_pipeline(config: dict, selected_features: np.ndarray, encoder, verbose: bool = True):
    if verbose:
        print("\nEncoder's features shape: ", selected_features.shape)
        print("\nCreating predictions from encoder...")

    y_hat_lh, y_hat_rh = create_predictions(encoder, reg_rh=None, features=selected_features)
    if verbose:
        print("Predictions created.\nPredicted fMRI shape (LH): ", y_hat_lh.shape) if y_hat_lh is not None else None
        print("\nPredicted fMRI shape (RH): ", y_hat_rh.shape) if y_hat_rh is not None else None

    if verbose:
        print("Creating suppression model...")
    X_M1, X_M2, target = create_supression_model(
        rng=config["rng_seed"],
        signal=y_hat_lh,
        suppresion_method=config["suppression_method"],
        features=selected_features,
        suppression_strength=config["suppression_strength"],
        mixing_dimension=config["mixing_dimension"],
        snr=config["snr"],
    )

    ca = commonality_analysis(X_M1, X_M2, target, method=config["method"])
    print('\nDone calculating commonality analysis.')
    m1 = torch.tensor(X_M1, dtype=torch.float64)
    m2 = torch.tensor(X_M2, dtype=torch.float64)
    t = torch.tensor(target, dtype=torch.float64)

    sources = [m1, m2]
    targets = [t]
    idep_class = Idep_multivariate_gauss(sources, targets, bias_correction=True)
    pid, mi = idep_class.idep()
    return ca, pid,mi


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
            "std": float(np.std(values, ddof=1)),
        }
    return summary


def save_all_seed_runs_results(seed_rows: list[dict], config: dict) -> Path:
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    test_name = datetime.now().strftime("%Y%m%d_%H%M%S") if config.get("test_name") is None else config["test_name"]
    prefix = config.get("all_runs_results_prefix", "seed_runs")
    file_path = results_dir / f"{prefix}_{test_name}.csv"

    config_to_save = dict(config)
    config_to_save["rng_seed"] = str(config_to_save.get("rng_seed"))
    config_json = json.dumps(config_to_save, sort_keys=True)

    if not seed_rows:
        metric_names = []
    else:
        metric_names = []
        for row in seed_rows:
            for key in row.keys():
                if key != "seed" and key not in metric_names:
                    metric_names.append(key)

    header = ["seed", *metric_names]

    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["config_json", config_json])
        writer.writerow([])
        writer.writerow(header)
        for row in seed_rows:
            writer.writerow([row.get(column, "") for column in header])

    return file_path


def run_multi_seed_experiment(config: dict) -> tuple[dict, list[dict]]:
    real_features, fmri_dict = load_model_and_fmri(config)
    all_component_results = []
    seed_rows = []

    seed_start = config["seed_start"]
    n_seeds = config["n_seeds"]
    progress_print_every = config.get("progress_print_every", 100)

    for seed in range(seed_start, seed_start + n_seeds):
        completed_runs = seed - seed_start + 1
        print(f"\nRunning seed {seed} ({completed_runs}/{n_seeds})...")
        run_config = dict(config)
        run_config["rng_seed"] = np.random.default_rng(seed=seed)

        encoder, selected_features = prepare_inputs(run_config, real_features, fmri_dict)
        ca_results, pid_results, mi_results = run_suppression_pipeline(
            run_config,
            selected_features,
            encoder,
            verbose=False,
        )
        single_run_results = extract_all_components(ca_results, pid_results, mi_results)
        all_component_results.append(single_run_results)
        row = {"seed": seed}
        row.update(single_run_results)
        seed_rows.append(row)

        if progress_print_every > 0 and completed_runs % progress_print_every == 0:
            running_summary = summarize_seed_results(all_component_results)
            print(f"\nIntermediate summary after {completed_runs} runs:")
            print_seed_summary(running_summary, completed_runs, seed_start)

    return summarize_seed_results(all_component_results), seed_rows


def print_seed_summary(summary: dict, n_seeds: int, seed_start: int) -> None:
    print("\n" + "=" * 70)
    print(f"CA + PID component summary across {n_seeds} seeds (start={seed_start})")
    print("=" * 70)
    for metric, stats in summary.items():
        print(f"{metric}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")


def save_seed_summary(summary: dict, config: dict) -> Path:
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    test_name = datetime.now().strftime("%Y%m%d_%H%M%S") if config.get("test_name") is None else config["test_name"]
    file_path = results_dir / f"{config['results_prefix']}_{test_name}.csv"

    config_to_save = dict(config)
    config_to_save["rng_seed"] = str(config_to_save.get("rng_seed"))
    config_json = json.dumps(config_to_save, sort_keys=True)

    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["config_json", config_json])
        writer.writerow([])
        writer.writerow(["metric", "mean", "std"])
        for metric, stats in summary.items():
            writer.writerow([
                metric,
                stats["mean"],
                stats["std"],
            ])

    return file_path


def print_results(outputs: dict, mi: dict, pid: dict) -> None:
    print("\nPID Commonality Results:")
    for key, value in outputs.items():
        print(f"- {key}: {value:.4f}")

    print("\n Mutual Information: ")
    for key, value in mi.items():
        print(f"- {key}: {value:.4f}")

    print("\nIdep PID results:")
    for key, value in pid.items():
        print(f"- {key}: {value:.4f}")


def main() -> None:
    config = get_run_config()
    summary, seed_rows = run_multi_seed_experiment(config)
    print_seed_summary(summary, config["n_seeds"], config["seed_start"])
    all_runs_path = save_all_seed_runs_results(seed_rows, config)
    saved_path = save_seed_summary(summary, config)
    print(f"\nSaved all seed run results to: {all_runs_path}")
    print(f"\nSaved summary to: {saved_path}")


if __name__ == "__main__":
    main()