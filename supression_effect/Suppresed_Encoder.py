import torch
import numpy as np
import joblib
import sys
from pathlib import Path


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
from utils import (
    run_multi_seed_experiment,
    append_seed_run_checkpoint,
    get_seed_runs_csv_path,
    save_seed_summary_csv,
    extract_all_components,
    print_seed_summary,
    create_test_histograms_with_kde,
    load_hist_kde_and_change_colors,
    seed_summary_to_table,
    save_seed_summary_table_image,
)


def get_run_config() -> dict:
    return {
        "data_dir": "/mnt/data4tb/data_algonauts/",
        "parent_submission_dir": "/mnt/data4tb/data_algonauts/submissions",
        "subj": 1,
        "method": "ridge_cv",
        "n_s": 7000,
        "n_f": 500,
        "rng_seed": np.random.default_rng(seed=30),
        "n_seeds": 20000,
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
    n_f = config["n_f"] if config["n_f"] is not None else real_features.shape[1]

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


def save_all_seed_runs_results(seed_rows: list[dict], config: dict) -> Path:
    if not seed_rows:
        return get_seed_runs_csv_path(config)

    metric_names = []
    for row in seed_rows:
        for key in row.keys():
            if key != "seed" and key not in metric_names:
                metric_names.append(key)

    file_path = get_seed_runs_csv_path(config)
    if file_path.exists():
        file_path.unlink()

    for row in seed_rows:
        append_seed_run_checkpoint(config, row=row, metric_names=metric_names)

    return file_path


def run_single_seed(
    seed: int,
    config: dict,
    real_features: np.ndarray,
    fmri_dict: dict,
) -> dict:
    run_config = dict(config)
    run_config["rng_seed"] = np.random.default_rng(seed=seed)

    encoder, selected_features = prepare_inputs(run_config, real_features, fmri_dict)
    ca_results, pid_results, mi_results = run_suppression_pipeline(
        run_config,
        selected_features,
        encoder,
        verbose=False,
    )
    return extract_all_components(ca_results, pid_results, mi_results)


def run_encoding_multi_seed_experiment(config: dict) -> tuple[dict, list[dict]]:
    real_features, fmri_dict = load_model_and_fmri(config)
    return run_multi_seed_experiment(
        config,
        per_seed_runner=lambda seed, run_config: run_single_seed(
            seed,
            run_config,
            real_features=real_features,
            fmri_dict=fmri_dict,
        ),
    )


def save_seed_summary(summary: dict, config: dict) -> Path:
    return save_seed_summary_csv(summary, config)


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
    summary, seed_rows = run_encoding_multi_seed_experiment(config)
    print_seed_summary(summary, config["n_seeds"], config["seed_start"])
    all_runs_path = get_seed_runs_csv_path(config)
    saved_path = save_seed_summary(summary, config)
    print(f"\nSaved all seed run results to: {all_runs_path}")
    print(f"\nSaved summary to: {saved_path}")


if __name__ == "__main__":
    print("Running suppression experiment with FBA encoding...")
    main()
    csv_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/Suppresion_FBA_Encoding/seed_runs_NoBiasCorrection-FBA-Encoding-suppresion_exp1.csv"
    output_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/Simulation_figs/Exp1_FBA_No_Bias_Correction"
    create_test_histograms_with_kde(csv_path, output_path,bar_color="#55A868", kde_color="#000000")

    summary_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/Suppresion_FBA_Encoding/seed_summary_NoBiasCorrection-FBA-Encoding-suppresion_exp1.csv"
    save_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/Suppresion_FBA_Encoding/NoBias_seed_summary_FBA-Encoding-suppresion_exp1_table.png"
    save_seed_summary_table_image(summary_path,image_path=save_path) 