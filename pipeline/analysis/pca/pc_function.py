"""Calculate PID and mutual information as target PCs are accumulated."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pipeline.pid_pipeline import PIDPipelineFunctions
from pipeline.analysis.pairwise import pairwise_utils
from pipeline.analysis.pca.plot_pc_results import (
    plot_pid_mi_as_function_of_pcs,
)
from pipeline.experiments.otc import PIPELINE_STEP_FUNCTIONS
from pipeline.phases.feature_manipulations import prepare_ridge_target
from pipeline.phases.sources_target_features import prepare_target
from pipeline.utils import pipeline_functions_from_config, print_pid_mi


DEFAULT_CONFIG_PATH = Path(__file__).with_name("pc_function.yaml")


def _save_pair_results(
    pair_results: dict[int, dict[str, Any]],
    model_1: str,
    model_2: str,
    results_dir: str | Path,
) -> Path:
    """Save the PC-dependent PID and MI results for one model pair.

    Inputs:
        pair_results: dict mapping target-PC counts to PID/MI result dicts.
        model_1: str containing the first model name.
        model_2: str containing the second model name.
        results_dir: str or Path naming the output directory.

    Outputs:
        Path to the written pickle file.
    """

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model_1 = model_1.replace("/", "_").replace("\\", "_")
    safe_model_2 = model_2.replace("/", "_").replace("\\", "_")
    output_path = output_dir / f"{safe_model_1}__{safe_model_2}_pc_results.pkl"
    with output_path.open("wb") as results_file:
        pickle.dump(pair_results, results_file)
    return output_path


def pc_function_analysis(
    config: dict[str, Any],
    functions: PIDPipelineFunctions,
    model1_name: list[str],
    model2_name: list[str],
    pc_path: str | Path,
    hdf_path: str | Path,
    pkl_info_path: str | Path,
    neural_data_path: str | Path,
    results_dir: str | Path | None = None,
    plot_dir: str | Path | None = None,
) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    """Calculate PID and MI for each model pair and cumulative target-PC count.

    Inputs:
        config: dict containing layer, saved-alpha, feature-extraction,
            execution, and PID settings.
        functions: PIDPipelineFunctions containing the configured feature
            extraction and PID callables.
        model1_name: list[str] containing the first-source model names.
        model2_name: list[str] containing the second-source model names.
        pc_path: str or Path pointing to the fitted target PCA model.
        hdf_path: str or Path pointing to the NSD stimulus HDF5 file.
        pkl_info_path: str or Path pointing to the NSD metadata pickle.
        neural_data_path: str or Path pointing to the neural response data.
        results_dir: str, Path, or None. When provided, one pickle is saved
            after completing each model pair.
        plot_dir: str, Path, or None. When provided, the absolute and
            normalized plots are saved immediately after each model pair.

    Outputs:
        dict nested as ``model_1 -> model_2 -> number_of_target_pcs``. Each
        PC-count entry contains the complete PID result with ``pid`` and
        ``mi`` dictionaries.
    """
    print(f"\nBias correction is set to: {config['pid_kwargs']['config']['bias_correction']}")
    print("Ridge predictions use the configured saved per-PC alphas.")
    print(f"Results will be saved to: {results_dir}")
    print(f"Plots will be saved to: {plot_dir}")
    print(f"Target PCA model will be loaded from: {pc_path}\n")

    feature_extraction = functions.feature_extraction
    pid_calculation = functions.pid_calculation
    if not callable(feature_extraction) or not callable(pid_calculation):
        raise ValueError("Feature extraction and PID calculation must be callable.")

    pairs = [(model_1, model_2) for model_1 in model1_name for model_2 in model2_name]
    required_models = pairwise_utils.required_models(pairs)
    layer_results_path = Path(config["choose_layer_kwargs"]["path_to_results"])
    seed = int(config["pid_kwargs"].get("rng_seed", 56))
    prefetch_next_model = bool(
        (config.get("execution") or {}).get("prefetch_next_model", True)
    )
    results: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}

    target_context = prepare_target(
        Path(hdf_path), Path(pkl_info_path), Path(neural_data_path))
    hdf_file = target_context.get("hdf_file")
    try:
        raw_target = pairwise_utils.pop_raw_target(target_context)
        shared_target, prepare_source = pairwise_utils.prepare_prediction_inputs(
            raw_target,
            target_context,
            config,
            required_models,
            layer_results_path,
            feature_extraction,
            Path(pc_path),
            seed,
            target_preparation=prepare_ridge_target,
        )
        del raw_target

        source_pairs = pairwise_utils.iter_prepared_pairs(
            pairs,
            prepare_source,
            prefetch_next_model=prefetch_next_model,
        )
        try:
            for model_1, model_2, source_1, _, source_2, _ in source_pairs:
                results.setdefault(model_1, {})
                print(
                    f"\nRunning target-PC analysis with Source 1: {model_1} and "
                    f"Source 2: {model_2} 😀"
                )
                pair_results: dict[int, dict[str, Any]] = {}

                for number_of_pcs in range(1, shared_target.shape[1] + 1):
                    selected_shared_target = shared_target[:, :number_of_pcs]
                    source_1_for_pid = source_1[:, :number_of_pcs]
                    source_2_for_pid = source_2[:, :number_of_pcs]
                    # Three (n_shared, n_target_pcs) arrays -> three
                    # (n_shared, number_of_pcs) arrays.
                    print(
                        f"Selecting the first {number_of_pcs} target PCs "
                        f"and running PID for {model_1} and {model_2} 😀"
                    )
                    print(
                        f"Source 1 shape: {source_1_for_pid.shape}, "
                        f"Source 2 shape: {source_2_for_pid.shape}"
                    )
                    print(f"Shared target shape: {selected_shared_target.shape}")
                    pid_results = pid_calculation(
                        selected_shared_target,
                        source_1_for_pid,
                        source_2_for_pid,
                        **(config.get("pid_kwargs") or {}),
                    )

                    pair_results[number_of_pcs] = pid_results
                    print(
                        f"\nResults after selecting the first {number_of_pcs} "
                        "target PCs:"
                    )
                    print_pid_mi(pid_results["pid"], pid_results["mi"])

                results[model_1][model_2] = pair_results
                if results_dir is not None:
                    _save_pair_results(pair_results, model_1, model_2, results_dir)
                if plot_dir is not None:
                    absolute_path, normalized_path = plot_pid_mi_as_function_of_pcs(
                        pair_results=pair_results,
                        model_1_name=model_1,
                        model_2_name=model_2,
                        output_dir=plot_dir,
                    )
                    print(f"Saved absolute plot: {absolute_path}")
                    print(f"Saved normalized plot: {normalized_path}")
        finally:
            source_pairs.close()
    finally:
        if hdf_file is not None:
            hdf_file.close()

    return results


def main() -> None:
    """Load the PC-function YAML, run all model pairs, and save their plots.

    Inputs:
        None. The function loads ``pc_function.yaml`` beside this
        module.

    Outputs:
        None. The function saves one result pickle and two plot files for each
        configured model pair.
    """

    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    configured_paths = {}
    for path_key in (
        "pc_path",
        "hdf_path",
        "pkl_info_path",
        "neural_data_path",
        "results_dir",
        "plot_dir",
    ):
        configured_path = Path(config[path_key]).expanduser()
        configured_paths[path_key] = (
            configured_path
            if configured_path.is_absolute()
            else repo_root / configured_path
        )

    layer_results_path = Path(
        config["choose_layer_kwargs"]["path_to_results"]
    ).expanduser()
    if not layer_results_path.is_absolute():
        layer_results_path = repo_root / layer_results_path
    config["choose_layer_kwargs"]["path_to_results"] = layer_results_path

    functions = pipeline_functions_from_config(
        config["functions"],
        PIPELINE_STEP_FUNCTIONS,
    )

    print(f"\n Starting Experiment: {config['experiment_name']}")
    #Create directories according ot bias correction and ridge settings
    if config["pid_kwargs"]["config"]["bias_correction"]:
        if config["feature_manipulation_kwargs"]["ridge"]:
            configured_paths["results_dir"] = Path(config["results_dir"]) / f"R_B_eigen_pid_{config['model1_name']}_{config['model2_name']}"
            configured_paths["plot_dir"] = Path(config["plot_dir"]) / f"R_B_eigen_pid_{config['model1_name']}_{config['model2_name' ]}"
        else:
            configured_paths["results_dir"] = Path(config["results_dir"]) / f"NR_B_eigen_pid_{config['model1_name']}_{config['model2_name']}"
            configured_paths["plot_dir"] = Path(config["plot_dir"]) / f"NR_B_eigen_pid_{config['model1_name']}_{config['model2_name']}"
    else:
        if config["feature_manipulation_kwargs"]["ridge"]:
            configured_paths["results_dir"] = Path(config["results_dir"]) / f"R_NB_eigen_pid_{config['model1_name']}_{config['model2_name']}"
            configured_paths["plot_dir"] = Path(config["plot_dir"]) / f"R_NB_eigen_pid_{config['model1_name']}_{config['model2_name']}"
        else:
            configured_paths["results_dir"] = Path(config["results_dir"]) / f"NR_NB_eigen_pid_{config['model1_name']}_{config['model2_name']}"
            configured_paths["plot_dir"] = Path(config["plot_dir"]) / f"NR_NB_eigen_pid_{config['model1_name']}_{config['model2_name']}"


    pc_function_analysis(
        config=config,
        functions=functions,
        model1_name=config["model1_name"],
        model2_name=config["model2_name"],
        pc_path=configured_paths["pc_path"],
        hdf_path=configured_paths["hdf_path"],
        pkl_info_path=configured_paths["pkl_info_path"],
        neural_data_path=configured_paths["neural_data_path"],
        results_dir=configured_paths["results_dir"],
        plot_dir=configured_paths["plot_dir"],
    )


if __name__ == "__main__":
    main()
