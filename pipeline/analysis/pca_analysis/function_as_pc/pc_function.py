"""Calculate PID and mutual information as target PCs are accumulated."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pipeline.pid_pipeline import PIDPipeline, PIDPipelineFunctions
from pipeline.analysis.pca_analysis.function_as_pc.plot_pc_results import (
    plot_pid_mi_as_function_of_pcs,
)
from pipeline.full_OTC.otc_experiment import PIPELINE_STEP_FUNCTIONS
from pipeline.pipeline_phases.feature_manipulations import prepare_ridge_target,pca_source
from pipeline.pipeline_phases.sources_target_features import prepare_target
from pipeline.pipeline_utils import pipeline_functions_from_config
from pipeline.ridge_find_alpha.find_alpha import find_alpha_per_pc


DEFAULT_CONFIG_PATH = Path(__file__).with_name("pc_function_config.yaml")


def _prepare_source_for_pid(
    source: np.ndarray,
    train_target: np.ndarray,
    shared_mask: np.ndarray,
    ridge: bool,
) -> np.ndarray:
    """Prepare one model source for PID on the held-out shared images.

    Inputs:
        source: np.ndarray with model features for every image.
        train_target: np.ndarray with target-PC scores for non-shared images.
        shared_mask: np.ndarray selecting the held-out shared images.
        ridge: bool indicating whether to predict target PCs with ridge.

    Outputs:
        np.ndarray containing held-out ridge predictions when ``ridge`` is
        true, or held-out source features otherwise.
    """

    if not ridge:
        return pca_source(source,shared_mask,train_target.shape[1])
        

    _, ridge_model = find_alpha_per_pc(source[~shared_mask], train_target)
    return ridge_model.predict(source[shared_mask])


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
        config: dict containing layer, feature-extraction, ridge, and PID kwargs.
        functions: PIDPipelineFunctions containing the configured layer,
            feature-extraction, source-extraction, and PID callables.
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

    target_context = prepare_target(
        Path(hdf_path),
        Path(pkl_info_path),
        Path(neural_data_path),
    )
    train_target, shared_target, shared_mask = prepare_ridge_target(
        target_context["target"],
        target_context,
        pc_path,
    )
    pipeline = PIDPipeline(functions)
    ridge = config["feature_manipulation_kwargs"]["ridge"]
    feature_kwargs = config.get("feature_extraction_kwargs", {})
    results: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}

    for model_1 in model1_name:
        print(f"\nRunning target-PC analysis with Source 1: {model_1} 😀")
        results[model_1] = {}
        source_1_raw = None

        for model_2 in model2_name:
            sources = pipeline.functions.sources_extraction(
                model_name_1=model_1,
                model_name_2=model_2,
            )
            selected_layers = pipeline.functions.choose_layer(
                sources,
                **(config.get("choose_layer_kwargs") or {}),
            )

            if source_1_raw is None:
                source_1_raw = pipeline.functions.feature_extraction(
                    sources["X1"],
                    selected_layers["X1"],
                    target_context,
                    **feature_kwargs,
                )

            source_2_raw = pipeline.functions.feature_extraction(
                sources["X2"],
                selected_layers["X2"],
                target_context,
                **feature_kwargs,
            )
            pair_results: dict[int, dict[str, Any]] = {}

            for number_of_pcs in range(1, shared_target.shape[1] + 1):
                selected_train_target = train_target[:, :number_of_pcs]
                selected_shared_target = shared_target[:, :number_of_pcs]
                print(
                    f"Selecting the first {number_of_pcs} target PCs, fitting "
                    f"ridge, and running PID for {model_1} and {model_2} 😀"
                )
                source_1_for_pid = _prepare_source_for_pid(
                    source_1_raw,
                    selected_train_target,
                    shared_mask,
                    ridge,
                )
                source_2_for_pid = _prepare_source_for_pid(
                    source_2_raw,
                    selected_train_target,
                    shared_mask,
                    ridge,
                )
                pair_results[number_of_pcs] = pipeline.functions.pid_calculation(
                    selected_shared_target,
                    source_1_for_pid,
                    source_2_for_pid,
                    **(config.get("pid_kwargs") or {}),
                )

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

    return results


def main() -> None:
    """Load the PC-function YAML, run all model pairs, and save their plots.

    Inputs:
        None. The function loads ``pc_function_config.yaml`` beside this
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
