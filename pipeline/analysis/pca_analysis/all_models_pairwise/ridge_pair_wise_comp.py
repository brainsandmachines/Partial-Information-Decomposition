"""Run resumable prediction-level ridge PID across unordered model pairs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pipeline.analysis.pca_analysis.all_models_pairwise import ridge_pairwise_utils
from pipeline.full_OTC import otc_experiment
from pipeline.pipeline_phases.feature_manipulations import prepare_ridge_target
from pipeline.pipeline_phases.preprocessing_layer import apply_saved_scaler
from pipeline.pipeline_utils import resolve_pipeline_function
from pipeline.plotting.plot_functions import plot_pairwise_pid_matrices


def run_pairwise_pid_pipeline(
    model_1_names: list[str],
    model_2_names: list[str],
    otc_config: dict[str, Any],
    csv_path: str | Path,
) -> Path:
    """Run prediction-level PID once per unfinished unordered model pair.

    Inputs:
        model_1_names: list[str] of ordered models eligible for source X1.
        model_2_names: list[str] of ordered models eligible for source X2.
        otc_config: dict[str, Any] containing the ridge pairwise configuration.
        csv_path: str or Path used for immediate checkpointing and resume.

    Outputs:
        Path to the compatible CSV containing old and newly appended results.

    The target is prepared once and each required model produces one cached
    held-out ridge prediction. PID receives ``(target, X1, X2)`` directly, so
    the retained pair orientation defines X1/unq1 and X2/unq2.
    """

    output_path = Path(csv_path)
    existing_results = ridge_pairwise_utils._load_or_create_checkpoint(output_path)
    unfinished_pairs = ridge_pairwise_utils._unfinished_unordered_pairs(
        model_1_names,
        model_2_names,
        existing_results,
    )
    if not unfinished_pairs:
        print(f"All requested unordered pairs are already complete in {output_path}.")
        return output_path

    required_models = ridge_pairwise_utils._required_models(unfinished_pairs)
    config = otc_config
    registry = otc_experiment.PIPELINE_STEP_FUNCTIONS
    target_extraction = resolve_pipeline_function(
        config["functions"], registry, "target_extraction", required=True
    )
    feature_extraction = resolve_pipeline_function(
        config["functions"], registry, "feature_extraction", required=True
    )
    pid_calculation = resolve_pipeline_function(
        config["functions"], registry, "pid_calculation", required=True
    )
    pid_report = resolve_pipeline_function(
        config["functions"], registry, "pid_report", required=False
    )
    if feature_extraction is None or pid_calculation is None:
        raise RuntimeError("Required configured pipeline functions did not resolve.")

    preprocess_kwargs = config.get("preprocess_kwargs", {})
    feature_manipulation_kwargs = config.get("feature_manipulation_kwargs", {})
    target_scaler_path = ridge_pairwise_utils._resolve_project_path(
        preprocess_kwargs["target_scaler_path"]
    )
    pc_target_path = ridge_pairwise_utils._resolve_project_path(
        feature_manipulation_kwargs["pc_target_path"]
    )
    for artifact_label, artifact_path in (
        ("target scaler", target_scaler_path),
        ("target PCA", pc_target_path),
    ):
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Configured {artifact_label} artifact does not exist: "
                f"{artifact_path}"
            )

    pid_kwargs = dict(config.get("pid_kwargs", {}))
    seed = int(pid_kwargs.get("rng_seed", 56))
    execution_config = config.get("execution", {})
    prefetch_enabled = bool(
        execution_config.get(
            "prefetch_ridge_predictions",
            execution_config.get("prefetch_model_context", True),
        )
    )
    print(
        f"Pending unordered pairs: {len(unfinished_pairs)}; required models: "
        f"{len(required_models)}; ridge-prediction prefetch: {prefetch_enabled}."
    )

    target_context = target_extraction(**dict(config.get("target_kwargs", {})))
    hdf_file = target_context.get("hdf_file")
    metadata = config.get("metadata", {})
    report_kwargs = dict(config.get("report_kwargs", {}))
    pid_config = pid_kwargs.get("config") or {}
    try:
        if "target" not in target_context:
            raise ValueError("Target extraction context is missing the 'target' array.")
        scaled_target = apply_saved_scaler(
            np.asarray(target_context["target"]),
            target_scaler_path,
        )
        target_context.pop("target", None)
        target_context.pop("neural_data", None)
        train_target, test_target, shared_mask = prepare_ridge_target(
            scaled_target,
            target_context,
            pc_target_path,
        )
        del scaled_target
        artifacts_by_model = ridge_pairwise_utils._validate_model_artifacts(
            required_models,
            config,
            expected_target_dim=test_target.shape[1],
        )
        prediction_pairs = ridge_pairwise_utils._iter_ridge_prediction_pairs(
            unfinished_pairs,
            artifacts_by_model,
            target_context,
            train_target,
            shared_mask,
            feature_extraction,
            config.get("feature_extraction_kwargs", {}),
            seed=seed,
            prefetch_ridge_predictions=prefetch_enabled,
        )
        try:
            for (
                model_1,
                model_2,
                source_1,
                layer_1,
                source_2,
                layer_2,
            ) in prediction_pairs:
                pid_results = pid_calculation(
                    test_target,
                    source_1,
                    source_2,
                    **pid_kwargs,
                )
                pid = pid_results["pid"]
                mi = pid_results["mi"]
                if pid_report is not None:
                    pid_report(
                        pid_results,
                        {
                            "target": test_target,
                            "source_1": source_1,
                            "source_2": source_2,
                            "model_1": model_1,
                            "model_2": model_2,
                        },
                        **report_kwargs,
                    )

                row = {
                    "model_1": model_1,
                    "model_2": model_2,
                    "layer_1": layer_1,
                    "layer_2": layer_2,
                    "subj_id": metadata.get("subj_id"),
                    "n_samples": test_target.shape[0],
                    "n_components_source_1": source_1.shape[1],
                    "n_components_source_2": source_2.shape[1],
                    "n_components_target": test_target.shape[1],
                    "pid_method": pid_results.get("method", pid_kwargs.get("method")),
                    "rng_seed": pid_kwargs.get("rng_seed"),
                    "bias_correction": pid_config.get("bias_correction"),
                    "red": pid["red"],
                    "unq1": pid["unq1"],
                    "unq2": pid["unq2"],
                    "syn": pid["syn"],
                    "bi_mi_1": mi["bi_mi_1"],
                    "bi_mi_2": mi["bi_mi_2"],
                    "tri_mi": mi["tri_mi"],
                }
                pd.DataFrame(
                    [row],
                    columns=ridge_pairwise_utils.PAIRWISE_RESULT_COLUMNS,
                ).to_csv(output_path, mode="a", header=False, index=False)
                print(
                    f"Completed PID for pair: {model_1}, {model_2}. "
                    f"Results appended to {output_path}"
                )
        finally:
            prediction_pairs.close()
    finally:
        if hdf_file is not None:
            hdf_file.close()

    return output_path


def main() -> None:
    """Run the YAML-configured ridge analysis and plot its exact written CSV.

    Inputs:
        None.

    Outputs:
        None. CSV checkpoints and heatmaps are written to configured paths.
    """

    config_path = "/home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/analysis/pca_analysis/all_models_pairwise/ridge_otc_pair_wise_comp.yaml"
    with open(config_path, "r", encoding="utf-8") as config_file:
        otc_config = yaml.safe_load(config_file)

    configured_models = otc_config.get("models")
    if not isinstance(configured_models, list) or not configured_models:
        raise ValueError("Config must define a non-empty 'models' list.")
    if not all(
        isinstance(model_name, str) and model_name for model_name in configured_models
    ):
        raise ValueError("Every config models entry must be a non-empty string.")

    output_config = otc_config.get("output", {})
    if "csv_path" not in output_config or "plot_dir" not in output_config:
        raise ValueError("Config output must define csv_path and plot_dir.")
    written_csv_path = run_pairwise_pid_pipeline(
        model_1_names=list(configured_models),
        model_2_names=list(configured_models),
        otc_config=otc_config,
        csv_path=ridge_pairwise_utils._resolve_project_path(output_config["csv_path"]),
    )
    plot_pairwise_pid_matrices(
        csv_path=written_csv_path,
        output_dir=ridge_pairwise_utils._resolve_project_path(
            output_config["plot_dir"]
        ),
    )


if __name__ == "__main__":
    main()
