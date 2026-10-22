"""Run resumable pairwise PID across unordered model pairs.

One runner serves both source modes, chosen by ``source_mode`` in the YAML:

- ``prediction`` (default; the intended design): the target T is the saved
  subject PCA of the neural responses, and X1, X2 are each model's held-out
  ridge predictions of those target PCs (``prediction.yaml``).
- ``representation``: T is a PCA of the neural responses fitted on the unique
  images, and X1, X2 are PCA projections of each model's best-layer features,
  also fitted on the unique images (``representation.yaml``).

In both modes PID is evaluated on the shared held-out images, every unordered
pair is computed once, and each finished pair is appended to a resumable CSV
whose heatmaps are plotted at the end.

Usage:
    python pipeline/analysis/pairwise/pairwise.py --config pipeline/analysis/pairwise/representation.yaml
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pipeline.analysis.pairwise import pairwise_utils
from pipeline.experiments import otc as otc_experiment
from pipeline.phases.feature_manipulations import prepare_ridge_target
from pipeline.plotting import plot_pairwise_pid_matrices
from pipeline.utils import resolve_pipeline_function


DEFAULT_CONFIG_PATH = Path(__file__).with_name("representation.yaml")
SOURCE_MODES = ("prediction", "representation")


def run_pairwise_pid_pipeline(model_1_names: list[str], model_2_names: list[str],
                              otc_config: dict[str, Any], csv_path: str | Path) -> Path:
    """Run PID once per unfinished unordered model pair and checkpoint to CSV.

    Inputs:
        model_1_names: list[str] of ordered models eligible for source X1.
        model_2_names: list[str] of ordered models eligible for source X2.
        otc_config: dict[str, Any] loaded from prediction.yaml or
            representation.yaml; ``source_mode`` selects the mode and
            defaults to "prediction".
        csv_path: str or Path used for immediate checkpointing and resume.

    Outputs:
        Path to the compatible CSV containing old and newly appended results.

    The target is prepared once and each required model is prepared once.
    PID receives ``(target, X1, X2)`` directly, so the retained pair
    orientation defines X1/unq1 and X2/unq2.
    """

    output_path = Path(csv_path)
    existing_results = pairwise_utils._load_or_create_checkpoint(output_path)
    unfinished_pairs = pairwise_utils._unfinished_unordered_pairs(model_1_names, model_2_names,
                                                                  existing_results)
    if not unfinished_pairs:
        print(f"All requested unordered pairs are already complete in {output_path}.")
        return output_path

    config = otc_config
    source_mode = config.get("source_mode", "prediction")
    registry = otc_experiment.PIPELINE_STEP_FUNCTIONS
    target_extraction = resolve_pipeline_function(config["functions"], registry,
                                                  "target_extraction", required=True)
    pid_calculation = resolve_pipeline_function(config["functions"], registry,
                                                "pid_calculation", required=True)
    pid_report = resolve_pipeline_function(config["functions"], registry,
                                           "pid_report", required=False)
    if target_extraction is None or pid_calculation is None:
        raise RuntimeError("Required configured pipeline functions did not resolve.")

    layer_results_path = pairwise_utils._layer_results_path(config)
    required_models = pairwise_utils.required_models(unfinished_pairs)
    pid_kwargs = dict(config.get("pid_kwargs", {}))
    seed = int(pid_kwargs.get("rng_seed", 56))
    report_kwargs = dict(config.get("report_kwargs", {}))
    prefetch_next_model = bool((config.get("execution") or {}).get("prefetch_next_model", True))
    # Validate the mode and bind its configuration before the expensive target load.
    prepare_mode_inputs = _build_mode_input_preparer(
        source_mode, config, registry, required_models, layer_results_path, seed)
    print(f"Source mode: {source_mode}; pending unordered pairs: {len(unfinished_pairs)}; "
          f"required models: {len(required_models)}; next-model prefetch: {prefetch_next_model}.")

    target_context = target_extraction(**dict(config.get("target_kwargs", {})))
    hdf_file = target_context.get("hdf_file")
    try:
        raw_target = pairwise_utils.pop_raw_target(target_context)
        test_target, prepare_source = prepare_mode_inputs(raw_target, target_context)
        # Free the full-voxel target before the long pair loop.
        del raw_target

        source_pairs = pairwise_utils.iter_prepared_pairs(
            unfinished_pairs, prepare_source, prefetch_next_model=prefetch_next_model)
        try:
            for model_1, model_2, source_1, layer_1, source_2, layer_2 in source_pairs:
                pid_results = pid_calculation(test_target, source_1, source_2, **pid_kwargs)
                if pid_report is not None:
                    pid_report(pid_results, {"target": test_target, "source_1": source_1,
                                             "source_2": source_2, "model_1": model_1,
                                             "model_2": model_2}, **report_kwargs)

                row = pairwise_utils._result_row(model_1, model_2, layer_1, layer_2,
                                                 test_target, source_1, source_2,
                                                 pid_results, config)
                row_frame = pd.DataFrame([row], columns=pairwise_utils.PAIRWISE_RESULT_COLUMNS)
                row_frame.to_csv(output_path, mode="a", header=False, index=False)
                print(f"Completed PID for pair: {model_1}, {model_2}. "
                      f"Results appended to {output_path}")
        finally:
            source_pairs.close()
    finally:
        if hdf_file is not None:
            hdf_file.close()

    return output_path


def _build_mode_input_preparer(
    source_mode: str,
    config: dict[str, Any],
    registry: dict[str, Any],
    required_models: list[str],
    layer_results_path: Path,
    seed: int,
) -> Callable[[np.ndarray, dict[str, Any]], tuple[np.ndarray, pairwise_utils.SourcePreparer]]:
    """Validate a source mode and bind its pairwise input-preparation inputs.

    Inputs:
        source_mode: str selecting prediction or representation sources.
        config: dict[str, Any] containing the selected mode's configuration.
        registry: dict[str, Any] mapping configured function names to callables.
        required_models: list[str] containing every model needed by pending pairs.
        layer_results_path: Path to the existing best-layer CSV.
        seed: int random seed used by ridge or PCA preparation.

    Outputs:
        Callable accepting the raw target and target context and returning the
        held-out target plus a reusable source preparer.
    """

    if source_mode == "prediction":
        feature_extraction, pc_target_path = pairwise_utils._check_prediction_config(
            config, registry)
        return partial(
            pairwise_utils.prepare_prediction_inputs,
            config=config,
            required_models=required_models,
            layer_results_path=layer_results_path,
            feature_extraction=feature_extraction,
            pc_target_path=pc_target_path,
            seed=seed,
            target_preparation=prepare_ridge_target,
        )
    if source_mode == "representation":
        n_source_components = pairwise_utils._check_representation_config(config)
        return partial(
            _representation_inputs,
            config=config,
            layer_results_path=layer_results_path,
            n_source_components=n_source_components,
            seed=seed,
        )
    raise ValueError(f"source_mode must be one of {SOURCE_MODES}, got {source_mode!r}.")


def _representation_inputs(raw_target: np.ndarray, target_context: dict[str, Any],
                           config: dict[str, Any], layer_results_path: Path,
                           n_source_components: int,
                           seed: int) -> tuple[np.ndarray, pairwise_utils.SourcePreparer]:
    """Prepare the representation-mode target and the PCA source preparer.

    Inputs:
        raw_target: np.ndarray neural target with shape (n_images, n_voxels).
        target_context: dict[str, Any] with "shared1000_subj" and stimuli.
        config: dict[str, Any] with feature_manipulation_kwargs and
            feature_extraction_kwargs.
        layer_results_path: Path to the existing best-layer CSV.
        n_source_components: int PCA size shared by both sources.
        seed: int seed for the randomized PCA solver.

    Outputs:
        tuple of the held-out target with shape (n_shared, n_target_pcs) and
        a SourcePreparer returning each model's projected shared features.
    """

    n_target_components = int(config.get("feature_manipulation_kwargs", {})["n_components_target"])
    test_target = pairwise_utils._fit_unique_project_shared(raw_target,
                                                            target_context["shared1000_subj"],
                                                            n_target_components, seed)
    # (n_images, n_voxels) -> (n_shared, n_target_pcs)
    print(f"Target PCA: projected shared target shape {test_target.shape}")
    prepare_source = partial(pairwise_utils._prepare_representation_projection,
                             target_context=target_context, layer_results_path=layer_results_path,
                             feature_extraction_kwargs=config.get("feature_extraction_kwargs", {}),
                             n_components=n_source_components, random_state=seed,
                             expected_shape=test_target.shape)
    return test_target, prepare_source


def main(config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    """Run the YAML-configured pairwise analysis and plot its exact written CSV.

    Inputs:
        config_path: str or Path to prediction.yaml, representation.yaml, or
            another config with ``models`` and ``output`` sections. Relative
            paths are resolved from the repository root.

    Outputs:
        None. CSV checkpoints and heatmaps are written to configured paths.
    """

    resolved_config_path = pairwise_utils._resolve_project_path(config_path)
    with resolved_config_path.open("r", encoding="utf-8") as config_file:
        otc_config = yaml.safe_load(config_file)

    configured_models = otc_config.get("models")
    if not isinstance(configured_models, list) or not configured_models:
        raise ValueError("Config must define a non-empty 'models' list.")
    if not all(isinstance(model_name, str) and model_name for model_name in configured_models):
        raise ValueError("Every config models entry must be a non-empty string.")

    output_config = otc_config.get("output", {})
    if "csv_path" not in output_config or "plot_dir" not in output_config:
        raise ValueError("Config output must define csv_path and plot_dir.")
    written_csv_path = run_pairwise_pid_pipeline(
        model_1_names=list(configured_models), model_2_names=list(configured_models),
        otc_config=otc_config,
        csv_path=pairwise_utils._resolve_project_path(output_config["csv_path"]))
    plot_pairwise_pid_matrices(
        csv_path=written_csv_path,
        output_dir=pairwise_utils._resolve_project_path(output_config["plot_dir"]))


def parse_args() -> argparse.Namespace:
    """Parse the command-line config path.

    Inputs:
        None. Reads sys.argv.

    Outputs:
        argparse.Namespace with ``config`` (Path), defaulting to prediction.yaml.
    """

    parser = argparse.ArgumentParser(
        description="Run pairwise PID across model pairs in prediction or representation mode.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="YAML config path; relative paths are resolved "
                             "from the repository root.")
    return parser.parse_args()


if __name__ == "__main__":
    import time
    start_time = time.time()
    main(parse_args().config)
    elapsed_time = time.time() - start_time
    print(f"Pairwise PID pipeline completed in {elapsed_time:.2f} seconds.")
