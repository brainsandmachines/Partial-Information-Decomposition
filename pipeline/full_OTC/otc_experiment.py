"""Config-driven full-OTC experiment runner built on PIDPipeline."""

from __future__ import annotations

import sys
from datetime import datetime
from inspect import signature
from pathlib import Path
from pprint import pformat
from typing import Any

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
    
from pipeline.pipeline_phases.sources_target_features import prepare_target

from pipeline.pipeline_utils import (
    COMMON_PIPELINE_STEP_FUNCTIONS,
    PipelineFunction,
    run_configured_pid_pipeline,
    validate_pipeline_config_sections,
)


def run_otc_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run one full-OTC experiment from an already-loaded config dictionary.

    Inputs:
        config: dict, YAML-loaded experiment config with function names and
            kwargs sections for PIDPipeline.run.

    Output:
        context: dict, full context returned by PIDPipeline.run. The configured
            workflow and local start/completion times are printed.
    """

    #validate_config(config)
    started_at = datetime.now().astimezone()
    _print_workflow(config, PIPELINE_STEP_FUNCTIONS, started_at)
    context = run_configured_pid_pipeline(
        config=config,
        function_registry=PIPELINE_STEP_FUNCTIONS,
    )
    print(
        "\nOTC pipeline completed at: "
        f"{datetime.now().astimezone().isoformat(timespec='seconds')}"
    )
    return context


def _print_workflow(
    config: dict[str, Any],
    function_registry: dict[str, PipelineFunction],
    started_at: datetime,
) -> None:
    """Print the configured OTC function order, inputs, kwargs, and start time.

    Inputs:
        config: dict[str, Any], validated OTC configuration containing function
            names and per-step kwargs sections.
        function_registry: dict[str, callable], maps configured names to the
            callables used by PIDPipeline.
        started_at: datetime, timezone-aware local pipeline start time.

    Output:
        None. Prints a preflight workflow summary without printing array data.
    """

    calls = (
        ("Target extraction", "target_extraction", "target_kwargs", "configured keyword inputs"),
        ("Sources extraction", "sources_extraction", "sources_kwargs", "configured keyword inputs"),
        ("Layer selection", "choose_layer", "choose_layer_kwargs", "sources <- step 2"),
        (
            "Feature extraction X1",
            "feature_extraction",
            "feature_extraction_kwargs",
            "sources['X1'], selected_layers['X1'], target_context",
        ),
        (
            "Feature extraction X2",
            "feature_extraction",
            "feature_extraction_kwargs",
            "sources['X2'], selected_layers['X2'], target_context",
        ),
        ("Preprocessing", "preprocess", "preprocess_kwargs", "source_1, source_2, target"),
        (
            "Feature manipulation",
            "feature_manipulation",
            "feature_manipulation_kwargs",
            "source_1, source_2, target",
        ),
        ("PID calculation", "pid_calculation", "pid_kwargs", "target, source_1, source_2"),
        ("PID report", "pid_report", "report_kwargs", "pid_results, pipeline context"),
    )

    print("\n" + "=" * 72)
    print("OTC PID PIPELINE WORKFLOW")
    print(f"Started at: {started_at.isoformat(timespec='seconds')}")
    print("=" * 72)
    for step_number, (label, function_key, kwargs_key, dynamic_inputs) in enumerate(calls, start=1):
        configured_name = config["functions"].get(function_key)
        print(f"\n{step_number}. {label}")
        if configured_name is None:
            print("   Status: SKIPPED")
            continue

        function = function_registry.get(configured_name)
        resolved_name = (
            f"{function.__module__}.{function.__qualname__}"
            if function is not None
            else "UNRESOLVED"
        )
        print(f"   Configured function: {configured_name}")
        print(f"   Resolved callable: {resolved_name}")
        if function is not None:
            function_signature = signature(function)
            print(f"   Signature: {function_signature}")
            print(f"   Runtime RNG injected: {'rng' in function_signature.parameters}")
        print(f"   Dynamic inputs: {dynamic_inputs}")
        print(f"   Configured kwargs ({kwargs_key}):")
        for line in pformat(config.get(kwargs_key, {}), sort_dicts=False).splitlines():
            print(f"      {line}")
    print("\n" + "=" * 72)
    print("Beginning OTC pipeline execution")
    print("=" * 72)


def nsd_otc_target(
    hdf_path: str | Path,
    pkl_info_path: str | Path,
    neural_data_path: str | Path,
    subj_id: str | None = None,
    voxel_index: int | None = None,
    n_images: int | None = None,
) -> dict[str, Any]:
    """Load the full OTC target matrix and expose it under the PIDPipeline target key.

    Inputs:
        hdf_path: str or Path, path to the NSD stimulus HDF5 file.
        pkl_info_path: str or Path, path to the NSD stimulus metadata pickle.
        neural_data_path: str or Path, path to the full OTC neural data array.
        subj_id: str or None, accepted for config symmetry and not used.
        voxel_index: int or None, accepted for config symmetry and ignored.
        n_images: int or None, optional number of images/samples to keep.

    Output:
        target_context: dict, full OTC target helper output with an added
            "target" key containing the full neural_data matrix.
    """

    del subj_id, voxel_index

    target_context = prepare_target(
        hdf_path=Path(hdf_path),
        pkl_info_path=Path(pkl_info_path),
        neural_data_path=Path(neural_data_path),
    )
    target_context = dict(target_context)
    if n_images is not None:
        target_context["target"] = target_context["target"][: int(n_images)]
        target_context["image_ids_for_subj"] = target_context["image_ids_for_subj"][: int(n_images)]
    target_context["target"] = target_context["target"]


    print(f"\nLoaded full OTC target with shape {target_context['target'].shape}.")
    return target_context


def _validate_config(config: dict[str, Any]) -> None:
    """Validate the config sections needed to call the full-OTC runner.

    Inputs:
        config: dict, experiment config loaded outside this function.

    Output:
        None. Raises ValueError when required sections or invalid layer choices
        are found.
    """

    validate_pipeline_config_sections(
        config,
        (
            "functions",
            "target_kwargs",
            "sources_kwargs",
            "choose_layer_kwargs",
            "feature_extraction_kwargs",
            "pid_kwargs",
        ),
    )
    if config["functions"].get("choose_layer") == "voxel_best_layer":
        raise ValueError("Full OTC experiments cannot use choose_layer='voxel_best_layer'.")


PIPELINE_STEP_FUNCTIONS: dict[str, PipelineFunction] = {
    **COMMON_PIPELINE_STEP_FUNCTIONS,
    "nsd_otc_target": nsd_otc_target,
}
