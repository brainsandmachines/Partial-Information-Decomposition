"""Utility helpers used by the thin PID pipeline orchestrator."""

from __future__ import annotations

from inspect import signature
from pathlib import Path
from typing import Any, Callable
import numpy as np

from pipeline.pid_pipeline import PIDPipeline, PIDPipelineFunctions
from pipeline.pipeline_phases.choosing_layer import overall_best_layer
from pipeline.pipeline_phases.feature_manipulations import pca_projection
PipelineFunction = Callable[..., Any]


def choose_random_sources(sources_list: list[str], size: int = 2, replace: bool = False) -> np.ndarray:
    """Randomly select a source from the list of available sources.
    
    Inputs:
        sources_list: list[str], available model or source names.
        size: int, number of sources to select.
        replace: bool, whether to sample with replacement.

    Output:
        selected_sources: np.ndarray, randomly selected source names.
    """

    selected_sources = np.random.choice(sources_list, size=size, replace=replace)
    return selected_sources


def run_configured_pid_pipeline(
    config: dict[str, Any],
    function_registry: dict[str, PipelineFunction],
    choose_layer_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run PIDPipeline from a config dictionary and function registry.

    Inputs:
        config: dict, YAML-loaded experiment config with function names and
            kwargs sections for PIDPipeline.run.
        function_registry: dict[str, callable], maps configured function names
            to concrete callables.
        choose_layer_kwargs: dict or None, optional already-arranged keyword
            arguments for the configured choose_layer function.

    Output:
        context: dict, full context returned by PIDPipeline.run.
    """

    functions = pipeline_functions_from_config(config["functions"], function_registry)
    pipeline = PIDPipeline(functions)
    return pipeline.run(
        target_kwargs=dict(config.get("target_kwargs", {})),
        sources_kwargs=dict(config.get("sources_kwargs", {})),
        choose_layer_kwargs=dict(choose_layer_kwargs or config.get("choose_layer_kwargs", {})),
        feature_extraction_kwargs=dict(config.get("feature_extraction_kwargs", {})),
        preprocess_kwargs=dict(config.get("preprocess_kwargs", {})),
        feature_manipulation_kwargs=dict(config.get("feature_manipulation_kwargs", {})),
        pid_kwargs=dict(config.get("pid_kwargs", {})),
        report_kwargs=dict(config.get("report_kwargs", {})),
    )


def pipeline_functions_from_config(
    function_config: dict[str, Any],
    function_registry: dict[str, PipelineFunction],
) -> PIDPipelineFunctions:
    """Resolve configured function names into PIDPipelineFunctions.

    Inputs:
        function_config: dict, mapping PIDPipeline step names to registry names
            or None.
        function_registry: dict[str, callable], allowed configured functions.

    Output:
        functions: PIDPipelineFunctions, resolved function bundle.
    """

    return PIDPipelineFunctions(
        target_extraction=resolve_pipeline_function(function_config, function_registry, "target_extraction", required=True),
        sources_extraction=resolve_pipeline_function(function_config, function_registry, "sources_extraction", required=True),
        choose_layer=resolve_pipeline_function(function_config, function_registry, "choose_layer", required=True),
        feature_extraction=resolve_pipeline_function(function_config, function_registry, "feature_extraction", required=True),
        preprocess=resolve_pipeline_function(function_config, function_registry, "preprocess", required=False),
        feature_manipulation=resolve_pipeline_function(function_config, function_registry, "feature_manipulation", required=False),
        pid_calculation=resolve_pipeline_function(function_config, function_registry, "pid_calculation", required=True),
        pid_report=resolve_pipeline_function(function_config, function_registry, "pid_report", required=False),
    )


def resolve_pipeline_function(
    function_config: dict[str, Any],
    function_registry: dict[str, PipelineFunction],
    step_name: str,
    required: bool,
) -> PipelineFunction | None:
    """Resolve one configured pipeline step name from a registry.

    Inputs:
        function_config: dict, configured function names by PIDPipeline step.
        function_registry: dict[str, callable], allowed configured functions.
        step_name: str, PIDPipeline function field name.
        required: bool, whether the step must be present.

    Output:
        func: callable or None, resolved pipeline function.
    """

    function_name = function_config.get(step_name)
    if function_name is None:
        if required:
            raise ValueError(f"Missing required function name for '{step_name}'.")
        return None
    if function_name not in function_registry:
        allowed = sorted(function_registry)
        raise ValueError(f"Unknown function name for '{step_name}': {function_name}. Allowed: {allowed}")
    return function_registry[function_name]


def validate_pipeline_config_sections(config: dict[str, Any], required_sections: tuple[str, ...]) -> None:
    """Validate that required config sections exist.

    Inputs:
        config: dict, experiment config loaded outside the runner.
        required_sections: tuple[str, ...], section names required by a runner.

    Output:
        None. Raises ValueError when a required section is missing.
    """

    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")


def nsd_sources(model_name_1: str, model_name_2: str) -> dict[str, dict[str, Any]]:
    """Load two model contexts and expose them under X1 and X2.

    Inputs:
        model_name_1: str, first source model name.
        model_name_2: str, second source model name.

    Output:
        sources: dict, model contexts under "X1" and "X2".
    """

    from pipeline.pipeline_phases.sources_target_features import prepare_sources

    sources = prepare_sources(model_name_1=model_name_1, model_name_2=model_name_2)
    return {"X1": sources["X1_context"], "X2": sources["X2_context"]}


def specific_layer_index(
    sources: dict[str, dict[str, Any]],
    X1_index: int,
    X2_index: int,
) -> dict[str, int]:
    """Select one configured layer index for each source.

    Inputs:
        sources: dict, source contexts under "X1" and "X2".
        X1_index: int, selected layer index for source X1.
        X2_index: int, selected layer index for source X2.

    Output:
        selected_layers: dict, layer indexes under "X1" and "X2".
    """

    from pipeline.pipeline_phases.choosing_layer import specific_index_layer_selection

    return {
        "X1": int(specific_index_layer_selection(_layer_index_values(sources, "X1", int(X1_index)), int(X1_index))),
        "X2": int(specific_index_layer_selection(_layer_index_values(sources, "X2", int(X2_index)), int(X2_index))),
    }


def random_layer_selection_for_sources(
    sources: dict[str, dict[str, Any]],
    random_seed: int | None = None,
) -> dict[str, int]:
    """Select a random valid layer index for each source.

    Inputs:
        sources: dict, source contexts under "X1" and "X2", each containing
            a "layers_ordered" list.
        random_seed: int or None, optional seed for deterministic selection.

    Output:
        selected_layers: dict, random layer indexes under "X1" and "X2".
    """

    rng = np.random.default_rng(random_seed)
    return {
        "X1": _random_layer_index_for_source(sources, "X1", rng),
        "X2": _random_layer_index_for_source(sources, "X2", rng),
    }


def voxel_best_layer_for_sources(
    sources: dict[str, dict[str, Any]],
    X1_index: int,
    X2_index: int,
    X1_path_to_results: str | Path,
    X2_path_to_results: str | Path,
) -> dict[str, int]:
    """Select each source model's best layer for one voxel index.

    Inputs:
        sources: dict, source contexts under "X1" and "X2".
        voxel_index: int, selected voxel index used for both best-layer CSV lookups.
        X1_path_to_results: str or Path, CSV path for X1 with columns
            "voxel_index" and "best_layer_index".
        X2_path_to_results: str or Path, CSV path for X2 with columns
            "voxel_index" and "best_layer_index".

    Output:
        selected_layers: dict, best layer indexes under "X1" and "X2".
    """

    del sources
    from pipeline.pipeline_phases.choosing_layer import voxel_best_layer

    selected_layers = {
        "X1": voxel_best_layer(voxel_index=int(X1_index), path_to_results=str(X1_path_to_results)),
        "X2": voxel_best_layer(voxel_index=int(X2_index), path_to_results=str(X2_path_to_results)),
    }
    missing_sources = [source_name for source_name, result in selected_layers.items() if result["l"] is None]
    if missing_sources:
        raise ValueError(f"Could not find best layer for sources: {missing_sources}")
    return {source_name: int(result["l"]) for source_name, result in selected_layers.items()}


def overall_best_layer_for_sources(
    sources: dict[str, dict[str, Any]],
    path_to_results: str | Path,
) -> dict[str, int]:
    """Select each source model's overall best OTC layer from one CSV file.

    Inputs:
        sources: dict, source contexts under "X1" and "X2", each containing
            "model_name".
        path_to_results: str or Path, CSV path with columns "model_name" and
            "best_layer_index".

    Output:
        selected_layers: dict, best layer indexes under "X1" and "X2".
    """


    selected_layers: dict[str, int] = {}
    for source_name in ("X1", "X2"):
        model_name = _model_name_for_source(sources, source_name)
        result = overall_best_layer(model_name=model_name, path_to_results=str(path_to_results))
        if result["l"] is None:
            available_names = _overall_best_layer_model_names(path_to_results)
            raise ValueError(
                f"No overall best layer found for model {model_name!r} in {path_to_results}. "
                f"First available model names: {available_names[:10]!r}"
            )
        selected_layers[source_name] = int(result["l"])
    return selected_layers


def nsd_feature_extraction(
    source_context: dict[str, Any],
    layer_index: int,
    target_context: dict[str, Any],
    batch_size_process: int,
    batch_size_dataloader: int = 128,
) -> Any:
    """Extract features for one NSD source and selected layer.

    Inputs:
        source_context: dict, one model context from nsd_sources.
        layer_index: int, selected layer index for the source.
        target_context: dict, target context containing image_ids_for_subj and stim.
        batch_size_process: int, number of images handled per outer batch.
        batch_size_dataloader: int, DataLoader batch size.

    Output:
        features: any, extracted features returned by feature_extraction.
    """

    from pipeline.pipeline_phases.sources_target_features import feature_extraction

    return feature_extraction(
        layer_index=int(layer_index),
        model_context=source_context,
        subj_image_ids=target_context["image_ids_for_subj"],
        stim_dataset=target_context["stim"],
        batch_size_process=int(batch_size_process),
        batch_size_dataloader=int(batch_size_dataloader),
    )


def pca_each_source(target: Any, source_1: Any, source_2: Any, n_components_source_1: int, n_components_source_2: int,n_components_target: int) -> tuple[Any, Any]:
    """Apply PCA separately to source_1 and source_2.

    Inputs:
        target: any, target feature matrix.
        source_1: any, first source feature matrix.
        source_2: any, second source feature matrix.
        n_components: int, number of PCA components to keep for each source.

    Output:
        reduced_sources: tuple, PCA-reduced source_1 and source_2.
    """

    if n_components_source_1 is not None:
        prj_source1 = pca_projection(source_1, n_components=int(n_components_source_1))
    else:
        prj_source1 = source_1

    if n_components_source_2 is not None:
        prj_source2 = pca_projection(source_2, n_components=int(n_components_source_2))
    else:
        prj_source2 = source_2

    if n_components_target is not None:
        prj_target = pca_projection(target, n_components=int(n_components_target))
    else:
        prj_target = target

    return (
        prj_target,
        prj_source1,
        prj_source2
    )


def pid_calc_adapter(
    target: Any,
    source_1: Any,
    source_2: Any,
    method: str,
    config: dict[str, Any] | None = None,
    rng_seed: int = 56,
    **pid_kwargs: Any,
) -> dict[str, Any]:
    """Call pid_calc using the strict PIDPipeline array order.

    Inputs:
        target: any, target samples T.
        source_1: any, first source samples X1.
        source_2: any, second source samples X2.
        method: str, PID method name passed to pid_calc.
        config: dict or None, PID config values to extend with dimensions.
        rng_seed: int, torch random seed for pid_calc.
        pid_kwargs: any, extra keyword arguments for pid_calc.

    Output:
        pid_results: dict, contains "pid", "mi", and "method".
    """

    import torch
    from Partial_Information_Decomposition.PID_calc import pid_calc

    target_tensor = _as_2d_tensor(target)
    source_1_tensor = _as_2d_tensor(source_1)
    source_2_tensor = _as_2d_tensor(source_2)
    pid_config = dict(config or {})
    pid_config.setdefault("dt", target_tensor.shape[1])
    pid_config.setdefault("dx1", source_1_tensor.shape[1])
    pid_config.setdefault("dx2", source_2_tensor.shape[1])
    pid_config.setdefault("n_samples", target_tensor.shape[0])
    pid_config.setdefault("bias_correction", False)

    rng = torch.Generator().manual_seed(int(rng_seed))
    pid, mi = pid_calc(
        config=pid_config,
        sources=[source_1_tensor, source_2_tensor],
        target=[target_tensor],
        rng=rng,
        method=method,
        **pid_kwargs,
    )
    return {"pid": pid, "mi": mi, "method": method}


def print_pid_mi_adapter(pid_results: dict[str, Any], context: dict[str, Any], **report_kwargs: Any) -> Any:
    """Print PID and MI outputs from pid_calc_adapter.

    Inputs:
        pid_results: dict, output from pid_calc_adapter.
        context: dict, full PIDPipeline context.
        report_kwargs: any, extra reporting kwargs reserved for future use.

    Output:
        report_output: any, output returned by print_pid_mi.
    """

    del context, report_kwargs
    from pipeline.pipeline_phases.report_results import print_pid_mi

    return print_pid_mi(pid_results["pid"], pid_results["mi"])


def _as_2d_tensor(value: Any) -> Any:
    """Convert samples to a 2D torch tensor.

    Inputs:
        value: any, array-like samples with shape (n_samples,) or
            (n_samples, n_features).

    Output:
        tensor: torch.Tensor, float tensor with shape (n_samples, n_features).
    """

    import torch

    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.reshape(-1, 1)
    if tensor.ndim != 2:
        raise ValueError(f"Expected 1D or 2D samples, got shape {tuple(tensor.shape)}")
    return tensor


def _random_layer_index_for_source(
    sources: dict[str, dict[str, Any]],
    source_name: str,
    rng: np.random.Generator,
) -> int:
    """Select one random layer index for one source context.

    Inputs:
        sources: dict, source contexts under "X1" and "X2".
        source_name: str, selected source key.
        rng: np.random.Generator, random number generator.

    Output:
        layer_index: int, random layer index for the source.
    """

    from pipeline.pipeline_phases.choosing_layer import random_layer_selection

    context = source_context(sources, source_name)
    layers = context.get("layers_ordered", []) if isinstance(context, dict) else []
    if not layers:
        raise ValueError(f"Source {source_name} has no layers_ordered entries for random layer selection.")
    state = np.random.get_state()
    try:
        np.random.seed(int(rng.integers(0, np.iinfo(np.int32).max)))
        return int(random_layer_selection(len(layers)))
    finally:
        np.random.set_state(state)


def _layer_index_values(sources: dict[str, dict[str, Any]], source_name: str, requested_index: int) -> list[int]:
    """Create valid layer-index values for one source.

    Inputs:
        sources: dict, source contexts under "X1" and "X2".
        source_name: str, selected source key.
        requested_index: int, configured layer index.

    Output:
        layer_indexes: list[int], index values accepted by specific index selection.
    """

    context = source_context(sources, source_name)
    layers = context.get("layers_ordered", []) if isinstance(context, dict) else []
    n_layers = max(len(layers), requested_index + 1)
    return list(range(n_layers))


def _model_name_for_source(sources: dict[str, dict[str, Any]], source_name: str) -> str:
    """Read a source model name from a source context.

    Inputs:
        sources: dict, source contexts under "X1" and "X2".
        source_name: str, selected source key.

    Output:
        model_name: str, model name stored in the source context.
    """

    context = source_context(sources, source_name)
    if not isinstance(context, dict) or "model_name" not in context:
        raise ValueError(f"Source {source_name} must contain a model_name for overall best-layer lookup.")
    return str(context["model_name"])


def _overall_best_layer_model_names(path_to_results: str | Path) -> list[str]:
    """Read model names from an overall best-layer CSV for diagnostics.

    Inputs:
        path_to_results: str or Path, CSV path with a "model_name" column.

    Output:
        model_names: list[str], model names read from the file, or an empty
            list if the file cannot be read.
    """

    import csv

    try:
        with Path(path_to_results).open("r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            return [row.get("model_name", "") for row in reader]
    except Exception:
        return []


COMMON_PIPELINE_STEP_FUNCTIONS: dict[str, PipelineFunction] = {
    "nsd_sources": nsd_sources,
    "random_layer_selection": random_layer_selection_for_sources,
    "specific_layer_index": specific_layer_index,
    "specific_index_layer_selection": specific_layer_index,
    "overall_best_layer": overall_best_layer_for_sources,
    "nsd_feature_extraction": nsd_feature_extraction,
    "pca_each_source": pca_each_source,
    "pid_calc": pid_calc_adapter,
    "print_pid_mi": print_pid_mi_adapter,
}


def source_context(sources: Any, source_name: str) -> Any:
    """Read one source context from the sources object.

    Inputs:
        sources: any, source contexts returned by prepare_sources.
        source_name: str, source key, either X1 or X2.

    Output:
        source_context: any, matching source context or None.
    """

    if not isinstance(sources, dict):
        return None
    return sources.get(f"{source_name}_context", sources.get(source_name))


def choose_one_layer(layer_func: Callable[..., Any], source_context_value: Any, layer_kwargs: dict[str, Any]) -> Any:
    """Choose one layer by adapting to the common layer-selection helper signatures.

    Inputs:
        layer_func: callable, layer-selection function.
        source_context_value: any, source context containing optional layers_ordered.
        layer_kwargs: dict, keyword inputs for the layer-selection function.

    Output:
        selected_layer: any, chosen layer name or index.
    """

    layer_names = []
    if isinstance(source_context_value, dict):
        layer_names = source_context_value.get("layers_ordered", [])

    params = signature(layer_func).parameters
    if "layer_names" in params:
        selected = layer_func(layer_names=layer_names, **layer_kwargs)
    elif "n_layers" in params:
        selected = layer_func(n_layers=len(layer_names), **layer_kwargs)
    else:
        selected = layer_func(layer_names, **layer_kwargs)

    if hasattr(selected, "__index__") and layer_names:
        return layer_names[int(selected)]
    return selected
