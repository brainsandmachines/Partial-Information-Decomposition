"""Config-driven voxel experiment runner built on PIDPipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pipeline.pid_pipeline import PIDPipeline, PIDPipelineFunctions

PipelineFunction = Callable[..., Any]


def run_voxel_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run one voxel experiment from an already-loaded config dictionary.

    Inputs:
        config: dict, YAML-loaded experiment config with function names and
            kwargs sections for PIDPipeline.run.

    Output:
        context: dict, full context returned by PIDPipeline.run.
    """

    _validate_config(config)
    functions = _pipeline_functions_from_config(config["functions"])
    choose_layer_kwargs = dict(config.get("choose_layer_kwargs", {}))
    if config["functions"].get("choose_layer") == "voxel_best_layer":
        choose_layer_kwargs.setdefault("voxel_index", config["target_kwargs"]["voxel_index"])
    pipeline = PIDPipeline(functions)
    return pipeline.run(
        target_kwargs=dict(config.get("target_kwargs", {})),
        sources_kwargs=dict(config.get("sources_kwargs", {})),
        choose_layer_kwargs=choose_layer_kwargs,
        feature_extraction_kwargs=dict(config.get("feature_extraction_kwargs", {})),
        preprocess_kwargs=dict(config.get("preprocess_kwargs", {})),
        feature_manipulation_kwargs=dict(config.get("feature_manipulation_kwargs", {})),
        pid_kwargs=dict(config.get("pid_kwargs", {})),
        report_kwargs=dict(config.get("report_kwargs", {})),
    )


def nsd_voxel_target(
    voxel_index: int,
    subj_id: str,
    hdf_path: str | Path,
    pkl_info_path: str | Path,
    neural_data_path: str | Path,
    n_images: int | None = None,
) -> dict[str, Any]:
    """Load one voxel target and expose it under the PIDPipeline target key.

    Inputs:
        voxel_index: int, selected voxel index in the neural response matrix.
        subj_id: str, subject identifier passed to the target helper.
        hdf_path: str or Path, path to the NSD stimulus HDF5 file.
        pkl_info_path: str or Path, path to the NSD stimulus metadata pickle.
        neural_data_path: str or Path, path to the neural data array.
        n_images: int or None, optional number of images/samples to keep.

    Output:
        target_context: dict, target helper output with an added "target" key.
    """

    if voxel_index is None:
        raise ValueError("target_kwargs['voxel_index'] must be an int for run_voxel_experiment.")


    from pipeline.pipeline_phases.sources_target_features import prepare_target_for_voxel

    target_context = prepare_target_for_voxel(
        voxel_index=int(voxel_index),
        subj_id=subj_id,
        hdf_path=Path(hdf_path),
        pkl_info_path=Path(pkl_info_path),
        neural_data_path=Path(neural_data_path),
    )
    target_context = dict(target_context)
    if n_images is not None:
        target_context["neural_data"] = target_context["neural_data"][: int(n_images)]
        target_context["image_ids_for_subj"] = target_context["image_ids_for_subj"][: int(n_images)]
    target_context["target"] = target_context["neural_data"]
    return target_context


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
    """Select one layer index for each source.

    Inputs:
        sources: dict, source contexts under "X1" and "X2".
        X1_index: int, selected layer index for source X1.
        X2_index: int, selected layer index for source X2.

    Output:
        selected_layers: dict, layer indexes under "X1" and "X2".
    """

    del sources
    return {"X1": int(X1_index), "X2": int(X2_index)}


def voxel_best_layer_for_sources(
    sources: dict[str, dict[str, Any]],
    voxel_index: int,
    X1_path_to_results: str | Path,
    X2_path_to_results: str | Path,
) -> dict[str, int]:
    """Select each source model's best layer for one voxel index.

    Inputs:
        sources: dict, source contexts under "X1" and "X2".
        voxel_index: int, selected voxel index used for both best-layer CSV lookups.
        X1_path_to_results: str or Path, CSV path for source X1 with columns
            "voxel_index" and "best_layer_index".
        X2_path_to_results: str or Path, CSV path for source X2 with columns
            "voxel_index" and "best_layer_index".

    Output:
        selected_layers: dict, best layer indexes under "X1" and "X2".
    """

    del sources
    from pipeline.pipeline_phases.choosing_layer import voxel_best_layer

    selected_layers = {
        "X1": voxel_best_layer(voxel_index=int(voxel_index), path_to_results=str(X1_path_to_results)),
        "X2": voxel_best_layer(voxel_index=int(voxel_index), path_to_results=str(X2_path_to_results)),
    }
    missing_sources = [source_name for source_name, result in selected_layers.items() if result["l"] is None]
    if missing_sources:
        raise ValueError(f"Could not find best layer for sources: {missing_sources}")
    return {source_name: int(result["l"]) for source_name, result in selected_layers.items()}


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


def pca_each_source(source_1: Any, source_2: Any, n_components: int) -> tuple[Any, Any]:
    """Apply PCA separately to source_1 and source_2.

    Inputs:
        source_1: any, first source feature matrix.
        source_2: any, second source feature matrix.
        n_components: int, number of PCA components to keep for each source.

    Output:
        reduced_sources: tuple, PCA-reduced source_1 and source_2.
    """


    from pipeline.pipeline_phases.feature_manipulations import pca_projection

    return (
        pca_projection(source_1, n_components=int(n_components)),
        pca_projection(source_2, n_components=int(n_components)),
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


def _pipeline_functions_from_config(function_config: dict[str, Any]) -> PIDPipelineFunctions:
    """Resolve configured function names into PIDPipelineFunctions.

    Inputs:
        function_config: dict, mapping PIDPipeline step names to registry names or None.

    Output:
        functions: PIDPipelineFunctions, resolved pipeline function bundle.
    """

    return PIDPipelineFunctions(
        target_extraction=_resolve_function(function_config, "target_extraction", required=True),
        sources_extraction=_resolve_function(function_config, "sources_extraction", required=True),
        choose_layer=_resolve_function(function_config, "choose_layer", required=True),
        feature_extraction=_resolve_function(function_config, "feature_extraction", required=True),
        preprocess=_resolve_function(function_config, "preprocess", required=False),
        feature_manipulation=_resolve_function(function_config, "feature_manipulation", required=False),
        pid_calculation=_resolve_function(function_config, "pid_calculation", required=True),
        pid_report=_resolve_function(function_config, "pid_report", required=False),
    )


def _resolve_function(function_config: dict[str, Any], step_name: str, required: bool) -> PipelineFunction | None:
    """Resolve one configured pipeline step name.

    Inputs:
        function_config: dict, configured function names by PIDPipeline step.
        step_name: str, PIDPipeline function field name.
        required: bool, whether the step must be present.

    Output:
        func: callable or None, resolved wrapper function.
    """

    function_name = function_config.get(step_name)
    if function_name is None:
        if required:
            raise ValueError(f"Missing required function name for '{step_name}'.")
        return None
    if function_name not in PIPELINE_STEP_FUNCTIONS:
        allowed = sorted(PIPELINE_STEP_FUNCTIONS)
        raise ValueError(f"Unknown function name for '{step_name}': {function_name}. Allowed: {allowed}")
    return PIPELINE_STEP_FUNCTIONS[function_name]


def _validate_config(config: dict[str, Any]) -> None:
    """Validate the config sections needed to call PIDPipeline.

    Inputs:
        config: dict, experiment config loaded outside this function.

    Output:
        None. Raises ValueError when required sections or voxel_index are invalid.
    """

    required_sections = (
        "functions",
        "target_kwargs",
        "sources_kwargs",
        "choose_layer_kwargs",
        "feature_extraction_kwargs",
        "pid_kwargs",
    )
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    voxel_index = config["target_kwargs"].get("voxel_index")
    if isinstance(voxel_index, bool) or not isinstance(voxel_index, int):
        raise ValueError("target_kwargs['voxel_index'] must be an int.")


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


PIPELINE_STEP_FUNCTIONS: dict[str, PipelineFunction] = {
    "nsd_voxel_target": nsd_voxel_target,
    "nsd_sources": nsd_sources,
    "specific_layer_index": specific_layer_index,
    "voxel_best_layer": voxel_best_layer_for_sources,
    "nsd_feature_extraction": nsd_feature_extraction,
    "pca_each_source": pca_each_source,
    "pid_calc": pid_calc_adapter,
    "print_pid_mi": print_pid_mi_adapter,
}
