"""Config-driven voxel experiment runner built on PIDPipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pipeline.pipeline_utils import (
    COMMON_PIPELINE_STEP_FUNCTIONS,
    PipelineFunction,
    run_configured_pid_pipeline,
    validate_pipeline_config_sections,
    voxel_best_layer_for_sources,
)

from pipeline.pipeline_phases.sources_target_features import prepare_target_for_voxel

def run_voxel_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run one voxel experiment from an already-loaded config dictionary.

    Inputs:
        config: dict, YAML-loaded experiment config with function names and
            kwargs sections for PIDPipeline.run.

    Output:
        context: dict, full context returned by PIDPipeline.run.
    """

    _validate_config(config)
    choose_layer_kwargs = _choose_layer_kwargs(config)
    return run_configured_pid_pipeline(
        config=config,
        function_registry=PIPELINE_STEP_FUNCTIONS,
        choose_layer_kwargs=choose_layer_kwargs,
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
        subj_id: str, subject identifier kept for config symmetry.
        hdf_path: str or Path, path to the NSD stimulus HDF5 file.
        pkl_info_path: str or Path, path to the NSD stimulus metadata pickle.
        neural_data_path: str or Path, path to the neural data array.
        n_images: int or None, optional number of images/samples to keep.

    Output:
        target_context: dict, target helper output with an added "target" key.
    """

    if voxel_index is None:
        raise ValueError("target_kwargs['voxel_index'] must be an int for run_voxel_experiment.")


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


def _choose_layer_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Arrange voxel-specific kwargs for the configured choose_layer function.

    Inputs:
        config: dict, validated voxel experiment config.

    Output:
        choose_layer_kwargs: dict, kwargs passed to PIDPipeline.run.
    """

    choose_layer_kwargs = dict(config.get("choose_layer_kwargs", {}))
    # if config["functions"].get("choose_layer") == "voxel_best_layer":
    #     choose_layer_kwargs.setdefault("voxel_index", config["target_kwargs"]["voxel_index"])
    return choose_layer_kwargs


def _validate_config(config: dict[str, Any]) -> None:
    """Validate the config sections needed to call the voxel runner.

    Inputs:
        config: dict, experiment config loaded outside this function.

    Output:
        None. Raises ValueError when required sections or voxel_index are invalid.
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
    voxel_index = config["target_kwargs"].get("voxel_index")
    if isinstance(voxel_index, bool) or not isinstance(voxel_index, int):
        raise ValueError("target_kwargs['voxel_index'] must be an int.")


PIPELINE_STEP_FUNCTIONS: dict[str, PipelineFunction] = {
    **COMMON_PIPELINE_STEP_FUNCTIONS,
    "nsd_voxel_target": nsd_voxel_target,
    "voxel_best_layer": voxel_best_layer_for_sources,
}
