"""Config-driven full-OTC experiment runner built on PIDPipeline."""

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
)


def run_otc_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run one full-OTC experiment from an already-loaded config dictionary.

    Inputs:
        config: dict, YAML-loaded experiment config with function names and
            kwargs sections for PIDPipeline.run.

    Output:
        context: dict, full context returned by PIDPipeline.run.
    """

    _validate_config(config)
    return run_configured_pid_pipeline(
        config=config,
        function_registry=PIPELINE_STEP_FUNCTIONS,
    )


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
    from pipeline.pipeline_phases.sources_target_features import prepare_target

    target_context = prepare_target(
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
