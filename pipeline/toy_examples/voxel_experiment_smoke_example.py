"""No-data smoke example for the config-driven voxel experiment runner."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    from pipeline.voxel_experiment import PIPELINE_STEP_FUNCTIONS, run_voxel_experiment
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from pipeline.voxel_experiment import PIPELINE_STEP_FUNCTIONS, run_voxel_experiment


def smoke_voxel_target(voxel_index: int, subj_id: str, n_images: int = 3) -> dict[str, Any]:
    """Create a tiny fake voxel target context.

    Inputs:
        voxel_index: int, fake selected voxel index.
        subj_id: str, fake subject identifier.
        n_images: int, number of fake samples to keep.

    Output:
        target_context: dict, fake voxel target context containing "target".
    """

    target = [[float(i + voxel_index)] for i in range(1, n_images + 1)]
    return {
        "target": target,
        "voxel_index": voxel_index,
        "subj_id": subj_id,
        "image_ids_for_subj": list(range(n_images)),
    }


def smoke_sources(model_name_1: str, model_name_2: str) -> dict[str, dict[str, Any]]:
    """Create tiny fake source contexts.

    Inputs:
        model_name_1: str, fake first model name.
        model_name_2: str, fake second model name.

    Output:
        sources: dict, fake model contexts under "X1" and "X2".
    """

    return {
        "X1": {
            "model_name": model_name_1,
            "layers": ["early", "late"],
            "features_by_layer": {
                "early": [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                "late": [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
            },
        },
        "X2": {
            "model_name": model_name_2,
            "layers": ["early", "late"],
            "features_by_layer": {
                "early": [[0.5, 5.0], [1.0, 10.0], [1.5, 15.0]],
                "late": [[2.0, 20.0], [2.5, 25.0], [3.0, 30.0]],
            },
        },
    }


def smoke_choose_layer(sources: dict[str, dict[str, Any]], X1_index: int, X2_index: int) -> dict[str, int]:
    """Choose fake layer indexes from config.

    Inputs:
        sources: dict, fake source contexts under "X1" and "X2".
        X1_index: int, selected fake X1 layer index.
        X2_index: int, selected fake X2 layer index.

    Output:
        selected_layers: dict, selected layer indexes under "X1" and "X2".
    """

    del sources
    return {"X1": X1_index, "X2": X2_index}


def smoke_feature_extraction(
    source_context: dict[str, Any],
    layer_index: int,
    target_context: dict[str, Any],
    feature_shift: float = 0.0,
) -> list[list[float]]:
    """Read fake source features for one selected layer.

    Inputs:
        source_context: dict, fake source context.
        layer_index: int, selected layer index.
        target_context: dict, fake target context used for sample count.
        feature_shift: float, value added to each feature for debugging kwargs.

    Output:
        features: list[list[float]], fake features aligned to target samples.
    """

    n_samples = len(target_context["target"])
    layer_name = source_context["layers"][layer_index]
    return [
        [value + feature_shift for value in row]
        for row in source_context["features_by_layer"][layer_name][:n_samples]
    ]


def smoke_feature_manipulation(
    source_1: list[list[float]],
    source_2: list[list[float]],
    keep_columns: int = 1,
) -> tuple[list[list[float]], list[list[float]]]:
    """Keep the first columns from both fake source matrices.

    Inputs:
        source_1: list[list[float]], first source features.
        source_2: list[list[float]], second source features.
        keep_columns: int, number of columns to keep.

    Output:
        reduced_sources: tuple, reduced source_1 and source_2.
    """

    return (
        [row[:keep_columns] for row in source_1],
        [row[:keep_columns] for row in source_2],
    )


def smoke_pid_calculation(
    target: list[list[float]],
    source_1: list[list[float]],
    source_2: list[list[float]],
    method: str = "smoke_pid",
) -> dict[str, Any]:
    """Return a tiny deterministic PID-like result.

    Inputs:
        target: list[list[float]], final target samples.
        source_1: list[list[float]], final X1 features.
        source_2: list[list[float]], final X2 features.
        method: str, fake PID method name.

    Output:
        pid_results: dict, fake PID result.
    """

    target_sum = sum(row[0] for row in target)
    x1_sum = sum(row[0] for row in source_1)
    x2_sum = sum(row[0] for row in source_2)
    red = min(x1_sum, x2_sum)
    return {
        "method": method,
        "pid": {
            "red": red,
            "unq1": x1_sum - red,
            "unq2": x2_sum - red,
            "syn": target_sum - red,
        },
        "mi": {"toy_total": target_sum + x1_sum + x2_sum},
    }


def smoke_report(pid_results: dict[str, Any], context: dict[str, Any]) -> str:
    """Print a compact smoke run report.

    Inputs:
        pid_results: dict, fake PID result.
        context: dict, full run context from run_voxel_experiment.

    Output:
        report: str, short text summary of the smoke run.
    """

    print("\nVoxel experiment smoke run")
    print("--------------------------")
    print("Selected layers:", context["selected_layers"])
    print("Target:", context["target"])
    print("X1:", context["source_1"])
    print("X2:", context["source_2"])
    print("PID result:", pid_results)
    return "voxel smoke report printed"


def register_smoke_functions() -> None:
    """Register smoke wrapper functions for this example run.

    Inputs:
        No inputs.

    Output:
        None. PIPELINE_STEP_FUNCTIONS is updated with smoke function names.
    """

    PIPELINE_STEP_FUNCTIONS.update(
        {
            "smoke_voxel_target": smoke_voxel_target,
            "smoke_sources": smoke_sources,
            "smoke_choose_layer": smoke_choose_layer,
            "smoke_feature_extraction": smoke_feature_extraction,
            "smoke_feature_manipulation": smoke_feature_manipulation,
            "smoke_pid": smoke_pid_calculation,
            "smoke_report": smoke_report,
        }
    )


def smoke_config() -> dict[str, Any]:
    """Create a small YAML-shaped config for run_voxel_experiment.

    Inputs:
        No inputs.

    Output:
        config: dict, smoke config with function names and per-step kwargs.
    """

    return {
        "functions": {
            "target_extraction": "smoke_voxel_target",
            "sources_extraction": "smoke_sources",
            "choose_layer": "smoke_choose_layer",
            "feature_extraction": "smoke_feature_extraction",
            "preprocess": None,
            "feature_manipulation": "smoke_feature_manipulation",
            "pid_calculation": "smoke_pid",
            "pid_report": "smoke_report",
        },
        "target_kwargs": {"voxel_index": 7, "subj_id": "smoke_subj", "n_images": 3},
        "sources_kwargs": {"model_name_1": "smoke_model_1", "model_name_2": "smoke_model_2"},
        "choose_layer_kwargs": {"X1_index": 0, "X2_index": 1},
        "feature_extraction_kwargs": {"feature_shift": 0.0},
        "feature_manipulation_kwargs": {"keep_columns": 1},
        "pid_kwargs": {"method": "smoke_pid_debug"},
        "report_kwargs": {},
    }


def main() -> dict[str, Any]:
    """Run the voxel experiment smoke example.

    Inputs:
        No inputs. The smoke config and functions are defined in this file.

    Output:
        context: dict, full context returned by run_voxel_experiment.
    """

    register_smoke_functions()
    return run_voxel_experiment(smoke_config())


if __name__ == "__main__":
    main()
