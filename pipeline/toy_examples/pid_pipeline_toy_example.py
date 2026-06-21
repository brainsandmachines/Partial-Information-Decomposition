"""Tiny no-data-loading example for debugging the strict PID pipeline flow."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    from pipeline.pid_pipeline import PIDPipeline, PIDPipelineFunctions
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from pipeline.pid_pipeline import PIDPipeline, PIDPipelineFunctions


def toy_target_extraction() -> dict[str, Any]:
    """Create tiny fake target data.

    Inputs:
        No inputs.

    Output:
        target_context: dict, fake target context containing "target".
    """

    return {"target": [[1.0], [2.0], [3.0]], "target_name": "toy_neural_response"}


def toy_sources_extraction(model_1: str, model_2: str) -> dict[str, dict[str, Any]]:
    """Create tiny fake source contexts with two layers per source.

    Inputs:
        model_1: str, name of the first fake model.
        model_2: str, name of the second fake model.

    Output:
        sources: dict, fake source contexts under "X1" and "X2".
    """

    return {
        "X1": {
            "model_name": model_1,
            "layers": ["early", "late"],
            "features_by_layer": {
                "early": [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                "late": [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
            },
        },
        "X2": {
            "model_name": model_2,
            "layers": ["early", "late"],
            "features_by_layer": {
                "early": [[0.5, 5.0], [1.0, 10.0], [1.5, 15.0]],
                "late": [[2.0, 20.0], [2.5, 25.0], [3.0, 30.0]],
            },
        },
    }


def toy_choose_layer(sources: dict[str, dict[str, Any]], layer_index: int = 0) -> dict[str, str]:
    """Choose one layer for each source by index.

    Inputs:
        sources: dict, source contexts under "X1" and "X2".
        layer_index: int, selected layer index for both sources.

    Output:
        selected_layers: dict, selected layer names under "X1" and "X2".
    """

    return {
        "X1": sources["X1"]["layers"][layer_index],
        "X2": sources["X2"]["layers"][layer_index],
    }


def toy_feature_extraction(
    source_context: dict[str, Any],
    layer_name: str,
    target_context: dict[str, Any],
) -> list[list[float]]:
    """Read fake features for one selected source layer.

    Inputs:
        source_context: dict, fake source context containing features_by_layer.
        layer_name: str, selected layer name.
        target_context: dict, target context passed through by the pipeline.

    Output:
        features: list[list[float]], fake source features for the selected layer.
    """

    del target_context
    return source_context["features_by_layer"][layer_name]


def toy_preprocess(
    target: list[list[float]],
    source_1: list[list[float]],
    source_2: list[list[float]],
    scale: float = 1.0,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """Scale target and source values together as a visible preprocessing step.

    Inputs:
        target: list[list[float]], target samples.
        source_1: list[list[float]], first source samples.
        source_2: list[list[float]], second source samples.
        scale: float, scalar multiplier applied to every value.

    Output:
        processed: tuple, scaled target, source_1, and source_2.
    """

    return (
        [[value * scale for value in row] for row in target],
        [[value * scale for value in row] for row in source_1],
        [[value * scale for value in row] for row in source_2],
    )


def toy_feature_manipulation(
    source_1: list[list[float]],
    source_2: list[list[float]],
    keep_columns: int = 1,
) -> tuple[list[list[float]], list[list[float]]]:
    """Keep the first columns from both source feature matrices.

    Inputs:
        source_1: list[list[float]], first source feature matrix.
        source_2: list[list[float]], second source feature matrix.
        keep_columns: int, number of columns to keep.

    Output:
        reduced_sources: tuple, reduced source_1 and source_2 feature matrices.
    """

    return (
        [row[:keep_columns] for row in source_1],
        [row[:keep_columns] for row in source_2],
    )


def toy_pid_calculation(
    target: list[list[float]],
    source_1: list[list[float]],
    source_2: list[list[float]],
    method_name: str = "toy_pid",
) -> dict[str, Any]:
    """Return a readable dummy PID result from tiny arrays.

    Inputs:
        target: list[list[float]], final target samples.
        source_1: list[list[float]], final X1 features.
        source_2: list[list[float]], final X2 features.
        method_name: str, label stored in the dummy result.

    Output:
        pid_results: dict, dummy PID-like values and method metadata.
    """

    target_total = sum(row[0] for row in target)
    x1_total = sum(row[0] for row in source_1)
    x2_total = sum(row[0] for row in source_2)
    red = min(x1_total, x2_total)
    return {
        "method": method_name,
        "red": red,
        "unq1": x1_total - red,
        "unq2": x2_total - red,
        "syn": target_total - red,
        "note": "Toy numbers only; this is not a scientific PID estimator.",
    }


def toy_pid_report(pid_results: dict[str, Any], context: dict[str, Any]) -> str:
    """Print a compact toy pipeline report.

    Inputs:
        pid_results: dict, dummy PID-like results.
        context: dict, full pipeline context from the toy run.

    Output:
        report: str, short text summary of the toy result.
    """

    print("\nToy PID pipeline run")
    print("--------------------")
    print("Selected layers:", context["selected_layers"])
    print("Target after preprocessing:", context["target"])
    print("X1 after manipulation:", context["source_1"])
    print("X2 after manipulation:", context["source_2"])
    print("PID results:", pid_results)
    return "toy report printed"


def main() -> dict[str, Any]:
    """Run the tiny toy PID pipeline.

    Inputs:
        No inputs. All toy data and functions are defined in this file.

    Output:
        context: dict, full PIDPipeline context after the toy run.
    """

    functions = PIDPipelineFunctions(
        target_extraction=toy_target_extraction,
        sources_extraction=toy_sources_extraction,
        choose_layer=toy_choose_layer,
        feature_extraction=toy_feature_extraction,
        preprocess=toy_preprocess,
        feature_manipulation=toy_feature_manipulation,
        pid_calculation=toy_pid_calculation,
        pid_report=toy_pid_report,
    )
    pipeline = PIDPipeline(functions)

    return pipeline.run(
        sources_kwargs={"model_1": "toy_model_1", "model_2": "toy_model_2"},
        choose_layer_kwargs={"layer_index": 0},
        preprocess_kwargs={"scale": 1.0},
        feature_manipulation_kwargs={"keep_columns": 1},
        pid_kwargs={"method_name": "toy_pid_debug"},
    )


if __name__ == "__main__":
    main()
