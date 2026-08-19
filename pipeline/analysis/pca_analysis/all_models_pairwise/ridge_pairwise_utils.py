"""Task-specific artifact, checkpoint, and prefetch helpers for ridge pairwise PID."""

from __future__ import annotations

import gc
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from external.mayas_project.features_and_encoding.feat_ext_and_encoding import (
    prepare_model_context,
)
from pipeline.analysis.anlysis_utils import to_deepdive_model_name
from pipeline.pipeline_phases.choosing_layer import overall_best_layer
from pipeline.pipeline_phases.feature_manipulations import (
    load_ridge_alphas,
    ridge_predict_shared,
)


repo_root = Path(__file__).resolve().parents[4]

PAIRWISE_RESULT_COLUMNS: list[str] = [
    "model_1",
    "model_2",
    "layer_1",
    "layer_2",
    "subj_id",
    "n_samples",
    "n_components_source_1",
    "n_components_source_2",
    "n_components_target",
    "pid_method",
    "rng_seed",
    "bias_correction",
    "red",
    "unq1",
    "unq2",
    "syn",
    "bi_mi_1",
    "bi_mi_2",
    "tri_mi",
]


@dataclass(frozen=True)
class RidgeModelArtifacts:
    """Store one model's validated layer and ridge-alpha artifact.

    Inputs:
        model_name: str identifier used by artifacts and pairwise results.
        layer_index: int selected model-layer index.
        alphas_path: Path to the per-target-PC alpha archive.
        alphas: np.ndarray containing one validated alpha per target PC.

    Outputs:
        Immutable ``RidgeModelArtifacts`` value used by model preprocessing.
    """

    model_name: str
    layer_index: int
    alphas_path: Path
    alphas: np.ndarray

def safe_model_name(model_name: str) -> str:
    """Replace path separators in a model identifier for artifact filenames.

    Inputs:
        model_name: str identifier that may contain path separators.

    Outputs:
        str identifier matching the ridge artifact-generation convention.
    """

    return str(model_name).replace("/", "_").replace("\\", "_")


def _resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a configured path relative to the repository when necessary.

    Inputs:
        path_value: str or Path that is absolute or repository-relative.

    Outputs:
        Path preserving absolute inputs and rooting relative inputs at checkout.
    """

    configured_path = Path(path_value).expanduser()
    if configured_path.is_absolute():
        return configured_path
    return repo_root / configured_path


def _load_or_create_checkpoint(output_path: Path) -> pd.DataFrame:
    """Load a compatible pairwise CSV or create an empty checkpoint.

    Inputs:
        output_path: Path used for resumable pairwise results.

    Outputs:
        pd.DataFrame containing existing rows or the required empty schema.

    Raises:
        ValueError: If an existing CSV is empty or has an incompatible header.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        try:
            existing_results = pd.read_csv(output_path)
        except pd.errors.EmptyDataError as error:
            raise ValueError(f"Existing CSV has no header: {output_path}") from error
        if list(existing_results.columns) != PAIRWISE_RESULT_COLUMNS:
            raise ValueError(
                f"Existing CSV has an incompatible schema: {output_path}. "
                f"Expected columns: {PAIRWISE_RESULT_COLUMNS}"
            )
        return existing_results

    existing_results = pd.DataFrame(columns=PAIRWISE_RESULT_COLUMNS)
    existing_results.to_csv(output_path, index=False)
    return existing_results


def _unfinished_unordered_pairs(
    source_1_names: list[str],
    source_2_names: list[str],
    existing_results: pd.DataFrame,
) -> list[tuple[str, str]]:
    """Choose the first requested orientation of every unfinished pair.

    Inputs:
        source_1_names: list[str] of ordered X1 model candidates.
        source_2_names: list[str] of ordered X2 model candidates.
        existing_results: pd.DataFrame containing completed checkpoint rows.

    Outputs:
        list[tuple[str, str]] without self, reverse, or completed duplicates.
    """

    seen_pairs = {
        frozenset((str(model_1), str(model_2)))
        for model_1, model_2 in zip(
            existing_results["model_1"],
            existing_results["model_2"],
        )
    }
    unfinished_pairs: list[tuple[str, str]] = []
    for model_1 in source_1_names:
        for model_2 in source_2_names:
            if model_1 == model_2:
                continue
            unordered_pair = frozenset((model_1, model_2))
            if unordered_pair in seen_pairs:
                continue
            unfinished_pairs.append((model_1, model_2))
            seen_pairs.add(unordered_pair)
    return unfinished_pairs


def _required_models(pairs: list[tuple[str, str]]) -> list[str]:
    """Return each model needed by pending pairs once in first-use order.

    Inputs:
        pairs: list[tuple[str, str]] containing ordered pending comparisons.

    Outputs:
        list[str] containing de-duplicated models in first-use order.
    """

    ordered_models: list[str] = []
    seen_models: set[str] = set()
    for pair in pairs:
        for model_name in pair:
            if model_name not in seen_models:
                ordered_models.append(model_name)
                seen_models.add(model_name)
    return ordered_models


def _model_artifact_path(
    artifact_config: Mapping[str, Any],
    model_name: str,
    artifact_label: str,
) -> Path:
    """Resolve and require one model-specific configured artifact path.

    Inputs:
        artifact_config: Mapping with directory and filename templates.
        model_name: str used to fill model and safe-model template fields.
        artifact_label: str used to identify the artifact in errors.

    Outputs:
        Path to the existing resolved artifact.

    Raises:
        ValueError: If template configuration is invalid.
        FileNotFoundError: If the resolved artifact does not exist.
    """

    missing_keys = {"directory_template", "filename_template"}.difference(
        artifact_config
    )
    if missing_keys:
        raise ValueError(
            f"{artifact_label} artifact config is missing keys: "
            f"{sorted(missing_keys)}"
        )
    template_values = {
        "model_name": model_name,
        "safe_model_name": safe_model_name(model_name),
    }
    try:
        directory = str(artifact_config["directory_template"]).format(**template_values)
        filename = str(artifact_config["filename_template"]).format(**template_values)
    except KeyError as error:
        raise ValueError(
            f"{artifact_label} templates contain an unsupported field: "
            f"{error.args[0]!r}. Allowed fields are model_name and "
            "safe_model_name."
        ) from error
    artifact_path = _resolve_project_path(Path(directory) / filename)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"{artifact_label} artifact for model {model_name!r} does not "
            f"exist: {artifact_path}"
        )
    return artifact_path


def _validate_model_artifacts(
    model_names: list[str],
    config: Mapping[str, Any],
    expected_target_dim: int,
) -> dict[str, RidgeModelArtifacts]:
    """Resolve and validate every required model artifact before extraction.

    Inputs:
        model_names: list[str] needed by unfinished comparisons.
        config: Mapping containing layer and artifact-template configuration.
        expected_target_dim: int number of PCA target columns and alphas.

    Outputs:
        dict[str, RidgeModelArtifacts] keyed by stored model identifier.

    Raises:
        ValueError: If layer selection or alpha metadata is incompatible.
        FileNotFoundError: If any required artifact is missing.
    """

    artifact_configs = config.get("artifact_templates")
    if not isinstance(artifact_configs, Mapping):
        raise ValueError("Config must contain an 'artifact_templates' mapping.")
    alphas_config = artifact_configs.get("ridge_alphas")
    if not isinstance(alphas_config, Mapping):
        raise ValueError(
            "artifact_templates must define a ridge_alphas mapping."
        )

    choose_layer_kwargs = config.get("choose_layer_kwargs", {})
    if "path_to_results" not in choose_layer_kwargs:
        raise ValueError("choose_layer_kwargs.path_to_results is required.")

    layer_results_path = _resolve_project_path(choose_layer_kwargs["path_to_results"])
    if not layer_results_path.is_file():
        raise FileNotFoundError(
            f"Configured best-layer results do not exist: {layer_results_path}"
        )

    artifacts_by_model: dict[str, RidgeModelArtifacts] = {}
    for model_name in model_names:
        layer_result = overall_best_layer(
            model_name=model_name,
            path_to_results=str(layer_results_path),
        )
        if layer_result["l"] is None:
            raise ValueError(f"No overall best layer found for model {model_name!r}.")
        layer_index = int(layer_result["l"])
        if layer_index < 0:
            raise ValueError(
                f"Layer index for model {model_name!r} must be nonnegative, "
                f"got {layer_index}."
            )

        alphas_path = _model_artifact_path(
            alphas_config,
            model_name,
            "ridge alphas",
        )
        alphas = load_ridge_alphas(
            alphas_path,
            model_name=model_name,
            expected_target_dim=expected_target_dim,
            expected_layer_index=layer_index,
        )
        artifacts_by_model[model_name] = RidgeModelArtifacts(
            model_name=model_name,
            layer_index=layer_index,
            alphas_path=alphas_path,
            alphas=alphas,
        )
        print(f"Validated ridge artifacts for {model_name} at layer {layer_index}.")
    return artifacts_by_model


def _load_model_context(model_name: str) -> tuple[dict[str, Any], float]:
    """Load one DeepDive model context and measure its loading duration.

    Inputs:
        model_name: str stored model identifier to load.

    Outputs:
        tuple[dict[str, Any], float] containing context and elapsed seconds.

    Raises:
        RuntimeError: If model loading or ordered-layer discovery fails.
    """

    started_at = time.perf_counter()
    try:
        model_context = prepare_model_context(to_deepdive_model_name(model_name))
    except Exception as error:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(
            f"Failed to load model context for {model_name!r}."
        ) from error
    elapsed = time.perf_counter() - started_at
    print(f"Model context load for {model_name}: {elapsed:.2f} seconds.")
    return model_context, elapsed


def _release_model_context(model_context: dict[str, Any] | None) -> None:
    """Clear one model context and release unused CPU and CUDA memory.

    Inputs:
        model_context: dict[str, Any] context, or None when none is live.

    Outputs:
        None. Context references and unused CUDA cache are released.
    """

    if model_context is not None:
        model_context.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _prepare_ridge_prediction(
    model_name: str,
    artifacts: RidgeModelArtifacts,
    target_context: dict[str, Any],
    train_target: np.ndarray,
    shared_mask: np.ndarray,
    feature_extraction: Callable[..., Any],
    feature_extraction_kwargs: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Create one model's held-out ridge prediction and release intermediates.

    Inputs:
        model_name: str identifying the model to preprocess.
        artifacts: RidgeModelArtifacts containing its layer and raw-feature
            ridge alphas.
        target_context: dict[str, Any] used by aligned feature extraction.
        train_target: np.ndarray containing non-shared PCA target rows.
        shared_mask: np.ndarray selecting held-out shared rows.
        feature_extraction: Callable producing one model's aligned features.
        feature_extraction_kwargs: Mapping passed to feature extraction.
        seed: int ridge random seed.

    Outputs:
        tuple[np.ndarray, int] containing the shared prediction and layer index.

    Raises:
        RuntimeError: If model loading fails.
        ValueError: If a selected model layer is unavailable.
        Exception: Propagates extraction and ridge failures after cleanup.
    """

    current_context: dict[str, Any] | None = None
    raw_features: Any = None
    try:
        current_context, _ = _load_model_context(model_name)
        layers_ordered = current_context.get("layers_ordered", [])
        if artifacts.layer_index >= len(layers_ordered):
            raise ValueError(
                f"Selected layer {artifacts.layer_index} for model "
                f"{model_name!r} is outside its {len(layers_ordered)} "
                "discovered layers."
            )

        extraction_started_at = time.perf_counter()
        raw_features = feature_extraction(
            source_context=current_context,
            layer_index=artifacts.layer_index,
            target_context=target_context,
            **dict(feature_extraction_kwargs),
        )
        print(
            f"Feature extraction for {model_name}: "
            f"{time.perf_counter() - extraction_started_at:.2f} seconds."
        )
    finally:
        _release_model_context(current_context)

    ridge_started_at = time.perf_counter()
    try:
        prediction = ridge_predict_shared(
            np.asarray(raw_features),
            train_target,
            shared_mask,
            artifacts.alphas,
            seed=seed,
        )
    finally:
        raw_features = None
    print(
        f"Ridge fitting and held-out prediction for {model_name}: "
        f"{time.perf_counter() - ridge_started_at:.2f} seconds."
    )
    return np.asarray(prediction), artifacts.layer_index


def _iter_ridge_prediction_pairs(
    pairs: list[tuple[str, str]],
    artifacts_by_model: Mapping[str, RidgeModelArtifacts],
    target_context: dict[str, Any],
    train_target: np.ndarray,
    shared_mask: np.ndarray,
    feature_extraction: Callable[..., Any],
    feature_extraction_kwargs: Mapping[str, Any],
    *,
    seed: int,
    prefetch_ridge_predictions: bool,
) -> Iterator[tuple[str, str, np.ndarray, int, np.ndarray, int]]:
    """Yield PID-ready pairs while preprocessing the next model in one worker.

    Inputs:
        pairs: list[tuple[str, str]] of unfinished ordered comparisons.
        artifacts_by_model: Mapping with validated artifacts for required models.
        target_context: dict[str, Any] used by feature extraction.
        train_target: np.ndarray containing non-shared PCA target rows.
        shared_mask: np.ndarray selecting held-out shared rows.
        feature_extraction: Callable producing aligned model features.
        feature_extraction_kwargs: Mapping passed to feature extraction.
        seed: int ridge random seed.
        prefetch_ridge_predictions: bool enabling full next-model preprocessing.

    Outputs:
        Iterator yielding model names, predictions, and layers for each PID pair.
        The next unseen model is loaded, extracted, and ridged without variance
        standardization in the background while the caller evaluates the
        currently yielded pair.

    Raises:
        RuntimeError: If a background model-preprocessing task fails.
        Exception: Propagates synchronous preprocessing failures after cleanup.
    """

    prediction_cache: dict[str, tuple[np.ndarray, int]] = {}
    executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="ridge-prediction")
        if prefetch_ridge_predictions
        else None
    )
    prefetch_future: Future[tuple[np.ndarray, int]] | None = None
    prefetched_model_name: str | None = None

    try:
        for pair_index, (model_1, model_2) in enumerate(pairs):
            for needed_model_name in (model_1, model_2):
                if needed_model_name in prediction_cache:
                    continue
                if prefetch_future is not None:
                    if prefetched_model_name is None:
                        raise RuntimeError(
                            "A ridge prefetch future has no associated model name."
                        )
                    future_model_name = prefetched_model_name
                    wait_started_at = time.perf_counter()
                    try:
                        prediction_cache[future_model_name] = prefetch_future.result()
                    except Exception as error:
                        raise RuntimeError(
                            "Prefetched ridge prediction failed for "
                            f"{future_model_name!r}."
                        ) from error
                    print(
                        f"Wait for prefetched ridge prediction "
                        f"{future_model_name}: "
                        f"{time.perf_counter() - wait_started_at:.2f} seconds."
                    )
                    prefetch_future = None
                    prefetched_model_name = None
                    if needed_model_name in prediction_cache:
                        continue
                prediction_cache[needed_model_name] = _prepare_ridge_prediction(
                    needed_model_name,
                    artifacts_by_model[needed_model_name],
                    target_context,
                    train_target,
                    shared_mask,
                    feature_extraction,
                    feature_extraction_kwargs,
                    seed=seed,
                )

            if executor is not None and prefetch_future is None:
                for future_pair in pairs[pair_index + 1 :]:
                    next_model_name = next(
                        (
                            model_name
                            for model_name in future_pair
                            if model_name not in prediction_cache
                        ),
                        None,
                    )
                    if next_model_name is None:
                        continue
                    prefetched_model_name = next_model_name
                    prefetch_future = executor.submit(
                        _prepare_ridge_prediction,
                        next_model_name,
                        artifacts_by_model[next_model_name],
                        target_context,
                        train_target,
                        shared_mask,
                        feature_extraction,
                        feature_extraction_kwargs,
                        seed=seed,
                    )
                    break

            source_1, layer_1 = prediction_cache[model_1]
            source_2, layer_2 = prediction_cache[model_2]
            yield model_1, model_2, source_1, layer_1, source_2, layer_2
    finally:
        if prefetch_future is not None:
            prefetch_future.cancel()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
