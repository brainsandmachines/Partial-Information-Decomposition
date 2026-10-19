"""Shared helpers for pairwise PID and cumulative target-PC analyses.

Checkpointing, unordered-pair bookkeeping, model loading, and next-model
prefetching are shared by both source modes. Each mode supplies one source
preparer that turns a model name into a held-out array and its layer index:

- prediction: best-layer features -> per-target-PC ridge -> shared-image
  predictions of the target PCs.
- representation: best-layer features -> optional SRP -> PCA fitted on the
  unique images -> shared-image projections.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from external.mayas_project.features_and_encoding.feat_ext_and_encoding import (
    batch_process, get_layer_feature_count, get_sparse_projection_gpu, prepare_model_context)
from pipeline.analysis.utils import to_deepdive_model_name
from pipeline.phases.choosing_layer import overall_best_layer
from pipeline.phases.feature_manipulations import load_ridge_alphas, ridge_predict_shared
from pipeline.phases.sources_target_features import batching
from pipeline.utils import resolve_pipeline_function


repo_root = Path(__file__).resolve().parents[3]

PAIRWISE_RESULT_COLUMNS: list[str] = ["model_1", "model_2", "layer_1", "layer_2", "subj_id",
                                      "n_samples", "n_components_source_1",
                                      "n_components_source_2", "n_components_target",
                                      "pid_method", "rng_seed", "bias_correction",
                                      "red", "unq1", "unq2", "syn",
                                      "bi_mi_1", "bi_mi_2", "tri_mi"]

# A source preparer maps one model name to (held-out source array, layer index).
SourcePreparer = Callable[[str], tuple[np.ndarray, int]]
# One PID-ready pair: (model_1, model_2, source_1, layer_1, source_2, layer_2).
PreparedPair = tuple[str, str, np.ndarray, int, np.ndarray, int]


# ---------------------------------------------------------------------------
# Paths, checkpoints, and pair bookkeeping (both modes)
# ---------------------------------------------------------------------------


def safe_model_name(model_name: str) -> str:
    """Replace path separators in a model identifier for artifact filenames.

    Inputs:
        model_name: str identifier that may contain path separators.

    Outputs:
        str identifier matching the ridge artifact-generation convention.
    """

    return str(model_name).replace("/", "_").replace("\\", "_")


def pop_raw_target(target_context: dict[str, Any]) -> np.ndarray:
    """Remove and return the full neural target from a target context.

    Inputs:
        target_context: dict[str, Any] returned by target extraction; it must
            contain "target" and may contain "neural_data".

    Outputs:
        np.ndarray raw target with shape (n_images, n_voxels). Both large
        arrays are removed from target_context so only the caller keeps them.

    Raises:
        ValueError: If target_context has no "target" array.
    """

    if "target" not in target_context:
        raise ValueError("Target extraction context is missing the 'target' array.")
    raw_target = np.asarray(target_context.pop("target"))
    target_context.pop("neural_data", None)
    return raw_target


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
            raise ValueError(f"Existing CSV has an incompatible schema: {output_path}. "
                             f"Expected columns: {PAIRWISE_RESULT_COLUMNS}")
        return existing_results

    existing_results = pd.DataFrame(columns=PAIRWISE_RESULT_COLUMNS)
    existing_results.to_csv(output_path, index=False)
    return existing_results


def _unfinished_unordered_pairs(source_1_names: list[str], source_2_names: list[str],
                                existing_results: pd.DataFrame) -> list[tuple[str, str]]:
    """Choose the first requested orientation of every unfinished pair.

    Inputs:
        source_1_names: list[str] of ordered X1 model candidates.
        source_2_names: list[str] of ordered X2 model candidates.
        existing_results: pd.DataFrame containing completed checkpoint rows.

    Outputs:
        list[tuple[str, str]] without self, reverse, or completed duplicates.
    """

    seen_pairs = {frozenset((str(model_1), str(model_2)))
                  for model_1, model_2 in zip(existing_results["model_1"],
                                              existing_results["model_2"])}
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


def required_models(pairs: list[tuple[str, str]]) -> list[str]:
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


def _result_row(model_1: str, model_2: str, layer_1: int, layer_2: int,
                target: np.ndarray, source_1: np.ndarray, source_2: np.ndarray,
                pid_results: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Build one CSV row for a finished model pair.

    Inputs:
        model_1: str model used as source X1.
        model_2: str model used as source X2.
        layer_1: int selected layer index of model_1.
        layer_2: int selected layer index of model_2.
        target: np.ndarray held-out target with shape (n_shared, d_T).
        source_1: np.ndarray held-out X1 with shape (n_shared, d_1).
        source_2: np.ndarray held-out X2 with shape (n_shared, d_2).
        pid_results: Mapping with "pid" (red, unq1, unq2, syn), "mi"
            (bi_mi_1, bi_mi_2, tri_mi), and optionally "method".
        config: Mapping containing "metadata" and "pid_kwargs" sections.

    Outputs:
        dict[str, Any] keyed exactly by PAIRWISE_RESULT_COLUMNS.
    """

    pid_kwargs = config.get("pid_kwargs") or {}
    pid = pid_results["pid"]
    mi = pid_results["mi"]
    return {"model_1": model_1,
            "model_2": model_2,
            "layer_1": layer_1,
            "layer_2": layer_2,
            "subj_id": (config.get("metadata") or {}).get("subj_id"),
            "n_samples": target.shape[0],
            "n_components_source_1": source_1.shape[1],
            "n_components_source_2": source_2.shape[1],
            "n_components_target": target.shape[1],
            "pid_method": pid_results.get("method", pid_kwargs.get("method")),
            "rng_seed": pid_kwargs.get("rng_seed"),
            "bias_correction": (pid_kwargs.get("config") or {}).get("bias_correction"),
            "red": pid["red"],
            "unq1": pid["unq1"],
            "unq2": pid["unq2"],
            "syn": pid["syn"],
            "bi_mi_1": mi["bi_mi_1"],
            "bi_mi_2": mi["bi_mi_2"],
            "tri_mi": mi["tri_mi"]}


# ---------------------------------------------------------------------------
# Configuration checks (run before the slow NSD target load)
# ---------------------------------------------------------------------------


def _layer_results_path(config: Mapping[str, Any]) -> Path:
    """Resolve and require the configured best-layer results CSV.

    Inputs:
        config: Mapping containing choose_layer_kwargs.path_to_results.

    Outputs:
        Path to the existing best-layer CSV.

    Raises:
        ValueError: If path_to_results is not configured.
        FileNotFoundError: If the resolved CSV does not exist.
    """

    choose_layer_kwargs = config.get("choose_layer_kwargs") or {}
    if "path_to_results" not in choose_layer_kwargs:
        raise ValueError("choose_layer_kwargs.path_to_results is required.")

    layer_results_path = _resolve_project_path(choose_layer_kwargs["path_to_results"])
    if not layer_results_path.is_file():
        raise FileNotFoundError(f"Configured best-layer results do not exist: {layer_results_path}")
    return layer_results_path


def _check_prediction_config(config: dict[str, Any],
                             registry: dict[str, Any]) -> tuple[Callable[..., Any], Path]:
    """Resolve prediction-mode inputs that can be checked before loading data.

    Inputs:
        config: dict[str, Any] with functions.feature_extraction and
            feature_manipulation_kwargs.pc_target_path.
        registry: dict[str, Any] mapping configured names to callables.

    Outputs:
        tuple of the feature-extraction callable and the existing target-PCA
        artifact Path.

    Raises:
        RuntimeError: If feature extraction does not resolve.
        FileNotFoundError: If the target-PCA artifact does not exist.
    """

    feature_extraction = resolve_pipeline_function(config["functions"], registry,
                                                   "feature_extraction", required=True)
    if feature_extraction is None:
        raise RuntimeError("Required configured pipeline functions did not resolve.")

    manipulation_kwargs = config.get("feature_manipulation_kwargs", {})
    pc_target_path = _resolve_project_path(manipulation_kwargs["pc_target_path"])
    if not pc_target_path.is_file():
        raise FileNotFoundError(f"Configured target PCA artifact does not exist: {pc_target_path}")
    return feature_extraction, pc_target_path


def _check_representation_config(config: dict[str, Any]) -> int:
    """Check representation-mode PCA sizes before loading data.

    Inputs:
        config: dict[str, Any] with feature_manipulation_kwargs containing
            n_components_source_1 and n_components_source_2.

    Outputs:
        int number of PCA components shared by both sources.

    Raises:
        ValueError: If the two sources request different PCA sizes.
    """

    manipulation_kwargs = config.get("feature_manipulation_kwargs", {})
    n_source_components = int(manipulation_kwargs["n_components_source_1"])
    if n_source_components != int(manipulation_kwargs["n_components_source_2"]):
        raise ValueError("Memory-cached pairwise comparisons require equal PCA dimensions "
                         "for source 1 and source 2.")
    return n_source_components


# ---------------------------------------------------------------------------
# Layer selection and model contexts (both modes)
# ---------------------------------------------------------------------------


def _best_layer_index(model_name: str, layer_results_path: Path) -> int:
    """Look up one model's overall best layer index.

    Inputs:
        model_name: str stored model identifier used in the best-layer CSV.
        layer_results_path: Path to the existing best-layer CSV.

    Outputs:
        int nonnegative layer index.

    Raises:
        ValueError: If the model has no best layer or a negative index.
    """

    layer_result = overall_best_layer(model_name=model_name,
                                      path_to_results=str(layer_results_path))
    if layer_result["l"] is None:
        raise ValueError(f"No overall best layer found for model {model_name!r}.")
    layer_index = int(layer_result["l"])
    if layer_index < 0:
        raise ValueError(f"Layer index for model {model_name!r} must be nonnegative, "
                         f"got {layer_index}.")
    return layer_index


def _check_layer_index(model_context: Mapping[str, Any], model_name: str, layer_index: int) -> None:
    """Require the selected layer to exist in a loaded model.

    Inputs:
        model_context: Mapping containing the model's "layers_ordered" list.
        model_name: str model identifier used in the error message.
        layer_index: int selected layer index.

    Outputs:
        None.

    Raises:
        ValueError: If layer_index is outside the discovered layers.
    """

    layers_ordered = model_context.get("layers_ordered", [])
    if layer_index >= len(layers_ordered):
        raise ValueError(f"Selected layer {layer_index} for model {model_name!r} is outside "
                         f"its {len(layers_ordered)} discovered layers.")


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
        raise RuntimeError(f"Failed to load model context for {model_name!r}.") from error
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


# ---------------------------------------------------------------------------
# Prediction mode: held-out ridge predictions of the saved target PCs
# ---------------------------------------------------------------------------


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


def _model_artifact_path(artifact_config: Mapping[str, Any], model_name: str,
                         artifact_label: str) -> Path:
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

    missing_keys = {"directory_template", "filename_template"}.difference(artifact_config)
    if missing_keys:
        raise ValueError(f"{artifact_label} artifact config is missing keys: "
                         f"{sorted(missing_keys)}")
    template_values = {"model_name": model_name, "safe_model_name": safe_model_name(model_name)}
    try:
        directory = str(artifact_config["directory_template"]).format(**template_values)
        filename = str(artifact_config["filename_template"]).format(**template_values)
    except KeyError as error:
        raise ValueError(f"{artifact_label} templates contain an unsupported field: "
                         f"{error.args[0]!r}. Allowed fields are model_name and "
                         "safe_model_name.") from error
    artifact_path = _resolve_project_path(Path(directory) / filename)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"{artifact_label} artifact for model {model_name!r} does not "
                                f"exist: {artifact_path}")
    return artifact_path


def _validate_model_artifacts(model_names: list[str], config: Mapping[str, Any],
                              layer_results_path: Path,
                              expected_target_dim: int) -> dict[str, RidgeModelArtifacts]:
    """Resolve and validate every required model artifact before extraction.

    Inputs:
        model_names: list[str] needed by unfinished comparisons.
        config: Mapping containing the artifact_templates configuration.
        layer_results_path: Path to the existing best-layer CSV.
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
        raise ValueError("artifact_templates must define a ridge_alphas mapping.")

    artifacts_by_model: dict[str, RidgeModelArtifacts] = {}
    for model_name in model_names:
        layer_index = _best_layer_index(model_name, layer_results_path)
        alphas_path = _model_artifact_path(alphas_config, model_name, "ridge alphas")
        alphas = load_ridge_alphas(alphas_path, model_name=model_name,
                                   expected_target_dim=expected_target_dim,
                                   expected_layer_index=layer_index)
        artifacts_by_model[model_name] = RidgeModelArtifacts(model_name=model_name,
                                                             layer_index=layer_index,
                                                             alphas_path=alphas_path, alphas=alphas)
        print(f"Validated ridge artifacts for {model_name} at layer {layer_index}.")
    return artifacts_by_model


def _prepare_ridge_prediction(model_name: str, *,
                              artifacts_by_model: Mapping[str, RidgeModelArtifacts],
                              target_context: dict[str, Any], train_target: np.ndarray,
                              shared_mask: np.ndarray, feature_extraction: Callable[..., Any],
                              feature_extraction_kwargs: Mapping[str, Any],
                              seed: int) -> tuple[np.ndarray, int]:
    """Create one model's held-out ridge prediction and release intermediates.

    This is the prediction-mode source preparer; bind every keyword argument
    with ``functools.partial`` to obtain a ``SourcePreparer``.

    Inputs:
        model_name: str identifying the model to preprocess.
        artifacts_by_model: Mapping from model name to RidgeModelArtifacts
            containing its layer and raw-feature ridge alphas.
        target_context: dict[str, Any] used by aligned feature extraction.
        train_target: np.ndarray containing non-shared PCA target rows.
        shared_mask: np.ndarray selecting held-out shared rows.
        feature_extraction: Callable producing one model's aligned features.
        feature_extraction_kwargs: Mapping passed to feature extraction.
        seed: int ridge random seed.

    Outputs:
        tuple[np.ndarray, int] containing the shared prediction with shape
        (n_shared, n_target_pcs) and the layer index.

    Raises:
        RuntimeError: If model loading fails.
        ValueError: If a selected model layer is unavailable.
        Exception: Propagates extraction and ridge failures after cleanup.
    """

    artifacts = artifacts_by_model[model_name]
    current_context: dict[str, Any] | None = None
    raw_features: Any = None
    try:
        current_context, _ = _load_model_context(model_name)
        _check_layer_index(current_context, model_name, artifacts.layer_index)

        extraction_started_at = time.perf_counter()
        raw_features = feature_extraction(source_context=current_context,
                                          layer_index=artifacts.layer_index,
                                          target_context=target_context,
                                          **dict(feature_extraction_kwargs))
        print(f"Feature extraction for {model_name}: "
              f"{time.perf_counter() - extraction_started_at:.2f} seconds.")
    finally:
        _release_model_context(current_context)

    ridge_started_at = time.perf_counter()
    try:
        prediction = ridge_predict_shared(np.asarray(raw_features), train_target, shared_mask,
                                          artifacts.alphas, seed=seed)
        # (n_images, n_features) -> (n_shared, n_target_pcs)
    finally:
        raw_features = None
    print(f"Ridge fitting and held-out prediction for {model_name}: "
          f"{time.perf_counter() - ridge_started_at:.2f} seconds.")
    return np.asarray(prediction), artifacts.layer_index


def prepare_prediction_inputs(
    raw_target: np.ndarray,
    target_context: dict[str, Any],
    config: Mapping[str, Any],
    required_models: list[str],
    layer_results_path: Path,
    feature_extraction: Callable[..., Any],
    pc_target_path: Path,
    seed: int,
    *,
    target_preparation: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, SourcePreparer]:
    """Prepare a held-out PCA target and reusable ridge source preparer.

    Inputs:
        raw_target: np.ndarray neural target with shape (n_images, n_voxels).
        target_context: dict[str, Any] containing the aligned shared-image mask
            and stimulus information used during feature extraction.
        config: Mapping[str, Any] containing artifact_templates and
            feature_extraction_kwargs.
        required_models: list[str] containing every model that will be used.
        layer_results_path: Path to the existing best-layer CSV.
        feature_extraction: Callable producing one model's aligned features.
        pc_target_path: Path to the saved target PCA artifact.
        seed: int ridge random seed.
        target_preparation: Callable projecting and splitting the raw target.

    Outputs:
        tuple containing the held-out target with shape
        (n_shared, n_target_pcs) and a SourcePreparer returning each model's
        held-out ridge prediction in the same target-PC space.
    """

    train_target, test_target, shared_mask = target_preparation(
        raw_target, target_context, pc_target_path)
    # (n_images, n_voxels) -> (n_unique, n_target_pcs), (n_shared, n_target_pcs)
    artifacts_by_model = _validate_model_artifacts(
        required_models, config, layer_results_path,
        expected_target_dim=test_target.shape[1])
    prepare_source = partial(
        _prepare_ridge_prediction,
        artifacts_by_model=artifacts_by_model,
        target_context=target_context,
        train_target=train_target,
        shared_mask=shared_mask,
        feature_extraction=feature_extraction,
        feature_extraction_kwargs=config.get("feature_extraction_kwargs", {}),
        seed=seed,
    )
    return test_target, prepare_source


# ---------------------------------------------------------------------------
# Representation mode: PCA projections of best-layer features
# ---------------------------------------------------------------------------


def _fit_unique_project_shared(data: np.ndarray, shared_mask: np.ndarray, n_components: int,
                               random_state: int) -> np.ndarray:
    """Fit PCA on the unique-image rows and project the shared-image rows.

    Used for both the neural target and the model features in representation
    mode, so neither PCA ever sees the held-out shared images.

    Inputs:
        data: np.ndarray with shape (n_images, n_features).
        shared_mask: np.ndarray bool with shape (n_images,); True marks the
            held-out shared images.
        n_components: int number of principal components to keep.
        random_state: int seed for the randomized SVD solver.

    Outputs:
        np.ndarray float64 with shape (n_shared, n_components).
    """

    data = np.asarray(data)
    shared_mask = np.asarray(shared_mask, dtype=bool)

    unique_rows = np.asarray(data[~shared_mask], dtype=np.float64)
    # (n_images, n_features) -> (n_unique, n_features)
    shared_rows = data[shared_mask]
    # (n_images, n_features) -> (n_shared, n_features)

    pca = PCA(n_components=int(n_components), svd_solver="randomized",
              random_state=random_state, copy=False)
    pca.fit(unique_rows)
    projected = np.asarray(pca.transform(shared_rows), dtype=np.float64)
    # (n_shared, n_features) -> (n_shared, n_components)
    return projected


def _extract_layer_features(model_context: dict[str, Any], layer_name: str,
                            target_context: Mapping[str, Any],
                            feature_extraction_kwargs: Mapping[str, Any]) -> np.ndarray:
    """Extract one layer's features for every image, optionally reduced by SRP.

    Inputs:
        model_context: dict[str, Any] with "model" and "image_transforms".
        layer_name: str name of the selected layer.
        target_context: Mapping with "stim", "image_ids_for_subj", and, when
            SRP is enabled without a configured size, "n_projections".
        feature_extraction_kwargs: Mapping with batch_size_process,
            batch_size_dataloader, and optional use_srp and srp_n_components.

    Outputs:
        np.ndarray float64 with shape (n_images, n_features). n_features is
        the layer size, or the SRP size when SRP is enabled and smaller.

    Raises:
        ValueError: If SRP is enabled without a positive projection size.
    """

    use_srp = bool(feature_extraction_kwargs.get("use_srp", False))
    srp_components = feature_extraction_kwargs.get("srp_n_components")
    if use_srp and srp_components is None:
        srp_components = target_context.get("n_projections")
    if use_srp and (srp_components is None or int(srp_components) < 1):
        raise ValueError("srp_n_components must be a positive integer when use_srp is true.")

    model = model_context["model"]
    raw_dimension = int(get_layer_feature_count(model, layer_name))
    n_features = min(int(srp_components), raw_dimension) if use_srp else raw_dimension
    n_images = len(target_context["image_ids_for_subj"])
    batch_size_process = int(feature_extraction_kwargs["batch_size_process"])
    batch_size_dataloader = int(feature_extraction_kwargs["batch_size_dataloader"])

    sparse_projection = None
    if raw_dimension > n_features:
        model_device = str(next(model.parameters()).device)
        print(f"Using sparse projection {raw_dimension} -> {n_features} on device {model_device}")
        sparse_projection = get_sparse_projection_gpu(raw_dimension, n_features,
                                                      device=model_device)

    features = np.empty((n_images, n_features), dtype=np.float64)
    for batch_start in range(0, n_images, batch_size_process):
        batch_end = min(batch_start + batch_size_process, n_images)
        if sparse_projection is None:
            batch_features = batching(model_context=model_context, batch_start=batch_start,
                                      batch_end=batch_end, stim_dataset=target_context["stim"],
                                      subj_image_ids=target_context["image_ids_for_subj"],
                                      layer_name=layer_name,
                                      batch_size_dataloader=batch_size_dataloader)
        else:
            batch_features = batch_process(model=model, batch_start=batch_start,
                                           batch_end=batch_end,
                                           batch_size_process=batch_size_process,
                                           batch_size_dataloader=batch_size_dataloader,
                                           layer_name=layer_name,
                                           sparse_projection_mat_gpu=sparse_projection,
                                           stim_dataset=target_context["stim"],
                                           image_ids_for_subj=target_context["image_ids_for_subj"],
                                           image_transforms=model_context["image_transforms"])
        features[batch_start:batch_end] = batch_features
        # (batch_end - batch_start, n_features) -> rows of (n_images, n_features)
        del batch_features
    return features


def _prepare_representation_projection(model_name: str, *, target_context: dict[str, Any],
                                       layer_results_path: Path,
                                       feature_extraction_kwargs: Mapping[str, Any],
                                       n_components: int, random_state: int,
                                       expected_shape: tuple[int, int]) -> tuple[np.ndarray, int]:
    """Create one model's PCA-projected shared-image features.

    This is the representation-mode source preparer; bind every keyword
    argument with ``functools.partial`` to obtain a ``SourcePreparer``.

    Inputs:
        model_name: str identifying the model to preprocess.
        target_context: dict[str, Any] with "stim", "image_ids_for_subj", and
            "shared1000_subj".
        layer_results_path: Path to the existing best-layer CSV.
        feature_extraction_kwargs: Mapping passed to _extract_layer_features.
        n_components: int number of source PCA components.
        random_state: int seed for the randomized PCA solver.
        expected_shape: tuple[int, int] shape of the held-out target, which
            the projected source must match.

    Outputs:
        tuple[np.ndarray, int] containing the projected shared features with
        shape (n_shared, n_components) and the layer index.

    Raises:
        RuntimeError: If model loading fails.
        ValueError: If the layer is unavailable or the projection shape does
            not match the target.
    """

    layer_index = _best_layer_index(model_name, layer_results_path)
    print(f"Extracting model {model_name} with "
          f"use_srp={bool(feature_extraction_kwargs.get('use_srp', False))} "
          f"and n_components={n_components}")
    model_context: dict[str, Any] | None = None
    try:
        model_context, _ = _load_model_context(model_name)
        _check_layer_index(model_context, model_name, layer_index)
        layer_name = model_context["layers_ordered"][layer_index]
        features = _extract_layer_features(model_context, layer_name, target_context,
                                           feature_extraction_kwargs)
        print(f"Completed feature extraction for model {model_name} at layer "
              f"{layer_index} ({layer_name}) with shape {features.shape}")
    finally:
        _release_model_context(model_context)

    projection = _fit_unique_project_shared(features, target_context["shared1000_subj"],
                                            n_components, random_state)
    # (n_images, n_features) -> (n_shared, n_components)
    if projection.shape != tuple(expected_shape):
        raise ValueError(f"Model {model_name} features shape {projection.shape} does not match "
                         f"target shape {tuple(expected_shape)}.")
    return projection, layer_index


# ---------------------------------------------------------------------------
# Pair iteration with optional next-model prefetch (both modes)
# ---------------------------------------------------------------------------


def iter_prepared_pairs(pairs: list[tuple[str, str]], prepare_source: SourcePreparer, *,
                        prefetch_next_model: bool) -> Iterator[PreparedPair]:
    """Yield PID-ready pairs, preparing each model once and optionally ahead.

    Inputs:
        pairs: list[tuple[str, str]] of unfinished ordered comparisons.
        prepare_source: SourcePreparer mapping a model name to its held-out
            source array and layer index.
        prefetch_next_model: bool enabling preparation of the next unseen
            model in one background thread while the caller evaluates the
            currently yielded pair.

    Outputs:
        Iterator yielding (model_1, model_2, source_1, layer_1, source_2,
        layer_2) for each pair, in the order of ``pairs``.

    Raises:
        RuntimeError: If a background source preparation fails.
        Exception: Propagates synchronous preparation failures.
    """

    source_cache: dict[str, tuple[np.ndarray, int]] = {}
    executor = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="pairwise-source")
                if prefetch_next_model else None)
    prefetch_future: Future[tuple[np.ndarray, int]] | None = None
    prefetched_model_name: str | None = None

    try:
        for pair_index, (model_1, model_2) in enumerate(pairs):
            for needed_model_name in (model_1, model_2):
                if needed_model_name in source_cache:
                    continue
                if prefetch_future is not None:
                    if prefetched_model_name is None:
                        raise RuntimeError("A source prefetch future has no associated model name.")
                    future_model_name = prefetched_model_name
                    wait_started_at = time.perf_counter()
                    try:
                        source_cache[future_model_name] = prefetch_future.result()
                    except Exception as error:
                        raise RuntimeError("Prefetched source preparation failed for "
                                           f"{future_model_name!r}.") from error
                    print(f"Wait for prefetched source {future_model_name}: "
                          f"{time.perf_counter() - wait_started_at:.2f} seconds.")
                    prefetch_future = None
                    prefetched_model_name = None
                    if needed_model_name in source_cache:
                        continue
                source_cache[needed_model_name] = prepare_source(needed_model_name)

            if executor is not None and prefetch_future is None:
                for future_pair in pairs[pair_index + 1 :]:
                    next_model_name = next((model_name for model_name in future_pair
                                            if model_name not in source_cache), None)
                    if next_model_name is None:
                        continue
                    prefetched_model_name = next_model_name
                    prefetch_future = executor.submit(prepare_source, next_model_name)
                    break

            source_1, layer_1 = source_cache[model_1]
            source_2, layer_2 = source_cache[model_2]
            yield model_1, model_2, source_1, layer_1, source_2, layer_2
    finally:
        if prefetch_future is not None:
            prefetch_future.cancel()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
