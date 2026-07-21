"""Run deterministic OTC PID comparisons across unordered model pairs."""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA

repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from external.mayas_project.features_and_encoding.feat_ext_and_encoding import (
    batch_process,
    get_layer_feature_count,
    get_sparse_projection_gpu,
    prepare_model_context,
)
from pipeline.full_OTC import otc_experiment
from pipeline.pipeline_phases.choosing_layer import overall_best_layer
from pipeline.pipeline_phases.sources_target_features import batching
from pipeline.pipeline_utils import resolve_pipeline_function
from pipeline.plotting.plot_functions import plot_pairwise_pid_matrices


#model_1_names = ['nf_resnet50_classification','hardcorenas_f_classification']

'''model_1_names = ['nf_resnet50_classification','hardcorenas_f_classification','eca_nfnet_l0_classification',
        'resnet50_classification','semnasnet_100_classification','cspresnet50_classification',
        'mobilenetv3_large_100_classification','ghostnet_100_classification','convnext_base_classification','xcit_nano_12_p8_224_classification'
        ,'xcit_nano_12_p16_224_classification','swin_large_patch4_window7_224_classification','jx_nest_tiny_classification',''
        'pit_ti_224_classification','vit_base_patch32_224_classification','vit_base_patch16_224_classification',
        'tnt_s_patch16_224_classification','crossvit_base_240_classification','deit_base_patch16_224_classification',
        'levit_128_classification','coat_lite_tiny_classification','visformer_small_classification',
        'convit_base_classification','ViT-B_32_clip','RN50_clip','RN101_clip','ViT-L_14_clip',
        'ResNet50-SimCLR_selfsupervised','ResNet50-DeepClusterV2-2x224_selfsupervised','ResNet50-SwAV-BS4096-2x224_selfsupervised',
        'ResNet50-PIRL_selfsupervised','ResNet50-ClusterFit-16K-RotNet_selfsupervised','ResNet50-MoCoV2-BS256_selfsupervised'
        ]''' #dino_resnet50_selfsupervised, dino_vitb16_selfsupervised - missing

model_1_names = ['nf_resnet50_classification','eca_nfnet_l0_classification','resnet50_classification','semnasnet_100_classification',
                 'cspresnet50_classification','mobilenetv3_large_100_classification','ghostnet_100_classification','convnext_base_classification','xcit_nano_12_p8_224_classification',
                 'xcit_nano_12_p16_224_classification']

model_2_names = model_1_names  # Compare all models against each other



def deterministic_pca(
    features: Any,
    n_components: int,
    random_state: int,
) -> np.ndarray:
    """Project one sample matrix with reproducible randomized PCA.

    Inputs:
        features: array-like, samples with shape (n_samples, n_features).
        n_components: int, requested number of principal components.
        random_state: int, seed used by randomized SVD.

    Output:
        projected: np.ndarray, float64 samples with shape
            (n_samples, min(n_components, n_samples, n_features)).
    """

    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"features must be two-dimensional, got shape {array.shape}")
    effective_components = min(int(n_components), *array.shape)
    if effective_components < 1:
        raise ValueError("n_components must leave at least one PCA component.")
    pca = PCA(
        n_components=effective_components,
        svd_solver="randomized",
        random_state=int(random_state),
        copy=False,
    )
    return np.asarray(pca.fit_transform(array), dtype=np.float64)


def extract_model_projection(
    model_name: str,
    target_context: dict[str, Any],
    choose_layer_kwargs: dict[str, Any],
    feature_extraction_kwargs: dict[str, Any],
    n_components: int,
    random_state: int,
) -> tuple[np.ndarray, int]:
    """Extract one selected model layer and return its memory-safe PCA projection.

    Inputs:
        model_name: str, stored model identifier used for layer lookup.
        target_context: dict[str, Any], contains stimulus images and ordered
            image IDs.
        choose_layer_kwargs: dict[str, Any], contains path_to_results for the
            overall best-layer CSV.
        feature_extraction_kwargs: dict[str, Any], contains outer and DataLoader
            batch sizes plus use_srp and srp_n_components. A null SRP dimension
            uses target_context["n_projections"] automatically.
        n_components: int, requested final PCA dimension.
        random_state: int, seed used by randomized PCA.

    Output:
        projection_and_layer: tuple[np.ndarray, int], the float64 projected
            features with shape (n_samples, n_components) and selected layer
            index. Raw batches are discarded immediately after reduction.
    """

    use_srp = bool(feature_extraction_kwargs.get("use_srp", False))
    srp_components = feature_extraction_kwargs.get("srp_n_components")
    if use_srp and srp_components is None:
        srp_components = target_context.get("n_projections")
    if use_srp and (srp_components is None or int(srp_components) < 1):
        raise ValueError("srp_n_components must be a positive integer when use_srp is true.")
    model_context = prepare_model_context(model_name)
    layer_result = overall_best_layer(
        model_name=model_name,
        path_to_results=str(choose_layer_kwargs["path_to_results"]),
    )
    if layer_result["l"] is None:
        raise ValueError(f"No overall best layer found for model {model_name!r}.")
    layer_index = int(layer_result["l"])
    layer_name = model_context["layers_ordered"][layer_index]
    model = model_context["model"]
    raw_dimension = get_layer_feature_count(model, layer_name)
    intermediate_dimension = (
        min(int(srp_components), int(raw_dimension))
        if use_srp
        else int(raw_dimension)
    )
    n_samples = len(target_context["image_ids_for_subj"])
    intermediate = np.empty(
        (n_samples, intermediate_dimension),
        dtype=np.float64,
    )
    batch_size_process = int(feature_extraction_kwargs["batch_size_process"])
    batch_size_dataloader = int(feature_extraction_kwargs["batch_size_dataloader"])
    sparse_projection = None

    try:
        if raw_dimension > intermediate_dimension:
            model_device = str(next(model.parameters()).device)
            sparse_projection = get_sparse_projection_gpu(
                raw_dimension,
                intermediate_dimension,
                device=model_device,
            )

        for batch_start in range(0, n_samples, batch_size_process):
            batch_end = min(batch_start + batch_size_process, n_samples)
            if sparse_projection is None:
                reduced_batch = batching(
                    model_context=model_context,
                    batch_start=batch_start,
                    batch_end=batch_end,
                    stim_dataset=target_context["stim"],
                    subj_image_ids=target_context["image_ids_for_subj"],
                    layer_name=layer_name,
                    batch_size_dataloader=batch_size_dataloader,
                )
            else:
                reduced_batch = batch_process(
                    model=model,
                    batch_start=batch_start,
                    batch_end=batch_end,
                    batch_size_process=batch_size_process,
                    batch_size_dataloader=batch_size_dataloader,
                    layer_name=layer_name,
                    sparse_projection_mat_gpu=sparse_projection,
                    stim_dataset=target_context["stim"],
                    image_ids_for_subj=target_context["image_ids_for_subj"],
                    image_transforms=model_context["image_transforms"],
                )
            intermediate[batch_start:batch_end] = reduced_batch
            del reduced_batch

        return (
            deterministic_pca(intermediate, n_components, random_state),
            layer_index,
        )
    finally:
        del model_context, model
        if sparse_projection is not None:
            del sparse_projection
        gc.collect()
        torch.cuda.empty_cache()


def run_pairwise_pid_pipeline(
    model_1_names: list[str],
    model_2_names: list[str],
    otc_config: dict[str, Any],
    csv_path: str | Path,
) -> Path:
    """Run OTC PID once per unordered model pair and checkpoint results to CSV.

    Inputs:
        model_1_names: list[str], model names to use as source X1.
        model_2_names: list[str], model names to use as source X2.
        otc_config: dict[str, Any], already-loaded OTC pipeline configuration.
        csv_path: str or Path, output CSV used for checkpointing and resuming.

    Output:
        output_path: Path, path to the CSV containing existing and newly
            calculated pair results.

    The target is projected once. Each model is extracted with optional
    batchwise SRP, projected with deterministic PCA, and retained only as its
    small final array in memory. Self-pairs and completed CSV pairs are skipped,
    and each successful row is appended immediately.
    """

    columns = [
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
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        try:
            existing_results = pd.read_csv(output_path)
        except pd.errors.EmptyDataError as error:
            raise ValueError(f"Existing CSV has no header: {output_path}") from error
        if list(existing_results.columns) != columns:
            raise ValueError(
                f"Existing CSV has an incompatible schema: {output_path}. "
                f"Expected columns: {columns}"
            )
    else:
        existing_results = pd.DataFrame(columns=columns)
        existing_results.to_csv(output_path, index=False)

    completed_pairs = {
        frozenset((model_1, model_2))
        for model_1, model_2 in zip(
            existing_results["model_1"].astype(str),
            existing_results["model_2"].astype(str),
        )
    }
    config = otc_config
    registry = otc_experiment.PIPELINE_STEP_FUNCTIONS
    target_extraction = resolve_pipeline_function(
        config["functions"], registry, "target_extraction", required=True
    )
    pid_calculation = resolve_pipeline_function(
        config["functions"], registry, "pid_calculation", required=True
    )
    pid_report = resolve_pipeline_function(
        config["functions"], registry, "pid_report", required=False
    )
    feature_kwargs = config.get("feature_manipulation_kwargs", {})
    extraction_kwargs = config.get("feature_extraction_kwargs", {})
    pid_kwargs = config.get("pid_kwargs", {})
    pid_config = pid_kwargs.get("config") or {}
    random_state = int(pid_kwargs.get("rng_seed", 56))
    source_components = int(feature_kwargs["n_components_source_1"])
    if source_components != int(feature_kwargs["n_components_source_2"]):
        raise ValueError(
            "Memory-cached pairwise comparisons require equal PCA dimensions "
            "for source 1 and source 2."
        )
    target_context = target_extraction(**config.get("target_kwargs", {}))
    try:
        target = deterministic_pca(
            target_context["target"],
            feature_kwargs["n_components_target"],
            random_state,
        )
    except Exception:
        hdf_file = target_context.get("hdf_file")
        if hdf_file is not None:
            hdf_file.close()
        raise
    target_context.pop("target", None)
    target_context.pop("neural_data", None)
    feature_cache: dict[str, tuple[np.ndarray, int]] = {}

    try:
        for model_1 in model_1_names:
            for model_2 in model_2_names:
                pair = frozenset((model_1, model_2))
                if model_1 == model_2 or pair in completed_pairs:
                    print(f"Skipping pair: {model_1}, {model_2} (self-pair or already completed)")
                    continue
                    

                if model_1 not in feature_cache:
                    feature_cache[model_1] = extract_model_projection(
                        model_1,
                        target_context,
                        config["choose_layer_kwargs"],
                        extraction_kwargs,
                        source_components,
                        random_state,
                    )
                    print(f"Extracted and cached features for model: {model_1}")
                if model_2 not in feature_cache:
                    feature_cache[model_2] = extract_model_projection(
                        model_2,
                        target_context,
                        config["choose_layer_kwargs"],
                        extraction_kwargs,
                        source_components,
                        random_state,
                    )
                    print(f"Extracted and cached features for model: {model_2}")
                source_1, layer_1 = feature_cache[model_1]
                source_2, layer_2 = feature_cache[model_2]
                pid_results = pid_calculation(
                    target,
                    source_1,
                    source_2,
                    **pid_kwargs,
                )
                pid = pid_results["pid"]
                mi = pid_results["mi"]
                if pid_report is not None:
                    pid_report(pid_results, {}, **config.get("report_kwargs", {}))

                row = {
                    "model_1": model_1,
                    "model_2": model_2,
                    "layer_1": layer_1,
                    "layer_2": layer_2,
                    "subj_id": config.get("target_kwargs", {}).get("subj_id"),
                    "n_samples": len(target),
                    "n_components_source_1": source_1.shape[1],
                    "n_components_source_2": source_2.shape[1],
                    "n_components_target": target.shape[1],
                    "pid_method": pid_results.get("method", pid_kwargs.get("method")),
                    "rng_seed": pid_kwargs.get("rng_seed"),
                    "bias_correction": pid_config.get("bias_correction"),
                    "red": pid["red"],
                    "unq1": pid["unq1"],
                    "unq2": pid["unq2"],
                    "syn": pid["syn"],
                    "bi_mi_1": mi["bi_mi_1"],
                    "bi_mi_2": mi["bi_mi_2"],
                    "tri_mi": mi["tri_mi"],
                }
                pd.DataFrame([row], columns=columns).to_csv(
                    output_path,
                    mode="a",
                    header=False,
                    index=False,
                )
                completed_pairs.add(pair)
                print(
                    f"Completed PID for pair: {model_1}, {model_2}. "
                    f"Results appended to {output_path}"
                )
    finally:
        hdf_file = target_context.get("hdf_file")
        if hdf_file is not None:
            hdf_file.close()

    return output_path

if __name__ == "__main__":
    analysis_dir = Path(
        "//home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/analysis/pca_analysis/all_models_pairwise/pair_wise_ridge"
    )
    config_path = Path('/home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/analysis/pca_analysis/all_models_pairwise/ridge_otc_pair_wise_comp.yaml')
    csv_path = analysis_dir / "pairwise_pid_results_ridge_pca.csv"
    plot_path = csv_path.parent / "pairwise_figures_ridge_pca"
    with open(config_path, "r") as config_file:
        otc_config = yaml.safe_load(config_file)

    run_pairwise_pid_pipeline(
        model_1_names=model_1_names,
        model_2_names=model_2_names,
        otc_config=otc_config,
        csv_path=csv_path,
    )
    csv_path = Path('/home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/analysis/pca_analysis/all_models_pairwise/pairwise_pid_results_ridge_pca.csv')
    plot_pairwise_pid_matrices(
        csv_path=csv_path,
        output_dir='/home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/analysis/pca_analysis/all_models_pairwise/pair_wise_ridge',
    )
