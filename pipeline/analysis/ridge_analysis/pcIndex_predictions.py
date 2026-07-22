from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

repo_root = Path(__file__).resolve().parents[3]
external_root = repo_root / "external"
for path in (repo_root, external_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from external.mayas_project.features_and_encoding.feat_ext_and_encoding import (
    prepare_model_context,
)
from pipeline.analysis.anlysis_utils import (
    _prepare_source_for_pid,
    to_deepdive_model_name,
)
from pipeline.pipeline_phases.choosing_layer import overall_best_layer
from pipeline.pipeline_phases.feature_manipulations import prepare_ridge_target
from pipeline.pipeline_phases.sources_target_features import prepare_target
from pipeline.pipeline_utils import nsd_feature_extraction


pc_path = Path("/home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/subj_PCs/saved_pcs_nostandardization/pca_by_variance=8000/subj01_pca_model.pkl")
hdf_path = Path("/groups/golan_neurogroup/bml_group/datasets/nsddata/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5")
pkl_info_path = Path("/groups/golan_neurogroup/bml_group/datasets/nsddata/nsddata/experiments/nsd/nsd_stim_info_merged.pkl")
neural_data_path = Path("/groups/golan_neurogroup/bml_group/datasets/nsddata/otc_betas/otc_betas_per_stim/subj01_OTC_betas_per_stimulus.zarr")
path_to_results = repo_root / "external/mayas_project/results_shared/encoding/best_layers/subj01_OTC_all_models_best_layer_overall.csv"


def each_pc_index_pred(
    model_name: str,
    n_pcs: list[int],
    pc_path: str | Path,
    hdf_path: str | Path,
    pkl_info_path: str | Path,
    neural_data_path: str | Path,
) -> tuple[np.ndarray, int]:
    """Predict individual target PCs from one model's selected layer.

    Inputs:
        model_name: str model identifier from the best-layer CSV.
        n_pcs: list[int] of zero-based target-PC indexes to predict.
        pc_path: str or Path to the fitted target PCA model.
        hdf_path: str or Path to the NSD stimulus HDF5 file.
        pkl_info_path: str or Path to the NSD stimulus-information pickle.
        neural_data_path: str or Path to the subject neural-data Zarr store.

    Outputs:
        tuple[np.ndarray, int] containing correlations in ``n_pcs`` order and
        the selected model-layer index.
    """

    target_context = prepare_target(Path(hdf_path), Path(pkl_info_path), Path(neural_data_path))
    train_target, shared_target, shared_mask = prepare_ridge_target(
        target_context["target"], target_context, pc_path
    )
    if any(pc_index < 0 or pc_index >= train_target.shape[1] for pc_index in n_pcs):
        raise IndexError(f"PC indexes must be between 0 and {train_target.shape[1] - 1}.")

    layer_index = overall_best_layer(model_name, path_to_results=str(path_to_results))["l"]
    if layer_index is None:
        raise ValueError(f"No best layer found for model {model_name!r}.")
    source_context = prepare_model_context(to_deepdive_model_name(model_name))
    print(f"Starting to extract features for {model_name} at layer {layer_index}")
    features = nsd_feature_extraction(source_context, layer_index, target_context=target_context,batch_size_process=128,batch_size_dataloader=128)

    print(f"\nFeatures shape: {features.shape}, Train target shape: {train_target.shape}, Shared target shape: {shared_target.shape}")
    print(f"Running ridge regression for {len(n_pcs)} PCs on model {model_name} at layer {layer_index}")

    correlations = np.full(len(n_pcs), np.nan, dtype=float)
    selected_train_target = train_target[:, n_pcs]
    # (n_train, n_all_pcs) -> (n_train, n_requested_pcs)
    selected_shared_target = shared_target[:, n_pcs]
    # (n_shared, n_all_pcs) -> (n_shared, n_requested_pcs)

    source_predictions = np.asarray(
        _prepare_source_for_pid(
            features,
            selected_train_target,
            shared_mask,
            ridge=True,
        )
    )
    # (n_samples, n_features) -> (n_shared, n_requested_pcs)

    print(f"Running analysis for {len(n_pcs)} PCs")
    for result_index, pc_index in enumerate(n_pcs):
        print(f"Running analysis for PC index {pc_index}")
        pc_test_target = selected_shared_target[:, result_index]
        # (n_shared, n_requested_pcs) -> (n_shared,)
        source_pred = source_predictions[:, result_index]
        # (n_shared, n_requested_pcs) -> (n_shared,)
        if np.std(pc_test_target) > 0 and np.std(source_pred) > 0:
            corr = pearsonr(pc_test_target, source_pred).statistic
            print(f"PC index {pc_index} correlation: {corr:.4f}")
            correlations[result_index] = corr
        else:
            print(f"PC index {pc_index} correlation: NaN (zero variance in target or prediction)")
            correlations[result_index] = np.nan
    return correlations, int(layer_index)


def save_correlations_to_csv(
    correlations: np.ndarray,
    model_name: str,
    layer_index: int,
    pc_indexes: list[int],
    output_path: str | Path,
) -> Path:
    """Append one completed model's per-PC correlations to a checkpoint CSV.

    Inputs:
        correlations: np.ndarray with one correlation per requested PC.
        model_name: str model identifier used as the resume key.
        layer_index: int selected model-layer index.
        pc_indexes: list[int] of zero-based PC indexes matching correlations.
        output_path: str or Path to the checkpoint CSV.

    Outputs:
        Path to the updated checkpoint CSV.
    """
    correlations = np.asarray(correlations, dtype=float).reshape(-1)
    if correlations.size != len(pc_indexes):
        raise ValueError("correlations and pc_indexes must have the same length.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["model_name", "layer_index", *(f"pc_{index}_correlation" for index in pc_indexes)]
    has_content = output_path.is_file() and output_path.stat().st_size > 0
    if has_content:
        with output_path.open("r", newline="", encoding="utf-8") as csv_file:
            if next(csv.reader(csv_file), None) != header:
                raise ValueError(f"Existing checkpoint has an incompatible header: {output_path}")
    with output_path.open("a" if has_content else "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not has_content:
            writer.writerow(header)
        writer.writerow([model_name, int(layer_index), *correlations.tolist()])
    print(f"Saved {model_name} correlations to {output_path} 😀")
    return output_path


def main() -> None:
    """Run every best-layer model and resume from a per-model CSV checkpoint.

    Inputs:
        None.

    Outputs:
        None. Saves one CSV row immediately after each completed model.
    """
    
    

    model_names =  ['dino_resnet50_selfsupervised', 'dino_vitb16_selfsupervised','nf_resnet50_classification','hardcorenas_f_classification','eca_nfnet_l0_classification',
        'resnet50_classification','semnasnet_100_classification','cspresnet50_classification',
        'mobilenetv3_large_100_classification','ghostnet_100_classification','convnext_base_classification'
        ,'swin_large_patch4_window7_224_classification','jx_nest_tiny_classification',''
        'pit_ti_224_classification','vit_base_patch32_224_classification','vit_base_patch16_224_classification',
        'tnt_s_patch16_224_classification','crossvit_base_240_classification','deit_base_patch16_224_classification',
        'levit_128_classification','coat_lite_tiny_classification','visformer_small_classification',
        'convit_base_classification','RN50_clip','RN101_clip',
        'ResNet50-SimCLR_selfsupervised','ResNet50-DeepClusterV2-2x224_selfsupervised','ResNet50-SwAV-BS4096-2x224_selfsupervised',
        'ResNet50-PIRL_selfsupervised','ResNet50-ClusterFit-16K-RotNet_selfsupervised','ResNet50-MoCoV2-BS256_selfsupervised'
        ]
    #oom kill: 'ViT-L_14_clip','ViT-B_32_clip','dino_resnet50_selfsupervised', 'dino_vitb16_selfsupervised''xcit_nano_12_p8_224_classification''xcit_nano_12_p16_224_classification',
    n_pcs = list(range(8000))  # zero-based PC indexes to predict
    output_path = Path(f'/home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/analysis/ridge_analysis/New_GPU_{max(n_pcs)+1}_pcs_correlations_by_model.csv')
    print("Results will be saved to:\n", output_path)
    header = ["model_name", "layer_index", *(f"pc_{index}_correlation" for index in n_pcs)]
    completed_models: set[str] = set()
    if output_path.is_file() and output_path.stat().st_size > 0:
        with output_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames != header:
                raise ValueError(f"Existing checkpoint has an incompatible header: {output_path}")
            completed_models = {row["model_name"].strip() for row in reader if row["model_name"].strip()}
    
    for model_name in model_names:
        if model_name in completed_models:
            print(f"Skipping completed model: {model_name}")
            continue
        print(f"Starting model: {model_name}")
        try:
            correlations, layer_index = each_pc_index_pred(
                model_name, n_pcs, pc_path, hdf_path, pkl_info_path, neural_data_path
            )
            save_correlations_to_csv(correlations, model_name, layer_index, n_pcs, output_path)
            completed_models.add(model_name)
            
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            continue

if __name__ == "__main__":
    main()
