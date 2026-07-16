from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline, make_pipeline

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from external.mayas_project.features_and_encoding.feat_ext_and_encoding import (
    prepare_model_context,
)
from pipeline.pipeline_phases.choosing_layer import overall_best_layer
from pipeline.pipeline_phases.sources_target_features import prepare_target
from pipeline.pipeline_utils import nsd_feature_extraction


def find_alpha_per_pc(
    predictor: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, Pipeline]:
    """Find one ridge alpha per target PC without variance standardization.

    Inputs:
        predictor:
            Array shaped (n_samples, n_features), containing model features.
        target:
            Array shaped (n_samples, n_components), containing target PC scores.

    Outputs:
        alphas_per_pc:
            Array shaped (n_components,), containing one alpha per target PC.
        ridge_model:
            Fitted raw-input pipeline that predicts all target PCs
            simultaneously. Ridge fits an intercept, but predictor columns are
            not divided by their standard deviations.
    """
    predictor = np.asarray(predictor, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if predictor.ndim != 2:
        raise ValueError("predictor must be a two-dimensional array.")

    if target.ndim != 2:
        raise ValueError("target must be a two-dimensional array.")

    if predictor.shape[0] != target.shape[0]:
        raise ValueError(
            "predictor and target must have the same number of samples."
        )

    alphas = np.logspace(4, 20, 50)

    print(
        f"Finding best ridge alphas for {target.shape[1]} target PCs "
        f"using {predictor.shape[1]} features and "
        f"{predictor.shape[0]} samples."
    )

    print("Using alphas: 10^4 to 10^20, 50 values spaced logarithmically.")

    ridge_model = make_pipeline(
        RidgeCV(
            alphas=alphas,
            cv=None,
            scoring=None, #mse-none
            fit_intercept=True,
            alpha_per_target=True,
            gcv_mode="auto",
        ),
        verbose=True,
    )

    ridge_model.fit(predictor, target)

    ridge_cv = ridge_model.named_steps["ridgecv"]

    alphas_per_pc = np.asarray(
        ridge_cv.alpha_,
        dtype=np.float64,
    )

    assert alphas_per_pc.shape == (target.shape[1],), (
        "Number of selected alphas must match the number of target PCs."
    )

    return alphas_per_pc, ridge_model







def load_and_apply_pca(
    data: np.ndarray,
    pca_path: str | Path,
) -> np.ndarray:
    """Transform raw neural data with a saved centered PCA model.

    Inputs:
        data: A 2D array shaped (n_samples, n_features). Its columns must match
            the raw, unstandardized feature ordering used to fit PCA.
        pca_path: str or Path to the saved centered PCA model.

    Outputs:
        A 2D array shaped (n_samples, n_components) containing PCA scores.
        PCA applies its fitted training mean but no variance standardization.
    """
    pca_path = Path(pca_path)
    pca = joblib.load(pca_path)
    transformed_data = pca.transform(np.asarray(data))

    return transformed_data


def split_alphas_csv_by_model(
    alphas_csv_path: str | Path,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Split an aggregate per-PC alpha CSV into one NumPy file per model.

    Inputs:
        alphas_csv_path:
            str or Path pointing to a CSV with the columns ``model_name``,
            ``layer_index``, ``pc_index``, and ``alpha``.
        output_dir:
            str, Path, or None specifying the folder for the model files. When
            None, a ``<source_csv_stem>_by_model`` folder is created beside the
            source CSV.

    Outputs:
        list[Path]:
            Paths to the created ``.npz`` files, ordered by each model's first
            appearance in the source CSV. Every file contains ``alphas``,
            ``pc_indices``, ``model_name``, and ``layer_index``.
    """
    source_path = Path(alphas_csv_path)
    expected_header = ["model_name", "layer_index", "pc_index", "alpha"]
    rows_by_model: dict[str, list[tuple[int, int, float]]] = {}

    with source_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        source_header = next(reader, None)

        if source_header != expected_header:
            raise ValueError(
                f"Alpha CSV has an incompatible header: {source_header}. "
                f"Expected: {expected_header}."
            )

        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(expected_header):
                raise ValueError(
                    f"Row {row_number} has {len(row)} columns; "
                    f"expected {len(expected_header)}."
                )

            model_name = row[0].strip()
            if not model_name:
                raise ValueError(f"Row {row_number} has an empty model_name.")

            try:
                layer_index = int(row[1])
                pc_index = int(row[2])
                alpha = float(row[3])
            except ValueError as error:
                raise ValueError(
                    f"Row {row_number} contains a non-numeric layer_index, "
                    "pc_index, or alpha."
                ) from error

            rows_by_model.setdefault(model_name, []).append(
                (layer_index, pc_index, alpha)
            )

    if not rows_by_model:
        raise ValueError("Alpha CSV does not contain any model rows.")

    output_folder = (
        Path(output_dir)
        if output_dir is not None
        else source_path.with_name(f"{source_path.stem}_by_model")
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    output_paths = []
    used_filenames = set()
    for model_name, model_rows in rows_by_model.items():
        safe_model_name = model_name.replace("/", "_").replace("\\", "_")
        output_filename = f"{source_path.stem}_{safe_model_name}.npz"

        if output_filename in used_filenames:
            raise ValueError(
                "Model names produce the same output filename after replacing "
                f"path separators: {output_filename}."
            )
        used_filenames.add(output_filename)

        layer_indices = {row[0] for row in model_rows}
        if len(layer_indices) != 1:
            raise ValueError(
                f"Model {model_name!r} has multiple layer indices: "
                f"{sorted(layer_indices)}."
            )

        source_pc_indices = np.asarray(
            [row[1] for row in model_rows],
            dtype=np.int64,
        )
        expected_pc_indices = np.arange(1, len(model_rows) + 1)
        if not np.array_equal(source_pc_indices, expected_pc_indices):
            raise ValueError(
                f"Model {model_name!r} must have ordered, one-based PC indices."
            )

        alphas_per_pc = np.asarray(
            [row[2] for row in model_rows],
            dtype=np.float64,
        )
        source_name = model_name
        layer_index = next(iter(layer_indices))
        output_path = output_folder / output_filename
        np.savez(
            output_path,
            alphas=np.asarray(alphas_per_pc, dtype=np.float64),
            pc_indices=np.arange(1, len(alphas_per_pc) + 1),
            model_name=str(source_name),
            layer_index=int(layer_index),
        )

        output_paths.append(output_path)

    return output_paths




def main(
    source_name: str,
    path_to_results: str | Path,
    pc_path: str | Path,
    hdf_path: str | Path,
    pkl_info_path: str | Path,
    neural_data_path: str | Path,
    alphas_csv_path: str | Path,
) -> tuple[np.ndarray, Pipeline]:
    """Find raw-feature ridge alphas and save model/layer/PC metadata.

    Inputs:
        source_name: str, model name stored in every CSV row.
        path_to_results: str or Path, best-layer results CSV.
        pc_path: str or Path, PCA fitted directly on raw neural responses.
        hdf_path: str or Path, NSD stimulus HDF5 file.
        pkl_info_path: str or Path, NSD stimulus-information pickle.
        neural_data_path: str or Path, subject neural-data path.
        alphas_csv_path: str or Path, destination CSV file.

    Outputs:
        tuple[np.ndarray, Pipeline] containing the per-PC alpha vector and the
        fitted raw-input RidgeCV pipeline. No target or predictor variance
    """

    target = prepare_target(hdf_path, pkl_info_path, neural_data_path)

    unique_mask = ~np.asarray(target["shared1000_subj"], dtype=bool)

    unique_target_context = target.copy()

    unique_target_context["image_ids_for_subj"] = np.asarray(
    target["image_ids_for_subj"])[unique_mask]

    unique_target_context["target"] = np.asarray(
    target["target"])[unique_mask]


    pca_target = load_and_apply_pca(
        unique_target_context["target"],
        pc_path,
    )

    #Save memory 
    del target 

    #Prepare model
    model_context = prepare_model_context(source_name)
    model_layer = overall_best_layer(source_name,path_to_results)
    layer_index = model_layer['l']

    if layer_index is None:
        raise ValueError(f"No best layer found for model {source_name}. Please check the results CSV at {path_to_results}.")

    features = nsd_feature_extraction(model_context,layer_index,unique_target_context,batch_size_process=28)

    #Save memory
    del model_context


    # Find the best ridge alpha for each target PC
    alphas_per_pc, ridge_model = find_alpha_per_pc(
        features,
        pca_target,
    )

    output_path = Path(alphas_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_header = ["model_name", "layer_index", "pc_index", "alpha"]
    csv_has_content = output_path.is_file() and output_path.stat().st_size > 0
    if csv_has_content:
        with output_path.open("r", newline="", encoding="utf-8") as csv_file:
            existing_header = next(csv.reader(csv_file), None)
        if existing_header != csv_header:
            raise ValueError(
                f"Existing alpha CSV has an incompatible header: {existing_header}. "
                f"Expected: {csv_header}."
            )

    write_mode = "a" if csv_has_content else "w"
    with output_path.open(write_mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not csv_has_content:
            writer.writerow(csv_header)
        writer.writerows(
            (source_name, int(layer_index), pc_index, float(alpha))
            for pc_index, alpha in enumerate(alphas_per_pc, start=1)
        )
        print("\nSelected alpha and cross-validation score for each PC:")

    ridge_cv = ridge_model.named_steps["ridgecv"]
    
    alphas_per_pc = np.asarray(
    ridge_cv.alpha_,
    dtype=np.float64,
    )

    scores_per_pc = np.asarray(
        ridge_cv.best_score_,
        dtype=np.float64,
    )

    print("Finished finding best ridge alphas for each target PC.✅")
    for pc_index, (alpha, score) in enumerate(
        zip(alphas_per_pc, scores_per_pc),
        start=1,
    ):
        print(
            f"\nPC {pc_index:>3}: "
            f"alpha = {alpha:.6g}, "
            f"CV score = {score:.6f}"
        )

    mean_score = float(np.mean(scores_per_pc))

    print(f"\nMean cross-validation score across PCs: {mean_score:.6f}")


    print(f"\n======================================================================================")

    return alphas_per_pc, ridge_model



if __name__ == "__main__":

    def check_path_exists(config: dict[str, Any]) -> None:
        """Require every configured input path before alpha generation.

        Inputs:
            config: dict[str, Any] containing input Paths or nested mappings.

        Outputs:
            None. Missing input paths raise ``FileNotFoundError``.
        """

        for key, value in config.items():
            if isinstance(value, Path) and not value.exists():
                raise FileNotFoundError(f"Path {value} does not exist.")
            elif isinstance(value, dict):
                check_path_exists(value)

    model_list = [
        "nf_resnet50_classification",
        "eca_nfnet_l0_classification",
        "resnet50_classification",
        "semnasnet_100_classification",
        "cspresnet50_classification",
        "mobilenetv3_large_100_classification",
        "ghostnet_100_classification",
        "convnext_base_classification",
        "xcit_nano_12_p8_224_classification",
        "xcit_nano_12_p16_224_classification",
        "swin_large_patch4_window7_224_classification",
        "jx_nest_tiny_classification",
        "pit_ti_224_classification",
        "vit_base_patch32_224_classification",
        "vit_base_patch16_224_classification",
        "tnt_s_patch16_224_classification",
        "crossvit_base_240_classification",
        "deit_base_patch16_224_classification",
        "levit_128_classification",
    ]
        #Path to best layer results
    path_to_results = Path('/home/ohadshee/Desktop/Partial-Information-Decomposition/external/mayas_project/results_shared/encoding/best_layers/subj01_OTC_all_models_best_layer_overall.csv')
    
    pc_path = Path('/home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/subj_PCs/saved_pcs/missmda_max=100_no_variance_standardization/subj01_pca_model.pkl')
    
    hdf_path = Path('/groups/golan_neurogroup/bml_group/datasets/nsddata/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5')
    pkl_info_path = Path('/groups/golan_neurogroup/bml_group/datasets/nsddata/nsddata/experiments/nsd/nsd_stim_info_merged.pkl')
    neural_data_path = Path('/groups/golan_neurogroup/bml_group/datasets/nsddata/otc_betas/otc_betas_per_stim/subj01_OTC_betas_per_stimulus.zarr')

    alphas_csv_path = Path('/home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/ridge_find_alpha/results/no_variance_standardization/alphas2.0_per_pc.csv')
    output_dir = Path('/home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/ridge_find_alpha/results/no_variance_standardization/alphas')
        
    path_config = {
        "path_to_results": path_to_results,
        "pc_path": pc_path,
        "hdf_path": hdf_path,
        "pkl_info_path": pkl_info_path,
        "neural_data_path": neural_data_path,
    }
    
    check_path_exists(path_config)
    
    for source_name in model_list:

        print("\nChosen model:", source_name  )


        main(
            source_name,
            path_to_results,
            pc_path,
            hdf_path,
            pkl_info_path,
            neural_data_path,
            alphas_csv_path,
        )
    

    split_alphas_csv_by_model(alphas_csv_path, output_dir)
