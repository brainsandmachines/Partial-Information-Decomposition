import csv

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from pathlib import Path
import sys
import time
import joblib
from external.mayas_project.features_and_encoding.feat_ext_and_encoding import prepare_model_context
from pipeline.pipeline_utils import nsd_feature_extraction
from pipeline.pipeline_phases.sources_target_features import prepare_target,prepare_sources
from pipeline.pipeline_phases.choosing_layer import overall_best_layer

def find_alpha_per_pc(predictor, target):
    """Find the best ridge alpha separately for each target PC."""

    base_ridge = RidgeCV(
        alphas=np.logspace(-3, 3, 50),
        cv=5,
        scoring="r2",
        fit_intercept=True,
    )

    ridge_per_pc = MultiOutputRegressor(base_ridge)
    ridge_per_pc.fit(predictor, target)

    alphas_per_pc = np.array([
        estimator.alpha_
        for estimator in ridge_per_pc.estimators_
    ])

    assert alphas_per_pc.shape[0] == target.shape[1], "Number of alphas should match number of target PCs"

    return alphas_per_pc, ridge_per_pc








def load_and_apply_pca(
    data: np.ndarray,
    pca_path: str | Path,
    scaler_path: str | Path,
) -> np.ndarray:
    """Scale data and transform it using a subject's fitted PCA model.

    Args:
        data: A 2D array shaped (n_samples, n_features). Its columns must match
            the features and ordering used when fitting the scaler and PCA.
        pca_path: Path to the saved PCA model.
        scaler_path: Path to the saved scaler model.

    Returns:
        A 2D array shaped (n_samples, n_components) containing PCA scores.
    """
    pca_path = Path(pca_path)
    scaler_path = Path(scaler_path)

    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)

    scaled_data = scaler.transform(data)
    transformed_data = pca.transform(scaled_data)

    return transformed_data




def main(
    source_name: str,
    path_to_results: str | Path,
    pc_path: str | Path,
    scaler_path: str | Path,
    hdf_path: Path,
    pkl_info_path: Path,
    neural_data_path: Path,
    alphas_csv_path: str | Path,
):
    """Find per-PC ridge alphas and save model, layer, and PC indexes.

    Inputs:
        source_name: str, model name stored in every CSV row.
        path_to_results: str or Path, best-layer results CSV.
        pc_path: str or Path, fitted PCA model.
        scaler_path: str or Path, fitted scaler model.
        hdf_path: Path, NSD stimulus HDF5 file.
        pkl_info_path: Path, NSD stimulus-information pickle.
        neural_data_path: Path, subject neural-data path.
        alphas_csv_path: str or Path, destination CSV file.

    Output:
        tuple, per-PC alpha array and fitted multi-output ridge model.
    """

    # Prepare the target using the loaded scaler
    target = prepare_target(hdf_path, pkl_info_path, neural_data_path)

    unique_mask = ~np.asarray(target["shared1000_subj"], dtype=bool)

    unique_target_context = target.copy()

    unique_target_context["image_ids_for_subj"] = np.asarray(
    target["image_ids_for_subj"])[unique_mask]

    unique_target_context["neural_data"] = np.asarray(
    target["neural_data"])[unique_mask]


    pca_target = load_and_apply_pca(unique_target_context["neural_data"], pc_path, scaler_path)


    #Prepare model
    model_context = prepare_model_context(source_name)
    model_layer = overall_best_layer(source_name,path_to_results)
    layer_index = model_layer['l']

    if layer_index is None:
        raise ValueError(f"No best layer found for model {source_name}. Please check the results CSV at {path_to_results}.")

    features = nsd_feature_extraction(model_context,layer_index,unique_target_context,batch_size_process=64)



    # Find the best ridge alpha for each target PC
    alphas_per_pc, ridge_model = find_alpha_per_pc(features, pca_target)

    output_path = Path(alphas_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["model_name", "layer_index", "pc_index", "alpha"])
        writer.writerows(
            (source_name, int(layer_index), pc_index, float(alpha))
            for pc_index, alpha in enumerate(alphas_per_pc, start=1)
        )

    return alphas_per_pc, ridge_model
