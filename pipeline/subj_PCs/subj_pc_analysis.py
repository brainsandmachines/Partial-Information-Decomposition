"""Fit subject-level PCA models and evaluate them on shared held-out images."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pipeline.pipeline_phases.sources_target_features import prepare_target


from Simulations.PCA_rank.eigenvector_pca import eigenvector_pca_cv
from library_wrappers.missmda_ncp import estimate_ncp_pca




def split_unique_shared(
    subj_id: str,
    hdf_path: str | Path,
    pkl_info_path: str | Path,
    neural_data_path: str | Path,
    variance_threshold: float = 0.99,
) -> dict[str, Any]:
    """Load one subject and split its rows into unique and shared-image sets.

    Inputs:
        subj_id: str, subject identifier retained in the returned context.
        hdf_path: str or Path, path to the NSD stimulus HDF5 file.
        pkl_info_path: str or Path, path to the NSD stimulus-information pickle.
        neural_data_path: str or Path, path to the subject's neural-data Zarr.
        variance_threshold: float, PCA threshold retained in the returned
            context for bookkeeping.

    Output:
        split_context: dict[str, Any], the target context plus Boolean masks,
            subject image IDs, and neural arrays for the unique training set
            and shared held-out set.
    """



    target_data = prepare_target(
        Path(hdf_path),
        Path(pkl_info_path),
        Path(neural_data_path),
    )
    shared_mask = np.asarray(target_data["shared1000_subj"], dtype=bool)
    neural_data = np.asarray(target_data["neural_data"])
    image_ids = np.asarray(target_data["image_ids_for_subj"])

    if neural_data.ndim != 2:
        raise ValueError(
            "Expected neural_data to have shape (n_images, n_voxels), "
            f"but received {neural_data.shape}."
        )
    if shared_mask.ndim != 1 or shared_mask.shape[0] != neural_data.shape[0]:
        raise ValueError(
            "shared1000_subj must be a one-dimensional mask with one entry "
            "per neural-data row."
        )
    if image_ids.ndim != 1 or image_ids.shape[0] != neural_data.shape[0]:
        raise ValueError(
            "image_ids_for_subj must contain one image ID per neural-data row."
        )

    unique_mask = ~shared_mask
    return {
        **target_data,
        "subj_id": subj_id,
        "variance_threshold": variance_threshold,
        "unique_mask": unique_mask,
        "shared_mask": shared_mask,
        "unique_images": image_ids[unique_mask],
        "shared_images": image_ids[shared_mask],
        "unique_neural_data": neural_data[unique_mask],
        "shared_neural_data": neural_data[shared_mask],
    }


def pca_by_variance(
    neural_data: np.ndarray,
    variance_threshold: float = 0.99,
) -> dict[str, Any]:
    """Fit standardized PCA retaining a requested training-variance fraction.

    Inputs:
        neural_data: np.ndarray of shape (n_samples, n_features), training
            neural responses.
        variance_threshold: float in (0, 1], fraction of training variance
            that the retained components should explain.

    Output:
        pca_results: dict[str, Any], containing the fitted PCA model under
            ``pca``, fitted StandardScaler under ``scaler``, projected training
            samples under ``transformed_data``, and ``variance_threshold``.
    """

    neural_array = np.asarray(neural_data)
    if neural_array.ndim != 2:
        raise ValueError(
            "neural_data must have shape (n_samples, n_features), "
            f"but received {neural_array.shape}."
        )
    if neural_array.shape[0] < 2 or neural_array.shape[1] < 1:
        raise ValueError("PCA requires at least two samples and one feature.")
    if not 0.0 < variance_threshold <= 1.0:
        raise ValueError("variance_threshold must be in the interval (0, 1].")

    scaler = StandardScaler()
    neural_data_scaled = scaler.fit_transform(neural_array)

    pca_components = None if variance_threshold == 1.0 else variance_threshold
    pca_model = PCA(n_components=pca_components, svd_solver="full")
    transformed_data = pca_model.fit_transform(neural_data_scaled)

    return {
        "pca": pca_model,
        "scaler": scaler,
        "transformed_data": transformed_data,
        "variance_threshold": variance_threshold,
    }


def heldout_pca(
    pca_model: PCA,
    scaler_model: StandardScaler,
    heldout_data: np.ndarray,
) -> np.ndarray:
    """Project held-out neural responses with fitted training transformations.

    Inputs:
        pca_model: PCA, fitted PCA model learned from the training data.
        scaler_model: StandardScaler, fitted scaler learned from training data.
        heldout_data: np.ndarray of shape (n_samples, n_features), held-out
            neural responses with the same feature order as the training data.

    Output:
        heldout_scores: np.ndarray of shape (n_samples, n_components), held-out
            samples represented in the fitted PCA basis.
    """

    heldout_array = np.asarray(heldout_data)
    if heldout_array.ndim != 2:
        raise ValueError(
            "heldout_data must have shape (n_samples, n_features), "
            f"but received {heldout_array.shape}."
        )

    heldout_data_scaled = scaler_model.transform(heldout_array)
    return pca_model.transform(heldout_data_scaled)


def pca_func(data,mode:str='eigenvector_CV',max_features:int=None):
    """The function that chooses which PCA function to run based on the mode.
    Inputs:
        mode: str, the mode of PCA to run. For now the options are: 
        pca_by_variance, rowwise_CV, eigenvector_CV and missmda_CV"""
    
    if max_features is None:
        max_features = data.shape[1]-1
    if mode == "pca_by_variance":
        return pca_by_variance(data)

    if mode == "eigenvector_CV":
        output = eigenvector_pca_cv(data,max_components=max_features,method_pca = 'SVD')
    if mode == "missmda_CV":
        output = estimate_ncp_pca(data,method_cv='kfold',ncp_max=max_features,method='EM')
    
    selected_n_components = output.selected_n_components
    
    scaler = StandardScaler()
    neural_data_scaled = scaler.fit_transform(data)
    pca_model = PCA(n_components=selected_n_components, svd_solver="full")
    transformed_data = pca_model.fit_transform(neural_data_scaled)
    
    return {
        "pca": pca_model,
        "scaler": scaler,
        "transformed_data": transformed_data,
        "selected_n_components": selected_n_components,
    }


def main(
    subj_id: str,
    hdf_path: str | Path,
    pkl_info_path: str | Path,
    neural_data_path: str | Path,
    variance_threshold: float = 0.99,
    save_models_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit PCA on unique images and measure each PC on shared held-out data.

    Inputs:
        subj_id: str, subject identifier used in saved filenames.
        hdf_path: str or Path, path to the NSD stimulus HDF5 file.
        pkl_info_path: str or Path, path to the NSD stimulus-information pickle.
        neural_data_path: str or Path, path to the subject's neural-data Zarr.
        variance_threshold: float in (0, 1], training variance fraction used to
            choose the number of retained PCs.
        save_models_path: str, Path, or None, output directory for the PCA
            model, scaler, and held-out variance CSV. When None, nothing is
            written to disk.

    Output:
        results: dict[str, Any], containing the fitted models, unique and
            held-out PC scores, held-out variance table, and optional saved
            file paths.

    Held-out explained-variance ratios use the total sample variance of the
    held-out data after applying the training scaler as their denominator.
    PC indices in the saved table are one-based.
    """

    data_split = split_unique_shared(
        subj_id,
        hdf_path,
        pkl_info_path,
        neural_data_path,
        variance_threshold=variance_threshold,
    )
    pca_results = pca_func(
        data_split["unique_neural_data"]
    )
    heldout_scores = heldout_pca(
        pca_results["pca"],
        pca_results["scaler"],
        data_split["shared_neural_data"],)

    if heldout_scores.shape[0] < 2:
        raise ValueError(
            "At least two shared held-out samples are required to estimate "
            "explained variance.")

    heldout_scaled = pca_results["scaler"].transform(
        data_split["shared_neural_data"])
    
    heldout_total_variance = float(
        np.var(heldout_scaled, axis=0, ddof=1).sum())
    if not np.isfinite(heldout_total_variance) or heldout_total_variance <= 0.0:
        raise ValueError(
            "Held-out standardized data must have positive finite variance.")

    heldout_component_variance = np.var(heldout_scores, axis=0, ddof=1)
    heldout_explained_ratio = (
        heldout_component_variance / heldout_total_variance)
    
    variance_table = pd.DataFrame(
        {
            "pc_index": np.arange(1, heldout_scores.shape[1] + 1),
            "heldout_component_variance": heldout_component_variance,
            "heldout_explained_variance_ratio": heldout_explained_ratio,
            "heldout_explained_variance_percent": (
                100.0 * heldout_explained_ratio
            ),
            "heldout_cumulative_explained_variance_ratio": np.cumsum(
                heldout_explained_ratio
            ),
            "training_explained_variance_ratio": (
                pca_results["pca"].explained_variance_ratio_
            ),
        }
    )

    saved_paths: dict[str, Path] = {}
    if save_models_path is not None:
        output_dir = Path(save_models_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = {
            "pca_model": output_dir / f"{subj_id}_pca_model.pkl",
            "scaler_model": output_dir / f"{subj_id}_scaler_model.pkl",
            "heldout_variance_csv": (
                output_dir / f"{subj_id}_heldout_pca_variance_explained.csv"
            ),
        }
        joblib.dump(pca_results["pca"], saved_paths["pca_model"])
        joblib.dump(pca_results["scaler"], saved_paths["scaler_model"])
        variance_table.to_csv(saved_paths["heldout_variance_csv"], index=False)

    print("PCA on unique data completed.")
    print("PCA on shared held-out data completed.")
    if "heldout_variance_csv" in saved_paths:
        print(
            "Held-out variance explained saved to "
            f"{saved_paths['heldout_variance_csv']}."
        )

    return {
        "pca_model": pca_results["pca"],
        "scaler_model": pca_results["scaler"],
        "unique_pca_scores": pca_results["transformed_data"],
        "heldout_pca_scores": heldout_scores,
        "heldout_variance_explained": variance_table,
        "saved_paths": saved_paths,
    }


if __name__ == "__main__":
    subject_id = "subj01"
    stimulus_hdf_path = (
        "/groups/golan_neurogroup/bml_group/datasets/nsddata/"
        "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
    )
    stimulus_info_path = (
        "/groups/golan_neurogroup/bml_group/datasets/nsddata/nsddata/"
        "experiments/nsd/nsd_stim_info_merged.pkl"
    )
    subject_neural_data_path = (
        "/groups/golan_neurogroup/bml_group/datasets/nsddata/otc_betas/"
        "otc_betas_per_stim/subj01_OTC_betas_per_stimulus.zarr"
    )

    save_path = '/home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/subj_PCs/subj01_324_pcs'
    main(
        subj_id=subject_id,
        hdf_path=stimulus_hdf_path,
        pkl_info_path=stimulus_info_path,
        neural_data_path=subject_neural_data_path,
        variance_threshold=0.60,
        save_models_path=Path(save_path),
    )
