"""Fit subject-level PCA models and evaluate them on shared held-out images."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

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
    neural_data = np.asarray(target_data["target"])
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
    """Fit centered PCA without per-feature variance standardization.

    Inputs:
        neural_data: np.ndarray of shape (n_samples, n_features), training
            neural responses.
        variance_threshold: float in (0, 1], fraction of training variance
            that the retained components should explain.

    Output:
        pca_results: dict[str, Any] containing the fitted PCA model under
            ``pca``, projected training samples under ``transformed_data``, and
            ``variance_threshold``. Scikit-learn PCA mean-centers columns but
            the input columns are not divided by their standard deviations.
    """

    neural_array = np.asarray(neural_data)
    if neural_array.ndim != 2:
        raise ValueError(
            "neural_data must have shape (n_samples, n_features), "
            f"but received {neural_array.shape}."
        )
    if neural_array.shape[0] < 2 or neural_array.shape[1] < 1:
        raise ValueError("PCA requires at least two samples and one feature.")
    pca_components = None if variance_threshold == 1.0 else variance_threshold
    pca_model = PCA(n_components=pca_components, svd_solver="full")
    transformed_data = pca_model.fit_transform(neural_array)

    return {
        "pca": pca_model,
        "transformed_data": transformed_data,
        "variance_threshold": variance_threshold,
    }


def heldout_pca(
    pca_model: PCA,
    heldout_data: np.ndarray,
) -> np.ndarray:
    """Project raw held-out responses with a fitted centered PCA model.

    Inputs:
        pca_model: PCA, fitted PCA model learned from the training data.
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

    return pca_model.transform(heldout_array)


def pca_func(
    data: np.ndarray,
    mode: str = "eigenvector_CV",
    max_features: int | None = None,
    variance_threshold: float = 0.99,
) -> dict[str, Any]:
    """Select a component count and fit centered, unstandardized PCA.

    Inputs:
        data: np.ndarray with samples in rows and neural features in columns.
        mode: str selecting ``pca_by_variance``, ``eigenvector_CV``, or
            ``missmda_CV``.
        max_features: int or None limiting candidate PCA components.
        variance_threshold: float training-variance fraction used by
            ``pca_by_variance``.

    Outputs:
        dict[str, Any] containing the fitted centered PCA, unstandardized
        training scores, and component-selection metadata.
    """

    data_array = np.asarray(data)
    if data_array.ndim != 2:
        raise ValueError("data must be a two-dimensional array.")
    if max_features is None:
        max_features = data_array.shape[1] - 1
    if mode == "pca_by_variance":
        print("Running PCA by variance threshold")
        return pca_by_variance(
            data_array,
            variance_threshold=variance_threshold,
        )

    if mode == "eigenvector_CV":
        print("Running Eigenvector PCA with cross-validation")
        csv_path = '/home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/subj_PCs/saved_pcs/eigenvector_max=100/checkpointmax=100.csv'
        output = eigenvector_pca_cv(
            data_array,
            max_components=max_features,
            method_pca="SVD",
            checkpoint_csv_path=csv_path,
        )
        selected_n_components = int(output.selected_n_components)
    elif mode == "missmda_CV":
        print("Running MissMDA PCA with cross-validation")
        output = estimate_ncp_pca(
            data_array,
            method_cv="Kfold",
            ncp_max=max_features,
            method="EM",
            scale=False,
            verbose=True,
            p_na=0.05,
            nbsim=5,
        )
        selected_n_components = int(output["ncp"])
    else:
        raise ValueError(f"Unsupported PCA mode: {mode!r}.")

    pca_model = PCA(n_components=selected_n_components, svd_solver="full")
    transformed_data = pca_model.fit_transform(data_array)

    return {
        "pca": pca_model,
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
    max_features: int | None = None,
    pca_mode: str = "missmda_CV",
) -> dict[str, Any]:
    """Fit centered unstandardized PCA and evaluate shared held-out data.

    Inputs:
        subj_id: str, subject identifier used in saved filenames.
        hdf_path: str or Path, path to the NSD stimulus HDF5 file.
        pkl_info_path: str or Path, path to the NSD stimulus-information pickle.
        neural_data_path: str or Path, path to the subject's neural-data Zarr.
        variance_threshold: float in (0, 1], training variance fraction used to
            choose the number of retained PCs.
        save_models_path: str, Path, or None, output directory for the PCA
            model and held-out variance CSV. When None, nothing is written.
        max_features: int or None, maximum number of features to use for PCA.
        pca_mode: str component-selection method passed to ``pca_func``.

    Output:
        results: dict[str, Any], containing the fitted models, unique and
            held-out PC scores, held-out variance table, and optional saved
            file paths.

    Held-out explained-variance ratios use the total raw held-out sample
    variance as their denominator. PCA still applies its fitted training mean;
    no column is divided by its standard deviation. PC indexes are one-based.
    """

    data_split = split_unique_shared(
        subj_id,
        hdf_path,
        pkl_info_path,
        neural_data_path,
        variance_threshold=variance_threshold,
    )

    print(f"Loaded subject {subj_id} with {data_split['unique_neural_data'].shape[0]} unique images and {data_split['shared_neural_data'].shape[0]} shared held-out images.")

    print(f"\n Fitting PCA on unique images")
    pca_results = pca_func(
        mode=pca_mode,
        data=data_split["unique_neural_data"],
        max_features=max_features,
        variance_threshold=variance_threshold,
    )

    heldout_scores = heldout_pca(
        pca_results["pca"],
        data_split["shared_neural_data"],
    )

    if heldout_scores.shape[0] < 2:
        raise ValueError(
            "At least two shared held-out samples are required to estimate "
            "explained variance.")

    heldout_total_variance = float(
        np.var(data_split["shared_neural_data"], axis=0, ddof=1).sum()
    )
    if not np.isfinite(heldout_total_variance) or heldout_total_variance <= 0.0:
        raise ValueError(
            "Held-out data must have positive finite total variance."
        )

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
            "heldout_variance_csv": (
                output_dir / f"{subj_id}_heldout_pca_variance_explained.csv"
            ),
        }
        joblib.dump(pca_results["pca"], saved_paths["pca_model"])
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

    save_path = '/home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/subj_PCs/saved_pcs/missmda_max=100_no_variance_standardization'
    main(
        subj_id=subject_id,
        hdf_path=stimulus_hdf_path,
        pkl_info_path=stimulus_info_path,
        neural_data_path=subject_neural_data_path,
        variance_threshold=1.0,
        save_models_path=Path(save_path),
        max_features=100,
        pca_mode="missmda_CV",
    )
