import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any
from subj_pc_analysis import split_unique_shared, pca_by_variance, heldout_pca







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
    pca_results = pca_by_variance(
        data_split["unique_neural_data"],
        variance_threshold=variance_threshold,
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