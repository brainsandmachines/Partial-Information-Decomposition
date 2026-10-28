"""Eigenvector cross-validation: predict each voxel of a held-out image from its others.

Two fold schemes share the same PRESS kernel (``pipeline.rank_selection.press``):
leave-one-image-out (``eigenvector_pca_cv``; PCA basis, optional GPU and
row-level checkpoints) and K image folds (``eigenvector_kfold_curve``; PCA or
GSN-signal basis, held-out images averaged over their repeats).
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

import numpy as np

from pipeline.rank_selection.data import RankData
from pipeline.rank_selection.methods import gsn
from pipeline.rank_selection.methods.eigenvector_checkpoint import (
    build_checkpoint_metadata,
    load_eigenvector_pca_checkpoint,
    write_eigenvector_pca_checkpoint,
)
from pipeline.rank_selection.press import (
    TORCH_IMPORT_ERROR,
    direct_press,
    eigenvector_press,
    is_orthonormal,
    to_numpy,
    torch,
)
from pipeline.rank_selection.result import RankResult, one_standard_error_rank

SKLEARN_IMPORT_ERROR = None
try:
    from sklearn.decomposition import PCA
except Exception as error:
    PCA = None
    SKLEARN_IMPORT_ERROR = error

PCAFitFunction = Callable[[np.ndarray, int], np.ndarray]
BASES = ("pca", "gsn")


@dataclass
class EigenvectorPCACVResult:
    selected_n_components: int
    press: np.ndarray
    msep: np.ndarray
    n_samples: int
    n_features: int
    time: float
    max_components: int


def fit_pca_loadings_svd(
    X_train: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Fit PCA loadings with SVD.

    Input: X_train is np.ndarray, or torch.Tensor for the GPU path, with shape
        (n_train_samples, n_features); n_components is int.
    Output: loading matrix of the same type as X_train with shape
        (n_features, n_components).
    """
    if torch is not None and isinstance(X_train, torch.Tensor):
        # On CUDA, the QR-based gesvd driver is exact and faster than the
        # default Jacobi driver for wide float64 matrices.
        driver = "gesvd" if X_train.is_cuda else None
        # (n_train_samples, n_features) -> Vt: (min(n_train_samples, n_features), n_features)
        _, _, Vt = torch.linalg.svd(X_train, full_matrices=False, driver=driver)
    else:
        # (n_train_samples, n_features) -> Vt: (min(n_train_samples, n_features), n_features)
        _, _, Vt = np.linalg.svd(X_train, full_matrices=False)
    # (min(n_train_samples, n_features), n_features) -> (n_features, n_components)
    return Vt[:n_components, :].T


def fit_pca_loadings_sklearn(
    X_train: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Fit PCA loadings with sklearn.

    Input: X_train is np.ndarray with shape (n_train_samples, n_features);
        n_components is int.
    Output: np.ndarray with shape (n_features, n_components).
    """
    if PCA is None:
        raise ImportError(
            "sklearn.decomposition.PCA is required for fit_pca_loadings_sklearn."
        ) from SKLEARN_IMPORT_ERROR

    model = PCA(n_components=n_components, svd_solver="full")
    model.fit(X_train)
    return model.components_.T


def _check_loadings(
    P: "np.ndarray | torch.Tensor",
    n_features: int,
    n_components: int,
) -> None:
    """Check the type and shape of a loading matrix returned by pca_fit_fn.

    Input: P is the loading matrix; n_features and n_components are int, the
        expected shape (n_features, n_components).
    Output: None. Raises TypeError or ValueError when P is invalid.
    """
    is_tensor = torch is not None and isinstance(P, torch.Tensor)
    if not (isinstance(P, np.ndarray) or is_tensor):
        raise TypeError("pca_fit_fn must return a NumPy array or torch tensor.")
    if tuple(P.shape) != (n_features, n_components):
        raise ValueError(
            f"pca_fit_fn returned shape {tuple(P.shape)}, "
            f"but expected {(n_features, n_components)}."
        )


def _eigenvector_pca_cv_sample_press(
    X: "np.ndarray | torch.Tensor",
    sample_index: int,
    max_components: int,
    pca_fit_fn: PCAFitFunction,
    center: bool,
    scale: bool,
    include_zero_components: bool,
    method_pca: str | None,
    eps: float,
) -> np.ndarray:
    """Calculate one held-out sample's PRESS contribution.

    Input: X is np.ndarray, or torch.Tensor for the GPU path, with shape
        (n_samples, n_features); sample_index and max_components are int;
        pca_fit_fn is PCAFitFunction; center, scale, and
        include_zero_components are bool; method_pca is str or None; eps is float.
    Output: np.ndarray of shape (max_components + 1,). With method_pca="SVD"
        and orthonormal loadings, all component counts come from one SVD
        through running residuals and leverages. Otherwise each count uses the
        leverage formula, falling back to the direct per-feature
        pseudoinverse calculation for non-orthonormal loadings.
    """
    n_features = X.shape[1]
    sample_press = np.zeros(max_components + 1, dtype=float)
    tolerance = max(1e-8, eps)

    # 1. Hold out one sample.
    if torch is not None and isinstance(X, torch.Tensor):
        X_train = torch.cat((X[:sample_index], X[sample_index + 1:]))  # (n_samples, n_features) -> (n_samples - 1, n_features)
    else:
        X_train = np.delete(X, sample_index, axis=0)  # (n_samples, n_features) -> (n_samples - 1, n_features)
    x_test = X[sample_index]  # (n_samples, n_features) -> (n_features,)

    # 2. Center and scale with training statistics only.
    if center:
        train_mean = X_train.mean(0)
        X_train = X_train - train_mean
        x_test = x_test - train_mean

    if scale:
        if torch is not None and isinstance(X_train, torch.Tensor):
            train_std = X_train.std(0, correction=1)
        else:
            train_std = X_train.std(axis=0, ddof=1)
        train_std[train_std < eps] = 1.0
        X_train = X_train / train_std
        x_test = x_test / train_std

    sample_press[0] = float((x_test ** 2).sum()) if include_zero_components else np.nan
    use_full_svd = method_pca == "SVD"
    if use_full_svd:
        P_full = pca_fit_fn(X_train, max_components)
        _check_loadings(P_full, n_features, max_components)

        # 3. Fast path. SVD loadings are nested and orthonormal, so one SVD
        #    gives every component count through the shared PRESS kernel;
        #    counts where some leverage reaches 1 use the direct calculation.
        if is_orthonormal(P_full, tolerance):
            # (n_features,) -> (1, n_features); press, stable: (max_components,)
            press, stable = eigenvector_press(x_test[None, :], P_full, tolerance)
            sample_press[1:] = press
            for f in np.flatnonzero(~stable) + 1:
                sample_press[f] = direct_press(
                    to_numpy(x_test), to_numpy(P_full[:, :f]), eps
                )
            return sample_press

    # 4. General path: refit for each component count, or slice
    #    non-orthonormal loadings.
    x_test = to_numpy(x_test)
    for f in range(1, max_components + 1):
        if f % 10 == 0:
            print(f"Processing component {f} of {max_components}...✅")
        P = P_full[:, :f] if use_full_svd else pca_fit_fn(X_train, f)
        _check_loadings(P, n_features, f)
        P = to_numpy(P)

        if is_orthonormal(P, tolerance):
            raw_error = x_test - (x_test @ P) @ P.T  # (n_features,) -> (n_features,)
            leverage = np.sum(P * P, axis=1)  # (n_features, f) -> (n_features,)
            denominator = 1.0 - leverage
            if np.all(np.abs(denominator) > tolerance):
                loo_error = raw_error / denominator
                sample_press[f] = np.sum(loo_error ** 2)
                continue

        sample_press[f] = direct_press(x_test, P, eps)

    return sample_press


def eigenvector_pca_cv(
    X: np.ndarray,
    max_components: int | None = None,
    pca_fit_fn: PCAFitFunction | None = None,
    center: bool = True,
    scale: bool = False,
    include_zero_components: bool = True,
    method_pca: str | None = None,
    eps: float = 1e-12,
    checkpoint_csv_path: str | Path | None = None,
    device: str | None = None,
) -> EigenvectorPCACVResult:
    """Run eigenvector cross-validation for PCA component selection.

    Input: X is np.ndarray with shape (n_samples, n_features); max_components
        is int or None; pca_fit_fn is PCAFitFunction or None; center, scale, and
        include_zero_components are bool; method_pca is str or None; eps is
        float; checkpoint_csv_path is str, Path, or None for row-level resume;
        device is str or None. None runs in NumPy. A torch device such as
        "cuda" runs the SVD and PRESS in float64 torch on that device; it
        requires method_pca="SVD" and pca_fit_fn=fit_pca_loadings_svd. The
        device is not part of the checkpoint metadata, so a NumPy checkpoint
        can be resumed on a GPU.
    Output: EigenvectorPCACVResult with selected count, PRESS, MSEP, dimensions,
        time placeholder, and max_components.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

    n_samples, n_features = X.shape
    if n_samples < 3:
        raise ValueError("Need at least 3 samples.")
    if n_features < 2:
        raise ValueError("Need at least 2 features.")

    max_allowed = min(n_samples - 2, n_features - 1)
    if max_components is None:
        max_components = max_allowed
    if max_components < 1:
        raise ValueError("max_components must be at least 1.")
    if max_components > max_allowed:
        raise ValueError(
            f"max_components={max_components} is too large. "
            f"Use max_components <= {max_allowed}."
        )

    if pca_fit_fn is None:
        pca_fit_fn = fit_pca_loadings_svd
        print("Using default PCA fitting function (SVD)!!!")

    X_backend = X
    if device is not None:
        if torch is None:
            raise ImportError(
                "torch is required when device is set."
            ) from TORCH_IMPORT_ERROR
        if method_pca != "SVD" or pca_fit_fn is not fit_pca_loadings_svd:
            raise ValueError(
                "device requires method_pca='SVD' and "
                "pca_fit_fn=fit_pca_loadings_svd."
            )
        X_backend = torch.as_tensor(X, dtype=torch.float64, device=device)

    completed_press: dict[int, np.ndarray] = {}

    if checkpoint_csv_path is not None:
        metadata = build_checkpoint_metadata(
            X,
            max_components,
            pca_fit_fn,
            center,
            scale,
            include_zero_components,
            method_pca,
            eps,
        )
        completed_press = load_eigenvector_pca_checkpoint(
            checkpoint_csv_path,
            metadata,
            max_components,
        )

    press = np.zeros(max_components + 1, dtype=float)
    for sample_press in completed_press.values():
        press += sample_press
    start = time.time()
    for i in range(n_samples):
        print(f"Processing sample {i + 1} of {n_samples}...")
        if i in completed_press:
            print(f"Skipping sample {i} (already completed)✅")
            continue

        sample_press = _eigenvector_pca_cv_sample_press(
            X_backend,
            i,
            max_components,
            pca_fit_fn,
            center,
            scale,
            include_zero_components,
            method_pca,
            eps,
        )
        press += sample_press
        completed_press[i] = sample_press

        if checkpoint_csv_path is not None:
            write_eigenvector_pca_checkpoint(
                checkpoint_csv_path,
                metadata,
                completed_press,
                max_components,
            )

            end = time.time()
            elapsed = end - start
            print("Checkpoint saved. Elapsed time: {:.2f} seconds.".format(elapsed))

    msep = press / (n_samples * n_features)
    selected_n_components = int(
        np.argmin(msep) if include_zero_components else np.argmin(msep[1:]) + 1
    )

    return EigenvectorPCACVResult(
        selected_n_components=selected_n_components,
        press=press,
        msep=msep,
        n_samples=n_samples,
        n_features=n_features,
        max_components=max_components,
        time=time.time() - start,
    )


def eigenvector_kfold_curve(
    responses: np.ndarray,
    basis: str = "pca",
    max_components: int | None = None,
    n_folds: int = 5,
    seed: int = 0,
) -> dict:
    """Eigenvector cross-validation over K held-out image folds.

    In each image fold the basis is fitted on the training images only:
    ``basis="pca"`` uses the right singular vectors of the training images'
    repeat-averaged, training-mean-centered responses; ``basis="gsn"`` uses
    the eigenvectors of GSN's signal covariance (all training repeats). Each
    held-out image is averaged over all its repeats and centered with the
    training mean, then every voxel is predicted from the image's other voxels
    (leave-one-voxel-out PRESS). The count with the lowest MSEP is selected.

    Inputs:
        responses: np.ndarray with shape (n_images, n_voxels, n_repeats); GSN
            needs at least two repeats.
        basis: str in BASES.
        max_components: int or None, largest candidate count; None tests
            n_voxels - 1 (and, for ``"pca"``, at most n_training_images - 1).
        n_folds: int, number of held-out image folds.
        seed: int, random seed used to assign images to folds (the same folds
            as ``repeat_reconstruction_curve`` for the same seed).

    Output:
        results: dict with ``candidate_n_components`` of shape
            (max_components + 1,) starting at 0, ``press`` (summed over images)
            and ``msep`` of that shape, ``fold_msep`` of shape
            (n_folds, max_components + 1), ``msep_standard_errors`` across
            folds, ``unstable_components`` (bool, True where some voxel had
            leverage within 1e-8 of 1; excluded from the selection), and
            ``best_n_components`` / ``selected_n_components`` (the MSEP minimum).
    """
    response_array = np.asarray(responses, dtype=float)
    if response_array.ndim != 3:
        raise ValueError(
            "responses must have shape (n_images, n_voxels, n_repeats)."
        )
    if not np.all(np.isfinite(response_array)):
        raise ValueError("responses must be finite.")
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}, got {basis!r}.")
    n_images, n_voxels, n_repeats = response_array.shape
    if n_images < 3:
        raise ValueError("Component selection requires at least three images.")
    if basis == "gsn" and n_repeats < 2:
        raise ValueError("GSN requires at least two repeats per image.")
    if not isinstance(n_folds, (int, np.integer)):
        raise TypeError("n_folds must be an integer.")
    if not 2 <= n_folds <= n_images:
        raise ValueError("n_folds must be between 2 and n_images.")
    largest = n_voxels - 1
    if basis == "pca":
        min_training = n_images - int(np.ceil(n_images / n_folds))
        largest = min(largest, min_training - 1)
    if max_components is None:
        max_components = largest
    if not isinstance(max_components, (int, np.integer)):
        raise TypeError("max_components must be an integer or None.")
    if not 1 <= max_components <= largest:
        raise ValueError(f"max_components must be between 1 and {largest}.")

    tolerance = 1e-8
    rng = np.random.default_rng(seed)
    # (n_images,) -> n_folds arrays with shape (n_validation_images,)
    validation_folds = np.array_split(rng.permutation(n_images), n_folds)

    fold_press = np.zeros((n_folds, max_components + 1))
    fold_sizes = np.empty(n_folds)
    unstable_components = np.zeros(max_components + 1, dtype=bool)
    for fold_index, validation_indices in enumerate(validation_folds):
        print(f"Eigenvector CV ({basis} basis): fold {fold_index + 1} of {n_folds}...")
        training_mask = np.ones(n_images, dtype=bool)
        training_mask[validation_indices] = False
        training_responses = response_array[training_mask]
        # (n_training_images, n_voxels, n_repeats) -> (n_voxels,)
        training_mean = training_responses.mean(axis=(0, 2))

        # 1. Basis from the training images only.
        if basis == "pca":
            # (n_training_images, n_voxels, n_repeats) -> (n_training_images, n_voxels)
            training_average = training_responses.mean(axis=2) - training_mean
            # (n_training_images, n_voxels) -> Vt: (min(n_training_images, n_voxels), n_voxels)
            _, _, Vt = np.linalg.svd(training_average, full_matrices=False)
            basis_vectors = Vt[:max_components].T  # -> (n_voxels, max_components)
        else:
            gsn_results = gsn.estimate_gsn_covariances(training_responses)
            pca_results = gsn.pca_signal_noise_covariances(
                gsn_results["cSb"],
                gsn_results["cNb"],
            )
            # (n_voxels, n_voxels) -> (n_voxels, max_components)
            basis_vectors = pca_results["signal_eigenvectors"][:, :max_components]

        # 2. Held-out images: average all repeats, center with training mean.
        # (n_validation_images, n_voxels, n_repeats) -> (n_validation_images, n_voxels)
        heldout = response_array[validation_indices].mean(axis=2) - training_mean
        fold_sizes[fold_index] = heldout.shape[0]
        fold_press[fold_index, 0] = np.sum(heldout ** 2)

        # 3. Leave-one-voxel-out PRESS for every count (shared kernel).
        press, stable = eigenvector_press(heldout, basis_vectors, tolerance)
        unstable_components[1:] |= ~stable
        fold_press[fold_index, 1:] = np.where(stable, press, 0.0)

    # (n_folds, max_components + 1) -> (max_components + 1,)
    press = fold_press.sum(axis=0)
    msep = press / (n_images * n_voxels)
    # (n_folds, max_components + 1) / (n_folds, 1) -> (n_folds, max_components + 1)
    fold_msep = fold_press / (fold_sizes[:, None] * n_voxels)
    # (n_folds, max_components + 1) -> (max_components + 1,)
    msep_standard_errors = fold_msep.std(axis=0, ddof=1) / np.sqrt(n_folds)
    press[unstable_components] = np.nan
    msep[unstable_components] = np.nan
    fold_msep[:, unstable_components] = np.nan
    msep_standard_errors[unstable_components] = np.nan
    best_n_components = int(np.nanargmin(msep))

    return {
        "selected_n_components": best_n_components,
        "best_n_components": best_n_components,
        "candidate_n_components": np.arange(max_components + 1),
        "press": press,
        "msep": msep,
        "fold_msep": fold_msep,
        "msep_standard_errors": msep_standard_errors,
        "unstable_components": unstable_components,
    }


def run(
    data: RankData,
    basis: str = "pca",
    folds: int | str = 5,
    max_components: int | None = None,
    seed: int = 0,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> RankResult:
    """Run eigenvector cross-validation on a dataset.

    Inputs:
        data: RankData; responses are averaged over repeats.
        basis: str in BASES; ``"gsn"`` only with K folds.
        folds: int (number of image folds) or ``"loo"`` (leave-one-image-out,
            ``eigenvector_pca_cv`` with PCA basis, centering, no scaling).
        max_components: int or None.
        seed: int fold seed (ignored by ``"loo"``).
        device: str or None, torch device for ``"loo"`` (e.g. ``"cuda"``).
        checkpoint_path: str, Path, or None, row-level checkpoint for ``"loo"``.
    Output: RankResult with ranks 0..max_components and MSEP as score; the
        selected rank is the MSEP minimum.
    """
    method = f"eigenvector[basis={basis},folds={folds}]"
    if folds == "loo":
        if basis != "pca":
            raise ValueError("Leave-one-image-out eigenvector CV supports only basis='pca'.")
        result = eigenvector_pca_cv(
            data.averaged(),
            max_components=max_components,
            method_pca="SVD",
            center=True,
            scale=False,
            checkpoint_csv_path=checkpoint_path,
            device=device,
        )
        return RankResult(
            method=method,
            dataset=data.name,
            seed=None,
            selected_rank=result.selected_n_components,
            selection_rule="minimum MSEP",
            score_name="MSEP (leave-one-voxel-out)",
            rank=np.arange(result.max_components + 1),
            score=result.msep,
            best_rank=result.selected_n_components,
            true_rank=data.true_rank,
            info={"runtime_seconds": result.time},
        )

    curve = eigenvector_kfold_curve(
        data.responses, basis=basis, max_components=max_components,
        n_folds=folds, seed=seed,
    )
    return RankResult(
        method=method,
        dataset=data.name,
        seed=seed,
        selected_rank=curve["selected_n_components"],
        selection_rule="minimum MSEP",
        score_name="MSEP (leave-one-voxel-out)",
        rank=curve["candidate_n_components"],
        score=curve["msep"],
        standard_error=curve["msep_standard_errors"],
        fold_scores=curve["fold_msep"],
        best_rank=curve["best_n_components"],
        one_se_rank=one_standard_error_rank(
            curve["candidate_n_components"], curve["msep"], curve["msep_standard_errors"]
        ),
        true_rank=data.true_rank,
        extra_curves={"unstable": curve["unstable_components"]},
    )
