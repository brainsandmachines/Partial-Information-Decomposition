"""Cross-covariance SVD directions evaluated with the existing GSN CV metric.

Responses are raw (image, voxel, repeat) arrays. No voxel z-scoring is added.
The SVD uses repetition-group-specific training means; reconstruction and its
baseline use the pooled training mean, exactly as in select_gsn_pca_components.
"""

from itertools import combinations
from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator, svds


def fit_cross_decomposition(
    first: np.ndarray,
    second: np.ndarray,
    n_components: int,
    seed: int = 0,
    solver: str = "auto",
) -> dict[str, Any]:
    """Fit training cross-covariance directions with memory-aware SVD.

    Inputs: first, second: finite float arrays (n_train, n_voxels), aligned by
        stimulus; n_components: int in [1, min(n_train - 1, n_voxels)]; seed:
        int for iterative SVD; solver: 'auto', 'iterative', or 'exact'.
    Output: dict with left_basis/right_basis (n_voxels, n_components), descending
        singular_values (n_components,), first_mean/second_mean (n_voxels,),
        and solver (str). Means are fitted only on these training observations.

    Exact mode factors A.T and B.T with reduced QR, then decomposes R_A R_B.T/n.
    Its square workspace has side min(n_train, n_voxels). Iterative mode applies
    C v = A.T (B v)/n through a LinearOperator, using O(n*d + d*K) storage.
    Counts beyond the centered sample-rank bound raise rather than assigning
    scientifically unidentified null-space directions arbitrary ranks.
    """
    # (n_train, n_voxels) -> (n_train, n_voxels)
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.ndim != 2 or second.shape != first.shape or not first.shape[1]:
        raise ValueError("first and second must have matching nonempty (image, voxel) shapes.")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Training responses must be finite.")
    n_train, n_voxels = first.shape
    rank_bound = min(n_train - 1, n_voxels)
    if not isinstance(n_components, (int, np.integer)) or isinstance(n_components, bool):
        raise TypeError("n_components must be an integer.")
    if not 1 <= n_components <= rank_bound:
        raise ValueError(f"n_components must be between 1 and the training rank bound {rank_bound}.")
    if solver not in ("auto", "exact", "iterative"):
        raise ValueError("solver must be 'auto', 'exact', or 'iterative'.")
    if solver == "auto":
        solver = "iterative" if n_components < rank_bound / 2 else "exact"
    # (n_train, n_voxels) -> (n_voxels,)
    first_mean, second_mean = first.mean(axis=0), second.mean(axis=0)
    # (n_train, n_voxels) - (n_voxels,) -> (n_train, n_voxels)
    a, b = first - first_mean, second - second_mean
    if solver == "iterative":
        if n_components >= n_voxels:
            raise ValueError("Iterative SVD requires n_components < n_voxels; use exact or auto.")

        def multiply(v: np.ndarray) -> np.ndarray:
            """Map float v (d,) or (d,q) to C v of the same shape."""
            # (d,q) -> (n_train,q) -> (d,q); also supports vectors (d,).
            return a.T @ (b @ v) / n_train

        def transpose_multiply(v: np.ndarray) -> np.ndarray:
            """Map float v (d,) or (d,q) to C.T v of the same shape."""
            # (d,q) -> (n_train,q) -> (d,q); also supports vectors (d,).
            return b.T @ (a @ v) / n_train

        operator = LinearOperator(
            (n_voxels, n_voxels), matvec=multiply, rmatvec=transpose_multiply,
            matmat=multiply, rmatmat=transpose_multiply, dtype=np.dtype(float),
        )
        # Scalar dimension -> (n_voxels,); explicit start makes ARPACK repeatable.
        start = np.random.default_rng(seed).normal(size=n_voxels)
        # Operator (d,d) -> (d,K), (K,), (K,d).
        left, values, right_t = svds(operator, k=n_components, v0=start, tol=1e-10)
    else:
        # (n_train,d).T -> (d,q), (q,n_train), q=min(n_train,d).
        qa, ra = np.linalg.qr(a.T, mode="reduced")
        qb, rb = np.linalg.qr(b.T, mode="reduced")
        # (q,n_train) @ (n_train,q) -> (q,q).
        core = ra @ rb.T / n_train
        # (q,q) -> (q,q), (q,), (q,q).
        core_left, values, core_right_t = np.linalg.svd(core, full_matrices=False)
        # (d,q) @ (q,K) -> (d,K); (K,q) @ (q,d) -> (K,d).
        left = qa @ core_left[:, :n_components]
        right_t = core_right_t[:n_components] @ qb.T
        # (q,) -> (K,).
        values = values[:n_components]
    # (K,) -> (K,); both solver paths return descending singular values.
    order = np.argsort(values)[::-1]
    return {
        # (d,K) -> (d,K); (K,d) -> (d,K); (K,) -> (K,).
        "left_basis": left[:, order], "right_basis": right_t[order].T,
        "singular_values": values[order], "first_mean": first_mean,
        "second_mean": second_mean, "solver": solver,
    }


def normalized_reconstruction_curve(
    observed: np.ndarray, target: np.ndarray, mean: np.ndarray, basis: np.ndarray,
) -> np.ndarray:
    """Return independent-repeat normalized MSE for K=0,...,basis.shape[1].

    Inputs: observed/target: finite arrays (n_validation, d); mean: training
        mean (d,); basis: orthonormal columns (d,K), supplied by SVD or eigh.
    Output: float array (K+1,), MSE divided by mean((target-mean)**2).
        K=0 represents reconstruction equal to mean and hence has error 1.
        Uses the exact orthogonal-projection SSE identity without constructing
        K separate reconstructed response matrices or a d-by-d projector.
    """
    if (observed.ndim != 2 or observed.shape != target.shape
            or mean.shape != (observed.shape[1],) or basis.ndim != 2
            or basis.shape[0] != observed.shape[1] or not observed.size):
        raise ValueError("Expected observed/target (n,d), mean (d,), basis (d,K).")
    if not all(np.isfinite(array).all() for array in (observed, target, mean, basis)):
        raise ValueError("Reconstruction inputs must be finite.")
    # (n,d) - (d,) -> (n,d).
    residual = target - mean
    baseline_sse = float(np.sum(residual ** 2))
    if baseline_sse == 0:
        raise ValueError("Validation responses contain no reconstructable variance.")
    # (n,d) @ (d,K) -> (n,K).
    scores = (observed - mean) @ basis
    target_scores = residual @ basis
    # (n,K) -> (K,): adding each orthogonal component changes SSE by this amount.
    increments = np.sum(scores ** 2 - 2 * scores * target_scores, axis=0)
    # (K,) -> (K+1,).
    return np.r_[1.0, 1.0 + np.cumsum(increments) / baseline_sse]


def validation_design(
    n_images: int, n_repeats: int, n_folds: int, seed: int,
    repeat_split_weighting: str,
) -> tuple[list[np.ndarray], list[tuple[tuple, tuple]], np.ndarray]:
    """Mirror the unchanged GSN selector's image and directed repetition splits.

    Inputs: n_images/n_repeats/n_folds: int sizes, seed: int, weighting: str
        'equal' or 'input_repeat_count'. Require >=2 training images per fold.
    Output: validation indices (list of n_folds integer arrays), directed repeat
        groups (list of tuple pairs), normalized split weights (n_splits,).
        Repeated calls with the same arguments give identical assignments.
    """
    if n_images < 3 or n_repeats < 2:
        raise ValueError("Require at least three images and two repeats.")
    if not isinstance(n_folds, (int, np.integer)) or isinstance(n_folds, bool):
        raise TypeError("n_folds must be an integer.")
    if not 2 <= n_folds <= n_images:
        raise ValueError("n_folds must be between 2 and n_images.")
    if repeat_split_weighting not in ("equal", "input_repeat_count"):
        raise ValueError("repeat_split_weighting must be 'equal' or 'input_repeat_count'.")
    # (n_images,) -> list of (n_validation,) arrays, exactly as in GSN selector.
    folds = np.array_split(np.random.default_rng(seed).permutation(n_images), n_folds)
    if n_images - max(map(len, folds)) < 2:
        raise ValueError("Each fold requires at least two training images.")
    splits = []
    for group_a in combinations(range(n_repeats), n_repeats // 2):
        group_b = tuple(i for i in range(n_repeats) if i not in group_a)
        if len(group_a) == len(group_b) and group_a > group_b:
            continue
        splits.extend(((group_a, group_b), (group_b, group_a)))
    # n_splits groups -> (n_splits,).
    weights = np.array([len(a) if repeat_split_weighting == "input_repeat_count"
                        else 1 for a, _ in splits], dtype=float)
    # (n_splits,) -> (n_splits,).
    return folds, splits, weights / weights.sum()


def select_cross_decomposition_components(
    responses: np.ndarray, max_components: int | None = None, n_folds: int = 5,
    seed: int = 0, repeat_split_weighting: str = "input_repeat_count",
    solver: str = "auto",
) -> dict[str, Any]:
    """Select cross-decomposition dimensions with the unchanged GSN CV protocol.

    Inputs: responses: finite raw array (images, voxels, repeats); max_components:
        int or None (=voxel count), n_folds/seed: int, weighting: 'equal' or
        'input_repeat_count', solver: 'auto', 'exact', 'iterative'. All candidate
        counts 1..max_components must fit within every training-fold rank bound.
    Output: GSN-compatible selection dict, including candidates (K,), fold errors
        (n_folds,K), means/SE (K,), best argmin count and one-SE selected count.
        Also returns directed errors (n_folds*n_splits,K), ordered fold-major,
        for auditing U/input-A and V/input-B reconstruction separately.
        These are validation curves, not independent final performance estimates.
    """
    # (images,voxels,repeats) -> (images,voxels,repeats).
    responses = np.asarray(responses, dtype=float)
    if responses.ndim != 3 or not responses.shape[1] or not np.isfinite(responses).all():
        raise ValueError("responses must be finite with shape (images, nonempty voxels, repeats).")
    n_images, n_voxels, n_repeats = responses.shape
    folds, splits, weights = validation_design(
        n_images, n_repeats, n_folds, seed, repeat_split_weighting,
    )
    max_components = n_voxels if max_components is None else max_components
    if not isinstance(max_components, (int, np.integer)) or isinstance(max_components, bool):
        raise TypeError("max_components must be an integer or None.")
    rank_bound = min(n_voxels, n_images - max(map(len, folds)) - 1)
    if not 1 <= max_components <= rank_bound:
        raise ValueError(
            f"max_components must be between 1 and {rank_bound}, the smallest training-fold "
            "rank bound. Set the same supported cap for both methods; no silent clipping."
        )
    # Scalar dimensions -> (n_folds,n_splits,K).
    direction_errors = np.empty((n_folds, len(splits), max_components))
    for fold_index, validation_indices in enumerate(folds):
        # Scalar n_images -> (n_images,) -> (n_train,voxels,repeats).
        mask = np.ones(n_images, dtype=bool)
        mask[validation_indices] = False
        training = responses[mask]
        # (images,voxels,repeats) -> (n_validation,voxels,repeats).
        validation = responses[validation_indices]
        # (n_train,voxels,repeats) -> (voxels,), same pooled mean as GSN.
        mean = training.mean(axis=(0, 2))
        # Forward/reverse splits are adjacent; one SVD supplies both bases.
        for split_index in range(0, len(splits), 2):
            group_a, group_b = splits[split_index]
            # (n_train,voxels,n_group) -> (n_train,voxels).
            first = training[:, :, group_a].mean(axis=2)
            second = training[:, :, group_b].mean(axis=2)
            decomposition = fit_cross_decomposition(first, second, max_components, seed, solver)
            # (n_validation,voxels,n_group) -> (n_validation,voxels).
            observed = validation[:, :, group_a].mean(axis=2)
            target = validation[:, :, group_b].mean(axis=2)
            # (K+1,) -> (K,); K=0 is diagnostic only, matching GSN's 1..K grid.
            direction_errors[fold_index, split_index] = normalized_reconstruction_curve(
                observed, target, mean, decomposition["left_basis"],
            )[1:]
            direction_errors[fold_index, split_index + 1] = normalized_reconstruction_curve(
                target, observed, mean, decomposition["right_basis"],
            )[1:]
    # (n_folds,n_splits,K) -> (n_folds,K).
    fold_errors = np.average(direction_errors, axis=1, weights=weights)
    # (n_folds,K) -> (K,).
    means = fold_errors.mean(axis=0)
    standard_errors = fold_errors.std(axis=0, ddof=1) / np.sqrt(n_folds)
    best = int(np.argmin(means))
    threshold = float(means[best] + standard_errors[best])
    selected = int(np.flatnonzero(means <= threshold)[0])
    return {
        "selected_n_components": selected + 1,
        "best_n_components": best + 1,
        # Scalar K -> (K,).
        "candidate_n_components": np.arange(1, max_components + 1),
        "fold_normalized_reconstruction_errors": fold_errors,
        # (n_folds,n_splits,K) -> (n_folds*n_splits,K).
        "direction_normalized_reconstruction_errors": direction_errors.reshape(-1, max_components),
        "mean_normalized_reconstruction_errors": means,
        "reconstruction_error_standard_errors": standard_errors,
        "one_standard_error_threshold": threshold,
        "repeat_splits_weighted_by_input_count": repeat_split_weighting == "input_repeat_count",
        "repeat_split_input_weight_ratio": float(weights.max() / weights.min()),
    }
