"""PyTorch port of missMDA's ``estim_ncpPCA`` (Josse & Husson, 2012, 2016).

The repeat-averaged responses form one (images x voxels) matrix, as in R.

- ``method_cv="kfold"``: hide a random ``p_na`` fraction of the cells,
  impute them with ``imputePCA`` (iterative, optionally regularized, SVD),
  and sum the squared error of the hidden cells; average over ``nbsim``
  random maskings. Selected rank: the minimum.
- ``method_cv="gcv"``: generalized cross-validation of the regularized
  rank-q fit. Selected rank: the first local minimum (R's rule).

Ported from github.com/husson/missMDA (R/estim_ncpPCA.R, R/imputePCA.R) and
FactoMineR's svd.triplet, with uniform row weights and no supplementary rows
or columns. Differences from R:

- The SVD is exact (``torch.linalg.svd``); R uses irlba, an iterative
  truncated SVD, when ncp < half the smaller dimension.
- GCV on complete data: R's ``imputePCA`` returns the bare matrix when
  nothing is missing, so ``estim_ncpPCA(method.cv="gcv")`` fails on complete
  data. Here the rank-q fit is the one ``imputePCA``'s loop converges to
  without missing cells (a single regularized SVD fit), and the criterion is
  R's formula with zero missing cells.
- The random masks come from NumPy, so K-fold numbers match R in
  distribution, not draw for draw.
- ``method.cv="loo"`` (one imputation per cell) is not ported.
"""

import warnings

import numpy as np
import torch

from pipeline.rank_selection.data import RankData
from pipeline.rank_selection.result import RankResult, one_standard_error_rank

METHODS_CV = ("gcv", "kfold")


def _regularized_noise_variance(
    singular_values: torch.Tensor,
    ncp: int,
    n_rows: int,
    n_columns: int,
    regularized: bool,
    coeff_ridge: float,
) -> torch.Tensor:
    """Estimate missMDA's noise variance sigma^2 that shrinks the kept singular values.

    Inputs:
        singular_values: torch.Tensor (..., min(n_rows, n_columns)), singular
            values of the standardized matrix divided by sqrt(n_rows), in
            decreasing order (FactoMineR's row-weighted SVD with weights 1/n).
        ncp: int, number of kept components, 1 <= ncp < len(singular_values).
        n_rows: int, number of rows (images).
        n_columns: int, number of columns (voxels).
        regularized: bool, False gives the EM version (sigma^2 = 0).
        coeff_ridge: float, multiplier of sigma^2 (R's ``coeff.ridge``).

    Output:
        sigma2: torch.Tensor (...,), the noise variance, capped at the
            (ncp + 1)-th squared singular value as in R.
    """

    if not regularized:
        return torch.zeros(singular_values.shape[:-1], dtype=singular_values.dtype,
                           device=singular_values.device)
    squared = singular_values ** 2
    # (..., k) -> (...,): variance left after the first ncp components (R's sumvp - sum(vs[1:ncp]^2))
    tail = squared[..., ncp:].sum(-1)
    degrees = (n_rows - 1) * n_columns - (n_rows - 1) * ncp - n_columns * ncp + ncp ** 2
    sigma2 = n_rows * n_columns / min(n_columns, n_rows - 1) * tail / degrees
    return torch.minimum(sigma2 * coeff_ridge, squared[..., ncp])


def impute_pca(
    values: torch.Tensor,
    missing: torch.Tensor,
    ncp: int,
    scale: bool = True,
    regularized: bool = True,
    coeff_ridge: float = 1.0,
    threshold: float = 1e-6,
    maxiter: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Impute missing cells of a batch of matrices, as missMDA's ``imputePCA``.

    Each matrix is iterated separately: fill the missing cells with the current
    fit, re-center (and re-scale) the columns, take the SVD, and refit with the
    first ``ncp`` components, shrinking each singular value s to
    (s^2 - sigma^2) / s. A matrix stops after at least 5 iterations once the
    relative change of the observed-cell residual (or the residual itself)
    falls below ``threshold``, or after ``maxiter`` iterations.

    Inputs:
        values: torch.Tensor (batch, n_rows, n_columns), float; entries at
            ``missing`` are ignored.
        missing: torch.Tensor (batch, n_rows, n_columns), bool, True where a
            cell is missing. Every matrix needs at least one missing cell
            (R returns the input unchanged otherwise).
        ncp: int, number of components, 0 <= ncp <= min(n_rows - 2, n_columns - 1).
            0 imputes each column's observed mean.
        scale: bool, standardize the columns (R's default True).
        regularized: bool, True for R's "Regularized", False for "EM".
        coeff_ridge: float, R's ``coeff.ridge``.
        threshold: float, convergence threshold.
        maxiter: int, maximum iterations per matrix.

    Output:
        completed: torch.Tensor (batch, n_rows, n_columns), ``values`` with
            the missing cells imputed (R's ``completeObs``).
        fitted: torch.Tensor (batch, n_rows, n_columns), the final low-rank
            fit in the original units (R's ``fittedX``).
        n_iterations: torch.Tensor (batch,), int64, iterations run per matrix.
    """

    _, n_rows, n_columns = values.shape
    if not 0 <= ncp <= min(n_rows - 2, n_columns - 1):
        raise ValueError(f"ncp={ncp} must lie in [0, {min(n_rows - 2, n_columns - 1)}].")
    present = ~missing
    zeroed = values.masked_fill(missing, 0.0)
    # (batch, n, p) -> (batch, p): observed count, mean, and RMS deviation of each column
    n_present = present.sum(1)
    mean = zeroed.sum(1) / n_present
    centered = (zeroed - mean[:, None, :]).masked_fill(missing, 0.0)
    sd = torch.sqrt((centered ** 2).sum(1) / n_present)
    standardized = centered / sd[:, None, :] if scale else centered
    fitted = standardized.clone()

    batch = values.shape[0]
    old = torch.full((batch,), float("inf"), dtype=values.dtype, device=values.device)
    n_iterations = torch.zeros(batch, dtype=torch.long, device=values.device)
    active = torch.full((batch,), ncp > 0, dtype=torch.bool, device=values.device)
    root_n = n_rows ** 0.5
    while bool(active.any()):
        index = torch.nonzero(active).squeeze(1)  # (n_active,)
        block_missing = missing[index]
        # 1. Fill the missing cells with the current fit and return to original units.
        filled = torch.where(block_missing, fitted[index], standardized[index])
        if scale:
            filled = filled * sd[index][:, None, :]
        filled = filled + mean[index][:, None, :]
        # 2. Re-center and re-scale every column, now over all rows.
        # (n_active, n, p) -> (n_active, p)
        block_mean = filled.mean(1)
        filled = filled - block_mean[:, None, :]
        block_sd = torch.sqrt((filled ** 2).mean(1))
        if scale:
            filled = filled / block_sd[:, None, :]
        # 3. Row-weighted SVD (weights 1/n) and the shrunk rank-ncp fit.
        # (n_active, n, p) -> (n_active, n, k), (n_active, k), (n_active, k, p); k = min(n, p)
        left, singular, right = torch.linalg.svd(filled / root_n, full_matrices=False)
        sigma2 = _regularized_noise_variance(singular, ncp, n_rows, n_columns,
                                             regularized, coeff_ridge)
        kept = singular[:, :ncp]
        shrunk = (kept ** 2 - sigma2[:, None]) / kept  # (n_active, ncp)
        # (n_active, n, ncp) @ (n_active, ncp, p) -> (n_active, n, p)
        block_fitted = root_n * (left[:, :, :ncp] * shrunk[:, None, :]) @ right[:, :ncp, :]
        # 4. Convergence on the observed cells (R's objective uses row weights 1/n).
        residual = (filled - block_fitted).masked_fill(block_missing, 0.0)
        objective = (residual ** 2).sum((1, 2)) / n_rows  # (n_active, n, p) -> (n_active,)
        criterion = torch.abs(1 - objective / old[index])

        standardized[index] = filled
        fitted[index] = block_fitted
        mean[index] = block_mean
        sd[index] = block_sd
        old[index] = objective
        n_iterations[index] += 1
        # R counts nb.iter from 1, so "nb.iter > 5" means at least 5 iterations.
        done = (~torch.isnan(criterion)
                & ((criterion < threshold) | (objective < threshold))
                & (n_iterations[index] >= 5))
        done |= n_iterations[index] >= maxiter
        active[index[done]] = False

    n_stopped = int((n_iterations >= maxiter).sum())
    if n_stopped:
        warnings.warn(f"impute_pca: {n_stopped} matrices stopped after {maxiter} iterations.")
    if scale:
        standardized = standardized * sd[:, None, :]
        fitted = fitted * sd[:, None, :]
    # The imputed cells are the fit inserted at the start of the last iteration, as in R.
    completed = torch.where(missing, standardized + mean[:, None, :], values)
    fitted = fitted + mean[:, None, :]
    return completed, fitted, n_iterations


def gcv_criterion(
    values: torch.Tensor,
    ranks: np.ndarray,
    scale: bool = True,
    regularized: bool = True,
    coeff_ridge: float = 1.0,
) -> np.ndarray:
    """Generalized cross-validation criterion of each rank on a complete matrix.

    For rank q, the regularized rank-q fit F_q (in original units) gives
    GCV(q) = mean(((n p) (X - F_q) / ((n - 1) p - q (n + p - q - 1)))^2), R's
    formula with no missing cells. Rank 0 is the column-centered mean square,
    unnormalized, as in R. All ranks come from one SVD: the residual of rank q
    is sqrt(n) sum_i r_i u_i v_i^T with r_i = sigma^2 / s_i for kept components
    and s_i otherwise, so its squared norm in original units is
    n sum_i r_i^2 sum_j sd_j^2 v_ij^2.

    Inputs:
        values: torch.Tensor (n_rows, n_columns), complete, float.
        ranks: np.ndarray (n_candidates,), int ranks in
            [0, min(n_rows - 2, n_columns - 1)].
        scale, regularized, coeff_ridge: as in ``impute_pca``.

    Output:
        criterion: np.ndarray (n_candidates,), the GCV value of each rank.
    """

    n_rows, n_columns = values.shape
    # (n, p) -> (p,)
    mean = values.mean(0)
    centered = values - mean
    sd = torch.sqrt((centered ** 2).mean(0))
    standardized = centered / sd if scale else centered
    # (n, p) -> (k,), (k, p); k = min(n, p)
    _, singular, right = torch.linalg.svd(standardized / n_rows ** 0.5, full_matrices=False)
    column_weight = sd ** 2 if scale else torch.ones_like(sd)
    # (k, p) x (p,) -> (k,): squared norm of each right singular vector in original units
    direction_weight = (right ** 2) @ column_weight

    criterion = []
    for rank in ranks:
        rank = int(rank)
        if rank == 0:
            criterion.append(float((centered ** 2).mean()))
            continue
        sigma2 = _regularized_noise_variance(singular, rank, n_rows, n_columns,
                                             regularized, coeff_ridge)
        residual_singular = singular.clone()  # (k,)
        residual_singular[:rank] = sigma2 / singular[:rank]
        squared_norm = n_rows * float((residual_singular ** 2 * direction_weight).sum())
        degrees = (n_rows - 1) * n_columns - rank * (n_rows + n_columns - rank - 1)
        criterion.append((n_rows * n_columns / degrees) ** 2 * squared_norm / (n_rows * n_columns))
    return np.asarray(criterion)


def first_local_minimum(ranks: np.ndarray, criterion: np.ndarray) -> int:
    """Select the rank before the criterion first increases, else the minimum (R's GCV rule).

    Inputs:
        ranks: np.ndarray (n_candidates,), ascending ranks.
        criterion: np.ndarray (n_candidates,).

    Output:
        rank: int, the selected rank.
    """

    increases = np.flatnonzero(np.diff(criterion) > 0)
    return int(ranks[increases[0]] if increases.size else ranks[np.argmin(criterion)])


def kfold_errors(
    values: torch.Tensor,
    ranks: np.ndarray,
    nbsim: int = 100,
    p_na: float = 0.05,
    seed: int = 0,
    scale: bool = True,
    regularized: bool = True,
    coeff_ridge: float = 1.0,
    threshold: float = 1e-4,
    maxiter: int = 1000,
    batch_size: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Squared imputation error of randomly hidden cells for each mask and rank.

    Each of ``nbsim`` masks hides round(p_na n p) cells drawn without
    replacement; every rank imputes them with ``impute_pca`` (rank 0: column
    means of the observed cells). Masks are processed ``batch_size`` at a time.

    Inputs:
        values: torch.Tensor (n_rows, n_columns), complete, float.
        ranks: np.ndarray (n_candidates,), int ranks.
        nbsim: int, number of random masks.
        p_na: float in (0, 1), fraction of hidden cells.
        seed: int, NumPy seed of the masks.
        scale, regularized, coeff_ridge, threshold, maxiter: as in ``impute_pca``
            (threshold 1e-4 is estim_ncpPCA's default).
        batch_size: int, masks imputed together.

    Output:
        errors: np.ndarray (nbsim, n_candidates), sum of squared errors of the
            hidden cells.
        n_iterations: np.ndarray (nbsim, n_candidates), imputation iterations.
    """

    n_rows, n_columns = values.shape
    n_hidden = int(round(p_na * n_rows * n_columns))
    rng = np.random.default_rng(seed)
    masks = np.zeros((nbsim, n_rows * n_columns), dtype=bool)
    for simulation in range(nbsim):
        masks[simulation, rng.choice(n_rows * n_columns, size=n_hidden, replace=False)] = True
    masks = masks.reshape(nbsim, n_rows, n_columns)  # (nbsim, n * p) -> (nbsim, n, p)

    errors = np.zeros((nbsim, len(ranks)))
    n_iterations = np.zeros((nbsim, len(ranks)), dtype=int)
    for start in range(0, nbsim, batch_size):
        stop = min(start + batch_size, nbsim)
        missing = torch.as_tensor(masks[start:stop], device=values.device)
        # (n, p) -> (batch, n, p)
        block = values.expand(stop - start, n_rows, n_columns)
        for column, rank in enumerate(ranks):
            completed, _, iterations = impute_pca(
                block, missing, int(rank), scale=scale, regularized=regularized,
                coeff_ridge=coeff_ridge, threshold=threshold, maxiter=maxiter,
            )
            # (batch, n, p) -> (batch,)
            hidden_error = ((completed - block) ** 2).masked_fill(~missing, 0.0).sum((1, 2))
            errors[start:stop, column] = hidden_error.cpu().numpy()
            n_iterations[start:stop, column] = iterations.cpu().numpy()
        print(f"missMDA K-fold: masks {stop}/{nbsim} done")
    return errors, n_iterations


def run(
    data: RankData,
    method_cv: str = "gcv",
    max_components: int | None = None,
    min_components: int = 0,
    nbsim: int = 100,
    p_na: float = 0.05,
    scale: bool = True,
    regularized: bool = True,
    coeff_ridge: float = 1.0,
    threshold: float = 1e-4,
    maxiter: int = 1000,
    seed: int = 0,
    device: str | None = None,
    batch_size: int = 10,
) -> RankResult:
    """Run the missMDA rank estimate on the repeat-averaged responses.

    Inputs:
        data: RankData; responses are averaged over repeats.
        method_cv: str in METHODS_CV, ``"gcv"`` (R's default) or ``"kfold"``.
        max_components: int or None, R's ``ncp.max``, capped at
            min(n_voxels - 1, n_images - 2) as in R; None uses that cap.
        min_components: int, R's ``ncp.min``.
        nbsim: int, K-fold masks (R's ``nbsim``).
        p_na: float, K-fold fraction of hidden cells (R's ``pNA``).
        scale: bool, standardize the voxels (R's default True).
        regularized: bool, ``"Regularized"`` (True) or ``"EM"`` (False).
        coeff_ridge: float, R's ``coeff.ridge``.
        threshold: float, K-fold imputation convergence threshold.
        maxiter: int, K-fold imputation iteration cap.
        seed: int, K-fold mask seed (GCV is deterministic).
        device: str or None, torch device such as ``"cuda"``; None is CPU.
            Computation is float64.
        batch_size: int, K-fold masks imputed together.

    Output:
        RankResult with ranks min_components..max_components and the
        criterion as score. GCV selects the first local minimum; K-fold
        selects the minimum mean error and also reports the 1-SE rank.
    """

    if method_cv not in METHODS_CV:
        raise ValueError(f"method_cv must be one of {METHODS_CV}, got {method_cv!r}.")
    averaged = data.averaged()  # (n_images, n_voxels, n_repeats) -> (n_images, n_voxels)
    n_images, n_voxels = averaged.shape
    cap = min(n_voxels - 1, n_images - 2)
    upper = cap if max_components is None else min(int(max_components), cap)
    ranks = np.arange(int(min_components), upper + 1)  # (n_candidates,)
    values = torch.as_tensor(averaged, dtype=torch.float64, device=device or "cpu")
    method = f"missmda[cv={method_cv},{'regularized' if regularized else 'em'},scale={scale}]"
    common = dict(method=method, dataset=data.name, rank=ranks, true_rank=data.true_rank)

    if method_cv == "gcv":
        criterion = gcv_criterion(values, ranks, scale, regularized, coeff_ridge)
        return RankResult(
            **common,
            seed=None,
            selected_rank=first_local_minimum(ranks, criterion),
            selection_rule="first local minimum of GCV (missMDA estim_ncpPCA)",
            score_name="GCV criterion",
            score=criterion,
            best_rank=int(ranks[np.argmin(criterion)]),
        )

    errors, n_iterations = kfold_errors(
        values, ranks, nbsim=nbsim, p_na=p_na, seed=seed, scale=scale,
        regularized=regularized, coeff_ridge=coeff_ridge, threshold=threshold,
        maxiter=maxiter, batch_size=batch_size,
    )
    # (nbsim, n_candidates) -> (n_candidates,)
    mean_error = errors.mean(0)
    standard_error = errors.std(0, ddof=1) / np.sqrt(nbsim) if nbsim > 1 else np.zeros_like(mean_error)
    best = int(ranks[np.argmin(mean_error)])
    return RankResult(
        **common,
        seed=seed,
        selected_rank=best,
        selection_rule="minimum mean squared error of hidden cells (missMDA estim_ncpPCA K-fold)",
        score_name="sum of squared errors of hidden cells (mean over masks)",
        score=mean_error,
        standard_error=standard_error,
        best_rank=best,
        one_se_rank=one_standard_error_rank(ranks, mean_error, standard_error),
        extra_curves={"mean_iterations": n_iterations.mean(0)},
        info={"nbsim": nbsim, "p_na": p_na, "max_iterations": int(n_iterations.max())},
    )
