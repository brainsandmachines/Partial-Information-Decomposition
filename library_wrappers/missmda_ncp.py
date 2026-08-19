"""Minimal in-process Python wrapper for R's ``missMDA::estim_ncpPCA``."""

from pathlib import Path
import warnings

import numpy as np

try:
    from rpy2 import robjects as ro
    from rpy2.robjects.packages import importr
except ImportError as error:
    ro = None
    importr = None
    RPY2_IMPORT_ERROR = error
else:
    RPY2_IMPORT_ERROR = None


RSCRIPT = "Rscript"


def estimate_ncp_pca(
    data,
    ncp_min=0,
    ncp_max=5,
    method="Regularized",
    scale=True,
    method_cv="gcv",
    nbsim=100,
    p_na=0.05,
    threshold=1e-4,
    seed=None,
    rscript=RSCRIPT,
    verbose=False,
):
    """Call ``missMDA::estim_ncpPCA`` on a Python sample table.

    Args:
        data: Numeric NumPy array, pandas DataFrame, or array-like object.
            Rows are samples, columns are continuous features, and missing
            entries are ``NaN``. This is raw data, not a covariance matrix.
        ncp_min: Integer minimum number of components to test.
        ncp_max: Integer maximum number of components to request.
        method: String ``"Regularized"`` or ``"EM"``.
        scale: Boolean passed to R to enable or disable variable scaling.
        method_cv: String ``"gcv"``, ``"loo"``, or ``"Kfold"``.
        nbsim: Integer K-fold simulation count.
        p_na: Float fraction of values masked during K-fold validation.
        threshold: Float convergence threshold.
        seed: Optional integer R random seed.
        rscript: Deprecated string or Path accepted for call-site
            compatibility. The wrapper now uses ``rpy2`` in the current
            Python process and does not start ``Rscript`` subprocesses.
        verbose: Boolean controlling the R progress display.

    Returns:
        Dictionary with integer ``ncp`` and ``criterion``, a mapping from each
        tested component count to its float MSEP value.

    Notes:
        The wrapper intentionally performs no preflight validation. NumPy,
        rpy2, and R errors are allowed to surface directly for debugging.
    """
    if rscript not in (None, RSCRIPT, Path(RSCRIPT)):
        warnings.warn(
            "estimate_ncp_pca no longer uses the rscript argument; rpy2 "
            "selects the embedded R runtime.",
            DeprecationWarning,
            stacklevel=2,
        )

    if RPY2_IMPORT_ERROR is not None:
        raise ImportError(
            "estimate_ncp_pca requires rpy2 so missMDA can run in-process. "
            "Install rpy2 and make sure the R package missMDA is available "
            "in the embedded R library path."
        ) from RPY2_IMPORT_ERROR

    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        array = array[:, None]

    if seed is not None:
        ro.r["set.seed"](int(seed))

    values = ro.FloatVector(array.ravel(order="F"))
    r_matrix = ro.r["matrix"](values, nrow=array.shape[0], ncol=array.shape[1])
    r_data = ro.r["as.data.frame"](r_matrix)

    miss_mda = importr("missMDA")
    result = miss_mda.estim_ncpPCA(
        r_data,
        ncp_min=int(ncp_min),
        ncp_max=int(ncp_max),
        method=method,
        scale=bool(scale),
        method_cv=method_cv,
        nbsim=int(nbsim),
        pNA=float(p_na),
        threshold=float(threshold),
        verbose=bool(verbose),
    )

    selected_ncp = int(result.rx2("ncp")[0])
    criterion = result.rx2("criterion")
    criterion_values = [float(value) for value in criterion]
    criterion_names = criterion.names
    if criterion_names is None:
        component_counts = range(int(ncp_min), int(ncp_min) + len(criterion_values))
    else:
        component_counts = [int(float(name)) for name in criterion_names]

    return {
        "ncp": selected_ncp,
        "criterion": {
            component: value
            for component, value in zip(component_counts, criterion_values)
        },
    }
