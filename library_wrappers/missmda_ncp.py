"""Minimal Python wrapper for R's ``missMDA::estim_ncpPCA``."""

from pathlib import Path
import subprocess
import tempfile

import numpy as np


RSCRIPT = "Rscript"

R_CODE = r"""
a <- commandArgs(TRUE)
if (nzchar(a[11])) set.seed(as.integer(a[11]))
result <- missMDA::estim_ncpPCA(
  read.csv(a[1], header=FALSE),
  ncp.min=as.integer(a[3]),
  ncp.max=as.integer(a[4]),
  method=a[5],
  scale=as.logical(a[6]),
  method.cv=a[7],
  nbsim=as.integer(a[8]),
  pNA=as.numeric(a[9]),
  threshold=as.numeric(a[10]),
  verbose=as.logical(a[12])
)
write.csv(data.frame(
  selected_ncp=as.integer(result$ncp),
  component=as.integer(names(result$criterion)),
  msep=as.numeric(result$criterion)
), a[2], row.names=FALSE)
"""


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
        rscript: String or Path to the Rscript executable.
        verbose: Boolean controlling the R progress display.

    Returns:
        Dictionary with integer ``ncp`` and ``criterion``, a mapping from each
        tested component count to its float MSEP value.

    Notes:
        The wrapper intentionally performs no validation. NumPy, subprocess,
        and R errors are allowed to surface directly for debugging.
    """
    with tempfile.TemporaryDirectory() as directory:
        input_csv = Path(directory) / "input.csv"
        output_csv = Path(directory) / "output.csv"
        np.savetxt(input_csv, np.asarray(data, dtype=float), delimiter=",")
        subprocess.run(
            [
                str(rscript),
                "-e",
                R_CODE,
                str(input_csv),
                str(output_csv),
                str(ncp_min),
                str(ncp_max),
                method,
                str(scale).upper(),
                method_cv,
                str(nbsim),
                str(p_na),
                str(threshold),
                "" if seed is None else str(seed),
                str(verbose).upper(),
            ],
            check=True,
        )
        rows = np.loadtxt(output_csv, delimiter=",", skiprows=1, ndmin=2)
    return {
        "ncp": int(rows[0, 0]),
        "criterion": {int(row[1]): float(row[2]) for row in rows},
    }
