# Minimal `estim_ncpPCA` wrapper

The wrapper has one function and calls the original R
`missMDA::estim_ncpPCA` implementation in-process through `rpy2`.

Install the compatible R packages with:

```bash
conda activate your-environment
conda install -c conda-forge r-missmda=1.21 r-factominer=2.13 rpy2
```

The wrapper does not start an `Rscript` subprocess. `rpy2` embeds R in the
current Python process, so R errors raised by `missMDA::estim_ncpPCA` surface
directly as Python exceptions and can stop an `sbatch` job. The deprecated
`rscript` keyword is still accepted for old call sites, but it is ignored.

Use it from Python:

```python
import numpy as np

from library_wrappers.missmda_ncp import estimate_ncp_pca

data = np.array(
    [
        [1.0, 1.2, 0.8],
        [2.0, 2.1, 1.7],
        [3.0, np.nan, 2.8],
        [4.0, 4.2, 4.1],
        [5.0, 5.1, 4.9],
    ]
)

result = estimate_ncp_pca(data, ncp_max=2)
print(result["ncp"])
print(result["criterion"])
```

Rows are samples and columns are continuous features. The input is raw data,
not a covariance matrix. The wrapper intentionally has no argparse interface
or custom error handling.

`missMDA` 1.21 cannot run GCV on a completely observed table. For complete
data, call:

```python
result = estimate_ncp_pca(
    data,
    method_cv="Kfold",
    seed=56,
)
```
