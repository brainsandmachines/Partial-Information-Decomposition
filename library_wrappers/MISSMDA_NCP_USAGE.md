# Minimal `estim_ncpPCA` wrapper

The wrapper has one function and calls the original R
`missMDA::estim_ncpPCA` implementation through `Rscript`.

Install the compatible R packages with:

```bash
/home/ohadshee/anaconda3/bin/conda install \
  -n PID_env -c conda-forge \
  r-missmda=1.21 r-factominer=2.13
```

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
not a covariance matrix. The wrapper intentionally has no shape checks,
assertions, argparse interface, or custom error handling.

`missMDA` 1.21 cannot run GCV on a completely observed table. For complete
data, call:

```python
result = estimate_ncp_pca(
    data,
    method_cv="Kfold",
    seed=56,
)
```
