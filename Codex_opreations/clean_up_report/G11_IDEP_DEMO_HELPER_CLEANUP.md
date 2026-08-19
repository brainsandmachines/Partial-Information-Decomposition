# G11 Idep Demo Helper Cleanup

## Summary

G11 is complete. The copied Idep equicorrelation demo helpers were removed from the solver implementation files and moved into one example module:

```text
Partial_Information_Decomposition/examples/idep_equicorr_demo.py
```

## Kept

| Function/Class | Location | What it does | Why keep it |
| --- | --- | --- | --- |
| `Idep_multivariate_gauss` | `Partial_Information_Decomposition/Idep_multivariate_gauss.py:30` | Main multivariate Gaussian Idep solver. | Active production solver used by simulations and tests. |
| `JackknifeIdepMultivariateGauss` | `Partial_Information_Decomposition/jacknife_Idep_multivariate_gauss.py:23` | Jackknife-oriented Idep solver variant. | Kept because it has different bias/jackknife behavior from the main solver. |
| `Idep_multivariate_gauss` compatibility alias | `Partial_Information_Decomposition/jacknife_Idep_multivariate_gauss.py:291` | Backwards-compatible alias to the renamed jackknife class. | Softens old imports while making the real class name explicit. |
| `para_Idep_multivariate_gauss` | `Partial_Information_Decomposition/parallel_Idep_multivariate_gauss.py:23` | Batched/parallel Idep solver. | Kept because its tensor shape contract differs from the main solver. |
| `ones` | `Partial_Information_Decomposition/examples/idep_equicorr_demo.py:27` | Builds column vectors of ones for equicorrelation blocks. | Canonical demo helper. |
| `equicorr_blocks` | `Partial_Information_Decomposition/examples/idep_equicorr_demo.py:31` | Builds P/Q/R equicorrelation blocks. | Canonical demo helper. |
| `build_full_cov` | `Partial_Information_Decomposition/examples/idep_equicorr_demo.py:48` | Builds the full covariance matrix from P/Q/R blocks. | Canonical demo helper. |
| `run_serial_example` | `Partial_Information_Decomposition/examples/idep_equicorr_demo.py:80` | Runs the main solver on one equicorrelation example. | Single example entry point for the main solver. |
| `run_jackknife_example` | `Partial_Information_Decomposition/examples/idep_equicorr_demo.py:95` | Runs the jackknife solver on one equicorrelation example. | Single example entry point for the jackknife solver. |
| `run_parallel_example` | `Partial_Information_Decomposition/examples/idep_equicorr_demo.py:114` | Runs the parallel solver on one equicorrelation example. | Single example entry point for the parallel solver. |
| `main` | `Partial_Information_Decomposition/examples/idep_equicorr_demo.py:131` | Prints the paper-style equicorrelation examples. | Keeps demo behavior outside solver files. |

## Removed Local Copies

| File | Removed duplicated helpers |
| --- | --- |
| `Partial_Information_Decomposition/Idep_multivariate_gauss.py` | `ones`, `equicorr_blocks`, `build_full_cov`, `pretty`, `run_one`, `main` |
| `Partial_Information_Decomposition/jacknife_Idep_multivariate_gauss.py` | `ones`, `equicorr_blocks`, `build_full_cov`, `pretty`, `run_one`, `main` |
| `Partial_Information_Decomposition/parallel_Idep_multivariate_gauss.py` | `ones`, `equicorr_blocks`, `build_full_cov`, `pretty`, `run_one`, `main` |

`pretty` was not kept because it was only local formatting for the copied demo code.

## Verification

Search check after cleanup shows the demo helpers only in `Partial_Information_Decomposition/examples/idep_equicorr_demo.py`.

Syntax parsing passed for the three solver files and the new example module.

`git diff --check` passed.
