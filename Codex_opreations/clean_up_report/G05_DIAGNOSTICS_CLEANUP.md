# G05 Diagnostics Helper Cleanup

## Summary

G05 is complete. The diagnostics helpers now have one canonical implementation in `Partial_Information_Decomposition/PID_util.py`, while `encoding_model/encoding_utils.py` re-exports those helpers for old imports.

This fixes the audit note about waiting "until callers are adapted to the PID return contract or the contracts are unified": the contract is now unified.

## Kept

| Function | Location | What it does | Why keep it |
| --- | --- | --- | --- |
| `correlation_matrix` | `Partial_Information_Decomposition/PID_util.py:409` | Converts a data block into a feature-feature correlation matrix. It now accepts 1D arrays and one-column blocks. | This is the canonical diagnostics helper used by PID-side singularity checks. |
| `singularity_report` | `Partial_Information_Decomposition/PID_util.py:445` | Builds singularity diagnostics for `M1`, `M2`, `Y`, and their joined blocks. | This is now the single return contract for diagnostics. By default it returns only `report`; `return_printing_required=True` returns `(report, printing_required)` for compatibility. |
| `diagnostic_plots` | `Partial_Information_Decomposition/PID_util.py:475` | Plots diagnostic correlation/cross-correlation views for model blocks and target. | The plotting helper belongs with the PID diagnostics utilities because it inspects the same block structure. |

## Removed Duplicate Implementations

| Removed duplicate | Old location | Replacement | Reason |
| --- | --- | --- | --- |
| `correlation_matrix` | `encoding_model/encoding_utils.py` | `Partial_Information_Decomposition.PID_util.correlation_matrix` | Same-purpose diagnostics helper. Keeping two versions made the singularity behavior drift. |
| `singularity_report` | `encoding_model/encoding_utils.py` | `Partial_Information_Decomposition.PID_util.singularity_report` | Same-purpose diagnostics helper with different return shape. The PID version is now adapted to the encoding-side default behavior. |
| `diagnostic_plots` | `encoding_model/encoding_utils.py` | `Partial_Information_Decomposition.PID_util.diagnostic_plots` | Same-purpose plotting helper. One canonical implementation is easier to maintain. |

Nothing was erased from callers without replacement. `encoding_model/encoding_utils.py:16` now imports and re-exports the canonical helpers, so older code using `from encoding_model.encoding_utils import singularity_report` still works.

## Contract Change

`singularity_report(X_M1, X_M2, y_real)` now returns:

```python
report
```

Compatibility mode remains available:

```python
report, printing_required = singularity_report(
    X_M1,
    X_M2,
    y_real,
    return_printing_required=True,
)
```

The function also now checks singularity on `correlation_matrix(block)` instead of checking raw rectangular data blocks. This removes the old failure path where rectangular feature matrices could raise `"Input isn't a square matrix"` before diagnostics were produced.

## Import Updates

| File | Change |
| --- | --- |
| `encoding_model/encoding_utils.py:16` | Re-exports `correlation_matrix`, `singularity_report`, and `diagnostic_plots` from `Partial_Information_Decomposition.PID_util`. |
| `toy_examples/toy_example.py:8` | Imports `diagnostic_plots` directly from `Partial_Information_Decomposition.PID_util`. |
| `encoding_model/suppression_core.py:1` | No longer imports diagnostics from `encoding_model.encoding_utils`. |
| `encoding_model/suppression_core.py:41` | Moved `LinearRegression` import into `create_encoder`, so importing toy examples does not eagerly import sklearn. |
| `encoding_model/suppression_core.py:147`, `encoding_model/suppression_core.py:200`, `encoding_model/suppression_core.py:343` | Moved `pandas` imports into the functions that actually build/read result tables. |

## Additional Import-Time Fix

`Partial_Information_Decomposition/PID_util.py` had heavy optional imports at module import time. Those are now lazy imports inside the functions that need them:

| Dependency | Current local import locations |
| --- | --- |
| `sklearn.linear_model.LinearRegression` | `Partial_Information_Decomposition/PID_util.py:12` |
| `sklearn.linear_model.RidgeCV` | `Partial_Information_Decomposition/PID_util.py:46` |
| `sklearn.covariance.LedoitWolf` | `Partial_Information_Decomposition/PID_util.py:79` |
| `sklearn.discriminant_analysis.StandardScaler` | `Partial_Information_Decomposition/PID_util.py:511` |
| `pandas` | `Partial_Information_Decomposition/PID_util.py:638`, `Partial_Information_Decomposition/PID_util.py:748`, `Partial_Information_Decomposition/PID_util.py:886` |

This was needed because importing diagnostics should not force pandas/sklearn binary imports.

`encoding_model/encoding_utils.py` also no longer imports `nilearn`, `matplotlib`, `PIL`, `torchvision`, `torch`, `joblib`, or `utils.py` at module import time. Those imports now live inside the functions/classes that need them, so the compatibility re-export can be imported even when optional visualization dependencies are not installed.

## Verification

Passed final syntax parse:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from pathlib import Path
paths = [
    'Partial_Information_Decomposition/PID_util.py',
    'encoding_model/encoding_utils.py',
    'encoding_model/suppression_core.py',
    'toy_examples/toy_example.py',
]
for path in paths:
    compile(Path(path).read_text(), path, 'exec')
PY
```

Passed smoke checks:

| Check | Result |
| --- | --- |
| `singularity_report(X1, X2, Y)` | Returned `dict`. |
| `singularity_report(X1, X2, Y, return_printing_required=True)` | Returned `(dict, bool)`. |
| Rectangular feature matrices with one-column `Y` | No old non-square covariance error. |
| `from toy_examples import toy_example` | Import passed. It emitted a fontconfig warning from matplotlib cache permissions, but not a nilearn/pandas/sklearn diagnostics failure. |
| `from encoding_model.encoding_utils import singularity_report, correlation_matrix, diagnostic_plots` | Import passed without requiring `nilearn`. |

Search check passed. The only real definitions are:

```text
Partial_Information_Decomposition/PID_util.py:409:def correlation_matrix(X):
Partial_Information_Decomposition/PID_util.py:445:def singularity_report(...)
Partial_Information_Decomposition/PID_util.py:475:def diagnostic_plots(...)
```

`git diff --check` passed after cleanup.

## Recommendation

Keep using `Partial_Information_Decomposition.PID_util` as the canonical diagnostics module. Keep the `encoding_utils.py` re-export for now because it is a low-risk compatibility layer, but future new code should import diagnostics directly from `PID_util.py`.
