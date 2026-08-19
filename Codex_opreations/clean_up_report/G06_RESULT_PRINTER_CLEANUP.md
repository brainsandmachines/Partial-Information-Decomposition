# G06 Result Printer Cleanup

## Summary

G06 is complete. The exact duplicate multivariate result printer in `supression_effect/supp_gauss_multivariate.py` was removed, and the script now imports the canonical implementation from `Partial_Information_Decomposition/PID_util.py`.

Follow-up cleanup also removed the univariate `compare_results` printer after standardizing the commonality result contract from `A`/`B` names to `X1`/`X2` names.

The stale `Partial_Information_Decomposition/Dump.py` file was deleted because it only contained an unused `check_supression_effect` helper and had no active imports or call sites.

## Kept

| Function | Location | What it does | Why keep it |
| --- | --- | --- | --- |
| `compare_results` | `Partial_Information_Decomposition/PID_util.py:576` | Prints side-by-side variance partitioning, PID, and optional MI results for the shared `X1`/`X2` key contract. | This is the canonical comparison printer. It now also supports univariate calls by inferring MI totals from PID components when `mi_results` is omitted. |
| `check_supression_effect` | `supression_effect/gauss_unvariate.py:76` | Prints whether the univariate variance partitioning/PID result looks like a suppression effect. | Kept because it has active local call sites in `gauss_unvariate.py`. |

## Removed

| Removed item | Replacement | Reason |
| --- | --- | --- |
| `supression_effect/supp_gauss_multivariate.py` local `compare_results` | `from Partial_Information_Decomposition.PID_util import compare_results` at `supression_effect/supp_gauss_multivariate.py:16` | The removed function was an exact duplicate of the PID utility. All later calls still use `compare_results(...)`, but now resolve to the canonical import. |
| `supression_effect/gauss_unvariate.py` local `compare_results` | `from Partial_Information_Decomposition.PID_util import compare_results` | Removed after changing commonality outputs and univariate readers from older A/B-style result keys to the shared X1/X2-style result keys. |
| `Partial_Information_Decomposition/Dump.py` | None | The file only contained an unused older `check_supression_effect` helper. Search found no active imports or call sites. |

## Call Site Notes

`supression_effect/supp_gauss_multivariate.py` still calls `compare_results(...)` in the experiment runner. Those calls now use the PID utility import:

```text
supression_effect/supp_gauss_multivariate.py:251
supression_effect/supp_gauss_multivariate.py:258
supression_effect/supp_gauss_multivariate.py:264
supression_effect/supp_gauss_multivariate.py:269
supression_effect/supp_gauss_multivariate.py:275
supression_effect/supp_gauss_multivariate.py:281
supression_effect/supp_gauss_multivariate.py:287
```

No call sites import `Partial_Information_Decomposition.Dump`.

`supression_effect/gauss_unvariate.py` now calls the canonical `compare_results(...)` and then calls `check_supression_effect(...)` separately, preserving its previous suppression-effect output.

The shared commonality result keys are now:

```text
encoding_model/commonality.py:85
R²_X1
R²_X2
R²_X12
unique_X1
unique_X2
common
unexplained
```

## Verification

Passed final syntax parse:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from pathlib import Path
paths = [
    'supression_effect/supp_gauss_multivariate.py',
    'supression_effect/gauss_unvariate.py',
    'encoding_model/commonality.py',
    'Partial_Information_Decomposition/PID_util.py',
    'toy_examples/suppression_toy_runner.py',
    'toy_examples/suppression_pipeline_example.py',
    'toy_examples/grid_search_example.py',
    'utils.py',
    'uni_tests/test_supression_core.py',
]
for path in paths:
    compile(Path(path).read_text(), path, 'exec')
PY
```

Passed smoke check:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib-codex python3 - <<'PY'
from Partial_Information_Decomposition.PID_util import compare_results
PY
```

Passed key-contract smoke check with sklearn avoided by monkeypatching the local
score function:

```text
['R²_X1', 'R²_X12', 'R²_X2', 'common', 'unexplained', 'unique_X1', 'unique_X2']
```

Search check after cleanup:

```text
Partial_Information_Decomposition/PID_util.py:576:def compare_results(...)
supression_effect/gauss_unvariate.py:76:def check_supression_effect(...)
```

Focused pytest was attempted:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q uni_tests/test_supression_core.py
```

It did not reach the tests because collection imports `sklearn`, which currently
fails in this environment with the existing NumPy/sklearn binary mismatch:
`ImportError: numpy.core.multiarray failed to import`.

`git diff --check` passed.

## Recommendation

Use `X1`/`X2` naming for new variance-partitioning/commonality outputs. Avoid adding new `A`/`B` result keys unless they are local temporary variables, not public result dictionary keys.
