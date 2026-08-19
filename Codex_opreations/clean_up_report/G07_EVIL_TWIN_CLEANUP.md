# G07 Evil-Twin Cleanup

## Summary

G07 is complete. The old `toy_examples/evil_twin.py` script was deleted, but the Sonic/Shadow covariance example was kept in a dedicated root-level importable package:

```text
evil_twin/
```

Use this import path for the kept example:

```python
from evil_twin import evil_twin_example_torch
```

## Kept

| Function | Location | What it does | Why keep it |
| --- | --- | --- | --- |
| `evil_twin_example_torch` | `evil_twin/covariance_example.py:70` | Generates Sonic and Shadow tensors, each as `(X1, X2, T)`. | This is the example you wanted to keep for showing the Sonic/Shadow covariance comparison. |
| `empirical_covariance_matrix_torch` | `evil_twin/covariance_example.py:6` | Computes the empirical covariance of `(X1, X2, T)`. | Needed to compare Sonic and Shadow covariance matrices. |
| `check_evil_twin_covariances_torch` | `evil_twin/covariance_example.py:24` | Computes Sonic/Shadow covariance matrices and reports their difference. | This is the direct covariance-comparison helper for the example. |
| `run_covariance_comparison` | `evil_twin/covariance_example.py:128` | Generates Sonic/Shadow data and compares their covariances in one call. | Useful as a small runnable demonstration without reintroducing top-level script execution. |

## Removed

| Removed item | Replacement | Reason |
| --- | --- | --- |
| `toy_examples/evil_twin.py` | `evil_twin/covariance_example.py` | The old file mixed reusable helpers with top-level execution and extra PID workflow code. The reusable covariance example now lives in its own root-level folder. |
| `evil_twin_idep` from the old script | None in G07 | Not needed for the requested Sonic/Shadow covariance demonstration. |
| `evil_twin_tilde_pid` from the old script | None in G07 | Not needed for the requested Sonic/Shadow covariance demonstration. |

## Import Notes

The package re-exports the kept helpers from `evil_twin/__init__.py`, so callers can use:

```python
from evil_twin import (
    check_evil_twin_covariances_torch,
    empirical_covariance_matrix_torch,
    evil_twin_example_torch,
    run_covariance_comparison,
)
```

Search found no active external imports of the old `toy_examples/evil_twin.py` file, so no caller rewrites were needed.

## Verification

Passed syntax parse:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from pathlib import Path
paths = [
    'evil_twin/__init__.py',
    'evil_twin/covariance_example.py',
]
for path in paths:
    compile(Path(path).read_text(), path, 'exec')
PY
```

Passed import and covariance smoke check:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from evil_twin import evil_twin_example_torch, check_evil_twin_covariances_torch
import torch

g = torch.Generator().manual_seed(0)
data = evil_twin_example_torch(g, n=40, p=3)
result = check_evil_twin_covariances_torch(data, verbose=False)
print(sorted(data.keys()))
print(result['Sigma_sonic'].shape)
PY
```

Observed:

```text
['shadow', 'sonic']
torch.Size([9, 9])
```

## Recommendation

Keep future Sonic/Shadow example code inside `evil_twin/`. If PID-specific evil-twin workflows are needed again, add them as separate modules in this folder rather than restoring top-level execution in a single script.
