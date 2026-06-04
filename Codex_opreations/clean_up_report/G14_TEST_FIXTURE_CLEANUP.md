# G14 Test Fixture Cleanup

## Summary

G14 is complete. The duplicated `random_data` test fixtures now live in:

```text
uni_tests/conftest.py
```

## Kept

| Fixture | Location | What it does | Why keep it |
| --- | --- | --- | --- |
| `dims` | `uni_tests/conftest.py:6` | Provides the multivariate test dimensions. | Shared fixture home for the multivariate tests. |
| `random_data` | `uni_tests/conftest.py:11` | Builds random `(T, M1, M2)` tensors for univariate and multivariate tests. | One fixture avoids three copied definitions. |

## Removed Local Copies

| File | Removed fixture(s) |
| --- | --- |
| `uni_tests/test_Idep.py` | `random_data` |
| `uni_tests/test_PID_util.py` | `random_data` |
| `uni_tests/test_idep_multivariate.py` | `dims`, `random_data` |

The shared fixture preserves the old behavior: univariate tests receive 1D tensors, while `test_idep_multivariate.py` receives the larger multivariate dimensions.

## Verification

Search check after cleanup shows only `uni_tests/conftest.py` defines `dims` and `random_data`.

Syntax parsing passed for the touched test files.

Focused pytest collection was attempted with:

```text
pytest -q --collect-only uni_tests/test_Idep.py uni_tests/test_PID_util.py uni_tests/test_idep_multivariate.py
```

Collection is blocked by the current environment because `sklearn` imports `pyarrow`, and `pyarrow` fails against NumPy 2.2.4 with `numpy.core.multiarray failed to import`.

`git diff --check` passed.
