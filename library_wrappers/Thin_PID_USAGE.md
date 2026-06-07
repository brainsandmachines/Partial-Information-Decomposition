# Thin-PID Wrapper Usage

This wrapper calls `exact_gauss_thin_pid` from `external/flow_pid/pid/thin_pid.py` without modifying the upstream `flow-pid` code.

## Input

- `--matrix-csv`: CSV containing a square covariance or correlation matrix.
- `--sizes`: comma-separated dimensions in local wrapper order: `source1,source2,target`.
- The CSV matrix must be ordered as `[source1, source2, target]`.
- The wrapper reorders the matrix to `[target, source1, source2]` before calling `exact_gauss_thin_pid(cov, dm, dx, dy)`.
- The matrix must be symmetric and have shape `(source1 + source2 + target, source1 + source2 + target)`.
- The wrapper does not whiten, normalize, standardize, discretize, or repair the matrix. The flow-pid implementation whitens internally and assumes the required covariance blocks are invertible.

## Output

The output is a one-row CSV with:

- `unique_source1`
- `unique_source2`
- `redundancy`
- `synergy`
- `I_source1_target`
- `I_source2_target`
- `joint_mutual_information`
- `union_information`
- `optimization_objective`
- `interaction_information`

## Commands

Run the default 1,1,1 evil-twin check:

```bash
python library_wrappers/Thin_PID.py
```

Run a custom covariance/correlation CSV:

```bash
python library_wrappers/Thin_PID.py --matrix-csv path/to/cov.csv --sizes 2,2,1 --output thin_result.csv --case MyCase
```

Request the optional unbiased correction:

```bash
python library_wrappers/Thin_PID.py --matrix-csv path/to/cov.csv --sizes 2,2,1 --sample-size 1000
```

## Dependencies

For this Thin-PID wrapper, the current environment needs only the relevant Gaussian solver dependencies: NumPy, SciPy, Matplotlib, and CVXPY as imported by the flow-pid utility file. The full `flow-pid` repository also lists packages such as `normflows` and PyTorch, but this wrapper avoids importing those modules because they are not needed for `pid/thin_pid.py`.
