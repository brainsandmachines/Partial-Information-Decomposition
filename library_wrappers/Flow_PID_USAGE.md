# Flow-PID Wrapper Usage

This wrapper calls `flow_pid` from `external/flow-pid/pid/flow_pid.py` without modifying the upstream `flow-pid` code.

## Input

- Flow-PID expects raw sample arrays, not covariance matrices.
- Rows are samples and columns are features.
- `--samples-csv` expects one no-header CSV ordered as `[source1, source2, target]`, matching the local wrapper convention.
- `--sizes` gives dimensions as `source1,source2,target`.
- The wrapper splits the combined CSV and calls `flow_pid(target, source1, source2, ...)`.
- Alternatively, pass separate files with `--m-csv`, `--x-csv`, and `--y-csv`; in that mode `m` is target, `x` is source1, and `y` is source2.

## Output

The output is a one-row CSV with:

- `unique_source1`
- `unique_source2`
- `redundancy`
- `synergy`
- `I_source1_target`
- `I_source2_target`
- `joint_mutual_information`
- `interaction_information`

## Commands

Run the built-in simple Gaussian example:

```bash
python library_wrappers/Flow_PID.py --example simple-gaussian --n-flows 1 --n-epochs 1 --batch-size 8 --output flow_simple.csv
```

Run a combined sample CSV:

```bash
python library_wrappers/Flow_PID.py --samples-csv path/to/samples.csv --sizes 2,2,1 --output flow_result.csv
```

Run separate sample CSVs:

```bash
python library_wrappers/Flow_PID.py --m-csv target.csv --x-csv source1.csv --y-csv source2.csv --sizes 2,2,1 --output flow_result.csv
```

## Dependencies

This wrapper requires the Flow-PID training dependencies, including PyTorch and `normflows`. The `normflows` package is listed in `external/flow-pid/requirements.txt`.
