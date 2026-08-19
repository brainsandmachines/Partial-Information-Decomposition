# Suppression Simulations: Commonality + CV/Figure Save Changes

Use this note as a direct handoff to replicate the same behavior on another machine.

## What Was Added

1. Run commonality analysis from suppression simulations (after PID runs).
2. Save per-seed commonality results to CSV.
3. Save a summary commonality figure at the end of multi-seed execution.
4. Return raw RVs from the simulation runner so commonality can be computed.
5. Add table/figure utility helpers for commonality rendering.

## Files Changed

- [Simulations/Theoretical_Examples/RVs_Story/story_batch_utils.py](Simulations/Theoretical_Examples/RVs_Story/story_batch_utils.py)
- [Simulations/Theoretical_Examples/RVs_Story/story_pid_utils.py](Simulations/Theoretical_Examples/RVs_Story/story_pid_utils.py)
- [Partial_Information_Decomposition/PID_util.py](Partial_Information_Decomposition/PID_util.py)
- [Simulations/Theoretical_Examples/rv_config.yaml](Simulations/Theoretical_Examples/rv_config.yaml)
- [Simulations/Theoretical_Examples/RVs_Story/suppresion_examples/all_supp_examples.py](Simulations/Theoretical_Examples/RVs_Story/suppresion_examples/all_supp_examples.py)

## 1) story_batch_utils.py

### Import commonality helper

Add import near top:

```python
from encoding_model.commonality import commonality_analysis
```

### In loop_examples(...)

Change runner call to unpack RVs:

```python
results, rvs = main_func(config, func)
```

After PID loop, add optional commonality block:

```python
ca = config.get("commonality_analysis", False)
if ca:
    print("\nRunning commonality analysis...")
    x1, x2, t = rvs
    ca_results = commonality_analysis(
        x1,
        x2,
        t,
        method=config.get("commonality_method", "ridge_cv"),
        alphas=config.get("commonality_alphas", None),
        scale_by_target_variance=config.get("commonality_scale_by_target_variance", False),
    )
    all_results["Commonality_Analysis"] = ca_results
    print("Finished commonality analysis.")
```

### Add commonality CSV path helper

```python
def commonality_csv_path(results_dir: Path, example: str) -> Path:
    """Build the per-example commonality CSV path."""
    example_name = example.replace(" ", "_")
    return results_dir / f"{example_name}_commonality_seeds.csv"
```

### Extend save_seed_csvs(...)

Update signature:

```python
def save_seed_csvs(seed: int, all_results: dict, results_dir: Path, save_commonality_csv: bool = False) -> None:
```

Inside loop, add special handling for Commonality_Analysis:

```python
if example == "Commonality_Analysis":
    if not save_commonality_csv:
        continue
    commonality_path = commonality_csv_path(results_dir, example)
    fieldnames = ["seed", "R²_X1", "R²_X2", "R²_X12", "Unique_X1", "Unique_X2", "Common", "Unexplained"]
    old_rows = []
    if commonality_path.exists():
        with commonality_path.open(newline="", encoding="utf-8") as handle:
            old_rows = [old for old in csv.DictReader(handle) if int(old["seed"]) != seed]
    row = {
        "R²_X1": results.get("R²_X1"),
        "R²_X2": results.get("R²_X2"),
        "R²_X12": results.get("R²_X12"),
        "Unique_X1": results.get("unique_X1"),
        "Unique_X2": results.get("unique_X2"),
        "Common": results.get("common"),
        "Unexplained": results.get("unexplained"),
    }
    with commonality_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(old_rows)
        writer.writerow({"seed": seed, **{key: _as_float(value) for key, value in row.items()}})
    continue
```

### In loop_examples_over_seeds(...)

Import additional saver:

```python
from Partial_Information_Decomposition.PID_util import save_commonality_comparison_table, save_pid_comparison_table
```

Track/save commonality in seed loop:

```python
last_commonality = None
save_commonality = bool(config.get("commonality_analysis", False))
...
last_commonality = all_results.get("Commonality_Analysis")
save_seed_csvs(seed, all_results, results_dir, save_commonality_csv=save_commonality)
```

Save summary commonality figure after PID mean figures:

```python
if save_commonality and last_commonality is not None:
    save_commonality_comparison_table(
        {"Commonality_Analysis": last_commonality},
        f"{results_dir}/commonality_over_{len(seed_values)}_seeds.png",
        title=f"Commonality Results - Seed Range {seed_values[0]}-{seed_values[-1]}",
        config=mean_config,
    )
```

## 2) story_pid_utils.py

In run_pid_story(...), change return value from only PID results to PID results + generated RVs:

```python
return results, [x1, x2, t]
```

This enables downstream commonality analysis without re-generating data.

## 3) PID_util.py

### Layout tweak in existing PID figure saver

In save_pid_comparison_table(...), title/legend vertical placement was adjusted:

```python
fig.suptitle(title, fontsize=14, fontweight="bold", y=0.91)
...
fig.text(0.5, 0.78, legend, ha="center", va="center", fontsize=9, color="#4b5563")
```

### New commonality table helpers

Add functions:

1. commonality_comparison_table(...)
2. save_commonality_comparison_table(...)
3. commanility_analysis(...)  # backward-compatible wrapper name kept intentionally

Purpose:

- Normalize possible key variants in commonality payload.
- Render/save a matplotlib table image for commonality metrics.
- Keep compatibility with legacy caller name typo (commanility_analysis).

## 4) rv_config.yaml

Commonality was enabled/configured in parameters:

```yaml
commonality_analysis: True
```

Also used in this run:

- n_samples changed to 1000000
- p changed to 10
- debias_factor_bool: True
- results_dir redirected to:
  /home/ohadshee/Desktop/Partial-Information-Decomposition/Simulations/Theoretical_Examples/RVs_Story/results_rvs/research_propsal

Optional extra keys supported by the new code path:

```yaml
commonality_method: ridge_cv
commonality_alphas: [0.01, 0.1, 1.0, 10.0]
commonality_scale_by_target_variance: False
```

If these are missing, defaults are used.

## 5) all_supp_examples.py

Current execution scope was narrowed for the run:

- functions_to_run changed from 3 examples to only full_suppresion
- example_names aligned to only full_suppresion
- num_seeds default in launcher changed from 100 to 20

This is not required for commonality support itself; it reflects current experiment scope.

## Outputs You Should See

When commonality is enabled:

1. Per-seed CSV file:
   - Commonality_Analysis_commonality_seeds.csv
2. End-of-run figure:
   - commonality_over_<N>_seeds.png
3. Existing PID per-method CSVs and mean PID figures continue as before.

## Quick Verification Checklist

1. Run suppression examples with commonality_analysis: True.
2. Confirm logs show:
   - Running commonality analysis...
   - Finished commonality analysis.
3. Confirm CSV exists and contains columns:
   - seed, R²_X1, R²_X2, R²_X12, Unique_X1, Unique_X2, Common, Unexplained
4. Confirm commonality summary PNG is created in results_dir.
