# G13 Config And Filename Utility Cleanup

## Summary

G13 is complete. The duplicated config merge helper and filename helper now have one canonical implementation each.

## Kept

| Function | Location | What it does | Why keep it |
| --- | --- | --- | --- |
| `make_pre_config` | `Partial_Information_Decomposition/Idep_Simulations/Simulation_utils.py:905` | Merges simulation config fragments for the Idep simulation runs. | Belongs with the simulation utility code and already existed there. |
| `safe_filename` | `Partial_Information_Decomposition/output_utils.py:4` | Converts output labels to strings before using them in file names. | Lightweight shared helper with no heavy imports. |

## Removed Local Copies

| File | Removed local helper | Replacement |
| --- | --- | --- |
| `Partial_Information_Decomposition/Idep_Simulations/idep_sim_runs.py` | `make_pre_config` | Imports `make_pre_config` from `Simulation_utils.py`. |
| `Partial_Information_Decomposition/Idep_Simulations/Simulation_utils.py` | `safe_filename` | Imports `safe_filename` from `output_utils.py`. |
| `Partial_Information_Decomposition/heatmap_plot.py` | `safe_filename` | Imports `safe_filename` from `output_utils.py`. |

## Verification

Search check after cleanup shows:

```text
Partial_Information_Decomposition/output_utils.py:4 def safe_filename
Partial_Information_Decomposition/Idep_Simulations/Simulation_utils.py:905 def make_pre_config
```

The old call sites still call `safe_filename` and `make_pre_config`, but now through imports.

Syntax parsing passed for the touched config/plotting files.

`git diff --check` passed.
