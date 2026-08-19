# G15 Multi-Seed Boilerplate Cleanup

## Summary

G15 is complete for the files listed in the duplicate audit. The repeated multi-seed `main()` boilerplate now lives in:

```text
utils.py:592 run_configured_multiseed
```

## Kept

| Function | Location | What it does | Why keep it |
| --- | --- | --- | --- |
| `run_configured_multiseed` | `utils.py:592` | Runs the multi-seed loop, prints the summary, saves the summary CSV, and prints output paths. | Canonical orchestration helper. |
| Experiment `get_run_config` functions | `supression_effect/Suppresed_Encoder.py`, `Simulations/Both_Unique_Encodrs.py`, `Simulations/both_unique.py`, `Simulations/turned_off_unqiue.py` | Define experiment-specific settings. | Kept because they encode different experiments. |
| Experiment `run_single_seed` functions | Same files | Run each experiment's seed-specific logic. | Kept because their generator/data logic differs. |

## Replaced Local Boilerplate

| File | Change |
| --- | --- |
| `supression_effect/Suppresed_Encoder.py:199` | `main()` now loads the model/fMRI data and delegates standard reporting to `run_configured_multiseed`. |
| `Simulations/Both_Unique_Encodrs.py:105` | `main()` now delegates summary/save/print boilerplate to `run_configured_multiseed`. |
| `Simulations/both_unique.py:164` | `main()` now delegates summary/save/print boilerplate to `run_configured_multiseed`. |
| `Simulations/turned_off_unqiue.py:290` | `main()` now delegates summary/save/print boilerplate to `run_configured_multiseed` and passes only the flat metric output from `run_single_seed`. |

## Not Merged Yet

The experiment-specific `run_single_seed` bodies were intentionally not merged. They still contain different data loading, generator, and analysis logic.

G12 M7/M8 simulation orchestration was not touched in this cleanup.

## Verification

Syntax parsing passed for `utils.py` and the four touched experiment files.

Search check confirms the listed `main()` functions call `run_configured_multiseed`.

`git diff --check` passed.
