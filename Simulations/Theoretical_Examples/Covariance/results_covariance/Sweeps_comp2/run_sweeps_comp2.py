"""Run the second theoretical PID timing sweep in its own tmux session."""

from functools import partial
from pathlib import Path

from Simulations.Theoretical_Examples.Covariance import run_gaussian_pid_examples as sweep

OUTPUT_DIR = Path(__file__).resolve().parent

print("Starting Sweeps_comp2.", flush=True)
sweep.EXAMPLES = tuple(
    {**example, "target_dim": 50, "source2_dim": 55}
    for example in sweep.EXAMPLES
)
one_dimensional = {
    **sweep.one_dimensional_target_example(),
    "target_dim": 1,
    "source2_dim": 55,
}
sweep.one_dimensional_target_example = partial(dict, one_dimensional)
sweep.main(
    OUTPUT_DIR / "theoretical_timing_runs.csv",
    OUTPUT_DIR / "theoretical_timing_summary.csv",
    OUTPUT_DIR / "theoretical_pid_and_runtime_sweeps.png",
    OUTPUT_DIR / "hyperparameters.yaml",
)
