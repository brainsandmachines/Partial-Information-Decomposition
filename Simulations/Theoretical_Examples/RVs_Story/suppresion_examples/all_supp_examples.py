"""Run all suppression RVs_Story examples across seeds."""

from functools import partial
from pathlib import Path
import sys

from full_suppresion import full_suppresion
from unq12_zero import unq12_zero
from unq2_zero import unq2_zero

STORY_ROOT = Path(__file__).resolve().parents[1]
if str(STORY_ROOT) not in sys.path:
    sys.path.append(str(STORY_ROOT))

from story_batch_utils import loop_examples_over_seeds
from story_pid_utils import load_story_config, run_pid_story, truth_pid_suppression


if __name__ == "__main__":
    config = load_story_config()
    config_dict = config["parameters"]
    functions_to_run = [full_suppresion]
    example_names = ["full_suppresion"]
    story_runner = partial(run_pid_story, truth_func=truth_pid_suppression)
    loop_examples_over_seeds(
        config_dict,
        functions_to_run,
        example_names,
        story_runner,
        num_seeds=config_dict.get("num_seeds", 20),
    )
    print("Finished all examples.")
