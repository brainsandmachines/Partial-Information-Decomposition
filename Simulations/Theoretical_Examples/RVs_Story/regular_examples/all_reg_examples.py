"""Run all regular RVs_Story examples across seeds."""

from functools import partial
from pathlib import Path
import sys

from equal_unique import equal_unique, equal_unique2

STORY_ROOT = Path(__file__).resolve().parents[1]
if str(STORY_ROOT) not in sys.path:
    sys.path.append(str(STORY_ROOT))

from story_batch_utils import loop_examples_over_seeds
from story_pid_utils import load_story_config, run_pid_story, truth_pid_equal_unique


if __name__ == "__main__":
    config = load_story_config()
    config_dict = config["parameters"]
    functions_to_run = [equal_unique, equal_unique2]
    example_names = ["equal_unique", "equal_unique2"]
    story_runner = partial(run_pid_story, truth_func=truth_pid_equal_unique)
    loop_examples_over_seeds(
        config_dict,
        functions_to_run,
        example_names,
        story_runner,
        num_seeds=config_dict.get("num_seeds", 100),
    )
    print("Finished all examples.")
