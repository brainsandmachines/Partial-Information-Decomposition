import torch
import yaml
import sys
import csv
from pathlib import Path
import numpy as np
from unq2_zero import unq2_zero
from full_suppresion import full_suppresion
from unq12_zero import unq12_zero
from core_model import main_func



root = Path(__file__).resolve().parents[4]
sys.path.append(str(root))  
from Partial_Information_Decomposition.PID_util import pid_comparison_table, save_pid_comparison_table



def _as_float(value):
    return float(value.detach().cpu().numpy()) if isinstance(value, torch.Tensor) else float(value)


def loop_examples(config:dict,function_to_run:list,example_name:list,save_image:bool=True):
    """Loop over all suppresion examples and run them"""
    all_results = {}
    for func, name in zip(function_to_run,example_name):
        print(f"\nRunning example {func.__name__}...")
        results_dict = main_func(config, func)
        all_results[name] = results_dict
        if save_image:
            save_pid_comparison_table(results_dict,f"{config['results_dir']}/{name}.png",config=config)
        print(f"Finished example {func.__name__}.")
    return all_results



def save_seed_csvs(seed:int, all_results:dict, results_dir:Path, overwrite:bool=False):
    """Save current seed values into one CSV per example and PID method."""
    opened = set()
    for example, results in all_results.items():
        for row in pid_comparison_table(results, print_table=False):
            method = row.pop("method")
            example_name = example.replace(" ", "_")
            method_name = method.replace(" ", "_")
            path = results_dir / f"{example_name}_{method_name}_seeds.csv"
            mode = "w" if (overwrite and path not in opened) or not path.exists() else "a"
            opened.add(path)
            with path.open(mode, newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["seed", *row.keys()])
                if mode == "w":
                    writer.writeheader()
                writer.writerow({"seed": seed, **{k: _as_float(v) for k, v in row.items()}})


def mean_over_seeds(seed_results:list[dict]):
    """Average the output of main_func across seeds."""
    mean_results = {}
    for method in seed_results[0]:
        pid_keys = seed_results[0][method][0]
        mi_keys = seed_results[0][method][1]
        pid = {k: np.mean([_as_float(r[method][0][k]) for r in seed_results]) for k in pid_keys}
        mi = {k: np.mean([_as_float(r[method][1][k]) for r in seed_results]) for k in mi_keys}
        mean_results[method] = (pid, mi)
    return mean_results


def loop_examples_over_seeds(config:dict,function_to_run:list,example_name:list,num_seeds:int=None,seeds:list=None):
    """Run all examples over seeds, save seed CSVs, then save averaged tables."""
    base_seed = config.get("seed", 0)
    seeds = list(seeds) if seeds is not None else list(range(base_seed, base_seed + (num_seeds or 1)))
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    by_example = {name: [] for name in example_name}

    for i, seed in enumerate(seeds):
        print(f"\n{'='*70}\nRunning all examples with seed={seed}\n{'='*70}")
        seed_config = {**config, "seed": seed}
        all_results = loop_examples(seed_config,function_to_run,example_name,save_image=False)
        save_seed_csvs(seed, all_results, results_dir, overwrite=(i == 0))
        for name, results in all_results.items():
            by_example[name].append(results)

    mean_config = {**config, "seed": f"{seeds[0]}-{seeds[-1]}"}
    for name, seed_results in by_example.items():
        mean_results = mean_over_seeds(seed_results)
        save_pid_comparison_table(
            mean_results,
            f"{results_dir}/{name}_mean_over_{len(seeds)}_seeds.png",
            title=f"PID Method Comparison - Mean Over {len(seeds)} Seeds",
            config=mean_config,
        )
    return by_example


if __name__ == "__main__":
    
    config_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Simulations/Theoretical_Examples/rv_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config_dict = config['parameters']
    function_to_run = [full_suppresion, unq2_zero, unq12_zero]
    example_name = ["full_suppresion", "unq2_zero", "unq12_zero"]
    _ = loop_examples_over_seeds(config_dict,function_to_run,example_name,num_seeds=config_dict.get("num_seeds", 2))
    print("Finished all examples.")
