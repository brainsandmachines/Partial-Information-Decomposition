import torch
import yaml
import sys
from pathlib import Path
from unq2_zero import unq2_zero
from full_suppresion import full_suppresion
from unq12_zero import unq12_zero
from core_model import main_func



root = Path(__file__).resolve().parents[3]
sys.path.append(str(root))  
from Partial_Information_Decomposition.PID_util import save_pid_comparison_table



def loop_exmaples(config:dict,function_to_run:list,example_name:str):
    """Loop over all suppresion examples and run them"""
    for func, name in zip(function_to_run,example_name):
        print(f"\nRunning example {func.__name__}...")
        results_dict = main_func(config, func)
        #Save Table
        save_pid_comparison_table(results_dict,f"{config['results_dir']}/{name}.png",config=config)
        print(f"Finished example {func.__name__}.")
    return results_dict


if __name__ == "__main__":
    
    config_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Simulations/Theoretical_Examples/rv_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config_dict = config['parameters']
    function_to_run = [full_suppresion, unq2_zero, unq12_zero]
    example_name = ["full_suppresion", "unq2_zero", "unq12_zero"]
    _ = loop_exmaples(config_dict,function_to_run,example_name)
    print("Finished all examples.")

