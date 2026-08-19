import torch
from pathlib import Path
import sys
from typing import Any
import yaml
root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from full_OTC.otc_experiment import run_otc_experiment


"""This file will do permuation analysis for features across images,
This is another test to see if everything works correctly and we get the expected results. 
for instance zero mutual inforamtion for both sources or 
If only one source gets permuted, we should only unique inofrmation for the other source."""





def permuatation_analysis(pipeline_config: dict, permuation_config: dict) -> dict[str, Any]:
    """Perform permutation analysis for the given pipeline and PCA configuration.
    
    Args:
        pipeline_config (dict): Configuration for the pipeline.
        permuation_config (dict): Configuration for the permutation analysis.
        n_permutations (int): Number of permutations to perform.

    Returns:
        dict[str, Any]: Dictionary containing the results of the permutation analysis.
    """
    pipeline_config['functions']['preprocess'] = 'permute_rv'
    pipeline_config['preprocess_kwargs'] = {'source1_perm': permuation_config['source1_perm'],
                                            'source2_perm': permuation_config['source2_perm'],
                                            'target_perm': permuation_config['target_perm']}
    
    results = run_otc_experiment(pipeline_config)

    return results




if __name__ == "__main__":
    # Load the pipeline configuration from YAML file
    with open(Path(__file__).with_name("permuation_config.yaml"), "r") as f:
        permuation_config = yaml.safe_load(f)

    with open(Path(__file__).resolve().parents[3] / "full_OTC" / "otc_config.yaml", "r") as f:
        pipeline_config = yaml.safe_load(f)

    # Perform permutation analysis
    results = permuatation_analysis(pipeline_config, permuation_config)

    # Save the results to a file
    output_path = Path(__file__).resolve().parents[1] / "permuation_results.yaml"
    with open(output_path, "w") as f:
        yaml.dump(results, f)






