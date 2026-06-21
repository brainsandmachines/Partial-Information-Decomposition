
import sys
from pathlib import Path
import yaml


def smoke_example_config():
    """Load the configuration for the smoke example experiment from a YAML file.

    This function reads the 'smoke_example.yaml' file, which contains all necessary parameters
    for running the smoke example experiment, including model names, layer choices, and data paths.

    Returns:
        config: dict, containing all configuration parameters for the smoke example experiment.
    """

    with open("pipeline/smoke_example.yaml", "r") as f:
        config = yaml.safe_load(f)


    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    print(f"Project root directory: {PROJECT_ROOT}")
    deepdive_path = PROJECT_ROOT / 'third_party' / 'DeepDive' / 'deepdive'
    for path in [deepdive_path, PROJECT_ROOT / 'model_opts']:
        if str(path) not in sys.path:
            sys.path.append(str(path))



    with open("pipeline/smoke_example.yaml", "r") as f:
        config = yaml.safe_load(f)




    MODEL_1_NAME = config["sources"]["source1_name"]
    MODEL_2_NAME = config["sources"]["source2_name"]
    DEBUG_LAYER_1 = config["sources"]["DEBUG_LAYER_1"]
    DEBUG_LAYER_2 = config["sources"]["DEBUG_LAYER_2"]
    N_DEBUG_IMAGES = config["images"]["N_DEBUG_IMAGES"]
    BATCH_SIZE_PROCESS = config["images"]["BATCH_SIZE_PROCESS"]
    BATCH_SIZE_DATALOADER = config["images"]["BATCH_SIZE_DATALOADER"]

    subject_name = config["target"]["target_name"]
    betas_general_roi = config["target"]["betas_roi"]

    # --- Paths and Directories ---
    # NSD data paths
    NSD_ROOT = Path(config["paths"]["NSD_ROOT"])   # Should be modified when running on cluster or locally
    HDF_PATH = NSD_ROOT / Path(config["paths"]["HDF_PATH"])
    PKL_INFO_PATH = NSD_ROOT / Path(config["paths"]["PKL_INFO_PATH"])

    # Path to neural data
    NEURAL_DATA_PATH = Path(config["paths"]["NEURAL_DATA_PATH"])
    NEURAL_DATA_PATH = NEURAL_DATA_PATH / f"{subject_name}_{betas_general_roi}_betas_per_stimulus.zarr"

    # Directory to save per-voxel results
    per_voxel_results_dir = PROJECT_ROOT / "results/encoding/per_voxel_results"
    per_voxel_results_dir.mkdir(parents=True, exist_ok=True)


    #Pipeline functions
    layer_func = config["functions"]["layer_func_name"]
    manipulation_func = config["functions"]["manipulation_func_name"]

    return dict(
        MODEL_1_NAME=MODEL_1_NAME,
        MODEL_2_NAME=MODEL_2_NAME,
        DEBUG_LAYER_1=DEBUG_LAYER_1,
        DEBUG_LAYER_2=DEBUG_LAYER_2,
        layer_func=layer_func,
        manipulation_func=manipulation_func,
        N_DEBUG_IMAGES=N_DEBUG_IMAGES,
        BATCH_SIZE_PROCESS=BATCH_SIZE_PROCESS,
        BATCH_SIZE_DATALOADER=BATCH_SIZE_DATALOADER,
        subject_name=subject_name,
        betas_general_roi=betas_general_roi,
        HDF_PATH=HDF_PATH,
        PKL_INFO_PATH=PKL_INFO_PATH,
        NEURAL_DATA_PATH=NEURAL_DATA_PATH,
        per_voxel_results_dir=per_voxel_results_dir,
    )