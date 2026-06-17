from pathlib import Path
import sys

from sources_target_features import feature_extraction, prepare_sources, prepare_target
# Import DeepDive modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(f"Project root directory: {PROJECT_ROOT}")
deepdive_path = PROJECT_ROOT / 'third_party' / 'DeepDive' / 'deepdive'
for path in [deepdive_path, PROJECT_ROOT / 'model_opts']:
    if str(path) not in sys.path:
        sys.path.append(str(path))

HDF_PATH = Path("PATH_TO_NSD_STIM_HDF5")
PKL_INFO_PATH = Path("PATH_TO_NSD_INFO_PKL")
NEURAL_DATA_PATH = Path("PATH_TO_NEURAL_DATA")
MODEL_1_NAME = "alexnet_random"
MODEL_2_NAME = "resnet18_random"
DEBUG_LAYER_1 = None
DEBUG_LAYER_2 = None
N_DEBUG_IMAGES = 8
BATCH_SIZE_PROCESS = 4
BATCH_SIZE_DATALOADER = 4

subject_name = "subj01"
betas_general_roi = "OTC"

# --- Paths and Directories ---
# NSD data paths
NSD_ROOT = Path("/groups/golan_neurogroup/bml_group/datasets/nsddata")   # Should be modified when running on cluster or locally
HDF_PATH = NSD_ROOT / "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
PKL_INFO_PATH = NSD_ROOT / "nsddata/experiments/nsd/nsd_stim_info_merged.pkl"

# Path to neural data
NEURAL_DATA_PATH = Path("/groups/golan_neurogroup/bml_group/datasets/nsddata/otc_betas/otc_betas_per_stim")
NEURAL_DATA_PATH = NEURAL_DATA_PATH / f"{subject_name}_{betas_general_roi}_betas_per_stimulus.zarr"

# Directory to save per-voxel results
per_voxel_results_dir = PROJECT_ROOT / "results/encoding/per_voxel_results"
per_voxel_results_dir.mkdir(parents=True, exist_ok=True)



def main():
    """Run a cluster smoke test using the constants above as inputs and printed diagnostics as output."""
    target = prepare_target(HDF_PATH, PKL_INFO_PATH, NEURAL_DATA_PATH)
    sources = prepare_sources(MODEL_1_NAME, MODEL_2_NAME)
    ids = target["image_ids_for_subj"][:N_DEBUG_IMAGES].astype("int64")
    y = target["neural_data"][:N_DEBUG_IMAGES]
    assert target["stim"].dtype.kind in "uifb", f"stim must be numeric image data, got {target['stim'].dtype}"
    layer1 = DEBUG_LAYER_1 or sources["X1_context"]["layers_ordered"][0]
    layer2 = DEBUG_LAYER_2 or sources["X2_context"]["layers_ordered"][0]
    x1 = feature_extraction(layer1, sources["X1_context"], ids, target["stim"], BATCH_SIZE_PROCESS, BATCH_SIZE_DATALOADER)
    x2 = feature_extraction(layer2, sources["X2_context"], ids, target["stim"], BATCH_SIZE_PROCESS, BATCH_SIZE_DATALOADER)
    print(f"models: {MODEL_1_NAME} / {MODEL_2_NAME}")
    print(f"layers: {layer1} / {layer2}")
    print(f"image_ids: {ids[0]} .. {ids[-1]} ({len(ids)})")
    print(f"shapes: X1={x1.shape}, X2={x2.shape}, T={y.shape}")
    assert x1.shape[0] == x2.shape[0] == y.shape[0] == len(ids)


if __name__ == "__main__":
    main()
