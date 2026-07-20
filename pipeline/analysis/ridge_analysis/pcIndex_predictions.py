import numpy as np
import torch 
from pathlib import Path
from scipy.stats import pearsonr
import sys
repo_root = Path(__file__).resolve().parents[3]
external_root = repo_root / "external"
for path in (repo_root, external_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

        
from pipeline.pipeline_phases.feature_manipulations import prepare_ridge_target
from pipeline.pipeline_phases.choosing_layer import overall_best_layer
from pipeline.pipeline_phases.sources_target_features import prepare_target
from pipeline.pipeline_utils import nsd_feature_extraction
from pipeline.analysis.anlysis_utils import _prepare_source_for_pid
from external.mayas_project.features_and_encoding.feat_ext_and_encoding import prepare_model_context






# Target data and the saved PCA used to order/select the target PCs.
pc_path= Path("pipeline/subj_PCs/saved_pcs_nostandardization/pca_by_variance=200/subj01_pca_model.pkl")
hdf_path= Path("/groups/golan_neurogroup/bml_group/datasets/nsddata/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5")
pkl_info_path= Path("/groups/golan_neurogroup/bml_group/datasets/nsddata/nsddata/experiments/nsd/nsd_stim_info_merged.pkl")
neural_data_path= Path("/groups/golan_neurogroup/bml_group/datasets/nsddata/otc_betas/otc_betas_per_stim/subj01_OTC_betas_per_stimulus.zarr")






def each_pc_index_pred(model_name,n_pcs,pc_path, hdf_path, pkl_info_path, neural_data_path):
    """This function runs ridge regression on each PC of the target data, after loading the model and extracting
    the best layer features. It returns the correlation between the predicted and actual PC values for each PC.
    Args:
        model_name (str): The name of the model to be used for feature extraction.
        n_pcs (list): A list of integers representing the indices of the PCs to be analyzed.
        pc_path (Path): Path to the saved PCA model.
        hdf_path (Path): Path to the HDF5 file containing the stimuli.
        pkl_info_path (Path): Path to the pickle file containing stimulus information.
        neural_data_path (Path): Path to the Zarr file containing neural data.
    Returns:
        correlations (np.ndarray): An array of correlation values between the predicted and actual PC values for each PC.
    """

    target_context = prepare_target(
        Path(hdf_path),
        Path(pkl_info_path),
        Path(neural_data_path),
    )

    #Split the data into train and test sets, and PCA the target data accoring to already existing pca model.
    train_target, shared_target, shared_mask = prepare_ridge_target(
        target_context["target"],
        target_context,
        pc_path,
    )


    # Prepare the source data for the two models
    source_context = prepare_model_context(model_name)

    
    path_to_results = Path('/home/ohadshee/Desktop/Partial-Information-Decomposition/external/mayas_project/results_shared/encoding/best_layers/subj01_OTC_all_models_best_layer_overall.csv')
    #Layers
    layer1 = overall_best_layer(model_name,path_to_results = path_to_results)
    print(f"Starting to extract features for {model_name} at layer {layer1['l']}")
    features1 = nsd_feature_extraction(source_context, layer1['l'],target_context=target_context)

    correlations = np.full(n_pcs, np.nan)
    
    print(f"Running analysis for {len(n_pcs)} PCs")
    for f in range(len(n_pcs)):
        n_pc = n_pcs[f]
        print(f"Running analysis for {n_pc} PCs")
        # Prepare the source data for the two models
        #Extract just one PC from the target data for the current iteration
        pc_train_target = train_target[:,n_pc]

        pc_test_target = shared_target[:,n_pc]
        # Predict the target PCs on the shared images using ridge regression for both models
        source_pred = _prepare_source_for_pid(features1, pc_train_target, shared_mask, ridge=True)

        if np.std(pc_train_target) > 0 and np.std(source_pred) > 0:
            correlations[n_pc] = pearsonr(pc_test_target, source_pred).statistic

    return correlations


def save_correlations_to_csv(correlations, output_path):
    """Saves the correlations array to a CSV file.
    Args:
        correlations (np.ndarray): An array of correlation values.
        output_path (Path): The path where the CSV file will be saved.
    """
    np.savetxt(output_path, correlations, delimiter=",")
    print(f"Correlations saved to {output_path}")



def main():
    model_name = 'RN50_clip'
    n_pcs = list(range(200))  # Analyze the first 200 PCs
    correlations = each_pc_index_pred(model_name, n_pcs, pc_path, hdf_path, pkl_info_path, neural_data_path)
    print("Correlations between predicted and actual PC values for each PC:")
    print(correlations)
    output_path = Path("correlations_per_pc.csv")
    save_correlations_to_csv(correlations, output_path)



if __name__ == "__main__":
    main()
