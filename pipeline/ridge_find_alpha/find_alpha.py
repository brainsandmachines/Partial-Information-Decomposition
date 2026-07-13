import csv

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from pathlib import Path
import sys
import time
import joblib
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from pipeline.pipeline_utils import nsd_feature_extraction
from pipeline.pipeline_phases.sources_target_features import prepare_target,prepare_sources
from pipeline.pipeline_phases.choosing_layer import overall_best_layer

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from external.mayas_project.features_and_encoding.feat_ext_and_encoding import prepare_model_context

import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import numpy as np

from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def find_alpha_per_pc(
    predictor: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, Pipeline, StandardScaler]:
    """Standardize predictors and find one ridge alpha per target PC.

    Inputs:
        predictor:
            Array shaped (n_samples, n_features), containing model features.
        target:
            Array shaped (n_samples, n_components), containing target PC scores.

    Outputs:
        alphas_per_pc:
            Array shaped (n_components,), containing one alpha per target PC.
        ridge_model:
            Fitted pipeline that predicts all target PCs simultaneously.
        predictor_scaler:
            Fitted StandardScaler used by the pipeline.
    """
    predictor = np.asarray(predictor, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if predictor.ndim != 2:
        raise ValueError("predictor must be a two-dimensional array.")

    if target.ndim != 2:
        raise ValueError("target must be a two-dimensional array.")

    if predictor.shape[0] != target.shape[0]:
        raise ValueError(
            "predictor and target must have the same number of samples."
        )

    alphas = np.logspace(0, 14, 50)

    print(
        f"Finding best ridge alphas for {target.shape[1]} target PCs "
        f"using {predictor.shape[1]} features and "
        f"{predictor.shape[0]} samples."
    )

    ridge_model = make_pipeline(
        StandardScaler(),
        RidgeCV(
            alphas=alphas,
            cv=None,
            scoring=None,
            fit_intercept=True,
            alpha_per_target=True,
            gcv_mode="auto",
        ),
        verbose=True,
    )

    ridge_model.fit(predictor, target)

    ridge_cv = ridge_model.named_steps["ridgecv"]

    alphas_per_pc = np.asarray(
        ridge_cv.alpha_,
        dtype=np.float64,
    )

    predictor_scaler = ridge_model.named_steps["standardscaler"]

    assert alphas_per_pc.shape == (target.shape[1],), (
        "Number of selected alphas must match the number of target PCs."
    )

    return alphas_per_pc, ridge_model, predictor_scaler







def load_and_apply_pca(
    data: np.ndarray,
    pca_path: str | Path,
    scaler_path: str | Path,
) -> np.ndarray:
    """Scale data and transform it using a subject's fitted PCA model.

    Args:
        data: A 2D array shaped (n_samples, n_features). Its columns must match
            the features and ordering used when fitting the scaler and PCA.
        pca_path: Path to the saved PCA model.
        scaler_path: Path to the saved scaler model.

    Returns:
        A 2D array shaped (n_samples, n_components) containing PCA scores.
    """
    pca_path = Path(pca_path)
    scaler_path = Path(scaler_path)

    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)

    scaled_data = scaler.transform(data)
    transformed_data = pca.transform(scaled_data)

    return transformed_data




def main(
    source_name: str,
    path_to_results: str | Path,
    pc_path: str | Path,
    scaler_path: str | Path,
    hdf_path: Path,
    pkl_info_path: Path,
    neural_data_path: Path,
    alphas_csv_path: str | Path,
    predictor_scaler_path: str | Path | None = None,
) -> tuple[np.ndarray, MultiOutputRegressor]:
    """Find per-PC ridge alphas and save model, layer, and PC indexes.

    Inputs:
        source_name: str, model name stored in every CSV row.
        path_to_results: str or Path, best-layer results CSV.
        pc_path: str or Path, fitted PCA model.
        scaler_path: str or Path, fitted scaler model.
        hdf_path: Path, NSD stimulus HDF5 file.
        pkl_info_path: Path, NSD stimulus-information pickle.
        neural_data_path: Path, subject neural-data path.
        alphas_csv_path: str or Path, destination CSV file.
        predictor_scaler_path: str, Path, or None, saved predictor scaler path.
            When None, save it beside the alpha CSV using a model-specific
            filename so runs for different models do not overwrite each other.

    Output:
        tuple, per-PC alpha array and fitted multi-output ridge model.
    """

    # Prepare the target using the loaded scaler
    target = prepare_target(hdf_path, pkl_info_path, neural_data_path)

    unique_mask = ~np.asarray(target["shared1000_subj"], dtype=bool)

    unique_target_context = target.copy()

    unique_target_context["image_ids_for_subj"] = np.asarray(
    target["image_ids_for_subj"])[unique_mask]

    unique_target_context["neural_data"] = np.asarray(
    target["neural_data"])[unique_mask]


    pca_target = load_and_apply_pca(unique_target_context["neural_data"], pc_path, scaler_path)

    #Save memory 
    del target 

    #Prepare model
    model_context = prepare_model_context(source_name)
    model_layer = overall_best_layer(source_name,path_to_results)
    layer_index = model_layer['l']

    if layer_index is None:
        raise ValueError(f"No best layer found for model {source_name}. Please check the results CSV at {path_to_results}.")

    features = nsd_feature_extraction(model_context,layer_index,unique_target_context,batch_size_process=64)

    #Save memory
    del model_context


    # Find the best ridge alpha for each target PC
    alphas_per_pc, ridge_model, predictor_scaler = find_alpha_per_pc(
        features,
        pca_target,
    )

    output_path = Path(alphas_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_source_name = str(source_name).replace("/", "_").replace("\\", "_")
    scaler_output_path = (
        Path(predictor_scaler_path)
        if predictor_scaler_path is not None
        else output_path.with_name(
            f"{output_path.stem}_{safe_source_name}_predictor_scaler.pkl"
        )
    )
    scaler_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor_scaler, scaler_output_path)

    csv_header = ["model_name", "layer_index", "pc_index", "alpha"]
    csv_has_content = output_path.is_file() and output_path.stat().st_size > 0
    if csv_has_content:
        with output_path.open("r", newline="", encoding="utf-8") as csv_file:
            existing_header = next(csv.reader(csv_file), None)
        if existing_header != csv_header:
            raise ValueError(
                f"Existing alpha CSV has an incompatible header: {existing_header}. "
                f"Expected: {csv_header}."
            )

    write_mode = "a" if csv_has_content else "w"
    with output_path.open(write_mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not csv_has_content:
            writer.writerow(csv_header)
        writer.writerows(
            (source_name, int(layer_index), pc_index, float(alpha))
            for pc_index, alpha in enumerate(alphas_per_pc, start=1)
        )
        print("\nSelected alpha and cross-validated R² for each PC:")

    ridge_cv = ridge_model.named_steps["ridgecv"]
    
    alphas_per_pc = np.asarray(
    ridge_cv.alpha_,
    dtype=np.float64,
    )

    scores_per_pc = np.asarray(
        ridge_cv.best_score_,
        dtype=np.float64,
    )

    print("Finished finding best ridge alphas for each target PC ✅")
    for pc_index, (alpha, score) in enumerate(
        zip(alphas_per_pc, scores_per_pc),
        start=1,
    ):
        print(
            f"\nPC {pc_index:>3}: "
            f"alpha = {alpha:.6g}, "
            f"CV R² = {score:.6f}"
        )

    mean_score = float(np.mean(scores_per_pc))

    print(f"\nMean cross-validated R² across PCs: {mean_score:.6f}")


    print(f"\n======================================================================================")

    return alphas_per_pc, ridge_model



if __name__ == "__main__":

    model_list = ['nf_resnet50_classification','eca_nfnet_l0_classification',
        'resnet50_classification','semnasnet_100_classification','cspresnet50_classification',
        'mobilenetv3_large_100_classification','ghostnet_100_classification','convnext_base_classification','xcit_nano_12_p8_224_classification'
        ,'xcit_nano_12_p16_224_classification','swin_large_patch4_window7_224_classification','jx_nest_tiny_classification',''
        'pit_ti_224_classification','vit_base_patch32_224_classification','vit_base_patch16_224_classification',
        'tnt_s_patch16_224_classification','crossvit_base_240_classification','deit_base_patch16_224_classification',
        'levit_128_classification','coat_lite_tiny_classification','visformer_small_classification',
        'convit_base_classification','ViT-B_32_clip','RN50_clip','RN101_clip','ViT-L_14_clip',
        'ResNet50-SimCLR_selfsupervised','ResNet50-DeepClusterV2-2x224_selfsupervised','ResNet50-SwAV-BS4096-2x224_selfsupervised',
        'ResNet50-PIRL_selfsupervised','ResNet50-ClusterFit-16K-RotNet_selfsupervised','ResNet50-MoCoV2-BS256_selfsupervised'
        ]
        #Path to best layer results
    path_to_results = Path('/home/ohadshee/Desktop/Partial-Information-Decomposition/external/mayas_project/results_shared/encoding/best_layers/subj01_OTC_all_models_best_layer_overall.csv')
    
    scaler_path = Path('/home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/subj_PCs/saved_pcs/pca_by_variance=60/subj01_scaler_model.pkl')
    pc_path = Path('/home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/subj_PCs/saved_pcs/pca_by_variance=60/subj01_pca_model.pkl')
    
    hdf_path = Path('/groups/golan_neurogroup/bml_group/datasets/nsddata/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5')
    pkl_info_path = Path('/groups/golan_neurogroup/bml_group/datasets/nsddata/nsddata/experiments/nsd/nsd_stim_info_merged.pkl')
    neural_data_path = Path('/groups/golan_neurogroup/bml_group/datasets/nsddata/otc_betas/otc_betas_per_stim/subj01_OTC_betas_per_stimulus.zarr')

    path_to_alphas_csv = Path('pipeline/ridge_find_alpha/results/alphas_per_pc2.csv')
    predictor_scaler_path = Path('pipeline/ridge_find_alpha/results/scalers')
    for source_name in model_list:

        print("\nChosen model:", source_name  )


        main(source_name,path_to_results,pc_path,scaler_path,hdf_path,pkl_info_path,neural_data_path,path_to_alphas_csv)
