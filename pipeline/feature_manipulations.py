import torch 
import numpy as np
from pathlib import Path
import sys


repo_root = Path(__file__).resolve().parents[1]
external_root = repo_root / "external"
for path in (repo_root, external_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)



"""This file will contatin feature manipulatioin function afer feature extraiction. For example, 
 n_project function or PCA or ICA etc..."""



def pca_projection(features, n_components):
    """Apply PCA projection to reduce dimensionality of features.
    Input: 
        - features: numpy array or torch tensor of shape (n_samples, n_features)
        - n_components: number of principal components to keep
        
    Output:
        - ft_reduced: numpy array of shape (n_samples, n_components) with the PCA-reduced features
    """
    from sklearn.decomposition import PCA

    if type(features) == torch.Tensor:
        features = features.cpu().numpy()
    pca = PCA(n_components=n_components)
    ft_reduced = pca.fit_transform(features)
    return ft_reduced


def jl_projection(features, n_samples, eps=0.1, jl_dim=None):
    """Apply Johnson-Lindenstrauss projection to reduce dimensionality of features.
    Input: 
        - features: numpy array or torch tensor of shape (n_samples, n_features)
        - n_samples: number of samples
        - eps: approximation error tolerance
        - jl_dim: target dimension for JL projection (if None, it will be calculated using johnson_lindenstrauss_min_dim)
    Output:
        - ft_reduced: numpy array of shape (n_samples, jl_dim) with the JL-reduced features
    """ 
    from sklearn.random_projection import johnson_lindenstrauss_min_dim
    from external.mayas_project.features_and_encoding.feat_ext_and_encoding import get_sparse_projection_gpu

    if type(features) == np.ndarray:
        features = torch.from_numpy(features).float()  # convert to torch tensor on CPU

    if jl_dim is None:
        #"Find a 'safe' number of components to randomly project to.
        jl_dim = johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps) 

    sparse_projection = get_sparse_projection_gpu(features.shape[1], jl_dim, device=str(features.device))  # shape: (jl_dim, n_features)


        # --- Reduce Features using Sparse Random Projection ---
    with torch.no_grad():  # no need to track gradients here
        # Multiply the features with the sparse projection matrix on GPU to reduce dimensionality

        # shape: ((jl_dim, n_features) @ (n_features, n_samples)).t() -> (n_samples, jl_dim)
        ft_reduced = torch.sparse.mm(sparse_projection, features.t()).t()   
        ft_reduced = ft_reduced.cpu().numpy()  # move back to CPU and convert to numpy array
    
    return ft_reduced


def ica_projection(features):
    """Placeholder for ICA feature reduction.

    Input:
        - features: numpy array or torch tensor of shape (n_samples, n_features)

    Output:
        - Raises NotImplementedError because ICA reduction is not implemented yet.
    """
    raise NotImplementedError("ica_projection is not implemented yet.")


def cca_projection(features1, features2, n_components):
    """Apply Canonical Correlation Analysis (CCA) to find linear combinations of two sets of features that are maximally correlated.
    Input: 
        - features1: numpy array or torch tensor of shape (n_samples, n_features1)
        - features2: numpy array or torch tensor of shape (n_samples, n_features2)
        - n_components: number of canonical components to keep
        
    Output:
        - X_c: numpy array of shape (n_samples, n_components) with the CCA-transformed features from features1
        - Y_c: numpy array of shape (n_samples, n_components) with the CCA-transformed features from features2
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_decomposition import CCA

    if type(features1) == torch.Tensor:
        features1 = features1.cpu().numpy()
    if type(features2) == torch.Tensor:
        features2 = features2.cpu().numpy()


    # Important: standardize before CCA
    sts1 = StandardScaler()
    sts2 = StandardScaler()

    X_scaled = sts1.fit_transform(features1)
    Y_scaled = sts2.fit_transform(features2)

    cca = CCA(n_components=n_components)
    X1_c, X2_c = cca.fit_transform(X_scaled, Y_scaled)
    return X1_c, X2_c


def run_feature_reduction_smoke(config_path: Path | str = repo_root / "pipeline" / "smoke_example.yaml") -> dict:
    """Run a small smoke test for the feature reduction functions using source features from the real-data smoke config.

    Input:
        - config_path: Path or str, path to the YAML config used by the source/target feature smoke example.

    Output:
        - results: dict, mapping each reduction name to its status and output shape, with printed diagnostics.
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    try:
        from pipeline.sources_target_features import prepare_sources, prepare_target, feature_extraction
    except Exception as exc:
        raise RuntimeError(
            "Could not import sources_target_features. Check the h5py/numpy environment and the source smoke config."
        ) from exc

    sources_config = config["sources"]
    target_config = config["target"]
    images_config = config["images"]
    paths_config = config["paths"]

    nsd_root = Path(paths_config["NSD_ROOT"])
    hdf_path = nsd_root / Path(paths_config["HDF_PATH"])
    pkl_info_path = nsd_root / Path(paths_config["PKL_INFO_PATH"])
    neural_data_path = Path(paths_config["NEURAL_DATA_PATH"]) / f"{target_config['target_name']}_{target_config['betas_roi']}_betas_per_stimulus.zarr"

    target = prepare_target(hdf_path, pkl_info_path, neural_data_path)
    sources = prepare_sources(sources_config["source1_name"], sources_config["source2_name"])

    n_debug_images = int(images_config["N_DEBUG_IMAGES"])
    batch_size_process = int(images_config["BATCH_SIZE_PROCESS"])
    batch_size_dataloader = int(images_config["BATCH_SIZE_DATALOADER"])
    ids = target["image_ids_for_subj"][:n_debug_images].astype("int64")
    y = target["neural_data"][:n_debug_images]

    layer1 = sources_config.get("DEBUG_LAYER_1") or sources["X1_context"]["layers_ordered"][0]
    layer2 = sources_config.get("DEBUG_LAYER_2") or sources["X2_context"]["layers_ordered"][0]
    x1 = feature_extraction(layer1, sources["X1_context"], ids, target["stim"], batch_size_process, batch_size_dataloader)
    x2 = feature_extraction(layer2, sources["X2_context"], ids, target["stim"], batch_size_process, batch_size_dataloader)

    pca_components = max(1, min(2, x1.shape[0], x1.shape[1]))
    jl_dim = max(1, min(2, x1.shape[1]))
    cca_components = max(1, min(2, x1.shape[0], x1.shape[1], x2.shape[1]))

    results = {
        "inputs": {
            "X1": tuple(x1.shape),
            "X2": tuple(x2.shape),
            "T": tuple(y.shape),
            "layer1": layer1,
            "layer2": layer2,
        }
    }

    pca_x1 = pca_projection(x1, pca_components)
    results["pca_projection"] = {"status": "ok", "shape": tuple(pca_x1.shape)}

    jl_x1 = jl_projection(torch.as_tensor(x1, dtype=torch.float32), n_samples=x1.shape[0], jl_dim=jl_dim)
    results["jl_projection"] = {"status": "ok", "shape": tuple(jl_x1.shape)}

    cca_x1, cca_x2 = cca_projection(x1, x2, cca_components)
    results["cca_projection"] = {"status": "ok", "X1_shape": tuple(cca_x1.shape), "X2_shape": tuple(cca_x2.shape)}

    try:
        ica_projection(x1)
    except NotImplementedError as exc:
        results["ica_projection"] = {"status": "skipped", "reason": str(exc)}

    print("Feature reduction smoke test")
    print(f"inputs: X1={x1.shape}, X2={x2.shape}, T={y.shape}")
    print(f"pca_projection: {results['pca_projection']}")
    print(f"jl_projection: {results['jl_projection']}")
    print(f"cca_projection: {results['cca_projection']}")
    print(f"ica_projection: {results['ica_projection']}")
    return results


if __name__ == "__main__":
    config_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else repo_root / "pipeline" / "smoke_example.yaml"
    run_feature_reduction_smoke(config_arg)
