import torch 
import numpy as np
from pathlib import Path
import sys
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import joblib
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



#Feature manipulation funtion: PCA target and ridge prediction from X1 and X2 to PCA(T) using the alphas per pc from the data
def feature_manipulation_ridge(source1,source2,target,target_context,seed,model_name_1,model_name_2,pc_target_path,alphas_source1_path,alphas_source2_path):
    """
    Perform feature manipulation on the source and target data.
    
    Args:
        source1 (np.ndarray): The first source data.
        source2 (np.ndarray): The second source data.
        target (np.ndarray): The target data.
        seed (int): The random seed for reproducibility.
        model_name_1 (str): The name of the first model.
        model_name_2 (str): The name of the second model.
        pc_target_path (str): The path to the PCA target data.
        alphas_source1_path (str): The path to the alphas for the first source.
        alphas_source2_path (str): The path to the alphas for the second source.
        shared1000_subj (np.ndarray): An array of shared subject IDs.
        """
    

    pca = joblib.load(pc_target_path)
    
    with np.load(alphas_source1_path, allow_pickle=True) as archive:
        alphas_source1 = np.asarray(
            archive["alphas"],
            dtype=np.float64,
        ).copy()

        if model_name_1 != archive["model_name"]:
            raise ValueError("Model name mismatch for source1.")
        
        pc_indices = np.asarray(archive["pc_indices"])

        if not np.array_equal(
            pc_indices,
            np.arange(1, len(alphas_source1) + 1),
        ):
            raise ValueError("Alpha PC ordering is invalid.")

    with np.load(alphas_source2_path, allow_pickle=False) as archive:
        alphas_source2 = np.asarray(
            archive["alphas"],
            dtype=np.float64,
        ).copy()

        pc_indices = np.asarray(archive["pc_indices"])

        if not np.array_equal(
            pc_indices,
            np.arange(1, len(alphas_source1) + 1),
        ):
            raise ValueError("Alpha PC ordering is invalid.")

        if model_name_2 != archive["model_name"]:
            raise ValueError("Model name mismatch for source2.")

    #PCA target data
    pca_target = pca.transform(target)

    shared_ids = target_context['shared1000_subj']


    pca_target_test = pca_target[shared_ids]
    source1_test = source1[shared_ids]
    source2_test = source2[shared_ids]


    training_indices = ~shared_ids

    #find betas 
    train_target = pca_target[training_indices]
    train_source1 = source1[training_indices]
    train_source2 = source2[training_indices]

    ridge_sourc1 = Ridge(alpha=alphas_source1, fit_intercept=True, random_state=seed)
    ridge_sourc1.fit(train_source1, train_target)

    ridge_sourc2 = Ridge(alpha=alphas_source2, fit_intercept=True, random_state=seed)
    ridge_sourc2.fit(train_source2, train_target)

    if pca_target.shape[1] != alphas_source1.shape[0]:
        raise ValueError(
            "There must be exactly one alpha for each target PC."
        )
    
    if pca_target.shape[1] != alphas_source2.shape[0]:
        raise ValueError(
        "There must be exactly one alpha for each target PC.")
    

        # Inputs should not contain invalid numerical values.
    if not np.isfinite(source1).all():
        raise ValueError("source1 contains NaN or infinite values.")

    if not np.isfinite(source2).all():
        raise ValueError("source2 contains NaN or infinite values.")

    if not np.isfinite(target).all():
        raise ValueError("target contains NaN or infinite values.")

    source1_pred = ridge_sourc1.predict(source1_test)
    source2_pred = ridge_sourc2.predict(source2_test)

        # Inputs should not contain invalid numerical values.
    if not np.isfinite(source1_pred).all():
        raise ValueError("source1 contains NaN or infinite values.")

    if not np.isfinite(source2_pred).all():
        raise ValueError("source2 contains NaN or infinite values.")

    return source1_pred, source2_pred, pca_target_test