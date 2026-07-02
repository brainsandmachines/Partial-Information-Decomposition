import torch 
import numpy as np
from pathlib import Path
import sys
from sklearn.decomposition import PCA


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
