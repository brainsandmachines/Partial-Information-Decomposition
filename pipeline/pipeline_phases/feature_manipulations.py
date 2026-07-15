import torch 
import numpy as np
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any
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



def prepare_ridge_target(
    target: np.ndarray,
    target_context: Mapping[str, Any],
    pc_target_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a target with a saved PCA and split it by the shared mask.

    Inputs:
        target:
            np.ndarray with shape ``(n_samples, n_target_features)``. The
            target must already be scaled with the scaler used to fit the
            saved PCA model.
        target_context:
            Mapping[str, Any] containing a one-dimensional Boolean
            ``shared1000_subj`` mask aligned with the target rows. ``True``
            rows form the held-out PID target.
        pc_target_path:
            str or Path pointing to the fitted target PCA artifact.

    Outputs:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            ``(train_target, test_target, shared_mask)``. The target arrays
            contain PCA scores for non-shared and shared rows respectively;
            ``shared_mask`` is a copied Boolean mask aligned to the original
            sample order.
    """
    target_array = np.asarray(target)
    if target_array.ndim != 2:
        raise ValueError("target must be a two-dimensional array.")
    if target_array.shape[0] == 0 or target_array.shape[1] == 0:
        raise ValueError("target must contain at least one sample and feature.")
    if not np.issubdtype(target_array.dtype, np.number) or np.iscomplexobj(target_array):
        raise TypeError("target must contain real numeric values.")
    if not np.isfinite(target_array).all():
        raise ValueError("target contains NaN or infinite values.")

    if not isinstance(target_context, Mapping):
        raise TypeError("target_context must be a mapping.")
    if "shared1000_subj" not in target_context:
        raise KeyError("target_context must contain 'shared1000_subj'.")

    shared_mask = np.asarray(target_context["shared1000_subj"])
    if shared_mask.ndim != 1 or shared_mask.dtype.kind != "b":
        raise ValueError("shared1000_subj must be a one-dimensional Boolean mask.")
    if shared_mask.shape[0] != target_array.shape[0]:
        raise ValueError(
            "shared1000_subj must contain one entry for each target sample."
        )
    if not shared_mask.any():
        raise ValueError("shared1000_subj must select at least one held-out sample.")
    if shared_mask.all():
        raise ValueError("shared1000_subj must leave at least one training sample.")
    shared_mask = shared_mask.copy()

    #LOAD the PCA model and project the target
    pca = joblib.load(Path(pc_target_path))
    if not callable(getattr(pca, "transform", None)):
        raise TypeError("The target PCA artifact must provide a transform method.")

    projected_target = np.asarray(pca.transform(target_array))

    #Validations
    if projected_target.ndim != 2:
        raise ValueError("The target PCA transform must return a two-dimensional array.")
    if projected_target.shape[0] != target_array.shape[0]:
        raise ValueError("The target PCA transform changed the number of samples.")
    if projected_target.shape[1] == 0:
        raise ValueError("The target PCA transform returned no components.")
    if not np.issubdtype(projected_target.dtype, np.number) or np.iscomplexobj(
        projected_target
    ):
        raise TypeError("The target PCA transform must return real numeric values.")
    if not np.isfinite(projected_target).all():
        raise ValueError("The PCA-projected target contains NaN or infinite values.")

    return (
        projected_target[~shared_mask],
        projected_target[shared_mask],
        shared_mask,
    )


def load_ridge_alphas(
    alphas_path: str | Path,
    *,
    model_name: str,
    expected_target_dim: int,
    expected_layer_index: int | None = None,
) -> np.ndarray:
    """Load and validate a model's per-target-PC ridge penalties.

    Inputs:
        alphas_path:
            str or Path pointing to an ``.npz`` archive with ``alphas``,
            ``pc_indices``, ``model_name``, and ``layer_index`` fields.
        model_name:
            str containing the exact model identifier expected in the
            archive.
        expected_target_dim:
            int giving the number of PCA target components and therefore the
            required number of ridge penalties.
        expected_layer_index:
            int or None. When provided, the archived layer index must equal
            this value; when None, the scalar layer metadata is still checked
            for validity but is not compared to a requested layer.

    Outputs:
        np.ndarray:
            A copied, one-dimensional ``float64`` array with one finite,
            non-negative ridge penalty per target component.
    """
    if isinstance(expected_target_dim, (bool, np.bool_)) or not isinstance(
        expected_target_dim, (int, np.integer)
    ):
        raise TypeError("expected_target_dim must be an integer.")
    expected_target_dim = int(expected_target_dim)
    if expected_target_dim <= 0:
        raise ValueError("expected_target_dim must be positive.")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string.")

    required_fields = {"alphas", "pc_indices", "model_name", "layer_index"}
    with np.load(Path(alphas_path), allow_pickle=False) as archive:
        missing_fields = required_fields.difference(archive.files)
        if missing_fields:
            missing_text = ", ".join(sorted(missing_fields))
            raise ValueError(f"The alpha archive is missing required fields: {missing_text}.")

        raw_alphas = np.asarray(archive["alphas"])
        pc_indices = np.asarray(archive["pc_indices"])
        archived_model_metadata = np.asarray(archive["model_name"])
        archived_layer_metadata = np.asarray(archive["layer_index"])

    if not np.issubdtype(raw_alphas.dtype, np.number) or np.iscomplexobj(raw_alphas):
        raise TypeError("Ridge alphas must contain real numeric values.")
    alphas = np.asarray(raw_alphas, dtype=np.float64).copy()
    if alphas.shape != (expected_target_dim,):
        raise ValueError(
            "There must be exactly one alpha for each target PC: "
            f"expected {expected_target_dim}, found {alphas.size}."
        )
    if not np.isfinite(alphas).all():
        raise ValueError("Ridge alphas contain NaN or infinite values.")
    if (alphas < 0).any():
        raise ValueError("Ridge alphas must be non-negative.")

    expected_pc_indices = np.arange(1, expected_target_dim + 1)
    pc_indices_are_numeric = np.issubdtype(pc_indices.dtype, np.number) and not np.iscomplexobj(
        pc_indices
    )
    if (
        pc_indices.shape != (expected_target_dim,)
        or not pc_indices_are_numeric
        or not np.isfinite(pc_indices).all()
        or not np.array_equal(pc_indices, expected_pc_indices)
    ):
        raise ValueError("Alpha PC indices must be ordered and one-based.")

    if archived_model_metadata.size != 1:
        raise ValueError("Alpha archive model_name metadata must be scalar.")
    archived_model_value = archived_model_metadata.reshape(-1)[0]
    if isinstance(archived_model_value, (bytes, np.bytes_)):
        archived_model_name = archived_model_value.decode("utf-8")
    else:
        archived_model_name = str(archived_model_value)
    if archived_model_name != model_name:
        raise ValueError(
            "Alpha archive model name mismatch: "
            f"expected {model_name!r}, found {archived_model_name!r}."
        )

    if archived_layer_metadata.size != 1:
        raise ValueError("Alpha archive layer_index metadata must be scalar.")
    archived_layer_value = archived_layer_metadata.reshape(-1)[0]
    if isinstance(archived_layer_value, (bool, np.bool_)):
        raise ValueError("Alpha archive layer_index metadata must be an integer.")
    try:
        archived_layer_index = int(archived_layer_value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "Alpha archive layer_index metadata must be an integer."
        ) from error
    try:
        layer_is_integral = float(archived_layer_value) == archived_layer_index
    except (TypeError, ValueError, OverflowError):
        layer_is_integral = False
    if not layer_is_integral:
        raise ValueError("Alpha archive layer_index metadata must be an integer.")
    if archived_layer_index < 0:
        raise ValueError("Alpha archive layer_index metadata must be non-negative.")

    if expected_layer_index is not None:
        if isinstance(expected_layer_index, (bool, np.bool_)) or not isinstance(
            expected_layer_index, (int, np.integer)
        ):
            raise TypeError("expected_layer_index must be an integer or None.")
        if int(expected_layer_index) < 0:
            raise ValueError("expected_layer_index must be non-negative.")
        if archived_layer_index != int(expected_layer_index):
            raise ValueError(
                "Alpha archive layer index mismatch: "
                f"expected {int(expected_layer_index)}, found {archived_layer_index}."
            )

    return alphas


def ridge_predict_shared(
    source: np.ndarray,
    train_target: np.ndarray,
    shared_mask: np.ndarray,
    alphas: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Fit per-target ridge models and predict only the held-out rows.

    Inputs:
        source:
            np.ndarray with shape ``(n_samples, n_source_features)``. Rows
            must follow the same sample order as ``shared_mask`` and the array
            must already be scaled.
        train_target:
            np.ndarray with shape ``(n_train, n_target_components)`` containing
            PCA target scores for rows where ``shared_mask`` is False.
        shared_mask:
            np.ndarray Boolean mask with shape ``(n_samples,)``. True rows are
            excluded from fitting and used only for prediction.
        alphas:
            np.ndarray with shape ``(n_target_components,)`` containing one
            non-negative ridge penalty per target component.
        seed:
            int passed to scikit-learn's Ridge estimator as ``random_state``.

    Outputs:
        np.ndarray:
            Finite held-out predictions with shape
            ``(shared_mask.sum(), n_target_components)``.
    """
    source_array = np.asarray(source)
    train_target_array = np.asarray(train_target)
    shared_mask_array = np.asarray(shared_mask)
    raw_alphas = np.asarray(alphas)
    if not np.issubdtype(raw_alphas.dtype, np.number) or np.iscomplexobj(raw_alphas):
        raise TypeError("Ridge alphas must contain real numeric values.")
    alphas_array = np.asarray(raw_alphas, dtype=np.float64)

    if source_array.ndim != 2:
        raise ValueError("source must be a two-dimensional array.")
    if source_array.shape[0] == 0 or source_array.shape[1] == 0:
        raise ValueError("source must contain at least one sample and feature.")
    if not np.issubdtype(source_array.dtype, np.number) or np.iscomplexobj(source_array):
        raise TypeError("source must contain real numeric values.")
    if not np.isfinite(source_array).all():
        raise ValueError("source contains NaN or infinite values.")

    if train_target_array.ndim != 2:
        raise ValueError("train_target must be a two-dimensional array.")
    if train_target_array.shape[0] == 0 or train_target_array.shape[1] == 0:
        raise ValueError("train_target must contain at least one sample and component.")
    if not np.issubdtype(train_target_array.dtype, np.number) or np.iscomplexobj(
        train_target_array
    ):
        raise TypeError("train_target must contain real numeric values.")
    if not np.isfinite(train_target_array).all():
        raise ValueError("train_target contains NaN or infinite values.")

    if shared_mask_array.ndim != 1 or shared_mask_array.dtype.kind != "b":
        raise ValueError("shared_mask must be a one-dimensional Boolean array.")
    if shared_mask_array.shape[0] != source_array.shape[0]:
        raise ValueError("shared_mask must contain one entry for each source sample.")
    n_test = int(shared_mask_array.sum())
    n_train = int((~shared_mask_array).sum())
    if n_test == 0:
        raise ValueError("shared_mask must select at least one held-out sample.")
    if n_train != train_target_array.shape[0]:
        raise ValueError(
            "The number of non-shared source rows must equal the number of "
            "train_target rows."
        )

    target_dim = train_target_array.shape[1]
    if alphas_array.shape != (target_dim,):
        raise ValueError("There must be exactly one alpha for each target PC.")
    if not np.isfinite(alphas_array).all():
        raise ValueError("Ridge alphas contain NaN or infinite values.")
    if (alphas_array < 0).any():
        raise ValueError("Ridge alphas must be non-negative.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")

    ridge = Ridge(alpha=alphas_array, fit_intercept=True, random_state=int(seed))
    ridge.fit(source_array[~shared_mask_array], train_target_array)
    predictions = np.asarray(ridge.predict(source_array[shared_mask_array]))
    if predictions.ndim == 1 and target_dim == 1:
        predictions = predictions.reshape(-1, 1)

    if predictions.shape != (n_test, target_dim):
        raise ValueError(
            "Ridge predictions have an unexpected shape: "
            f"expected {(n_test, target_dim)}, found {predictions.shape}."
        )
    if not np.isfinite(predictions).all():
        raise ValueError("Ridge predictions contain NaN or infinite values.")

    return predictions


# Feature manipulation function: PCA target and ridge prediction from X1 and
# X2 to PCA(T) using the saved per-PC alphas.
def feature_manipulation_ridge(
    source1: np.ndarray,
    source2: np.ndarray,
    target: np.ndarray,
    target_context: Mapping[str, Any],
    seed: int,
    model_name_1: str,
    model_name_2: str,
    pc_target_path: str | Path,
    alphas_source1_path: str | Path,
    alphas_source2_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create held-out ridge predictions for two sources against PCA(T).

    Inputs:
        source1:
            np.ndarray with shape ``(n_samples, n_source1_features)``. The
            first model features must already be scaled.
        source2:
            np.ndarray with shape ``(n_samples, n_source2_features)``. The
            second model features must already be scaled.
        target:
            np.ndarray with shape ``(n_samples, n_target_features)``. The
            neural target must already be scaled.
        target_context:
            Mapping[str, Any] containing the Boolean ``shared1000_subj`` mask
            that identifies held-out rows.
        seed:
            int passed to each Ridge estimator as ``random_state``.
        model_name_1:
            str identifying source 1; it must match its alpha archive.
        model_name_2:
            str identifying source 2; it must match its alpha archive.
        pc_target_path:
            str or Path pointing to the fitted target PCA artifact.
        alphas_source1_path:
            str or Path pointing to source 1's per-PC alpha archive.
        alphas_source2_path:
            str or Path pointing to source 2's per-PC alpha archive.

    Outputs:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            ``(source1_prediction, source2_prediction, test_target)``. Every
            array contains only shared held-out rows, and all three use the
            saved target-PCA component space.
    """
    train_target, test_target, shared_mask = prepare_ridge_target(
        target,
        target_context,
        pc_target_path,
    )
    target_dim = train_target.shape[1]

    alphas_source1 = load_ridge_alphas(
        alphas_source1_path,
        model_name=model_name_1,
        expected_target_dim=target_dim,
    )
    alphas_source2 = load_ridge_alphas(
        alphas_source2_path,
        model_name=model_name_2,
        expected_target_dim=target_dim,
    )

    source1_prediction = ridge_predict_shared(
        source1,
        train_target,
        shared_mask,
        alphas_source1,
        seed=seed,
    )
    source2_prediction = ridge_predict_shared(
        source2,
        train_target,
        shared_mask,
        alphas_source2,
        seed=seed,
    )

    return source1_prediction, source2_prediction, test_target
