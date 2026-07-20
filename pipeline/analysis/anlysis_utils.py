
from pipeline.pipeline_phases.feature_manipulations import pca_source
import numpy as np

DEEPDIVE_MODEL_NAME_CONVERSIONS: dict[str, str] = {
    "ViT-B_32_clip": "ViT-B/32_clip",
    "ViT-L_14_clip": "ViT-L/14_clip",
}



def to_deepdive_model_name(model_name: str) -> str:
    """Convert a stored model alias to the canonical name expected by DeepDive.

    Inputs:
        model_name: str, model identifier used by the pairwise model list and
            best-layer results CSV.

    Output:
        deepdive_model_name: str, canonical DeepDive model identifier. Names
            that do not require conversion are returned unchanged.
    """

    deepdive_model_name = DEEPDIVE_MODEL_NAME_CONVERSIONS.get(model_name, model_name)
    if deepdive_model_name != model_name:
        print(f"Changed model name: {model_name} -> {deepdive_model_name}")
    return deepdive_model_name


def _prepare_source_for_pid(
    source: np.ndarray,
    train_target: np.ndarray,
    shared_mask: np.ndarray,
    ridge: bool,
) -> np.ndarray:
    """Prepare one model source for PID on the held-out shared images.

    Inputs:
        source: np.ndarray with model features for every image.
        train_target: np.ndarray with target-PC scores for non-shared images.
        shared_mask: np.ndarray selecting the held-out shared images.
        ridge: bool indicating whether to predict target PCs with ridge.

    Outputs:
        np.ndarray containing held-out ridge predictions when ``ridge`` is
        true, or held-out source features otherwise.
    """

    if not ridge:
        return pca_source(source, shared_mask, train_target.shape[1])

    from pipeline.ridge_find_alpha.find_alpha import find_alpha_per_pc

    _, ridge_model = find_alpha_per_pc(source[~shared_mask], train_target)
    return ridge_model.predict(source[shared_mask])



