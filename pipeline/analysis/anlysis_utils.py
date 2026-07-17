
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