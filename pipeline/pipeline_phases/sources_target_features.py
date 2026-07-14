import numpy as np
from pathlib import Path
import sys
import time
import joblib
repo_root = Path(__file__).resolve().parents[1]
external_root = repo_root / "external"
for path in (repo_root, external_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from external.mayas_project.features_and_encoding.feat_ext_and_encoding import DataLoader, ImageDatasetNSD, extract_features_per_layer, prepare_subject_context, prepare_model_context
""""This file utilized mayas_project for feature extraction."""

#NOTE: I need to add feature manipulation function and maybe move the n_project outside of feature_extraction. 




def prepare_sources(model_name_1:str,model_name_2:str) -> dict[str, dict]:
    """Prepare sources for feature extraction.

    Inputs:
        model_name_1: str, name of the first model/source.
        model_name_2: str, name of the second model/source.

    Output:
        sources: dict, model contexts under "X1_context" and "X2_context".
    """


    # Prepare model context
    model_1 = prepare_model_context(model_name_1)

    model_2 = prepare_model_context(model_name_2)

    return {'X1_context': model_1, 'X2_context': model_2}



def prepare_target(hdf_path:Path,pkl_info_path:Path,neural_data_path:Path) -> dict:
    """Prepare target for feature extraction. 
    Inputs:
        hdf_path: path to hdf file containing neural data
        pkl_info_path: path to pkl file containing info about the neural data
        neural_data_path: path to the directory containing the neural data

    Outputs: 
        per_subject_context = {
        "neural_data": neural_data,
        "stim": stim, ------->     Images presented to the subject during the experiment
        "hdf_file": hdf_file,
        "image_ids_for_subj": image_ids_for_subj,
        "shared1000_subj": shared1000_subj,
        "n_projections": n_projections,
        "kf": kf
        }

    """
    target_context = prepare_subject_context(hdf_path,pkl_info_path,neural_data_path)
    target_context["target"] = target_context.pop('neural_data')  # Add 'target' key for compatibility with PID pipeline
    return target_context


def shared1000_subj_target(
    hdf_path: str | Path,
    pkl_info_path: str | Path,
    neural_data_path: str | Path,
    pca_model_path: str | Path | None = None,
    scaler_model_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Load a subject's shared 1,000 stimuli and aligned neural responses.

    When both saved model paths are supplied, the neural responses are first
    transformed by the saved scaler and then projected into the saved PCA
    space. When neither model path is supplied, the raw shared neural
    responses are returned.

    Inputs:
        hdf_path: str or Path, NSD HDF5 file containing the global stimulus
            image dataset.
        pkl_info_path: str or Path, pickle file containing the NSD shared-1,000
            flags.
        neural_data_path: str or Path, subject neural-data Zarr directory.
        pca_model_path: str, Path, or None, saved fitted PCA model. It must be
            supplied together with ``scaler_model_path``.
        scaler_model_path: str, Path, or None, saved fitted scaler model. It
            must be supplied together with ``pca_model_path``.

    Output:
        shared_target: dict[str, np.ndarray], with ``neural_data`` of shape
            (1000, n_voxels) without models or (1000, n_components) with
            models, and ``stim`` containing the 1,000 aligned stimulus images.
    """

    has_pca_path = pca_model_path is not None
    has_scaler_path = scaler_model_path is not None
    if has_pca_path != has_scaler_path:
        raise ValueError(
            "pca_model_path and scaler_model_path must be provided together."
        )

    pca_model = None
    scaler_model = None
    if has_pca_path:
        pca_path = Path(pca_model_path)
        scaler_path = Path(scaler_model_path)
        if not pca_path.is_file():
            raise FileNotFoundError(f"PCA model does not exist: {pca_path}")
        if not scaler_path.is_file():
            raise FileNotFoundError(
                f"Scaler model does not exist: {scaler_path}"
            )
        pca_model = joblib.load(pca_path)
        scaler_model = joblib.load(scaler_path)

    subject_context = prepare_subject_context(
        Path(hdf_path),
        Path(pkl_info_path),
        Path(neural_data_path),
    )
    hdf_file = subject_context.get("hdf_file")

    try:
        neural_data = np.asarray(subject_context["neural_data"])
        shared_mask = np.asarray(
            subject_context["shared1000_subj"],
            dtype=bool,
        )
        image_ids = np.asarray(subject_context["image_ids_for_subj"])

        if neural_data.ndim != 2:
            raise ValueError(
                "neural_data must have shape (n_images, n_voxels), "
                f"but received {neural_data.shape}."
            )
        if shared_mask.ndim != 1 or shared_mask.shape[0] != neural_data.shape[0]:
            raise ValueError(
                "shared1000_subj must be a one-dimensional mask with one "
                "entry per neural-data row."
            )
        if image_ids.ndim != 1 or image_ids.shape[0] != neural_data.shape[0]:
            raise ValueError(
                "image_ids_for_subj must contain one image ID per "
                "neural-data row."
            )

        shared_count = int(np.count_nonzero(shared_mask))
        if shared_count != 1000:
            raise ValueError(
                "Expected exactly 1000 shared images, "
                f"but found {shared_count}."
            )

        shared_neural_data = neural_data[shared_mask]
        shared_image_ids = image_ids[shared_mask]
        if not np.issubdtype(shared_image_ids.dtype, np.integer):
            raise TypeError("Shared NSD image IDs must be integers.")

        stim_dataset = subject_context["stim"]
        if np.any(shared_image_ids < 0) or np.any(
            shared_image_ids >= stim_dataset.shape[0]
        ):
            raise IndexError(
                "A shared NSD image ID is outside the stimulus dataset."
            )

        shared_stim = np.empty(
            (shared_count, *stim_dataset.shape[1:]),
            dtype=stim_dataset.dtype,
        )
        for output_index, image_id in enumerate(shared_image_ids):
            shared_stim[output_index] = stim_dataset[int(image_id)]

        if pca_model is not None and scaler_model is not None:
            scaled_neural_data = scaler_model.transform(shared_neural_data)
            shared_neural_data = pca_model.transform(scaled_neural_data)
            expected_components = getattr(pca_model, "n_components_", None)
            if (
                shared_neural_data.ndim != 2
                or shared_neural_data.shape[0] != shared_count
                or (
                    expected_components is not None
                    and shared_neural_data.shape[1] != expected_components
                )
            ):
                raise ValueError(
                    "PCA-transformed neural data has an unexpected shape: "
                    f"{shared_neural_data.shape}."
                )

        return {
            "neural_data": np.asarray(shared_neural_data),
            "stim": shared_stim,
        }
    finally:
        if hdf_file is not None:
            hdf_file.close()


def prepare_target_for_voxel(voxel_index:int, subj_id:str, hdf_path:Path,pkl_info_path:Path,neural_data_path:Path) -> dict:
    """Prepare target for feature extraction for a specific voxel. 
    Inputs:
        voxel_index: index of the voxel to prepare the target for
        subj_id: subject ID
        hdf_path: path to hdf file containing neural data
        pkl_info_path: path to pkl file containing info about the neural data
        neural_data_path: path to the directory containing the neural data
        
    Outputs:
        per_subject_context = {
        "neural_data": neural_data, ----> For a given voxel, the neural data will be a 1D array of shape (n_images,) containing the responses of that voxel to each image.
        "stim": stim, ------->     Images presented to the subject during the experiment
        "hdf_file": hdf_file,
        "image_ids_for_subj": image_ids_for_subj,
        "shared1000_subj": shared1000_subj,
        "n_projections": n_projections,
        "kf": kf
        }"""
    
    subject_context = prepare_target(hdf_path,pkl_info_path,neural_data_path)
    neural_data = subject_context["neural_data"][:, voxel_index]

    return {**subject_context, "neural_data": neural_data}


def make_nsd_dataloader(model_context: dict, stim_dataset, image_ids: np.ndarray, batch_size: int) -> DataLoader:
    """Create a DataLoader for an ordered subset of NSD images.

    Inputs:
        model_context: dict, model context containing an "image_transforms" entry.
        stim_dataset: h5py.Dataset-like object, dataset containing stimulus images.
        image_ids: np.ndarray, ordered image IDs to include in this DataLoader.
        batch_size: int, number of images per DataLoader batch.

    Outputs:
        dataloader: DataLoader, loader over the selected transformed images in the same order as image_ids.
    """
    image_dataset = ImageDatasetNSD(stim_dataset, image_ids, transform=model_context["image_transforms"])

    dataloader = DataLoader(
      image_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=0)

    return dataloader


def batching(model_context: dict, batch_start: int, batch_end: int, stim_dataset, subj_image_ids: np.ndarray,
             layer_name: str, batch_size_dataloader: int) -> np.ndarray:
    """Batch process a range of images for feature extraction.
    Inputs:
        model_context: dict, context for the model to extract features from.
        batch_start: int, starting index of the batch
        batch_end: int, ending index of the batch
        stim_dataset: h5py.Dataset-like object, dataset containing the stimuli images
        subj_image_ids: np.ndarray, ordered image IDs for this subject
        layer_name: str, name of the layer to extract features from
        batch_size_dataloader: int, number of images per DataLoader batch

    Outputs:
        features_batch: np.ndarray, extracted features for this batch with rows aligned to subj_image_ids[batch_start:batch_end].
    """

    batch_image_ids = subj_image_ids[batch_start:batch_end]
    batched_dataloader = make_nsd_dataloader(
        model_context=model_context,
        stim_dataset=stim_dataset,
        image_ids=batch_image_ids,
        batch_size=batch_size_dataloader)
    
    features_batch = extract_features_per_layer(model_context["model"], dataloader=batched_dataloader, layer_name=layer_name)

    if hasattr(features_batch, "detach"):
        features_batch = features_batch.detach().cpu().numpy()

    return features_batch


def feature_extraction(layer_index: int, model_context: dict, subj_image_ids: np.ndarray,
                       stim_dataset, batch_size_process: int, batch_size_dataloader: int = 128) -> np.ndarray:
    """Extract features from the models and the neural data.
    
    Inputs:
        layer_name: str, name of the layer to extract features from.
        layer_index: int, index of the layer to extract features from.
        model_context: dict, context for a model, containing the model and its image transforms.
        subj_image_ids: np.ndarray, ordered image IDs for this subject.
        stim_dataset: h5py.Dataset-like object, dataset containing stimulus images.
        batch_size_process: int, number of subject images handled in each outer batch.
        batch_size_dataloader: int, number of images per DataLoader batch.
        
    Outputs:
        model_features: np.ndarray, extracted features for the model for the given layer.
    """


        # --- Get model-level context ---
    model_name = model_context["model_name"]
    layers_ordered = model_context["layers_ordered"]
    layer_name = layers_ordered[layer_index]

    # --- Check if layer name is valid ---
    print("\n" + "="*50)
    print(f"Processing layer: {layer_name} from model: {model_name}")
    assert layer_name in layers_ordered, f"Layer {layer_name} not found in model layers. Available layers: {layers_ordered}"

    num_images = subj_image_ids.shape[0]
    start_extraction = time.perf_counter()  # start timer for feature extraction
    model_features = None  # Initialize model_features to None
    print(f"\nFor model {model_name} Layer {layer_name}: Extracting features for {num_images} images in batches of {batch_size_process}...")
    for batch_start in range(0, num_images, batch_size_process):
        batch_end = min(batch_start + batch_size_process, num_images)
        features_batch = batching(
            model_context=model_context,
            batch_start=batch_start,
            batch_end=batch_end,
            stim_dataset=stim_dataset,
            subj_image_ids=subj_image_ids,
            layer_name=layer_name,
            batch_size_dataloader=batch_size_dataloader)
        if model_features is None:
            model_features = np.zeros((num_images, features_batch.shape[1]), dtype=np.float32)
        model_features[batch_start:batch_end] = features_batch
        del features_batch  # free memory after each batch

    end_extraction = time.perf_counter()  # end timer for feature extraction
    print(f"Feature extraction for model {model_name} layer {layer_name} completed in {end_extraction - start_extraction:.2f} seconds.")
    return model_features
