import numpy as np
from pathlib import Path
import sys
import time
from sklearn.externals import joblib
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
    return prepare_subject_context(hdf_path,pkl_info_path,neural_data_path)


def shared1000_subj_target(hdf_path:Path,pkl_info_path:Path,neural_data_path:Path, pca_model_path:Path,scaler_model_path:Path) -> np.ndarray:
        """This function returns the shared1000 images for a given and subject 
        and the corresponding neural data PCA'ed and scaled (if models are provided).

        If a pca_model and scaler model are provided, the function will use them to transform the neural data.

        if not returns the full neural data not scaled.

        Inputs:
            hdf_path: path to hdf file containing neural data
            pkl_info_path: path to pkl file containing info about the neural data
            neural_data_path: path to the directory containing the neural data
            pca_model_path: path to the PCA model
            scaler_model_path: path to the scaler model

        Outputs:
            shared1000_subj: np.ndarray, shared1000 images for the subject."""



        subject_context = prepare_subject_context(hdf_path,pkl_info_path,neural_data_path)

        if pca_model_path.exists() and scaler_model_path.exists():

            pca_model = joblib.load(pca_model_path)
            scaler_model = joblib.load(scaler_model_path)

            neural_data = subject_context["neural_data"]
            shared_neural_data = neural_data[subject_context["shared1000_subj"]]
            stim = subject_context["stim"][subject_context["shared1000_subj"]]
            neural_data_scaled = pca_model.transform(shared_neural_data)
            neural_data_pca = scaler_model.transform(neural_data_scaled) #(1000, n_components_)
            assert (neural_data_pca.shape[0],neural_data_pca.shape[1]) == (1000, pca_model.n_components_), "Shape mismatch between PCA transformed data and expected dimensions"

            return {'neural_data':neural_data_pca, 'stim': stim}


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
