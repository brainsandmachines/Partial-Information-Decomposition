import numpy as np
from pathlib import Path
import sys
import time


repo_root = Path("/home/ohadshee/Desktop/Thesis_Ohad_Sheelo")
sys.path.append(str(repo_root))

from external.mayas_project.features_and_encoding.feat_ext_and_encoding import DataLoader, ImageDatasetNSD, extract_features_per_layer, prepare_subject_context, prepare_model_context
""""This file utilized mayas_project for feature extraction."""

#NOTE: I need to add feature manipulation function and maybe move the n_project outside of feature_extraction. 




def prepare_sources(model_1:str,model_2:str) -> dict[str, dict]:
    """Prepare sources for feature extraction.
    
    loads the model contexts for the two models and prepares them for feature extraction.
    
    
    Output:
        per_model_context = {
        "model_name": model_name,
        "model": model,
        "image_transforms": image_transforms,
        "layers_ordered": layers_ordered
    }"""


    # Prepare model context
    model_1 = prepare_model_context(model_1)

    model_2 = prepare_model_context(model_2)

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


def extract_NSD_model_transform(model,stim_dataset,subj_image_ids):
    """Create a full-subject NSD DataLoader using a model's image transforms.
    
    Inputs:
        model: dict, model context containing an "image_transforms" entry.
        stim_dataset: h5py.Dataset-like object, dataset containing the stimuli images.
        subj_image_ids: np.ndarray, ordered image IDs for this subject.
    
    Outputs:
        dataloader: DataLoader, loader over all selected subject images in subj_image_ids order.
    """
    return make_nsd_dataloader(
        model_context=model,
        stim_dataset=stim_dataset,
        image_ids=subj_image_ids,
        batch_size=subj_image_ids.shape[0])



def feature_extraction(layer_name: str, model_context: dict, subj_image_ids: np.ndarray,
                       stim_dataset, batch_size_process: int, batch_size_dataloader: int = 128) -> np.ndarray:
    """Extract features from the models and the neural data.
    
    Inputs:
        layer_name: str, name of the layer to extract features from.
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
    


def features_pipeline(model1,model2,subj_id,hdf_path:Path,pkl_info_path:Path,neural_data_path:Path) -> dict:
    """Main function to run the feature extraction pipeline.
    
    Inputs: 
        model1: str, name of the first model
        model2: str, name of the second model
        subj_id: str, subject ID
        hdf_path: Path, path to the hdf file containing neural data
        pkl_info_path: Path, path to the pkl file containing info about the neural data
        neural_data_path: Path, path to the directory containing neural data
        
    Outputs: 
        context = {
            'sources_context': sources_context,
            'target_context': target_context
        }
        
        source_context example: {
            'X1_context': {
                'model_name': model_name,
                'model': model,
                'image_transforms': image_transforms,
                'layers_ordered': layers_ordered,
                'dataloader': dataloader,
                'features': features
            
        target_context example: {
            'neural_data': neural_data,
            'stim': stim, ------->     Images presented to the subject during the experiment
            'hdf_file': hdf_file,
            'image_ids_for_subj': image_ids_for_subj,
            'shared1000_subj': shared1000_subj
        }"""
    

    sources_context = prepare_sources(model_1=model1, model_2=model2)
    
    target_context = prepare_target(hdf_path=hdf_path,
                                    pkl_info_path=pkl_info_path,
                                    neural_data_path=neural_data_path)

    
    #NSD batch
    subj_imgs = target_context["image_ids_for_subj"]
    stim = target_context["stim"]

    # --- Extract features for a specific layer ---
    layer_name = "layer3"  # specify the layer to extract features from
    
    features_X1 = feature_extraction(layer_name=layer_name, model_context=sources_context['X1_context'],
                                     subj_image_ids=subj_imgs, stim_dataset=stim, batch_size_process=128)
    
    features_X2 = feature_extraction(layer_name=layer_name, model_context=sources_context['X2_context'],
                                     subj_image_ids=subj_imgs, stim_dataset=stim, batch_size_process=128)
    

    sources_context['X1_context']['features'] = features_X1
    sources_context['X2_context']['features'] = features_X2

    return {'sources_context': sources_context, 'target_context': target_context}

