import torch
import numpy as np
import sys
import os 
from pathlib import Path
import joblib
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline.pipeline_phases.sources_target_features import prepare_target,prepare_sources
from sklearn.linear_model import Ridge



"""This file will contain the functions to create the PID pipeline
after loading PCA components for the the target. Scaling and Ridge predictions from X1 X2 -> PCA(T) and then PID pipeline  
function that are already implemented will be imported from pipieline phases folder.
"""



def load_target(target_kwargs):
    """
    Load the target data from the given path, PCA matrix and scaler and PCA data.
    
    Args:
        target_kwargs (dict): A dictionary containing the keyword arguments for the target data.

    Returns:
        dict: A dictionary containing the loaded target PCA and scaler.
    """


    target_pca_path = target_kwargs['pca_path']
    target_scaler_path = target_kwargs['scaler_path']

    # Load the target PCA and scaler
    target_pca = joblib.load(target_pca_path)
    target_scaler = joblib.load(target_scaler_path)

    hdf_path = target_kwargs['hdf_path']
    pkl_info_path = target_kwargs['pkl_info_path']
    neural_data_path = target_kwargs['neural_data_path']

    # Boolean flag to determine if only shared images data or all images
    only_shared = target_kwargs['only_shared']

    target_context = prepare_target(hdf_path, pkl_info_path, neural_data_path)



    #Load scaler:
    scaler = joblib.load(target_scaler_path)

    #Load PCA model: 
    pca = joblib.load(target_pca_path)

    neural_data = target_context["neural_data"]
    # Scale the neural data
    scaled_neural_data = scaler.transform(neural_data)

    #PCA neural data
    pca_neural_data = pca.transform(scaled_neural_data)

    target_context['pca_target'] = pca_neural_data

    
    return target_context


def load_source_context(source_kwargs):

    source1_name = source_kwargs['source1_name']
    source2_name = source_kwargs['source2_name']

    sources_context = prepare_sources(source1_name, source2_name)

    X1_context = sources_context['X1_context']
    X2_context = sources_context['X2_context']


    scaler_X1 = joblib.load(source_kwargs['source1_scaler_path'])
    scaler_X2 = joblib.load(source_kwargs['source2_scaler_path'])


    pass



#Prepropessing function: Scaling the data
def scale_func(source1,source2,target,preprocess_kwargs):
    """
    Scale the data using the provided scaler.
    
    Args:
        data_kwargs (dict): A dictionary containing the data and scaler.

    Returns:
        np.ndarray: The scaled data.
    """
    source1_scaler_path = preprocess_kwargs['source1_scaler_path']
    source2_scaler_path = preprocess_kwargs['source2_scaler_path']
    target_scaler_path = preprocess_kwargs['target_scaler_path']

    scaler_source1 = joblib.load(source1_scaler_path)
    scaler_source2 = joblib.load(source2_scaler_path)
    scaler_target = joblib.load(target_scaler_path)

    # Scale the data
    scaled_source1 = scaler_source1.transform(source1)
    scaled_source2 = scaler_source2.transform(source2)
    scaled_target = scaler_target.transform(target)

    return scaled_source1, scaled_source2, scaled_target


#Feature manipulation funtion: PCA target and ridge prediction from X1 and X2 to PCA(T) using the alphas per pc from the data
def feature_manipulation_func(source1,source2,target,feature_manipulation_kwargs):
    """
    Perform feature manipulation on the source and target data.
    
    Args:
        source1 (np.ndarray): The first source data.
        source2 (np.ndarray): The second source data.
        target (np.ndarray): The target data.
        feature_manipulation_kwargs (dict): A dictionary containing the keyword arguments for feature manipulation.
        """
    
    seed = feature_manipulation_kwargs['seed']
    pca_target_path = feature_manipulation_kwargs['pca_target_path']
    alpahs_source1_path = feature_manipulation_kwargs['alphas_source1_path']
    alphas_source2_path = feature_manipulation_kwargs['alphas_source2_path']

    pca = joblib.load(pca_target_path)
    alphas_source1 = joblib.load(alpahs_source1_path)
    alphas_source2 = joblib.load(alphas_source2_path)

    #PCA target data
    pca_target = pca.transform(target)

    shared_ids = feature_manipulation_kwargs['shared1000_subj']


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





       

    #Ridge prediction from X1 and X2 to PCA(T) using the alphas per pc from the data

def load_sources_target(target_kwargs,source1_kwargs,source2_kwargs):
    """
    Load the source and target data from the given paths.
    
    Args:
        target_kwargs (dict): A dictionary containing the keyword arguments for the target data.
        source1_kwargs (dict): A dictionary containing the keyword arguments for the first source data.
        source2_kwargs (dict): A dictionary containing the keyword arguments for the second source data."""
    
    target_data = 
    target_pca_path = target_kwargs['pca_path']
    source1_pca_path = source1_kwargs['pca_path']
    source2_pca_path = source2_kwargs['pca_path']

    # Load the target data
    target_pca = joblib.load(target_pca_path)
    
    # Load the source data
    source1_pca = joblib.load(source1_pca_path)
    source2_pca = joblib.load(source2_pca_path)

    target_scaler_path = target_kwargs['scaler_path']
    source1_scaler_path = source1_kwargs['scaler_path']
    source2_scaler_path = source2_kwargs['scaler_path']

    # Load the scalers
    target_scaler = joblib.load(target_scaler_path)
    source1_scaler = joblib.load(source1_scaler_path)
    source2_scaler = joblib.load(source2_scaler_path)

    # Transform the data using the loaded scalers
    target_data = target_scaler.transform(target_data)
    source1_data = source1_scaler.transform(source1_data)
    source2_data = source2_scaler.transform(source2_data)


    return {'targetPCA': target_pca, 'source1PCA': source1_pca, 'source2PCA': source2_pca,
            'targetScaler': target_scaler, 'source1Scaler': source1_scaler, 'source2Scaler': source2_scaler,}



def ridge_source(source1,selected_layer,target_context):
    """
    Perform Ridge regression on the source data to predict the target data.
    
    Args:
        source1 (np.ndarray): The first source data.
        selected_layer (np.ndarray): The selected layer from the source data.
        target_context (dict): A dictionary containing the target data and other relevant information.
            target_context must contain the alpha per pc for the source1
    
    Returns:
        dict: A dictionary containing the Ridge regression results, including predictions and coefficients.
    """
    