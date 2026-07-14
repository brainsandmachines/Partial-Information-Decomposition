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




#Prepropessing function: Scaling the data
def scale_func(source1,source2,target,source1_scaler_path,source2_scaler_path,target_scaler_path):
    """
    Scale the data using the provided scaler.
    
    Args:
        source1 (np.ndarray): The first source data.
        source2 (np.ndarray): The second source data.
        target (np.ndarray): The target data.
        source1_scaler_path (str): The path to the scaler for the first source.
        source2_scaler_path (str): The path to the scaler for the second source.
        target_scaler_path (str): The path to the scaler for the target.

    Returns:
        np.ndarray: The scaled data.
    """

    scaler_source1 = joblib.load(source1_scaler_path)
    scaler_source2 = joblib.load(source2_scaler_path)
    scaler_target = joblib.load(target_scaler_path)

    # Scale the data
    scaled_source1 = scaler_source1.transform(source1)
    scaled_source2 = scaler_source2.transform(source2)
    scaled_target = scaler_target.transform(target)

    return scaled_source1, scaled_source2, scaled_target



       
#Feature manipulation funtion: PCA target and ridge prediction from X1 and X2 to PCA(T) using the alphas per pc from the data
def feature_manipulation_func(source1,source2,target,seed,source1_name,source2_name,pc_target_path,alphas_source1_path,alphas_source2_path,shared1000_subj):
    """
    Perform feature manipulation on the source and target data.
    
    Args:
        source1 (np.ndarray): The first source data.
        source2 (np.ndarray): The second source data.
        target (np.ndarray): The target data.
        seed (int): The random seed for reproducibility.
        source1_name (str): The name of the first source data.
        source2_name (str): The name of the second source data.
        pc_target_path (str): The path to the PCA target data.
        alphas_source1_path (str): The path to the alphas for the first source.
        alphas_source2_path (str): The path to the alphas for the second source.
        shared1000_subj (np.ndarray): An array of shared subject IDs.
        """
    

    pca = joblib.load(pc_target_path)
    
    with np.load(alphas_source1_path, allow_pickle=False) as archive:
        alphas_source1 = np.asarray(
            archive["alphas"],
            dtype=np.float64,
        ).copy()

        if source1_name != archive["model_name"]:
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

        if source2_name != archive["model_name"]:
            raise ValueError("Model name mismatch for source2.")

    #PCA target data
    pca_target = pca.transform(target)

    shared_ids = shared1000_subj


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