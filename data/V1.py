import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import zarr

import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from encoding_model.algoanut_data import load_data_algonauts, argObj
from encoding_model.encoding_utils import map_correlation_to_rois, roi_fmri_data,get_specific_roi_fmri, split_dataset,fmri_data_loader,save_model
from encoding_model.pred_pipeline import pipeline
from encoding_model.fmri_model import encoding_model
from encoding_model.suppresion_model import train_save_or_load
from encoding_model.suppression_core import create_encoder, create_predictions, suppression_analysis_pipeline, grid_search_suppression_analysis


def load_roi_data(args,roi_name='V1v'):
    # Load fmri data:
    fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri') #/mnt/data4tb/data_algonauts/subj01/training_split/training_fmri
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))
    print("LH fMRI shape: ", lh_fmri.shape)

    training_img_dir = os.path.join(args.data_dir, 'training_split', 'training_images')
    # Create lists will all training and test image file names, sorted
    train_img_list = os.listdir(training_img_dir)
    train_img_list.sort()
    print('Training images: ' + str(len(train_img_list)))
    lh_roi_fmri, rh_roi_fmri = get_specific_roi_fmri(args,lh_fmri,rh_fmri, roi_name=roi_name)
    return lh_roi_fmri, rh_roi_fmri, train_img_list, training_img_dir




def roi_encoding_model(train_data_loader,val_data_loader,lh_roi_fmri_train, rh_roi_fmri_train,lh_roi_fmri_val, rh_roi_fmri_val,layer_name='features.2',model=None,features=None, batch_size=500,ncomponents=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoding_model_instance = encoding_model(device=device,model_layer=layer_name,model=model,features=features)
    reg_lh, reg_rh, lh_correlation, rh_correlation = encoding_model_instance.run_model(train_data_loader,val_data_loader,lh_roi_fmri_train,
                                                                                         rh_roi_fmri_train,lh_roi_fmri_val,rh_roi_fmri_val,batch_size=500,ncomponents=None)      
    features_val_pred_lh = encoding_model_instance.lh_fmri_val_pred if hasattr(encoding_model_instance, 'lh_fmri_val_pred') else None
    features_val_pred_rh = encoding_model_instance.rh_fmri_val_pred if hasattr(encoding_model_instance, 'rh_fmri_val_pred') else None
    features_train = encoding_model_instance.features_train if hasattr(encoding_model_instance, 'features_train') else None
    features_val_trained = encoding_model_instance.features_val if hasattr(encoding_model_instance, 'features_val') else None
    
    print(f"ROI Encoding model completed for layer: {layer_name}")
    print(f"Left hemisphere correlation: {np.mean(lh_correlation)}")
    print(f"Right hemisphere correlation: {np.mean(rh_correlation)}")

    output_dict = {
        'reg_lh': reg_lh,
        'reg_rh': reg_rh,
        'lh_correlation': lh_correlation,
        'rh_correlation': rh_correlation,
        'features_val_pred_lh': features_val_pred_lh,
        'features_val_pred_rh': features_val_pred_rh,
        'features_train': features_train,
        'features_val_trained': features_val_trained
    }
    return output_dict




def main():
    data_dir  = '/mnt/data4tb/data_algonauts/'
    parent_submission_dir = '/mnt/data4tb/data_algonauts/submissions'
    subj = 1
    args = argObj(data_dir, parent_submission_dir, subj)

    #Create encoding for V1 ROI
    """lh_v1_fmri, rh_v1_fmri, train_img_list, training_img_dir = load_roi_data(args,roi_name='V1v')
    fmri_dict, train_data_loader,val_data_loader = fmri_data_loader(lh_v1_fmri,rh_v1_fmri,train_img_list,test_img_list=None,train_img_dir=training_img_dir,test_img_dir=None,batch_size=500,train_p=80)
    lh_v1_fmri_train = fmri_dict['lh_fmri_train']
    lh_v1_fmri_val = fmri_dict['lh_fmri_val']
    rh_v1_fmri_train = fmri_dict['rh_fmri_train']
    rh_v1_fmri_val = fmri_dict['rh_fmri_val']
    layer_name = 'features.2'
    model = 'alexnet'
    output_dict = roi_encoding_model(train_data_loader,val_data_loader,lh_v1_fmri_train, rh_v1_fmri_train,lh_v1_fmri_val, rh_v1_fmri_val,layer_name=layer_name,model=model,features=None)
    print("ROI encoding model finished.")
    print('Saving models...')

    folder_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/trained_models/roi_models'
    model_name = f'V1v_{model}_{layer_name}_subj{str(args.subj)}.pth'
    save_model(folder_path=folder_path, model_name=model_name, save_dict=output_dict)"""

    #Load encoding model for V1 ROI
    model_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/trained_models/roi_models/V1v_alexnet_features.2_subj01.pth/V1v_alexnet_features.2_subj01.pth_encoding_model.joblib'
    loaded_model_dict = train_save_or_load(path_to_load=model_path)
    print("Loaded model keys: ", loaded_model_dict.keys())

    v1_reg_lh = loaded_model_dict['reg_lh']
    features_train = loaded_model_dict['features_train']

    """results = suppression_analysis_pipeline(
        features=features_train,
        reg_lh=v1_reg_lh,
        reg_rh=None,  
        hemisphere='left',
        suppression_strength=0.7,
        snr=50,
        mixing_dimension=100,  
        analysis_methods=['ridge_cv'],  
        rng_seed=1
    )"""

    grid_search_suppression_analysis(features=features_train,
        reg_lh=v1_reg_lh,
        reg_rh=None,
        suppression_strength_list=[0.3, 0.5, 0.7],
        snr_list=[0.5,1.0, 10.0, 20.0,700],
        mixing_dimension_list=[None, 30, 50,100,250,500,1000],
        rng_seed_list=list(np.arange(1, 100)),
        hemisphere='left',
        suppresion_method='permutate',
        output_dir='./encoding_model/grid_search_results',
        verbose=True    
    )

if __name__ == "__main__":
    main()

