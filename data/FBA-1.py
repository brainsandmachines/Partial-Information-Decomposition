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

from data.V1 import load_roi_data,roi_encoding_model

data_dir  = '/mnt/data4tb/data_algonauts/'
parent_submission_dir = '/mnt/data4tb/data_algonauts/submissions'
subj = 1
args = argObj(data_dir, parent_submission_dir, subj)

#Create encoding for FBA-1 ROI
# lh_fba1_fmri, rh_fba1_fmri, train_img_list, training_img_dir = load_roi_data(args,roi_name='FBA-1')
# fmri_dict, train_data_loader,val_data_loader = fmri_data_loader(lh_fba1_fmri,rh_fba1_fmri,train_img_list,test_img_list=None,train_img_dir=training_img_dir,test_img_dir=None,batch_size=500,train_p=80)
# lh_fba1_fmri_train = fmri_dict['lh_fmri_train']
# lh_fba1_fmri_val = fmri_dict['lh_fmri_val']
# rh_fba1_fmri_train = fmri_dict['rh_fmri_train']
# rh_fba1_fmri_val = fmri_dict['rh_fmri_val']
# layer_name = 'features.8'
# model = 'alexnet'
# output_dict = roi_encoding_model(train_data_loader,val_data_loader,lh_fba1_fmri_train, rh_fba1_fmri_train,lh_fba1_fmri_val, rh_fba1_fmri_val,layer_name=layer_name,model=model,features=None)
# print("ROI encoding model finished.")
# print('Saving models...')

# folder_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/trained_models/roi_models'
# model_name = f'FBA-1_{model}_{layer_name}_subj{str(args.subj)}.pth'
# save_model(folder_path=folder_path, model_name=model_name, save_dict=output_dict)


#Load encoding model for FBA-1 ROI
model_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/trained_models/roi_models/FBA-1_alexnet_features.8_subj01_1.pth/FBA-1_alexnet_features.8_subj01.pth_encoding_model.joblib'
loaded_model_dict = train_save_or_load(path_to_load=model_path)
print("Loaded model keys: ", loaded_model_dict.keys())

fba_reg_lh = loaded_model_dict['reg_lh']
features_train = loaded_model_dict['features_train']

# results = suppression_analysis_pipeline(
#     features=features_train,
#     reg_lh=fba_reg_lh,
#     reg_rh=None,  
#     hemisphere='left',
#     suppression_strength=0.4,
#     snr=0.9,
#     mixing_dimension=100,  
#     analysis_methods=['ridge_cv'],  
#     rng_seed=1
# )


grid_search_suppression_analysis(features=features_train,
    reg_lh=fba_reg_lh,
    reg_rh=None,
    suppression_strength_list=[0.3, 0.5, 0.7],
    snr_list=[0.5,1.0, 10.0, 20.0,100],
    mixing_dimension_list=[None, 30, 50,100,250],
    rng_seed_list=list(np.arange(1, 100)),
    hemisphere='left',
    suppresion_method='permutate',
    output_dir='./encoding_model/grid_search_results',
    grid_name='lh-FBA-1_features.8_alexnet',
    verbose=True    
)
