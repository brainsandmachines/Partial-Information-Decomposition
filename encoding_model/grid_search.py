import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from pathlib import Path
from tqdm import tqdm
import sys
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from encoding_model.algoanut_data import argObj, load_data_algonauts
from toy_example_new import run_experiment
from utils import check_file_exists, create_permuation
from typing import Optional
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from encoding_model.fmri_model import encoding_model
from pred_pipeline import pipeline
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from encoding_model.encoding_utils import split_dataset, visualize_encdoing_accuaracy,fmri_data_loader,save_model
from scipy.stats import pearsonr as corr
from suppresion_model import create_supression_model,commonality_analysis











data_dir  = '/mnt/data4tb/data_algonauts/'
parent_submission_dir = '/mnt/data4tb/data_algonauts/submissions'
fmri_fig_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/fmri_figs'
correlation_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/correlations_fig'
subj = 1
args = argObj(data_dir, parent_submission_dir, subj)
model = 'alexnet'
layer_name = 'features.2'
model_path='pytorch/vision:v0.10.0'
batch_size=500
ncomponents=200


grid_search = False
if grid_search:
    def grid_search_ols(method,suppresions_strengths_list,snr_list,mixing_dimensions_list,signal):
        results = []
        for _,suppression_strength in tqdm(enumerate(suppresions_strengths_list), total=len(suppresions_strengths_list)):
            for snr in snr_list:
                for mixing_dimension in mixing_dimensions_list:
                    X_M1, X_M2,target = create_supression_model(rng=np.random.default_rng(0),signal=signal, features=features, suppression_strength=suppression_strength,snr=snr,mixing_dimension=mixing_dimension)
                    results_dict = commonality_analysis(X_M1, X_M2, target, method=method,snr=snr)
                    results_dict.update({
                        'suppression_strength': suppression_strength,
                        'snr': snr,
                        'mixing_dimension': mixing_dimension
                    })
                    results.append(results_dict)
        df = pd.DataFrame(results)
        data_frame_sorted = df = df.sort_values(by="common", ascending=True)
        return data_frame_sorted

    suppresions_strengths_list = [0.5]
    snr_list = [1.0, 5.0, 10.0, 20.0,50.0]
    mixing_dimensions_list = [None,20, 50, 70,150,200,300,120,180,190,250]
    df_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/data_frames'
    exp_name = 'grid_seach_ols_commonality_analysis_real_model_' + model + '_' + layer_name + '_subj' + str(subj) + 'ncomponents' + str(ncomponents)+'.csv'
    grid_search_result = grid_search_ols(method='standard',suppresions_strengths_list=suppresions_strengths_list,snr_list=snr_list,mixing_dimensions_list=mixing_dimensions_list,signal=y_hat_lh)
    print("\nGrid Search Commonality Analysis finished:")
    path = f"{df_path}/{exp_name}"
    path = check_file_exists(path)
    grid_search_result.to_csv(path)
    print(f"Results saved to {path}")
    print(grid_search_result)