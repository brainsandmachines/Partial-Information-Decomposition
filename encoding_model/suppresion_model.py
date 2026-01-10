import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import json
from pathlib import Path
from tqdm import tqdm
import sys
from algoanut_data import argObj, load_data_algonauts
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))  
from toy_example_new import run_experiment
from utils import check_file_exists, create_permuation,Tee,meta_exists
from typing import Optional
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from encoding_model.fmri_model import encoding_model
from pred_pipeline import pipeline
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from encoding_utils import split_dataset, visualize_encdoing_accuaracy,fmri_data_loader,save_model,compute_r2,compute_ols_cv_r2,compute_ridge_cv_r2
from scipy.stats import pearsonr as corr
from encoding_model.suppression_core import *

log = open("run.log", "w")

sys.stdout = Tee(sys.stdout, log)
sys.stderr = Tee(sys.stderr, log)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

output_dict, data_dict = load_data_algonauts(paths_dict={'data_dir': data_dir, 'parent_submission_dir': parent_submission_dir}, args=args, subj=subj)

train_img_list = output_dict['train_img_list']
test_img_list = output_dict['test_img_list']
lh_fmri = output_dict['lh_fmri']
rh_fmri = output_dict['rh_fmri']

train_img_dir = data_dict['train_img_dir']
test_img_dir = data_dict['test_img_dir']
    

fmri_dict, train_data_loader,val_imgs_dataloader = fmri_data_loader(lh_fmri,rh_fmri,train_img_list,test_img_list,train_img_dir,test_img_dir,batch_size=500,train_p=100)

def real_model_func(model, layer_name,model_path,batch_size,ncomponents,train_data_loader):
    """Create and train the real encoding model using fMRI data and image features.
    
    Args:
        model: Pretrained model name (e.g., 'alexnet').
        layer_name: Specific layer name in the model to extract features from.
        model_path: Path or identifier for the pretrained model.
        batch_size: Batch size for data loading.
        ncomponents: Number of PCA components.
        args: Argument object containing data directories.
        data_dir: Base data directory.
        parent_submission_dir: Parent submission directory.
        subj: Subject number.
    Returns:
    model - The Instance from encoingding_model class after training.
    real_reg_lh - Trained regression model for left hemisphere.
    real_reg_rh - Trained regression model for right hemisphere.
    pca - Fitted PCA object.
    real_feature - Extracted features from training data."""
    model = encoding_model(device=device,model=model,model_layer=layer_name,model_path=model_path)


    pca = model.fit_pca(train_data_loader,batch_size=batch_size,ncomponents=ncomponents)

    real_feature = model.extract_features(train_data_loader,pca)

    real_reg_lh,real_reg_rh = model.train(train_data_loader,fmri_dict['lh_fmri_train'],fmri_dict['rh_fmri_train'],features_train=real_feature)
    return model, real_reg_lh, real_reg_rh, pca, real_feature

def train_save_or_load(folder_path=None, model_name=None,path_to_load=None):
    """Load a trained encoding model from disk."""
    if path_to_load is None:
        trained_real_model = real_model_func(model, layer_name,model_path,batch_size,ncomponents,train_data_loader,val_imgs_dataloader)
        real_model, real_reg_lh, real_reg_rh, pca, real_feature = trained_real_model
        #Save trained model:
        save_dict= {'real_model': real_model, 'reg_lh': real_reg_lh, 'reg_rh': real_reg_rh, 'pca': pca, 'real_feature': real_feature}
        save_model(folder_path, model_name, save_dict)
        print(f"\n Trained model saved to folder: {folder_path}/{model_name}")
    else:
    #Load trained model:
        assert check_file_exists(path_to_load), f"File {path_to_load} does not exist."
        trained_real_model = joblib.load(path_to_load)
        print(f"\n Trained model loaded from: {path_to_load}")

    return trained_real_model



def main(dict,suppression_strength=0.5,rng_seed=0):
    """Run the 2x3 factorial experiment design."""
    # Common parameters
    rng_seed = np.random.default_rng(0)
    suppression_strength = 0.7
    
    # =============================================================================
    # LOW SNR experiments (SNR = 1.0)
    # =============================================================================

    print("\n" + "="*70)
    print("Experiment 1: LOW SNR + NO MIXING")
    print("="*70)
    run_all_methods(rng_seed,suppresion_method='permutate' ,suppression_strength=suppression_strength, mixing_dimension=None, snr=1,models_and_features_dict=dict)

    print("\n" + "="*70)
    print("Experiment 2: LOW SNR + INVERTIBLE MIXING (30)")
    print("="*70)
    run_all_methods(rng_seed, suppresion_method='permutate' ,suppression_strength=suppression_strength, mixing_dimension=30, snr=1,models_and_features_dict=dict)

    print("\n" + "="*70)
    print("Experiment 3: LOW SNR + LOSSY MIXING (50)")
    print("="*70)
    run_all_methods(rng_seed, suppresion_method='permutate' ,suppression_strength=suppression_strength, mixing_dimension=50, snr=1,models_and_features_dict=dict)

    # =============================================================================
    # HIGH SNR experiments (SNR = 10.0)
    # =============================================================================

    print("\n" + "="*70)
    print("Experiment 4: HIGH SNR + NO MIXING")
    print("="*70)
    run_all_methods(rng_seed, suppresion_method='permutate' ,suppression_strength=suppression_strength, mixing_dimension=None, snr=10.0,models_and_features_dict=dict)

    print("\n" + "="*70)
    print("Experiment 5: HIGH SNR + INVERTIBLE MIXING (70)")
    print("="*70)
    run_all_methods(rng_seed, suppresion_method='permutate' ,suppression_strength=suppression_strength, mixing_dimension=30, snr=10.0,models_and_features_dict=dict)

    print("\n" + "="*70)
    print("Experiment 6: HIGH SNR + LOSSY MIXING (50)")
    print("="*70)
    run_all_methods(rng_seed, suppresion_method='permutate' ,suppression_strength=suppression_strength, mixing_dimension=50, snr=10.0,models_and_features_dict=dict)

    # =============================================================================
    # HIGHER SNR experiments (SNR = 50.0)
    # =============================================================================
    
    print("\n" + "="*70)
    print("Experiment 7: HIGHER SNR + NO MIXING")
    print("="*70)
    run_all_methods(rng_seed, suppresion_method='permutate' ,suppression_strength=suppression_strength, mixing_dimension=None, snr=50.0,models_and_features_dict=dict)

    print("\n" + "="*70)
    print("Experiment 8: HIGHER SNR + INVERTIBLE MIXING (30)")
    print("="*70)
    run_all_methods(rng_seed, suppresion_method='permutate' ,suppression_strength=suppression_strength, mixing_dimension=30, snr=50.0,models_and_features_dict=dict)


    print("\n" + "="*70)
    print("Experiment 9: HIGHER SNR + LOSSY MIXING (50)")
    print("="*70)
    run_all_methods(rng_seed, suppresion_method='permutate' ,suppression_strength=suppression_strength, mixing_dimension=50, snr=50.0,models_and_features_dict=dict)

def test_run(run_name,save_dir,features,fmri_dict,rng_seeds,suppression_method,suppression_strength=[0.5],n_samples=[1000],n_features=[100],snr=[1.0],mixing_dimension=[None]):
    print("\n" + "="*70)
    print('\nStarting test run...')
    csv_path = save_dir / f"{run_name}.csv"
    pkl_path = save_dir / f"{run_name}.pkl"
    records = []
    print(f"\nResults will be saved to: {csv_path} and {pkl_path}")
    print(f'\n Starting test run with features shape: {features.shape} and fmri shape (LH): {fmri_dict["lh_fmri_train"].shape}')
    for num in rng_seeds:
        rng_seed = np.random.default_rng(num)
        for n_s in n_samples:
            for n_f in n_features:
                for s in suppression_strength:
                    for sn in snr:
                        for md in mixing_dimension:
                            # ----- meta_data (hyperparameters) -----
                            meta_data = {
                                "rng_seed": num,
                                "suppression_method": suppression_method,
                                "n_samples": n_s,
                                "n_features": n_f,
                                "suppression_strength": s,
                                "snr": sn,
                                "mixing_dimension": md
                            }

                            if meta_exists(meta_data, csv_path):
                                print(f"\nSkipping already completed test with parameters: {meta_data}")
                                continue

                            print("\n" + "="*70)
                            print(f"\nTest Run: n_samples={n_s}, n_features={n_f}, suppression_strength={s}, snr={sn}, mixing_dimension={md}")

                            lh_fmri_train = fmri_dict['lh_fmri_train'][:n_s,:]
                            real_features = features[:n_s,:]
                            encoder,selected_features = create_encoder(rng_seed, real_features,lh_fmri_train,n_features=n_f)

                            print("\nEncoder's features shape: ", selected_features.shape)
                            print("\nCreating predictions from encoder...")
                            y_hat_lh, y_hat_rh = create_predictions(encoder,reg_rh=None, features=selected_features) #From model1 
                            print("Predictions created.\nPredicted fMRI shape (LH): ", y_hat_lh.shape) if y_hat_lh is not None else None
                            print("\nPredicted fMRI shape (RH): ", y_hat_rh.shape) if y_hat_rh is not None else None
                            

                            models_and_features_dict = {'X_M1': None, 'X_M2': None, 'target': None,'signal': y_hat_lh,'real_feature': selected_features}
                            outputs = run_all_methods(rng_seed,suppresion_method=suppression_method ,mixing_dimension=md, snr=sn, suppression_strength=s,models_and_features_dict=models_and_features_dict)
                            
                            record =  {**meta_data, **outputs}

                            df_new = pd.DataFrame([record])

                            if csv_path.exists():
                                # append without overwriting
                                df_new.to_csv(csv_path, mode="a", header=False, index=False)
                            else:
                                # first time: create file with header
                                df_new.to_csv(csv_path, index=False)

                            # keep pickle in sync (overwrite is fine)
                            if pkl_path.exists():
                                df_old = pd.read_pickle(pkl_path)
                                df_all = pd.concat([df_old, df_new], ignore_index=True)
                            else:
                                df_all = df_new

                            df_all.to_pickle(pkl_path)

    return 
    
if __name__ == "__main__":
    #path_to_load = None
    # n_stimuli = 1000
    # n_features = 100
    # rng_seed = np.random.default_rng(0)
    # folder_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/trained_models'
    # model_name = 'real_model_' + model + '_' + layer_name + '_subj' + str(subj) + 'ncomponents' + str(ncomponents)
    path_to_load = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/trained_models/RidgeCV_subj1_model_alexnet_features.2/RidgeCV_subj1_model_alexnet_features.2_encoding_model.joblib'

    loaded_model = train_save_or_load(path_to_load=path_to_load)
    real_reg_lh, real_reg_rh, real_features =loaded_model['reg_lh'], loaded_model['reg_rh'], loaded_model['features_train'] 
    fmri_dict = joblib.load('/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/fmri_dicts/subj1_fmri_dicts.joblib')
    
    run_name = "test_run_RidgeCV_Encoder"
    save_dir = Path('/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/test_runs')
    features = real_features
    suppression_method = 'permutate'
    rngs_seed = [2,32,10,6]
    supression_strength = [0.3,0.5,0.8]
    n_samples = [1000,2000,3000]
    n_features = [100,300,500] #WARNING: n_features should be less than n_samples
    snr = [20.0,50.0,80.0]
    mixing_dimension = [None,30,50,70,100]
 
    df = test_run(run_name, save_dir, features, fmri_dict, rngs_seed, suppression_method, supression_strength, n_samples, n_features, snr, mixing_dimension)
    print("\nTest run completed. Results saved.")
    print(df)
    print("\n" + "="*70)

    # features_train = real_features[:n_stimuli,:] #In case I want to less data for faster testing: i.e: features_train[:1000,:]
    # lh_fmri_train = fmri_dict['lh_fmri_train'][:n_stimuli,:] #In case I want to less data for faster testing: i.e: fmri_dict['lh_fmri_test'][:1000,:]
    # rh_fmri_train = fmri_dict['rh_fmri_train'][:n_stimuli,:] #In case I want to less data for faster testing: i.e:

    # #Create a model with less features: 
    # encoder,selected_features = create_encoder(rng_seed, features_train,lh_fmri_train,n_features=n_features)
    
    # print("Encoder 1 features shape: ", features_train.shape)
    # print("\nCreating predictions from encoder...")
    # y_hat_lh, y_hat_rh = create_predictions(real_reg_lh,reg_rh=None, features=features_train) #From model1 
    # print("Predictions created.\nPredicted fMRI shape (LH): ", y_hat_lh.shape) if y_hat_lh is not None else None
    # print("\nPredicted fMRI shape (RH): ", y_hat_rh.shape) if y_hat_rh is not None else None
    

    # models_and_features_dict = {'X_M1': None, 'X_M2': None, 'target': None,'signal': y_hat_lh,'real_feature': features_train}
    # main(models_and_features_dict)


    # lh_results_dict = commonality_analysis(X_M1, X_M2, target, method='standard')
    # df = pd.DataFrame.from_dict(lh_results_dict, orient='index', columns=['value'])
    # print("\nLH Commonality Analysis Results:")
    # print(df)
