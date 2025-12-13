import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from pathlib import Path
from tqdm import tqdm
import sys
from algoanut_data import argObj, load_data_algonauts
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from utils import check_file_exists, create_permuation
from typing import Optional
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from fmri_model import encoding_model
from pred_pipeline import pipeline
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from encoding_utils import split_dataset, visualize_encdoing_accuaracy,fmri_data_loader,save_model
from scipy.stats import pearsonr as corr

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

def real_model_func(model, layer_name,model_path,batch_size,ncomponents,train_data_loader,val_imgs_dataloader):
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
        subj: Subject number."""
    model = encoding_model(device=device,model=model,model_layer=layer_name,model_path=model_path)


    pca = model.fit_pca(train_data_loader,batch_size=batch_size,ncomponents=ncomponents)

    real_feature = model.extract_features(train_data_loader,pca)

    real_reg_lh,real_reg_rh = model.train(train_data_loader,fmri_dict['lh_fmri_train'],fmri_dict['rh_fmri_train'],features_train=real_feature)
    return model, real_reg_lh, real_reg_rh, pca, real_feature


def create_predictions(reg_lh, reg_rh, features):
    """
    Create fMRI predictions using trained regression models.
    
    Args:
        reg_lh: Trained regression model for left hemisphere.
        reg_rh: Trained regression model for right hemisphere.
        features: Feature matrix (shape: [n_samples, n_features]).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Predicted fMRI responses for left and right hemispheres.
    """
    y_hat_lh = reg_lh.predict(features)
    y_hat_rh = reg_rh.predict(features)
    return y_hat_lh, y_hat_rh

def create_supression_model(rng,signal, features, suppression_strength=0.5,snr=1.0,mixing_dimension=None):

    n,p = features.shape
    
    n_real_dim = 1-suppression_strength
    real_dim = int(p*n_real_dim)

    std = np.std(signal)
    noise_std = std.item() / snr
    signal_dim1 , signal_dim2 = signal.shape[0], signal.shape[1]
    target = signal +  noise_std * rng.standard_normal((signal_dim1 , signal_dim2))

    real_feature = features[:,:real_dim]
    spurious_feature = features[:,real_dim:]

    rand_perm = rng.permutation(n)

    shuffled_real = real_feature[rand_perm]
    shuffled_spurious = spurious_feature[rand_perm]

    X_M1 = np.hstack([real_feature, shuffled_spurious])
    X_M2 = np.hstack([shuffled_real, shuffled_spurious])

        # Remove any linear predictability of X_M2 from real_features
    ortho_model = LinearRegression(fit_intercept=False)
    ortho_model.fit(features, X_M2)
    X_M2_pred = ortho_model.predict(features)
    X_M2 = X_M2 - X_M2_pred

    if mixing_dimension is not None:
        # Create mixed features: entangle real and spurious with a mixing matrix
        mixing_matrix_M1 = rng.standard_normal((X_M1.shape[1], mixing_dimension))
        X_M1 = X_M1 @ mixing_matrix_M1
        mixing_matrix_M2 = rng.standard_normal((X_M2.shape[1], mixing_dimension))
        X_M2 = X_M2 @ mixing_matrix_M2

    return X_M1, X_M2,target

def compute_ols_cv_r2(X, y):
    """
    Compute cross-validated R² using leave-one-out cross-validation.
    
    Uses RidgeCV with near-zero regularization (alpha=1e-16) which is
    effectively OLS but leverages the efficient GCV formula.
    
    Args:
        X (np.ndarray): Design matrix WITHOUT intercept (shape: [n, p]).
        y (np.ndarray): Target variable (shape: [n,]).
        
    Returns:
        float: Cross-validated R² (can be negative if model overfits badly)
    """
    ridge_cv = RidgeCV(alphas=[1e-16], fit_intercept=True, scoring='r2', cv=None)
    ridge_cv.fit(X, y)
    return ridge_cv.best_score_


def compute_ridge_cv_r2(X, y, alphas=None):
    """
    Compute cross-validated R² using RidgeCV with efficient LOO cross-validation.
    
    RidgeCV uses generalized cross-validation (GCV) which is an efficient 
    approximation to leave-one-out CV for ridge regression.
    
    Args:
        X (np.ndarray): Design matrix WITHOUT intercept (shape: [n, p]).
        y (np.ndarray): Target variable (shape: [n,]).
        alphas (array-like, optional): Array of alpha values to try.
            Defaults to DEFAULT_RIDGE_ALPHAS.
        
    Returns:
        float: Best cross-validated R² across all alpha values.
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 50)
    
    # RidgeCV with leave-one-out CV (efficient GCV approximation)
    # cv=None means use efficient LOO via GCV
    ridge_cv = RidgeCV(alphas=alphas, fit_intercept=True, scoring='r2', cv=None)
    ridge_cv.fit(X, y)
    
    return ridge_cv.best_score_


def compute_r2(X, y):
    """
    Compute in-sample R² for OLS regression.
    
    Args:
        X (np.ndarray): Design matrix WITHOUT intercept (shape: [n, p]).
        y (np.ndarray): Target variable (shape: [n,]).
        
    Returns:
        float: In-sample R².
    """
    model = LinearRegression()
    model.fit(X, y)
    return model.score(X, y)

def commonality_analysis(features_A, features_B, target, method='standard', alphas=None, snr=1.0):
    # Select R² computation function based on method
    if method == 'standard':
        compute_r2_fn = compute_r2
    elif method == 'ols_cv':
        compute_r2_fn = compute_ols_cv_r2
    elif method == 'ridge_cv':
        compute_r2_fn = lambda X, y: compute_ridge_cv_r2(X, y, alphas)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'standard', 'ols_cv', or 'ridge_cv'.")
    
    #Define joint model features
    features_AB = np.hstack([features_A, features_B])
    # Compute R² for each model
    r2_A = compute_r2_fn(features_A, target)
    r2_B = compute_r2_fn(features_B, target)
    r2_AB = compute_r2_fn(features_AB, target)
    
    # Commonality analysis decomposition
    unique_A = (r2_AB - r2_B)
    unique_B = (r2_AB - r2_A)
    common_AB = (r2_A + r2_B - r2_AB)
    unexplained = (1 - r2_AB)
    
    return {
        'R²_A': r2_A,
        'R²_B': r2_B,
        'R²_AB': r2_AB,
        'unique_A': unique_A,
        'unique_B': unique_B,
        'common': common_AB,
        'unexplained': unexplained
    }

def train_save_or_load(folder_path, model_name,path_to_load=None):
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
        path_to_load = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/trained_models/real_model_alexnet_features.2_subj1/real_model_alexnet_features.2_subj1_encoding_model.joblib'
        assert check_file_exists(path_to_load), f"File {path_to_load} does not exist."
        trained_real_model = joblib.load(path_to_load)
        print(f"\n Trained model loaded from: {path_to_load}")

    return trained_real_model
if __name__ == "__main__":
    folder_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/trained_models'
    model_name = 'real_model_' + model + '_' + layer_name + '_subj' + str(subj) + 'ncomponents' + str(ncomponents)
    path_to_load = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/trained_models/real_model_alexnet_features.2_subj1ncomponents200'
    #path_to_load = None

    loaded_model = train_save_or_load(folder_path, model_name,path_to_load=path_to_load)
    real_model, real_reg_lh, real_reg_rh, pca, real_feature = loaded_model['real_model'], loaded_model['reg_lh'], loaded_model['reg_rh'], loaded_model['pca'], loaded_model['real_feature']
    #features_val = real_model.extract_features(val_imgs_dataloader, pca)
    #lh_correlation,rh_correlation = real_model.validate(features_val, real_reg_lh, real_reg_rh)
    n_preds = 500
    features = real_feature
    y_hat_lh, y_hat_rh = create_predictions(real_reg_lh, real_reg_rh, real_feature)
    X_M1, X_M2,target = create_supression_model(rng=np.random.default_rng(0),signal = y_hat_lh,features=real_feature,suppression_strength=0.5,mixing_dimension=None,snr=10.0)

    lh_results_dict = commonality_analysis(X_M1, X_M2, target, method='standard')
    df = pd.DataFrame.from_dict(lh_results_dict, orient='index', columns=['value'])
    print("\nLH Commonality Analysis Results:")
    print(df)

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