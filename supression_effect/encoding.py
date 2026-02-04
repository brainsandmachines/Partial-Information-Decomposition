import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Tuple
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root)) 

from encoding_model.suppresion_model import train_save_or_load   
from encoding_model.suppression_core import *
from Partial_Information_Decomposition.Idep_multivariate_gauss import Idep_multivariate_gauss


method = 'ridge_cv'  #Method for variance partitioning
n_s = 200  #Number of samples
n_f = 100    #Number of features to use in the encoder
rng_seed = np.random.default_rng(seed=42)  #Random number generator seed
snr = 20  #Signal to noise ratio
mixing_dimension = 70  #Mixing dimension for suppression model
suppression_strength = 0.2  #Suppression strength
suppression_method = 'permutate'
path_to_load = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/trained_models/RidgeCV_subj1_model_alexnet_features.2/RidgeCV_subj1_model_alexnet_features.2_encoding_model.joblib'

loaded_model = train_save_or_load(path_to_load=path_to_load)
real_reg_lh, real_reg_rh, real_features =loaded_model['reg_lh'], loaded_model['reg_rh'], loaded_model['features_train'] 
fmri_dict = joblib.load('/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/fmri_dicts/subj1_fmri_dicts.joblib')

run_name = "test_run_RidgeCV_Encoder"
save_dir = Path('/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/test_runs')
features = real_features
suppression_method = 'permutate'
n_features = n_f

lh_fmri_train = fmri_dict['lh_fmri_train'][:n_s,:]
real_features = features[:n_s,:]
encoder,selected_features = create_encoder(rng=None, features=real_features, target=lh_fmri_train, n_features=n_f)

print("\nEncoder's features shape: ", selected_features.shape)
print("\nCreating predictions from encoder...")

y_hat_lh, y_hat_rh = create_predictions(encoder,reg_rh=None, features=selected_features) #From model1 
print("Predictions created.\nPredicted fMRI shape (LH): ", y_hat_lh.shape) if y_hat_lh is not None else None
print("\nPredicted fMRI shape (RH): ", y_hat_rh.shape) if y_hat_rh is not None else None

print("Creating suppression model...")
X_M1, X_M2,target = create_supression_model(rng=rng_seed,signal = y_hat_lh,suppresion_method=suppression_method,features=selected_features,suppression_strength=suppression_strength,mixing_dimension=mixing_dimension,snr=snr)


#outputs = commonality_analysis(X_M1, X_M2, target, method=method)

M1 = torch.tensor(X_M1,dtype=torch.float64)
M2 = torch.tensor(X_M2,dtype=torch.float64)
T = torch.tensor(target,dtype=torch.float64)

sources = [M1,M2]
targets = [T]
idep_class = Idep_multivariate_gauss(sources,targets)
pid = idep_class.idep()
print("\nIdep PID results:")
for key, value in pid.items():
    print(f"- {key}: {value:.4f}")