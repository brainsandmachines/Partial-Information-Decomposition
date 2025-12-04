import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os
from algoanut_data import argObj, ImageDataset, load_data_algonauts
from fmri_model import encoding_model
from encoding_utils import plot_fmri, split_dataset, map_correlation_to_rois, visualize_encdoing_accuaracy,save_corellation,save_model
import torchvision.transforms as transforms
from pathlib import Path
import joblib
from pyparsing import Optional

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir  = '/mnt/data4tb/data_algonauts/'
parent_submission_dir = '/mnt/data4tb/data_algonauts/submissions'
fmri_fig_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/fmri_figs'
correlation_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/correlations_fig'
subj = 1
args = argObj(data_dir, parent_submission_dir, subj)

def pipeline(data_dir, parent_submission_dir, subj,args):
    """Main pipeline to run the encoding model on Algonauts data for a given subject.
    Args:
        data_dir (str): Base data directory.
        parent_submission_dir (str): Parent submission directory.
        subj (int): Subject number.
        args (argObj): Argument object containing data directories.

    Returns: 
        dict: Dictionary containing:
            Corellation values for left hemisphere fMRI data. (lh_correlation)
            Correlation values for right hemisphere fMRI data. (rh_correlation)
            trained left hemisphere regression model. (reg_lh)
            trained right hemisphere regression model. (reg_rh)
    """
    output_dict, data_dict = load_data_algonauts(paths_dict={'data_dir': data_dir, 'parent_submission_dir': parent_submission_dir}, args=args, subj=subj)

    train_img_list = output_dict['train_img_list']
    test_img_list = output_dict['test_img_list']
    lh_fmri = output_dict['lh_fmri']
    rh_fmri = output_dict['rh_fmri']

    train_img_dir = data_dict['train_img_dir']
    test_img_dir = data_dict['test_img_dir']


    idxs_train, idxs_val, idxs_test = split_dataset(train_img_list=train_img_list,test_img_list=test_img_list)

    transform = transforms.Compose([
        transforms.Resize((224,224)), # resize the images to 224x24 pixels
        transforms.ToTensor(), # convert the images to a PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
    ])

    batch_size = 500 #@param
    # Get the paths of all image files
    train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))
    test_imgs_paths = sorted(list(Path(test_img_dir).iterdir()))

    # The DataLoaders contain the ImageDataset class
    train_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_train, transform),
        batch_size=batch_size
    )
    val_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_val, transform),
        batch_size=batch_size
    )
    test_imgs_dataloader = DataLoader(
        ImageDataset(test_imgs_paths, idxs_test, transform),
        batch_size=batch_size
    )

    lh_fmri_train = lh_fmri[idxs_train]
    lh_fmri_val = lh_fmri[idxs_val]
    rh_fmri_train = rh_fmri[idxs_train]
    rh_fmri_val = rh_fmri[idxs_val]

    encoding_model_instance = encoding_model(device=device)
    reg_lh, reg_rh, lh_correlation, rh_correlation = encoding_model_instance.run_model(train_imgs_dataloader,val_imgs_dataloader,lh_fmri_train,rh_fmri_train,lh_fmri_val,rh_fmri_val)

    return  {'reg_lh': reg_lh, 'reg_rh': reg_rh, 'lh_correlation': lh_correlation, 'rh_correlation': rh_correlation}
   

if __name__ == "__main__":
    output_dict = pipeline(data_dir, parent_submission_dir, subj,args)
    reg_lh = output_dict['reg_lh']
    reg_rh = output_dict['reg_rh']
    lh_correlation = output_dict['lh_correlation']
    rh_correlation = output_dict['rh_correlation']

    hemisphere = 'left' #@param ['left', 'right'] {allow-input: true}
    map_correlation_to_rois(args,lh_correlation,rh_correlation,hemisphere=hemisphere)
    
    #Save model and corellation values
    trained_model_dir = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/trained_models'   
    model_name = f'subj{subj}_model'

    models_folder = save_model(reg_lh, reg_rh, folder_path=trained_model_dir, model_name=model_name)
    roi_names, lh_mean_roi_correlation, rh_mean_roi_correlation = visualize_encdoing_accuaracy(args,lh_correlation,rh_correlation,correlation_path=models_folder,plot=True)
    save_corellation(roi_names, lh_mean_roi_correlation, rh_mean_roi_correlation, correlation_path=models_folder, experiment_name=model_name)
    print("=====================================================================")
    print(f"\nTrained model and correlation values saved to: {models_folder}")
