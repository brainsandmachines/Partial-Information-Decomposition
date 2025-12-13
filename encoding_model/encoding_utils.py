from nilearn import plotting
import os
import numpy as np
from numpy import ndarray
from nilearn import datasets
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, List
import joblib
#Import utils
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from utils import check_file_exists,check_folder_exists
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch

class ImageDataset(Dataset):
    def __init__(self, imgs_paths, idxs, transform):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img).to(self.device)
        return img
def plot_fmri(path,args, hemi, title=''):
    """Plot fMRI data on a brain surface and save the figure.

    Args:
        path (str): Path to save the figure.
        hemi (str): 'left' or 'right' hemisphere.
        title (str, optional): Title of the plot. Defaults to ''.
        vmax (float, optional): Maximum value for color scaling. Defaults to None.
        arguments (argObj): Argument object containing data directories."""
    
    hemisphere = hemi

    hemisphere2 = 'left' if hemisphere=='l' else 'right'

    # Load the brain surface map of all vertices
    roi_dir = os.path.join(args.data_dir, 'roi_masks',
        hemisphere[0]+'h.all-vertices_fsaverage_space.npy')
    fsaverage_all_vertices = np.load(roi_dir)

    # Create the interactive brain surface map
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    fig = plotting.plot_surf(
        surf_mesh=fsaverage['infl_'+hemisphere2],
        surf_map=fsaverage_all_vertices,
        bg_map=fsaverage['sulc_'+hemisphere2],
        threshold=1e-14,
        cmap='cool'
    ).figure

    fig.savefig(f"{path}/{hemisphere2}_surface_view.png", dpi=300, bbox_inches="tight")

def fmri_response_image(path,args,hemisphere,img_idx,train_img_dir,train_img_list,lh_fmri,rh_fmri):
    """This function outputs the fmri response that matches the image shown.
    accoring to the NSD dataset structure.
    Args:
        path (str): Path to save the figure.
        hemi (str): 'left' or 'right' hemisphere.
        img_idx (int): Index of the image shown.
        arguments (argObj): Argument object containing data directories."""
    img = 0 #@param
    hemisphere = 'left' #@param ['left', 'right'] {allow-input: true}

    # Load the image
    img_dir = os.path.join(train_img_dir, train_img_list[img])
    train_img = Image.open(img_dir).convert('RGB')

    # Plot the image
    plt.figure()
    plt.axis('off')
    plt.imshow(train_img)
    plt.title('Training image: ' + str(img+1));
    plt.savefig(f"{path}/training_image_{img_idx}.png", dpi=300, bbox_inches="tight")

    # Load the brain surface map of all vertices
    roi_dir = os.path.join(args.data_dir, 'roi_masks',
        hemisphere[0]+'h.all-vertices_fsaverage_space.npy')
    fsaverage_all_vertices = np.load(roi_dir)

    # Map the fMRI data onto the brain surface map
    fsaverage_response = np.zeros(len(fsaverage_all_vertices))
    if hemisphere == 'left':
        fsaverage_response[np.where(fsaverage_all_vertices)[0]] = lh_fmri[img]
    elif hemisphere == 'right':
        fsaverage_response[np.where(fsaverage_all_vertices)[0]] = rh_fmri[img]

    # Create the interactive brain surface map
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    fig = plotting.plot_surf(
        surf_mesh=fsaverage['infl_'+hemisphere],
        surf_map=fsaverage_response,
        bg_map=fsaverage['sulc_'+hemisphere],
        threshold=1e-14,
        cmap='cold_hot',
        colorbar=True,
        title='All vertices, '+hemisphere+' hemisphere'
        ).figure
    fig.savefig(f"{path}/{hemisphere}_fmri_response_{img_idx}.png", dpi=300, bbox_inches="tight")


def split_dataset(train_img_list,test_img_list,rand_seed=5,train_p=90):
     
    np.random.seed(rand_seed)

    # Calculate how many stimulus images correspond to 90% of the training data
    num_train = int(np.round(len(train_img_list) / 100 * train_p))
    # Shuffle all training stimulus images
    idxs = np.arange(len(train_img_list))
    np.random.shuffle(idxs)
    # Assign 90% of the shuffled stimulus images to the training partition,
    # and 10% to the test partition
    idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]
    # No need to shuffle or split the test stimulus images
    idxs_test = np.arange(len(test_img_list))

    print('Training stimulus images: ' + format(len(idxs_train)))
    print('\nValidation stimulus images: ' + format(len(idxs_val)))
    print('\nTest stimulus images: ' + format(len(idxs_test)))
    
    return idxs_train, idxs_val, idxs_test

def fmri_data_loader(lh_fmri,rh_fmri,train_img_list,test_img_list,train_img_dir,test_img_dir,batch_size=500,train_p=90):
    
    

    idxs_train, idxs_val, idxs_test = split_dataset(train_img_list=train_img_list,test_img_list=test_img_list,train_p=train_p)
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
    fmri_dict = {}
    fmri_dict['lh_fmri_train'] = lh_fmri_train
    fmri_dict['lh_fmri_val'] = lh_fmri_val
    fmri_dict['rh_fmri_train'] = rh_fmri_train
    fmri_dict['rh_fmri_val'] = rh_fmri_val

    return fmri_dict,train_imgs_dataloader,val_imgs_dataloader



def map_correlation_to_rois(args,lh_correlation,rh_correlation,hemisphere):
    """Map correlation values to ROIs.

    Args:
        corrs (np.array): Array of correlation values for each vertex.
        roi_mask (np.array): Array of ROI labels for each vertex.

    Returns:
        dict: Dictionary mapping ROI labels to mean correlation values.
    """
        # Load the brain surface map of all vertices
    roi_dir = os.path.join(args.data_dir, 'roi_masks',
        hemisphere[0]+'h.all-vertices_fsaverage_space.npy')
    fsaverage_all_vertices = np.load(roi_dir)

    # Map the correlation results onto the brain surface map
    fsaverage_correlation = np.zeros(len(fsaverage_all_vertices))
    if hemisphere == 'left':
        fsaverage_correlation[np.where(fsaverage_all_vertices)[0]] = lh_correlation
    elif hemisphere == 'right':
        fsaverage_correlation[np.where(fsaverage_all_vertices)[0]] = rh_correlation

    # Create the interactive brain surface map
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    fig = plotting.plot_surf(
        surf_mesh=fsaverage['infl_'+hemisphere],
        surf_map=fsaverage_correlation,
        bg_map=fsaverage['sulc_'+hemisphere],
        threshold=1e-14,
        cmap='cold_hot',
        colorbar=True,
        title='Encoding accuracy, '+hemisphere+' hemisphere'
        ).figure
    
def visualize_encdoing_accuaracy(args,lh_correlation,rh_correlation,correlation_path,plot=True):
    """Visualize encoding accuracy with a bar graph and return ROI correlation values and 
    ROI names for left and right hemispheres for a given subject.

    Args:
        args (argObj): Argument object containing data directories.
        lh_correlation (np.array): Array of correlation values for left hemisphere vertices.
        rh_correlation (np.array): Array of correlation values for right hemisphere vertices.
        fmri_fig_path (str): Path to save the figure.
        
        Returns: 
            x- axis: ROIs
            y- axis: Mean Pearson's r
    """
        # Load the ROI classes mapping dictionaries
    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
        'mapping_floc-faces.npy', 'mapping_floc-places.npy',
        'mapping_floc-words.npy', 'mapping_streams.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(args.data_dir, 'roi_masks', r),
            allow_pickle=True).item())

    # Load the ROI brain surface maps
    lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
        'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
        'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
        'lh.streams_challenge_space.npy']
    rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
        'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
        'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
        'rh.streams_challenge_space.npy']
    lh_challenge_rois = []
    rh_challenge_rois = []
    for r in range(len(lh_challenge_roi_files)):
        lh_challenge_rois.append(np.load(os.path.join(args.data_dir, 'roi_masks',
            lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(args.data_dir, 'roi_masks',
            rh_challenge_roi_files[r])))

    # Select the correlation results vertices of each ROI
    roi_names = []
    lh_roi_correlation = []
    rh_roi_correlation = []
    for r1 in range(len(lh_challenge_rois)):
        for r2 in roi_name_maps[r1].items():
            if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                roi_names.append(r2[1])
                lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                lh_roi_correlation.append(lh_correlation[lh_roi_idx])
                rh_roi_correlation.append(rh_correlation[rh_roi_idx])
    roi_names.append('All vertices')
    lh_roi_correlation.append(lh_correlation)
    rh_roi_correlation.append(rh_correlation)
    if plot:
        # Create the plot
        lh_mean_roi_correlation = [np.mean(lh_roi_correlation[r])
            for r in range(len(lh_roi_correlation))]
        rh_mean_roi_correlation = [np.mean(rh_roi_correlation[r])
            for r in range(len(rh_roi_correlation))]
        plt.figure(figsize=(18,6))
        x = np.arange(len(roi_names))
        width = 0.30
        plt.bar(x - width/2, lh_mean_roi_correlation, width, label='Left Hemisphere')
        plt.bar(x + width/2, rh_mean_roi_correlation, width,
            label='Right Hemishpere')
        plt.xlim(left=min(x)-.5, right=max(x)+.5)
        plt.ylim(bottom=0, top=1)
        plt.xlabel('ROIs')
        plt.xticks(ticks=x, labels=roi_names, rotation=60)
        plt.ylabel('Mean Pearson\'s $r$')
        plt.legend(frameon=True, loc=1)
        fig_dir = check_file_exists(os.path.join(correlation_path,
            'mean_roi_correlation.png'))
        plt.savefig(fig_dir,
            dpi=300, bbox_inches="tight")
    
    return roi_names, lh_mean_roi_correlation, rh_mean_roi_correlation
    

def save_corellation(roi_names,lh_correlation,rh_correlation,correlation_path,experiment_name):
    """Save correlation values to .npy files.

    Args:
        lh_correlation (np.array): Array of correlation values for left hemisphere vertices.
        rh_correlation (np.array): Array of correlation values for right hemisphere vertices.
        correlation_path (str): Path to save the correlation files.
    """
    #Define file name: 
    lh_name = f'{experiment_name}_lh_correlation.npy' if experiment_name else 'lh_correlation.npy'
    rh_name = f'{experiment_name}_rh_correlation.npy' if experiment_name else 'rh_correlation.npy'
    roi_names_name = f'{experiment_name}_roi_names.npy' if experiment_name else 'roi_names.npy'


    lh_corr_file = check_file_exists(os.path.join(correlation_path,
        lh_name))
    rh_corr_file = check_file_exists(os.path.join(correlation_path,
        rh_name))
    roi_names_file = check_file_exists(os.path.join(correlation_path,
        roi_names_name))
    
    np.save(lh_corr_file, lh_correlation)
    np.save(rh_corr_file, rh_correlation)
    np.save(roi_names_file, roi_names)
    print(f"Correlation files: {roi_names_file}, {lh_corr_file}, {rh_corr_file} saved to: {correlation_path}")


def save_model(folder_path, model_name,save_dict,reg_lh:Optional[ndarray]=None, reg_rh:Optional[ndarray]=None,features_val_pred_lh:Optional[List]=None,features_val_pred_rh:Optional[List]=None,features_train:Optional[ndarray]=None,features_val_trained:Optional[ndarray]=None,predict_array:Optional[ndarray]=None,
               roi_names:Optional[List]=None,lh_correlation:Optional[ndarray]=None,rh_correlation:Optional[ndarray]=None):
    """Save the trained encoding model. with its corellation values and roi names and figs`

    Args:
        reg_lh (LinearRegression): Trained left hemisphere regression model.
        reg_rh (LinearRegression): Trained right hemisphere regression model.
        folder_path (str): Path to save the model's folder.
        model_name (str): Name of the model.
        roi_names (list, optional): List of ROI names. Defaults to None.
        lh_correlation (np.array, optional): Array of correlation values for left hemisphere vertices. Defaults to None.
        rh_correlation (np.array, optional): Array of correlation values for right
        
        Returns:
            Saves the model and correlation values to specified folder with the specific rois.
    """
    model_name_joblib = f'{model_name}_encoding_model.joblib' if model_name else 'encoding_model.joblib'
    
    models_folder = check_folder_exists(f'{folder_path}/{model_name}')
    model_save_path = os.path.join(models_folder, model_name_joblib)
    if save_dict is None:
        save_dict = {'reg_lh': reg_lh, 'reg_rh': reg_rh, 'features_val_pred_lh': features_val_pred_lh, 'features_val_pred_rh': features_val_pred_rh, 'features_train': features_train,'features_val_trained': features_val_trained, 'predict_array': predict_array}
    joblib.dump(save_dict, model_save_path)
    
    if roi_names is not None and lh_correlation is not None and rh_correlation is not None:
        save_corellation(roi_names,lh_correlation,rh_correlation,models_folder,experiment_name=model_name)

    print(f"Encoding model saved to: {model_save_path}")
    
    return models_folder