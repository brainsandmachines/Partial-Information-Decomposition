import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os
from algoanut_data import argObj, ImageDataset, device
from fmri_model import encoding_model
from encoding_utils import plot_fmri, split_dataset, map_correlation_to_rois, visualize_encdoing_accuaracy
import torchvision.transforms as transforms
from pathlib import Path
from pyparsing import Optional

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir  = '/mnt/data4tb/data_algonauts/'
parent_submission_dir = '/mnt/data4tb/data_algonauts/submissions'
subj = 1
args = argObj(data_dir, parent_submission_dir, subj)

# Load fmri data:
fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

print('LH training fMRI data shape:')
print(lh_fmri.shape)
print('(Training stimulus images × LH vertices)')

print('\nRH training fMRI data shape:')
print(rh_fmri.shape)
print('(Training stimulus images × RH vertices)')


train_img_dir  = os.path.join(args.data_dir, 'training_split', 'training_images')
test_img_dir  = os.path.join(args.data_dir, 'test_split', 'test_images')

# Create lists will all training and test image file names, sorted
train_img_list = os.listdir(train_img_dir)
train_img_list.sort()
test_img_list = os.listdir(test_img_dir)
test_img_list.sort()
print('Training images: ' + str(len(train_img_list)))
print('Test images: ' + str(len(test_img_list)))

train_img_file = train_img_list[0]
print('Training image file name: ' + train_img_file)
print('73k NSD images ID: ' + train_img_file[-9:-4])

"""================================================="""
# Visualize fMRI data on brain surface:
hemisphere = 'l'  # 'l' or 'r'
fmri_fig_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/correlations_fig'
#plot_fmri(path=fmri_fig_path,args=args,hemi=hemisphere)
"""================================================="""


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
lh_correlation, rh_correlation = encoding_model_instance.run_model(train_imgs_dataloader,val_imgs_dataloader,lh_fmri_train,rh_fmri_train,lh_fmri_val,rh_fmri_val)
hemisphere = 'left' #@param ['left', 'right'] {allow-input: true}
map_correlation_to_rois(args,lh_correlation,rh_correlation,hemisphere=hemisphere)
visualize_encdoing_accuaracy(args,lh_correlation,rh_correlation,fmri_fig_path)