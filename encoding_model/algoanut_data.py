import os
import numpy as np
from pathlib import Path
from PIL import Image
from pyparsing import Optional
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr
from encoding_utils import map_correlation_to_rois, plot_fmri,fmri_response_image, split_dataset,visualize_encdoing_accuaracy



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class argObj:
  def __init__(self, data_dir, parent_submission_dir, subj):

    self.subj = format(subj, '02')
    self.data_dir = os.path.join(data_dir, 'subj'+self.subj)
    self.parent_submission_dir = parent_submission_dir
    self.subject_submission_dir = os.path.join(self.parent_submission_dir,
        'subj'+self.subj)

    # Create the submission directory if not existing
    if not os.path.isdir(self.subject_submission_dir):
        os.makedirs(self.subject_submission_dir)

class ImageDataset(Dataset):
    def __init__(self, imgs_paths, idxs, transform):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img).to(device)
        return img





if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  data_dir  = '/mnt/data4tb/data_algonauts/'
  parent_submission_dir = '/mnt/data4tb/data_algonauts/submissions'
  subj = 1
  args = argObj(data_dir, parent_submission_dir, subj)

  # Load fmri data:
  fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri') #/mnt/data4tb/data_algonauts/subj01/training_split/training_fmri
  lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
  rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

  print('LH training fMRI data shape:')
  print(lh_fmri.shape)
  print('(Training stimulus images × LH vertices)')

  print('\nRH training fMRI data shape:')
  print(rh_fmri.shape)
  print('(Training stimulus images × RH vertices)')


  train_img_dir  = os.path.join(args.data_dir, 'training_split', 'training_images') #'/mnt/data4tb/data_algonauts/subj01/training_split/training_images'
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


  fmri_response_image(path=fmri_fig_path,
                      args=args,
                      hemisphere='left',
                      img_idx=0,
                      train_img_dir=train_img_dir,
                      train_img_list=train_img_list,
                      lh_fmri=lh_fmri,
                      rh_fmri=rh_fmri)

