import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from scipy.stats import pearsonr as corr
from encoding_utils import map_correlation_to_rois, plot_fmri,fmri_response_image, split_dataset,visualize_encdoing_accuaracy
from typing import Optional

class encoding_model():
  def __init__(self,device  ,model:str='alexnet',model_layer:str='features.2',model_path:str ='pytorch/vision:v0.10.0' ,features:Optional[np.ndarray]=None):
        self.device = device

        self.model = model
        self.model_path = model_path
        self.model_layer = model_layer
        self.features = features
        self.trained_dict = {}
        self.feature_dict  = {}

        print("\n==========================================================")

        # Load model (now that self is defined)
        print(f"No features provided, extracting features using {self.model} at layer {self.model_layer}...")
        model = torch.hub.load(self.model_path, self.model)
        model.to(self.device)
        model.eval()

        train_nodes, _ = get_graph_node_names(model) #Extract name of layers
        print(f"Train nodes: {train_nodes}")

        # create feature extractor for the chosen layer
        self.feature_extractor = create_feature_extractor(model, return_nodes=[self.model_layer])

  def fit_pca(self,dataloader, batch_size=500,ncomponents=100):
      print("\nFitting PCA on the extracted features...")
      print(f"Number of PCA components: {ncomponents}")
      print(f"Batch size: {batch_size}")
      pca = IncrementalPCA(n_components=ncomponents, batch_size=batch_size)
      for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
          ft = self.feature_extractor(d)
          ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
          pca.partial_fit(ft.detach().cpu().numpy())
      return pca

  def extract_features(self,dataloader,pca):  
      features = []
      for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
          ft = self.feature_extractor(d)
          ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
          ft = pca.transform(ft.cpu().detach().numpy())
          features.append(ft)
      return np.vstack(features)

  def train(self,train_data_loader,lh_fmri_train,rh_fmri_train,features_train:Optional[np.ndarray]=None):

    self.features_train = self.features_train if self.features_train is not None else features_train

    print('\nTraining images features:')
    # Fit linear regressions on the training data
    self.reg_lh = LinearRegression().fit(self.features_train, lh_fmri_train)
    self.reg_rh = LinearRegression().fit(self.features_train, rh_fmri_train)

    print("\nTraining completed.")

    self.trained_dict['reg_lh'] = self.reg_lh
    self.trained_dict['reg_rh'] = self.reg_rh

    return self.reg_lh, self.reg_rh

  def validate(self,reg_lh, reg_rh, lh_fmri_val,rh_fmri_val,features_val):
    """"This funciton validates the encoding model on the validation set and returns the correlation scores for each hemisphere."""
    # Use fitted linear regressions to predict the validation and test fMRI data
    
    print("\nValidating the model on the validation set...")

    reg_model_lh = self.reg_lh if self.reg_lh is not None else reg_lh
    reg_model_rh = self.reg_rh if self.reg_rh is not None else reg_rh

    lh_fmri_val_pred = reg_model_lh.predict(features_val)
    rh_fmri_val_pred = reg_model_rh.predict(features_val)

    print(f'\n Finding mean correlation for each hemisphere...')
        # Empty correlation array of shape: (LH vertices)
    lh_correlation = np.zeros(lh_fmri_val_pred.shape[1])
    # Correlate each predicted LH vertex with the corresponding ground truth vertex
    for v in tqdm(range(lh_fmri_val_pred.shape[1])):
        lh_correlation[v] = corr(lh_fmri_val_pred[:,v], lh_fmri_val[:,v])[0]

    # Empty correlation array of shape: (RH vertices)
    rh_correlation = np.zeros(rh_fmri_val_pred.shape[1])
    # Correlate each predicted RH vertex with the corresponding ground truth vertex
    for v in tqdm(range(rh_fmri_val_pred.shape[1])):
        rh_correlation[v] = corr(rh_fmri_val_pred[:,v], rh_fmri_val[:,v])[0]

    return lh_correlation, rh_correlation

  def test(self,dataloader,lh_fmri_val,rh_fmri_val):
    """This function is to test the model on the test set.
    but for my case there isnt any test fmri data so I will skip this part."""
    # # Use fitted linear regressions to predict the validation and test fMRI data
    # features_test = self.extract_features(test_imgs_dataloader)
    # lh_fmri_test_pred = reg_lh.predict(features_test)
    # rh_fmri_test_pred = reg_rh.predict(features_test)
    pass


  def run_model(self,train_imgs_dataloader,val_imgs_dataloader,lh_fmri_train,rh_fmri_train,lh_fmri_val,rh_fmri_val):
    """This function runs the entire encoding model pipeline: feature extraction, training, validation without testing."""
    pca = self.fit_pca(train_imgs_dataloader)  #Fit PCA on the data loader used for training and validation

    #Training
    print("\nExtracting features for the training set...")
    self.features_train = self.extract_features(train_imgs_dataloader, pca)
    self.feature_dict['features_train'] = self.features_train

    reg_lh,reg_rh = self.train(train_imgs_dataloader, lh_fmri_train, rh_fmri_train)

    #Validation
    print("\nExtracting features for the validation set...")
    features_val = self.extract_features(val_imgs_dataloader, pca)
    self.feature_dict['features_val'] = features_val

   

    lh_correlation, rh_correlation = self.validate(reg_lh, reg_rh, lh_fmri_val, rh_fmri_val, features_val)
    return lh_correlation, rh_correlation