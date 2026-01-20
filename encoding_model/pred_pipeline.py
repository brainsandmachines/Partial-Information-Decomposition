import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os
from encoding_model.algoanut_data import argObj, load_data_algonauts
from encoding_model.fmri_model import encoding_model
from encoding_model.encoding_utils import check_folder_exists,plot_fmri, split_dataset, map_correlation_to_rois,ImageDataset
from encoding_model.encoding_utils import visualize_encdoing_accuaracy,save_corellation,save_model,fmri_data_loader 
import torchvision.transforms as transforms
from pathlib import Path
import joblib
from pyparsing import Optional
from scipy.stats import pearsonr as corr
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir  = '/mnt/data4tb/data_algonauts/'
parent_submission_dir = '/mnt/data4tb/data_algonauts/submissions'
fmri_fig_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/fmri_figs'
correlation_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/correlations_fig'
subj = 1
args = argObj(data_dir, parent_submission_dir, subj)

def pipeline(data_dir, parent_submission_dir, subj,args,layer_name='features.2',model=None,features=None, only_validate=False,train_p=80):
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
    
        


    idxs_train, idxs_val, idxs_test = split_dataset(train_img_list=train_img_list,test_img_list=test_img_list,train_p=train_p)

    output_dict, data_dict = load_data_algonauts(paths_dict={'data_dir': data_dir, 'parent_submission_dir': parent_submission_dir}, args=args, subj=subj)

    mri_dict, train_data_loader,val_imgs_dataloader = fmri_data_loader(lh_fmri,rh_fmri,train_img_list,test_img_list,train_img_dir,test_img_dir,batch_size=500,train_p=100)
    train_img_list = output_dict['train_img_list']
    test_img_list = output_dict['test_img_list']
    lh_fmri = output_dict['lh_fmri']
    rh_fmri = output_dict['rh_fmri']

    train_img_dir = data_dict['train_img_dir']
    test_img_dir = data_dict['test_img_dir']
    


    lh_fmri_train = lh_fmri[idxs_train]
    lh_fmri_val = lh_fmri[idxs_val]
    rh_fmri_train = rh_fmri[idxs_train]
    rh_fmri_val = rh_fmri[idxs_val]
    fmri_dict = {}
    fmri_dict['lh_fmri_train'] = lh_fmri_train
    fmri_dict['lh_fmri_val'] = lh_fmri_val
    fmri_dict['rh_fmri_train'] = rh_fmri_train
    fmri_dict['rh_fmri_val'] = rh_fmri_val

    if not only_validate:
        encoding_model_instance = encoding_model(device=device,model_layer=layer_name,model=model,features=features)
        reg_lh, reg_rh, lh_correlation, rh_correlation = encoding_model_instance.run_model(train_data_loader,val_imgs_dataloader,lh_fmri_train,
                                                                                    rh_fmri_train,lh_fmri_val,rh_fmri_val,batch_size=500,ncomponents=None)
        features_val_pred_lh = encoding_model_instance.lh_fmri_val_pred if hasattr(encoding_model_instance, 'lh_fmri_val_pred') else None
        features_val_pred_rh = encoding_model_instance.rh_fmri_val_pred if hasattr(encoding_model_instance, 'rh_fmri_val_pred') else None
        features_train = encoding_model_instance.features_train if hasattr(encoding_model_instance, 'features_train') else None
        features_val_trained = encoding_model_instance.features_val if hasattr(encoding_model_instance, 'features_val') else None
        #save_fmri
        #fmri_dicts_paths = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/fmri_dicts'
        #fmri_save_path = os.path.join(fmri_dicts_paths, f'subj{subj}_fmri_dicts.joblib')
        #joblib.dump(fmri_dict, fmri_save_path)
    
    else:
        encoding_model_instance = encoding_model(device=device,model_layer=layer_name,model=model,features=features)
        zero_predict_list = []
        for _, d in tqdm(enumerate(val_imgs_dataloader), total=len(val_imgs_dataloader)):
            zero_predict = encoding_model_instance.feature_extractor(d)
            zero_predict_list.append(zero_predict[layer_name].cpu().detach().numpy())
        zero_predict_array = np.vstack(zero_predict_list)
        features_val = zero_predict_array.reshape(zero_predict_array.shape[0], -1)
        W_lh = np.random.randn(46656, 19004)
        features_val_lh = features_val[:, :19004]
        W_rh = np.random.randn(46656, 20544)
        features_val_rh = features_val[:, :20544]

        
        #pca = encoding_model_instance.fit_pca(train_imgs_dataloader)
        #features_val = encoding_model_instance.extract_features(val_imgs_dataloader, pca)
        print(f'\n Finding mean correlation for each hemisphere...')
        # Empty correlation array of shape: (LH vertices)
        lh_correlation = np.zeros(features_val_lh.shape[1])
        # Correlate each predicted LH vertex with the corresponding ground truth vertex
        for v in tqdm(range(features_val_lh.shape[1])):
            lh_correlation[v] = corr(features_val_lh[:,v], lh_fmri_val[:,v])[0]

        # Empty correlation array of shape: (RH vertices)
        rh_correlation = np.zeros(features_val_rh.shape[1])
        # Correlate each predicted RH vertex with the corresponding ground truth vertex
        for v in tqdm(range(features_val_rh.shape[1])):
            rh_correlation[v] = corr(features_val_rh[:,v], rh_fmri_val[:,v])[0]
        reg_lh = None
        reg_rh = None
        features_val_pred_lh = features_val_lh
        features_val_pred_rh = features_val_rh
        predict_array = zero_predict_array
        diction = {}
        diction['predict_array'] = predict_array if predict_array is not None else None 
        features_train = None
    diction = {}
    diction['reg_lh'] = reg_lh if reg_lh is not None else None
    diction['reg_rh'] = reg_rh if reg_rh is not None else None
    diction['lh_correlation'] = lh_correlation if lh_correlation is not None else None
    diction['rh_correlation'] = rh_correlation if rh_correlation is not None else None
    diction['features_val_pred_lh'] = features_val_pred_lh if features_val_pred_lh is not None else None
    diction['features_val_pred_rh'] = features_val_pred_rh if features_val_pred_rh is not None else None
    diction['features_train'] = features_train if features_train is not None else None
    diction['features_val_trained'] = encoding_model_instance.features_val if hasattr(encoding_model_instance, 'features_val') else None
    
    return diction


def trained_model(layer_name,model,model_name,train_p):
    output_dict = pipeline(data_dir, parent_submission_dir, subj,args,layer_name,model=model,train_p=train_p)
    reg_lh = output_dict['reg_lh']
    reg_rh = output_dict['reg_rh']

    if train_p != 100:
        lh_correlation = output_dict['lh_correlation'] if 'lh_correlation' in output_dict else None
        rh_correlation = output_dict['rh_correlation'] if 'rh_correlation' in output_dict else None
        features_val_pred_lh = output_dict['features_val_pred_lh'] if 'features_val_pred_lh' in output_dict else None
        features_val_pred_rh = output_dict['features_val_pred_rh'] if 'features_val_pred_rh' in output_dict else None
        features_train = output_dict['features_train']
        features_val_trained = output_dict['features_val_trained'] if 'features_val_trained' in output_dict else None


    #Save model and corellation values
    trained_model_dir = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/trained_models'   


    models_folder = save_model(save_dict=output_dict,folder_path=trained_model_dir, model_name=model_name)
    print(f'Model: {model_name} saved successfully.!!!')

    hemisphere = 'left' #@param ['left', 'right'] {allow-input: true}
    map_correlation_to_rois(args,lh_correlation,rh_correlation,hemisphere=hemisphere)
    
    roi_names, lh_mean_roi_correlation, rh_mean_roi_correlation = visualize_encdoing_accuaracy(args,lh_correlation,rh_correlation,correlation_path=models_folder,plot=True)
    save_corellation(roi_names, lh_mean_roi_correlation, rh_mean_roi_correlation, correlation_path=models_folder, experiment_name=model_name)
    print("==================================== =================================")
    print(f"\nTrained model and correlation values saved to: {models_folder}")

def just_validate(layer_name, model):
    output_dict = pipeline(data_dir, parent_submission_dir, subj,args,layer_name,model=model, only_validate=True)
    reg_lh = output_dict['reg_lh']
    reg_rh = output_dict['reg_rh']
    lh_correlation = output_dict['lh_correlation']
    rh_correlation = output_dict['rh_correlation']
    features_val_pred_lh = output_dict['features_val_pred_lh']
    features_val_pred_rh = output_dict['features_val_pred_rh']
    features_train = output_dict['features_train']
    features_val_trained = output_dict['features_val_trained']
    predict_array = output_dict['predict_array']

    hemisphere = 'left' #@param ['left', 'right'] {allow-input: true}
    map_correlation_to_rois(args,lh_correlation,rh_correlation,hemisphere=hemisphere)
    #Save model and corellation values
    trained_model_dir = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/trained_models'   
    model_name = f'subj{subj}_model_{layer_name}'

    models_folder = save_model(reg_lh, reg_rh,features_val_pred_lh,features_val_pred_rh,features_train,features_val_trained,predict_array,folder_path=trained_model_dir, model_name=model_name)
    roi_names, lh_mean_roi_correlation, rh_mean_roi_correlation = visualize_encdoing_accuaracy(args,lh_correlation,rh_correlation,correlation_path=models_folder,plot=True)
    save_corellation(roi_names, lh_mean_roi_correlation, rh_mean_roi_correlation, correlation_path=models_folder, experiment_name=model_name)
    print("=====================================================================")
    print(f"\nTrained model and correlation values saved to: {models_folder}")
    return models_folder

def check_corr_between_models(model_1_path,model_2_path):
    """This function checks the correlation between two trained models' correlation values."""
    model_1 = joblib.load(model_1_path)
    model_2 = joblib.load(model_2_path)
    

    lh_corr_1 = model_1['features_val_pred_lh']
    rh_corr_1 = model_1['features_val_pred_rh']

    lh_corr_2 = model_2['features_val_pred_lh']
    rh_corr_2 = model_2['features_val_pred_rh']
    lh_correlation_between_models = corr(lh_corr_1, lh_corr_2)[0]
    rh_correlation_between_models = corr(rh_corr_1, rh_corr_2)[0]
    between_models_folder = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/between_models_cor'
    roi_names, lh_mean_roi_correlation, rh_mean_roi_correlation = visualize_encdoing_accuaracy(args,lh_correlation_between_models,rh_correlation_between_models,correlation_path=between_models_folder,plot=True)
    save_corellation(roi_names, lh_mean_roi_correlation, rh_mean_roi_correlation, correlation_path=between_models_folder, experiment_name='between_models_correlation')

    print(f'Correlation between left hemisphere models: {lh_correlation_between_models}')
    print(f'Correlation between right hemisphere models: {rh_correlation_between_models}')
    return lh_correlation_between_models, rh_correlation_between_models

def pred_two_models(trained_model_path,untrained_model_path,fmri_path):
    trained_model = joblib.load(trained_model_path)
    untrained_model = joblib.load(untrained_model_path)
    fmri_dict = joblib.load(fmri_path)
    train_fmri_data_lh = fmri_dict['lh_fmri_train']
    train_fmri_data_rh = fmri_dict['rh_fmri_train']
    lh_fmri_val = fmri_dict['lh_fmri_val']
    rh_fmri_val = fmri_dict['rh_fmri_val']

    reg_lh_trained = trained_model['reg_lh']
    reg_rh_trained = trained_model['reg_rh']
    features_train = trained_model['features_train']
    features_val_trained = trained_model['features_val_trained']

    reg_lh_untrained = untrained_model['predict_array']
    reg_rh_untrained = untrained_model['predict_array']
    features_val_untrained = untrained_model['features_val_pred_lh']


    joint_model = np.concatenate(((reg_lh_trained.coef_).T, features_val_untrained))

    model_lh = LinearRegression()
    reg_lh = model_lh.fit(joint_model, train_fmri_data_lh)

    model_rh = LinearRegression()
    reg_rh = model_rh.fit(joint_model, train_fmri_data_rh)

    predict_lh = reg_lh.predict(features_val_trained)
    predict_rh = reg_rh.predict(features_val_trained)

    print(f'\n Finding mean correlation for each hemisphere...')
        # Empty correlation array of shape: (LH vertices)
    lh_correlation = np.zeros(predict_lh.shape[1])
    # Correlate each predicted LH vertex with the corresponding ground truth vertex
    for v in tqdm(range(predict_lh.shape[1])):
        lh_correlation[v] = corr(predict_lh[:,v], lh_fmri_val[:,v])[0]

    # Empty correlation array of shape: (RH vertices)
    rh_correlation = np.zeros(predict_rh.shape[1])
    # Correlate each predicted RH vertex with the corresponding ground truth vertex
    for v in tqdm(range(predict_rh.shape[1])):
        rh_correlation[v] = corr(predict_rh[:,v], rh_fmri_val[:,v])[0]

    return lh_correlation, rh_correlation


if __name__ == "__main__":

    model = 'alexnet' #@param ['alexnet', 'vgg16', 'resnet50'] {allow-input: true}
    layer_name = 'features.2' #@param ['features.2', 'features.5', 'features.10', 'features.12', 'features.16'] {allow-input: true}
    model_name = f'RidgeCV_subj{subj}_model_{model}_{layer_name}'
    trained_model(layer_name, model,model_name=model_name,train_p=100)

    
    # #models_folder = just_validate(layer_name, model)
    # models_folder = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/trained_models/subj1_model_features_1.2'
    # model_1_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/trained_models/subj1_model_features.2/subj1_model_features.2_encoding_model.joblib'
    # model_2_path = f'{models_folder}/subj1_model_features.2_encoding_model.joblib'
    # """ROI PPA is the desired ROI to compare between models
    # model 1 which is the trained model has high correlation in PPA
    # model 2 which is the validated model has low correlation in PPA
    # but they have correlation between them in the given ROI PPA"""
    # #check_corr_between_models(model_1_path,model_2_path)
    # fmri_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/fmri_dicts/subj1_fmri_dicts.joblib'
    # lh_correlation_joint, rh_correlation_joint = pred_two_models(model_1_path,model_2_path,fmri_path=fmri_path)

