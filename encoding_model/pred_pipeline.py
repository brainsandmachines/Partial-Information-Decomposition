import torch
import os
from encoding_model.algoanut_data import argObj, load_data_algonauts
from encoding_model.fmri_model import encoding_model
from encoding_model.encoding_utils import check_folder_exists,plot_fmri, split_dataset, map_correlation_to_rois,ImageDataset
from encoding_model.encoding_utils import visualize_encdoing_accuaracy,save_corellation,save_model,fmri_data_loader 
from pyparsing import Optional
from scipy.stats import pearsonr as corr


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir  = '/mnt/data4tb/data_algonauts/'
parent_submission_dir = '/mnt/data4tb/data_algonauts/submissions'
fmri_fig_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/fmri_figs'
correlation_path = '/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/correlations_fig'
subj = 1
args = argObj(data_dir, parent_submission_dir, subj)

def pipeline(data_dir, parent_submission_dir, subj,args,layer_name='features.2',model=None,features=None, only_validate=False,train_p=80,data_fmri=None,data_imgs=None):
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
    if data_fmri is not None and data_imgs is not None:
        lh_fmri = data_fmri['lh_fmri']
        rh_fmri = data_fmri['rh_fmri']
        train_img_list = data_imgs['train_img_list']
        test_img_list = data_imgs['test_img_list'] if 'test_img_list' in data_imgs else None
    else:
        print(f"No data was provided, loading from {data_dir}...")
        output_dict, data_dict = load_data_algonauts(paths_dict={'data_dir': data_dir, 'parent_submission_dir': parent_submission_dir}, args=args, subj=subj)

        train_img_list = output_dict['train_img_list']
        test_img_list = output_dict['test_img_list']
        lh_fmri = output_dict['lh_fmri']
        rh_fmri = output_dict['rh_fmri']

        train_img_dir = data_dict['train_img_dir']
        test_img_dir = data_dict['test_img_dir']
        

        idxs_train, idxs_val, idxs_test = split_dataset(train_img_list=train_img_list,test_img_list=test_img_list,train_p=train_p)

        fmri_dict, train_data_loader,val_imgs_dataloader = fmri_data_loader(lh_fmri,rh_fmri,train_img_list,test_img_list,train_img_dir,test_img_dir,batch_size=500,train_p=100)
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


    encoding_model_instance = encoding_model(device=device,model_layer=layer_name,model=model,features=features)
    reg_lh, reg_rh, lh_correlation, rh_correlation = encoding_model_instance.run_model(train_data_loader,val_imgs_dataloader,lh_fmri_train,
                                                                                rh_fmri_train,lh_fmri_val,rh_fmri_val,batch_size=500,ncomponents=None)
    features_val_pred_lh = encoding_model_instance.lh_fmri_val_pred if hasattr(encoding_model_instance, 'lh_fmri_val_pred') else None
    features_val_pred_rh = encoding_model_instance.rh_fmri_val_pred if hasattr(encoding_model_instance, 'rh_fmri_val_pred') else None
    features_train = encoding_model_instance.features_train if hasattr(encoding_model_instance, 'features_train') else None
    features_val_trained = encoding_model_instance.features_val if hasattr(encoding_model_instance, 'features_val') else None
    

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




if __name__ == "__main__":

    model = 'alexnet' #@param ['alexnet', 'vgg16', 'resnet50'] {allow-input: true}
    layer_name = 'features.2' #@param ['features.2', 'features.5', 'features.10', 'features.12', 'features.16'] {allow-input: true}
    model_name = f'RidgeCV_subj{subj}_model_{model}_{layer_name}'
    trained_model(layer_name, model,model_name=model_name,train_p=100)

    
