import torch
import numpy as np 
import matplotlib.pyplot as plt
from unique_m7_m8 import * 
from Simulation_utils import *
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from PID_util import *




def create_cov_lr(config:dict,data) -> dict:
    """Take the Random Varibles and create predictions of the X1 and X2 using T"""
    cov, rv_list = data
    X1,X2,T = rv_list[0],rv_list[1],rv_list[2]
    lr_type = config['lr_type']
    if lr_type == 'ols':
        model1 = LinearRegression_fit(T,X1)
        model2 = LinearRegression_fit(T,X2)
    elif lr_type == 'ridge':
        alpha = config.get('ridge_alpha', None)
        _,model1 = compute_ridge_cv_r2(T,X1,alpha)
        _,model2 = compute_ridge_cv_r2(T,X2,alpha)
    T_np = T.detach().cpu().numpy()
    X1_pred = model1.predict(T_np)
    X2_pred = model2.predict(T_np)

    X1_pred = torch.from_numpy(X1_pred).to(config['device'])
    X2_pred = torch.from_numpy(X2_pred).to(config['device'])

    rvs = [X1_pred,X2_pred,T]
    cov_dict = create_cov_matrix(rvs=rvs,device=config['device'])


    return {'cov_dict': cov_dict,'cov': cov_dict['full_cov'] ,'rv_list': rvs}



