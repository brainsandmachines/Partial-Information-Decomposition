import torch
import numpy as np 
import matplotlib.pyplot as plt
from unique_m7_m8 import * 
from Simulation_utils import *
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from PID_util import *




def create_cov_lr(config:dict,cov:torch.Tensor,rv:torch.Tensor) -> dict:
    """Take the Random Varibles and create predictions of the X1 and X2 using T"""

    X1,X2,T = rv[0],rv[1],rv[2]
    lr_type = config['lr_type']
    if lr_type == 'ols':
        model1 = LinearRegression_fit(X1,T)
        model2 = LinearRegression_fit(X2,T)
    elif lr_type == 'ridge':
        alpha = config.get('ridge_alpha', 1.0)
        _,model1 = compute_ridge_cv_r2(X1,T,alpha)
        _,model2 = compute_ridge_cv_r2(X2,T,alpha)

    X1_pred = model1.predict(X1)
    X2_pred = model2.predict(X2)

    rvs = [X1_pred,X2_pred,T]
    cov_dict = create_cov_matrix(rvs=rvs,device=config['device'])


    return {'cov_dict': cov_dict,'cov': cov_dict['full_cov'] ,'rv_list': rvs}



