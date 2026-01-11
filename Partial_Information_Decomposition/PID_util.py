import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics import r2_score
from itertools import chain, combinations
from typing import List, Tuple, Union
import torch
from torch.linalg import inv, slogdet


def LinearRegression_fit(X,y):
    model = LinearRegression()
    model.fit(X,y)
    return model


def cond_cov(sigma_1,sigma_2,sigma12,sigma21):
    """This function will compute the conditional covariance matrix of two Gaussian variables
    Sigma_1|2 = Sigma_1 - Sigma12*inv(Sigma_2)*Sigma21

    input: sigma_1,sigma_2 are torch tensors of shape (d,d)
    d is the dimension of each observation.

    output: a torch tensor of shape (d,d)
    covariance(sigma_1|sigma_2)"""
    inv_sigma_2 = inv(sigma_2)
    cond_cov = sigma_1 - sigma12 @ inv_sigma_2 @ sigma21
    return cond_cov


def create_cov_matrix(X0,X1,X2):
    """This function will create the covariance matrix for the three variables M1,M2,T
    input: M1,M2,T are torch tensors of shape (N,d) 
    N is the number of observations, 
    d is the dimension of each observation.

    output: a 3*dX3*d covariance matrix"""
    # Stack all variables side by side
    Z = torch.hstack([X0, X1, X2])   # shape (N, d_T+d_M1+d_M2)

    Sigma = torch.cov(Z.T,correction=1) #Correction means unbiased estimator (N-1 in denominator)
    cov_dict = {}
    print(f"\nFull covariance matrix shape: {Sigma.shape}")
    x1_dim = X1.shape[1]
    x2_dim = X2.shape[1]
    x0_dim = X0.shape[1]
    N = X1.shape[0] #number of observations
    
    dt_dx1 = x0_dim + x1_dim
    d_all = x0_dim + x1_dim + x2_dim

    #Full covariance matrix
    cov_dict['full_cov'] = Sigma #Full covariance matrix ΣX0X1X2

    #Cross-Covariances:
    cov_dict['cross_x0_x1'] = Sigma[0:x0_dim, x0_dim:dt_dx1] #ΣX0,X1
    cov_dict['cross_x0_x2'] = Sigma[0:x0_dim, dt_dx1:d_all] #ΣX0,X2
    cov_dict['cross_x1_x2'] = Sigma[x0_dim:dt_dx1, dt_dx1:d_all]#ΣX1,X2
    cov_dict['cross_x12_x0'] = Sigma[x0_dim:d_all, 0:x0_dim] #ΣX1X2,X0

    #Auto-Covariances
    cov_dict['cov_x0'] = Sigma[0:x0_dim, 0:x0_dim] #ΣX0
    cov_dict['cov_x1'] = Sigma[x0_dim:dt_dx1, x0_dim:dt_dx1] #ΣX1
    cov_dict['cov_x2'] = Sigma[dt_dx1:d_all, dt_dx1:d_all] #ΣX2
    cov_dict['auto_x01'] = Sigma[0:dt_dx1, 0:dt_dx1]  #ΣX0X1
    cov_dict['auto_x02'] = Sigma[0:x0_dim, dt_dx1:d_all]  #ΣX0X2   
    cov_dict['auto_x12'] = Sigma[x0_dim:d_all, x0_dim:d_all]  #ΣX1X2

 
    ##ΣX0,X2:
    a = torch.cat((cov_dict['cov_x0'], cov_dict['cross_x0_x2']),dim=1)
    b = torch.cat((cov_dict['cross_x0_x2'].T, cov_dict['cov_x2']),dim=1)
    cov_dict['auto_x02'] = torch.cat((a,b),dim=0)


    return cov_dict
