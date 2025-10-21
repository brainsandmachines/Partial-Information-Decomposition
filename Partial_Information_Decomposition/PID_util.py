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


def create_cov_matrix(T,M1,M2):
    """This function will create the covariance matrix for the three variables M1,M2,T
    input: M1,M2,T are torch tensors of shape (N,d) 
    N is the number of observations, 
    d is the dimension of each observation.

    output: a 3*dX3*d covariance matrix"""
    # Stack all variables side by side
    Z = torch.hstack([T, M1, M2])   # shape (N, d_T+d_M1+d_M2)

    Sigma = torch.cov(Z.T,correction=1) #Correction means unbiased estimator (N-1 in denominator)
    cov_dict = {}

    m1_dim = M1.shape[1]
    m2_dim = M2.shape[1]
    t_dim = T.shape[1]
    N = M1.shape[0] #number of observations
    
    dt_dm1 = t_dim + m1_dim
    dt_dm2 = t_dim + m2_dim
    dm1_dm2 = m1_dim + m2_dim
    d_all = t_dim + m1_dim + m2_dim

    cov_dict['cov_t'] = Sigma[0:t_dim, 0:t_dim] #ΣT
    cov_dict['cov_m1'] = Sigma[t_dim:dt_dm1, t_dim:dt_dm1] #ΣM1
    cov_dict['cov_m2'] = Sigma[dt_dm1:d_all, dt_dm1:d_all] #ΣM2
    cov_dict['cov_t_m1'] = Sigma[0:t_dim, t_dim:dt_dm1] #ΣT,M1
    cov_dict['cov_t_m2'] = Sigma[0:t_dim, dt_dm1:d_all] #ΣT,M2
    cov_dict['cross_cov_m1_m2'] = Sigma[t_dim:dt_dm1, dt_dm1:d_all]#ΣM1,M2
    cov_dict['cross_cov_m12_t'] = Sigma[t_dim:d_all, 0:t_dim] #ΣM1M2,T 
    cov_dict['auto_cov_m12'] = Sigma[t_dim:d_all, t_dim:d_all]  #ΣM1M2 
    cov_dict['cov_tm1'] = Sigma[0:dt_dm1, 0:dt_dm1] #ΣT,M1
    ##ΣTM2:
    a = torch.cat((cov_dict['cov_t'], cov_dict['cov_t_m2']),dim=1)
    b = torch.cat((cov_dict['cov_t_m2'].T, cov_dict['cov_m2']),dim=1)
    cov_dict['cov_tm2'] = torch.cat((a,b),dim=0)
    
    cov_dict['full_cov'] = Sigma #Full covariance matrix ΣTM1M2

    return cov_dict
