import torch
import numpy as np
import matplotlib.pyplot as plt
from PID_util import *
from Partial_Information_Decomposition.Toy_Simulations.Bias_Corr_simulations import theoretical_covariance, sample_cov_simulation
from Idep_multivariate_gauss import Idep_multivariate_gauss














def run_pqr_simulation(seed,N,dims):
    
    total_dim = sum(dims)
    A = torch.randn(total_dim, total_dim)
    Sigma_theoretical_old = A @ A.T + torch.eye(total_dim) * 10.0  # Ensure positive definiteness with a large diagonal
    cov_dic = create_cov_matrix(Sigma=Sigma_theoretical_old, dims=dims)
    pid_class = Idep_multivariate_gauss
    pid_class.cov_dict = cov_dic
    P,Q,R = pid_class.P, pid_class.Q, pid_class.R
    
    