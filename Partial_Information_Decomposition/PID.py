
import numpy as np
from Commonality_Analysis import CommonalityAnalysis  # not used here; keep if you need it elsewhere
from Partial_Information_Decomposition.RV_toy import generate_random_variables
import admUI
from admUI import admUI_numpy
import torch
from torch.distributions import Distribution, Normal 
from sklearn.linear_model import LinearRegression



num_obsv = 10      
vec_size = 1
base_dist = Normal(loc=0.0, scale=1.0)
random_variables = ['S1', 'S2', 'S12', 'N1', 'N2', 'NT']

def toy_example(rv_dict):
    # M1 = S12 + S1 + N1
    # M2 = S12 + S2 + N2
    # T  = S12 + S1 + S2 + NT
    m1 = rv_dict['S12'] + rv_dict['S1'] + rv_dict['N1']
    m2 = rv_dict['S12'] + rv_dict['S2'] + rv_dict['N2']
    t  = rv_dict['S12'] + rv_dict['S1'] + rv_dict['S2'] + rv_dict['NT']
    return {'M1': m1, 'M2': m2, 'target': t}

# Your generator should return tensors of shape (n, 1)
result, _ = generate_random_variables(toy_example, num_obsv,vec_size ,base_dist, *random_variables)


M1, M2, target = result['M1'], result['M2'], result['target']

linear_model = LinearRegression()
X = torch.hstack([M1, M2])

linear_model.fit(X.numpy(), target.numpy())
beta0 = linear_model.intercept_.item()
beta1, beta2 = linear_model.coef_[:,0].item(), linear_model.coef_[:,1].item()
print("Coefficients:", (beta0,beta1, beta2))
print("R^2:", linear_model.score(X.numpy(), target.numpy()))


def calculate_conditional_probablility(coef,intercept,mue,std):
    pass
