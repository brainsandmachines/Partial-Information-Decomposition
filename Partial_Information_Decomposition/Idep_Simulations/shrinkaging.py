import torch
import numpy as np 
from sklearn.covariance import LedoitWolf,ShrunkCovariance
from Simulation_utils import *
from wrapper_M7_M8_models import create_m7_cov, make_random_true_cov



def ledoit_wolf_cov(X):
    lw = LedoitWolf().fit(X)
    return lw.covariance_


def shrunk_cov(X, alpha=0.1):
    if type(alpha) == list:
        sc_list = []
        for shrinkage in alpha:
            sc = ShrunkCovariance(shrinkage=shrinkage).fit(X)
            sc_list.append(sc.covariance_)
        return sc_list
    else:
        sc = ShrunkCovariance(shrinkage=alpha).fit(X)
    return sc.covariance_



def shrinkage_covariance(X, method='ledoit_wolf', alpha=0.1):
    if method == 'ledoit_wolf':
        return ledoit_wolf_cov(X)
    elif method == 'shrunk_cov':
        return shrunk_cov(X, alpha)
    else:
        raise ValueError("Unsupported method. Use 'ledoit_wolf' or 'shrunk_cov'.")
    


def shrinkage_m7_m8_simulation(config:dict,evluation_func:callable = None,data=None):
    """This function takes true covriances and return a smaple with shrinkage covariance estimation for both M7 and M8 models. 
    It also returns the true covariances for both models. The function can be used to evaluate the performance of shrinkage covariance estimation methods in the context of M7 and M8 models."""
    n0 = config['n0'] #X0 dim
    n1 = config['n1'] #X1 dim
    n2 = config['n2'] #T dim
    seed = config['seed'] 
    device = config.get('device', 'cpu')
    white_normalize = config.get('white_normalize', True)
    alphas = config['alpha_list'] #List of alpha values to test for shrinkage
    rng = torch.Generator(device=config['device']).manual_seed(seed)

    #Set true covrainces for m7 and m8
    true_cov_m8, true_cov_m7 = data

    sample_cov_m8 = sample_data_from_cov(config, true_cov_m8, rng)
    ledoit_wolf_cov_m8 = ledoit_wolf_cov(sample_cov_m8)
    shrunk_cov_m8 = shrunk_cov(sample_cov_m8, alphas)

    sample_cov_m7 = create_m7_cov(config,sample_cov_m8,whitening_normalize=white_normalize)
    ledoit_wolf_cov_m7 = ledoit_wolf_cov(sample_cov_m7)
    shrunk_cov_m7 = shrunk_cov(sample_cov_m7, alphas)

    return {'M8': {'true_cov': true_cov_m8, 'sample_cov': sample_cov_m8, 'ledoit_wolf_cov': ledoit_wolf_cov_m8, 'shrunk_cov': shrunk_cov_m8},
            'M7': {'true_cov': true_cov_m7, 'sample_cov': sample_cov_m7, 'ledoit_wolf_cov': ledoit_wolf_cov_m7, 'shrunk_cov': shrunk_cov_m7}}




def evaluate_shrinkage(config:dict,results_dict:dict):
    """Will evaluate the preformance of the shrinkage covriance according to 
    some evaluation function (e.g. Frobenius norm between the true covariance and the estimated covariance)"""

    evaluation_func = config['evaluation_func']
    evaluation_results = {}
    for model in ['M7', 'M8']:
        true_cov = results_dict[model]['true_cov']
        sample_cov = results_dict[model]['sample_cov']
        ledoit_wolf_cov = results_dict[model]['ledoit_wolf_cov']
        shrunk_cov_list = results_dict[model]['shrunk_cov']
        evaluation_results[model] = {
            'ledoit_wolf': evaluation_func(true_cov, ledoit_wolf_cov),
            'shrunk_cov': [evaluation_func(true_cov, shrunk_cov) for shrunk_cov in shrunk_cov_list],
            'sample_cov': evaluation_func(true_cov, sample_cov)
        }
    return evaluation_results