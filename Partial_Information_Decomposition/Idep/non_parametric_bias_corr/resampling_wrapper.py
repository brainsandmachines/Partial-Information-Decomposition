import torch
import numpy as np
from sklearn.model_selection import LeaveOneOut
from zarr import config
from PID_util import *
import pandas as pd
from joblib import Parallel, delayed
import time
import sys
from pathlib import Path
from Partial_Information_Decomposition.Idep.Idep_Simulations.Simulation_utils import on_covariance
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from Partial_Information_Decomposition.Idep.non_parametric_bias_corr.jackknife import *
from Partial_Information_Decomposition.Idep.non_parametric_bias_corr.bootstrap import *




def resampleing(resample_inputs: dict,rng) -> dict:

    assert resample_inputs['resample_method'] in ['jackknife', 'bootstrap'], "resample method must be either jackknife or bootstrap"

    #Extract resampling parameters
    resample_func = resample_inputs['resample_method']
    data = resample_inputs['data']
    n_samples = resample_inputs['n_samples']


    org_pop,resample_pop = resample_func(data, n_samples, rng)
    return {'org_pop': org_pop, 'resample_pop': resample_pop}


def calculate_statistic(config:dict,calc_func:callable ,population:dict,rng:np.random.Generator) -> dict:
    #Extract data
    seed = config['seed']
    #Calculate statistic on the population
    statistic = calc_func(population, rng)
    return {'statistic': statistic, 'org_pop': population['org_pop'],}


def calculate_bias(config:dict,statistic_dict:dict,bias_func:callable) -> dict:
    #Extract data
    org_pop = statistic_dict['org_pop']
    statistic = statistic_dict['statistic']
    #Calculate bias
    bias = bias_func(statistic)
    return {'bias': bias, 'statistic': statistic, 'org_pop': org_pop}


def bias_resampling(config:dict,calc_func:callable=None) -> dict:
    """This function will calculate the statistics value and it's and will return a dictionary with the following keys:
    
    Input: data: the data to calculate the statistic on (covariance etc )
    config: a dictionary with data

    return:
        results_dict[statistic_key] = {
        'sample': statistic_model['sample'],
        'avg': statistic_model['avg'],
        'std': statistic_model['std'],
        'emp_bias': statistic_model['emp_bias'],
        'corrected_statistic': model_corr_values,
        'ground_truth': statistic_model['ground_truth']
        }
    """
    bias_method = config['bias_method']

    if bias_method[0] == 'jackknife':
        resample_method = jackknife_resample
        bias_func = jackkinfe_func

    elif bias_method[0] == 'bootstrap':
        resample_method = bootstrap_resample
        bias_func = bootstrap_func

    calc_func = config['calc_statistic_func'] if calc_func is None else calc_func

    _,resample_pop,resample_pop_whitened = resample_method(config)

    bias = bias_func(config,resample_pop_whitened,calc_func)

    return bias