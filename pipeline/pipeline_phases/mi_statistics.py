import torch
import numpy as np
import yaml
from pathlib import Path
import sys
from Partial_Information_Decomposition.mi_functions import *
from Partial_Information_Decomposition.PID_util import *


"""This file calcualtes statistic or assertion on PID and MI values"""


def assert_mi(own_mi,pid_mi):
    """This function checks if the mutual information calculated by the PID method is equal 
    to the mutual information calculated by the own method.

    - I(X_1;T)
    - I(X_2;T)
    - I(X_1,X_2;T)
    Input:
        - own_mi: mutual information calculated by the own method (float)
        - pid_mi: mutual information calculated by the PID method (float)"""
    

    mi_names = ['bi_mi_1_t','bi_mi_2_t','tri_mi']
    mi_real_names = ['I(X_1;T)','I(X_2;T)','I(X_1,X_2;T)']

    for mi_name, mi_real_name in zip(mi_names, mi_real_names):
        if not np.isclose(own_mi[mi_name], pid_mi[mi_name], atol=1e-5):
            print(f"Error: Mutual information {mi_real_name} calculated by own method ({own_mi[mi_name]}) is not equal to mutual information calculated by PID method ({pid_mi[mi_name]}).")
        else:
            print(f"Mutual information {mi_real_name} calculated by own method is equal to mutual information calculated by PID method.")

    return True


