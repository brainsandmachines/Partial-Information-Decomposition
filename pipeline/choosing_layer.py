import torch
import numpy as np
import yaml
import sys
from pathlib import Path


""""A file with functions help to choose the layer to each source"""



def random_layer_selection(n_layers):
    """Choose a random layer index from the available layers.
    
    Input:
        - n_layers: total number of layers available (integer)
        
    Output:
        - layer_idx: randomly selected layer index (integer in range [0, n_layers-1])
    """

    layer_idx = np.random.randint(0, n_layers)
    return layer_idx


def specific_index_layer_selection(layer_names, index):
    """Choose a specific layer index from the available layers.
    
    Input:
        - layer_names: list of available layer names
        - index: specific layer index to select (integer in range [0, len(layer_names)-1])"""
    

    return layer_names[index]




def best_layer(layer_names):
    """This function chooses the most predictive layer 
    for a given source  on a given subject"""
    pass