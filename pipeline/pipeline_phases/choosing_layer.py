import csv

import torch
import numpy as np
import yaml
import sys
from pathlib import Path
import pandas as pd

""""A file with functions help to choose the layer to each source"""


def choose_layer_function(layer_func_name:str):
    LAYER_FUNC_NAME = ['random_layer_selection', 'specific_index_layer_selection', 'voxel_best_layer']


    if layer_func_name not in LAYER_FUNC_NAME:
        raise ValueError(f"Invalid layer function name: {layer_func_name}. Must be one of {LAYER_FUNC_NAME}.")

    else:
        if layer_func_name == 'random_layer_selection':
            return random_layer_selection
        elif layer_func_name == 'specific_index_layer_selection':
            return specific_index_layer_selection
        elif layer_func_name == 'voxel_best_layer':
            return voxel_best_layer
        
    
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
        - index: specific layer index to select (integer in range [0, len(layer_names)-1])
        
        
    Output:
        - layer_name: name of the selected layer (string)
        """
    
    
    

    return layer_names[index]





def voxel_best_layer(voxel_index: int = None, index_layer: int = None, path_to_results: str = None) -> dict:
    """Choose the best model layer for one voxel, or a representative voxel for one layer.

    Input:
        - voxel_index: voxel index to look up (integer or None).
        - index_layer: best layer index to look up (integer or None).
        - path_to_results: path to a CSV file with columns 'voxel_index' and
          'best_layer_index' (string or None when both indexes are provided).

    Output:
        - dict with keys 'v' and 'l', where 'v' is the selected voxel index
          and 'l' is the selected best layer index. Missing or failed lookups
          return {'v': None, 'l': None}.
    """

    if voxel_index is not None and index_layer is not None:
        return {'v': int(voxel_index), 'l': int(index_layer)}

    if path_to_results is None:
        print("No path_to_results provided for layer lookup.")
        return {'v': None, 'l': None}

    try:
        df = pd.read_csv(path_to_results)
        required_columns = {'voxel_index', 'best_layer_index'}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            print(f"Missing required columns in results CSV: {sorted(missing_columns)}")
            return {'v': None, 'l': None}

        if voxel_index is not None:
            voxel_row = df[df['voxel_index'] == voxel_index]
            if voxel_row.empty:
                print(f"No results found for voxel index {voxel_index}")
                return {'v': None, 'l': None}
            index_layer = voxel_row['best_layer_index'].iloc[0]

        elif index_layer is not None:
            layer_row = df[df['best_layer_index'] == index_layer]
            if layer_row.empty:
                print(f"No results found for layer index {index_layer}")
                return {'v': None, 'l': None}
            voxel_index = layer_row['voxel_index'].iloc[0]

        else:
            print("No voxel index or layer index provided. Choosing random layer.")
            unique_layers = df['best_layer_index'].dropna().unique()
            if len(unique_layers) == 0:
                print("No layer indexes found in results CSV.")
                return {'v': None, 'l': None}
            index_layer = np.random.choice(unique_layers)
            layer_row = df[df['best_layer_index'] == index_layer]
            voxel_index = layer_row['voxel_index'].iloc[0]

    except Exception as e:
        print(f"Error loading best layer selection results: {e}")
        return {'v': None, 'l': None}

    return {'v': int(voxel_index), 'l': int(index_layer)}

    

    
