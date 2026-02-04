import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from nilearn import datasets, plotting
from PIL import Image
from pathlib import Path
import pandas as pd


def check_file_exists(file_path):
    """Check if a file exists at the given path.
    if it exists change it's name by adding a number at the end.

    Args:
        file_path (str): The path to the file."""
    
    base, ext = os.path.splitext(file_path)
    counter = 1
    new_file_path = file_path
    while os.path.exists(new_file_path):
        new_file_path = f"{base}_{counter}{ext}"
        counter += 1
    return new_file_path

def check_folder_exists(folder_path):
    """Check if a folder exists at the given path.
    if it doesn't exist, create it.

    Args:
        folder_path (str): The path to the folder."""
    base, ext = os.path.splitext(folder_path)
    counter = 1
    new_folder_path = folder_path
    while os.path.exists(new_folder_path):
        new_folder_path = f"{base}_{counter}{ext}"
        counter += 1
    os.makedirs(new_folder_path)
    return new_folder_path
    
def create_permuation(list_to_permute):
    """This function take a range of indices 
    and return a permuted version of it.
    Args:    
        list_to_permute (list,np.array,torch.Tensor): list to permute
        
    Returns:
        permuted_list (list,np.array,torch.Tensor): permuted list
    """
    permute_type = type(list_to_permute)

    if not isinstance(list_to_permute, (np.ndarray)):
        list_to_permute = np.array(list_to_permute)

    list_to_permute = list_to_permute[np.random.permutation(len(list_to_permute))]

    return permute_type(list_to_permute) 



class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

def check_equal_type_invariance(a,b) -> bool:
    """Check if two inputs are equal in value and type invariance.
    
    Args:
        a: First input.
        b: Second input."""
    if type(a) == type(b):
        return a == b
    # if b is None:
    #     
    element_a = a[0] if isinstance(a, (list, np.ndarray, torch.Tensor)) and len(a) > 0 else a
    
    if pd.isna(element_a) and pd.isna(b):
        return True
    try:
        b_converted = type(element_a)(b)
        return b_converted == a
    except (ValueError, TypeError):
        return False

def meta_exists(meta_data: dict, csv_path) -> bool:
    """
    Check whether a row with identical meta_data already exists in a CSV file. 
    it is invariant to type differences (e.g., int vs float vs str).
    
    Args:
        meta_data (dict): hyperparameter dictionary
        csv_path (Path or str): path to csv file
    
    Returns:
        bool: True if meta_data already exists, False otherwise
    """
    if not csv_path.exists():
        return False

    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    if df.empty:
        return False


    cols = meta_data.keys()

    
    for record in records:
        mask_list = []
        is_equal = False
        for col in cols:
            is_equal = check_equal_type_invariance(record[col], meta_data[col])
            mask_list.append(is_equal)
        if all(mask_list):
            return True
    else:
        return False
            

