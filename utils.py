import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from nilearn import datasets, plotting
from PIL import Image
from pathlib import Path



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
