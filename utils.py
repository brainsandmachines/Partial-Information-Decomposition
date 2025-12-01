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
    