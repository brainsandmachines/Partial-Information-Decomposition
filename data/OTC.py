import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import zarr

def load_OTC(subject_id: int, path_to_data: str) -> dict:
    """
    Load OTC fMRI data for a given subject.
    (This function assumes data files are zarr files stored in the specified path.)

    Args:
        subject_id (int): The ID of the subject.
        path_to_data (str): Path to the directory containing the fMRI data files.

    Returns:
        dict: A dictionary containing left and right hemisphere fMRI data.
    """
    
    z = zarr.open(os.path.join(path_to_data, f'subj{subject_id}_OTC_betas.zarr'),mode='r')
    x = torch.from_numpy(np.array(z))

    return x





def main():
    data_dir = '/mnt/data/NSD_data/processed_data/OTC_betas'  # Update this path to your data directory
    subject_id = '01'  # Example subject ID
    fmri_data = load_OTC(subject_id, data_dir)
    print(f"Loaded fMRI data shape for subject {subject_id}: {fmri_data.shape}")

if __name__ == "__main__":
    main()