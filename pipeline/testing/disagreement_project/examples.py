import torch
import numpy as np
import matplotlib.pyplot as plt

"""This file will contain controlled examples of disagreement between the 3 methods. 
Regular PCA, GSN-PCA, and Cross-Decomposition."""






def KPQ_example(seed,n_samples,true_rank,n_repetions,snr):
    """Easy example that all methods should agree on for discovering the true rank of the signal.
    """

    rng = np.random.default_rng(seed)

    signal = rng.normal(size=(n_samples,true_rank))
    signal_repited = np.repeat(signal[:,np.newaxis,:],n_repetions,axis=1)
    noise = rng.normal(size=(n_samples,n_repetions,true_rank))*snr
    responses = signal_repited + noise
    return responses



def LudiPQ_example():
    pass






def Zakum_example():
    pass




if __name__ == "__main__":
    responses = KPQ_example(seed=0,n_samples=100,true_rank=5,n_repetions=3,snr=0.1)
