import torch
import scipy
import numpy as np
from sklearn.model_selection import LeaveOneOut



def jackknife_bias_term(X,X_bar_logdet):
    """This function utilized LeaveOneOut resampling to estimate the bias of the logdet estimator for a given dataset X.
    Args:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        X_bar_logdet (np.ndarray): Logdet of the whole sample.
        
        """
    loo = LeaveOneOut()
    bias_terms = []
    for train_idx, test_idx in loo.split(X):
        X_train = X[train_idx]
        X_test = X[test_idx]
        # Calculate the logdet for the training set
        logdet_train = logdet_estimator(X_train)
        # Calculate the logdet for the test set
        logdet_test = logdet_estimator(X_test)
        # Calculate the bias term
        bias_term = logdet_train - logdet_test
        bias_terms.append(bias_term)
    # Return the mean of the bias terms
    return np.mean(bias_terms)