import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics import r2_score
from itertools import chain, combinations
from typing import List, Tuple, Union



def LinearRegression_fit(X,y):
    model = LinearRegression()
    model.fit(X,y)
    return model