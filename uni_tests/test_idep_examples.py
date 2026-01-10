import numpy as np
import torch
import pytest
from sklearn.linear_model import LinearRegression
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))  
from Partial_Information_Decomposition.Idep_univariabe_gauss import Idep_univariate_gauss

