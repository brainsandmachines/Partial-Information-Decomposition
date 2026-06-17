import torch 
import numpy as np
import os 
from pathlib import Path
import sys



from external.mayas_project.old_scripts.features_extraction.final_feature_extraction import extract_features_for_batch

repo_root = Path("/home/ohadshee/Desktop/Thesis_Ohad_Sheelo")
sys.path.append(str(repo_root))

from external.mayas_project.features_and_encoding.feat_ext_and_encoding import DataLoader, ImageDatasetNSD, get_layer_feature_count, get_sparse_projection_gpu, prepare_subject_context, prepare_model_context



"""This file will contatin feature manipulatioin function afer feature extraiction. For example, 
 n_project function or PCA or ICA etc..."""