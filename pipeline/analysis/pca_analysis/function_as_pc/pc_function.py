import numpy as np 
import torch
from pathlib import Path
import sys
from typing import Any
import yaml
root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from full_OTC.otc_experiment import run_otc_experiment
from pipeline_phases.sources_target_features import prepare_sources, prepare_target
from pipeline_phases.feature_manipulations import prepare_ridge_target,ridge_predict_shared
from pipeline.pid_pipeline import PIDPipeline
from pipeline.ridge_find_alpha.find_alpha import find_alpha_per_pc
from sklearn.linear_model import Ridge







def pc_function_analysis(config,functions:dict,model1_name:list,model2_name:list,pc_path:Path,hdf_path:Path,pkl_info_path:Path,neural_data_path:Path) -> dict[str, Any]:







    #Load data
    target_context = prepare_target(hdf_path, pkl_info_path, neural_data_path)
    shared_maked = target_context['shared1000_subj']
    unique_masked = ~shared_maked
    #Devide data into shared and unique components
    unique_neural,shared_neural,_ = prepare_ridge_target(target_context['neural_data'],target_context,pc_path)

    
    pipeline = PIDPipeline(functions)
    
    for model_1 in model1_name:
        print(f"\nRunning PID with Source 1: {model_1} 😀")
        source1_raw = None
        for model_2 in model2_name:
            sources = prepare_sources(model_1,model_2,target_context,pc_path)
            sources_layer = pipeline.functions.choose_layer(sources, **(config['choose_layer_kwargs'] or {}))['X1']

            if source1_raw is None:
                source1_raw = pipeline.functions.feature_extraction(
                    sources["X1"],
                    sources_layer["X1"],
                    target_context,
                    config['feature_extraction_kwargs']['target_context'],
                )

            source2_raw = pipeline.functions.feature_extraction(
                sources["X2"],
                sources_layer["X2"],
                target_context,
                config['feature_extraction_kwargs']['target_context'])
            
            source1_shared = source1_raw[shared_maked]
            source2_shared = source2_raw[shared_maked]

            source1_unique = source1_raw[unique_masked]
            source2_unique = source2_raw[unique_masked]
            for f in len(range(shared_neural.shape[1])):
                print(f"Running PID with Source 1: {model_1} and Source 2: {model_2} for PC {f} 😀")
                target_f = shared_neural[:,f]
                ridge = config['feature_manipulation_kwargs']['ridge']

                if ridge:
                    print(f"Running Ridge Regression for PC {f} 😀")
                    _, source1_model = find_alpha_per_pc(
                        source1_shared,
                        target_f)
                    _, source2_model = find_alpha_per_pc(
                        source2_shared,
                        target_f)
                    
                    source_1_pred = source1_model.predict(source1_raw)
                    source_2_pred = source2_model.predict(source2_raw)

                else:
                    source_1_pred = source1_unique
                    source_2_pred = source2_unique

                pid_results = pipeline.functions.pid_calculation(
                    target_f,
                    source_1_pred,
                    source_2_pred,
                    **(config['pid_kwargs'] or {}),
                )

            
            

            
            



    

    

