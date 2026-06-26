import torch 
import numpy as np
import yaml
from pathlib import Path
import sys





"""This file will contain the preprocessing layer of the pipeline. For example: 
(Gaussian copulas, normalizing, standardizing, etc...)

For now it is empty, but in the future it will contain functions that will be applied to the features after feature extraction and before feature manipulation."""




def permute_rv(target,source1,source2,source1_perm=False,source2_perm=False,target_perm=False,rng_seed=56):
    """Permute the random variable rv according to the configuration provided in config.
    If rv is a tuple, all blocks are permuted with the same permutation.
    X is kept fixed, so any internal structure within X is preserved.
    
    input: 
        Experiment configuration
        rvs: tuple of random variables (source1, source2, target)
        source1: bool, whether to permute source1
        source2: bool, whether to permute source2
        target: bool, whether to permute target
    output:
        permuted_rvs: tuple of permuted random variables (source1, source2, target)
    """
    
    print("Running permuation on with RNG seed: ", rng_seed)
    n = target.shape[0]
    rng = torch.Generator()
    rng.manual_seed(rng_seed)

    idx = torch.randperm(n,generator=rng)

    source1 = source1[idx] if source1_perm and source1 is not None else source1
    source2 = source2[idx] if source2_perm and source2 is not None else source2
    target = target[idx] if target_perm and target is not None else target

    return source1 ,source2,target
    