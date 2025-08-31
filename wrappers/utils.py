import torch
import matplotlib.pyplot as plt
import numpy as np
import typing
from torch import rand, randn
from torch.distributions import Distribution, Normal 



def generate_random_variables(func,num_obsv,vec_size,dist,*random_variables):
    """This function will generate i.i.d random variables from a given distribution.
    args: 
    num_samples: Number of observations for each random variable
    dist: A torch.distributions.Distribution object
    *random_variables: A list to hold the generated random variables
    func: A function to apply to the generated random variables
    """
    assert isinstance(dist,Distribution), "Argument must be a torch.distributions.Distribution object"

    rv_dict = {}
    for rv in random_variables:
        rv_sample = dist.sample((num_obsv,vec_size))
        rv_dict[rv] = rv_sample

    return func(rv_dict)