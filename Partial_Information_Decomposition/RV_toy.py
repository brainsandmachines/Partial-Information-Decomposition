import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Any,Optional
from torch import rand, randn
from torch.distributions import Distribution, Normal 



def generate_random_variables(func: Optional[Callable], num_obsv: int, vec_size: int, dist: Distribution, *random_variables: str) -> Any:
    """This function will generate i.i.d random variables from a given distribution.
    args: 
    num_samples: Number of observations for each random variable
    vec_size: Size of the vector for each random variable
    dist: A torch.distributions.Distribution object
    *random_variables: A list with a name to the random variables
    func: A function to apply to the generated random variables
    
    the purpose of this function is just to generate random variables as many as
    the  wishes and to apply a function to them if needed. else returns the random variables
    
    returns: 
    what func outputs
    """
    assert isinstance(dist,Distribution), "Argument must be a torch.distributions.Distribution object"

    rv_dict = {}
    for rv in random_variables:
        rv_sample = dist.sample((num_obsv,vec_size))
        rv_dict[rv] = rv_sample

    if func is not None:
        func_output = func(rv_dict)
        return func_output, rv_dict

    return rv_dict




def main():
    num_obsv = 1
    vec_size = 1
    dist = Normal(loc=0.0, scale=1.0)
    random_variables = ['S1', 'S2','S12','N1', 'N2','NT']

    def toy_example(rv_dict):
        m1 = rv_dict['S12'] + rv_dict['S1'] + rv_dict['N1']
        m2 = rv_dict['S12'] + rv_dict['S2'] + rv_dict['N2']
        target = rv_dict['S12'] + rv_dict['S1'] + rv_dict['S2'] +rv_dict['NT']
        return {'M1':m1, 'M2':m2, 'target':target}

    result, _ = generate_random_variables(toy_example, num_obsv,vec_size ,dist, *random_variables)
    print(result)


if __name__ == "__main__":
    main()