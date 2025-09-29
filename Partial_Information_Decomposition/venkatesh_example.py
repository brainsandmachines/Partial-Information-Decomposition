import numpy as np
import matplotlib.pyplot as plt
from Venkatesh_broja import broja_venkatesh
from typing import Callable, Any,Optional
from torch import rand, randn
from torch.distributions import Distribution, Normal 


def PID_BROJA(func,vec_size,num_obsv):

    M1, M2, T = func(vec_size,num_obsv)
    M1, M2, T = M1.numpy(), M2.numpy(), T.numpy()
    broja = broja_venkatesh(M1, M2, T, vec_size,bias=True)
    decomposition_values = broja.compute_decomposition()
    return decomposition_values


def pure_unique(vec_size,num_obsv):
    dist = Normal(loc=0.0, scale=1.0)
    T = dist.sample((num_obsv, vec_size))

    M1 = T + dist.sample((num_obsv, vec_size))
    M2 = dist.sample((num_obsv, vec_size))
    return M1, M2, T

def pure_redundant(vec_size,num_obsv):
    dist = Normal(loc=0.0, scale=1.0)
    T = dist.sample((num_obsv, vec_size))

    dist_noise = Normal(loc=0.0, scale=0.1)
    noise = dist_noise.sample((num_obsv, vec_size))
    M1 = T + noise
    M2 = T + noise
    return M1, M2, T

def pure_synargy(vec_size,num_obsv,sigma=1500):
    dist_T = Normal(loc=0.0, scale=1)
    dist_noise = Normal(loc=0.0, scale=sigma)
    T = dist_T.sample((num_obsv, vec_size))
    noise = dist_noise.sample((num_obsv, vec_size))
    M1 = T + noise
    M2 = noise
    return M1, M2, T


def run_exp(PID,goal_func,trial_num,vec_size,num_obsv):
    Unique_M1_all = 0
    Unique_M2_all = 0
    Redundant_M1_M2_all = 0
    Synergy_M1_M2_all = 0

    for i in range(trial_num):
        results = PID(goal_func,vec_size,num_obsv)
        if i % 10 == 0:
            print(f"Trial {i+1}: {results}")
        Unique_M1_all += results['Unique_M1']
        Unique_M2_all += results['Unique_M2']
        Redundant_M1_M2_all += results['Redundant_M1_M2']
        Synergy_M1_M2_all += results['Synergy_M1_M2']

    Unique_M1_avg = Unique_M1_all / trial_num
    Unique_M2_avg = Unique_M2_all / trial_num
    Redundant_M1_M2_avg = Redundant_M1_M2_all / trial_num
    Synergy_M1_M2_avg = Synergy_M1_M2_all / trial_num
    print(f"\n================= Average Results =================")
    print(f"\nAverage Unique_M1: {Unique_M1_avg}")
    print(f"\nAverage Unique_M2: {Unique_M2_avg}")
    print(f"\nAverage Redundant_M1_M2: {Redundant_M1_M2_avg}")
    print(f"\nAverage Synergy_M1_M2: {Synergy_M1_M2_avg}")
    return

if __name__ == "__main__":
    num_obsv = 3000
    vec_size = 1

    #pure_unique_ = PID_BROJA(pure_unique,vec_size,num_obsv)
    
    #pure_redundant_ = PID_BROJA(pure_redundant,vec_size,num_obsv)

    #pure_synargy_ = PID_BROJA(pure_synargy,vec_size,num_obsv)

    run_exp(PID_BROJA,pure_redundant,50,vec_size,num_obsv)
