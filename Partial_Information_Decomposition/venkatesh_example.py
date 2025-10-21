import numpy as np
import matplotlib.pyplot as plt
from Venkatesh_broja import broja_venkatesh
from typing import Callable, Any,Optional
from torch import rand, randn
from torch.distributions import Distribution, Normal 
from U_LB import unique_lb_gauss

def PID_BROJA(func,vec_size,num_obsv,bias:Optional[bool]=False):

    M1, M2, T = func(vec_size,num_obsv)
    M1, M2, T = M1.numpy(), M2.numpy(), T.numpy()
    broja = broja_venkatesh(M1, M2, T, vec_size,bias=bias)
    decomposition_values = broja.compute_decomposition()
    return decomposition_values

def PID_ULB(func,vec_size,num_obsv):
    M1, M2, T = func(vec_size,num_obsv)
    ulb = unique_lb_gauss(T,M1,M2)
    ULB_M1, ULB_M2 = ulb.unique_lb_gauss()
    return ULB_M1, ULB_M2


def pure_unique(vec_size,num_obsv):
    dist = Normal(loc=0.0, scale=1.0)
    T = dist.sample((num_obsv, vec_size))
    noise_1 = Normal(loc=0.0, scale=1.0)
    noise_2 = Normal(loc=0.0, scale=1.0)

    M1 = T + noise_1.sample((num_obsv, vec_size))
    M2 = noise_2.sample((num_obsv, vec_size))
    return M1, M2, T

def pure_redundant(vec_size,num_obsv):
    dist = Normal(loc=0.0, scale=1.0)
    T = dist.sample((num_obsv, vec_size))

    dist_noise = Normal(loc=0.0, scale=0.1)
    noise = dist_noise.sample((num_obsv, vec_size))
    M1 = T + noise
    M2 = T + noise
    return M1, M2, T

def pure_synargy(vec_size,num_obsv,sigma=15000):
    dist_T = Normal(loc=0.0, scale=1)
    dist_noise_1 = Normal(loc=0.0, scale=sigma)
    dist_noise_2 = Normal(loc=0.0, scale=sigma)
    T = dist_T.sample((num_obsv, vec_size))
    noise_1 = dist_noise_1.sample((num_obsv, vec_size))
    noise_2 = dist_noise_2.sample((num_obsv, vec_size))
    M1 = T +  noise_1
    M2 = noise_2
    return M1, M2, T

def both_unique(vec_size,num_obsv):
    dist = Normal(loc=0.0, scale=1.0)
    T = dist.sample((num_obsv, vec_size))
    noise_1 = Normal(loc=0.0, scale=1.0)
    noise_2 = Normal(loc=0.0, scale=1.0)

    M1 = T + noise_1.sample((num_obsv, vec_size))
    M2 = T + noise_2.sample((num_obsv, vec_size))
    return M1, M2, T

def run_exp(PID,goal_func,trial_num,vec_size,num_obsv,bias:Optional[bool]=False):
    Unique_M1_all = 0
    Unique_M2_all = 0
    Redundant_M1_M2_all = 0
    Synergy_M1_M2_all = 0

    for i in range(trial_num):
        results = PID(goal_func,vec_size,num_obsv,bias=True)
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
    num_obsv = 50000
    vec_size = 1

    #pure_unique_ = PID_BROJA(pure_unique,vec_size,num_obsv,bias=True)
    
    #pure_redundant_ = PID_BROJA(pure_redundant,vec_size,num_obsv,bias=True)

    #pure_synargy_ = PID_BROJA(pure_synargy,vec_size,num_obsv,bias=True) #NOTE: For this to work well, maybe I need to do linear regression first

    #run_exp(PID_BROJA,pure_redundant,500,vec_size,num_obsv,bias=True)

    #pure_synargy_ulb = PID_ULB(pure_synargy,vec_size,num_obsv) #Both unique_lb should be close to 0 

    #pure_unique_ulb = PID_ULB(pure_unique,vec_size,num_obsv) #One unique greater than 0, other close to 0

    #pure_redundant_ulb = PID_ULB(pure_redundant,vec_size,num_obsv)

    both_unique_ulb = PID_ULB(both_unique,vec_size=50,num_obsv=num_obsv) #Both unique greater than 0