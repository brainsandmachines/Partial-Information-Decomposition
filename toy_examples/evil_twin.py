import torch
import numpy as np











def evil_twin_example(rng,n,p):
    """
    Create an "evil twin" example where X and Y are identical copies of S, but with some noise.
    This should yield a high unique information for both X and Y, and zero shared information.
    """
    #Twin sonic (Unique for both)
    R_so = rng.standard_normal((n, p))
    R_so *= (0.5 / p) ** 0.5 #Var(R_so) = 0.5

    N = rng.standard_normal((n, p))
    N *= (2.5 / p) ** 0.5 #Var(N) = 2.5

    N_t = rng.standard_normal((n, p))

    U1_so = rng.standard_normal((n, p))
    U1_so *= (2.5 / p) ** 0.5 #Var(U1_so) = 2.5

    U2_so = rng.standard_normal((n, p))
    U2_so *= (0.5 / p) ** 0.5 #Var(U2_so) = 0.5

    X1_so = R_so + N + U1_so
    X2_so = R_so + N + U2_so

    T_so = R_so + U1_so + U2_so + N_t



    # Twin Shadow (No unique for X2)
    R_sh = rng.standard_normal((n, p))
    R_sh *= (1 / p) ** 0.5 #Var(R_sh) = 1
    
    N_sh = rng.standard_normal((n, p))
    N_sh *= (2 / p) ** 0.5 #Var(N_sh) = 2
    E1_sh = rng.standard_normal((n, p))
    E1_sh *= (0.5 / p) ** 0.5

    E2_sh = rng.standard_normal((n, p))
    E2_sh *= (0.5 / p) ** 0.5

    U1_sh = rng.standard_normal((n, p))
    U1_sh *= (2 / p) ** 0.5 #Var(U1_sh) = 2

    N_t_sh = rng.standard_normal((n, p))

    X1_sh = R_sh + N_sh + U1_sh + E1_sh
    X2_sh = R_sh + N_sh + E2_sh
    T_sh = R_sh + U1_sh + N_t_sh

    return {'sonic': (X1_so, X2_so, T_so), 'shadow': (X1_sh, X2_sh, T_sh)}


def main():
    rng = np.random.default_rng(0)
    n = 10000
    p = 10
    data = evil_twin_example(rng, n, p)
    