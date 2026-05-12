import torch
import numpy as np











def evil_twin_example(rng,n,p):
    """
    Create an "evil twin" example where X and Y are identical copies of S, but with some noise.
    This should yield a high unique information for both X and Y, and zero shared information.
    """
    #Twin sonic (Unique for both)
    R_s = rng.standard_normal((n, p))
    R_s *= (0.5 / p) ** 0.5 #Var(R_s) = 0.5

    N = rng.standard_normal((n, p))
    N *= (0.5 / p) ** 0.5 #Var(N) = 0.5

    N_t = rng.standard_normal((n, p))

    U1 = rng.standard_normal((n, p))

    U2 = rng.standard_normal((n, p))

    X1 = R_s + N + U1
    X2 = R_s + N + U2

    T = R_s + U1 + U2 + N_t



    # Twin Shadow (No unique for X2)
    