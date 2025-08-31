from dit import Distribution
from dit.pid import PID_BROJA

def bern_pair_probs(p1=0.5, p2=0.5):
    """Return joint probs P(X1=a, X2=b) for independent Bernoulli(p1), Bernoulli(p2)."""
    return {
        (0,0): (1-p1)*(1-p2),
        (0,1): (1-p1)*p2,
        (1,0): p1*(1-p2),
        (1,1): p1*p2,
    }

def broja_xor(p1=0.5, p2=0.5):
    """
    Exact distribution for XOR:
      M1 = X1, M2 = X2, T = X1 XOR X2
    Outcomes are tuples (M1, M2, T).
    """
    pm = bern_pair_probs(p1, p2)
    supp = []
    pmf  = []
    for (x1,x2), pr in pm.items():
        t = (x1 ^ x2)  # XOR
        supp.append((x1, x2, t))
        pmf.append(pr)
    dist = Distribution(supp, pmf)
    pid  = PID_BROJA(dist, inputs=[[0],[1]], output=[2])
    return dist, pid

def broja_two_bit_copy(p1=0.5, p2=0.5):
    """
    Exact distribution for two-bit copy:
      M1 = X1, M2 = X2, T = (X1, X2)   (literal pair)
    Outcomes are (M1, M2, T_tuple).
    """
    pm = bern_pair_probs(p1, p2)
    supp = []
    pmf  = []
    for (x1,x2), pr in pm.items():
        t = (x1, x2)      # T is the pair
        supp.append((x1, x2, t))
        pmf.append(pr)
    dist = Distribution(supp, pmf)
    pid  = PID_BROJA(dist, inputs=[[0],[1]], output=[2])
    return dist, pid

# --- Examples (uncomment to run) ---
d_xor, pid_xor = broja_xor(0.5, 0.5)
print("XOR BROJA:", pid_xor)
# d_copy, pid_copy = broja_two_bit_copy(0.5, 0.5)
# print("Two-bit copy BROJA:", pid_copy.get_measures())
