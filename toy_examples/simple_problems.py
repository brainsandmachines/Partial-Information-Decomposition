import admUI
import numpy as np
from admUI import admUI_numpy
from dit.shannon import conditional_entropy as dit_conditional_entropy
from dit import Distribution
from itertools import product

def bern_pair_as_channels(p1=0.5, p2=0.5):
    """
    Convert two independent Bernoulli(p1), Bernoulli(p2) into
    PXgS, PYgS, PS format compatible with computeQUI_numpy.
    """
    # Define joint distribution P(X1, X2)
    probs = {
        (0,0): (1-p1)*(1-p2),
        (0,1): (1-p1)*p2,
        (1,0): p1*(1-p2),
        (1,1): p1*p2,
    }

    # There are 4 possible states of S, each representing (x1, x2)
    states = [(0,0), (0,1), (1,0), (1,1)]

    # Prior over S (column vector)
    PS = np.array([[probs[s]] for s in states])

    # Channel PX1|S : rows = S, cols = values of X1
    PX1gS = np.zeros((4,2))
    for i,(x1,x2) in enumerate(states):
        PX1gS[i, x1] = 1.0

    # Channel PX2|S : rows = S, cols = values of X2
    PX2gS = np.zeros((4,2))
    for i,(x1,x2) in enumerate(states):
        PX2gS[i, x2] = 1.0

    return PX1gS, PX2gS, PS


PXgS = np.array([[ 2./3,  0.],
                    [ 1./3,  1.]])
PYgS = PXgS
PS = np.array([[ 0.75], [ 0.25]])
Q = admUI_numpy.computeQUI_numpy(PXgS, PYgS, PS)
print(PXgS.shape)
print(PYgS.shape)
print(PS.shape)
print(Q.shape)
print(Q)

PX1gS, PX2gS, PS = bern_pair_as_channels()
print(PX1gS.shape)
print(PX2gS.shape)
print(PS)
Q_min = admUI_numpy.computeQUI_numpy(PX1gS.T, PX2gS.T, PS)
print("QUI for two independent Bernoulli(0.5) sources:", Q_min)

print('Calculating U')
def dist_from_tensor(Q, names=('S','X','Y')):
    Q = np.asarray(Q, float)
    assert np.isclose(Q.sum(), 1.0), "Q must sum to 1"
    nS, nX1, nX2 = Q.shape
    outcomes = list(product(range(nS), range(nX1), range(nX2)))  # (s,x1,x2)
    pmf = Q.reshape(-1)
    d = Distribution(outcomes, pmf)
    d.set_rv_names(list(names))
    return d

Q = dist_from_tensor(Q)
U_M1 = dit_conditional_entropy(Q, 'S', 'X') + dit_conditional_entropy(Q, 'X', 'Y') - dit_conditional_entropy(Q, 'SX', 'Y')
print(U_M1)