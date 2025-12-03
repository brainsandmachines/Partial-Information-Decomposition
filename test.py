""""A file to test random stuffs"""
"""========================================="""



import dit
from dit.example_dists import Xor,Unq
from dit.pid import PID_dep

d = Xor()
d.set_rv_names(['X','Y','Z'])

pid = PID_dep(dist=d,sources=['Y','Z'], target='X')
unique = pid._measure(d, sources=('Y','Z'), target=('X'))
print("dit version:", dit.__version__)
print("BROJA unique information:", unique)
