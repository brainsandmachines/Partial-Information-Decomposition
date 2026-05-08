import sys
from zipfile import Path
from scipy.special import digamma
from networkx import constraint
import torch
import numpy as np
import matplotlib.pyplot as plt 
from torch.distributions import Normal
from sklearn.linear_model import LinearRegression
from sklearn.covariance import LedoitWolf

if __name__ != "__main__":
    from .PID_util import create_cov_matrix,whiten_block,block_singularity_check
else:
    from PID_util import create_cov_matrix, cond_cov, standardize, assert_full_rank,singularity_report
from typing import Optional


import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))  
from Partial_Information_Decomposition.bias_functions import bias_func,logdet_wishart_bias

"""This files implement the Idep univariate source and target method for univariate gaussian variables as described in:
Ince et al. 2018: (Exact Partial Information Decompositions for Gaussian Systems Based on Dependency Constraints)"""


torch.set_default_dtype(torch.float64)
class Idep_multivariate_gauss:
    def __init__(self, rng=None,sources: Optional[list] = None, targets:Optional[list] = None,cov_matrix: Optional[torch.tensor]=None,base_e: bool =True,bias_correction: bool = False):
        """Initialize the Idep multivariate gaussian class

        input: M1,M2,T are torch tensors of shape (N,P)
        N is the number of observations
        P is the number of variables in each observation

        """
        self.base_e = base_e  # Default to natural logarithm
        self.rng = rng

        self.M1 = sources[0] if sources is not None else None
        self.M2 = sources[1] if sources is not None else None
        self.T = targets[0] if targets is not None else None
        self.cov_dict = None
        self.N = self.M1.shape[0] if self.M1 is not None else None
        df = self.N - 1  if self.M1 is not None else None
        self.dim_m1 = self.M1.shape[1] if self.M1 is not None else 0
        self.dim_m2 = self.M2.shape[1] if self.M2 is not None else 0
        self.dim_t = self.T.shape[1] if self.T is not None else 0
        self.bias_correction = bias_correction
        self.I0 = torch.eye(self.dim_m1)
        self.I1 = torch.eye(self.dim_m2)
        self.I2 = torch.eye(self.dim_t)

        if self.M1 is not None and self.M2 is not None and self.T is not None:
            self.cov_dict = create_cov_matrix(rvs=[self.M1,self.M2,self.T])
            self.cov_matrix = self.cov_dict['full_cov']
            #assert_full_rank(self.cov_matrix)
        elif cov_matrix is not None:
            self.cov_matrix = cov_matrix

        if self.cov_dict is not None:
            self.sigma00 = self.cov_dict['cov_x1']
            self.sigma11 = self.cov_dict['cov_x2']
            self.sigma22 = self.cov_dict['cov_xt']
            self.sigma01 = self.cov_dict['cross_x1_x2']
            self.sigma02 = self.cov_dict['cross_x1_xt']
            self.sigma12 = self.cov_dict['cross_x2_xt']


            self.P = whiten_block(self.sigma00, self.sigma01, self.sigma11)
            self.Q = whiten_block(self.sigma00, self.sigma02, self.sigma22)
            self.R = whiten_block(self.sigma11, self.sigma12, self.sigma22)


            assert self.P.shape == (self.dim_m1,self.dim_m2), f"Covariance matrix dimensions {self.P.shape} do not match the provided source dimensions: {self.dim_m1,self.dim_m2}."
            assert self.Q.shape == (self.dim_m1,self.dim_t), f"Covariance matrix dimensions {self.Q.shape} do not match the provided source and target dimensions: {self.dim_m1,self.dim_t}."
            assert self.R.shape == (self.dim_m2,self.dim_t), f"Covariance matrix dimensions {self.R.shape} do not match the provided source and target dimensions: {self.dim_m2,self.dim_t}."
            assert self.dim_m1 + self.dim_m2 + self.dim_t == self.cov_matrix.shape[0], f"Covariance matrix dimensions {self.cov_matrix.shape} do not match the provided source and target dimensions: {self.dim_m1 + self.dim_m2 + self.dim_t}."
            
            # Singularity check for P,Q and R blocks
            p = (torch.eye(self.P.shape[0]) - self.P @ self.P.T).detach().cpu().numpy()
            q = (torch.eye(self.Q.shape[0]) - self.Q @ self.Q.T).detach().cpu().numpy()
            r = (torch.eye(self.R.shape[0]) - self.R @ self.R.T).detach().cpu().numpy()
            block_singularity_check(np.array([p,q,r]))
            block_singularity_check(np.array([p.T,q.T,r.T]))
        if self.bias_correction :
            print("Calculating bias correction terms...")
            config_m8 = {
                'model': 'M8',
                'rng': self.rng,
                'n0': self.dim_m1,
                'n1': self.dim_m2,
                'n2': self.dim_t,
                'n_samples': self.N,
                'X1': self.M1,
                'X2': self.M2,
                'T': self.T,
                'device': self.M1.device if self.M1 is not None else 'cpu',
                'n_perm': 20
            }
            config_m7 = config_m8.copy()
            config_m7['model'] = 'M7'
            self.m8_bias = bias_func(config_m8,'M8')
            self.m7_bias = bias_func(config_m7,'M7')
        else: 
            self.m8_bias = {'j':0,'k':0} 
            self.m7_bias = {'i':0,'h':0}
            
        self.I_dep_values = {}
        self.PID_values = {}


            
        

    def log_base(self,x:Optional[torch.Tensor]) -> torch.Tensor:
        if self.base_e:
            return torch.log(x)
        LN2 = torch.log(torch.tensor(2.0, dtype=torch.float64))
        return torch.log(x) / LN2
    

    def create_model_M(self,block1:Optional[torch.tensor]=None,block2:Optional[torch.tensor]=None,block3:Optional[torch.tensor]=None) -> torch.tensor:
        """This function will create the dependency matrix for the given blocks
        input: 
        block (1,2,3) is a torch tensor of shape (d,d) (Defined byu the paper as P or Q or R or a multiplication of them)
        
        output: a torch tensor of shape (d,d)
        """

        M = torch.block_diag(self.I0, self.I1, self.I2)
        
        if block1 is not None:
            M[:self.dim_m1, self.dim_m1:self.dim_m1 + self.dim_m2] = block1
            M[self.dim_m1:self.dim_m1 + self.dim_m2, :self.dim_m1] = block1.T
        if block2 is not None:
            M[:self.dim_m1, self.dim_m1 + self.dim_m2:] = block2
            M[self.dim_m1 + self.dim_m2:, :self.dim_m1] = block2.T
        if block3 is not None:
            M[self.dim_m1:self.dim_m1 + self.dim_m2, self.dim_m1 + self.dim_m2:] = block3
            M[self.dim_m1 + self.dim_m2:, self.dim_m1:self.dim_m1 + self.dim_m2] = block3.T

        assert M.shape == (self.dim_m1 + self.dim_m2 + self.dim_t, self.dim_m1 + self.dim_m2 + self.dim_t), f"Created matrix shape {M.shape} does not match expected shape {(self.dim_m1 + self.dim_m2 + self.dim_t, self.dim_m1 + self.dim_m2 + self.dim_t)}."
        #assert_full_rank(M)
        return M


    def dependency_matrix(self,constraints: list,cov_matrix: Optional[torch.tensor]=None,cov_dict: Optional[dict]=None)-> dict:
        """This function will create the dependency matrix for the given constraint

        input: cov_matrix is a torch tensor of shape (d,d)
        d is the dimension of each observation.

        cov_dict is a dictionary containing the covariance matrices of the variables

        constraint is a string indicating the constraint to be applied
        'M1T' : M1 and T are dependent
        'M2T' : M2 and T are dependent
        'M1M2' : M1 and M2 are dependent

        output: a torch tensor of shape (d,d)
        dependency matrix"""

        cov_matrix = self.cov_matrix if cov_matrix is None else cov_matrix
        assert (cov_dict is None) != (cov_matrix is None), "Either cov_dict or cov_matrix must be provided, but not both."


        possible_inputs = ['c_model_1','c_model_2','c_model_3','c_model_4','c_model_5','c_model_6','c_model_7','c_model_8']
        assert np.all([constraint in possible_inputs for constraint in constraints]), f"Constraint {constraints} not recognized. Available constraints: {possible_inputs}" 

        self.constraint_cov_dict = {}
        print(f"\nCovariance matrix shape: {cov_matrix.shape}")
        print(f"dim_m1: {self.dim_m1}, dim_m2: {self.dim_m2}, dim_t: {self.dim_t}")



        
        if 'c_model_1' in constraints:
            self.constraint_cov_dict['c_model_1'] = I = torch.eye(cov_matrix.shape[0], device=cov_matrix.device, dtype=cov_matrix.dtype)
        #No constraints, all independent

        if 'c_model_2' in constraints:
            M2 = self.create_model_M(block1=self.P)
            self.constraint_cov_dict['c_model_2'] = M2 #M0 and M1 dependent

        if 'c_model_3' in constraints:
            M3 = self.create_model_M(block2=self.Q)
            self.constraint_cov_dict['c_model_3'] = M3 #M0 and T dependent

        if 'c_model_4' in constraints:
            M4 = self.create_model_M(block3=self.R)
            self.constraint_cov_dict['c_model_4'] = M4 #M1 and T dependent

        if 'c_model_5' in constraints:
            P_Q = (self.P).T @ self.Q
            M5 = self.create_model_M(block1=self.P,block2=self.Q,block3=P_Q) 
            self.constraint_cov_dict['c_model_5'] = M5 #M1 and T dependent, M1 and M2 dependent

        if 'c_model_6' in constraints:
            P_R = self.P @ self.R  
            M6 = self.create_model_M(block1=self.P,block2=P_R,block3=self.R)    
            self.constraint_cov_dict['c_model_6'] = M6 #M2 and T dependent, M1 and M2 dependent

        if 'c_model_7' in constraints:
            Q_R = self.Q @ self.R.T
            M7 = self.create_model_M(block1=Q_R,block2=self.Q,block3=self.R)   
            self.constraint_cov_dict['c_model_7'] = M7 #M1 and T dependent, M2 and T dependent

        if 'c_model_8' in constraints: 
            M8 = self.create_model_M(self.P,self.Q,self.R) #Full covariance, all dependent
            self.constraint_cov_dict['c_model_8'] = M8


        return self.constraint_cov_dict
    
    def compute_Idep(self)-> dict:
        """This function calcualtes the mutual information for a given covariance matrix - U models in the lattice"""
        assert hasattr(self, "constraint_cov_dict"), "Run dependency_matrix(...) before compute_Idep(...)."
        
        self.q_logdet = torch.logdet(self.I2 - (self.Q.T @ self.Q))
        self.i_m1_t = -0.5*self.q_logdet 
        
        self.r_logdet = torch.logdet(self.I2 - (self.R.T @ self.R))
        self.i_m2_t = -0.5*self.r_logdet

        #M7:
        mat = self.constraint_cov_dict['c_model_7']
        block7 = mat[:self.dim_m1, self.dim_m1:self.dim_m1 + self.dim_m2] #Q@R.T
        nume7 = torch.logdet(self.I1-(block7.T@block7)) 
        deno7 = torch.logdet(self.I2 - (self.Q.T @ self.Q)) + torch.logdet(self.I2 - (self.R.T @ self.R))  
        m7_raw = 0.5*(nume7- deno7) 
        
        #M8:
        mat = self.constraint_cov_dict['c_model_8']
        nume8 = torch.logdet(self.I1 - (self.P.T @ self.P))
        deno8 = torch.logdet(mat)
        m8_raw = 0.5*(nume8-deno8) 

        #Calculate unique 1
        i = m7_raw  - self.i_m2_t- self.m7_bias['i']
        k = m8_raw  - self.i_m2_t- self.m8_bias['k']
        d = b = self.i_m1_t
     

        unique_1 = torch.min(torch.stack([i,k])) #James proved that b,d are not relavnt therefore it's just a sanity check
        assert unique_1 == i or unique_1 == k, f"Unique information for source 1 should be determined by either U7 or U8. Got unique_1={unique_1}, i={i}, k={k}."
        self.I_dep_values['unique_1'] = unique_1.item()

        #Calculate unique 2
        h = m7_raw  - self.i_m1_t- self.m7_bias['h']
        j = m8_raw  - self.i_m1_t- self.m8_bias['j']
        f = c = self.i_m2_t
        unique_2 = torch.min(torch.stack([h,j])) #James proved that c,f are not relavnt therefore it's just a sanity check

        assert unique_2 == h or unique_2 == j, f"Unique information for source 2 should be determined by either U7 or U8. Got unique_2={unique_2}, h={h}, j={j}."
        self.I_dep_values['unique_2'] = unique_2.item()
        #Check for nan values
        assert not torch.isnan(unique_1), f"unique_0 = {unique_1} was not calculated properly."
        assert not torch.isnan(unique_2), f"unique_2 = {unique_2} was not calculated properly." 
        
        return self.I_dep_values
    
    def pid_values(self,unique_1, unique_2):
        """This function will compute the PID values using the I_dep values
        input: unique_0, unique_1 are the unique informations for source 0 and source 1
        output: a dictionary with the PID values
        keys: 'red', 'unq1', 'unq2', 'syn'"""



        if self.bias_correction:
            df = self.N - 1
            d = self.dim_m1 + self.dim_m2 + self.dim_t
            bias_x1 = logdet_wishart_bias(df, self.dim_m1)
            bias_x2 = logdet_wishart_bias(df, self.dim_m2)
            bias_t = logdet_wishart_bias(df, self.dim_t)
            tri_mi_bias = 0.5*(logdet_wishart_bias(df, self.dim_m1 + self.dim_m2) - (bias_x1 + bias_x2)) - 0.5*(logdet_wishart_bias(df, d)-(bias_x1 + bias_x2 + bias_t))
            bi_mi_bias_1 = 0.5*logdet_wishart_bias(df, self.dim_m2) + 0.5*logdet_wishart_bias(df, self.dim_t) - 0.5*logdet_wishart_bias(df, self.dim_m1 + self.dim_t)
            bi_mi_bias_2 = 0.5*logdet_wishart_bias(df, self.dim_m2) + 0.5*logdet_wishart_bias(df, self.dim_t) - 0.5*logdet_wishart_bias(df, self.dim_m2 + self.dim_t)
        else:
            tri_mi_bias = 0
            bi_mi_bias_1 = 0
            bi_mi_bias_2 = 0
        
        
        self.i_m1_m2_t = 0.5*torch.logdet((self.I1 - self.P.T @ self.P)) - 0.5*torch.logdet(self.constraint_cov_dict['c_model_8']) 
        
        #Bias correction for the mutual information terms (0 if bias_correction is False)
        self.i_m1_m2_t -= tri_mi_bias
        self.i_m1_t -= bi_mi_bias_1
        self.i_m2_t -= bi_mi_bias_2
        
        # Redundant information
        red0 = self.i_m1_t - unique_1
        red1 = self.i_m2_t - unique_2
        assert abs(red0 - red1) < 1e-8, f"Redundant information from both sources not equal. red0: {red0}, red1: {red1}"
        red = red0
        # Synergistic information
        syn = self.i_m1_m2_t - self.i_m1_t - unique_2
        assert not torch.isnan(red), f"Redundant={red} information not calculated properly."
        assert not torch.isnan(syn), f"Synergistic={syn} information not calculated properly."
        self.PID_values = {
            'red': red.item(),
            'unq1': unique_1,
            'unq2': unique_2,
            'syn': syn.item()
        }
        return self.PID_values
    
    def idep(self,cov_matrix: Optional[torch.tensor]=None)-> dict:
        """This function will compute the full Idep PID decomposition

        input: cov_matrix is a torch tensor of shape (d,d) in case you want to provide a different covariance matrix

        output: a dictionary with the PID values

        keys: 'red', 'unq1', 'unq2', 'syn'"""

        self.cov_matrix = self.cov_matrix if cov_matrix is None else cov_matrix

        self.dependency_matrix(constraints=[
            'c_model_1','c_model_2','c_model_3','c_model_4',
            'c_model_5','c_model_6','c_model_7','c_model_8'
        ],cov_matrix=cov_matrix)

        idep_values = self.compute_Idep()
        pid = self.pid_values(idep_values['unique_1'], idep_values['unique_2'])
        mi = {'I(M1;T)': self.i_m1_t.item(), 'I(M2;T)': self.i_m2_t.item(), 'I(M1,M2;T)': self.i_m1_m2_t.item()}
        return pid , mi
    
    




# import your class
# from your_module import Idep_multivariate_gauss

def ones(n, device="cpu", dtype=torch.float64):
    return torch.ones((n, 1), device=device, dtype=dtype)

def equicorr_blocks(n0, n1, n2, p, q, r, device="cpu", dtype=torch.float64):
    """
    Eq (63) in the paper:
      P = p 1_{n0} 1_{n1}^T
      Q = q 1_{n0} 1_{n2}^T
      R = r 1_{n1} 1_{n2}^T
    """
    P = p * (ones(n0, device, dtype) @ ones(n1, device, dtype).T)  # (n0,n1)
    Q = q * (ones(n0, device, dtype) @ ones(n2, device, dtype).T)  # (n0,n2)
    R = r * (ones(n1, device, dtype) @ ones(n2, device, dtype).T)  # (n1,n2)
    return P, Q, R

def build_full_cov(n0, n1, n2, P, Q, R, device="cpu", dtype=torch.float64):
    """
    Full covariance/correlation (Model 8):
    [[I0, P,  Q],
     [P^T,I1, R],
     [Q^T,R^T,I2]]
    """
    I0 = torch.eye(n0, device=device, dtype=dtype)
    I1 = torch.eye(n1, device=device, dtype=dtype)
    I2 = torch.eye(n2, device=device, dtype=dtype)

    row1 = torch.cat([I0,    P,   Q], dim=1)
    row2 = torch.cat([P.T,  I1,   R], dim=1)
    row3 = torch.cat([Q.T,  R.T, I2], dim=1)
    return torch.cat([row1, row2, row3], dim=0)

def pretty(d):
    return {k: float(d[k]) for k in ["unq0", "unq1", "red", "syn"]}

def run_one(n0, n1, n2, p, q, r):
    P, Q, R = equicorr_blocks(n0, n1, n2, p, q, r)
    Sigma = build_full_cov(n0, n1, n2, P, Q, R)

    # IMPORTANT: you pass cov_matrix, but your class needs self.cov_dict=None fix
    obj = Idep_multivariate_gauss(sources=None, targets=None, cov_matrix=Sigma,base_e=True,bias_correction=False)

    # You must set these because when cov_matrix is given, your code currently
    # doesn't populate P,Q,R and dims (unless you refactor __init__)
    obj.dim_m1, obj.dim_m2, obj.dim_t = n0, n1, n2
    obj.I0 = torch.eye(n0)
    obj.I1 = torch.eye(n1)
    obj.I2 = torch.eye(n2)
    obj.P, obj.Q, obj.R = P, Q, R
    obj.cov_matrix = Sigma

    pid = obj.idep(cov_matrix=Sigma)
    # Your pid dict uses keys: red, unq0, unq1, syn
    return pid

def main():
    # Keep expected values in log2 (bits) as in the paper table.
    # Convert to base e (nats) at runtime when base_e=True.
    base_e = True
    ln2 = float(torch.log(torch.tensor(2.0, dtype=torch.float64)))

    examples = [
        # (n0,n1,n2, p,q,r, expected Idep row)
        ((3,4,3), (-0.15, 0.15, 0.15), {"unq1":0.1227, "unq2":0.1865, "red":0.0406, "syn":2.4772}),
        ((4,4,2), (-0.2, -0.2, 0.3),   {"unq1":0.0893, "unq2":0.7293, "red":0.1889, "syn":0.0087}),
        ((4,2,4), (-0.1, 0.15, -0.2),  {"unq1":0.2336, "unq2":0.1899, "red":0.0883, "syn":0.0345}),
    ]

    for (n0,n1,n2), (p,q,r), expected_bits in examples:
        got,_ = run_one(n0,n1,n2,p,q,r)
        got_fmt = {
            "unq1": got["unq1"],
            "unq2": got["unq2"],
            "red":  got["red"],
            "syn":  got["syn"],
        }
        expected = {k: (v * ln2 if base_e else v) for k, v in expected_bits.items()}

        got_3dp = {k: f"{v:.3f}" for k, v in got_fmt.items()}
        expected_3dp = {k: f"{v:.3f}" for k, v in expected.items()}

        print("\n========================================")
        print(f"Example (n0,n1,n2)=({n0},{n1},{n2}), (p,q,r)=({p},{q},{r})")
        print("Got:      ", got_3dp)
        print("Expected: ", expected_3dp)



if __name__ == "__main__":
    main()