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
from sklearn.model_selection import LeaveOneOut
if __name__ != "__main__":
    from .PID_util import create_cov_matrix, cond_cov, standardize, assert_full_rank,singularity_report,block_singularity_check
    from .bias_corr import entropy_bias_term,matrix_beta_mi_bias
else:
    from PID_util import create_cov_matrix, cond_cov, standardize, assert_full_rank,singularity_report
from typing import Optional
"""This files implement the Idep univariate source and target method for univariate gaussian variables as described in:
Ince et al. 2018: (Exact Partial Information Decompositions for Gaussian Systems Based on Dependency Constraints)"""


torch.set_default_dtype(torch.float64)
class Idep_multivariate_gauss:
    def __init__(self, sources: Optional[list] = None, targets:Optional[list] = None,cov_matrix: Optional[torch.tensor]=None,bias_correction: bool = False,verbose:bool = True):
        """Initialize the Idep multivariate gaussian class

        input: M1,M2,T are torch tensors of shape (N,P)
        N is the number of observations
        P is the number of variables in each observation

        """
        self.verbose = verbose
        self.M1 = sources[0] if sources is not None else None
        self.M2 = sources[1] if sources is not None else None
        self.T = targets[0] if targets is not None else None
        self.cov_dict = None
        self.N = self.M1.shape[0] if self.M1 is not None else None
        self.df = self.N - 1  if self.M1 is not None else 0
        self.dim_m1 = self.M1.shape[1] if self.M1 is not None else 0
        self.dim_m2 = self.M2.shape[1] if self.M2 is not None else 0
        self.dim_t = self.T.shape[1] if self.T is not None else 0

        self.I0 = torch.eye(self.dim_m1)
        self.I1 = torch.eye(self.dim_m2)
        self.I2 = torch.eye(self.dim_t)

        if self.M1 is not None and self.M2 is not None and self.T is not None:
            self.cov_dict = create_cov_matrix(self.M1,self.M2,self.T,self.verbose)
            self.cov_matrix = self.cov_dict['full_cov']
            #assert_full_rank(self.cov_matrix)
        elif cov_matrix is not None:
            self.cov_matrix = cov_matrix

        if self.cov_dict is not None:
            self.sigma00 = self.cov_dict['cov_x0']
            self.sigma11 = self.cov_dict['cov_x1']
            self.sigma22 = self.cov_dict['cov_x2']
            self.sigma01 = self.cov_dict['cross_x0_x1']
            self.sigma02 = self.cov_dict['cross_x0_x2']
            self.sigma12 = self.cov_dict['cross_x1_x2']


            self.P = self.whiten_block(self.sigma00, self.sigma01, self.sigma11)
            self.Q = self.whiten_block(self.sigma00, self.sigma02, self.sigma22)
            self.R = self.whiten_block(self.sigma11, self.sigma12, self.sigma22)


            assert self.P.shape == (self.dim_m1,self.dim_m2), f"Covariance matrix dimensions {self.P.shape} do not match the provided source dimensions: {self.dim_m1,self.dim_m2}."
            assert self.Q.shape == (self.dim_m1,self.dim_t), f"Covariance matrix dimensions {self.Q.shape} do not match the provided source and target dimensions: {self.dim_m1,self.dim_t}."
            assert self.R.shape == (self.dim_m2,self.dim_t), f"Covariance matrix dimensions {self.R.shape} do not match the provided source and target dimensions: {self.dim_m2,self.dim_t}."
            assert self.dim_m1 + self.dim_m2 + self.dim_t == self.cov_matrix.shape[0], f"Covariance matrix dimensions {self.cov_matrix.shape} do not match the provided source and target dimensions: {self.dim_m1 + self.dim_m2 + self.dim_t}."
            
            # Singularity check for P,Q and R blocks
            p = (self.I1 - self.P.T @ self.P).detach().cpu().numpy()
            q = (self.I2 - self.Q.T @ self.Q).detach().cpu().numpy()
            r = (self.I2 - self.R.T @ self.R).detach().cpu().numpy()
            block_singularity_check(np.array([p]))
            block_singularity_check(np.array([q]))
            block_singularity_check(np.array([r]))
        self.I_dep_values = {}
        self.PID_values = {}

    def whiten_block(self,
                        Sigma_xx: torch.Tensor,
                        Sigma_xy: torch.Tensor,
                        Sigma_yy: torch.Tensor) -> torch.Tensor:
        """
        return Ux^{-T} @ Sigma_xy @ Uy^{-1}
        where Sigma_xx = Ux^T Ux, Sigma_yy = Uy^T Uy, and Ux,Uy are upper triangular.
        """
        Ux = torch.linalg.cholesky(Sigma_xx).T
        Uy = torch.linalg.cholesky(Sigma_yy).T

        tmp = torch.linalg.solve_triangular(Uy.T, Sigma_xy.T, upper=False).T
        K   = torch.linalg.solve_triangular(Ux.T, tmp,        upper=False)

        return K
              

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


    def dependency_matrix(self,cov_matrix: Optional[torch.tensor]=None,cov_dict: Optional[dict]=None)-> dict:
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
        

        self.constraint_cov_dict = {}
        if self.verbose:
            print(f"\nCovariance matrix shape: {cov_matrix.shape}")
            print(f"dim_m1: {self.dim_m1}, dim_m2: {self.dim_m2}, dim_t: {self.dim_t}")

        self.constraint_cov_dict['c_model_1'] = I = torch.eye(cov_matrix.shape[0], device=cov_matrix.device, dtype=cov_matrix.dtype)
    #No constraints, all independent

        M2 = self.create_model_M(block1=self.P)
        self.constraint_cov_dict['c_model_2'] = M2 #M0 and M1 dependent


        M3 = self.create_model_M(block2=self.Q)
        self.constraint_cov_dict['c_model_3'] = M3 #M0 and T dependent


        M4 = self.create_model_M(block3=self.R)
        self.constraint_cov_dict['c_model_4'] = M4 #M1 and T dependent


        P_Q = (self.P).T @ self.Q
        M5 = self.create_model_M(block1=self.P,block2=self.Q,block3=P_Q) 
        self.constraint_cov_dict['c_model_5'] = M5 #M1 and T dependent, M1 and M2 dependent


        P_R = self.P @ self.R  
        M6 = self.create_model_M(block1=self.P,block2=P_R,block3=self.R)    
        self.constraint_cov_dict['c_model_6'] = M6 #M2 and T dependent, M1 and M2 dependent

        Q_R = self.Q @ self.R.T
        M7 = self.create_model_M(block1=Q_R,block2=self.Q,block3=self.R)   
        self.constraint_cov_dict['c_model_7'] = M7 #M1 and T dependent, M2 and T dependent

        M8 = self.create_model_M(self.P,self.Q,self.R) #Full covariance, all dependent
        self.constraint_cov_dict['c_model_8'] = M8


        return self.constraint_cov_dict
    
    def compute_Idep(self)-> dict:
        """This function calcualtes the mutual information for a given covariance matrix - U models in the lattice"""
        assert hasattr(self, "constraint_cov_dict"), "Run dependency_matrix(...) before compute_Idep(...)."
        
        self.q_logdet = torch.logdet(self.I2 - (self.Q.T @ self.Q))
        
        self.r_logdet = torch.logdet(self.I2 - (self.R.T @ self.R))

        #Mutual Information
        self.i_m1_t = -0.5*self.q_logdet
        self.i_m2_t = -0.5*self.r_logdet

        #M7:
        mat = self.constraint_cov_dict['c_model_7']
        block7 = mat[:self.dim_m1, self.dim_m1:self.dim_m1 + self.dim_m2] #Q@R.T
        nume7 = torch.logdet(self.I1-(block7.T@block7)) 
        deno7 = self.q_logdet + self.r_logdet
        m7 = 0.5*(nume7- deno7) 

        
        
        #M8:
        mat = self.constraint_cov_dict['c_model_8']
        nume8 = torch.logdet(self.I1 - (self.P.T @ self.P))
        deno8 = torch.logdet(mat)
        m8 = 0.5*(nume8-deno8)
        
        #Calculate unique 1
        i = m7  - self.i_m2_t 
        k = m8  - self.i_m2_t
        b = self.i_m1_t 
        d = self.i_m1_t 
        self.unique_1 = torch.min(torch.stack([i,k])) #James proved that b,d are not relavnt therefore it's just a sanity check
        assert self.unique_1 == i or self.unique_1 == k, f"Unique information for source 1 should be determined by either U7 or U8. Got unique_1={self.unique_1}, i={i}, k={k}."
        self.I_dep_values['unique_1'] = self.unique_1.item()
    
        #Calculate unique 2
        h = m7  - self.i_m1_t 
        j = m8  - self.i_m1_t
        c = self.i_m2_t 
        f = self.i_m2_t  
        self.unique_2 = torch.min(torch.stack([h,j])) #James proved that c,f are not relavnt therefore it's just a sanity check
        self.I_dep_values['unique_2'] = self.unique_2.item()
        assert self.unique_2 == h or self.unique_2 == j, f"Unique information for source 2 should be determined by either U7 or U8. Got unique_2={self.unique_2}, h={h}, j={j}."

        #Check for nan values
        assert not torch.isnan(self.unique_1), f"unique_1 = {self.unique_1} was not calculated properly."
        assert not torch.isnan(self.unique_2), f"unique_2 = {self.unique_2} was not calculated properly."  
        return self.I_dep_values
    

    
    def pid_values(self,unique_1, unique_2):
        """This function will compute the PID values using the I_dep values
        input: unique_0, unique_1 are the unique informations for source 0 and source 1
        output: a dictionary with the PID values
        keys: 'red', 'unq1', 'unq2', 'syn'"""

        
        self.i_m1_m2_t = 0.5*torch.logdet((self.I1 - self.P.T @ self.P)) - 0.5*torch.logdet(self.constraint_cov_dict['c_model_8']) 
        # Redundant information
        red0 = self.i_m1_t.item() - unique_1
        red1 = self.i_m2_t.item() - unique_2
        assert abs(red0 - red1) < 1e-8, f"Redundant information from both sources not equal. red0: {red0}, red1: {red1}"
        red = red0
        # Synergistic information
        syn = self.i_m1_m2_t.item() - self.i_m1_t.item() - unique_2
        assert not torch.isnan(torch.tensor([red])), f"Redundant={red} information not calculated properly."
        assert not torch.isnan(torch.tensor([syn])), f"Synergistic={syn} information not calculated properly."
        self.PID_values = {
            'red': red,
            'unq1': unique_1,
            'unq2': unique_2,
            'syn': syn
        }
        self.mi = {
            'I(M1;T)': self.i_m1_t,
            'I(M2;T)': self.i_m2_t,
            'I(M1,M2;T)': self.i_m1_m2_t
        }
        return self.PID_values, self.mi
    
    def idep(self,cov_matrix: Optional[torch.tensor]=None,bias_correction: bool=False)-> dict:
        """This function will compute the full Idep PID decomposition

        input: cov_matrix is a torch tensor of shape (d,d) in case you want to provide a different covariance matrix

        output: a dictionary with the PID values

        keys: 'red', 'unq1', 'unq2', 'syn'"""

        self.cov_matrix = self.cov_matrix if cov_matrix is None else cov_matrix

        self.dependency_matrix(cov_matrix=cov_matrix)
 
        idep_values = self.compute_Idep()
        self.unique_1_raw = idep_values['unique_1']
        self.unique_2_raw = idep_values['unique_2']
        pid_raw, mi_raw = self.pid_values(self.unique_1_raw, self.unique_2_raw)
        if bias_correction:
            jackknife_pid,jackknife_mi = self.jackknife_pid(self.M1.detach().cpu(), self.M2.detach().cpu(), self.T.detach().cpu(),func=Idep_multivariate_gauss) 


            pid_b = {k: self.df * (jackknife_pid[k + '_jack'] - pid_raw[k]) for k in pid_raw}
            mi_b = {k: self.df * (jackknife_mi[k + '_jack'] - mi_raw[k]) for k in mi_raw}

            pid = {k: pid_raw[k] - pid_b[k] for k in pid_raw}
            mi = {k: mi_raw[k] - mi_b[k] for k in mi_raw}

        else:
            self.unique_1 = self.unique_1_raw
            self.unique_2 = self.unique_2_raw
            pid,mi = pid_raw, mi_raw
        
            
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
    obj = Idep_multivariate_gauss(sources=None, targets=None, cov_matrix=Sigma,bias_correction=False)

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
