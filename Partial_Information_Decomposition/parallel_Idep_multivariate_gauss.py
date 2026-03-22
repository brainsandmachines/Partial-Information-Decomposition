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
    from Partial_Information_Decomposition.PID_util import para_create_cov_matrix,block_singularity_check
    from Partial_Information_Decomposition.bias_corr import entropy_bias_term
else:
    from PID_util import para_create_cov_matrix
from typing import Optional
"""This files implement the Idep univariate source and target method for univariate gaussian variables as described in:
Ince et al. 2018: (Exact Partial Information Decompositions for Gaussian Systems Based on Dependency Constraints)"""


torch.set_default_dtype(torch.float64)
class para_Idep_multivariate_gauss:
    def __init__(self,N=None,df=None,device = 'cuda',sources: Optional[list] = None, targets:Optional[list] = None,cov_matrix: Optional[torch.tensor]=None,dims: Optional[list] = None,bias_correction: bool = False):
        """Initialize the Idep multivariate gaussian class

        input: M1,M2,T are torch tensors of shape (N,P)
        N is the number of observations
        P is the number of variables in each observation

        """

        self.device = device
        self.N = N
        self.cov_dict = None
        assert sources is not None or dims is not None, "Either sources or dims must be provided. If sources are provided, they should be a list of three torch tensors [M1,M2,T]. If dims are provided, they should be a list of three integers [dim_m1, dim_m2, dim_t]."
        self.M1,self.M2,self.T = sources if sources is not None else (None, None, None)
        is_sorces_provided = sources is not None
        self.df = df if df is not None else (self.N - 1 if self.M1 is not None else None)

        # Find features dimensions from the provided sources or from the provided dims
        self.dim_m1 = self.M1.shape[1] if is_sorces_provided else dims[0]
        self.dim_m2 = self.M2.shape[1] if is_sorces_provided else dims[1]
        self.dim_t = self.T.shape[1] if is_sorces_provided else dims[2]
        self.dims = [self.dim_m1, self.dim_m2, self.dim_t] if dims is None else dims
        self.dall = self.dim_m1 + self.dim_m2 + self.dim_t

        self.I0 = torch.eye(self.dim_m1, device=self.device)
        self.I0 = self.I0.repeat(self.N, 1, 1) if self.N > 1 else self.I0.unsqueeze(0)
        self.I1 = torch.eye(self.dim_m2, device=self.device)
        self.I1 = self.I1.repeat(self.N, 1, 1) if self.N > 1 else self.I1.unsqueeze(0)
        self.I2 = torch.eye(self.dim_t, device=self.device)
        self.I2 = self.I2.repeat(self.N, 1, 1) if self.N > 1 else self.I2.unsqueeze(0)
        if cov_matrix is not None:
            self.cov_dict = para_create_cov_matrix(dims=self.dims,Sigmas=cov_matrix)
            self.cov_matrix = self.cov_dict['full_cov'] 


        if self.cov_dict is not None:
            self.sigma00 = self.cov_dict['cov_x0'] #(N, dim_m1, dim_m1)
            self.sigma11 = self.cov_dict['cov_x1'] #(N, dim_m2, dim_m2)
            self.sigma22 = self.cov_dict['cov_x2'] #(N, dim_t, dim_t)
            self.sigma01 = self.cov_dict['cross_x0_x1'] #(N, dim_m1, dim_m2)
            self.sigma02 = self.cov_dict['cross_x0_x2'] #(N, dim_m1, dim_t)
            self.sigma12 = self.cov_dict['cross_x1_x2'] #(N, dim_m2, dim_t)


            self.P = self.whiten_block(self.sigma00, self.sigma01, self.sigma11) #(N, dim_m1, dim_m2)
            self.Q = self.whiten_block(self.sigma00, self.sigma02, self.sigma22) #(N, dim_m1, dim_t)
            self.R = self.whiten_block(self.sigma11, self.sigma12, self.sigma22) #(N, dim_m2, dim_t)
            # 1. Build the three "rows" of the block matrix by concatenating along the column dimension (dim=-1)
            row1 = torch.cat([self.I0,   self.P,    self.Q],    dim=-1)
            row2 = torch.cat([self.P.mT, self.I1,   self.R],    dim=-1)
            row3 = torch.cat([self.Q.mT, self.R.mT, self.I2],   dim=-1)

            # 2. Stack the rows vertically by concatenating along the row dimension (dim=-2)
            self.cov_whiten = torch.cat([row1, row2, row3], dim=-2)

            assert self.cov_whiten.shape == (self.N,self.dall,self.dall), f"Whitened covariance matrix shape {self.cov_whiten.shape} does not match expected shape {(self.dall,self.dall)}."
            assert self.P.shape == (self.N,self.dim_m1,self.dim_m2), f"Covariance matrix dimensions {self.P.shape} do not match the provided source dimensions: {self.dim_m1,self.dim_m2}."
            assert self.Q.shape == (self.N,self.dim_m1,self.dim_t), f"Covariance matrix dimensions {self.Q.shape} do not match the provided source and target dimensions: {self.dim_m1,self.dim_t}."
            assert self.R.shape == (self.N,self.dim_m2,self.dim_t), f"Covariance matrix dimensions {self.R.shape} do not match the provided source and target dimensions: {self.dim_m2,self.dim_t}."
            assert self.dim_m1 + self.dim_m2 + self.dim_t == self.cov_matrix.shape[1], f"Covariance matrix dimensions {self.cov_matrix.shape} do not match the provided source and target dimensions: {self.dim_m1 + self.dim_m2 + self.dim_t}."
            
            # Singularity check for P,Q and R blocks
            p = (torch.eye(self.P.shape[1],device=self.device) - self.P @ self.P.mT).detach().cpu().numpy()
            q = (torch.eye(self.Q.shape[1],device=self.device) - self.Q @ self.Q.mT).detach().cpu().numpy()
            r = (torch.eye(self.R.shape[1],device=self.device) - self.R @ self.R.mT).detach().cpu().numpy()
            #block_singularity_check(np.array([p,q,r]))
            #block_singularity_check(np.array([p.T,q.T,r.T]))
        if bias_correction:
            self.bm1 = entropy_bias_term(self.df, self.dim_m1) 
            self.bm2 = entropy_bias_term(self.df, self.dim_m2)
            self.bt = entropy_bias_term(self.df, self.dim_t)
            
            self.bq = entropy_bias_term(self.df, self.dim_t + self.dim_m1)
            self.br = entropy_bias_term(self.df, self.dim_t + self.dim_m2)
            self.bp = entropy_bias_term(self.df, self.dim_m1 + self.dim_m2)
            self.ball = entropy_bias_term(self.df, self.dim_m1 + self.dim_m2 + self.dim_t)
            
        else: 
            self.ball = self.b_tmi = self.bm1 = self.bm2 = self.bt = self.bq = self.br = self.bp = 0
            
        self.I_dep_values = {}
        self.PID_values = {}

    def whiten_block(self,
                    Sigma_xx: torch.Tensor,
                    Sigma_xy: torch.Tensor,
                    Sigma_yy: torch.Tensor) -> torch.Tensor:
        """
        Computes: Ux^{-T} @ Sigma_xy @ Uy^{-1}
        where Sigma_xx = Ux^T Ux, Sigma_yy = Uy^T Uy, and Ux,Uy are upper triangular.
        Supports batched inputs of shape (N, d, d).
        """
        # Use .mT to transpose only the last two dimensions (matrix transpose)
        Ux = torch.linalg.cholesky(Sigma_xx).mT
        Uy = torch.linalg.cholesky(Sigma_yy).mT

        # Uy.mT is lower triangular. 
        # solve_triangular computes X where Uy.mT @ X = Sigma_xy.mT
        tmp = torch.linalg.solve_triangular(Uy.mT, Sigma_xy.mT, upper=False).mT
        
        # Ux.mT is lower triangular.
        # solve_triangular computes Y where Ux.mT @ Y = tmp
        K = torch.linalg.solve_triangular(Ux.mT, tmp, upper=False)

        return K

    
    def compute_Idep(self)-> dict:
        """This function calcualtes the mutual information for a given covariance matrix - U models in the lattice"""
        
        self.q_logdet = torch.logdet(self.I2 - (self.Q.mT @ self.Q))
        
        self.r_logdet = torch.logdet(self.I2 - (self.R.mT @ self.R))

        self.q_correction = (self.bm1 + self.bt - self.bq)
        self.r_correction = (self.bm2 + self.bt - self.br)
        self.b_tmi = (self.bp + self.bt - self.ball)
        #Mutual Information
        self.i_m1_t_raw = -0.5*self.q_logdet 
        self.i_m2_t_raw = -0.5*self.r_logdet 

        #M7:
        block7 = self.Q @ self.R.mT #Q@R.T
        nume7 = torch.logdet(self.I1-(block7.mT @block7)) 
        deno7 = self.q_logdet + self.r_logdet
        m7 = 0.5*(nume7- deno7)

        
        
        #M8:
        mat = self.cov_whiten
        nume8 = torch.logdet(self.I1 - (self.P.mT @ self.P))
        deno8 = torch.logdet(mat)
        m8 = 0.5*(nume8-deno8) 
        
        #Calculate unique 1
        i = m7  - self.i_m2_t_raw
        k = m8  - self.i_m2_t_raw
        b = self.i_m1_t_raw
        d = self.i_m1_t_raw

        bias_cmi_m1_given_m2 = self.bp + self.br - self.bm2 - self.ball
        bias_cmi_m2_given_m1 = self.bp + self.bq - self.bm1 - self.ball
        self.unique_1 = torch.min(torch.stack([i,k]),dim=0).values #James proved that b,d are not relavnt therefore it's just a sanity check
        #assert self.unique_1 == i or self.unique_1 == k, f"Unique information for source 1 should be determined by either U7 or U8. Got unique_1={self.unique_1}, i={i}, k={k}."
        self.unique_1 += bias_cmi_m1_given_m2
        self.I_dep_values['unique_1'] = self.unique_1
    
        #Calculate unique 2
        h = m7  - self.i_m1_t_raw
        j = m8  - self.i_m1_t_raw
        c = self.i_m2_t_raw
        f = self.i_m2_t_raw
        self.unique_2 = torch.min(torch.stack([h,j]),dim=0).values #James proved that c,f are not relavnt therefore it's just a sanity check
        self.unique_2 += bias_cmi_m2_given_m1
        self.I_dep_values['unique_2'] = self.unique_2
        #assert self.unique_2 == h or self.unique_2 == j, f"Unique information for source 2 should be determined by either U7 or U8. Got unique_2={self.unique_2}, h={h}, j={j}."

        #Check for nan values
        assert not torch.all(torch.isnan(self.unique_1)), f"unique_1 = {self.unique_1} was not calculated properly."
        assert not torch.all(torch.isnan(self.unique_2)), f"unique_2 = {self.unique_2} was not calculated properly."  
        return self.I_dep_values
    
    
    def pid_values(self,unique_1, unique_2):
        """This function will compute the PID values using the I_dep values
        input: unique_0, unique_1 are the unique informations for source 0 and source 1
        output: a dictionary with the PID values
        keys: 'red', 'unq1', 'unq2', 'syn'"""

        
        self.i_m1_m2_t = 0.5*torch.logdet((self.I1 - self.P.mT @ self.P)) - 0.5*torch.logdet(self.cov_whiten)
        self.i_m1_m2_t += self.b_tmi
        self.i_m1_t = self.i_m1_t_raw + self.q_correction
        self.i_m2_t = self.i_m2_t_raw + self.r_correction
        # Redundant information
        red0 = self.i_m1_t - unique_1
        red1 = self.i_m2_t - unique_2
        assert torch.all(abs(red0 - red1)< 1e-8), f"Redundant information from both sources not equal. red0: {red0}, red1: {red1}"
        red = red0
        # Synergistic information
        syn = self.i_m1_m2_t - self.i_m1_t - unique_2
        assert not torch.all(torch.isnan(red)), f"Redundant={red} information not calculated properly."
        assert not torch.all(torch.isnan(syn)), f"Synergistic={syn} information not calculated properly."
        self.PID_values = {
            'red': red,
            'unq1': unique_1,
            'unq2': unique_2,
            'syn': syn
        }
        return self.PID_values
    
    def idep(self,cov_matrix: Optional[torch.tensor]=None)-> dict:
        """This function will compute the full Idep PID decomposition

        input: cov_matrix is a torch tensor of shape (d,d) in case you want to provide a different covariance matrix

        output: a dictionary with the PID values

        keys: 'red', 'unq1', 'unq2', 'syn'"""

        self.cov_matrix = self.cov_matrix if cov_matrix is None else cov_matrix


        idep_values = self.compute_Idep()
        pid = self.pid_values(idep_values['unique_1'], idep_values['unique_2'])
        mi = {'I(M1;T)': self.i_m1_t, 'I(M2;T)': self.i_m2_t, 'I(M1,M2;T)': self.i_m1_m2_t}
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    P, Q, R = equicorr_blocks(n0, n1, n2, p, q, r, device=device)
    Sigma = build_full_cov(n0, n1, n2, P, Q, R,device=device)
    Sigma = Sigma.unsqueeze_(0)  # Add batch dimension, shape becomes (1, d, d)
    # IMPORTANT: you pass cov_matrix, but your class needs self.cov_dict=None fix

    dims = [n0, n1, n2]
    obj = para_Idep_multivariate_gauss(N =1 ,device = device, sources=None, targets=None, cov_matrix=Sigma,dims=dims,bias_correction=False)


    pid = obj.idep()
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