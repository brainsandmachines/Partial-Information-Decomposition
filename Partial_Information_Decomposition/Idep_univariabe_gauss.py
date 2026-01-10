from networkx import constraint
import torch
import numpy as np
import matplotlib.pyplot as plt 
from torch.distributions import Normal
from sklearn.linear_model import LinearRegression

if __name__ != "__main__":
    from .PID_util import create_cov_matrix, cond_cov
else:
    from PID_util import create_cov_matrix, cond_cov
from typing import Optional
"""This files implement the Idep univariate source and target method for univariate gaussian variables as described in:
Ince et al. 2018: (Exact Partial Information Decompositions for Gaussian Systems Based on Dependency Constraints)"""



class Idep_univariate_gauss:
    def __init__(self, sources: Optional[list] = None, targets:Optional[list] = None,cov_matrix: Optional[torch.tensor]=None):
        """Initialize the Idep univariate gaussian class

        input: M1,M2,T are torch tensors of shape (N,1)
        N is the number of observations

        """
        self.M1 = sources[0] if sources is not None else None
        self.M2 = sources[1] if sources is not None else None
        self.T = targets[0] if targets is not None else None
        if self.M1 is not None and self.M2 is not None and self.T is not None:
            self.cov_dict = create_cov_matrix(self.M1,self.M2,self.T)
            self.cov_matrix = self.cov_dict['full_cov']
        elif cov_matrix is not None:
            self.cov_matrix = cov_matrix

            p = self.cov_matrix[0,1]
            q = self.cov_matrix[0,2]
            r = self.cov_matrix[1,2]
            det = 1 - p**2 - q**2 - r**2 + 2*p*q*r
            assert det > 0, "U8 covariance/correlation not positive definite."

        self.I_dep_values = {}
        self.PID_values = {}



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
        p = cov_matrix[0,1]
        q = cov_matrix[0,2]
        r = cov_matrix[1,2] 

        if 'c_model_8' in constraints: 
            self.constraint_cov_dict['c_model_8'] = cov_matrix #Full covariance, all dependent

        if 'c_model_1' in constraints:
            self.constraint_cov_dict['c_model_1'] = torch.eye(cov_matrix.shape[0]) #No constraints, all independent

        if 'c_model_2' in constraints:
            I = torch.eye(cov_matrix.shape[0])
            I[0,1] = I[1,0] = p
            self.constraint_cov_dict['c_model_2'] = I #M0 and M1 dependent

        if 'c_model_3' in constraints:
            I = torch.eye(cov_matrix.shape[0])
            I[0,2] = I[2,0] = q
            self.constraint_cov_dict['c_model_3'] = I #M0 and T dependent

        if 'c_model_4' in constraints:
            I = torch.eye(cov_matrix.shape[0])
            I[1,2] = I[2,1] = r
            self.constraint_cov_dict['c_model_4'] = I #M1 and T dependent

        if 'c_model_5' in constraints:
            I = torch.eye(cov_matrix.shape[0])
            I[0,1] = I[1,0] = p
            I[0,2] = I[2,0] = q
            I[1,2] = I[2,1] = p*q    
            self.constraint_cov_dict['c_model_5'] = I #M1 and T dependent, M1 and M2 dependent
        if 'c_model_6' in constraints:
            I = torch.eye(cov_matrix.shape[0])
            I[0,1] = I[1,0] = p
            I[0,2] = I[2,0] = p*r
            I[1,2] = I[2,1] = r    
            self.constraint_cov_dict['c_model_6'] = I #M2 and T dependent, M1 and M2 dependent
        if 'c_model_7' in constraints:
            I = torch.eye(cov_matrix.shape[0])
            I[0,1] = I[1,0] = q*r
            I[0,2] = I[2,0] = q
            I[1,2] = I[2,1] = r    
            self.constraint_cov_dict['c_model_7'] = I #M1 and T dependent, M2 and T dependent

        return self.constraint_cov_dict
    
    def compute_Idep(self,unique:list = [0,1])-> dict:
        """This function calcualtes the mutual information for a given covariance matrix - U models in the lattice"""
        assert hasattr(self, "constraint_cov_dict"), "Run dependency_matrix(...) before compute_Idep(...)."
        self.i_m0_t = 0.5*torch.log(1/(1 - self.cov_matrix[0,2]**2))
        self.i_m1_t = 0.5*torch.log(1/(1 - self.cov_matrix[1,2]**2))
        
        if 0 in unique:
            # calculate b and d both equal to I(M0;T)
            b = self.i_m0_t 
            d = self.i_m0_t 

            #calculate i with U7:
            mat = self.constraint_cov_dict['c_model_7']
            q_u7 = mat[0,2]
            r_u7 = mat[1,2]
            i = 0.5*torch.log((1-(q_u7*r_u7)**2)/((1-q_u7**2)*(1-r_u7**2))) - self.i_m1_t 

            #calculate k with U8:
            mat = self.constraint_cov_dict['c_model_8']
            q_u8 = mat[0,2]
            r_u8 = mat[1,2]
            p_u8 = mat[0,1]
            k = 0.5*torch.log((1 - p_u8**2)/(1 - p_u8**2 - q_u8**2 - r_u8**2 + 2*p_u8*q_u8*r_u8)) - self.i_m1_t


            unique_0 = torch.min(torch.stack([b,d,i,k]))
            self.I_dep_values['unique_0'] = unique_0.item()



        if 1 in unique:
            # calculate c and f both equal to I(M1;T)
            c = self.i_m1_t 
            f = self.i_m1_t  

            #calculate h with U7:
            mat = self.constraint_cov_dict['c_model_7']
            q_u7 = mat[0,2]
            r_u7 = mat[1,2]
            h = 0.5*torch.log((1-(q_u7*r_u7)**2)/(((1-q_u7**2)*(1-r_u7**2)))) - self.i_m0_t

            #calculate j with U8:
            mat = self.constraint_cov_dict['c_model_8']
            q_u8 = mat[0,2]
            r_u8 = mat[1,2]
            p_u8 = mat[0,1]
            j = 0.5*torch.log((1 - p_u8**2)/(1 - p_u8**2 - q_u8**2 - r_u8**2 + 2*p_u8*q_u8*r_u8)) - self.i_m0_t

            unique_1 = torch.min(torch.stack([c,f,h,j]))
            self.I_dep_values['unique_1'] = unique_1.item()

        return self.I_dep_values
    def pid_values(self,unique_0, unique_1):
        """This function will compute the PID values using the I_dep values
        input: unique_0, unique_1 are the unique informations for source 0 and source 1
        output: a dictionary with the PID values
        keys: 'red', 'unq0', 'unq1', 'syn'"""
        i_m0_t = self.i_m0_t if self.i_m0_t is not None else 0.5*torch.log(1/(1 - self.cov_matrix[0,2]**2))
        i_m1_t = self.i_m1_t if self.i_m1_t is not None else 0.5*torch.log(1/(1 - self.cov_matrix[1,2]**2))
        self.i_m0_m1_t = 0.5*torch.log((1 - self.cov_matrix[0,1]**2)/(1 - self.cov_matrix[0,1]**2 - self.cov_matrix[0,2]**2 - self.cov_matrix[1,2]**2 + 2*self.cov_matrix[0,1]*self.cov_matrix[0,2]*self.cov_matrix[1,2]))
        # Redundant information
        red0 = i_m0_t - unique_0
        red1 = i_m1_t - unique_1
        assert abs(red0 - red1) < 1e-10, "Redundant information from both sources not equal."
        red = red0
        # Synergistic information
        syn = self.i_m0_m1_t - (red + unique_0 + unique_1)

        self.PID_values = {
            'red': red.item(),
            'unq0': unique_0,
            'unq1': unique_1,
            'syn': syn.item()
        }
        return self.PID_values
    
    def idep(self,cov_matrix: Optional[torch.tensor]=None)-> dict:
        """This function will compute the full Idep PID decomposition

        input: cov_matrix is a torch tensor of shape (d,d) in case you want to provide a different covariance matrix

        output: a dictionary with the PID values

        keys: 'red', 'unq0', 'unq1', 'syn'"""

        self.cov_matrix = self.cov_matrix if cov_matrix is None else cov_matrix

        self.dependency_matrix(constraints=[
            'c_model_1','c_model_2','c_model_3','c_model_4',
            'c_model_5','c_model_6','c_model_7','c_model_8'
        ],cov_matrix=cov_matrix)

        idep_values = self.compute_Idep(unique=[0,1])
        pid = self.pid_values(idep_values['unique_0'], idep_values['unique_1'])
        return pid
    
    


#=======Canoniacal test examples from the paper =======
def test_idep_gauss_q0_example(p=0.3, r=0.5, tol=1e-8):
    """
    Test Example 1 from the paper:
      q = corr(X0, Y) = 0
      p = corr(X0, X1) != 0
      r = corr(X1, Y) != 0

    Expected:
      unq0 = 0
      red  = 0
      unq1 = 1/2 log(1 / (1 - r^2))
      syn  = 1/2 log(((1 - p^2)(1 - r^2)) / (1 - p^2 - r^2))
    """

    # --- build covariance (X0, X1, Y) ---
    Sigma = np.array([
        [1.0, p,   0.0],
        [p,   1.0, r  ],
        [0.0, r,   1.0]
    ])

    idep_class = Idep_univariate_gauss()
    # --- run Idep Gaussian PID ---
    pid = idep_class.idep(
        cov_matrix=torch.tensor(Sigma, dtype=torch.float64))
 

    # --- analytic values ---
    unq1_expected = 0.5 * np.log(1.0 / (1.0 - r**2))
    syn_expected  = 0.5 * np.log(((1.0 - p**2) * (1.0 - r**2)) /
                                 (1.0 - p**2 - r**2))

    print("=== Expected ===")
    print(f"red  = 0")
    print(f"unq0 = 0")
    print(f"unq1 = {unq1_expected}")
    print(f"syn  = {syn_expected}")

    print("\n=== Your code ===")
    for k in ["red", "unq0", "unq1", "syn"]:
        print(f"{k} = {pid[k]}")

    # --- checks ---
    assert abs(pid["red"])  < tol, "red should be 0"
    assert abs(pid["unq0"]) < tol, "unq0 should be 0"
    assert abs(pid["unq1"] - unq1_expected) < tol, "unq1 mismatch"
    assert abs(pid["syn"]  - syn_expected)  < tol, "synergy mismatch"

    print("\n✅ Example 1 (q=0) PASSED")

def check_idep_gauss_r0_example(p=0.3, q=0.5, tol=1e-8):
    """
    Example 2 from the paper:
      r = corr(X1, Y) = 0
      p = corr(X0, X1) != 0
      q = corr(X0, Y)  != 0

    Expected:
      unq1 = 0
      red  = 0
      unq0 = 1/2 log(1 / (1 - q^2))
      syn  = 1/2 log(((1 - p^2)(1 - q^2)) / (1 - p^2 - q^2))
    """

    # --- build covariance (X0, X1, Y) ---
    Sigma = np.array([
        [1.0, p,   q  ],
        [p,   1.0, 0.0],
        [q,   0.0, 1.0]
    ])

    # --- run Idep Gaussian PID ---
    idep_class = Idep_univariate_gauss()
    pid = idep_class.idep(
        cov_matrix=torch.tensor(Sigma, dtype=torch.float64))

    # --- analytic values ---
    unq0_expected = 0.5 * np.log(1.0 / (1.0 - q**2))
    syn_expected  = 0.5 * np.log(((1.0 - p**2) * (1.0 - q**2)) /
                                 (1.0 - p**2 - q**2))

    print("=== Expected ===")
    print(f"red  = 0")
    print(f"unq1 = 0")
    print(f"unq0 = {unq0_expected}")
    print(f"syn  = {syn_expected}")

    print("\n=== Your code ===")
    for k in ["red", "unq0", "unq1", "syn"]:
        print(f"{k} = {pid[k]}")

    # --- checks ---
    assert abs(pid["red"])  < tol, "red should be 0"
    assert abs(pid["unq1"]) < tol, "unq1 should be 0"
    assert abs(pid["unq0"] - unq0_expected) < tol, "unq0 mismatch"
    assert abs(pid["syn"]  - syn_expected)  < tol, "synergy mismatch"

    print("\n✅ Example 2 (r=0) PASSED")

import numpy as np
import torch

def check_idep_gauss_p0_example(q=0.3, r=0.5, tol=1e-8):
    """
    Example 3 from the paper:
      p = corr(X0, X1) = 0
      q = corr(X0, Y)  != 0
      r = corr(X1, Y)  != 0

    Expected:
      unq0 = 1/2 log((1 - q^2 r^2) / (1 - q^2))
      unq1 = 1/2 log((1 - q^2 r^2) / (1 - r^2))
      red  = 1/2 log(1 / (1 - q^2 r^2))
      syn  = 1/2 log(((1 - q^2)(1 - r^2)) /
                     ((1 - q^2 - r^2)(1 - q^2 r^2)))
    """

    # --- build covariance (X0, X1, Y) ---
    Sigma = np.array([
        [1.0, 0.0, q  ],
        [0.0, 1.0, r  ],
        [q,   r,   1.0]
    ])

    # --- run Idep Gaussian PID ---
    idep_class = Idep_univariate_gauss()
    pid = idep_class.idep(
        cov_matrix=torch.tensor(Sigma, dtype=torch.float64))
    # --- analytic values ---
    unq0_expected = 0.5 * np.log((1.0 - q**2 * r**2) / (1.0 - q**2))
    unq1_expected = 0.5 * np.log((1.0 - q**2 * r**2) / (1.0 - r**2))
    red_expected  = 0.5 * np.log(1.0 / (1.0 - q**2 * r**2))
    syn_expected  = 0.5 * np.log(((1.0 - q**2) * (1.0 - r**2)) /
                                 ((1.0 - q**2 - r**2) * (1.0 - q**2 * r**2)))

    print("=== Expected ===")
    print(f"red  = {red_expected}")
    print(f"unq0 = {unq0_expected}")
    print(f"unq1 = {unq1_expected}")
    print(f"syn  = {syn_expected}")

    print("\n=== Your code ===")
    for k in ["red", "unq0", "unq1", "syn"]:
        print(f"{k} = {pid[k]}")

    # --- checks ---
    assert abs(pid["red"]  - red_expected)  < tol, "red mismatch"
    assert abs(pid["unq0"] - unq0_expected) < tol, "unq0 mismatch"
    assert abs(pid["unq1"] - unq1_expected) < tol, "unq1 mismatch"
    assert abs(pid["syn"]  - syn_expected)  < tol, "synergy mismatch"

    print("\n✅ Example 3 (p=0) PASSED")


def tests():
    
    #Easy covariance matrix to test:
    cov_matrix = torch.tensor([
    [1.0, 0.5, 0],
    [0.5, 1.0, 0],
    [0, 0, 1.0]
    ])
    constraints = ['c_model_1','c_model_2','c_model_3','c_model_4','c_model_5','c_model_6','c_model_7','c_model_8'
    ]
    solver = Idep_univariate_gauss(None,None,None,cov_matrix=cov_matrix)
    res_dict = solver.dependency_matrix(constraints,cov_matrix=cov_matrix)
    print("\nDependency matrix for c_model_7:")
    print(res_dict['c_model_7'])
    dict = solver.compute_Idep(unique=[0,1])
    print("\nI_dep values:")
    print(dict)




if __name__ == "__main__":
    #tests()
    print("="*70)
    print("\nRunning test example 1 from the paper...")
    test_idep_gauss_q0_example()

    print("\nRunning test example 2 from the paper...")
    check_idep_gauss_r0_example()

    print("\nRunning test example 3 from the paper...")
    check_idep_gauss_p0_example()
    print("\nAll tests passed! ✅ ")