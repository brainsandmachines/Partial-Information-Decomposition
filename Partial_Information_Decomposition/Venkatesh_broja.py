from collections import deque
import numpy as np
import torch
from RV_toy import generate_random_variables
from torch.distributions import Distribution, Normal 
from numpy.linalg import inv, det, slogdet, eigh, pinv
from numpy import block, sign, ones, log , eye
from scipy.linalg import sqrtm




class broja_venkatesh():
    def __init__(self,M1,M2,T,vec_size,bias=False):
        
        self.bias = bias
        self.gamma = 1e-12
        #Initialize the class with the three variables M1,M2,T
        self.M1 = M1
        self.M2 = M2
        self.T = T
        self.vec_size = vec_size
        
        #if not (isinstance(M1, np.ndarray) or isinstance(M2, np.ndarray) or isinstance(T, np.ndarray)):
            #raise ValueError("M1, M2, and T must be numpy arrays.")

        #Define dimension for futher use
        self.m1_dim = M1.shape[1]
        self.m2_dim = M2.shape[1]
        self.t_dim = T.shape[1]
        self.N = M1.shape[0] #number of observations
        
        self.dt_dm1 = self.t_dim + self.m1_dim
        self.dt_dm2 = self.t_dim + self.m2_dim
        self.dm1_dm2 = self.m1_dim + self.m2_dim
        self.d_all = self.t_dim + self.m1_dim + self.m2_dim

        # Create the matrix ΣTM1M2
        self.create_cov_matrix()

        # Store last 20 solutions to check for convergence
        self.solutions_queue = deque(maxlen=20)

        # Define the unique information for each variable
        self.unique_t_m1 = None
        self.unique_t_m2 = None
        self.gradient_descend_converged = False

        #Define the constant difference that will be used to find the real solution after gradient descent
        """We can use ΣX,Y|M in place of ΣX,Y because they differ by a constant: ΣX,Y|M - ΣX,Y is an off-diagonal
        block in ΣXY - ΣXY |M, which is equal to ΣXY,M @ inv(ΣM) @ transpose(ΣXY,M)
        which is constant across P and Q Venkatesh et al 2023"""

        self.constant_auto_cov = (self.cov_dict['cross_cov_m12_t'] @ inv(self.cov_dict['cov_t']) @ self.cov_dict['cross_cov_m12_t'].T)
        self.constant_diff = self.constant_auto_cov[0:self.m1_dim,self.m1_dim:self.m1_dim+self.m2_dim] #Equals to   ΣM1,M2 - ΣM1,M2|T
        

    def create_cov_matrix(self):
        """This function will create the covariance matrix for the three variables M1,M2,T
        input: M1,M2,T are torch tensors of shape (N,d) 
        N is the number of observations, 
        d is the dimension of each observation.

        output: a 3*dX3*d covariance matrix"""
        # Stack all variables side by side
        Z = np.hstack([self.T, self.M1, self.M2])   # shape (N, d_T+d_M1+d_M2)

        self.Sigma = np.cov(Z, rowvar=False, bias=False) #WARNING bias=False or True?
        
        self.cov_dict = {}

        self.cov_dict['cov_t'] = self.Sigma[0:self.t_dim, 0:self.t_dim] #ΣT
        self.cov_dict['cov_m1'] = self.Sigma[self.t_dim:self.dt_dm1, self.t_dim:self.dt_dm1] #ΣM1
        self.cov_dict['cov_m2'] = self.Sigma[self.dt_dm1:self.d_all, self.dt_dm1:self.d_all] #ΣM2
        self.cov_dict['cov_t_m1'] = self.Sigma[0:self.t_dim, self.t_dim:self.dt_dm1] #ΣT,M1
        self.cov_dict['cov_t_m2'] = self.Sigma[0:self.t_dim, self.dt_dm1:self.d_all] #ΣT,M2
        self.cov_dict['cross_cov_m1_m2'] = self.Sigma[self.t_dim:self.dt_dm1, self.dt_dm1:self.d_all]#ΣM1,M2
        self.cov_dict['cross_cov_m12_t'] = self.Sigma[self.t_dim:self.d_all, 0:self.t_dim] #ΣM1M2,T #TODO: Not sure if its the correct shape chatGPT says its correct
        self.cov_dict['auto_cov_m12'] = self.Sigma[self.t_dim:self.d_all, self.t_dim:self.d_all]  #ΣM1M2 

        return 


    def create_positive_semidefinite(self,matrix):
            """This function creates the component of the projection step
            from Venkatesh et al 2023
            ΣM1,M2|T := VΛ(bar)VT
            Let λi(bar) := max(0,λi)
            Λ(bar) := diag(λi(bar)).

            where  ΣXY|M =: VΛVT is the eigendecomposition of ΣXY|M
            (This will be done on ΣM1|T and ΣM2|T as well )

            input: ΣM1,M2|T which is the off digonal block of ΣXY|M
            """
            #Create ΣXY|M according to the paper appendix shape(2*vec_size,2*vec_size)

            eig_decomp = eigh(matrix)
            eig_values = eig_decomp[0]
            eig_vectors = eig_decomp[1]
            
            self.id = eye(eig_values.shape[0])
            max_eig_diag = np.diag(np.maximum(eig_values, 0))

            conditional_cov = eig_vectors @ max_eig_diag @ eig_vectors.T
            return conditional_cov

    def projection_step(self, sol):
        """This function is the projection operator from Venkatesh et al 2023
        It will project the solution to the space of valid covariance matrices
        input: sol is the current solution ΣM1,M2|T
        output: projected solution"""

        
        conditional_m1 = self.cov_dict['cov_m1'] - self.cov_m1_t @ inv(self.cov_dict['cov_t']) @ self.cov_m1_t.T
        conditional_m2 = self.cov_dict['cov_m2'] - self.cov_m2_t @ inv(self.cov_dict['cov_t']) @ self.cov_m2_t.T

        conditional_cov_m12_t = block([[conditional_m1, sol],
                                    [sol.T, conditional_m2]]) #TODO: Check if the identitiy needs to be there or somemthing eles

        assert conditional_cov_m12_t.shape == (self.m1_dim + self.m2_dim, self.m1_dim + self.m2_dim)

        positive_semidef = self.create_positive_semidefinite(conditional_cov_m12_t) #ΣM1M2|T

        cov_m1_2_cond_t = positive_semidef[0:self.m1_dim, self.m1_dim:self.m1_dim+self.m2_dim] ##ΣX,Y |M maybe I can just use sol (?)


        bar_conditional_m1 = positive_semidef[0:self.m1_dim, 0:self.m1_dim] #ΣM1|T
        bar_conditional_m2 = positive_semidef[self.m1_dim:self.m1_dim+self.m2_dim, self.m1_dim:self.m1_dim+self.m2_dim] #ΣM2|T

        bar_conditional_m1_sqrt = sqrtm(bar_conditional_m1)
        bar_conditional_m2_sqrt = sqrtm(bar_conditional_m2)

        semi_def = False
        j = 0

        while not semi_def:
            """where γ is slowly increased from 10-12, by a factor of 10 in each step, until g(Σproj
                X,Y |M) ≽ 0 (Venkatesh et al 2023)"""
            j += 1
            projected_sol = inv((self.gamma*eye(conditional_m1.shape[0])) +  bar_conditional_m1_sqrt) @ cov_m1_2_cond_t @ inv((self.gamma*eye(conditional_m2.shape[0]) + bar_conditional_m2_sqrt))

            if all(eigh(projected_sol)[0] >= 0):
                semi_def = True
                self.gamma = 1e-12 #Reset gamma for next projection step
                print(f"Projection step converged after {j} iterations with eigenvalue::={eigh(projected_sol)[0]}")
                j = 0 #Reset j for next projection step

            else:
                self.gamma *= 10

        return projected_sol
    
    
    def stopping_criteria(self,sol):
        """Check if the last 20 solutions are converged"""
        if len(self.solutions_queue) >= 20:
            array_solutions = np.array(self.solutions_queue)
            if np.all(np.allclose(sol, array_solutions, atol=1e-6)):
                return True
            self.solutions_queue.popleft()
            self.solutions_queue.append(sol)
            
        else:
            self.solutions_queue.append(sol)
        return False

    def gradient_descent(self, alpha=0.999, num_iterations=1000):
        """This function will perform gradient descent to minimize the Union information
        given the covariance matrix.
        
        Args:
            cov_matrix (np.ndarray): The covariance matrix of shape (3d, 3d).
            learning_rate (float): The learning rate for gradient descent.
            num_iterations (int): The number of iterations for gradient descent."""

        """
        B = (H(M1) - Σ(M1M2|T) @ H(M2))
        H(M1) = cov(M1,T)
        H(M2) = cov(M2,T) 

        S = (gamma*I - ΣM1,M2|T @ ΣM1,M2|T(transpose))
        S^-1 = inv(gamma*I - ΣM1,M2|T @ ΣM1,M2|T(transpose))
        """
        print(f"\nStarting gradient descent...")
        print(f"\nWith paprameters: learning_rate={alpha}, num_iterations={num_iterations}")
        print(f"\n------------------------------------(-_-)-------------------------------------------")
       
        #Define H(M1) and H(M2)
        self.cov_t_m1 = self.cov_dict['cov_t_m1']  #Σ(M1,T)
        self.cov_t_m2 = self.cov_dict['cov_t_m2']  #Σ(M2,T)

        self.cov_m1_t = self.cov_t_m1.T  #H(M1) = Σ(M1,T).T
        self.cov_m2_t = self.cov_t_m2.T  #H(M2) = Σ(M2,T).T
        #define solution:  ΣX,Y|M
        sol_0 = self.projection_step(self.cov_m1_t @ pinv(self.cov_m2_t)) 

        #Define parameters
        mue = ones((sol_0.shape[0],sol_0.shape[1]))*1e-3  # Step size parameter for Rprop
        beta =ones((sol_0.shape[0],sol_0.shape[1]))*0.9   # Decay factor for Rprop
        epsilon = 1e-7
        
        #Define B
        B = self.cov_m1_t - sol_0 @ self.cov_m2_t #B = H(M1) - Σ(M1,M2|T) @ H(M2)

        # Define S
        I_t = eye(self.t_dim)
        S = eye(sol_0.shape[0]) - sol_0 @ sol_0.T #Not sure if this is needed
        inv_S = inv((1+epsilon)*eye(sol_0.shape[0]) - sol_0 @ sol_0.T)



        for i in range(num_iterations):
            """Preforming gradient descent to minimize the objective function
            using Rprop algorithm"""
            # Compute gradients'

            
            
            if i == 0:
                sol = sol_0

                # Compute gradient
            grad = inv_S@ B @ inv(I_t + self.cov_m2_t.T @ self.cov_m2_t + B.T @ inv_S @ B) @ (B.T @ inv_S @ sol - self.cov_m2_t.T)
            sign_grad_curr = sign(grad)

            if i > 0:
                mue *=  beta**(-(sign_grad_curr*sign_grad_prev))

            
            
            # Update solution
            sol -= ((alpha)**i) * mue * sign_grad_curr # *  - Hadamard product
            if np.isnan(sol).any():
                print("NaN detected in solution, stopping gradient descent.")
                break

            # Projection step to ensure the solution remains valid
            sol_new = self.projection_step(sol)

            # Update B,S and mue for next iteration
            B = self.cov_m1_t - sol_new @ self.cov_m2_t

            S = eye(sol_0.shape[0]) - sol_new @ sol_new.T #Not sure if this is needed
            inv_S = inv((1+epsilon)*eye(sol_0.shape[0]) - sol_new @ sol_new.T)

            #Keep previous gradient sign
            sign_grad_prev = sign_grad_curr

            #Update solution
            sol = sol_new

            # Check for convergence
            if self.stopping_criteria(sol):
                print(f"Converged after {i} iterations.")
                break


        #The real solution that needs to be found is ΣM1,M2
        # Update the covariance matrix with the new solution everything else is constant
        self.cov_dict['cross_cov_m1_m2'] = sol + self.constant_diff #ΣM1,M2 (= ΣM1,M2|T + constant difference)
        self.cov_dict['m1_2_cond_t'] = sol #ΣM1,M2|T
        self.gradient_descend_converged = True

        return self.cov_dict['cross_cov_m1_m2']

    def unique_pid(self): #TODO: finish this function
        """This function will compute the unique information U(T;M1\M2) and U(T;M2\M1)
        using the covariance matrix obtained from gradient descent."""

        if not self.gradient_descend_converged:
            print(f"\nStarting gradient descent with default parameters...")
            self.gradient_descent()
        
       
        cov_t = self.cov_dict['cov_t']
        
        
        cov_m12_t = self.cov_dict['cross_cov_m12_t'] #ΣM1M2,T
        cov_m1_2_cond_t = self.cov_dict['m1_2_cond_t'] #ΣM1,M2|T
        cov_m1_cond_t = self.cov_dict['cov_m1'] - self.cov_m1_t @ inv(cov_t) @ self.cov_m1_t.T #ΣM1|T
        cov_m2_cond_t = self.cov_dict['cov_m2'] - self.cov_m2_t @ inv(cov_t) @ self.cov_m2_t.T #ΣM2|T

        cov_m12_cond_t = block([[cov_m1_cond_t, cov_m1_2_cond_t],
                                  [cov_m1_2_cond_t.T, cov_m2_cond_t]]) #ΣM1M2|T
        #Find I(T;M1,M2):
        no_name = inv(cov_t) @ cov_m12_t.T @ inv(cov_m12_cond_t) @ cov_m12_t #need to find this a name
        identity = eye(no_name.shape[0])
        
        log_det_info_t_m1_m2 = slogdet(identity + no_name)

        self.info_t_m1_m2 = 0.5*log_det_info_t_m1_m2[1] #More stable the log(det(.))
        
        print(f"\nI(T;M1,M2) = {self.info_t_m1_m2}")



        #Find I(T;M1)
        _, m1_logdet = slogdet(2*np.pi*np.e*self.cov_dict['cov_m1']) #More stable the log(det(.))
        entropy_m1 = 0.5*m1_logdet

        _, m1_logdet_cond_t = slogdet(2*np.pi*np.e*cov_m1_cond_t)
        entropy_m1_cond_t = 0.5*m1_logdet_cond_t

        self.info_t_m1 = entropy_m1 - entropy_m1_cond_t
        print(f"I(T;M1) = {self.info_t_m1}")


        #Find I(T;M2)
        _, m2_logdet = slogdet(2*np.pi*np.e*self.cov_dict['cov_m2']) #More stable the log(det(.))
        entropy_m2 = 0.5*m2_logdet

        _, m2_logdet_cond_t = slogdet(2*np.pi*np.e*cov_m2_cond_t)
        entropy_m2_cond_t = 0.5*m2_logdet_cond_t

        self.info_t_m2 = entropy_m2 - entropy_m2_cond_t
        print(f"I(T;M2) = {self.info_t_m2}")


     
        
        #Unique information for U(T;M1\M2)
        unique_t_m1 = self.info_t_m1_m2 - self.info_t_m1
        unique_t_m2 = self.info_t_m1_m2 - self.info_t_m2

        if self.bias:
            print(f"\nApplying bias correction...")

            bias_m1_m2_t = self.bias_correction(self.t_dim,self.m1_dim+self.m2_dim,self.N)
            self.info_t_m1_m2_corrected = self.info_t_m1_m2 * (1 - (bias_m1_m2_t/self.info_t_m1_m2))
            print(f"I(T;M1,M2)[corr] = {self.info_t_m1_m2_corrected}")

            #TODO: Check if bias need to be applied to I(T;M1) and I(T;M2) and what bias?
            bias_m1_t = self.bias_correction(self.t_dim,self.m1_dim,self.N)
            self.info_t_m1_corrected = self.info_t_m1 * (1 - (bias_m1_t/self.info_t_m1))
            print(f"I(T;M1)[corr] = {self.info_t_m1_corrected}")

            bias_m2_t = self.bias_correction(self.t_dim,self.m2_dim,self.N)
            self.info_t_m2_corrected = self.info_t_m2 * (1 - (bias_m2_t/self.info_t_m2))
            print(f"I(T;M2)[corr] = {self.info_t_m2_corrected}")

            unique_t_m1 = self.info_t_m1_m2_corrected - self.info_t_m2 #NOTE: For now no bias correction on I(T;M2) 
            unique_t_m2 = self.info_t_m1_m2_corrected - self.info_t_m1 #NOTE: For now no bias correction on I(T;M1) 


        else:
            print(f"\nUnique information U(T;M1\M2) = {unique_t_m1}")
            print(f"\nUnique information U(T;M2\M1) = {unique_t_m2}")

        print("------------------------------------")
        print("Partial Information Decomposition complete!")
        return dict(U_T_M1=unique_t_m1, U_T_M2=unique_t_m2)

    def bias_correction(self,p,q,sample_size):  
        """This function will perform bias correction on the unique information
        using the method described in Venkatesh et al 2023"""
        p_array = np.arange(1,p+1)
        q_array = np.arange(1,q+1)
        p_q_array = np.arange(1,p+q+1)

        bias_p = np.sum(log(1 - p_array/sample_size))
        bias_q = np.sum(log(1 - q_array/sample_size))
        bias_p_plus_q = np.sum(log(1 - p_q_array/sample_size))
        
        bias = (bias_p + bias_q - bias_p_plus_q)
        return bias

    def compute_decomposition(self):
        """This function will compute the partial information decomposition
        using the covariance matrix obtained from gradient descent."""
        if (self.unique_t_m1) or (self.unique_t_m2 is None):
            print("Computing Partial Information Decomposition...")
            print("------------------------------------")
            pid = self.unique_pid()
            self.unique_t_m1 = pid['U_T_M1']
            self.unique_t_m2 = pid['U_T_M2']
            
            self.redundant_m1_ = self.info_t_m1 - self.unique_t_m1
            self.redundant_m2_ = self.info_t_m2 - self.unique_t_m2
            #print(f"\nRedundant information from M1: R(T;M1) = {self.redundant_m1_}")
            #print(f"\nRedundant information from M2: R(T;M2) = {self.redundant_m2_}")
            assert np.isclose(self.redundant_m1_, self.redundant_m2_,atol=1e-1), "Redundant information from M1 and M2 are not equal"
            self.redundant =self.redundant_m1_
            
            self.synergy = self.info_t_m1_m2 - self.unique_t_m1 - self.unique_t_m2 - self.redundant
            print(f"\nRedundant information R(T;M1,M2) = {self.redundant.item()}")
            print(f"\nSynergy information S(T;M1,M2) = {self.synergy.item()}")
            print(f"\nUnique(T;M1\M2) = {self.unique_t_m1.item()}")
            print(f"\nUnique(T;M2\M1) = {self.unique_t_m2.item()}")

        return dict(Unique_M1=self.unique_t_m1, Unique_M2=self.unique_t_m2, Redundant_M1_M2=self.redundant, Synergy_M1_M2=self.synergy)



        

        # Compute the covariance matrices needed for PID calculation


def main():
    num_obsv = 300
    vec_size = 1
    dist = Normal(loc=0.0, scale=1.0)
    random_variables = ['S1', 'S2','S12','N1', 'N2','NT']

    def toy_example(rv_dict):
        m1 = rv_dict['S12'] + rv_dict['S1'] + rv_dict['N1']
        m2 = rv_dict['S12'] + rv_dict['S2'] + rv_dict['N2']
        target = rv_dict['S12'] + rv_dict['S1'] + rv_dict['S2'] +rv_dict['NT']
        return {'M1':m1, 'M2':m2, 'target':target}

    result, _ = generate_random_variables(toy_example, num_obsv,vec_size ,dist, *random_variables)

    broja = broja_venkatesh(result['M1'], result['M2'], result['target'],vec_size=vec_size,bias=True)
    broja.compute_decomposition()

    """print(broja.cov_matrix)
    print("Covariance matrix shape:", broja.cov_matrix.shape)
    print(broja.cov_matrix[vec_size:2*vec_size, 0:vec_size])
    print(np.diag([1,1]))"""

if __name__ == "__main__":
    main()