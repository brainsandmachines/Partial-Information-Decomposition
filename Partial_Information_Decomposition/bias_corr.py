from matplotlib.pylab import eigvals
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import OAS
from scipy.special import digamma
import torch
from sklearn.model_selection import LeaveOneOut 
import pandas as pd
from joblib import Parallel, delayed
import time
# --- 1. Helper Functions ---

def entropy_bias_term(N, d):
    """ Analytical bias for Standard MLE (Wishart) """
    # Returns negative value (underestimation of entropy)
    return 0.5 * (np.sum([digamma((N - i + 1) / 2.0) for i in range(1, d + 1)]) + d * np.log(2.0 / N))


# def jackknife_wrapper(raw_output,N,func,data):
#     """This a warpper to calcualte the bias term for a given function
#     using the jackknife method. It takes as input the function and the data, and returns the bias term.
#     Input:
#         raw_output: the output of the function on the whole dataset
#         N: the number of samples in the dataset
#         func: the function for which we want to calculate the bias term. The function should take as input the data and return a scalar value.
#     Output:
#         bias_term: the bias term calculated using the jackknife method."""
    
#     loo = LeaveOneOut()
#     jackknife_estimates = []
#     for train_index, test_index in loo.split(data):
    

# def asymptotic_entropy_bias(df, p):
#     """
#     Calculates the asymptotic bias correction specifically for entropy.
#     Applies the -0.5 factor required when converting a log determinant 
#     into differential entropy.
    
#     Parameters:
#     df (int or float): Degrees of freedom (number of samples - 1).
#     p (int or float): Number of dimensions (features).
    
#     Returns:
#     float: The entropy bias correction term.
#     """
#     return -0.5 * ((p * (p + 1)) / (2.0 * df))

# def jackknife_bias_term(self,X_M1,X_M2,X_T,func):
#     """This function utilized LeaveOneOut resampling to estimate the bias of the logdet estimator for a given dataset X.
#     Args:
#         X (np.ndarray): Input data of shape (n_samples, n_features).
#         X_bar_logdet (np.ndarray): Logdet of the whole sample.
        
#         """
#     if X_M1 is None or X_M2 is None or X_T is None:
#         return {k: torch.tensor(0.0) for k in ['logdetq_jack', 'logdetp_jack', 'logdetr_jack', 'm7_jack', 'm8_jack']}
    
#     loo = LeaveOneOut()
#     keys = ['logdetq_jack','logdetp_jack','logdetr_jack','m7_jack','m8_jack']
#     logdet_dict_jack = {key: [] for key in keys}
#     l = 1
#     for train_idx, test_idx in loo.split(X_M1):
#         print(f"Jackknife iteration {l}/{self.N}", end="\r")
#         X_M1_train = X_M1[train_idx]
#         X_T_train = X_T[train_idx]
#         X_M2_train = X_M2[train_idx]

#         jack_idep = func(sources=[X_M1_train,X_M2_train], targets=[X_T_train], bias_correction=False,verbose=False)

#         P_jack = jack_idep.P
#         Q_jack = jack_idep.Q
#         R_jack = jack_idep.R

            
#         logdetp_jack = torch.logdet(jack_idep.I1 - (P_jack.T @ P_jack))
#         logdetr_jack = torch.logdet(jack_idep.I2 - (R_jack.T @ R_jack))
#         logdetq_jack = torch.logdet(jack_idep.I2 - (Q_jack.T @ Q_jack))
#         assert not torch.isnan(logdetp_jack), f"Jackknife logdetp_jack is NaN at iteration {l}."
#         assert not torch.isnan(logdetr_jack), f"Jackknife logdetr_jack is NaN at iteration {l}."
#         assert not torch.isnan(logdetq_jack), f"Jackknife logdetq_jack is NaN at iteration {l}."

#         dep_mat_jack = jack_idep.dependency_matrix(jack_idep.cov_matrix)
#         #M7:
#         mat = dep_mat_jack['c_model_7']
#         block7_jack = mat[:self.dim_m1, self.dim_m1:self.dim_m1 + self.dim_m2] #Q@R.T
#         nume7_jack = torch.logdet(torch.eye(block7_jack.shape[0]) - (block7_jack.T @ block7_jack))
#         deno7_jack = logdetq_jack + logdetr_jack 
#         m7 = 0.5*(nume7_jack- deno7_jack) 
#         assert not torch.isnan(m7), f"Jackknife m7 is NaN at iteration {l}."
        
#         #M8:s
#         mat_jack = dep_mat_jack['c_model_8']
#         nume8_jack = logdetp_jack
#         deno8_jack = torch.logdet(mat_jack)
#         m8 = 0.5*(nume8_jack-deno8_jack)
#         assert not torch.isnan(m8), f"Jackknife m8 is NaN at iteration {l}."
        
#         logdet_dict_jack['logdetq_jack'].append(logdetq_jack.item())
#         logdet_dict_jack['logdetp_jack'].append(logdetp_jack.item())
#         logdet_dict_jack['logdetr_jack'].append(logdetr_jack.item())
#         logdet_dict_jack['m7_jack'].append(m7.item())
#         logdet_dict_jack['m8_jack'].append(m8.item())
#         l += 1
#     jack_mean = {k: torch.mean(torch.tensor(logdet_dict_jack[k])) for k in logdet_dict_jack}

#     return jack_mean


# def jackknife_pid(self,X_M1,X_M2,X_T,func):
#     """This function utilized LeaveOneOut resampling to estimate the bias of the unique information estimator for a given dataset X.
#     Args:
#         X (np.ndarray): Input data of shape (n_samples, n_features).
#         X_bar_logdet (np.ndarray): Logdet of the whole sample.
        
#         """
#     if X_M1 is None or X_M2 is None or X_T is None:
#         raise ValueError("Input data for Jackknife PID calculation cannot be None.")
    
#     loo = LeaveOneOut()

#     pid_keys = ['unq1_jack','unq2_jack','red_jack','syn_jack']
#     pid_dict_jack = {key: [] for key in pid_keys}

#     mi_keys = ['I(M1;T)_jack','I(M2;T)_jack','I(M1,M2;T)_jack']
#     mi_dict_jack = {key: [] for key in mi_keys}

#     l = 1
#     for train_idx, test_idx in loo.split(X_M1):
#         print(f"Unique Jackknife iteration {l}/{self.N}", end="\r")
#         X_M1_train = X_M1[train_idx]
#         X_T_train = X_T[train_idx]
#         X_M2_train = X_M2[train_idx]

#         jack_idep = func(sources=[X_M1_train,X_M2_train], targets=[X_T_train], bias_correction=False,verbose=False)

#         jack_idep.dependency_matrix()
#         idep_values_jack = jack_idep.compute_Idep()
#         pid_values_jack, mi_values_jack = jack_idep.pid_values(idep_values_jack['unique_1'], idep_values_jack['unique_2'])
        
#         #PID Values
#         pid_dict_jack['unq1_jack'].append(idep_values_jack['unique_1'])
#         pid_dict_jack['unq2_jack'].append(idep_values_jack['unique_2'])
#         pid_dict_jack['red_jack'].append(pid_values_jack['red'])
#         pid_dict_jack['syn_jack'].append(pid_values_jack['syn'])

#         #Mutual Information Values
#         mi_dict_jack['I(M1;T)_jack'].append(mi_values_jack['I(M1;T)'])
#         mi_dict_jack['I(M2;T)_jack'].append(mi_values_jack['I(M2;T)'])
#         mi_dict_jack['I(M1,M2;T)_jack'].append(mi_values_jack['I(M1,M2;T)'])
#         l += 1
#     pid_jack_mean = {k: torch.mean(torch.tensor(pid_dict_jack[k])) for k in pid_dict_jack}
#     mi_jack_mean = {k: torch.mean(torch.tensor(mi_dict_jack[k])) for k in mi_dict_jack}
#     return pid_jack_mean, mi_jack_mean

# def get_oas_entropy(X):
#     """ Helper: Calculate H(X) using OAS covariance """
#     oas = OAS(assume_centered=False)
#     oas.fit(X)
#     # 0.5 * log|Sigma| (ignoring constants for MI diff)
#     sign, logdet = np.linalg.slogdet(oas.covariance_)
#     return 0.5 * logdet if sign > 0 else -np.inf

# def get_mi_estimates(X, Y, N, dx, dy):
#     dz = dx + dy
#     Z = np.hstack((X, Y))
    
#     # --- Method A: Naive MLE ---
#     Cx = np.cov(X, rowvar=False, bias=True)
#     Cy = np.cov(Y, rowvar=False, bias=True)
#     Cz = np.cov(Z, rowvar=False, bias=True)
    
#     def get_logdet(C):
#         sign, ld = np.linalg.slogdet(C)
#         return ld

#     mi_naive = 0.5 * (get_logdet(Cx) + get_logdet(Cy) - get_logdet(Cz))
    
#     # --- Method B: Analytical Correction ---
#     bx = entropy_bias_term(N, dx)
#     by = entropy_bias_term(N, dy)
#     bz = entropy_bias_term(N, dz)
#     correction = bx + by - bz 
#     mi_analytic = mi_naive + correction

#     # --- Method C: OAS + Permutation ---
#     # 1. Raw OAS MI
#     h_x = get_oas_entropy(X)
#     h_y = get_oas_entropy(Y)
#     h_z = get_oas_entropy(Z)
#     mi_oas_raw = h_x + h_y - h_z
    
#     # 2. Permutation (Shuffle Y to find noise floor)
#     # We only need to shuffle a few times to get a stable mean for high N
#     null_mis = []
#     Y_shuff = Y.copy()
#     for _ in range(5):
#         np.random.shuffle(Y_shuff) 
#         h_z_null = get_oas_entropy(np.hstack((X, Y_shuff)))
#         mi_null = h_x + h_y - h_z_null
#         null_mis.append(mi_null)
    
#     bias_est = np.mean(null_mis)
#     mi_oas_perm = max(0, mi_oas_raw - bias_est)
    
#     return mi_naive, mi_analytic, mi_oas_perm


# def theortical_covraince(dx1,dx2,dt,correlation):
#     """ Helper: Create a covariance matrix with specified correlation between X and Y """
#     theoretical_cov = np.eye(dx1 + dx2)
#     theoretical_cov[:dx1, dx1:dx1+dx2] = correlation * np.ones((dx1, dx2))
#     theoretical_cov[dx1:dx1+dx2, :dx1] = correlation * np.ones((dx2, dx1))

#     eigvals = np.linalg.eigvalsh(theoretical_cov)
#     if np.any(eigvals <= 0):
#         raise ValueError("Theoretical covariance is not positive definite.")
#     return theoretical_cov


# def sample_cov_simulation(seed, N, px, py,theo_cov_matrix):
#     """ Helper: Sample data from a Gaussian with specified covariance """
#     rng = np.random.default_rng(seed)
#     mean = np.zeros(px + py)
#     X = rng.multivariate_normal(mean, theo_cov_matrix, size=N)
#     X1 = X[:, :px]
#     X2 = X[:, px:]
#     X1_torch = torch.from_numpy(X1).to(torch.float64)
#     X2_torch = torch.from_numpy(X2).to(torch.float64)
#     rv_list = [X1, X2]
#     cov_blocks = create_cov_matrix(rvs=[X1_torch, X2_torch],verbose=True)
#     sample_covariance = cov_blocks['full_cov'].numpy()
#     return rv_list, sample_covariance

# def mi_simulation(seed, N, px, py, correlation):
    
#     theoretical_cov = theortical_covraince(px, py,dt=0 ,correlation=correlation)
#     rvs, sample_cov = sample_cov_simulation(seed, N, px, py, theoretical_cov)

#     assert sample_cov.shape == theoretical_cov.shape, "Sample covariance shape does not match theoretical covariance shape."
#     assert sample_cov.shape[0] == sample_cov.shape[1], "Sample covariance dimension does not match total dimension (px + py)."
#     p = px + py

#     # Theoretical Entropy Constants
#     _, log_det_theo = np.linalg.slogdet(theoretical_cov)
#     _, log_det_theox1 = np.linalg.slogdet(theoretical_cov[:px, :px])
#     _, log_det_theox2 = np.linalg.slogdet(theoretical_cov[px:, px:])

#     # Sample Entropy
#     _, log_det_sample = np.linalg.slogdet(sample_cov)
#     _, log_det_samplex1 = np.linalg.slogdet(sample_cov[:px, :px])
#     _, log_det_samplex2 = np.linalg.slogdet(sample_cov[px:, px:])

#     # Theoretical MI
#     mi_theoretical = 0.5 * (log_det_theox1 + log_det_theox2 - log_det_theo)
#     # No bias correction MI
#     mi_sample_no_bias = 0.5 * (log_det_samplex1 + log_det_samplex2 - log_det_sample)

#     # Analytical Bias Correction
#     df = N-1
#     bias =   entropy_bias_term(df, px) + entropy_bias_term(df, py) - entropy_bias_term(df, px + py)
#     mi_sample_with_bias = mi_sample_no_bias + bias


#     return mi_theoretical, mi_sample_no_bias, mi_sample_with_bias

# def theoretical_cov_simulation(seeds, N, px, py, correlation):
#     """ Helper: Run multiple simulations to compare theoretical vs sample covariance """
#     results_dict = {}
#     for seed in seeds:
#         mi_theoretical, mi_sample_no_bias, mi_sample_with_bias = mi_simulation(seed, N, px, py, correlation)
#         if seed % 10 == 0:
#             print(f"Completed seed {seed}/{len(seeds)}")
            
#         results_dict[seed] = {'mi_theoretical': mi_theoretical, 'mi_sample_no_bias': mi_sample_no_bias, 'mi_sample_with_bias': mi_sample_with_bias}
#     return results_dict



# --- 2. Simulation Logic ---

# def run_simulation():
#     np.random.seed(42)
    
#     # Dimensions: High dimensional setting
#     dx = 100
#     dy = 100
#     p = dx + dy
    
#     # Sample Sizes
#     sample_sizes = [500,1000]
    
#     # --- SCENARIO 1: ZERO MI (Independent) ---
#     print(f"\n{'='*30}\n SCENARIO 1: True MI = 0.0\n{'='*30}")
#     print(f"{'N':<6} | {'Naive':<10} | {'Analytic':<10} | {'OAS+Perm':<10}")
    
#     res_zero = {'naive': [], 'analytic': [], 'oas': []}
    
#     for N in sample_sizes:
#         X = np.random.randn(N, dx)
#         Y = np.random.randn(N, dy)
        
#         naive, analytic, oas_perm = get_mi_estimates(X, Y, N, dx, dy)
        
#         res_zero['naive'].append(naive)
#         res_zero['analytic'].append(analytic)
#         res_zero['oas'].append(oas_perm)
        
#         print(f"{N:<6} | {naive:.3f}      | {analytic:.3f}      | {oas_perm:.3f}")

#     # --- SCENARIO 2: POSITIVE MI (Correlated) ---
#     # Create a fixed Ground Truth Covariance
#     print(f"\n{'='*30}\n SCENARIO 2: True MI > 0\n{'='*30}")
    
#     # Generate random covariance with signal
#     A = np.random.randn(p, p)
#     True_Sigma = np.dot(A, A.T)
    
#     # Calculate True MI of this matrix
#     _, ld_x = np.linalg.slogdet(True_Sigma[:dx, :dx])
#     _, ld_y = np.linalg.slogdet(True_Sigma[dx:, dx:])
#     _, ld_z = np.linalg.slogdet(True_Sigma)
#     TRUE_MI = 0.5 * (ld_x + ld_y - ld_z)
    
#     print(f"Ground Truth MI: {TRUE_MI:.3f} nats")
#     print(f"{'N':<6} | {'Naive':<10} | {'Analytic':<10} | {'OAS+Perm':<10}")

#     res_pos = {'naive': [], 'analytic': [], 'oas': []}

#     for N in sample_sizes:
#         # Generate data from the True Covariance
#         Z = np.random.multivariate_normal(np.zeros(p), True_Sigma, size=N)
#         X = Z[:, :dx]
#         Y = Z[:, dx:]
        
#         naive, analytic, oas_perm = get_mi_estimates(X, Y, N, dx, dy)
        
#         res_pos['naive'].append(naive)
#         res_pos['analytic'].append(analytic)
#         res_pos['oas'].append(oas_perm)
        
#         print(f"{N:<6} | {naive:.3f}      | {analytic:.3f}      | {oas_perm:.3f}")

#     # --- 3. Plotting ---
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
#     # Plot Zero MI
#     ax = axes[0]
#     ax.plot(sample_sizes, res_zero['naive'], 'r-o', label='Naive MLE', linewidth=2)
#     ax.plot(sample_sizes, res_zero['analytic'], 'b-s', label='Analytic Corrected', linewidth=2)
#     ax.plot(sample_sizes, res_zero['oas'], 'g-^', label='OAS + Permutation', linewidth=2)
#     ax.axhline(y=0, color='k', linestyle='--', label='True MI (0.0)')
#     ax.set_title('Scenario 1: Noise (True MI=0)')
#     ax.set_xlabel('Samples (N)')
#     ax.set_ylabel('MI (nats)')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
    
#     # Plot Positive MI
#     ax = axes[1]
#     ax.plot(sample_sizes, res_pos['naive'], 'r-o', label='Naive MLE', linewidth=2)
#     ax.plot(sample_sizes, res_pos['analytic'], 'b-s', label='Analytic Corrected', linewidth=2)
#     ax.plot(sample_sizes, res_pos['oas'], 'g-^', label='OAS + Permutation', linewidth=2)
#     ax.axhline(y=TRUE_MI, color='k', linestyle='--', label=f'True MI ({TRUE_MI:.2f})')
#     ax.set_title('Scenario 2: Signal (True MI > 0)')
#     ax.set_xlabel('Samples (N)')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     #run_simulation()
#     seeds_list = range(50)
#     N = 500
#     px = 50
#     py = 50
#     correlation = 0
#     sim_results = theoretical_cov_simulation(seeds=seeds_list, N=N, px=px, py=py, correlation=correlation)
#     save_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/Bias_Corr_Sim"
#     df = pd.DataFrame.from_dict(sim_results, orient="index")
#     df.index.name = "seed"
#     df.to_csv(f"{save_path}/MI_bias_corr_simulation_results.csv")  
