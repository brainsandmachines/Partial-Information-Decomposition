import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import OAS
from scipy.special import digamma

# --- 1. Helper Functions ---

def entropy_bias_term(df, d):
    """ Analytical bias for Standard MLE (Wishart) """
    # Returns negative value (underestimation of entropy)
    return -0.5 * (np.sum([digamma((df - i) / 2.0) for i in range(1, d + 1)]) + d * np.log(2.0 / df))


import torch

def matrix_beta_mi_bias(df: float, p: int, q: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Calculates the exact bias for Mutual Information I(X; Y)
    based on the Type I Matrix Variate Beta distribution (Wilks' Lambda).
    
    df: Degrees of freedom (N - 1)
    p: Dimension of the source variable
    q: Dimension of the target variable
    """
    if p == 0 or q == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float64)
        
    i = torch.arange(1, p + 1, device=device, dtype=torch.float64)
    
    # Expected value of ln|I - P^T P| (which is strictly negative)
    term1 = torch.digamma((df - q - i + 1) / 2.0)
    term2 = torch.digamma((df - i + 1) / 2.0)
    expected_logdet = torch.sum(term1 - term2)
    
    # Mutual Information is calculated as -0.5 * ln|I - P^T P|.
    # Because empirical MI overestimates true MI, this positive bias 
    # must be SUBTRACTED from your raw MI calculations.
    mi_bias = -0.5 * expected_logdet 
    
    return mi_bias

def entropy_bias_term2(df: float, d: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """ 
    Analytical bias for Standard MLE (Wishart).
    Returns a negative value (underestimation of entropy).
    Add this term to your raw entropy estimate to correct it.
    """
    if d == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float64)
        
    # i ranges from 1 to d
    i = torch.arange(1, d + 1, device=device, dtype=torch.float64)
    
    # Corrected: (df - i + 1)
    digamma_sum = torch.sum(torch.digamma((df - i + 1) / 2.0))
    
    # d * ln(2/df)
    log_term = d * torch.log(torch.tensor(2.0 / df, device=device, dtype=torch.float64))
    
    bias = -1*(digamma_sum + log_term)
    
    return bias

def get_oas_entropy(X):
    """ Helper: Calculate H(X) using OAS covariance """
    oas = OAS(assume_centered=False)
    oas.fit(X)
    # 0.5 * log|Sigma| (ignoring constants for MI diff)
    sign, logdet = np.linalg.slogdet(oas.covariance_)
    return 0.5 * logdet if sign > 0 else -np.inf
    

def get_mi_estimates(X, Y, N, dx, dy):
    dz = dx + dy
    Z = np.hstack((X, Y))
    
    # --- Method A: Naive MLE ---
    Cx = np.cov(X, rowvar=False, bias=True)
    Cy = np.cov(Y, rowvar=False, bias=True)
    Cz = np.cov(Z, rowvar=False, bias=True)
    
    def get_logdet(C):
        sign, ld = np.linalg.slogdet(C)
        return ld

    mi_naive = 0.5 * (get_logdet(Cx) + get_logdet(Cy) - get_logdet(Cz))
    
    # --- Method B: Analytical Correction ---
    bx = entropy_bias_term(N-1, dx)
    by = entropy_bias_term(N-1, dy)
    bz = entropy_bias_term(N-1 , dz)
    correction = bx + by - bz 
    mi_analytic = mi_naive + correction

    # --- Method B2: Analytical Correction with PyTorch ---
    bx2 = entropy_bias_term2(N-1, dx)
    by2 = entropy_bias_term2(N-1, dy)
    bz2 = entropy_bias_term2(N-1, dz)
    correction2 = bx2 + by2 - bz2
    mi_analytic2 = mi_naive + correction2.item()  # Convert tensor to scalar

    # --- Method C: OAS + Permutation ---
    # 1. Raw OAS MI
    h_x = get_oas_entropy(X)
    h_y = get_oas_entropy(Y)
    h_z = get_oas_entropy(Z)
    mi_oas_raw = h_x + h_y - h_z
    
    # 2. Permutation (Shuffle Y to find noise floor)
    # We only need to shuffle a few times to get a stable mean for high N
    null_mis = []
    Y_shuff = Y.copy()
    for _ in range(5):
        np.random.shuffle(Y_shuff) 
        h_z_null = get_oas_entropy(np.hstack((X, Y_shuff)))
        mi_null = h_x + h_y - h_z_null
        null_mis.append(mi_null)
    
    bias_est = np.mean(null_mis)
    mi_oas_perm = max(0, mi_oas_raw - bias_est)
    
    return mi_naive, mi_analytic,mi_analytic2, mi_oas_perm

# --- 2. Simulation Logic ---

def run_simulation():
    np.random.seed(42)
    
    # Dimensions: High dimensional setting
    dx = 500
    dy = 500
    p = dx + dy
    
    # Sample Sizes
    sample_sizes = [1100, 2000, 5000, 7000, 9000,50000,100000]
    
    # --- SCENARIO 1: ZERO MI (Independent) ---
    print(f"\n{'='*30}\n SCENARIO 1: True MI = 0.0\n{'='*30}")
    print(f"{'N':<6} | {'Naive':<10} | {'Analytic':<10} |{'Analytic2':<10} | {'OAS+Perm':<10}")
    
    res_zero = {'naive': [], 'analytic': [], 'analytic2': [], 'oas': []}
    
    for N in sample_sizes:
        X = np.random.randn(N, dx)
        Y = np.random.randn(N, dy)
        
        naive, analytic, analytic2, oas_perm = get_mi_estimates(X, Y, N, dx, dy)
        
        res_zero['naive'].append(naive)
        res_zero['analytic'].append(analytic)
        res_zero['analytic2'].append(analytic2)
        res_zero['oas'].append(oas_perm)
        
        print(f"{N:<6} | {naive:.3f}      | {analytic:.3f}  | {analytic2:.3f} | {oas_perm:.3f}")

    # --- SCENARIO 2: POSITIVE MI (Correlated) ---
    # Create a fixed Ground Truth Covariance
    print(f"\n{'='*30}\n SCENARIO 2: True MI > 0\n{'='*30}")
    
    # Generate random covariance with signal
    A = np.random.randn(p, p)
    True_Sigma = np.dot(A, A.T)
    
    # Calculate True MI of this matrix
    _, ld_x = np.linalg.slogdet(True_Sigma[:dx, :dx])
    _, ld_y = np.linalg.slogdet(True_Sigma[dx:, dx:])
    _, ld_z = np.linalg.slogdet(True_Sigma)
    TRUE_MI = 0.5 * (ld_x + ld_y - ld_z)
    
    print(f"Ground Truth MI: {TRUE_MI:.3f} nats")
    print(f"{'N':<6} | {'Naive':<10} | {'Analytic':<10} | {'Analytic2':<10} | {'OAS+Perm':<10}")

    res_pos = {'naive': [], 'analytic': [], 'analytic2': [], 'oas': []}

    for N in sample_sizes:
        # Generate data from the True Covariance
        Z = np.random.multivariate_normal(np.zeros(p), True_Sigma, size=N)
        X = Z[:, :dx]
        Y = Z[:, dx:]
        
        naive, analytic, analytic2, oas_perm = get_mi_estimates(X, Y, N, dx, dy)
        
        res_pos['naive'].append(naive)
        res_pos['analytic'].append(analytic)
        res_pos['analytic2'].append(analytic2)
        res_pos['oas'].append(oas_perm)
        
        print(f"{N:<6} | {naive:.3f}      | {analytic:.3f}      | {analytic2:.3f}      | {oas_perm:.3f}")

    # --- 3. Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Zero MI
    ax = axes[0]
    ax.plot(sample_sizes, res_zero['naive'], 'r-o', label='Naive MLE', linewidth=2)
    ax.plot(sample_sizes, res_zero['analytic'], 'b-s', label='Analytic Corrected', linewidth=2)
    ax.plot(sample_sizes, res_zero['analytic2'], 'm-d', label='Analytic2', linewidth=2)
    ax.plot(sample_sizes, res_zero['oas'], 'g-^', label='OAS + Permutation', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', label='True MI (0.0)')
    ax.set_title('Scenario 1: Noise (True MI=0)')
    ax.set_xlabel('Samples (N)')
    ax.set_ylabel('MI (nats)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot Positive MI
    ax = axes[1]
    ax.plot(sample_sizes, res_pos['naive'], 'r-o', label='Naive MLE', linewidth=2)
    ax.plot(sample_sizes, res_pos['analytic'], 'b-s', label='Analytic Corrected', linewidth=2)
    ax.plot(sample_sizes, res_pos['analytic2'], 'm-d', label='Analytic2', linewidth=2)
    ax.plot(sample_sizes, res_pos['oas'], 'g-^', label='OAS + Permutation', linewidth=2)
    ax.axhline(y=TRUE_MI, color='k', linestyle='--', label=f'True MI ({TRUE_MI:.2f})')
    ax.set_title('Scenario 2: Signal (True MI > 0)')
    ax.set_xlabel('Samples (N)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()