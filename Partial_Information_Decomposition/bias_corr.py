import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import OAS
from scipy.special import digamma
import torch
from astropy.stats import jackknife_stats

# --- 1. Helper Functions ---

def entropy_bias_term(df, d):
    """ Analytical bias for Standard MLE (Wishart) """
    return -0.5 * (np.sum([digamma((df - i) / 2.0) for i in range(1, d + 1)]) + d * np.log(2.0 / df))

def matrix_beta_mi_bias(df: float, p: int, q: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    if p == 0 or q == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float64)
        
    i = torch.arange(1, p + 1, device=device, dtype=torch.float64)
    term1 = torch.digamma((df - q - i + 1) / 2.0)
    term2 = torch.digamma((df - i + 1) / 2.0)
    expected_logdet = torch.sum(term1 - term2)
    return -0.5 * expected_logdet 

def entropy_bias_term2(df: float, d: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    if d == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float64)
        
    i = torch.arange(1, d + 1, device=device, dtype=torch.float64)
    digamma_sum = torch.sum(torch.digamma((df - i + 1) / 2.0))
    log_term = d * torch.log(torch.tensor(2.0 / df, device=device, dtype=torch.float64))
    
    bias = -1 * (digamma_sum + log_term)
    return bias

def get_oas_entropy(X):
    """ Helper: Calculate H(X) using OAS covariance """
    oas = OAS(assume_centered=False)
    oas.fit(X)
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
    mi_analytic2 = mi_naive + correction2.item()

    # --- Method C: OAS + Permutation ---
    h_x = get_oas_entropy(X)
    h_y = get_oas_entropy(Y)
    h_z = get_oas_entropy(Z)
    mi_oas_raw = h_x + h_y - h_z
    
    null_mis = []
    Y_shuff = Y.copy()
    for _ in range(5):
        np.random.shuffle(Y_shuff) 
        h_z_null = get_oas_entropy(np.hstack((X, Y_shuff)))
        mi_null = h_x + h_y - h_z_null
        null_mis.append(mi_null)
    
    bias_est = np.mean(null_mis)
    mi_oas_perm = max(0, mi_oas_raw - bias_est)

    # --- Method D: Astropy Jackknife ---
    # Memory/Compute safeguard for Astropy array allocation
    # --- Method D: Astropy Jackknife ---
    # Memory/Compute safeguard for Astropy array allocation
    if N <= 2000:
        def mi_statistic(idx):
            # idx is a 1D array of indices (length N-1) passed by astropy
            # We use it to slice the global Z array safely
            data = Z[idx.astype(int)]
            
            # Recalculate MLE MI for the N-1 subset
            C_x = np.cov(data[:, :dx], rowvar=False, bias=True)
            C_y = np.cov(data[:, dx:], rowvar=False, bias=True)
            C_z = np.cov(data, rowvar=False, bias=True)
            ld_x = np.linalg.slogdet(C_x)
            ld_y = np.linalg.slogdet(C_y)
            ld_z = np.linalg.slogdet(C_z)
            return 0.5 * (ld_x + ld_y - ld_z)

        try:
            # Pass a 1D array of row indices to avoid astropy's flattening bug
            indices = np.arange(N)
            estimate, bias, stderr, conf_interval = jackknife_stats(indices, mi_statistic)
            mi_jackknife = estimate
        except MemoryError:
            mi_jackknife = np.nan
    else:
        mi_jackknife = np.nan

    return mi_naive, mi_analytic, mi_analytic2, mi_oas_perm, mi_jackknife

# --- 2. Simulation Logic ---

def run_simulation():
    np.random.seed(42)
    
    dx = 500
    dy = 500
    p = dx + dy
    
    sample_sizes = [1100, 2000]
    
    # --- SCENARIO 1: ZERO MI (Independent) ---
    print(f"\n{'='*40}\n SCENARIO 1: True MI = 0.0\n{'='*40}")
    print(f"{'N':<6} | {'Naive':<8} | {'Analytic':<8} | {'OAS+Perm':<8} | {'Jackknife':<8}")
    
    res_zero = {'naive': [], 'analytic': [], 'analytic2': [], 'oas': [], 'jackknife': []}
    
    for N in sample_sizes:
        X = np.random.randn(N, dx)
        Y = np.random.randn(N, dy)
        
        naive, analytic, analytic2, oas_perm, jackknife = get_mi_estimates(X, Y, N, dx, dy)
        
        res_zero['naive'].append(naive)
        res_zero['analytic'].append(analytic)
        res_zero['analytic2'].append(analytic2)
        res_zero['oas'].append(oas_perm)
        res_zero['jackknife'].append(jackknife)
        
        jack_str = f"{jackknife:.3f}" if not np.isnan(jackknife) else "N/A"
        print(f"{N:<6} | {naive:>8.3f} | {analytic:>8.3f} | {oas_perm:>8.3f} | {jack_str:>8}")

    # --- SCENARIO 2: POSITIVE MI (Correlated) ---
    print(f"\n{'='*40}\n SCENARIO 2: True MI > 0\n{'='*40}")
    
    A = np.random.randn(p, p)
    True_Sigma = np.dot(A, A.T)
    
    _, ld_x = np.linalg.slogdet(True_Sigma[:dx, :dx])
    _, ld_y = np.linalg.slogdet(True_Sigma[dx:, dx:])
    _, ld_z = np.linalg.slogdet(True_Sigma)
    TRUE_MI = 0.5 * (ld_x + ld_y - ld_z)
    
    print(f"Ground Truth MI: {TRUE_MI:.3f} nats")
    print(f"{'N':<6} | {'Naive':<8} | {'Analytic':<8} | {'OAS+Perm':<8} | {'Jackknife':<8}")

    res_pos = {'naive': [], 'analytic': [], 'analytic2': [], 'oas': [], 'jackknife': []}

    for N in sample_sizes:
        Z = np.random.multivariate_normal(np.zeros(p), True_Sigma, size=N)
        X = Z[:, :dx]
        Y = Z[:, dx:]
        
        naive, analytic, analytic2, oas_perm, jackknife = get_mi_estimates(X, Y, N, dx, dy)
        
        res_pos['naive'].append(naive)
        res_pos['analytic'].append(analytic)
        res_pos['analytic2'].append(analytic2)
        res_pos['oas'].append(oas_perm)
        res_pos['jackknife'].append(jackknife)
        
        jack_str = f"{jackknife:.3f}" if not np.isnan(jackknife) else "N/A"
        print(f"{N:<6} | {naive:>8.3f} | {analytic:>8.3f} | {oas_perm:>8.3f} | {jack_str:>8}")

    # --- 3. Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax = axes
    ax.plot(sample_sizes, res_zero['naive'], 'r-o', label='Naive MLE', linewidth=2)
    ax.plot(sample_sizes, res_zero['analytic'], 'b-s', label='Analytic Corrected', linewidth=2)
    ax.plot(sample_sizes, res_zero['oas'], 'g-^', label='OAS + Permutation', linewidth=2)
    ax.plot(sample_sizes, res_zero['jackknife'], 'm-X', label='Astropy Jackknife', markersize=8, linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', label='True MI (0.0)')
    ax.set_title('Scenario 1: Noise (True MI=0)')
    ax.set_xlabel('Samples (N)')
    ax.set_ylabel('MI (nats)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes
    ax.plot(sample_sizes, res_pos['naive'], 'r-o', label='Naive MLE', linewidth=2)
    ax.plot(sample_sizes, res_pos['analytic'], 'b-s', label='Analytic Corrected', linewidth=2)
    ax.plot(sample_sizes, res_pos['oas'], 'g-^', label='OAS + Permutation', linewidth=2)
    ax.plot(sample_sizes, res_pos['jackknife'], 'm-X', label='Astropy Jackknife', markersize=8, linewidth=2)
    ax.axhline(y=TRUE_MI, color='k', linestyle='--', label=f'True MI ({TRUE_MI:.2f})')
    ax.set_title('Scenario 2: Signal (True MI > 0)')
    ax.set_xlabel('Samples (N)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()