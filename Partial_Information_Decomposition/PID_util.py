import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics import r2_score
from itertools import chain, combinations
from typing import List, Tuple, Union
import torch
from torch.linalg import inv, slogdet
from sklearn.covariance import MinCovDet
import statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.covariance import LedoitWolf

def LinearRegression_fit(X,y):
    model = LinearRegression()
    model.fit(X,y)
    return model


def cond_cov(sigma_1,sigma_2,sigma12,sigma21):
    """This function will compute the conditional covariance matrix of two Gaussian variables
    Sigma_1|2 = Sigma_1 - Sigma12*inv(Sigma_2)*Sigma21

    input: sigma_1,sigma_2 are torch tensors of shape (d,d)
    d is the dimension of each observation.

    output: a torch tensor of shape (d,d)
    covariance(sigma_1|sigma_2)"""
    inv_sigma_2 = inv(sigma_2)
    cond_cov = sigma_1 - sigma12 @ inv_sigma_2 @ sigma21
    return cond_cov


def ledoit_wolf_cov_torch(X: torch.Tensor, assume_centered: bool = False) -> torch.Tensor:
    """
    Fit Ledoit-Wolf on X (N,פ) and return covariance as torch.Tensor on same device/dtype.
    """
    X_np = X.detach().cpu().numpy()
    lw = LedoitWolf(assume_centered=assume_centered).fit(X_np)
    Sigma = torch.from_numpy(lw.covariance_).to(device=X.device, dtype=X.dtype)
    return Sigma


def create_cov_matrix(X0,X1,X2):
    """This function will create the covariance matrix for the three variables M1,M2,T
    input: M1,M2,T are torch tensors of shape (N,p) 
    N is the number of observations, 
    p is the dimension of each observation.

    output: a 3*pX3*dp covariance matrix"""

    Z = torch.hstack([X0, X1, X2])   # shape (N, d_T+d_M1+d_M2)

    Sigma = torch.cov(Z.T,correction=1) #Correction means unbiased estimator (N-1 in denominator)
    #eigvenvalue_summary(Sigma.detach().cpu().numpy())
    min_eig, is_singular = block_singularity_check(Sigma.detach().cpu().numpy())
    if is_singular:
        print(f"Warning: Full covariance matrix is singular or ill-conditioned with min eigenvalue: {min_eig:.2e}")

    cov_dict = {}
    print(f"\nFull covariance matrix shape: {Sigma.shape}")
    x0_dim = X0.shape[1]
    x1_dim = X1.shape[1]
    x2_dim = X2.shape[1]
    N = X1.shape[0] #number of observations
    
    dt_dx1 = x0_dim + x1_dim
    d_all = x0_dim + x1_dim + x2_dim

    #Full covariance matrix
    cov_dict['full_cov'] = Sigma #Full covariance matrix ΣX0X1X2

    #Cross-Covariances:
    cov_dict['cross_x0_x1'] = Sigma[0:x0_dim, x0_dim:dt_dx1] #ΣX0,X1
    cov_dict['cross_x0_x2'] = Sigma[0:x0_dim, dt_dx1:d_all] #ΣX0,X2
    cov_dict['cross_x1_x2'] = Sigma[x0_dim:dt_dx1, dt_dx1:d_all]#ΣX1,X2
    cov_dict['cross_x12_x0'] = Sigma[x0_dim:d_all, 0:x0_dim] #ΣX1X2,X0

    #Auto-Covariances
    cov_dict['cov_x0'] = Sigma[0:x0_dim, 0:x0_dim] #ΣX0
    cov_dict['cov_x1'] = Sigma[x0_dim:dt_dx1, x0_dim:dt_dx1] #ΣX1
    cov_dict['cov_x2'] = Sigma[dt_dx1:d_all, dt_dx1:d_all] #ΣX2
    cov_dict['auto_x01'] = Sigma[0:dt_dx1, 0:dt_dx1]  #ΣX0X1
    cov_dict['auto_x12'] = Sigma[x0_dim:d_all, x0_dim:d_all]  #ΣX1X2

 
    ##ΣX0,X2:
    a = torch.cat((cov_dict['cov_x0'], cov_dict['cross_x0_x2']),dim=1)
    b = torch.cat((cov_dict['cross_x0_x2'].T, cov_dict['cov_x2']),dim=1)
    cov_dict['auto_x02'] = torch.cat((a,b),dim=0)


    return cov_dict

def plot_cov_blocks(cov_dict, x0_dim, x1_dim, x2_dim,
                    *, title="Covariance (block view)",
                    cmap="Blues", vmin=None, vmax=None,
                    fine_grid=False, show_colorbar=True):
    Sigma = cov_dict["full_cov"]

    # torch -> numpy
    if hasattr(Sigma, "detach"):
        M = Sigma.detach().cpu().numpy()
    else:
        M = np.asarray(Sigma)

    n = M.shape[0]
    assert n == x0_dim + x1_dim + x2_dim, "dims don't match full_cov"

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="equal")

    # Thick block boundaries
    cuts = [x0_dim, x0_dim + x1_dim]
    for c in cuts:
        ax.axvline(c - 0.5, linewidth=3)
        ax.axhline(c - 0.5, linewidth=3)

    # Optional fine grid for the “pixel lattice” look
    if fine_grid:
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", linewidth=0.25)
        ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/random_plots/{title.replace(' ', '_')}.png", dpi=300)
def standardize(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Standardize columns of X to zero mean and unit variance.

    X shape: (N, P)
    """
    mean = X.mean(dim=0, keepdim=True)
    std  = X.std(dim=0, unbiased=False, keepdim=True)
    return (X - mean) / (std + eps)



def assert_full_rank(X: torch.Tensor,jitter=0) -> None:
    """
    Assert that the input matrix X is full rank.

    X shape: (N, P)
    """
    n,m = X.shape
    full_rank = min(n,m)

    rank = torch.linalg.matrix_rank(X)
    if rank < full_rank and jitter > 0:
        # Try adding jitter to the diagonal and recompute rank
        print(f"Matrix is rank-deficient (rank={rank}). Adding jitter to check for full rank.")
        jitter_matrix = jitter * torch.eye(n, m)
        rank = torch.linalg.matrix_rank(X + jitter_matrix)
        X += jitter_matrix
        print(f"New rank after adding jitter: {rank}")
        return X
    if rank < full_rank:
        raise ValueError(f"Input matrix is rank-deficient (rank={rank}, expected full rank={full_rank}).")

    return X


def correlation_matrix(X):
    """Compute the correlation matrix of the columns of X."""
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    stddev = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(stddev, stddev)
    return corr_matrix

def block_singularity_check(X, tol=1e-10):
    """Check if a block is singular or ill-conditioned."""

    if len(X.shape) != 2:
        singular_dict = {}
        for i, block in enumerate(X):
            min_eig, is_singular = block_singularity_check(block, tol)
            singular_dict[i] = (min_eig,is_singular)
        if any(is_singular for _, (_, is_singular) in singular_dict.items()):
            for i, (min_eig, is_singular) in singular_dict.items():
                status = "SINGULAR" if is_singular else "OK"
                if is_singular:
                    print(f"Block {i}: min eigenvalue = {min_eig:.2e} -> {status}")
        return singular_dict
    
    if X.shape[0] != X.shape[1]:
        raise ValueError("Input isn't a square matrix. it might no be a co-variance matrix")
    eigvals = np.linalg.eigvalsh(X)
    min_eig = float(eigvals.min())
    return min_eig, min_eig <= tol

def singularity_report(X_M1, X_M2, y_real, tol=1e-10):
    """Return min eigenvalue and singularity flag for blocks and combinations."""
    blocks = {
        "M1": X_M1,
        "M2": X_M2,
        "Y": y_real,
        "M1+M2": np.hstack([X_M1, X_M2]),
        "M1+Y": np.hstack([X_M1, y_real]),
        "M2+Y": np.hstack([X_M2, y_real]),
        "M1+M2+Y": np.hstack([X_M1, X_M2, y_real]),
    }
    report = {}
    printing_required = False
    for name, block in blocks.items():
        min_eig, is_singular = block_singularity_check(block, tol)
        report[name] = {"min_eigval": min_eig, "is_singular": is_singular}
        if is_singular:
            printing_required = True
    # print report if any block is singular or ill-conditioned
    if printing_required:
        for name, info in report.items():
            status = "SINGULAR" if info["is_singular"] else "OK"
            print(f"Block {name}: min eigenvalue = {info['min_eigval']:.2e} -> {status}")
    return report,printing_required

def diagnostic_plots(X_M1, X_M2, y_real, method, mixing_dimension):
    def cross_correlation(X, Y):
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        n = Xc.shape[0] - 1
        cov = (Xc.T @ Yc) / n
        sx = np.sqrt(np.diag((Xc.T @ Xc) / n))
        sy = np.sqrt(np.diag((Yc.T @ Yc) / n))
        with np.errstate(divide='ignore', invalid='ignore'):
            return cov / np.outer(sx, sy)

    blocks, labels = [X_M1, X_M2, y_real], ["M1", "M2", "Y"]
    counts = [b.shape[1] for b in blocks]
    fig = plt.figure(figsize=(9, 9))
    gs = plt.GridSpec(3, 3, width_ratios=counts, height_ratios=counts, wspace=0.05, hspace=0.05)
    axes, im = [], None
    for i in range(3):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            corr = cross_correlation(blocks[i], blocks[j])
            im = ax.imshow(corr, cmap='bwr', vmin=-1, vmax=1, aspect='auto')
            max_text = "max: n/a" if np.all(np.isnan(corr)) else f"max: {np.nanmax(corr):.2f}"
            ax.text(0.02, 0.98, max_text, transform=ax.transAxes, ha='left', va='top', fontsize=8,
                    color='black', bbox={'boxstyle': 'round,pad=0.2', 'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none'})
            if i == 0:
                ax.set_title(labels[j])
            if j == 0:
                ax.set_ylabel(labels[i])
            ax.set_xticks([]); ax.set_yticks([])
            axes.append(ax)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02).set_label('Correlation Coefficient')
    fig.suptitle(f'Correlation Matrix - Method: {method}, Mixing Dim: {mixing_dimension}')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def vif_summary(X):
    if len(X.shape) == 2:
          X = np.expand_dims(X, axis=0)
    max_vif_list = []
    for block in X:
        block = StandardScaler().fit_transform(block)
        
        vifs = np.array([variance_inflation_factor(block, i) 
                        for i in range(block.shape[1])])
        
        R = np.corrcoef(block, rowvar=False)
        eigvals = np.linalg.eigvalsh(R)
        max_vif_list.append(np.max(vifs))
        print("\n===== VIF Summary =====")
        print("Max VIF:", np.max(vifs))
        print("Mean VIF:", np.mean(vifs))
        print("Median VIF:", np.median(vifs))
        print("Num > 10:", np.sum(vifs > 10))
        print("Min eigenvalue (corr):", np.min(eigvals))
        print("Condition number (corr):", eigvals.max()/eigvals.min())
        
    
    for max_vif in max_vif_list:
        if max_vif > 10:
            print("Warning: High multicollinearity detected (max VIF > 10). Consider removing or combining features.")


import numpy as np

def std_scaling_summary(X):
    if len(X.shape) == 2:
        X = np.expand_dims(X, axis=0)
        
    for block in X:
        block = np.asarray(block)

        # Standard deviations
        stds = np.std(block, axis=0, ddof=1)
        variances = stds**2
        print("\n===== STD Scaling Summary =====")
        print("Max std:", np.max(stds))
        print("Min std:", np.min(stds))
        print("Std ratio (max/min):", np.max(stds)/np.min(stds))
        print("Max variance:", np.max(variances))
        print("Min variance:", np.min(variances))
        print("Variance ratio (max/min):", np.max(variances)/np.min(variances))


def eigvenvalue_summary(X):
    assert X.shape[0] == X.shape[1], "Input must be a square matrix to compute eigenvalues."
    eigvals = np.linalg.eigvalsh(X)
    print("\n===== Eigenvalue Summary =====")
    print("Min eigenvalue:", np.min(eigvals))
    print("Max eigenvalue:", np.max(eigvals))
    print("Eigenvalue ratio (max/min):", np.max(eigvals)/np.min(eigvals))

def compare_results(vp_results,pid_results,mi_results):
    """Compare Variance Partitioning and Partial Information Decomposition results.
    Parameters
    ----------
    vp_results : dict
        Results from variance partitioning.
    pid_results : dict
        Results from Partial Information Decomposition.
    mi_results : dict
        Mutual information results from PID.

    Returns
    -------
    None
    """
    print("\n" + "="*70)
    print("Comparison of Variance Partitioning and PID Results")
    print("="*70)
    print("Results:")
    print(f"M1 R² (VP): {vp_results['R²_X1']:.4f} | I(T;M1): {mi_results['I(M0;T)']:.4f}")
    print(f"M2 R² (VP): {vp_results['R²_X2']:.4f} | I(T;M2): {mi_results['I(M1;T)']:.4f}")
    print(f"Both M1 and M2 R² (VP): {vp_results['R²_X12']:.4f} | I(T;M1,M2): {mi_results['I(M0,M1;T)']:.4f}")
    print(f"\nUnique to M1 (VP): {vp_results['unique_X1']:.4f} | Unique(T;M1\\M2): {pid_results['unq0']:.4f}")
    print(f"Unique to M2 (VP): {vp_results['unique_X2']:.4f} | Unique(T;M2\\M1): {pid_results['unq1']:.4f}")
    print(f"Common (VP): {vp_results['common']:.4f} | Redundant (PID): {pid_results['red']:.4f}")
    print(f"Synergy (PID): {pid_results['syn']:.4f}")