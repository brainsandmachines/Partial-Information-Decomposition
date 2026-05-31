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
import os
from matplotlib.colors import LogNorm
from sklearn.linear_model import RidgeCV, LinearRegression


def LinearRegression_fit(X,y):
    model = LinearRegression()
    if X.device != 'cpu':
        X = X.cpu().numpy()
    if y.device != 'cpu':
        y = y.cpu().numpy()
    model.fit(X,y)
    return model




def compute_ridge_cv_r2(X, y, alphas=None):
    """
    Compute cross-validated R² using RidgeCV with efficient LOO cross-validation.
    
    RidgeCV uses generalized cross-validation (GCV) which is an efficient 
    approximation to leave-one-out CV for ridge regression.
    
    Args:
        X (np.ndarray): Design matrix WITHOUT intercept (shape: [n, p]).
        y (np.ndarray): Target variable (shape: [n,]).
        alphas (array-like, optional): Array of alpha values to try.
            Defaults to DEFAULT_RIDGE_ALPHAS.
        
    Returns:
        float: Best cross-validated R² across all alpha values.
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 50)
    
    # RidgeCV with leave-one-out CV (efficient GCV approximation)
    # cv=None means use efficient LOO via GCV
    ridge_cv = RidgeCV(alphas=alphas, fit_intercept=True, scoring='r2', cv=None)
    
    if X.device != 'cpu':
        X = X.cpu().numpy()
    if y.device != 'cpu':
        y = y.cpu().numpy()
    
    ridge_cv.fit(X, y)
    
    return ridge_cv.best_score_, ridge_cv



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


def create_cov_matrix(rvs:list=[],verbose=False,Sigma=None,dims:list=None,device='cpu',check_singular=True):
    """This function will create the covariance matrix for the three variables M1,M2,T
    input: M1,M2,T are torch tensors of shape (N,p) 
    rvs is a list of the three variables [M1,M2,T]
    N is the number of observations, 
    p is the dimension of each observation.

    output: a len(rvs)*len(rvs)*p covariance matrix"""

    if Sigma is None:
        assert len(rvs) == 2 or len(rvs) == 3, "Length of random variable list should be either 2 or 3"
        Z = torch.hstack(rvs).to(torch.float64)   # shape (N, len(rvs)*len(rvs)*p)    
        Sigma = torch.cov(Z.T,correction=1) #Correction means unbiased estimator (N-1 in denominator)
        Sigma = Sigma.to(device)
    
    if verbose:
        eigvenvalue_summary(Sigma.detach().cpu().numpy())

    if check_singular:
        min_eig, is_singular = block_singularity_check(Sigma.detach().cpu().numpy())
        if is_singular:
            print(f"Warning: Full covariance matrix is singular or ill-conditioned with min eigenvalue: {min_eig:.2e}")

    cov_dict = {}
    if verbose:
        print(f"\nFull covariance matrix shape: {Sigma.shape}")
    x1_dim = rvs[0].shape[1] if rvs else dims[0]
    x2_dim = rvs[1].shape[1] if rvs else dims[1]
    xt_dim = rvs[2].shape[1] if (rvs and len(rvs) == 3) else dims[2] if dims else 0


    dt_dx2 = x2_dim + xt_dim
    dx1_dx2 = x1_dim + x2_dim
    d_all = x1_dim + x2_dim + xt_dim

    #Full covariance matrix
    cov_dict['full_cov'] = Sigma #Full covariance matrix ΣX1X2T
    #Cross-Covariances:
    cov_dict['cross_x1_x2'] = Sigma[0:x1_dim, x1_dim:dx1_dx2] #ΣX1,X2   
    #Auto-Covariances
    cov_dict['cov_x1'] = Sigma[0:x1_dim, 0:x1_dim] #ΣX1
    cov_dict['cov_x2'] = Sigma[x1_dim:dx1_dx2, x1_dim:dx1_dx2] #ΣX2
   
    cov_dict['joint_x1_x2'] = Sigma[0:dx1_dx2, 0:dx1_dx2]  #ΣX1X2


    if (rvs and len(rvs) == 3) or (dims and len(dims) == 3):
        #Cross-Covariances:
        cov_dict['cross_x2_t'] = Sigma[x1_dim:dx1_dx2, dx1_dx2:d_all]#ΣX2,XT
        cov_dict['cross_x2t_x1'] = Sigma[x1_dim:d_all, 0:x1_dim] #ΣX2XT,X1
        cov_dict['cross_x1_t'] = Sigma[0:x1_dim, dx1_dx2:d_all] #ΣX1,XT

        #Auto-Covariances:
        cov_dict['joint_x2_t'] = Sigma[x1_dim:d_all, x1_dim:d_all]  #ΣX2XT
        cov_dict['cov_t'] = Sigma[dx1_dx2:d_all, dx1_dx2:d_all] #ΣXT

        ##ΣX1,XT:
        a = torch.cat((cov_dict['cov_x1'], cov_dict['cross_x1_t']),dim=1,)
        b = torch.cat((cov_dict['cross_x1_t'].T, cov_dict['cov_t']),dim=1)
        cov_dict['joint_x1_t'] = torch.cat((a,b),dim=0)


    return cov_dict


def reorder_cov_blocks(
    Sigma: torch.Tensor,
    dims: dict[str, int],
    old_order: list[str],
    new_order: list[str],
) -> torch.Tensor:
    """
    Reorder covariance matrix blocks according to variable names.

    Example:
        Sigma is ordered as [X1, X2, T]
        reorder to [T, X1, X2]

        Sigma_new = reorder_cov_blocks(
            Sigma,
            dims={'X1': d1, 'X2': d2, 'T': dt},
            old_order=['X1', 'X2', 'T'],
            new_order=['T', 'X1', 'X2']
        )
    """

    assert set(old_order) == set(new_order), "old_order and new_order must contain the same variables"

    start = {}
    current = 0
    for var in old_order:
        start[var] = current
        current += dims[var]

    indices = []
    for var in new_order:
        s = start[var]
        e = s + dims[var]
        indices.append(torch.arange(s, e, device=Sigma.device))

    new_idx = torch.cat(indices)

    return Sigma[new_idx][:, new_idx]



def para_create_cov_matrix(dims,Sigmas=None,verbose=False):
    """This function will create the covariance matrix for the three variables M1,M2,T

    output: a len(rvs)*len(rvs)*p covariance matrix"""

    
    cov_dict = {}
    if verbose:
        print(f"\nFull covariance matrix shape: {Sigmas.shape}")

    x1_dim = dims[0]
    x2_dim = dims[1]
    xt_dim = dims[2] if len(dims) == 3 else 0


    dt_dx2 = x2_dim + xt_dim
    dx1_dx2 = x1_dim + x2_dim
    d_all = x1_dim + x2_dim + xt_dim

    #Full covariance matrix
    cov_dict['full_cov'] = Sigmas #Full covariance matrix ΣX1X2T
    #Cross-Covariances:
    cov_dict['cross_x1_x2'] = Sigmas[:,0:x1_dim, x1_dim:dx1_dx2] #ΣX1,X2   
    #Auto-Covariances
    cov_dict['cov_x1'] = Sigmas[:,0:x1_dim, 0:x1_dim] #ΣX1
    cov_dict['cov_x2'] = Sigmas[:,x1_dim:dx1_dx2, x1_dim:dx1_dx2] #ΣX2
   
    cov_dict['joint_x1_x2'] = Sigmas[:,0:dx1_dx2, 0:dx1_dx2]  #ΣX1X2


    if ( len(dims) == 3):
        #Cross-Covariances:
        cov_dict['cross_x2_xt'] = Sigmas[:,x1_dim:dx1_dx2, dx1_dx2:d_all]#ΣX2,XT
        cov_dict['cross_x2t_x1'] = Sigmas[:,x1_dim:d_all, 0:x1_dim] #ΣX2XT,X1
        cov_dict['cross_x1_xt'] = Sigmas[:,0:x1_dim, dx1_dx2:d_all] #ΣX1,XT

        #Auto-Covariances:
        cov_dict['joint_x2_xt'] = Sigmas[:,x1_dim:d_all, x1_dim:d_all]  #ΣX2XT
        cov_dict['cov_xt'] = Sigmas[:,dx1_dx2:d_all, dx1_dx2:d_all] #ΣXT

        ##ΣX1,XT:
        a = torch.cat((cov_dict['cov_x1'], cov_dict['cross_x1_xt']),dim=2)
        b = torch.cat((cov_dict['cross_x1_xt'].mT, cov_dict['cov_xt']),dim=2)
        cov_dict['joint_x1_xt'] = torch.cat((a,b),dim=1)


    return cov_dict

def old_para_create_cov_matrix(dims,Sigmas=None,verbose=False):
    """This function will create the covariance matrix for the three variables M1,M2,T
    input: M1,M2,T are torch tensors of shape (N,p) 
    rvs is a list of the three variables [M1,M2,T]
    N is the number of observations, 
    p is the dimension of each observation.

    output: a len(rvs)*len(rvs)*p covariance matrix"""

    cov_dict = {}
    if verbose:
        print(f"\nFull covariance matrix shape: {Sigmas.shape}")
    x0_dim = dims[0]
    x1_dim = dims[1]
    x2_dim = dims[2] if len(dims) == 3 else 0

    dt_dx1 = x0_dim + x1_dim
    d_all = x0_dim + x1_dim + x2_dim

    #Full covariance matrix
    cov_dict['full_cov'] =  Sigmas #Full covariance matrix ΣX0X1X2
    #Cross-Covariances:
    cov_dict['cross_x0_x1'] = Sigmas[:,0:x0_dim, x0_dim:dt_dx1] #ΣX0,X1   
    #Auto-Covariances
    cov_dict['cov_x0'] = Sigmas[:,0:x0_dim, 0:x0_dim] #ΣX0
    cov_dict['cov_x1'] = Sigmas[:,x0_dim:dt_dx1, x0_dim:dt_dx1] #ΣX1
   
    cov_dict['joint_x0_x1'] = Sigmas[:,0:dt_dx1, 0:dt_dx1]  #ΣX0X1


    if len(dims) == 3:
        #Cross-Covariances:
        cov_dict['cross_x1_x2'] = Sigmas[:,x0_dim:dt_dx1, dt_dx1:d_all]#ΣX1,X2
        cov_dict['cross_x12_x0'] = Sigmas[:,x0_dim:d_all, 0:x0_dim] #ΣX1X2,X0
        cov_dict['cross_x0_x2'] = Sigmas[:,0:x0_dim, dt_dx1:d_all] #ΣX0,X2

        #Auto-Covariances:
        cov_dict['joint_x1_x2'] = Sigmas[:,x0_dim:d_all, x0_dim:d_all]  #ΣX1X2
        cov_dict['cov_x2'] = Sigmas[:,dt_dx1:d_all, dt_dx1:d_all] #ΣX2

        ##ΣX0,X2:
        a = torch.cat((cov_dict['cov_x0'], cov_dict['cross_x0_x2']),dim=2)
        b = torch.cat((cov_dict['cross_x0_x2'].mT, cov_dict['cov_x2']),dim=2)
        cov_dict['auto_x02'] = torch.cat((a,b),dim=1)


    return cov_dict


def whiten_block(
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


def para_whiten_block(
                Sigma_xx: torch.Tensor,
                Sigma_xy: torch.Tensor,
                Sigma_yy: torch.Tensor) -> torch.Tensor:
    """
    Computes: Ux^{-T} @ Sigma_xy @ Uy^{-1}
    where Sigma_xx = Ux^T Ux, Sigma_yy = Uy^T Uy, and Ux,Uy are upper triangular.
    Supports batched inputs of shape (N, d, d).
    """
    if Sigma_xx.ndim == 2 and Sigma_yy.ndim == 2 and Sigma_xy.ndim == 2:
        return whiten_block(Sigma_xx, Sigma_xy, Sigma_yy)
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
    print(f"M1 R² (VP): {vp_results['R²_X1']:.4f} | I(T;M1): {mi_results['I(M1;T)']:.4f}")
    print(f"M2 R² (VP): {vp_results['R²_X2']:.4f} | I(T;M2): {mi_results['I(M2;T)']:.4f}")
    print(f"Both M1 and M2 R² (VP): {vp_results['R²_X12']:.4f} | I(T;M1,M2): {mi_results['I(M1,M2;T)']:.4f}")
    print(f"\nUnique to M1 (VP): {vp_results['unique_X1']:.4f} | Unique(T;M1\\M2): {pid_results['unq1']:.4f}")
    print(f"Unique to M2 (VP): {vp_results['unique_X2']:.4f} | Unique(T;M2\\M1): {pid_results['unq2']:.4f}")
    print(f"Common (VP): {vp_results['common']:.4f} | Redundant (PID): {pid_results['red']:.4f}")
    print(f"Synergy (PID): {pid_results['syn']:.4f}")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_mi_heatmap(
    csv_path,
    value_col,
    *,
    n_col="N",
    p_col="p",
    figsize=(7, 5),
    title=None,
    save_path=None,
    annotate=True,
    fmt=".3f",
    cmap="viridis"
):
    """
    Plot a block heatmap from an averaged CSV.

    Parameters
    ----------
    csv_path : str or Path
        Path to the averaged CSV.
    value_col : str
        Column to plot. For example:
        'mi_theoretical', 'mi_sample_no_bias', or 'mi_sample_with_bias'
    n_col : str
        Column for sample size N.
    p_col : str
        Column for total dimension p = px1 + px2.
    figsize : tuple
        Figure size.
    title : str or None
        Plot title.
    save_path : str or Path or None
        Where to save the figure. If None, does not save.
    annotate : bool
        Whether to write the numeric value inside each cell.
    fmt : str
        Format for annotations.
    cmap : str
        Matplotlib colormap name.
    """
    df = pd.read_csv(csv_path)

    # Remove possible unnamed index columns
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    # Keep only needed columns
    required_cols = {n_col, p_col, value_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Make sure numeric
    df[n_col] = pd.to_numeric(df[n_col])
    df[p_col] = pd.to_numeric(df[p_col])
    df[value_col] = pd.to_numeric(df[value_col])

    # Pivot to heatmap matrix
    heatmap_df = df.pivot(index=p_col, columns=n_col, values=value_col)

    # Sort axes
    heatmap_df = heatmap_df.sort_index(axis=0).sort_index(axis=1)

    values = heatmap_df.to_numpy(dtype=float)

    # Mask missing values
    masked_values = np.ma.masked_invalid(values)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(masked_values, aspect="auto", origin="lower", cmap=cmap_obj)

    # Tick labels
    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_xticklabels([int(x) if float(x).is_integer() else x for x in heatmap_df.columns])

    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_yticklabels([int(y) if float(y).is_integer() else y for y in heatmap_df.index])

    ax.set_xlabel("N")
    ax.set_ylabel("p = px1 + px2")

    if title is None:
        title = f"Heatmap of {value_col}"
    ax.set_title(title)

    # Add grid lines between blocks
    ax.set_xticks(np.arange(-0.5, len(heatmap_df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(heatmap_df.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)

    # Annotate cells
    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if not np.isnan(values[i, j]):
                    ax.text(
                        j, i,
                        format(values[i, j], fmt),
                        ha="center", va="center",
                        color="black"
                    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(f'{save_path}/{title}.png', dpi=300, bbox_inches="tight")

    plt.show()



def plot_all_mi_heatmaps(
    csv_path,
    title="Mutual Information Heatmaps",
    *,
    n_col="N",
    p_col="p",
    figsize=(16, 5),
    save_path=None,
    annotate=True,
    mean_fmt=".2f",
    std_fmt=".2f",
    log_scale=False,
    cmap="viridis",
    annotation_mode="pm",   # "pm" or "paren"
    fontsize=9,
    aggfunc="mean",         # handles duplicate (N,p) rows
):
    """
    Plot theoretical, naive, and bias-corrected MI heatmaps in one figure.

    Cell color is determined by the mean value.
    Cell annotation shows mean and std.

    Expected columns:
    N, p,
    mi_theoretical_mean, mi_theoretical_std,
    mi_sample_no_bias_mean, mi_sample_no_bias_std,
    mi_sample_with_bias_mean, mi_sample_with_bias_std
    """

    df = pd.read_csv(csv_path)

    # Remove possible unnamed index columns
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    # Ensure numeric
    df[n_col] = pd.to_numeric(df[n_col], errors="coerce")
    df[p_col] = pd.to_numeric(df[p_col], errors="coerce")

    mean_std_pairs = [
        ("mi_theoretical_mean", "mi_theoretical_std", "Theoretical MI"),
        ("mi_sample_no_bias_mean", "mi_sample_no_bias_std", "Naive MI"),
        ("mi_sample_with_bias_mean", "mi_sample_with_bias_std", "Bias-corrected MI"),
    ]

    required_cols = [n_col, p_col]
    for mean_col, std_col, _ in mean_std_pairs:
        required_cols.extend([mean_col, std_col])

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Check duplicates
    dup_mask = df.duplicated(subset=[n_col, p_col], keep=False)
    if dup_mask.any():
        print("Warning: duplicate (N, p) pairs found. Aggregating with pivot_table.")
        print(df.loc[dup_mask, [n_col, p_col]].value_counts().sort_index())

    # Global color scaling from mean columns only
    mean_columns = [pair[0] for pair in mean_std_pairs]
    all_mean_values = df[mean_columns].to_numpy(dtype=float)

    if log_scale:
        positive_vals = all_mean_values[all_mean_values > 0]
        if positive_vals.size == 0:
            raise ValueError("No positive mean values found. Cannot use log scale.")
        vmin = positive_vals.min()
        vmax = positive_vals.max()
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin = np.nanmin(all_mean_values)
        vmax = np.nanmax(all_mean_values)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    last_im = None

    for ax, (mean_col, std_col, panel_title) in zip(axes, mean_std_pairs):
        mean_df = df.pivot_table(
            index=p_col,
            columns=n_col,
            values=mean_col,
            aggfunc=aggfunc
        )
        std_df = df.pivot_table(
            index=p_col,
            columns=n_col,
            values=std_col,
            aggfunc=aggfunc
        )

        mean_df = mean_df.sort_index().sort_index(axis=1)
        std_df = std_df.reindex(index=mean_df.index, columns=mean_df.columns)

        mean_values = mean_df.to_numpy(dtype=float)
        std_values = std_df.to_numpy(dtype=float)

        plot_values = mean_values.copy()
        if log_scale:
            plot_values[plot_values <= 0] = np.nan

        im = ax.imshow(
            plot_values,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
        )
        last_im = im

        ax.set_xticks(np.arange(len(mean_df.columns)))
        ax.set_xticklabels(mean_df.columns.astype(int))

        ax.set_yticks(np.arange(len(mean_df.index)))
        ax.set_yticklabels(mean_df.index.astype(int))

        ax.set_xlabel("N")
        ax.set_ylabel("p = dx0,px1,pt")
        ax.set_title(panel_title)

        if annotate:
            for i in range(mean_values.shape[0]):
                for j in range(mean_values.shape[1]):
                    mean_val = mean_values[i, j]
                    std_val = std_values[i, j]

                    if np.isnan(mean_val):
                        text = "nan"
                    else:
                        mean_text = format(mean_val, mean_fmt)
                        std_text = "nan" if np.isnan(std_val) else format(std_val, std_fmt)

                        if annotation_mode == "pm":
                            text = f"{mean_text}\n±{std_text}"
                        elif annotation_mode == "paren":
                            text = f"{mean_text}\n({std_text})"
                        else:
                            raise ValueError("annotation_mode must be 'pm' or 'paren'")

                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=fontsize,
                    )

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.85)
    cbar.set_label("Mutual Information (mean)")

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        filename = f"{title}.png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches="tight")

    plt.show()



def plot_block_heatmap(csv_path, save_path=None):

    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    N_vals = sorted(df["N"].unique())
    p_vals = sorted(df["p"].unique())

    nN = len(N_vals)
    nP = len(p_vals)

    # each cell split into 3 rows
    block = np.full((nP * 3, nN), np.nan)

    for i, p in enumerate(p_vals):
        for j, N in enumerate(N_vals):

            row = df[(df["p"] == p) & (df["N"] == N)]
            if row.empty:
                continue

            block[i*3 + 0, j] = row["mi_theoretical"].values[0]
            block[i*3 + 1, j] = row["mi_sample_no_bias"].values[0]
            block[i*3 + 2, j] = row["mi_sample_with_bias"].values[0]

    fig, ax = plt.subplots(figsize=(8,6))

    im = ax.imshow(block, origin="lower", aspect="auto")

    # x axis
    ax.set_xticks(range(nN))
    ax.set_xticklabels(N_vals)
    ax.set_xlabel("N")

    # y axis (only show p labels centered in blocks)
    ax.set_yticks([i*3 + 1 for i in range(nP)])
    ax.set_yticklabels(p_vals)
    ax.set_ylabel("p = px1 + px2")

    # grid lines between big blocks
    for i in range(nP+1):
        ax.axhline(i*3 - 0.5, color="white", linewidth=2)

    for j in range(nN+1):
        ax.axvline(j - 0.5, color="white", linewidth=2)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("MI value")

    plt.title("Mutual Information Bias Simulation")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def anchored_oas_shrinkage(Sigma_full: torch.Tensor, cov_loo_all: torch.Tensor, n_samples: int):
    """
    Calculates OAS parameters ONCE on the full matrix, 
    and applies the EXACT SAME linear shrinkage to all LOO matrices.
    """
    # Fix: Ensure Sigma_full is 2D [d, d] for trace and matrix multiplication
    if Sigma_full.ndim == 3:
        Sigma_full = Sigma_full.squeeze(0)
    
    p = Sigma_full.shape[0]
    
    # 1. Calculate Anchor Parameters strictly from the Full Matrix
    tr_S = torch.trace(Sigma_full)
    tr_S2 = torch.trace(Sigma_full @ Sigma_full) # Now works because Sigma_full is 2D
    mu_anchor = tr_S / p
    
    # OAS formula using the full degrees of freedom (N - 1)
    N_for_formula = n_samples - 1
    numerator = (1.0 - 2.0/p) * tr_S2 + (tr_S ** 2)
    denominator = (N_for_formula + 1.0 - 2.0/p) * (tr_S2 - (tr_S ** 2) / p)
    
    # Handle edge case where denominator is 0
    if denominator == 0:
        alpha_anchor = torch.tensor(1.0, device=Sigma_full.device)
    else:
        alpha_anchor = numerator / denominator
        
    alpha_anchor = torch.clamp(alpha_anchor, 0.0, 1.0)
    
    T_anchor = mu_anchor * torch.eye(p, dtype=Sigma_full.dtype, device=Sigma_full.device)
    
    # 2. Apply the EXACT SAME constant anchor to everything
    Sigma_full_shrunk = (1.0 - alpha_anchor) * Sigma_full + alpha_anchor * T_anchor
    
    # Expand T_anchor to match the batch size of the LOO matrices (N, d, d)
    T_batch = T_anchor.unsqueeze(0).expand_as(cov_loo_all)
    cov_loo_all_shrunk = (1.0 - alpha_anchor) * cov_loo_all + alpha_anchor * T_batch
    
    # Return Sigma_full_shrunk as [1, d, d] to maintain consistency with the rest of your pipeline
    return Sigma_full_shrunk.unsqueeze(0), cov_loo_all_shrunk

def oas_cov_torch(S: torch.Tensor, N: int) -> torch.Tensor:
    """
    Apply Oracle Approximating Shrinkage (OAS) to a covariance matrix.
    Requires ONLY the sample covariance matrix S and sample size N.
    """
    p = S.shape[0]
    
    # Trace of S and Trace of S^2
    tr_S = torch.trace(S)
    tr_S2 = torch.trace(S @ S)
    
    # Calculate target matrix T (scaled identity)
    mu = tr_S / p
    T = mu * torch.eye(p, dtype=S.dtype, device=S.device)
    
    # Calculate optimal shrinkage alpha
    numerator = (1.0 - 2.0/p) * tr_S2 + (tr_S ** 2)
    denominator = (N + 1.0 - 2.0/p) * (tr_S2 - (tr_S ** 2) / p)
    
    # Handle edge case where denominator is 0 (perfectly spherical data)
    if denominator == 0:
        alpha = torch.tensor(1.0, dtype=S.dtype, device=S.device)
    else:
        alpha = numerator / denominator
        alpha = torch.clamp(alpha, min=0.0, max=1.0)
        
    # Apply shrinkage
    S_shrunk = (1.0 - alpha) * S + alpha * T
    return S_shrunk


def residual_rvs(rv_list:list,predictor_index=0):
    """Given a list of random variables (Torch.Tensors),
    returns a list where we predict the second rv using the first rv and return the residuls. 
    
    input: 
        list of  two random variables [rv1, rv2]
    
    output:
        [rv1,residual of rv2 after regressing out rv1]"""

    
    if len(rv_list) != 2:
        raise ValueError("This function is designed for exactly two random variables.")
    
    target_index = 1 - predictor_index
    predictor = rv_list[predictor_index]
    target = rv_list[target_index]

    _,model_fit = compute_ridge_cv_r2(predictor, target)

    target_pred = model_fit.predict(predictor)
    residual = target - target_pred
    return [predictor, residual]