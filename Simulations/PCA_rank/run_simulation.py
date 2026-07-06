from eigenvector_pca import eigenvector_pca_cv,fit_pca_loadings_sklearn
from pca_simulation import generate_rank_simulation_data
from rowwise_PCA import rowwise_loo_pca_variance_threshold
import numpy as np
from library_wrappers.missmda_ncp import estimate_ncp_pca
from time import perf_counter


def run_simulation(
    n_samples: int,
    n_features: int,
    rank: int,
    loading_corr: float,
    noise_std: float,
    random_state: int,
) -> dict:
    """Run PCA simulations with eigenvector and row-wise LOOCV methods."""
    data = generate_rank_simulation_data(
        n_samples=n_samples,
        n_features=n_features,
        rank=rank,
        loading_corr=loading_corr,
        noise_std=noise_std,
        random_state=random_state,
    )

    X = data["X"]

    # Run eigenvector PCA with cross-validation
    max_components = min(n_samples, n_features) - 1


    # EM-wold
    start = perf_counter()
    ncp_estimate = estimate_ncp_pca(X,ncp_max = max_components ,method="EM",method_cv = 'Kfold' ,nbsim=5,seed  = random_state)
    end_ncp =  perf_counter()
    ncp_estimate['time'] = end_ncp- start 
    
    
    eigenvector_result  = eigenvector_pca_cv(
        X,
        max_components=max_components,
        pca_fit_fn=fit_pca_loadings_sklearn,
        center=True,
        scale=False,
    )

    end_eigenvector = perf_counter() 
    eigenvector_result.time = end_eigenvector - end_ncp

    # Run row-wise LOOCV PCA
    rowwise_result = rowwise_loo_pca_variance_threshold(X, variance_threshold=0.95)
    end_rowwise = perf_counter() 
    
    rowwise_result.time = end_rowwise - end_eigenvector

    return {
        "eigenvector_result": eigenvector_result,
        "rowwise_result": rowwise_result,
        "simulation_data": data,
        "ncp_estimate": ncp_estimate,
    }


if __name__ == "__main__":
    # Example usage
    n_samples = 100
    n_features = 50
    rank = 10
    loading_corr = 0.4
    noise_std = 0.3
    random_state = 56 
    results = run_simulation(
        n_samples=n_samples,
        n_features=n_features,
        rank=rank,
        loading_corr=loading_corr,
        noise_std=noise_std,
        random_state=random_state,
    )
    
    eigenvector_result = results["eigenvector_result"]
    rowwise_result = results["rowwise_result"]
    ncp_estimate = results["ncp_estimate"]

    print("\nEigenvector PCA selected components:", eigenvector_result.selected_n_components, "\n It took", eigenvector_result.time, "seconds to run")
    


    print("\nEM-wold estimated components:", ncp_estimate['ncp'], "\n It took", ncp_estimate['time'], "seconds to run")


    print("\nRow-wise LOOCV PCA selected components mean over fold:", np.mean(np.array(rowwise_result.n_components_per_fold)),"\n It took", rowwise_result.time, "seconds to run")
    print('\n Row-wise LOOCV PCA selected components minimum fold:', min(rowwise_result.n_components_per_fold))