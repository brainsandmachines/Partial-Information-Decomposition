import torch 
from Partial_Information_Decomposition.Idep.Idep_Simulations.Simulation_utils import *
from Partial_Information_Decomposition.Idep.Idep_Simulations.shrinkaging import ledoit_wolf_cov, oracle_shrinkage_cov, shrunk_cov

def on_covariance(config,data):
    """This will call an intermidate function on the covariance
    
    Input: data: a tuple containing the covariance matrix and a list of random variables
    
    config: a dictionary with data and parameters for the function to apply on the covariance matrix.
        config['on_cov'] = 'ledoit_wolf' or 'oas' or 'shrunk_cov' or 'False etc...'
        config['alpha'] = the alpha parameter for the shrunk_cov function if on_cov is 'shrunk_cov'
  
    
    Output: the covariance matrix after applying the function on it."""

    covariance_matrix = data
    cov_list = []
    on_cov = config['on_covariance'] #Check for srhinkage method 
    if on_cov == 'False':
        return {'cov': covariance_matrix}
    
    if covariance_matrix.ndim == 2:
        covariance_matrix = covariance_matrix.unsqueeze(0)

    for cov in covariance_matrix:
        if torch.any(torch.isnan(cov)):
            raise ValueError("Covariance matrix contains NaN values.")
        if torch.any(torch.isinf(cov)):
            raise ValueError("Covariance matrix contains Inf values.")
        
        elif on_cov == 'ledoit_wolf':
            cov =  ledoit_wolf_cov(cov.cpu().numpy())
        
        elif on_cov == 'shrunk_cov':
            alpha = config['alpha']
            cov = shrunk_cov(cov.cpu().numpy(), alpha)
        
        if on_cov == 'oas':
            cov =  oracle_shrinkage_cov(cov.cpu().numpy())

        if type(cov) != torch.Tensor:
            cov = torch.from_numpy(cov).to(covariance_matrix.device).to(covariance_matrix.dtype)
        cov_list.append(cov)
    cov = torch.stack(cov_list, dim=0)    
    assert cov.shape == covariance_matrix.shape, f"Expected output shape {covariance_matrix.shape}, got {cov.shape}." 
    return {'cov':cov}