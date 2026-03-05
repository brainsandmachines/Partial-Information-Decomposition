import numpy as np
from sklearn.linear_model import RidgeCV, LinearRegression
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from sklearn.linear_model import MultiTaskLassoCV,LassoCV
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from sklearn.linear_model import LassoCV
from sklearn.multioutput import MultiOutputRegressor
from encoding_model.encoding_utils import (compute_r2,
                                           compute_ols_cv_r2,
                                           compute_ridge_cv_r2,
                                            compute_lasso_cv_r2,
                                           diagnostic_plots, 
                                           singularity_report)
from Partial_Information_Decomposition.Idep_multivariate_gauss import Idep_multivariate_gauss






class commonality_analysis:
    def __init__(self,features_x1,features_x2,target,method='standard',alphas=None):
        self.features_x1 = features_x1
        self.features_x2 = features_x2
        self.target = target
        self.method = method
        self.alphas = alphas



    def compute_ridge_cv_r2(self,X, y, alphas=None):
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

        else:
            if not isinstance(alphas, np.ndarray):
                alphas = np.array(alphas)
        
        # RidgeCV with leave-one-out CV (efficient GCV approximation)
        # cv=None means use efficient LOO via GCV
        ridge_cv = RidgeCV(alphas=alphas, fit_intercept=True, scoring='r2', cv=None,store_cv_results=True)
        ridge_cv.fit(X, y)


        
        return ridge_cv,ridge_cv.best_score_

    def ca(self, features_x1, features_x2, target, method='standard', alphas=None):
        """
        Decomposes the variance of the target variable into contributions from features_x1 and features_x2.
        
        This is done using commonality analysis, which does not assume uncorrelated sources.
        Supports three methods:
        - 'standard': In-sample R² (prone to overfitting)
        - 'ols_cv': Cross-validated R² using PRESS residuals
        - 'ridge_cv': Cross-validated R² using RidgeCV with GCV
        
        Args:
            features_x1 (np.ndarray): Feature matrix X1 (shape: [n, p_X1] or [n,] for 1D).
            features_x2 (np.ndarray): Feature matrix X2 (shape: [n, p_X2] or [n,] for 1D).
            target (np.ndarray): Target variable (shape: [n,]).
            method (str): Which R² computation method to use: 'standard', 'ols_cv', or 'ridge_cv'.
            alphas (array-like, optional): Alpha values for RidgeCV (only used if method='ridge_cv').
            
        Returns:
            dict: A dictionary with R² values and variance decomposition.
        """
        n = len(target)
        
        # Ensure 2D
        if features_x1.ndim == 1:
            self.features_x1 = features_x1.reshape(-1, 1)
        if features_x2.ndim == 1:
            self.features_x2 = features_x2.reshape(-1, 1)
        
        # Total sum of squares (using N-1 for unbiased sample variance)
        tss = np.sum((target - target.mean())**2)
        var_y = tss / (n - 1)
        
        # Build combined features
        self.features_AB = np.hstack([self.features_x1, self.features_x2])
        
        # Select R² computation function based on method
        if method == 'standard':
            compute_r2_fn = compute_r2
        elif method == 'ols_cv':
            compute_r2_fn = compute_ols_cv_r2
        elif method == 'ridge_cv':
            compute_r2_fn = lambda X, y: self.compute_ridge_cv_r2(X, y, alphas=alphas)
        elif method == 'lasso_cv':
            compute_r2_fn = compute_lasso_cv_r2
        else:
            raise ValueError(f"Unknown method: {method}. Use 'standard', 'ols_cv', or 'ridge_cv'.")
        
        # Compute R² for each model
        modelx1,r2_x1 = self.compute_ridge_cv_r2(self.features_x1, self.target, alphas=alphas)
        modelx2,r2_x2 = self.compute_ridge_cv_r2(self.features_x2, self.target, alphas=alphas)
        modelx12,r2_x12 = self.compute_ridge_cv_r2(self.features_AB, self.target, alphas=alphas)
        
        self.modelx1_cv_results = modelx1.cv_results_
        self.modelx2_cv_results = modelx2.cv_results_
        self.modelx12_cv_results = modelx12.cv_results_
        # Commonality analysis decomposition
        unique_A = (r2_x12 - r2_x2) 
        unique_B = (r2_x12 - r2_x1) 
        common_AB = (r2_x1 + r2_x2 - r2_x12) 
        unexplained = (1 - r2_x12) 
        
        return {
            'R²_X1': r2_x1,
            'R²_X2': r2_x2,
            'R²_X12': r2_x12,
            'unique_X1': unique_A,
            'unique_X2': unique_B,
            'common': common_AB,
            'unexplained': unexplained,
            'betas_X1': modelx1.coef_ if hasattr(modelx1, 'coef_') else None,
            'betas_X2': modelx2.coef_ if hasattr(modelx2, 'coef_') else None,
            'betas_X12': modelx12.coef_ if hasattr(modelx12, 'coef_') else None
        }
    
    def find_best_alpha(self, alphas: np.ndarray) -> float:
        alphas = self.alphas if alphas is None else alphas
        self.features_x12 = np.hstack([self.features_x1, self.features_x2])
        ridge_model, best_score = self.compute_ridge_cv_r2(self.features_x12, self.target, alphas=alphas)
        best_alpha = ridge_model.alpha_

        return best_alpha
    
