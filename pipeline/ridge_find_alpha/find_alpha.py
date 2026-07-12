import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor



def find_alpha_per_pc(predictor,target):
    """This function finds the best alpha on ridge regression for each principal component in the target
    
    Input:
        predictor: array-like, shape (n_samples, n_features)
            The input data
        target: array-like, shape (n_samples, n_components)
            The target values

    Output:
        best_alpha: float
            The best alpha value for each principal component
    """

    base_ridge = RidgeCV(
    alphas=np.logspace(-3, 3, 50),
    cv=5,
    scoring="r2",
    fit_intercept=True,
    )

    ridge_per_pc = MultiOutputRegressor(base_ridge)
    ridge_per_pc.fit(X_train, target_pca_train)

    target_pca_pred = ridge_per_pc.predict(X_test)

    alphas_per_pc = np.array([
        estimator.alpha_
        for estimator in ridge_per_pc.estimators_
    ])